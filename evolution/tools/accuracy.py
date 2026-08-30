"""Keep an evolved tool description true to the schema it describes.

PLAN.md's Phase 2 constraint list ends with one nothing else enforces:
descriptions "must remain factually accurate (can't claim a tool does something
it doesn't)". The size budgets are arithmetic and the growth limit is
arithmetic, but accuracy is a claim about meaning, and an optimizer rewarded
for selection accuracy has every incentive to overclaim. A description that
promises ``read_file`` can take a ``recursive`` flag wins the examples where
recursion is mentioned and then produces a call the agent cannot make.

The checks come in two tiers, and the deterministic tier is the one that always
runs.

**Structural checks, no model required.** The frozen half of the schema -
parameter names, types, enums, required list - is ground truth, and text can be
checked against it without asking anyone's opinion:

* every backticked or quoted identifier that reads as a parameter reference
  must be a real parameter of that tool
* a value claimed for a parameter that has an enum must be one of the enum's
  values
* a parameter the schema marks required may not be described as optional
* a promise that the caller can supply something - a timeout, an encoding, a
  retry count - must correspond to a parameter that can carry it

**What is deliberately not checked.** The schema cannot express behaviour, so
behaviour claims are not contradicted by it. "Searches recursively" is a fact
about the handler, not about the parameter list, and flagging it would punish
accurate text. Only claims about the *call interface* are adjudicated here,
because those are the only claims the frozen schema can settle. Everything
softer is what the optional entailment tier is for.

**Optional entailment check, one model call.** :class:`DescriptionEntailment`
asks whether the evolved text makes any claim the baseline description and the
schema do not support. It is injectable and it is optional: with no predictor
supplied, or no LM configured, the checker reports that the tier was skipped
and returns the structural findings alone. A constraint that silently needs a
network to work is a constraint that silently stops working.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import dspy

from evolution.tools.tool_catalog import ToolCatalog, ToolDescriptions

__all__ = [
    "KIND_UNKNOWN_PARAMETER",
    "KIND_ENUM_CONTRADICTION",
    "KIND_REQUIREDNESS",
    "KIND_UNSUPPORTED_CAPABILITY",
    "KIND_ENTAILMENT",
    "AccuracyFinding",
    "AccuracyReport",
    "ToolSchemaFacts",
    "DescriptionEntailment",
    "FactualAccuracyChecker",
    "facts_from_catalog",
]

KIND_UNKNOWN_PARAMETER = "unknown_parameter"
KIND_ENUM_CONTRADICTION = "enum_contradiction"
KIND_REQUIREDNESS = "requiredness"
KIND_UNSUPPORTED_CAPABILITY = "unsupported_capability"
KIND_ENTAILMENT = "entailment"

SOURCE_STRUCTURAL = "structural"
SOURCE_ENTAILMENT = "entailment"

# Backticked, single-quoted or double-quoted runs. The identifier filter below
# does the real work: "the file's name and the user's copy" matches here and is
# discarded there, because it is not one bare token.
_QUOTED = re.compile(r"`([^`\n]{1,64})`|'([^'\n]{1,64})'|\"([^\"\n]{1,64})\"")

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# ``mode='replace'``, ``replace_all=true``, ``target = files``.
_ASSIGNMENT = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[`'\"]?([A-Za-z0-9_.*/-]{1,40})[`'\"]?"
)

# Words that mark the quoted token beside them as a parameter reference rather
# than as prose. Either side counts.
_PARAM_WORDS = {
    "parameter",
    "parameters",
    "param",
    "params",
    "argument",
    "arguments",
    "arg",
    "args",
    "option",
    "options",
    "flag",
    "flags",
    "field",
    "fields",
    "key",
    "keys",
}
_PARAM_VERBS = {"pass", "set", "specify", "provide", "supply", "omit", "include"}

# Values a description may quote that are never parameter claims.
_SCHEMA_WORDS = {
    "string",
    "integer",
    "number",
    "boolean",
    "object",
    "array",
    "null",
    "true",
    "false",
    "none",
    "default",
    "required",
    "optional",
}

# Caller-supplied affordances. A description may only promise the caller can
# set one of these when a parameter exists that could carry it. Behaviour verbs
# ("use", "reads") are excluded on purpose: they describe what the tool does,
# not what the caller may hand it.
_CAPABILITY_VERBS = (
    r"(?:pass(?:es|ing)?|sets?|setting|specif(?:y|ies|ying)|provid(?:e|es|ing)"
    r"|suppl(?:y|ies|ying)|configur(?:e|es|ing)|choos(?:e|es|ing)"
    r"|controls?|controlling|adjusts?|adjusting)"
)
_CAPABILITY_CLAIMS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("timeout", ("timeout", "deadline", "seconds"), "a timeout"),
    ("encoding", ("encoding", "charset"), "a text encoding"),
    ("retries", ("retry", "retries", "attempts"), "a retry count"),
    ("retry count", ("retry", "retries", "attempts"), "a retry count"),
    ("concurrency", ("concurren", "parallel", "workers", "jobs"), "a concurrency level"),
)

# Compiled once: the claims and verbs are module constants, so building the
# pattern per call in the check loop was pure rework.
_CAPABILITY_PATTERNS: tuple[tuple[re.Pattern, tuple[str, ...], str], ...] = tuple(
    (
        re.compile(
            rf"{_CAPABILITY_VERBS}\b[^.;\n]{{0,60}}?\b{re.escape(phrase)}\b",
            re.IGNORECASE,
        ),
        keywords,
        human,
    )
    for phrase, keywords, human in _CAPABILITY_CLAIMS
)

_NO_ARGUMENTS = re.compile(
    r"\b(?:takes|requires|needs|accepts)\s+no\s+(?:arguments|parameters|args|params)\b",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────────
# Findings
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AccuracyFinding:
    """One way a description says something the schema does not support."""

    tool: str
    target: str  # "read_file" or "read_file.limit"
    kind: str
    message: str
    evidence: str = ""
    source: str = SOURCE_STRUCTURAL

    def describe(self) -> str:
        """The target, the problem, and the text it was found in."""
        text = f"{self.target}: {self.message}"
        if self.evidence:
            text += f" (in: {self.evidence!r})"
        return text

    def to_dict(self) -> dict:
        """Serialise one factual finding."""
        return {
            "tool": self.tool,
            "target": self.target,
            "kind": self.kind,
            "message": self.message,
            "evidence": self.evidence,
            "source": self.source,
        }


@dataclass
class AccuracyReport:
    """Every factual finding across a bundle, and whether the LLM tier ran."""

    findings: list[AccuracyFinding] = field(default_factory=list)
    entailment_ran: bool = False
    entailment_skipped: str = ""
    checked: int = 0

    @property
    def passed(self) -> bool:
        """True when nothing was found."""
        return not self.findings

    def for_target(self, target: str) -> list[AccuracyFinding]:
        """Findings recorded against one target."""
        return [f for f in self.findings if f.target == target]

    def by_target(self) -> dict[str, list[AccuracyFinding]]:
        """Findings grouped by target, name-sorted."""
        grouped: dict[str, list[AccuracyFinding]] = {}
        for finding in self.findings:
            grouped.setdefault(finding.target, []).append(finding)
        return {key: grouped[key] for key in sorted(grouped)}

    def summary(self) -> str:
        """One line naming the finding count and whether entailment ran."""
        head = (
            f"factual accuracy: {len(self.findings)} finding(s) over "
            f"{self.checked} description(s) checked"
        )
        if self.entailment_ran:
            return head + "; entailment check ran"
        if self.entailment_skipped:
            return head + f"; entailment check skipped ({self.entailment_skipped})"
        return head + "; structural checks only"

    def to_dict(self) -> dict:
        """Serialise the accuracy report, entailment status included."""
        return {
            "passed": self.passed,
            "checked": self.checked,
            "entailment_ran": self.entailment_ran,
            "entailment_skipped": self.entailment_skipped,
            "findings": [f.to_dict() for f in self.findings],
        }


# ──────────────────────────────────────────────────────────────────────────
# The frozen half of a schema
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolSchemaFacts:
    """What one tool's schema actually guarantees, as text checks need it.

    ``known_tools`` is here because tool descriptions legitimately quote other
    tools - the real ``write_file`` text says to use 'patch' for targeted edits -
    and a checker that reads every quoted name as a parameter claim would reject
    accurate descriptions.
    """

    tool_name: str
    params: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    types: dict[str, str] = field(default_factory=dict)
    enums: dict[str, tuple[str, ...]] = field(default_factory=dict)
    known_tools: tuple[str, ...] = ()

    @property
    def enum_values(self) -> set[str]:
        """Every declared enum value across the schema, flattened."""
        return {value for values in self.enums.values() for value in values}

    def render(self) -> str:
        """The schema as a compact block, for a prompt or an error message."""
        lines = [f"tool: {self.tool_name}"]
        for name in self.params:
            bits = [self.types.get(name, "any")]
            if name in self.required:
                bits.append("required")
            if name in self.enums:
                bits.append("one of " + ", ".join(self.enums[name]))
            lines.append(f"  {name}: {'; '.join(bits)}")
        if not self.params:
            lines.append("  (no parameters)")
        return "\n".join(lines)

    @classmethod
    def from_entry(
        cls,
        entry: Any,
        known_tools: Sequence[str] = (),
    ) -> "ToolSchemaFacts":
        """Build from a :class:`~evolution.tools.tool_catalog.ToolEntry`."""
        enums = {
            name: tuple(str(v) for v in values)
            for name, values in getattr(entry, "param_enums", {}).items()
            if values
        }
        return cls(
            tool_name=entry.tool_name,
            params=tuple(entry.param_names),
            required=tuple(entry.required),
            types=dict(entry.param_types),
            enums=enums,
            known_tools=tuple(known_tools),
        )


def facts_from_catalog(catalog: ToolCatalog) -> dict[str, ToolSchemaFacts]:
    """``{tool_name: ToolSchemaFacts}`` for a whole catalogue."""
    names = list(catalog.names)
    return {
        entry.tool_name: ToolSchemaFacts.from_entry(entry, known_tools=names)
        for entry in catalog
    }


# ──────────────────────────────────────────────────────────────────────────
# The optional entailment tier
# ──────────────────────────────────────────────────────────────────────────


class DescriptionEntailment(dspy.Signature):
    """Decide whether an evolved tool description overclaims.

    You are given a tool's frozen schema, the description it shipped with, and
    a rewritten description. Answer one question: does the rewrite make any
    claim that neither the baseline description nor the schema supports?

    Say 'no' to supported only for a real overclaim: a capability the tool does
    not have, a parameter that does not exist, a guarantee about results the
    baseline never made. Rewording, reordering, shortening, sharpening the
    wording, and adding guidance about when to prefer another tool are all
    supported. Being shorter is not an overclaim.
    """

    tool_name: str = dspy.InputField(desc="Name of the tool")
    tool_schema: str = dspy.InputField(desc="Frozen schema: parameters, types, enums, required")
    baseline_description: str = dspy.InputField(desc="The description that ships today")
    evolved_description: str = dspy.InputField(desc="The rewritten description under review")
    supported: str = dspy.OutputField(desc="'yes' if every claim is supported, otherwise 'no'")
    unsupported_claim: str = dspy.OutputField(
        desc="The strongest unsupported claim, quoted from the evolved text, or 'none'"
    )


def _dedupe(findings: list[AccuracyFinding]) -> list[AccuracyFinding]:
    """Same complaint twice is still one complaint."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[AccuracyFinding] = []
    for finding in findings:
        key = (finding.target, finding.kind, finding.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique


def _truthy_no(value: Any) -> bool:
    """True when a judge said the description is not supported."""
    text = str(value or "").strip().strip(".").casefold()
    return text.startswith("no") or text in {"false", "unsupported", "0"}


# ──────────────────────────────────────────────────────────────────────────
# The checker
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class FactualAccuracyChecker:
    """Structural accuracy checks, plus an optional model-backed entailment tier.

    ``entailment`` is any callable with the :class:`DescriptionEntailment`
    signature. Leaving it ``None`` is a supported configuration, not a
    degraded one: the structural checks are the part that must always run.
    """

    facts: dict[str, ToolSchemaFacts] = field(default_factory=dict)
    entailment: Any = None
    lm: Any = None
    skipped_reason: str = ""
    entailment_ran: bool = False

    # ── plumbing ────────────────────────────────────────────────────────
    def facts_for(self, tool: str) -> Optional[ToolSchemaFacts]:
        """Schema facts for *tool*, or None when it is not in the catalogue."""
        return self.facts.get(tool)

    def _lm_available(self) -> bool:
        if self.lm is not None:
            return True
        try:
            return getattr(dspy.settings, "lm", None) is not None
        except Exception:
            return False

    # ── structural checks ───────────────────────────────────────────────
    def check_tool(
        self,
        tool: str,
        description: str,
        baseline_description: str = "",
    ) -> list[AccuracyFinding]:
        """Every structural and entailment finding for one tool description."""
        facts = self.facts_for(tool)
        if facts is None:
            return []
        findings = _dedupe(self._structural(facts, tool, description))
        entailed = self._entailment(facts, tool, description, baseline_description)
        if entailed is not None:
            findings.append(entailed)
        return findings

    def check_param(
        self,
        tool: str,
        param: str,
        description: str,
        baseline_description: str = "",
    ) -> list[AccuracyFinding]:
        """Structural findings for one parameter description.

        A parameter's own description is checked against its own enum over the
        whole text, not through a proximity window: the subject is not in doubt.
        """
        facts = self.facts_for(tool)
        if facts is None:
            return []
        target = f"{tool}.{param}"
        findings = self._structural(facts, tool, description, target=target)
        allowed = facts.enums.get(param)
        for value, evidence, _ in self._quoted_identifiers(description) if allowed else []:
            if value in allowed:
                continue
            if value in facts.params or value in facts.known_tools:
                continue
            if value.casefold() in _SCHEMA_WORDS:
                continue
            findings.append(
                AccuracyFinding(
                    tool=tool,
                    target=target,
                    kind=KIND_ENUM_CONTRADICTION,
                    message=(
                        f"names {value!r} as a value of {param!r}, but the schema "
                        f"allows only: {', '.join(allowed)}"
                    ),
                    evidence=evidence,
                )
            )
        return _dedupe(findings)

    def _structural(
        self,
        facts: ToolSchemaFacts,
        tool: str,
        text: str,
        target: Optional[str] = None,
    ) -> list[AccuracyFinding]:
        target = target or tool
        findings: list[AccuracyFinding] = []
        findings.extend(self._unknown_parameters(facts, tool, target, text))
        findings.extend(self._enum_contradictions(facts, tool, target, text))
        findings.extend(self._requiredness(facts, tool, target, text))
        findings.extend(self._capabilities(facts, tool, target, text))
        return findings

    @staticmethod
    def _quoted_identifiers(text: str) -> list[tuple[str, str, bool]]:
        """``(identifier, surrounding text, was_backticked)`` per quoted token.

        The backtick flag matters: hermes-agent's shipped descriptions contain
        no backticks at all, so a backticked token is something a rewrite added
        and is code by intent. A single-quoted word may just be prose.
        """
        found: list[tuple[str, str, bool]] = []
        for match in _QUOTED.finditer(text or ""):
            token = next(group for group in match.groups() if group is not None)
            token = token.strip()
            if not _IDENTIFIER.match(token):
                continue
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            found.append((token, text[start:end].strip(), match.group(0).startswith("`")))
        return found

    def _unknown_parameters(
        self,
        facts: ToolSchemaFacts,
        tool: str,
        target: str,
        text: str,
    ) -> list[AccuracyFinding]:
        """Quoted identifiers that read as parameters the schema does not have.

        Three things qualify a token: it was backticked, which in this corpus
        only ever means code; it is snake_case, which no English word is; or it
        sits next to a word like "parameter" or a verb like "pass". A plainly
        quoted single word in prose is left alone, because descriptions quote
        modes, formats and other tools' names all the time and none of those are
        claims about this tool's arguments.
        """
        findings: list[AccuracyFinding] = []
        seen: set[str] = set()
        # A token on the right of an equals sign is a value, not a parameter
        # name. ``target='both'`` is an enum problem, and reporting it twice
        # under two headings helps nobody.
        assigned = {
            match.group(2).strip("`'\"") for match in _ASSIGNMENT.finditer(text or "")
        }
        for token, evidence, backticked in self._quoted_identifiers(text):
            lowered = token.casefold()
            if token in facts.params or token in facts.known_tools:
                continue
            if token in assigned:
                continue
            if token == facts.tool_name or lowered in _SCHEMA_WORDS:
                continue
            if token in facts.enum_values:
                continue
            if token in seen:
                continue
            if not (backticked or "_" in token or self._reads_as_parameter(text, token)):
                continue
            seen.add(token)
            known = ", ".join(facts.params) or "none"
            findings.append(
                AccuracyFinding(
                    tool=tool,
                    target=target,
                    kind=KIND_UNKNOWN_PARAMETER,
                    message=(
                        f"names {token!r} as if it were a parameter, but "
                        f"{facts.tool_name} has only: {known}"
                    ),
                    evidence=evidence,
                )
            )
        return findings

    @staticmethod
    def _reads_as_parameter(text: str, token: str) -> bool:
        """True when *token* is used in a parameter context somewhere in *text*."""
        for match in re.finditer(re.escape(token), text):
            before = text[max(0, match.start() - 32):match.start()].casefold()
            after = text[match.end():match.end() + 24].casefold()
            words = re.findall(r"[a-z_]+", before)[-3:] + re.findall(r"[a-z_]+", after)[:2]
            if _PARAM_WORDS.intersection(words) or _PARAM_VERBS.intersection(words):
                return True
            if after.lstrip("`'\"").startswith("="):
                return True
        return False

    def _enum_contradictions(
        self,
        facts: ToolSchemaFacts,
        tool: str,
        target: str,
        text: str,
    ) -> list[AccuracyFinding]:
        """Values claimed for an enum parameter that the enum does not allow.

        Both an explicit ``mode='rewrite'`` and a quoted value sitting next to
        the parameter's name count. The proximity window stops at the next
        parameter mention so a value belonging to one parameter is never read as
        a claim about another.
        """
        findings: list[AccuracyFinding] = []
        reported: set[tuple[str, str]] = set()

        for match in _ASSIGNMENT.finditer(text or ""):
            param, value = match.group(1), match.group(2).strip("`'\"")
            allowed = facts.enums.get(param)
            if not allowed or value in allowed:
                continue
            key = (param, value)
            if key in reported:
                continue
            reported.add(key)
            findings.append(
                AccuracyFinding(
                    tool=tool,
                    target=target,
                    kind=KIND_ENUM_CONTRADICTION,
                    message=(
                        f"claims {param}={value!r}, but the schema allows only: "
                        f"{', '.join(allowed)}"
                    ),
                    evidence=match.group(0),
                )
            )

        for param, allowed in facts.enums.items():
            for window, evidence in self._windows_after(facts, text, param):
                for token, _, _backticked in self._quoted_identifiers(window):
                    if token in allowed or token in facts.params:
                        continue
                    if token in facts.known_tools or token.casefold() in _SCHEMA_WORDS:
                        continue
                    key = (param, token)
                    if key in reported:
                        continue
                    reported.add(key)
                    findings.append(
                        AccuracyFinding(
                            tool=tool,
                            target=target,
                            kind=KIND_ENUM_CONTRADICTION,
                            message=(
                                f"offers {token!r} as a value of {param!r}, but the "
                                f"schema allows only: {', '.join(allowed)}"
                            ),
                            evidence=evidence,
                        )
                    )
        return findings

    @staticmethod
    def _windows_after(
        facts: ToolSchemaFacts,
        text: str,
        param: str,
        span: int = 90,
    ) -> list[tuple[str, str]]:
        """Text right after each mention of *param*, cut at the next subject.

        A window ends at the first sentence break or the first mention of a
        different parameter, so "path (required), offset (optional)" never lets
        offset's words be read as a claim about path.
        """
        windows: list[tuple[str, str]] = []
        others = [p for p in facts.params if p != param]
        for match in re.finditer(rf"\b{re.escape(param)}\b", text or ""):
            window = text[match.end():match.end() + span]
            cut = len(window)
            for stop in (". ", "; ", "\n"):
                index = window.find(stop)
                if index != -1:
                    cut = min(cut, index)
            for other in others:
                found = re.search(rf"\b{re.escape(other)}\b", window)
                if found:
                    cut = min(cut, found.start())
            windows.append((window[:cut], (param + window[:cut]).strip()))
        return windows

    def _requiredness(
        self,
        facts: ToolSchemaFacts,
        tool: str,
        target: str,
        text: str,
    ) -> list[AccuracyFinding]:
        """A required parameter described as optional.

        Only this direction is checked. The reverse - text calling a
        schema-optional parameter required - is how the real ``patch`` tool
        documents "REQUIRED when mode='replace'", a conditional requirement JSON
        Schema cannot express. Flagging that would punish the honest text.
        """
        findings: list[AccuracyFinding] = []
        for param in facts.required:
            for window, evidence in self._windows_after(facts, text, param, span=60):
                phrase = window.casefold()
                if "optional" not in phrase and "may be omitted" not in phrase:
                    continue
                if "not optional" in phrase:
                    continue
                findings.append(
                    AccuracyFinding(
                        tool=tool,
                        target=target,
                        kind=KIND_REQUIREDNESS,
                        message=(
                            f"describes the required parameter {param!r} as optional"
                        ),
                        evidence=evidence,
                    )
                )
                break
        no_arguments = _NO_ARGUMENTS.search(text or "")
        if facts.required and no_arguments:
            findings.append(
                AccuracyFinding(
                    tool=tool,
                    target=target,
                    kind=KIND_REQUIREDNESS,
                    message=(
                        f"says the tool takes no arguments, but the schema requires: "
                        f"{', '.join(facts.required)}"
                    ),
                    evidence=no_arguments.group(0),
                )
            )
        return findings

    def _capabilities(
        self,
        facts: ToolSchemaFacts,
        tool: str,
        target: str,
        text: str,
    ) -> list[AccuracyFinding]:
        """Promises the caller can supply something with nowhere to put it."""
        findings: list[AccuracyFinding] = []
        params = " ".join(facts.params).casefold()
        for pattern, keywords, human in _CAPABILITY_PATTERNS:
            match = pattern.search(text or "")
            if not match:
                continue
            if any(keyword in params for keyword in keywords):
                continue
            findings.append(
                AccuracyFinding(
                    tool=tool,
                    target=target,
                    kind=KIND_UNSUPPORTED_CAPABILITY,
                    message=(
                        f"promises the caller can set {human}, but no parameter of "
                        f"{facts.tool_name} carries it"
                    ),
                    evidence=match.group(0).strip(),
                )
            )
        return findings

    # ── entailment tier ─────────────────────────────────────────────────
    def _entailment(
        self,
        facts: ToolSchemaFacts,
        tool: str,
        description: str,
        baseline_description: str,
    ) -> Optional[AccuracyFinding]:
        """One model call, or nothing at all. Never an exception."""
        if self.entailment is None:
            self.skipped_reason = self.skipped_reason or "no entailment predictor supplied"
            return None
        if not self._lm_available():
            self.skipped_reason = "no LM configured"
            return None
        if not baseline_description:
            # Nothing to entail against. The structural tier still applies.
            return None
        try:
            if self.lm is not None:
                with dspy.context(lm=self.lm):
                    result = self.entailment(
                        tool_name=tool,
                        tool_schema=facts.render(),
                        baseline_description=baseline_description,
                        evolved_description=description,
                    )
            else:
                result = self.entailment(
                    tool_name=tool,
                    tool_schema=facts.render(),
                    baseline_description=baseline_description,
                    evolved_description=description,
                )
        except Exception as exc:  # a judge that fails must not fail the run
            self.skipped_reason = f"entailment check errored: {exc}"
            return None

        self.entailment_ran = True
        if not _truthy_no(getattr(result, "supported", "yes")):
            return None
        claim = str(getattr(result, "unsupported_claim", "") or "").strip()
        if claim.casefold().strip(".") in {"", "none", "n/a"}:
            claim = "an unsupported claim the judge did not quote"
        return AccuracyFinding(
            tool=tool,
            target=tool,
            kind=KIND_ENTAILMENT,
            message=f"makes a claim the baseline description and schema do not support: {claim}",
            evidence=claim,
            source=SOURCE_ENTAILMENT,
        )

    # ── bundles ─────────────────────────────────────────────────────────
    def check_bundle(
        self,
        candidate: dict[str, ToolDescriptions],
        baseline: Optional[dict[str, ToolDescriptions]] = None,
        allowed: Optional[Iterable[str]] = None,
    ) -> AccuracyReport:
        """Check every description that changed, or every one when unchanged is unknown."""
        baseline = baseline or {}
        permitted = set(allowed) if allowed is not None else None
        self.entailment_ran = False
        # Reset alongside entailment_ran: a reason left over from an earlier
        # bundle would be reported as this bundle's skip, which it is not.
        self.skipped_reason = ""
        report = AccuracyReport()

        for tool_name in sorted(candidate):
            if permitted is not None and tool_name not in permitted:
                continue
            proposed = candidate[tool_name]
            before = baseline.get(tool_name)
            if before is None or proposed.description != before.description:
                report.checked += 1
                report.findings.extend(
                    self.check_tool(
                        tool_name,
                        proposed.description,
                        before.description if before else "",
                    )
                )
            for param in sorted(proposed.params):
                text = proposed.params[param]
                if before is not None and before.params.get(param) == text:
                    continue
                report.checked += 1
                report.findings.extend(
                    self.check_param(
                        tool_name,
                        param,
                        text,
                        before.params.get(param, "") if before else "",
                    )
                )

        report.entailment_ran = self.entailment_ran
        report.entailment_skipped = "" if self.entailment_ran else self.skipped_reason
        return report

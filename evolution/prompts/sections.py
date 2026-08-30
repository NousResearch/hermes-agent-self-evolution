"""Wrap hermes-agent's evolvable system-prompt sections as optimizable units.

Phase 3 is the widest blast radius in the plan. A skill only affects the
sessions that load it and a tool description only affects the sessions that
see that tool, but a system prompt section is in front of every single turn of
every single session. So this module is deliberately more about refusal than
about optimization: it decides what may be touched, how far it may move, and
when it may be written.

Four rules from PLAN.md are enforced here as code rather than as prose:

1. **Allowlist.** Only the four string constants in
   :data:`~evolution.core.artifact_io.EVOLVABLE_PROMPT_SECTIONS` are
   optimizable. ``PLATFORM_HINTS`` is listed as evolvable in PLAN.md, but in
   the real ``agent/prompt_builder.py`` it is a dict of eleven per-platform
   strings, not a string. Rewriting one platform's hint is a different
   operation with different accuracy rules ("don't tell Telegram to use ANSI
   codes"), so it is reported here as present-but-not-evolvable-in-this-phase
   instead of crashing the discovery pass or being silently dropped.

2. **Growth ceiling.** "Each section must not exceed its current size by >20%."
   The ceiling comes from ``EvolutionConfig.max_prompt_growth`` and is checked
   on the ratio, so a candidate that is exactly 20% larger passes and one char
   more fails.

3. **Identity preservation.** The identity section must still read as helpful,
   direct, and willing to admit uncertainty. That is checked with an explicit
   trait table: every trait needs at least one of its phrasings present, and a
   candidate that optimizes the uncertainty clause out of existence is rejected
   with the trait named.

4. **Caching boundary.** The assembled static prefix has to stay inside the
   budget the deployment reserves for it. Prompt caching keys on an exact
   prefix, so two things follow. The prefix must fit the budget, or every
   session pays full price for it. And an edit invalidates the cached prefix,
   which is the mechanical reason an evolved section deploys on the NEXT
   session and is never hot-swapped into a running one. That is why
   :func:`detect_active_session` exists and why :func:`write_sections` is a
   separate, explicit step rather than something the optimizer does on its own.

Nothing here calls an LLM or the network. Every write is routed through
``artifact_io.apply_prompt_section`` so the span-exact, re-parsed guarantees
hold, and :func:`verify_only_sections_changed` additionally proves that no
neighbouring constant moved.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

from evolution.core.artifact_io import (
    EVOLVABLE_PROMPT_SECTIONS,
    PromptSection,
    SourceSpan,
    apply_prompt_section,
    discover_prompt_sections,
)

__all__ = [
    "CHARS_PER_TOKEN",
    "CACHE_BLOCK_TOKENS",
    "DEFAULT_CACHE_BUDGET_TOKENS",
    "STRUCTURED_SECTION_NAMES",
    "IDENTITY_SECTION",
    "IDENTITY_TRAITS",
    "IdentityTrait",
    "SectionCheck",
    "SectionValidation",
    "EvolvableSection",
    "StructuredSection",
    "SectionInventory",
    "UnknownSection",
    "SectionWriteError",
    "ActiveSessionReport",
    "WriteResult",
    "restore_from_backup",
    "load_sections",
    "validate_section_names",
    "estimate_tokens",
    "constant_strings",
    "verify_only_sections_changed",
    "write_sections",
    "staged_prompt_write",
    "detect_active_session",
    "NEXT_SESSION_NOTICE",
]


# Rough token estimate. Four characters per token is the usual English prose
# approximation and it is deliberately provider-independent - this module must
# not depend on a tokenizer library that is not in pyproject.toml.
CHARS_PER_TOKEN = 4

# Anthropic-class models only cache a prefix once it reaches a minimum block
# size, and they account for cached prefixes in blocks of that size.
CACHE_BLOCK_TOKENS = 1024

# How much of the static prefix the deployment reserves for the assembled
# system prompt. Past this the prefix stops being a stable, cacheable prefix.
DEFAULT_CACHE_BUDGET_TOKENS = 8192

# Constants in prompt_builder.py that PLAN.md calls evolvable but that are not
# plain strings, so they cannot be routed through apply_prompt_section.
STRUCTURED_SECTION_NAMES = ("PLATFORM_HINTS",)

IDENTITY_SECTION = "DEFAULT_AGENT_IDENTITY"

NEXT_SESSION_NOTICE = (
    "Evolved sections take effect on the NEXT session. A running session has "
    "already assembled and cached its system prompt, so nothing is hot-swapped."
)


class UnknownSection(KeyError):
    """Raised when a caller names a section that is not in the inventory."""


class SectionWriteError(RuntimeError):
    """Raised when a write would change something other than the target sections."""


# ──────────────────────────────────────────────────────────────────────────
# Identity traits
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IdentityTrait:
    """One core trait the identity section has to keep expressing.

    A trait is satisfied when any of its phrasings appears. The alternatives
    exist so the optimizer can rewrite the sentence around a trait without
    tripping the check - what is forbidden is dropping the trait, not
    rephrasing it.
    """

    key: str
    label: str
    patterns: tuple[str, ...]

    def matched_by(self, text: str) -> Optional[str]:
        """Return the phrasing that satisfied this trait, or None."""
        for pattern in self.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return pattern
        return None


IDENTITY_TRAITS: tuple[IdentityTrait, ...] = (
    IdentityTrait(
        key="helpful",
        label="helpful",
        patterns=(
            r"\bhelpful\b",
            r"\bhelps?\b",
            r"\bgenuinely useful\b",
            r"\bassists?\b",
            r"\bassist(?:ing|ance)\b",
        ),
    ),
    IdentityTrait(
        key="direct",
        label="direct",
        patterns=(
            r"\bdirect(?:ly|ness)?\b",
            r"\bstraightforward\b",
            r"\bto the point\b",
            r"\bwithout (?:padding|preamble)\b",
            r"\bconcise\b",
        ),
    ),
    IdentityTrait(
        key="admits_uncertainty",
        label="admits uncertainty",
        patterns=(
            r"\badmits?\b[^.]{0,40}\buncertain",
            r"\buncertaint(?:y|ies)\b",
            r"\backnowledge[sd]?\b[^.]{0,40}\bunknown",
            r"\bsays? so when (?:you|it)\b[^.]{0,20}\b(?:do not|don'?t) know\b",
            r"\b(?:do not|don'?t) (?:pretend|guess)\b",
            r"\bflags? what (?:you|it) (?:do not|don'?t) know\b",
        ),
    ),
)


# ──────────────────────────────────────────────────────────────────────────
# Check results
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class SectionCheck:
    """One constraint outcome for a candidate section.

    ``severity`` separates a hard rejection from a caution. An "error" blocks
    the candidate outright. A "warning" blocks only under strict mode, which is
    how a cache-block boundary crossing is handled: it is a real cost signal
    but not, on its own, a reason to throw away a better prompt.
    """

    name: str
    passed: bool
    message: str
    details: str = ""
    severity: str = "error"

    @property
    def is_error(self) -> bool:
        """A failed check severe enough to block."""
        return not self.passed and self.severity == "error"

    @property
    def is_warning(self) -> bool:
        """A failed check that reports without blocking."""
        return not self.passed and self.severity == "warning"

    def to_dict(self) -> dict:
        """Serialise one section check."""
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "severity": self.severity,
        }


@dataclass
class SectionValidation:
    """Every constraint outcome for one candidate section."""

    section: str
    checks: list[SectionCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no error-severity check failed."""
        return not any(c.is_error for c in self.checks)

    @property
    def passed_strict(self) -> bool:
        """True when nothing failed at all, warnings included."""
        return all(c.passed for c in self.checks)

    @property
    def errors(self) -> list[SectionCheck]:
        """The blocking failures."""
        return [c for c in self.checks if c.is_error]

    @property
    def warnings(self) -> list[SectionCheck]:
        """The non-blocking failures."""
        return [c for c in self.checks if c.is_warning]

    def summary(self) -> str:
        """One line per check, marked pass or fail."""
        icons = {True: "✓", False: "✗"}
        return "\n".join(
            f"{icons[c.passed]} {c.name}: {c.message}" for c in self.checks
        )

    def to_dict(self) -> dict:
        """Serialise the section report, strict verdict included."""
        return {
            "section": self.section,
            "passed": self.passed,
            "passed_strict": self.passed_strict,
            "checks": [c.to_dict() for c in self.checks],
        }


# ──────────────────────────────────────────────────────────────────────────
# The optimizable unit
# ──────────────────────────────────────────────────────────────────────────


def estimate_tokens(text: str, chars_per_token: int = CHARS_PER_TOKEN) -> int:
    """Estimate a token count from a character count.

    Deliberately crude and dependency free. It is used for budget headroom, so
    it only has to be stable and roughly right, not exact.
    """
    if not text:
        return 0
    return -(-len(text) // max(1, chars_per_token))  # ceiling division


@dataclass
class EvolvableSection:
    """One system prompt section, with the ceiling it has to stay under."""

    name: str
    path: Path
    baseline_text: str
    # None when the section was constructed rather than discovered from source.
    # A write re-discovers the span anyway, since spans are only valid against
    # the exact source string they were read from.
    span: Optional[SourceSpan]
    max_growth: float = 0.2

    @classmethod
    def from_prompt_section(
        cls, section: PromptSection, max_growth: float = 0.2
    ) -> "EvolvableSection":
        """Wrap a discovered :class:`PromptSection` as an evolvable one."""
        return cls(
            name=section.name,
            path=section.path,
            baseline_text=section.text,
            span=section.span,
            max_growth=max_growth,
        )

    # ── shape ────────────────────────────────────────────────────────────
    @property
    def baseline_size(self) -> int:
        """Baseline length in characters."""
        return len(self.baseline_text)

    @property
    def baseline_tokens(self) -> int:
        """Estimated tokens in the baseline text."""
        return estimate_tokens(self.baseline_text)

    @property
    def max_chars(self) -> int:
        """Largest candidate size the growth ceiling allows, in characters."""
        return int(self.baseline_size * (1.0 + self.max_growth))

    @property
    def is_identity(self) -> bool:
        """True for the identity section, which is held to stricter rules."""
        return self.name == IDENTITY_SECTION

    def growth_of(self, candidate: str) -> float:
        """Fractional size change of *candidate* against the baseline."""
        return (len(candidate) - self.baseline_size) / max(1, self.baseline_size)

    # ── checks ───────────────────────────────────────────────────────────
    def check_non_empty(self, candidate: str) -> SectionCheck:
        """Reject a blank candidate, which would silently delete guidance."""
        if candidate.strip():
            return SectionCheck(
                "non_empty", True, f"{len(candidate):,} chars of content"
            )
        return SectionCheck(
            "non_empty",
            False,
            "candidate section is empty - a blank section silently removes guidance",
        )

    def check_growth(self, candidate: str) -> SectionCheck:
        """Reject a candidate that grows the section past its ceiling."""
        growth = self.growth_of(candidate)
        # Compare on the ratio, with a float-noise epsilon, so exactly +20%
        # passes and one character beyond it does not.
        if growth <= self.max_growth + 1e-9:
            return SectionCheck(
                "growth_ceiling",
                True,
                f"{growth:+.1%} vs baseline (ceiling {self.max_growth:+.0%}, "
                f"{len(candidate):,}/{self.max_chars:,} chars)",
            )
        return SectionCheck(
            "growth_ceiling",
            False,
            f"grew {growth:+.1%}, ceiling is {self.max_growth:+.0%} "
            f"({len(candidate):,} chars, max {self.max_chars:,})",
        )

    def check_identity_traits(self, candidate: str) -> list[SectionCheck]:
        """Verify the identity section still carries its core traits.

        Returns an empty list for every other section: the traits are a claim
        about the persona, not about memory or skills guidance.
        """
        if not self.is_identity:
            return []

        kept: list[str] = []
        lost: list[str] = []
        for trait in IDENTITY_TRAITS:
            if trait.matched_by(candidate):
                kept.append(trait.label)
            else:
                lost.append(trait.label)

        if not lost:
            return [
                SectionCheck(
                    "identity_traits",
                    True,
                    f"core traits retained: {', '.join(kept)}",
                )
            ]

        return [
            SectionCheck(
                "identity_traits",
                False,
                f"core traits dropped: {', '.join(lost)}",
                details=(
                    "PLAN.md requires the identity section to keep reading as "
                    "helpful, direct, and willing to admit uncertainty. "
                    f"Retained: {', '.join(kept) or 'none'}."
                ),
            )
        ]

    def validate(self, candidate: str) -> SectionValidation:
        """Run every section-local constraint against *candidate*."""
        checks = [self.check_non_empty(candidate), self.check_growth(candidate)]
        checks.extend(self.check_identity_traits(candidate))
        return SectionValidation(section=self.name, checks=checks)

    def to_dict(self) -> dict:
        """Serialise the evolvable section and its baseline measurements."""
        return {
            "name": self.name,
            "path": str(self.path),
            "baseline_size": self.baseline_size,
            "baseline_tokens": self.baseline_tokens,
            "max_chars": self.max_chars,
            "max_growth": self.max_growth,
            "is_identity": self.is_identity,
        }


@dataclass
class StructuredSection:
    """A prompt constant that is not a plain string, so not evolvable here.

    ``PLATFORM_HINTS`` is the only one today. It is surfaced rather than
    ignored so the CLI can say "this exists, it is out of scope for this
    phase" instead of leaving the operator to wonder why PLAN.md lists five
    evolvable sections and the tool found four.
    """

    name: str
    path: Path
    kind: str
    keys: tuple[str, ...]
    total_chars: int
    reason: str

    def to_dict(self) -> dict:
        """Serialise a structured prompt constant and its keys."""
        return {
            "name": self.name,
            "path": str(self.path),
            "kind": self.kind,
            "keys": list(self.keys),
            "total_chars": self.total_chars,
            "reason": self.reason,
        }


# ──────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────


def _prompt_builder_path(hermes_repo: Path) -> Path:
    return Path(hermes_repo) / "agent" / "prompt_builder.py"


def _iter_module_assignments(tree: ast.Module) -> Iterator[tuple[str, ast.AST]]:
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    yield target.id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.value is not None:
                yield stmt.target.id, stmt.value


def _discover_structured(
    path: Path, names: Sequence[str] = STRUCTURED_SECTION_NAMES
) -> list[StructuredSection]:
    """Describe the non-string prompt constants without trying to rewrite them."""
    if not path.is_file():
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError, UnicodeDecodeError):
        return []

    wanted = set(names)
    out: list[StructuredSection] = []
    for const_name, value in _iter_module_assignments(tree):
        if const_name not in wanted:
            continue
        if isinstance(value, ast.Dict):
            keys: list[str] = []
            total = 0
            for k, v in zip(value.keys, value.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.append(k.value)
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    total += len(v.value)
            out.append(
                StructuredSection(
                    name=const_name,
                    path=path,
                    kind="dict",
                    keys=tuple(keys),
                    total_chars=total,
                    reason=(
                        f"{const_name} is a dict of {len(keys)} per-platform strings, "
                        "not a single string. Per-platform text has its own accuracy "
                        "rules and is out of scope for this phase."
                    ),
                )
            )
        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
            # Defensive: if hermes-agent ever flattens it to a string, say so
            # rather than pretending it is still a dict.
            out.append(
                StructuredSection(
                    name=const_name,
                    path=path,
                    kind="str",
                    keys=(),
                    total_chars=len(value.value),
                    reason=(
                        f"{const_name} is a string here, but it is not on the "
                        "evolvable allowlist, so it is reported only."
                    ),
                )
            )
        else:
            out.append(
                StructuredSection(
                    name=const_name,
                    path=path,
                    kind=type(value).__name__,
                    keys=(),
                    total_chars=0,
                    reason=f"{const_name} is a computed value with no literal to rewrite.",
                )
            )
    return out


@dataclass
class SectionInventory:
    """Everything Phase 3 found in ``agent/prompt_builder.py``."""

    prompt_builder: Optional[Path]
    sections: list[EvolvableSection] = field(default_factory=list)
    structured: list[StructuredSection] = field(default_factory=list)
    missing: tuple[str, ...] = ()
    cache_budget_tokens: int = DEFAULT_CACHE_BUDGET_TOKENS

    # ── lookup ───────────────────────────────────────────────────────────
    @property
    def names(self) -> list[str]:
        """Every section name, in file order."""
        return [s.name for s in self.sections]

    def __len__(self) -> int:
        return len(self.sections)

    def __iter__(self) -> Iterator[EvolvableSection]:
        return iter(self.sections)

    def get(self, name: str) -> EvolvableSection:
        """The section named *name*, or raise UnknownSection naming what does exist."""
        for section in self.sections:
            if section.name == name:
                return section
        raise UnknownSection(
            f"{name} was not found in {self.prompt_builder} "
            f"(found: {', '.join(self.names) or 'nothing'})"
        )

    # ── assembly and caching ─────────────────────────────────────────────
    def assembled_prompt(self, overrides: Optional[Mapping[str, str]] = None) -> str:
        """Join the evolvable sections into a usable system prompt.

        Sections are ordered as PLAN.md lists them, with *overrides* substituted
        in. This is what gets handed to ``--ephemeral_system_prompt``, so it
        contains only real text. The volatile parts of the deployed prompt - the
        memory block, the skills index, context files - are absent on purpose:
        they are per-session data, not the artifact under optimization.
        """
        overrides = dict(overrides or {})
        order = {name: i for i, name in enumerate(EVOLVABLE_PROMPT_SECTIONS)}
        ordered = sorted(self.sections, key=lambda s: (order.get(s.name, 99), s.name))
        return "\n\n".join(overrides.get(s.name, s.baseline_text) for s in ordered)

    def assembled_preview(self, overrides: Optional[Mapping[str, str]] = None) -> str:
        """The assembled prompt padded to its worst case, for token counting.

        Exactly one platform hint is present in any real session, so the widest
        one is added as filler. This string is a measuring stick and nothing
        else - never send it to a model.
        """
        parts = [self.assembled_prompt(overrides)]
        widest = max(
            (s.total_chars for s in self.structured if s.kind == "dict"), default=0
        )
        if widest:
            parts.append("x" * widest)
        return "\n\n".join(parts)

    def estimated_tokens(self, overrides: Optional[Mapping[str, str]] = None) -> int:
        """Estimated token count of the assembled prompt."""
        return estimate_tokens(self.assembled_preview(overrides))

    def cache_blocks(self, overrides: Optional[Mapping[str, str]] = None) -> int:
        """Cache blocks the assembled prompt would occupy, rounded up."""
        tokens = self.estimated_tokens(overrides)
        return -(-tokens // CACHE_BLOCK_TOKENS)

    def check_caching_boundary(
        self,
        overrides: Optional[Mapping[str, str]] = None,
        budget_tokens: Optional[int] = None,
    ) -> list[SectionCheck]:
        """Check the assembled prefix against the caching budget.

        Two outcomes, deliberately at different severities. Blowing the budget
        is an error: the prefix stops being cacheable and every session pays
        for it. Pushing the prefix into another cache block is a warning: it
        costs real money per session but a genuinely better prompt can be worth
        it, so the decision belongs to the operator (or to --strict-gates).
        """
        budget = int(budget_tokens or self.cache_budget_tokens)
        tokens = self.estimated_tokens(overrides)
        baseline_tokens = self.estimated_tokens(None)
        blocks = self.cache_blocks(overrides)
        baseline_blocks = self.cache_blocks(None)

        checks: list[SectionCheck] = []

        if tokens <= budget:
            checks.append(
                SectionCheck(
                    "caching_budget",
                    True,
                    f"~{tokens:,} tokens of {budget:,} budgeted "
                    f"({budget - tokens:,} headroom)",
                    details=f"baseline ~{baseline_tokens:,} tokens",
                )
            )
        else:
            checks.append(
                SectionCheck(
                    "caching_budget",
                    False,
                    f"~{tokens:,} tokens exceeds the {budget:,} token caching budget "
                    f"by {tokens - budget:,}",
                    details=(
                        "Past the budget the static prefix is no longer a stable "
                        "cacheable prefix and every session pays full price for it."
                    ),
                )
            )

        if blocks <= baseline_blocks:
            checks.append(
                SectionCheck(
                    "cache_block_stability",
                    True,
                    f"still {blocks} cache block(s) of {CACHE_BLOCK_TOKENS} tokens",
                )
            )
        else:
            checks.append(
                SectionCheck(
                    "cache_block_stability",
                    False,
                    f"crosses into cache block {blocks} (baseline used {baseline_blocks})",
                    details=(
                        "Every session pays for the extra block. Worth it only if "
                        "the behavioral gain is real."
                    ),
                    severity="warning",
                )
            )

        return checks

    # ── validation ───────────────────────────────────────────────────────
    def validate(
        self,
        name: str,
        candidate: str,
        budget_tokens: Optional[int] = None,
    ) -> SectionValidation:
        """Validate a candidate for *name*, caching boundary included."""
        section = self.get(name)
        validation = section.validate(candidate)
        validation.checks.extend(
            self.check_caching_boundary({name: candidate}, budget_tokens)
        )
        return validation

    def to_dict(self) -> dict:
        """Serialise the whole discovery result, missing sections included."""
        return {
            "prompt_builder": str(self.prompt_builder) if self.prompt_builder else None,
            "sections": [s.to_dict() for s in self.sections],
            "structured": [s.to_dict() for s in self.structured],
            "missing": list(self.missing),
            "cache_budget_tokens": self.cache_budget_tokens,
            "assembled_tokens": self.estimated_tokens(),
            "cache_blocks": self.cache_blocks(),
        }


def validate_section_names(names: Sequence[str]) -> list[str]:
    """Return the requested names that are not on the evolvable allowlist.

    Kept separate from discovery so the CLI can reject a bad ``--section``
    before it touches the filesystem, and so the error can name PLATFORM_HINTS
    specifically rather than saying "not found".
    """
    return [n for n in names if n not in EVOLVABLE_PROMPT_SECTIONS]


def load_sections(
    hermes_repo: Path,
    names: Sequence[str] = EVOLVABLE_PROMPT_SECTIONS,
    max_growth: float = 0.2,
    cache_budget_tokens: int = DEFAULT_CACHE_BUDGET_TOKENS,
) -> SectionInventory:
    """Discover the evolvable prompt sections in a hermes-agent checkout.

    Unknown names are refused up front. Names that are on the allowlist but
    absent from the file are reported in ``missing`` rather than raising, so a
    hermes-agent that renames a constant produces a clear inventory instead of
    a stack trace.
    """
    unknown = validate_section_names(names)
    if unknown:
        raise UnknownSection(
            f"not evolvable prompt sections: {', '.join(unknown)} "
            f"(allowed: {', '.join(EVOLVABLE_PROMPT_SECTIONS)})"
        )

    path = _prompt_builder_path(hermes_repo)
    if not path.is_file():
        return SectionInventory(
            prompt_builder=None,
            sections=[],
            structured=[],
            missing=tuple(names),
            cache_budget_tokens=cache_budget_tokens,
        )

    found = discover_prompt_sections(Path(hermes_repo), names=tuple(names))
    sections = [EvolvableSection.from_prompt_section(s, max_growth) for s in found]
    discovered = {s.name for s in sections}

    return SectionInventory(
        prompt_builder=path,
        sections=sections,
        structured=_discover_structured(path),
        missing=tuple(n for n in names if n not in discovered),
        cache_budget_tokens=cache_budget_tokens,
    )


# ──────────────────────────────────────────────────────────────────────────
# Active session detection
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ActiveSessionReport:
    """Best-effort answer to "is a Hermes session running right now?".

    Detection is best effort by design. A false positive costs an operator one
    retry; a false negative would let an evolved prompt land under a session
    that has already cached the old one, which is exactly the hot-swap PLAN.md
    forbids. So the checks are conservative and every signal is named in
    ``evidence`` so the refusal can be argued with.
    """

    active: bool
    evidence: tuple[str, ...] = ()
    checked: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        """One line saying whether a live session was detected, and on what evidence."""
        if not self.active:
            return "no active Hermes session detected"
        return "active session: " + "; ".join(self.evidence)

    def to_dict(self) -> dict:
        """Serialise the session probe and everything it checked."""
        return {
            "active": self.active,
            "evidence": list(self.evidence),
            "checked": list(self.checked),
        }


_SESSION_ENV_VARS = (
    "HERMES_SESSION_ID",
    "HERMES_SESSION_PLATFORM",
    "HERMES_KANBAN_TASK",
)

_LOCK_GLOBS = ("*.lock", "sessions/*.lock", "locks/*.lock")
_PID_FILES = ("run.pid", "agent.pid", "hermes.pid")


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def detect_active_session(
    hermes_home: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> ActiveSessionReport:
    """Look for signs that a Hermes session is live.

    Three signals, all cheap and all local: a session environment variable
    (which means *this* process was launched inside a session), a lock file
    under the Hermes home, and a pid file whose process still exists.
    """
    env = os.environ if env is None else env
    home = Path(
        hermes_home
        or env.get("HERMES_HOME")
        or (Path.home() / ".hermes")
    ).expanduser()

    evidence: list[str] = []
    checked: list[str] = [f"env {', '.join(_SESSION_ENV_VARS)}", f"home {home}"]

    for var in _SESSION_ENV_VARS:
        value = env.get(var)
        if value:
            evidence.append(f"{var}={value}")

    if home.is_dir():
        for pattern in _LOCK_GLOBS:
            for lock in sorted(home.glob(pattern)):
                evidence.append(f"lock file {lock}")
        for pid_name in _PID_FILES:
            pid_path = home / pid_name
            if not pid_path.is_file():
                continue
            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip() or "0")
            except (ValueError, OSError):
                continue
            if pid > 0 and _pid_is_alive(pid):
                evidence.append(f"live pid {pid} from {pid_path}")

    return ActiveSessionReport(
        active=bool(evidence),
        evidence=tuple(evidence),
        checked=tuple(checked),
    )


# ──────────────────────────────────────────────────────────────────────────
# Writes
# ──────────────────────────────────────────────────────────────────────────


def constant_strings(source: str) -> dict[str, str]:
    """Map every module-level string constant in *source* to its value.

    Used to prove that a rewrite touched only the sections it claimed to. The
    values are the parsed strings, not the source spelling, so a rewrite that
    reflows an implicit-concat block without changing its value is correctly
    treated as unchanged.
    """
    tree = ast.parse(source)
    out: dict[str, str] = {}
    for name, value in _iter_module_assignments(tree):
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            out[name] = value.value
    return out


def verify_only_sections_changed(
    before: str, after: str, expected: Sequence[str]
) -> None:
    """Raise :class:`SectionWriteError` if any other constant moved.

    ``apply_prompt_section`` already guarantees the file still parses and that
    only one span was replaced. This is the complementary check from the other
    direction: parse both versions and compare every module-level string
    constant, so a neighbouring constant that shifted, vanished, or appeared is
    caught by value rather than by trusting the span arithmetic.
    """
    try:
        after_constants = constant_strings(after)
    except SyntaxError as exc:
        raise SectionWriteError(f"rewritten prompt_builder does not parse: {exc}") from exc

    before_constants = constant_strings(before)
    expected_set = set(expected)

    added = sorted(set(after_constants) - set(before_constants))
    removed = sorted(set(before_constants) - set(after_constants))
    if added or removed:
        raise SectionWriteError(
            f"module-level string constants changed (added={added}, removed={removed})"
        )

    drifted = sorted(
        name
        for name, value in before_constants.items()
        if name not in expected_set and after_constants[name] != value
    )
    if drifted:
        raise SectionWriteError(
            f"rewrite changed constants it was not asked to change: {', '.join(drifted)}"
        )


@dataclass
class WriteResult:
    """What a (possibly dry) write to prompt_builder.py did."""

    path: Path
    updated: tuple[str, ...]
    source: str
    before_chars: int
    after_chars: int
    dry_run: bool

    def to_dict(self) -> dict:
        """Serialise the write result, with sizes before and after."""
        return {
            "path": str(self.path),
            "updated": list(self.updated),
            "before_chars": self.before_chars,
            "after_chars": self.after_chars,
            "dry_run": self.dry_run,
        }


def write_sections(
    hermes_repo: Path,
    updates: Mapping[str, str],
    dry_run: bool = False,
) -> WriteResult:
    """Rewrite one or more prompt sections in place.

    Spans are only valid against the exact source they were read from, so each
    section is applied to a staged copy and the next section is re-discovered
    from that copy. The staged file is the only thing that is edited until
    every update has landed and :func:`verify_only_sections_changed` has
    approved the result, which also means ``dry_run=True`` exercises the whole
    path without the repo ever being touched.
    """
    if not updates:
        raise SectionWriteError("no sections to write")

    unknown = validate_section_names(list(updates))
    if unknown:
        raise SectionWriteError(
            f"refusing to write non-evolvable sections: {', '.join(unknown)}"
        )

    path = _prompt_builder_path(hermes_repo)
    if not path.is_file():
        raise SectionWriteError(f"prompt_builder.py not found at {path}")

    original = path.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="hase-prompt-") as tmp:
        staging = Path(tmp)
        (staging / "agent").mkdir(parents=True)
        staged_file = staging / "agent" / "prompt_builder.py"
        staged_file.write_text(original, encoding="utf-8")

        for name, new_text in updates.items():
            found = {s.name: s for s in discover_prompt_sections(staging)}
            section = found.get(name)
            if section is None:
                raise SectionWriteError(
                    f"{name} is not a string constant in {path} - nothing to rewrite"
                )
            current = staged_file.read_text(encoding="utf-8")
            staged_file.write_text(
                apply_prompt_section(current, section, new_text), encoding="utf-8"
            )

        final = staged_file.read_text(encoding="utf-8")

    verify_only_sections_changed(original, final, list(updates))

    if not dry_run:
        path.write_text(final, encoding="utf-8")

    return WriteResult(
        path=path,
        updated=tuple(updates),
        source=final,
        before_chars=len(original),
        after_chars=len(final),
        dry_run=dry_run,
    )


@contextmanager
def staged_prompt_write(
    hermes_repo: Path,
    updates: Mapping[str, str],
    enabled: bool = True,
    backup_path: Optional[Path] = None,
) -> Iterator[Optional[WriteResult]]:
    """Temporarily put candidate sections in place, then always put them back.

    Gates that actually exercise the agent - pytest, benchmarks - have to see
    the candidate on disk, and there is no ephemeral-prompt channel for those.
    So the file is written, the gates run, and the original is restored in a
    ``finally``. A copy of the original is left at *backup_path* first, so a
    process killed mid-gate can still be recovered by hand.

    The restore writes the captured original back verbatim. Anything that
    edits ``prompt_builder.py`` while the block is open - a gate, a tool
    under test, a parallel process - is overwritten when it closes. The block
    owns the file for its whole duration; that is the contract, not a hazard.
    """
    if not enabled or not updates:
        yield None
        return

    path = _prompt_builder_path(hermes_repo)
    if not path.is_file():
        raise SectionWriteError(f"prompt_builder.py not found at {path}")

    original = path.read_text(encoding="utf-8")
    if backup_path is not None:
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(original, encoding="utf-8")

    result = write_sections(hermes_repo, updates, dry_run=False)
    try:
        yield result
    finally:
        path.write_text(original, encoding="utf-8")


def restore_from_backup(hermes_repo: Path, backup_path: Path) -> Path:
    """Put a backup taken by :func:`staged_prompt_write` back in place."""
    path = _prompt_builder_path(hermes_repo)
    shutil.copyfile(Path(backup_path), path)
    return path

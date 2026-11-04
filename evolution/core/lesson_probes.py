"""Lesson probes: a lesion-validated regression gate for incident-derived lessons.

The problem this solves (see issue #200)
----------------------------------------
A skill accumulates *lessons*: passages that exist because something once went
wrong ("gate the commit on the test command's exit code, not its output";
"refuse to merge when a required review is missing"). The evaluation set is
drawn from ordinary tasks, so a lesson that only matters in a rare situation is
invisible to the fitness score. Deleting it costs nothing and reads as "more
concise", so the optimizer deletes it and scores higher.

Checks that compare the candidate's *text* to the baseline's text can catch a
deletion, but they cannot tell a dropped lesson from a reworded one, and they
cannot tell whether the lesson ever changed the agent's behavior at all.

The idea
--------
Treat every lesson as a regression test, and validate the test itself:

    lesson   a passage in the skill that exists because of a past failure
    probe    a scenario that reproduces that failure, plus an expectation on
             the agent's output
    lesion   the baseline skill with that one lesson removed

A probe is trusted only if it *discriminates*: the baseline passes it and the
lesion fails it. That single comparison sorts probes into four kinds:

    discriminating   baseline passes, lesion fails    -> gates the candidate
    dead             baseline passes, lesion passes   -> tests nothing; reported
    broken           baseline fails                   -> lesson is already inert
    orphaned         anchor text not found in baseline

Only discriminating probes gate. The candidate is rejected if it fails a probe
that the baseline passes and the lesion fails, i.e. it has lost a behavior the
lesson demonstrably provided. Rewording that keeps the behavior passes;
deletion that loses the behavior fails. No text similarity is involved.

Because probes are validated by lesion rather than by trust, they can come from
a sidecar file written by a human *or* from a generator (an LLM) without the
generator's mistakes leaking into the gate: a hallucinated probe that passes
with or without the lesson is simply dead.

This module has no LLM dependency. The agent is a ``Responder`` callable
``(skill_text, scenario) -> output`` injected by the caller; the CLI wires it to
``SkillModule`` and tests use a deterministic stub.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Optional

# (skill_text, scenario) -> agent output
Responder = Callable[[str, str], str]

# lesson_text -> probe fields (scenario / must_include / ...) or None
ProbeGenerator = Callable[[str], Optional[dict]]


# ── Probe definition ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LessonProbe:
    """A scenario that should only be handled correctly *because* of a lesson."""

    lesson_id: str
    anchor: str  # exact substring of the lesson as it appears in the baseline body
    scenario: str  # task_input that reproduces the original incident
    must_include: tuple[str, ...] = ()  # every item must appear in the output (case-insensitive)
    must_not_include: tuple[str, ...] = ()  # no item may appear in the output
    must_match: Optional[str] = None  # regex that must match the output (IGNORECASE | DOTALL)
    source: str = "file"  # "file" or "generated"

    def check(self, output: str) -> tuple[bool, str]:
        """Evaluate the expectation against an agent output."""
        low = output.lower()
        missing = [s for s in self.must_include if s.lower() not in low]
        if missing:
            return False, f"missing: {missing}"
        present = [s for s in self.must_not_include if s.lower() in low]
        if present:
            return False, f"forbidden present: {present}"
        if self.must_match and not re.search(self.must_match, output, re.IGNORECASE | re.DOTALL):
            return False, f"regex not matched: {self.must_match!r}"
        return True, "ok"

    @classmethod
    def from_dict(cls, d: dict) -> "LessonProbe":
        if not d.get("lesson_id") or not d.get("anchor") or not d.get("scenario"):
            raise ValueError("probe needs lesson_id, anchor and scenario")
        if not (d.get("must_include") or d.get("must_not_include") or d.get("must_match")):
            raise ValueError(f"probe {d['lesson_id']!r} has no expectation")
        return cls(
            lesson_id=str(d["lesson_id"]),
            anchor=str(d["anchor"]),
            scenario=str(d["scenario"]),
            must_include=tuple(d.get("must_include") or ()),
            must_not_include=tuple(d.get("must_not_include") or ()),
            must_match=d.get("must_match"),
            source=str(d.get("source", "file")),
        )


def load_probes(path: Path) -> tuple[list[LessonProbe], list[str]]:
    """Load probes from a JSONL file. Returns (probes, errors); bad lines are skipped."""
    probes: list[LessonProbe] = []
    errors: list[str] = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            probes.append(LessonProbe.from_dict(json.loads(line)))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            errors.append(f"{path.name}:{n}: {e}")
    return probes, errors


# ── Lesion ──────────────────────────────────────────────────────────────────


def _paragraphs(text: str) -> list[str]:
    return re.split(r"\n[ \t]*\n", text)


def _lesson_body(paragraph: str) -> str:
    """The prose of a paragraph: leading heading lines dropped, code fences ignored."""
    lines = paragraph.strip().splitlines()
    while lines and lines[0].lstrip().startswith("#"):
        lines.pop(0)
    text = "\n".join(lines).strip()
    return "" if text.startswith("```") else text


def lesion_text(baseline: str, anchor: str) -> Optional[str]:
    """Return the baseline with the paragraph containing ``anchor`` removed.

    Removing the whole paragraph (rather than the anchor characters alone)
    keeps the lesion a well-formed document and matches what a deleting
    optimizer actually does. If the anchor spans paragraphs, only the anchor
    itself is removed. Returns None when the anchor is not present.
    """
    if not anchor or anchor not in baseline:
        return None
    paras = _paragraphs(baseline)
    kept = [p for p in paras if anchor not in p]
    if len(kept) == len(paras):  # anchor crosses a paragraph boundary
        return baseline.replace(anchor, "", 1)
    return "\n\n".join(kept)


# ── Outcomes ────────────────────────────────────────────────────────────────


class ProbeStatus(str, Enum):
    DISCRIMINATING = "discriminating"
    DEAD = "dead"
    BROKEN = "broken"
    ORPHANED = "orphaned"
    UNVERIFIED = "unverified"  # lesion check disabled; baseline passes, trusted as-is


@dataclass
class ProbeOutcome:
    probe: LessonProbe
    status: ProbeStatus
    baseline_passed: Optional[bool] = None
    lesion_passed: Optional[bool] = None
    candidate_passed: Optional[bool] = None
    detail: str = ""

    @property
    def gates(self) -> bool:
        return self.status in (ProbeStatus.DISCRIMINATING, ProbeStatus.UNVERIFIED)

    @property
    def regressed(self) -> bool:
        return self.gates and self.candidate_passed is False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["gates"] = self.gates
        d["regressed"] = self.regressed
        return d


@dataclass
class LessonProbeReport:
    outcomes: list[ProbeOutcome] = field(default_factory=list)
    unprobed_lessons: list[str] = field(default_factory=list)
    responder_calls: int = 0

    def by_status(self, status: ProbeStatus) -> list[ProbeOutcome]:
        return [o for o in self.outcomes if o.status == status]

    @property
    def gating(self) -> list[ProbeOutcome]:
        return [o for o in self.outcomes if o.gates]

    @property
    def regressions(self) -> list[ProbeOutcome]:
        return [o for o in self.outcomes if o.regressed]

    @property
    def passed(self) -> bool:
        return not self.regressions

    def summary(self) -> str:
        n = len(self.outcomes)
        parts = [
            f"{len(self.gating)}/{n} probes gate",
            f"{len(self.regressions)} regression(s)",
        ]
        for st in (ProbeStatus.DEAD, ProbeStatus.BROKEN, ProbeStatus.ORPHANED):
            k = len(self.by_status(st))
            if k:
                parts.append(f"{k} {st.value}")
        if self.unprobed_lessons:
            parts.append(f"{len(self.unprobed_lessons)} lesson-like passage(s) without a probe")
        return "; ".join(parts)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "summary": self.summary(),
            "responder_calls": self.responder_calls,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "unprobed_lessons": list(self.unprobed_lessons),
        }


# ── Runner ──────────────────────────────────────────────────────────────────


def run_lesson_probes(
    baseline_body: str,
    candidate_body: str,
    probes: Iterable[LessonProbe],
    responder: Responder,
    *,
    lesion_check: bool = True,
) -> LessonProbeReport:
    """Classify every probe by lesion, then test the candidate on the gating ones.

    Responder budget per probe: baseline (1) + lesion (1, if enabled) +
    candidate (1, only for gating probes). Dead, broken and orphaned probes
    never cost a candidate call.
    """
    probes = list(probes)
    report = LessonProbeReport()
    calls = 0

    def ask(skill_text: str, scenario: str) -> str:
        nonlocal calls
        calls += 1
        return responder(skill_text, scenario) or ""

    for probe in probes:
        lesion = lesion_text(baseline_body, probe.anchor)
        if lesion is None:
            report.outcomes.append(
                ProbeOutcome(probe, ProbeStatus.ORPHANED, detail="anchor not found in baseline")
            )
            continue

        base_ok, base_why = probe.check(ask(baseline_body, probe.scenario))
        if not base_ok:
            report.outcomes.append(
                ProbeOutcome(probe, ProbeStatus.BROKEN, baseline_passed=False,
                             detail=f"baseline fails its own probe ({base_why})")
            )
            continue

        if lesion_check:
            les_ok, _ = probe.check(ask(lesion, probe.scenario))
            if les_ok:
                report.outcomes.append(
                    ProbeOutcome(probe, ProbeStatus.DEAD, baseline_passed=True, lesion_passed=True,
                                 detail="passes without the lesson; probe does not test it")
                )
                continue
            status, les = ProbeStatus.DISCRIMINATING, False
        else:
            status, les = ProbeStatus.UNVERIFIED, None

        cand_ok, cand_why = probe.check(ask(candidate_body, probe.scenario))
        report.outcomes.append(
            ProbeOutcome(probe, status, baseline_passed=True, lesion_passed=les,
                         candidate_passed=cand_ok,
                         detail="ok" if cand_ok else f"candidate lost the behavior ({cand_why})")
        )

    report.unprobed_lessons = find_unprobed_lessons(baseline_body, probes)
    report.responder_calls = calls
    return report


# ── Finding lessons that have no probe ──────────────────────────────────────

# Words that tend to mark a rule born from a failure. Deliberately loose: this
# feeds a visibility report, never a verdict.
_LESSON_MARKERS = re.compile(
    r"(\bnever\b|\balways\b|\bmust(?: not)?\b|\bdo not\b|\bdon't\b|\brefuse\b|\breject\b|"
    r"\bonly (?:if|when)\b|\brather than\b|\binstead of\b|\bfail(?:s|ed|ure)?\b|\bbroke\b|"
    r"\bincident\b|\blesson\b|\bpost-?mortem\b|\bregression\b|\blearned\b|\broot cause\b|"
    r"\bbecause\b|(?:PR|issue)\s*#\d+)",
    re.IGNORECASE,
)


def find_unprobed_lessons(body: str, probes: list[LessonProbe], *, preview: int = 90) -> list[str]:
    """Paragraphs that read like incident lessons but are not anchored by any probe.

    This is a heuristic for *visibility*, not a gate: it tells the operator
    which passages the optimizer may delete for free. Headings and code
    fences are ignored.
    """
    anchored = [p.anchor for p in probes]
    found: list[str] = []
    for para in _paragraphs(body):
        text = _lesson_body(para)
        if not text or not _LESSON_MARKERS.search(text):
            continue
        if any(a in para for a in anchored):
            continue
        one_line = " ".join(text.split())
        found.append(one_line if len(one_line) <= preview else one_line[: preview - 1] + "…")
    return found


# ── Probe generation hook ───────────────────────────────────────────────────


def generate_probes(
    body: str,
    generator: ProbeGenerator,
    existing: list[LessonProbe],
    *,
    max_probes: int = 20,
) -> list[LessonProbe]:
    """Ask ``generator`` for a probe per unprobed lesson-like paragraph.

    The generator may be an LLM. Its output is *not* trusted: every generated
    probe still goes through the lesion check in ``run_lesson_probes``, where
    a probe that cannot distinguish the lesson from its absence is marked dead
    and ignored. The anchor is always the paragraph itself, so generated
    probes can never be orphaned.
    """
    anchored = [p.anchor for p in existing]
    out: list[LessonProbe] = []
    for idx, para in enumerate(_paragraphs(body)):
        if len(out) >= max_probes:
            break
        text = _lesson_body(para)
        if not text or not _LESSON_MARKERS.search(text) or any(a in para for a in anchored):
            continue
        try:
            spec = generator(text)
        except Exception:
            spec = None
        if not spec:
            continue
        spec = dict(spec)
        spec.setdefault("lesson_id", f"gen-{idx}")
        spec["anchor"] = text
        spec["source"] = "generated"
        try:
            out.append(LessonProbe.from_dict(spec))
        except ValueError:
            continue
    return out

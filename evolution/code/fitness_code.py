"""Composite fitness for an evolved hermes-agent code candidate.

PLAN.md specifies four signals for Phase 4 and one of them is not like the
others:

    pytest          hard gate - any failure is immediate rejection
    benchmarks      broad capability check, scored
    bug repro       did this mutation fix the bug it was aimed at
    code quality    the heuristics in safety.py, scored

"Hard gate" is taken literally here. A candidate whose test run fails scores
``0.0``, not "0.0 for tests and full marks everywhere else". There is no
weighting that lets a red suite through, because a code change that breaks
tests is not a partially good code change.

Availability is never assumed. hermes-agent ships no benchmark directories
today, so :mod:`evolution.core.gates` reports them ``UNAVAILABLE`` and this
module drops them from the weighted average rather than scoring an absent
benchmark as a zero (which would reject everything) or a pass (which would
certify nothing). Under ``strict=True`` an unavailable gate is a rejection
instead, for a release process that must prove every gate actually ran.

Two of those signals are measurements rather than facts, and are treated as
such. A reproduction script run once is a single Bernoulli trial, so
``repro_runs`` runs it n times and the fix rate carries a Wilson interval:
"fixed" becomes a measured rate instead of one lucky pass, and a patch that
clears the reproduction three times in five is visibly not a fix. And when
per-test outcomes exist for both the baseline and the candidate, the two runs
are compared as paired binary outcomes over the tests they share, which says
whether the suite actually moved rather than only whether it is still green.
Neither softens anything: pytest failing is still an outright rejection, and
the statistics are information layered on top of that verdict, never a vote
against it.

Every score carries its **evidence coverage**: the fraction of the intended
weight that was actually measured. Renormalizing over whichever components
happened to be available keeps the number on a 0-1 scale, but it also lets
0.85 from a single heuristic look exactly like 0.85 from tests plus benchmarks
plus a reproduction. The coverage figure travels next to the score everywhere
the score appears so the two cannot be mistaken for each other.

The candidate source is expected to be **on disk already** when
:meth:`CodeFitnessEvaluator.evaluate` is called - pytest and the reproduction
script run against the working tree, not against a string. :class:`CodeOrganism`
is what puts it there.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

from evolution.core.gates import (
    GateChain,
    GateResult,
    GateStatus,
    run_benchmark_gate,
    run_pytest_gate,
)
from evolution.core.stats import (
    Interval,
    PairedBinary,
    compare_paired_binary,
    wilson_interval,
)
from evolution.code.safety import (
    QualitySignals,
    SafetyReport,
    quality_signals,
    run_safety_checks,
)

__all__ = [
    "FitnessError",
    "ReproStatus",
    "ReproResult",
    "ReproTrials",
    "BugReproduction",
    "PASSING_OUTCOMES",
    "FAILING_OUTCOMES",
    "parse_pytest_outcomes",
    "pytest_outcomes_from_result",
    "PerTestGateResult",
    "PerTestPytestRunner",
    "SuiteComparison",
    "compare_test_suites",
    "FitnessWeights",
    "CodeFitness",
    "DEFAULT_RANKING_RESOLUTION",
    "CandidateRanking",
    "rank_candidates",
    "BaselineSnapshot",
    "CodeFitnessEvaluator",
]


class FitnessError(RuntimeError):
    """Raised when a candidate cannot be scored honestly."""


# ──────────────────────────────────────────────────────────────────────────
# Bug reproduction
# ──────────────────────────────────────────────────────────────────────────


class ReproStatus(str, Enum):
    """Whether the bug a candidate claims to fix still reproduces."""
    FIXED = "fixed"
    PRESENT = "present"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


# A repro script can be explicit about its verdict instead of relying on its
# exit code, which matters for scripts that also print a diagnostic.
FIXED_MARKERS = ("BUG_FIXED", "BUG FIXED")
PRESENT_MARKERS = ("BUG_PRESENT", "BUG_REPRODUCED", "BUG PRESENT")


# 0.500 - 0.480 is 0.020000000000000018, so a margin equal to the resolution
# tested as strictly greater and got reported as resolved.
_FLOAT_SLACK = 1e-9


@dataclass
class ReproResult:
    """What one run of a reproduction script said."""

    status: ReproStatus
    message: str
    exit_code: Optional[int] = None
    details: str = ""
    duration_s: float = 0.0

    @property
    def fixed(self) -> bool:
        """True when this run reports the bug no longer reproduces."""
        return self.status is ReproStatus.FIXED

    @property
    def measured(self) -> bool:
        """True when the script ran and returned a verdict."""
        return self.status in (ReproStatus.FIXED, ReproStatus.PRESENT)

    def to_dict(self) -> dict:
        """Serialise the reproduction result for the run artifacts."""
        return {
            "status": self.status.value,
            "message": self.message,
            "exit_code": self.exit_code,
            "details": self.details,
            "duration_s": round(self.duration_s, 2),
        }


@dataclass
class ReproTrials:
    """Several runs of one reproduction script, and the fix rate they measured.

    One run is one Bernoulli trial. Flaky reproductions are the norm in agent
    tooling - a race, a temp directory, a timeout under load - so a single
    clean run cannot tell a fix from a lucky draw. Running the script n times
    turns "it passed" into a rate with an interval around it, which is a claim
    a reviewer can check.

    The verdict stays conservative. :attr:`fixed` is true only when every run
    reported the bug fixed: a patch that clears the reproduction three times in
    five has not fixed the bug, and the interval says how little three of five
    pins down even when it is all the evidence there is.
    """

    runs: list[ReproResult] = field(default_factory=list)
    confidence: float = 0.95

    @property
    def n(self) -> int:
        """How many times the script was run."""
        return len(self.runs)

    @property
    def measured_runs(self) -> int:
        """Runs that returned a verdict. A timeout measures nothing."""
        return sum(1 for r in self.runs if r.measured)

    @property
    def fixes(self) -> int:
        """How many runs reported the bug fixed."""
        return sum(1 for r in self.runs if r.fixed)

    @property
    def fix_rate(self) -> float:
        """Fraction of measured runs that reported the bug fixed."""
        return self.fixes / self.measured_runs if self.measured_runs else 0.0

    def interval(self) -> Interval:
        """Wilson interval on the fix rate, over the runs that measured one.

        Wilson rather than Wald because these counts are tiny and sit against
        the boundary: at 1 of 1 Wald reports [1.0, 1.0], which is a lie about
        what one run establishes.
        """
        return wilson_interval(self.fixes, self.measured_runs, self.confidence)

    @property
    def status(self) -> ReproStatus:
        """The aggregate verdict, worst news first.

        A single run that still reproduces outranks any number of clean ones,
        because the bug demonstrably survives the patch.
        """
        if not self.runs:
            return ReproStatus.UNAVAILABLE
        if any(r.status is ReproStatus.PRESENT for r in self.runs):
            return ReproStatus.PRESENT
        if any(r.status is ReproStatus.ERROR for r in self.runs):
            return ReproStatus.ERROR
        if all(r.status is ReproStatus.FIXED for r in self.runs):
            return ReproStatus.FIXED
        return ReproStatus.UNAVAILABLE

    @property
    def fixed(self) -> bool:
        """True when the representative run reports the bug fixed."""
        return self.status is ReproStatus.FIXED

    @property
    def measured(self) -> bool:
        """True when the runs produced a verdict worth scoring."""
        return self.status in (ReproStatus.FIXED, ReproStatus.PRESENT)

    @property
    def flaky(self) -> bool:
        """True when the reproduction disagreed with itself across runs."""
        return 0 < self.fixes < self.measured_runs

    @property
    def reproduced(self) -> bool:
        """True when at least one run showed the bug - there is something to fix."""
        return any(r.status is ReproStatus.PRESENT for r in self.runs)

    @property
    def representative(self) -> Optional[ReproResult]:
        """The single run that stands for the set, for reports built on one.

        The first run matching the aggregate verdict, so a three-of-five flake
        is represented by a run that actually showed the bug rather than by
        whichever run happened to go first.
        """
        if not self.runs:
            return None
        status = self.status
        for result in self.runs:
            if result.status is status:
                return result
        return self.runs[0]

    @property
    def message(self) -> str:
        """The representative run's message, with the trial tally when there are several."""
        representative = self.representative
        base = representative.message if representative else "reproduction was not run"
        if self.n <= 1:
            return base
        return f"{base} ({self.describe()})"

    @property
    def power_note(self) -> Optional[str]:
        """What this many clean runs still could not rule out.

        Reported only when every run came back clean, which is exactly when the
        number is easiest to over-read. The bound is the interval's own lower
        edge rather than a threshold picked to look reassuring.
        """
        if not self.measured_runs or not self.fixed:
            return None
        return (
            f"{self.fixes}/{self.measured_runs} clean run(s) is consistent with a "
            f"fix rate as low as {self.interval().low:.1%} - raise the run count "
            "to narrow that"
        )

    def describe(self) -> str:
        """Fix count, rate and interval across the trials."""
        if not self.measured_runs:
            return f"no verdict from {self.n} run(s)"
        text = (
            f"fixed {self.fixes}/{self.measured_runs} run(s), "
            f"{self.interval().describe()}"
        )
        # Runs that errored or timed out measured nothing, so they are not in
        # the denominator. Saying so keeps "fixed 1/1" from reading like a clean
        # single-trial pass when in fact two runs happened and one blew up.
        unmeasured = self.n - self.measured_runs
        if unmeasured:
            text += f" ({unmeasured} of {self.n} run(s) returned no verdict)"
        if self.flaky:
            text += " - flaky, not a fix"
        return text

    def to_dict(self) -> dict:
        """Serialise the trial set for the run artifacts."""
        return {
            "runs": self.n,
            "measured_runs": self.measured_runs,
            "fixes": self.fixes,
            "fix_rate": round(self.fix_rate, 6),
            "fix_rate_ci": self.interval().to_dict(),
            "status": self.status.value,
            "fixed": self.fixed,
            "flaky": self.flaky,
            "reproduced": self.reproduced,
            "results": [r.to_dict() for r in self.runs],
        }


@dataclass
class BugReproduction:
    """A script that demonstrates one specific bug.

    Contract: the script exits ``0`` when the bug is **fixed** and non-zero
    when it still reproduces, which is what a pytest file expressing the
    desired behaviour does for free. A script can override that by printing
    ``BUG_FIXED`` or ``BUG_PRESENT``.

    ``test_*.py`` files run under pytest; any other ``.py`` file runs as a
    plain script; anything else must be executable and is run directly.

    ``exec_fn`` is how the run is actually executed, defaulting to
    ``subprocess.run``. The code phase passes the sandbox's runner here,
    because a reproduction executes the candidate's code and belongs behind
    the same boundary as the evolver that produced it.
    """

    script: Path
    issue: Optional[str] = None
    timeout: int = 300
    python: Optional[str] = None
    exec_fn: Callable[..., subprocess.CompletedProcess] = subprocess.run

    def __post_init__(self) -> None:
        self.script = Path(self.script).expanduser()

    @property
    def name(self) -> str:
        """File name of the reproduction script."""
        return self.script.name

    def available(self) -> bool:
        """True when the reproduction script exists and can be run."""
        return self.script.is_file()

    def command(self, python: Optional[str] = None) -> list[str]:
        """Build the command that runs this reproduction."""
        interpreter = self.python or python or "python"
        if self.script.suffix == ".py":
            stem = self.script.stem
            if stem.startswith("test_") or stem.endswith("_test"):
                return [interpreter, "-m", "pytest", str(self.script), "-q", "--tb=short"]
            return [interpreter, str(self.script)]
        if os.access(self.script, os.X_OK):
            return [str(self.script)]
        raise FitnessError(
            f"do not know how to run reproduction script {self.script} "
            "(use a .py file, or mark the script executable)"
        )

    def run(self, repo: Path, python: Optional[str] = None) -> ReproResult:
        """Run the reproduction against the working tree at *repo*."""
        if not self.available():
            return ReproResult(
                ReproStatus.UNAVAILABLE,
                f"reproduction script not found: {self.script}",
            )

        try:
            cmd = self.command(python)
        except FitnessError as exc:
            return ReproResult(ReproStatus.UNAVAILABLE, str(exc))

        started = time.time()
        try:
            proc = self.exec_fn(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(repo),
            )
        except subprocess.TimeoutExpired:
            return ReproResult(
                ReproStatus.ERROR,
                f"reproduction timed out after {self.timeout}s",
                duration_s=time.time() - started,
            )
        except OSError as exc:
            return ReproResult(
                ReproStatus.UNAVAILABLE, f"could not run reproduction: {exc}"
            )

        elapsed = time.time() - started
        output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        tail = "\n".join(output.strip().splitlines()[-20:])

        if any(marker in output for marker in FIXED_MARKERS):
            return ReproResult(
                ReproStatus.FIXED,
                "reproduction reports the bug is fixed",
                proc.returncode,
                tail,
                elapsed,
            )
        if any(marker in output for marker in PRESENT_MARKERS):
            return ReproResult(
                ReproStatus.PRESENT,
                "reproduction reports the bug is still present",
                proc.returncode,
                tail,
                elapsed,
            )

        if proc.returncode == 0:
            return ReproResult(
                ReproStatus.FIXED,
                "reproduction exited 0 - bug appears fixed",
                proc.returncode,
                tail,
                elapsed,
            )
        return ReproResult(
            ReproStatus.PRESENT,
            f"reproduction exited {proc.returncode} - bug still reproduces",
            proc.returncode,
            tail,
            elapsed,
        )

    def run_many(
        self,
        repo: Path,
        python: Optional[str] = None,
        runs: int = 1,
        confidence: float = 0.95,
    ) -> ReproTrials:
        """Run the reproduction *runs* times and report the measured fix rate.

        Built on :meth:`run`, so a subclass that overrides a single run gets
        the aggregate for free. The loop stops early on ``UNAVAILABLE``: a
        script that is not there will not appear on the next attempt, and
        repeating it would pad n with runs that measured nothing.
        """
        if runs < 1:
            raise ValueError(f"reproduction run count must be at least 1, got {runs}")
        results: list[ReproResult] = []
        for _ in range(runs):
            result = self.run(repo, python)
            results.append(result)
            if result.status is ReproStatus.UNAVAILABLE:
                break
        return ReproTrials(runs=results, confidence=confidence)


# ──────────────────────────────────────────────────────────────────────────
# Per-test outcomes
# ──────────────────────────────────────────────────────────────────────────

# XFAIL counts as a pass: the suite declared that failure expected, so the run
# is still green. XPASS counts as a pass for the same reason - it did pass.
# SKIPPED appears in neither set on purpose. A skipped test asserted nothing,
# so it carries no outcome to pair, and inventing one either way would put a
# fabricated observation into a real test.
PASSING_OUTCOMES = ("PASSED", "XPASS", "XFAIL")
FAILING_OUTCOMES = ("FAILED", "ERROR")

_OUTCOME_WORDS = "|".join((*PASSING_OUTCOMES, *FAILING_OUTCOMES))

# "PASSED tests/test_x.py::test_y" - the -rA short summary, which is the only
# form that lists tests that passed.
_SUMMARY_LINE = re.compile(rf"^({_OUTCOME_WORDS})\s+(\S.*?)(?:\s+-\s.*)?$")

# "tests/test_x.py::test_y PASSED [ 50%]" - the -v progress form.
_VERBOSE_LINE = re.compile(rf"^(\S.*?::.+?)\s+({_OUTCOME_WORDS})\b")

# Node ids of failing tests kept in a serialised gate result. A full suite has
# thousands of tests and the run record does not need every passing name; the
# ones that failed are what a reviewer opens the file for.
_MAX_REPORTED_TESTS = 25


def parse_pytest_outcomes(output: str) -> dict[str, bool]:
    """Map pytest node ids to pass/fail from a run's captured output.

    Reads both shapes pytest emits: the ``-rA`` short summary
    (``PASSED path::test``) and the ``-v`` progress lines
    (``path::test PASSED``). Only lines carrying a node id are read, so
    collection errors and skip lines - which name a file, not a test - are
    left out rather than guessed at.

    A node id seen twice with conflicting verdicts resolves to the failure.
    Two verdicts for one test means something reran it, and the run that
    failed is the one worth keeping.
    """
    outcomes: dict[str, bool] = {}
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _SUMMARY_LINE.match(line)
        if match:
            outcome, nodeid = match.group(1), match.group(2).strip()
        else:
            match = _VERBOSE_LINE.match(line)
            if not match:
                continue
            nodeid, outcome = match.group(1).strip(), match.group(2)
        if "::" not in nodeid:
            continue
        passed = outcome in PASSING_OUTCOMES
        outcomes[nodeid] = outcomes.get(nodeid, True) and passed
    return outcomes


@dataclass
class PerTestGateResult(GateResult):
    """A pytest gate result that also carries the outcome of every test.

    A subclass rather than a wider :class:`GateResult`, because gates.py is
    shared by every phase and per-test capture is a Phase 4 need. Everything
    that accepts a GateResult keeps accepting this one; the extra field is
    what makes a paired comparison of two runs possible at all.
    """

    outcomes: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise the gate result, adding the per-test outcome tally."""
        blob = super().to_dict()
        failing = sorted(name for name, ok in self.outcomes.items() if not ok)
        blob["tests_measured"] = len(self.outcomes)
        blob["failing_tests"] = failing[:_MAX_REPORTED_TESTS]
        return blob


def pytest_outcomes_from_result(result: GateResult) -> dict[str, bool]:
    """Best-effort per-test outcomes for a pytest gate result.

    Prefers the structured field on a :class:`PerTestGateResult`, and falls
    back to parsing whatever output the gate kept. The fallback matters because
    an operator can widen the gate through ``--pytest-subset -v`` without
    swapping the runner. An empty mapping means "not measured", which callers
    must report as absent evidence rather than as no change.
    """
    outcomes = getattr(result, "outcomes", None)
    if outcomes:
        return dict(outcomes)
    return parse_pytest_outcomes(result.details or "")


def _last_summary_line(stdout: str) -> str:
    for line in reversed(stdout.strip().splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            return line.strip()
    return ""


@dataclass
class PerTestPytestRunner:
    """A pytest gate runner that records how every individual test ended.

    :func:`evolution.core.gates.run_pytest_gate` answers the only question the
    hard gate asks - did anything fail - and keeps a tail of the output when
    something did. That is enough to gate on and not enough to compare two runs
    test by test: a green run carries no per-test detail at all, so a baseline
    and a candidate cannot be lined up as paired outcomes.

    This runner asks pytest for the full short summary and keeps the parsed
    result on the gate result whatever the verdict. The verdict itself is
    unchanged: the exit status decides, exactly as before. An extra reporting
    flag cannot turn a red run green, and nothing here can accept a candidate
    the plain gate would have rejected.

    ``exec_fn`` defaults to ``subprocess.run``; the code phase passes the
    sandbox's runner, since a suite run with a candidate applied executes
    that candidate's code.
    """

    extra_args: tuple[str, ...] = ("-rA",)
    exec_fn: Callable[..., subprocess.CompletedProcess] = subprocess.run

    def __call__(
        self,
        repo: Path,
        subset: Optional[Sequence[str]] = None,
        timeout: int = 900,
        python: Optional[str] = None,
    ) -> GateResult:
        repo = Path(repo)
        if not repo.is_dir():
            return PerTestGateResult(
                "pytest", GateStatus.UNAVAILABLE, f"repo not found: {repo}"
            )
        if not (repo / "tests").is_dir():
            return PerTestGateResult(
                "pytest", GateStatus.UNAVAILABLE, f"no tests/ directory under {repo}"
            )

        cmd = [
            python or "python",
            "-m",
            "pytest",
            *(subset or ["tests/"]),
            "-q",
            "--tb=short",
            *self.extra_args,
        ]
        started = time.time()
        try:
            proc = self.exec_fn(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=str(repo)
            )
        except subprocess.TimeoutExpired:
            return PerTestGateResult(
                "pytest",
                GateStatus.FAILED,
                f"test suite timed out after {timeout}s",
                duration_s=time.time() - started,
            )
        except OSError as exc:
            return PerTestGateResult(
                "pytest", GateStatus.UNAVAILABLE, f"could not run pytest: {exc}"
            )

        elapsed = time.time() - started
        stdout = proc.stdout or ""
        outcomes = parse_pytest_outcomes(stdout)
        summary = _last_summary_line(stdout)

        if proc.returncode == 0:
            return PerTestGateResult(
                "pytest",
                GateStatus.PASSED,
                summary or "all tests passed",
                duration_s=elapsed,
                outcomes=outcomes,
            )
        return PerTestGateResult(
            "pytest",
            GateStatus.FAILED,
            summary or f"pytest exited {proc.returncode}",
            details="\n".join(stdout.strip().splitlines()[-25:]),
            duration_s=elapsed,
            outcomes=outcomes,
        )


@dataclass
class SuiteComparison:
    """Baseline vs candidate over the tests both runs actually ran.

    Pairing is by node id, never by position: a candidate can add, remove, skip
    or reorder tests, so index i in one run is not index i in the other. Tests
    only one run knows about are reported on their own rather than padded with
    an invented outcome, which would put a pair nobody observed into the test.

    This is information, not a gate. Any failing test already rejects the
    candidate outright, so there is no tolerance here for a significance test
    to soften - and with a zero tolerance there is no underpowered case to
    report either, because the gate is already as strict as a gate can be.
    """

    paired: PairedBinary
    shared: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    newly_failing: tuple[str, ...] = ()
    newly_passing: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        """Number of tests shared by both runs, which is what the pairing uses."""
        return self.paired.n

    @property
    def significant_regression(self) -> bool:
        """True when the paired test finds a regression at alpha."""
        return self.paired.significant_regression

    @property
    def significant_improvement(self) -> bool:
        """True when the paired test finds an improvement at alpha."""
        return self.paired.significant_improvement

    @property
    def coverage_changed(self) -> bool:
        """True when the candidate run did not cover the same set of tests.

        A test that stopped being collected is not a test that passed. Fifty of
        a hundred tests silently disappearing has to read as a coverage change,
        not as an identical run.
        """
        return bool(self.added or self.removed)

    @property
    def unchanged(self) -> bool:
        """True when the same tests ran and not one of them changed outcome."""
        return self.paired.discordant == 0 and not self.coverage_changed

    @property
    def verdict(self) -> str:
        """One phrase for the outcome: regression, improvement, or coverage change."""
        if self.significant_regression:
            return "significant regression"
        if self.significant_improvement:
            return "significant improvement"
        if self.coverage_changed:
            return "coverage changed"
        if self.unchanged:
            return "identical outcomes"
        return "no significant change"

    def describe(self) -> str:
        """The verdict, the paired statistics, and any newly failing or passing tests."""
        parts = [f"{self.verdict}: {self.paired.describe()}"]
        if self.newly_failing:
            parts.append(f"{len(self.newly_failing)} newly failing")
        if self.newly_passing:
            parts.append(f"{len(self.newly_passing)} newly passing")
        if self.added or self.removed:
            parts.append(
                f"{len(self.added)} added and {len(self.removed)} removed "
                "(unpaired, excluded)"
            )
        return "; ".join(parts)

    def to_dict(self) -> dict:
        """Serialise the comparison for the run artifacts."""
        return {
            "verdict": self.verdict,
            "paired": self.paired.to_dict(),
            "shared": len(self.shared),
            "added": list(self.added[:_MAX_REPORTED_TESTS]),
            "removed": list(self.removed[:_MAX_REPORTED_TESTS]),
            "newly_failing": list(self.newly_failing[:_MAX_REPORTED_TESTS]),
            "newly_passing": list(self.newly_passing[:_MAX_REPORTED_TESTS]),
        }


def compare_test_suites(
    baseline: Mapping[str, bool],
    candidate: Mapping[str, bool],
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> Optional[SuiteComparison]:
    """Compare two runs of a suite as paired binary outcomes.

    Returns None when the two runs share no test, which is the honest answer
    when there is nothing to pair. An unpaired comparison of two unrelated test
    sets would be worse than no comparison at all.
    """
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        return None
    base_outcomes = [bool(baseline[name]) for name in shared]
    cand_outcomes = [bool(candidate[name]) for name in shared]
    return SuiteComparison(
        paired=compare_paired_binary(
            base_outcomes, cand_outcomes, alpha=alpha, confidence=confidence
        ),
        shared=tuple(shared),
        added=tuple(sorted(set(candidate) - set(baseline))),
        removed=tuple(sorted(set(baseline) - set(candidate))),
        newly_failing=tuple(n for n in shared if baseline[n] and not candidate[n]),
        newly_passing=tuple(n for n in shared if candidate[n] and not baseline[n]),
    )


# ──────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FitnessWeights:
    """Relative weight of each *scored* component.

    pytest is absent from this list on purpose: it gates rather than scores.
    Weights are renormalized over whatever components were actually
    measurable, so a run with no benchmark installed still produces a
    meaningful number instead of silently capping at 0.7.
    """

    bug_fix: float = 0.5
    benchmark: float = 0.3
    quality: float = 0.2

    def total(self) -> float:
        """The intended weight, measured or not.

        The denominator for evidence coverage: how much of the score a
        candidate could have earned if every component had actually run.
        """
        return self.bug_fix + self.benchmark + self.quality

    def to_dict(self) -> dict:
        """Serialise the score components."""
        return {
            "bug_fix": self.bug_fix,
            "benchmark": self.benchmark,
            "quality": self.quality,
        }


@dataclass
class CodeFitness:
    """The full verdict on one candidate."""

    label: str
    accepted: bool
    total: float
    safety: SafetyReport
    quality: QualitySignals
    pytest_result: GateResult
    benchmark_results: list[GateResult] = field(default_factory=list)
    repro: Optional[ReproResult] = None
    components: dict[str, float] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    rejection_reason: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    repro_trials: Optional[ReproTrials] = None
    suite: Optional[SuiteComparison] = None
    evidence_coverage: float = 0.0
    missing_evidence: list[str] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        """True when the candidate did not clear the guardrails."""
        return not self.accepted

    def score_line(self) -> str:
        """The score and the evidence behind it, which never travel apart.

        A 0.85 measured by tests, a benchmark and a reproduction and a 0.85
        measured by one source heuristic print as the same number. This is the
        one place that difference is always visible.
        """
        line = f"{self.total:.3f} (evidence {self.evidence_coverage:.0%}"
        if self.missing_evidence:
            line += f", no {' or '.join(self.missing_evidence)}"
        return line + ")"

    def to_dict(self) -> dict:
        """Serialise the fitness verdict, its score and its evidence coverage."""
        return {
            "label": self.label,
            "accepted": self.accepted,
            "total": round(self.total, 4),
            "evidence_coverage": round(self.evidence_coverage, 4),
            "missing_evidence": list(self.missing_evidence),
            "rejection_reason": self.rejection_reason,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
            "safety": self.safety.to_dict(),
            "quality": self.quality.to_dict(),
            "pytest": self.pytest_result.to_dict(),
            "benchmarks": [b.to_dict() for b in self.benchmark_results],
            "repro": self.repro.to_dict() if self.repro else None,
            "repro_trials": self.repro_trials.to_dict() if self.repro_trials else None,
            "suite": self.suite.to_dict() if self.suite else None,
            "notes": list(self.notes),
        }


# ──────────────────────────────────────────────────────────────────────────
# Ranking several candidates
# ──────────────────────────────────────────────────────────────────────────

# How much of a gap in the composite score means anything at all. The score is
# a weighted average of coarse measurements: a binary bug fix, a benchmark pass
# rate over a handful of tasks, and a quality heuristic whose penalties come in
# steps of 0.05. Nothing in it resolves finer than a couple of points, so two
# candidates inside this distance are not ranked, they are shuffled.
DEFAULT_RANKING_RESOLUTION = 0.02


@dataclass
class CandidateRanking:
    """Which candidate won, by how much, and whether that gap means anything.

    Picking the top of a sorted list is easy. The useful part is saying when
    the sort order was arbitrary: presenting a 0.004 lead as a result implies a
    precision the measurements do not have, and a reviewer who trusts it spends
    their afternoon on the wrong patch.

    ``tied`` includes the winner. It is the group of candidates nothing
    separates, not a list of runners-up, so a length of one means the winner
    stands alone.
    """

    winner: str
    winner_score: float
    resolution: float = DEFAULT_RANKING_RESOLUTION
    runner_up: Optional[str] = None
    runner_up_score: Optional[float] = None
    tied: tuple[str, ...] = ()
    winner_coverage: float = 0.0
    runner_up_coverage: Optional[float] = None
    winner_fix_rate: Optional[Interval] = None
    runner_up_fix_rate: Optional[Interval] = None
    considered: int = 1

    @property
    def margin(self) -> Optional[float]:
        """The winner's lead over the runner-up, or None with no runner-up."""
        if self.runner_up_score is None:
            return None
        return self.winner_score - self.runner_up_score

    @property
    def separated(self) -> bool:
        """True when the winner is ahead by more than the score can resolve.

        A sole survivor is separated by default: there is no second candidate
        for it to be confused with.
        """
        margin = self.margin
        return margin is None or margin > self.resolution + _FLOAT_SLACK

    @property
    def within_noise(self) -> bool:
        """True when the top two candidates are not separated by the data."""
        return not self.separated

    @property
    def fix_rate_inconclusive(self) -> bool:
        """True when the reproduction evidence does not separate the top two.

        Only interesting when the two measured different fix rates. Identical
        rates trivially overlap, and saying so would add nothing.
        """
        first, second = self.winner_fix_rate, self.runner_up_fix_rate
        if first is None or second is None or first.point == second.point:
            return False
        return first.low <= second.high and second.low <= first.high

    @property
    def thinner_evidence(self) -> bool:
        """True when the winner rests on less measured evidence than the runner-up."""
        if self.runner_up_coverage is None:
            return False
        return self.winner_coverage < self.runner_up_coverage

    def describe(self) -> str:
        """The winner, its margin over the runner-up, and whether that margin means anything."""
        if self.margin is None:
            return (
                f"{self.winner} wins at {self.winner_score:.3f} - the only candidate "
                "that survived the guardrails, so there is nothing to rank it against"
            )
        if self.separated:
            text = (
                f"{self.winner} wins at {self.winner_score:.3f}, ahead of "
                f"{self.runner_up} by {self.margin:.3f} "
                f"(resolution {self.resolution:.3f})"
            )
        else:
            text = (
                f"{self.winner} leads at {self.winner_score:.3f} but only by "
                f"{self.margin:.3f} over {self.runner_up}, inside the "
                f"{self.resolution:.3f} this score can resolve - the ranking is "
                "within noise and the pick is arbitrary"
            )
            if len(self.tied) > 1:
                text += f" (nothing separates {', '.join(self.tied)})"
        if self.fix_rate_inconclusive:
            text += (
                f"; their measured fix rates ({self.winner_fix_rate.describe()} vs "
                f"{self.runner_up_fix_rate.describe()}) have overlapping intervals, "
                "so the reproduction does not separate them either"
            )
        if self.thinner_evidence:
            text += (
                f"; the winner's score rests on less evidence "
                f"({self.winner_coverage:.0%}) than {self.runner_up}'s "
                f"({self.runner_up_coverage:.0%})"
            )
        return text

    def to_dict(self) -> dict:
        """Serialise the ranking for the run artifacts."""
        return {
            "winner": self.winner,
            "winner_score": round(self.winner_score, 6),
            "runner_up": self.runner_up,
            "runner_up_score": (
                None if self.runner_up_score is None else round(self.runner_up_score, 6)
            ),
            "margin": None if self.margin is None else round(self.margin, 6),
            "resolution": self.resolution,
            "separated": self.separated,
            "within_noise": self.within_noise,
            "tied": list(self.tied),
            "considered": self.considered,
            "winner_coverage": round(self.winner_coverage, 4),
            "runner_up_coverage": (
                None
                if self.runner_up_coverage is None
                else round(self.runner_up_coverage, 4)
            ),
            "fix_rate_inconclusive": self.fix_rate_inconclusive,
            "thinner_evidence": self.thinner_evidence,
            "summary": self.describe(),
        }


def _fix_rate_interval(fitness: CodeFitness) -> Optional[Interval]:
    trials = fitness.repro_trials
    if trials is None or not trials.measured_runs:
        return None
    return trials.interval()


def rank_candidates(
    fitnesses: Sequence[CodeFitness],
    resolution: float = DEFAULT_RANKING_RESOLUTION,
) -> Optional[CandidateRanking]:
    """Rank scored candidates and say how much the order is worth.

    Sorting is stable and descending, so an exact tie keeps the order the
    candidates arrived in and this function picks the same winner ``max`` would.
    What it adds is the margin, the set of candidates the winner is not
    meaningfully ahead of, and the two ways the evidence can undercut the
    ordering: overlapping fix-rate intervals, and a winner measured more thinly
    than the candidate below it.
    """
    ranked = sorted(fitnesses, key=lambda f: f.total, reverse=True)
    if not ranked:
        return None

    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    return CandidateRanking(
        winner=winner.label,
        winner_score=winner.total,
        resolution=resolution,
        runner_up=runner_up.label if runner_up else None,
        runner_up_score=runner_up.total if runner_up else None,
        tied=tuple(
            f.label for f in ranked if winner.total - f.total <= resolution + _FLOAT_SLACK
        ),
        winner_coverage=winner.evidence_coverage,
        runner_up_coverage=runner_up.evidence_coverage if runner_up else None,
        winner_fix_rate=_fix_rate_interval(winner),
        runner_up_fix_rate=_fix_rate_interval(runner_up) if runner_up else None,
        considered=len(ranked),
    )


@dataclass
class BaselineSnapshot:
    """The state of the repo before any mutation, for comparison and sanity."""

    source: str
    pytest_result: GateResult
    repro: Optional[ReproResult] = None
    benchmark_results: list[GateResult] = field(default_factory=list)
    repro_trials: Optional[ReproTrials] = None
    test_outcomes: dict[str, bool] = field(default_factory=dict)

    @property
    def tests_green(self) -> bool:
        """True when the hard pytest gate passed."""
        return self.pytest_result.status is GateStatus.PASSED

    @property
    def bug_reproduces(self) -> bool:
        """True when the repro script confirms the bug exists to be fixed.

        With one run this is the old question unchanged. With several it is the
        honest version of it: a bug that shows up twice in five runs is still a
        bug to fix, and :attr:`repro_trials` carries how often it showed.
        """
        if self.repro_trials is not None:
            return self.repro_trials.reproduced
        return self.repro is not None and self.repro.status is ReproStatus.PRESENT

    def benchmark_baselines(self) -> dict[str, Optional[float]]:
        """Each benchmark's baseline score by name, None where it did not run."""
        return {r.name: r.score for r in self.benchmark_results}

    def to_dict(self) -> dict:
        """Serialise every gate result that fed this fitness verdict."""
        return {
            "pytest": self.pytest_result.to_dict(),
            "repro": self.repro.to_dict() if self.repro else None,
            "repro_trials": self.repro_trials.to_dict() if self.repro_trials else None,
            "benchmarks": [b.to_dict() for b in self.benchmark_results],
            "tests_green": self.tests_green,
            "bug_reproduces": self.bug_reproduces,
            "tests_measured": len(self.test_outcomes),
        }


PytestRunner = Callable[..., GateResult]
BenchmarkRunner = Callable[..., GateResult]
OutcomeReader = Callable[[GateResult], dict]


class CodeFitnessEvaluator:
    """Score candidates against the ladder PLAN.md specifies.

    The runners are injectable so the scoring logic can be tested without a
    hermes-agent checkout, and so an operator can substitute a cheaper test
    subset for the 2550-test suite during a long run.

    ``repro_runs`` is how many times the reproduction is run per candidate. The
    default of 1 preserves the old single-trial behaviour exactly; anything
    higher buys a measured fix rate with an interval instead of one Bernoulli
    draw. ``outcome_reader`` pulls per-test outcomes out of a pytest gate
    result, which is what makes the paired suite comparison possible; with the
    shared gate runner there is nothing to read and the comparison is reported
    as absent rather than as "no change".
    """

    def __init__(
        self,
        repo: Path,
        *,
        target: Optional[Path] = None,
        repro: Optional[BugReproduction] = None,
        benchmarks: Sequence[str] = (),
        benchmark_baselines: Optional[dict[str, Optional[float]]] = None,
        weights: Optional[FitnessWeights] = None,
        python: Optional[str] = None,
        pytest_subset: Optional[Sequence[str]] = None,
        pytest_timeout: int = 900,
        regression_threshold: float = 0.02,
        strict: bool = False,
        require_bug_fix: bool = True,
        repro_runs: int = 1,
        alpha: float = 0.05,
        confidence: float = 0.95,
        baseline_test_outcomes: Optional[Mapping[str, bool]] = None,
        safety_checks: Optional[Iterable[Callable[[str, str], object]]] = None,
        pytest_runner: PytestRunner = run_pytest_gate,
        benchmark_runner: BenchmarkRunner = run_benchmark_gate,
        outcome_reader: OutcomeReader = pytest_outcomes_from_result,
    ) -> None:
        if repro_runs < 1:
            raise ValueError(
                f"repro_runs must be at least 1, got {repro_runs}"
            )
        self.repo = Path(repo)
        self.target = Path(target) if target else None
        self.repro = repro
        self.benchmarks = tuple(benchmarks)
        self.benchmark_baselines = dict(benchmark_baselines or {})
        self.weights = weights or FitnessWeights()
        self.python = python
        self.pytest_subset = list(pytest_subset) if pytest_subset else None
        self.pytest_timeout = pytest_timeout
        self.regression_threshold = regression_threshold
        self.strict = strict
        self.require_bug_fix = require_bug_fix
        self.repro_runs = repro_runs
        self.alpha = alpha
        self.confidence = confidence
        # Filled in by snapshot_baseline, the same way benchmark_baselines is,
        # so a candidate is compared against the run that actually preceded it.
        self.baseline_test_outcomes = dict(baseline_test_outcomes or {})
        self.safety_checks = safety_checks
        self.pytest_runner = pytest_runner
        self.benchmark_runner = benchmark_runner
        self.outcome_reader = outcome_reader

    # ── gates ───────────────────────────────────────────────────────────

    def _run_pytest(self) -> GateResult:
        return self.pytest_runner(
            self.repo,
            subset=self.pytest_subset,
            timeout=self.pytest_timeout,
            python=self.python,
        )

    def _run_benchmark(self, name: str, fast: bool = True) -> GateResult:
        return self.benchmark_runner(
            self.repo,
            name,
            baseline=self.benchmark_baselines.get(name),
            regression_threshold=self.regression_threshold,
            fast=fast,
        )

    def _run_repro(self) -> Optional[ReproTrials]:
        if self.repro is None:
            return None
        return self.repro.run_many(
            self.repo,
            self.python,
            runs=self.repro_runs,
            confidence=self.confidence,
        )

    # ── baseline ────────────────────────────────────────────────────────

    def snapshot_baseline(self, source: str) -> BaselineSnapshot:
        """Measure the repo before evolution starts.

        Two things this catches early, both of which invalidate a whole run:
        a baseline test suite that is already red, and a reproduction script
        that already passes (so there is no bug to fix, or the script does not
        actually reproduce it).

        It also records the per-test outcomes of the baseline run, which is the
        left-hand side of every paired suite comparison that follows. Without a
        baseline captured on the same machine, in the same session, there is
        nothing legitimate to pair a candidate against - so this measurement
        replaces any outcomes handed to the constructor, including when it
        measured none. A comparison skipped for want of a baseline is a smaller
        loss than a comparison run against last week's.
        """
        pytest_result = self._run_pytest()
        benchmarks = [self._run_benchmark(name) for name in self.benchmarks]
        for result in benchmarks:
            if result.score is not None:
                self.benchmark_baselines.setdefault(result.name, result.score)
        trials = self._run_repro()
        outcomes = self.outcome_reader(pytest_result)
        self.baseline_test_outcomes = dict(outcomes)
        return BaselineSnapshot(
            source=source,
            pytest_result=pytest_result,
            repro=trials.representative if trials else None,
            benchmark_results=benchmarks,
            repro_trials=trials,
            test_outcomes=dict(outcomes),
        )

    # ── candidate scoring ───────────────────────────────────────────────

    def evaluate(
        self,
        before_source: str,
        after_source: str,
        label: str = "candidate",
    ) -> CodeFitness:
        """Score one candidate that is already written to the working tree."""
        if self.target is not None and self.target.is_file():
            on_disk = self.target.read_text(encoding="utf-8")
            if on_disk != after_source:
                raise FitnessError(
                    f"{self.target} does not contain the candidate being scored - "
                    "apply the mutation before evaluating it"
                )

        safety = run_safety_checks(before_source, after_source, self.safety_checks)
        quality = quality_signals(before_source, after_source)

        if after_source == before_source:
            # A candidate identical to the baseline cannot have fixed anything,
            # and running 2550 tests to confirm that would be a waste.
            return CodeFitness(
                label=label,
                accepted=False,
                total=0.0,
                safety=safety,
                quality=quality,
                pytest_result=GateResult(
                    "pytest",
                    GateStatus.SKIPPED,
                    "not run - candidate is identical to the baseline",
                ),
                rejection_reason="no change from the baseline",
            )

        if not safety.passed:
            failure = safety.first_failure()
            return CodeFitness(
                label=label,
                accepted=False,
                total=0.0,
                safety=safety,
                quality=quality,
                pytest_result=GateResult(
                    "pytest",
                    GateStatus.SKIPPED,
                    "not run - candidate failed the safety guardrails",
                ),
                rejection_reason=f"safety: {failure.message}" if failure else "safety",
                notes=["expensive gates skipped: guardrails rejected the candidate"],
            )

        chain = GateChain(strict=self.strict).run(
            self._run_pytest,
            *[self._benchmark_thunk(name) for name in self.benchmarks],
        )
        pytest_result = chain.results[0]
        benchmark_results = list(chain.results[1:])
        suite = self._compare_suite(pytest_result)

        notes: list[str] = []
        if pytest_result.status is GateStatus.UNAVAILABLE:
            notes.append(
                "pytest could not run - the hard gate did not actually verify anything"
            )
        if suite is not None:
            notes.append(f"suite vs baseline: {suite.describe()}")
        notes.extend(self._benchmark_power_notes(benchmark_results))

        if not chain.passed:
            blocker = chain.blockers[0]
            reason = (
                "pytest failed - hard gate, no partial credit"
                if blocker.name == "pytest" and blocker.status is GateStatus.FAILED
                else f"{blocker.name}: {blocker.message}"
            )
            return CodeFitness(
                label=label,
                accepted=False,
                total=0.0,
                safety=safety,
                quality=quality,
                pytest_result=pytest_result,
                benchmark_results=benchmark_results,
                rejection_reason=reason,
                notes=notes,
                suite=suite,
            )

        trials = self._run_repro()
        if (
            trials is not None
            and self.require_bug_fix
            and trials.status is not ReproStatus.UNAVAILABLE
            and not trials.fixed
        ):
            return CodeFitness(
                label=label,
                accepted=False,
                total=0.0,
                safety=safety,
                quality=quality,
                pytest_result=pytest_result,
                benchmark_results=benchmark_results,
                repro=trials.representative,
                repro_trials=trials,
                rejection_reason=f"bug not fixed: {trials.message}",
                notes=notes,
                suite=suite,
            )

        components, weights_used = self._score_components(
            trials, benchmark_results, quality
        )
        total = self._weighted_total(components, weights_used)
        if trials is not None and trials.power_note:
            notes.append(trials.power_note)

        return CodeFitness(
            label=label,
            accepted=True,
            total=total,
            safety=safety,
            quality=quality,
            pytest_result=pytest_result,
            benchmark_results=benchmark_results,
            repro=trials.representative if trials else None,
            repro_trials=trials,
            components=components,
            weights_used=weights_used,
            notes=notes,
            suite=suite,
            evidence_coverage=self._evidence_coverage(weights_used),
            missing_evidence=self._missing_evidence(weights_used),
        )

    # ── internals ───────────────────────────────────────────────────────

    def _benchmark_thunk(self, name: str) -> Callable[[], GateResult]:
        return lambda: self._run_benchmark(name)

    def _compare_suite(self, pytest_result: GateResult) -> Optional[SuiteComparison]:
        """Pair the candidate's per-test outcomes against the baseline's.

        Returns None when either side has no per-test detail, which is the
        common case with the shared pytest gate: it keeps a failure tail and
        nothing more. Absent evidence is reported as absent. "No comparison
        available" and "no change detected" are different claims and only one
        of them is true here.
        """
        if not self.baseline_test_outcomes:
            return None
        candidate = self.outcome_reader(pytest_result)
        if not candidate:
            return None
        return compare_test_suites(
            self.baseline_test_outcomes,
            candidate,
            alpha=self.alpha,
            confidence=self.confidence,
        )

    def _benchmark_power_notes(
        self, benchmark_results: Sequence[GateResult]
    ) -> list[str]:
        """Say out loud what the benchmark tolerance is and is not doing.

        A benchmark gate compares two aggregate pass rates against a fixed
        tolerance. That is a point-estimate comparison with no sample size
        attached, so nothing here can say whether a drop inside the tolerance
        is real or whether one past it is noise, and a benchmark reporting only
        a rate gives no per-task outcomes to pair. The note records that the
        tolerance is being enforced on a point estimate rather than on
        evidence, which is a smaller problem when it is written down.
        """
        notes: list[str] = []
        for result in benchmark_results:
            if result.score is None or result.baseline is None:
                continue
            notes.append(
                f"{result.name}: {result.baseline:.1%} -> {result.score:.1%} judged "
                f"against a fixed {abs(self.regression_threshold):.1%} tolerance on "
                "the point estimate - the benchmark reports no per-task outcomes, "
                "so this comparison carries no sample size and no significance"
            )
        return notes

    def _evidence_coverage(self, weights_used: Mapping[str, float]) -> float:
        """How much of the intended weight was actually measured.

        The weighted total renormalizes over whatever ran, which is the right
        way to keep the number comparable and the wrong way to represent how
        much is known. This is the missing half of that story.
        """
        intended = self.weights.total()
        if intended <= 0:
            return 0.0
        return min(1.0, sum(weights_used.values()) / intended)

    def _missing_evidence(self, weights_used: Mapping[str, float]) -> list[str]:
        """Named components that carried weight but were never measured."""
        return [
            name
            for name in ("bug_fix", "benchmark", "quality")
            if name not in weights_used and getattr(self.weights, name) > 0
        ]

    def _score_components(
        self,
        repro: Optional[ReproTrials],
        benchmark_results: Sequence[GateResult],
        quality: QualitySignals,
    ) -> tuple[dict[str, float], dict[str, float]]:
        components: dict[str, float] = {}
        weights: dict[str, float] = {}

        if repro is not None and repro.measured:
            # The measured fix rate, not a verdict on one run. With the default
            # single run this is still exactly 1.0 or 0.0; with more runs a
            # flaky fix scores like the partial thing it is.
            components["bug_fix"] = repro.fix_rate
            weights["bug_fix"] = self.weights.bug_fix

        scored = [r.score for r in benchmark_results if r.score is not None]
        if scored:
            components["benchmark"] = sum(scored) / len(scored)
            weights["benchmark"] = self.weights.benchmark

        components["quality"] = quality.score
        weights["quality"] = self.weights.quality
        return components, weights

    @staticmethod
    def _weighted_total(
        components: dict[str, float], weights: dict[str, float]
    ) -> float:
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return 0.0
        weighted = sum(components[k] * weights[k] for k in components)
        return max(0.0, min(1.0, weighted / total_weight))

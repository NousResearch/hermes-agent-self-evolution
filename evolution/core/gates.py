"""Validation gates shared by Phases 2-4.

PLAN.md draws a hard line between *fitness* and *gates*: fitness asks whether
the evolved artifact does its job better, gates ask whether it broke anything
else. A variant that improves fitness but trips a gate is rejected outright.

The gate ladder, in the order PLAN.md specifies:

    pytest              functional correctness - hard floor, must be 100%
    benchmark (fast)    quick capability check on a task subset
    benchmark (full)    thorough regression check on top candidates
    coherence           long-horizon behaviour check

Availability is checked, never assumed. PLAN.md was written expecting
``environments/benchmarks/tblite`` and ``.../yc_bench`` inside hermes-agent;
neither path exists in the repo today. Rather than silently scoring an absent
benchmark as a pass - which would let an unvalidated variant ship - a missing
benchmark yields ``status="unavailable"``. :class:`GateChain` treats that as a
pass only in permissive mode, and as a failure under ``strict=True``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

__all__ = [
    "GateStatus",
    "GateResult",
    "GateChain",
    "run_pytest_gate",
    "find_benchmark",
    "run_benchmark_gate",
    "BenchmarkSpec",
    "KNOWN_BENCHMARKS",
]


class GateStatus(str, Enum):
    """The four outcomes a gate can report.

    UNAVAILABLE is deliberately distinct from PASSED and SKIPPED. A benchmark
    that is not installed has proved nothing, and must never be counted as
    though it had passed.
    """
    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


@dataclass
class GateResult:
    """Outcome of one gate."""

    name: str
    status: GateStatus
    message: str
    score: Optional[float] = None
    baseline: Optional[float] = None
    details: str = ""
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        """True only for PASSED, so UNAVAILABLE never reads as success."""
        return self.status is GateStatus.PASSED

    @property
    def blocking(self) -> bool:
        """True when this result should stop a candidate from shipping."""
        return self.status is GateStatus.FAILED

    def to_dict(self) -> dict:
        """Serialise the gate result for the run artifacts."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "score": self.score,
            "baseline": self.baseline,
            "details": self.details,
            "duration_s": round(self.duration_s, 2),
        }


@dataclass
class BenchmarkSpec:
    """Where a benchmark lives and how to invoke it."""

    name: str
    # Candidate paths relative to the hermes-agent repo root, first hit wins.
    candidate_paths: tuple[str, ...]
    runner: tuple[str, ...] = ()
    fast_task_count: int = 20
    full_task_count: int = 100


# PLAN.md names these three. The paths below are every location they have
# plausibly lived; find_benchmark reports honestly when none resolve.
KNOWN_BENCHMARKS: dict[str, BenchmarkSpec] = {
    "tblite": BenchmarkSpec(
        name="tblite",
        candidate_paths=(
            "environments/benchmarks/tblite",
            "benchmarks/tblite",
            "environments/tblite",
        ),
        fast_task_count=20,
        full_task_count=100,
    ),
    "terminalbench2": BenchmarkSpec(
        name="terminalbench2",
        candidate_paths=(
            "environments/benchmarks/terminalbench2",
            "benchmarks/terminalbench2",
        ),
        fast_task_count=20,
        full_task_count=89,
    ),
    "yc_bench": BenchmarkSpec(
        name="yc_bench",
        candidate_paths=(
            "environments/benchmarks/yc_bench",
            "benchmarks/yc_bench",
        ),
        fast_task_count=50,
        full_task_count=100,
    ),
}


# ──────────────────────────────────────────────────────────────────────────
# pytest gate
# ──────────────────────────────────────────────────────────────────────────

_PYTEST_SUMMARY = re.compile(
    r"(?:(\d+) failed)?[,\s]*(?:(\d+) passed)?[,\s]*(?:(\d+) error)?",
)


def _summarise_pytest(stdout: str) -> str:
    for line in reversed(stdout.strip().splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            return line.strip()
    return ""


def run_pytest_gate(
    repo: Path,
    subset: Optional[Sequence[str]] = None,
    timeout: int = 900,
    python: Optional[str] = None,
) -> GateResult:
    """Run a repo's pytest suite. Anything but a clean exit is a failure.

    *subset* narrows the run to specific paths or ``-k`` expressions, which is
    what makes the gate affordable to run on every candidate rather than only
    on finalists.
    """
    import time

    repo = Path(repo)
    if not repo.is_dir():
        return GateResult(
            "pytest", GateStatus.UNAVAILABLE, f"repo not found: {repo}"
        )
    if not (repo / "tests").is_dir():
        return GateResult(
            "pytest", GateStatus.UNAVAILABLE, f"no tests/ directory under {repo}"
        )

    cmd = [python or "python", "-m", "pytest", *(subset or ["tests/"]), "-q", "--tb=short"]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(repo),
        )
    except subprocess.TimeoutExpired:
        return GateResult(
            "pytest",
            GateStatus.FAILED,
            f"test suite timed out after {timeout}s",
            duration_s=time.time() - started,
        )
    except OSError as exc:
        return GateResult(
            "pytest", GateStatus.UNAVAILABLE, f"could not run pytest: {exc}"
        )

    elapsed = time.time() - started
    summary = _summarise_pytest(proc.stdout or "")

    if proc.returncode == 0:
        return GateResult(
            "pytest",
            GateStatus.PASSED,
            summary or "all tests passed",
            duration_s=elapsed,
        )

    # Exit 5 is pytest's "no tests were collected", not "tests failed". Treating
    # it as a failure turns a too-narrow -k expression into a red suite that
    # blocks every candidate, and tells the operator their tests are broken when
    # the truth is that the filter matched nothing. An empty selection verifies
    # nothing, so it is UNAVAILABLE, and strict mode still refuses to ship on it.
    if proc.returncode == 5:
        selection = " ".join(subset) if subset else "tests/"
        return GateResult(
            "pytest",
            GateStatus.UNAVAILABLE,
            f"no tests matched the selection ({selection}), so nothing was verified",
            duration_s=elapsed,
        )

    tail = "\n".join((proc.stdout or "").strip().splitlines()[-25:])
    return GateResult(
        "pytest",
        GateStatus.FAILED,
        summary or f"pytest exited {proc.returncode}",
        details=tail,
        duration_s=elapsed,
    )


# ──────────────────────────────────────────────────────────────────────────
# Benchmark gates
# ──────────────────────────────────────────────────────────────────────────


def find_benchmark(repo: Path, name: str) -> Optional[Path]:
    """Resolve a benchmark directory inside *repo*, or None if absent.

    An explicit ``HERMES_BENCH_<NAME>`` env var overrides discovery, so an
    operator running a benchmark kept outside the repo can still gate on it.
    """
    spec = KNOWN_BENCHMARKS.get(name)
    if spec is None:
        return None

    override = os.getenv(f"HERMES_BENCH_{name.upper()}")
    if override:
        p = Path(override).expanduser()
        return p if p.exists() else None

    repo = Path(repo)
    for rel in spec.candidate_paths:
        candidate = repo / rel
        if candidate.exists():
            return candidate
    return None


# A textual score has to name itself as one. Scanning backwards for any
# percentage or any N/M pair makes the last progress line a runner printed into
# the result: a trailing "[100%]" reports a perfect benchmark and a "12/20"
# counter reports 0.6, neither of which the benchmark ever claimed, and
# run_benchmark_gate then reports PASSED on the strength of it. An unlabelled
# number now parses as None, which this module already treats as a gate that
# could not be measured rather than one that passed.
_RESULT_LABEL = r"(?:score|pass[\s_-]?rate|accuracy|passed|correct|result|final|total)"
_LABELLED_FRACTION = re.compile(
    _RESULT_LABEL + r"[^\d\n]{0,24}(\d+)\s*/\s*(\d+)", re.IGNORECASE
)
_LABELLED_PERCENT = re.compile(
    _RESULT_LABEL + r"[^\d\n]{0,24}(\d+(?:\.\d+)?)\s*%", re.IGNORECASE
)


def _parse_benchmark_score(stdout: str) -> Optional[float]:
    """Pull a pass rate out of a benchmark's stdout.

    Accepts a JSON object carrying ``score``/``pass_rate``/``accuracy``, or a
    line that labels its ``N/M`` or percentage as a result. Returns None when
    nothing parses, which the caller treats as a failed gate rather than a zero
    score.
    """
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                blob = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("score", "pass_rate", "accuracy"):
                if isinstance(blob.get(key), (int, float)):
                    return float(blob[key])
        m = _LABELLED_FRACTION.search(line)
        if m and int(m.group(2)) > 0:
            return int(m.group(1)) / int(m.group(2))
        m = _LABELLED_PERCENT.search(line)
        if m:
            return float(m.group(1)) / 100.0
    return None


def run_benchmark_gate(
    repo: Path,
    name: str,
    baseline: Optional[float] = None,
    regression_threshold: float = 0.02,
    fast: bool = True,
    timeout: int = 7200,
    runner: Optional[Sequence[str]] = None,
    exec_fn: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> GateResult:
    """Run a benchmark and compare it against *baseline*.

    Returns ``UNAVAILABLE`` when the benchmark is not present in the repo -
    the current state of every benchmark PLAN.md names. A drop of more than
    *regression_threshold* below the baseline fails the gate; PLAN.md sets
    that tolerance at 2% for TBLite.

    *exec_fn* is how the benchmark process is executed, defaulting to
    ``subprocess.run``. The code phase passes its sandbox runner here,
    because a benchmark run with a candidate applied executes that
    candidate's code.
    """
    import time

    location = find_benchmark(repo, name)
    if location is None:
        return GateResult(
            name,
            GateStatus.UNAVAILABLE,
            f"benchmark '{name}' not found in {repo} "
            f"(set HERMES_BENCH_{name.upper()} to point at it)",
        )

    spec = KNOWN_BENCHMARKS[name]
    count = spec.fast_task_count if fast else spec.full_task_count
    cmd = list(runner or spec.runner) or [
        "python", "-m", f"environments.benchmarks.{name}", "--limit", str(count)
    ]

    if shutil.which(cmd[0]) is None and not Path(cmd[0]).exists():
        return GateResult(
            name, GateStatus.UNAVAILABLE, f"runner not executable: {cmd[0]}"
        )

    started = time.time()
    try:
        proc = exec_fn(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=str(repo)
        )
    except subprocess.TimeoutExpired:
        return GateResult(
            name,
            GateStatus.FAILED,
            f"benchmark timed out after {timeout}s",
            duration_s=time.time() - started,
        )
    except OSError as exc:
        return GateResult(name, GateStatus.UNAVAILABLE, f"could not run: {exc}")

    elapsed = time.time() - started
    score = _parse_benchmark_score(proc.stdout or "")

    if score is None:
        return GateResult(
            name,
            GateStatus.FAILED,
            "could not parse a score from benchmark output",
            details="\n".join((proc.stdout or "").splitlines()[-20:]),
            duration_s=elapsed,
        )

    if baseline is None:
        return GateResult(
            name,
            GateStatus.PASSED,
            f"scored {score:.1%} (no baseline to compare)",
            score=score,
            duration_s=elapsed,
        )

    delta = score - baseline
    if delta < -abs(regression_threshold):
        return GateResult(
            name,
            GateStatus.FAILED,
            f"regressed {delta:+.1%} ({baseline:.1%} -> {score:.1%}), "
            f"tolerance {regression_threshold:.1%}",
            score=score,
            baseline=baseline,
            duration_s=elapsed,
        )

    return GateResult(
        name,
        GateStatus.PASSED,
        f"held at {score:.1%} ({delta:+.1%} vs baseline)",
        score=score,
        baseline=baseline,
        duration_s=elapsed,
    )


# ──────────────────────────────────────────────────────────────────────────
# Chaining
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class GateChain:
    """Run gates in order, stopping at the first blocking failure.

    ``strict`` decides what an unavailable gate means. Permissive (the default)
    records it and continues, which is what lets the pipeline run today against
    a hermes-agent that ships no benchmarks. Strict turns it into a failure, so
    a release process can demand that every requested gate actually ran.
    """

    strict: bool = False
    results: list[GateResult] = field(default_factory=list)

    def run(self, *gates) -> "GateChain":
        """Run each gate in order, stopping at the first blocking failure."""
        for gate in gates:
            result = gate() if callable(gate) else gate
            self.results.append(result)
            if self._is_blocking(result):
                break
        return self

    def _is_blocking(self, result: GateResult) -> bool:
        if result.status is GateStatus.FAILED:
            return True
        return self.strict and result.status is GateStatus.UNAVAILABLE

    @property
    def passed(self) -> bool:
        """True when no gate in the chain is blocking."""
        return not any(self._is_blocking(r) for r in self.results)

    @property
    def blockers(self) -> list[GateResult]:
        """The gate results that block, which depends on strict mode."""
        return [r for r in self.results if self._is_blocking(r)]

    def summary(self) -> str:
        """One line per gate, iconised by status."""
        icon = {
            GateStatus.PASSED: "✓",
            GateStatus.FAILED: "✗",
            GateStatus.UNAVAILABLE: "○",
            GateStatus.SKIPPED: "-",
        }
        return "\n".join(
            f"{icon[r.status]} {r.name}: {r.message}" for r in self.results
        )

    def to_dict(self) -> dict:
        """Serialise the whole chain, strict flag included."""
        return {
            "strict": self.strict,
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
        }

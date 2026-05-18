"""Composite fitness scoring for Phase 4 candidates.

The scoring policy is intentionally conservative:

- pytest must pass 100% on the candidate (hard gate)
- ruff must be clean on the candidate (soft penalty if not)
- signature/registry/error-handling freeze must hold (hard gate)
- if a bug-reproduction test is supplied, it must transition
  ``FAIL on baseline -> PASS on candidate``; otherwise the candidate is
  treated as "no demonstrated improvement"

Any hard-gate failure yields ``score == 0.0``.
"""

from __future__ import annotations

from dataclasses import dataclass

from evolution.code.sandbox import SandboxResult
from evolution.code.signature_freeze import FreezeReport


@dataclass(frozen=True)
class CompositeFitness:
    """Bundle of per-signal results and the aggregated score."""

    score: float
    pytest_passed: bool
    pytest_duration_seconds: float
    ruff_clean: bool
    freeze_passed: bool
    bug_repro_transitioned: bool
    bug_repro_evaluated: bool
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "pytest_passed": self.pytest_passed,
            "pytest_duration_seconds": self.pytest_duration_seconds,
            "ruff_clean": self.ruff_clean,
            "freeze_passed": self.freeze_passed,
            "bug_repro_transitioned": self.bug_repro_transitioned,
            "bug_repro_evaluated": self.bug_repro_evaluated,
            "notes": list(self.notes),
        }


def compute_fitness(
    *,
    pytest_candidate: SandboxResult,
    ruff_candidate: SandboxResult | None,
    freeze: FreezeReport,
    bug_repro_baseline: SandboxResult | None = None,
    bug_repro_candidate: SandboxResult | None = None,
) -> CompositeFitness:
    """Aggregate signals into a single 0..1 fitness score."""
    notes: list[str] = []
    pytest_passed = pytest_candidate.passed
    if not pytest_passed:
        notes.append(
            f"pytest failed (rc={pytest_candidate.returncode}, "
            f"timed_out={pytest_candidate.timed_out})",
        )

    ruff_clean = True if ruff_candidate is None else ruff_candidate.passed
    if ruff_candidate is not None and not ruff_clean:
        notes.append(f"ruff issues (rc={ruff_candidate.returncode})")

    freeze_passed = freeze.passed
    if not freeze_passed:
        if freeze.signature_violations:
            notes.append("signature freeze violated")
        if freeze.registry_violations:
            notes.append("registry freeze violated")
        if freeze.error_handling_decreased:
            notes.append(
                f"error-handling count decreased "
                f"({freeze.error_handling_baseline} -> {freeze.error_handling_candidate})",
            )

    bug_repro_evaluated = bug_repro_baseline is not None and bug_repro_candidate is not None
    bug_repro_transitioned = False
    if bug_repro_evaluated:
        baseline_failed = not bug_repro_baseline.passed
        candidate_passed = bug_repro_candidate.passed
        if baseline_failed and candidate_passed:
            bug_repro_transitioned = True
            notes.append("bug repro: baseline FAIL -> candidate PASS")
        elif not baseline_failed:
            notes.append("bug repro: baseline already passed; nothing to fix")
        elif baseline_failed and not candidate_passed:
            notes.append("bug repro: candidate still failing")

    if not pytest_passed or not freeze_passed:
        score = 0.0
    elif bug_repro_evaluated:
        score = 1.0 if bug_repro_transitioned else 0.25
    else:
        score = 0.5
    if ruff_candidate is not None and not ruff_clean and score > 0.0:
        score = max(0.0, score - 0.05)

    return CompositeFitness(
        score=score,
        pytest_passed=pytest_passed,
        pytest_duration_seconds=pytest_candidate.duration_seconds,
        ruff_clean=ruff_clean,
        freeze_passed=freeze_passed,
        bug_repro_transitioned=bug_repro_transitioned,
        bug_repro_evaluated=bug_repro_evaluated,
        notes=notes,
    )

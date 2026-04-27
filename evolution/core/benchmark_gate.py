"""Benchmark regression gates for evolved artifacts.

An evolved skill must not regress on external benchmarks beyond a configured
threshold. This module defines the gate interface and a stub TBLite runner.

The TBLite integration itself is intentionally a stub: running TBLite (or any
substantive eval harness) requires the hermes-agent batch_runner and a real
benchmark set, which are out of scope for this package. The stub keeps the
config surface real (so `run_tblite=True` produces a deterministic result
instead of being silently ignored) and gives a single seam for a real runner
to be wired in.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evolution.core.config import EvolutionConfig


@dataclass
class BenchmarkResult:
    """Outcome of a single benchmark gate check."""

    gate_name: str
    skipped: bool
    passed: bool
    baseline_score: Optional[float] = None
    evolved_score: Optional[float] = None
    regression: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""

    @property
    def display_message(self) -> str:
        if self.skipped:
            return f"{self.gate_name}: skipped — {self.message}"
        if self.passed:
            return f"{self.gate_name}: passed (regression {self.regression:+.3%} ≤ threshold {self.threshold:+.3%})"
        return f"{self.gate_name}: FAILED — {self.message}"


class BenchmarkGate:
    """Run benchmark regression checks as part of the evolution pipeline.

    Each gate compares the evolved skill against the baseline on an external
    benchmark and rejects the evolved version if regression exceeds a threshold.
    """

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def run_all(
        self,
        baseline_skill_path: Path,
        evolved_skill_text: str,
    ) -> list[BenchmarkResult]:
        """Run every benchmark gate enabled by config. Returns one result per gate."""
        results: list[BenchmarkResult] = []
        if self.config.run_tblite:
            results.append(self.run_tblite(baseline_skill_path, evolved_skill_text))
        return results

    def run_tblite(
        self,
        baseline_skill_path: Path,
        evolved_skill_text: str,
    ) -> BenchmarkResult:
        """Run the TBLite regression check.

        Stub implementation — returns a `skipped=True` result with an explanatory
        message. A real implementation would:
          1. Materialize `evolved_skill_text` to a temp skill location.
          2. Invoke hermes-agent's batch_runner against the TBLite benchmark.
          3. Compare evolved score to baseline and compute regression.
          4. Pass when regression <= self.config.tblite_regression_threshold.
        """
        return BenchmarkResult(
            gate_name="tblite_regression",
            skipped=True,
            passed=True,
            threshold=self.config.tblite_regression_threshold,
            message=(
                "TBLite runner not implemented in this package. "
                "Wire hermes-agent batch_runner here to enforce the gate."
            ),
        )

"""Tests for the BenchmarkGate stub.

The TBLite runner is intentionally a stub — these tests verify the gate
honours config flags, returns properly typed results, and does not silently
succeed when the user explicitly opted in.
"""

from pathlib import Path

from evolution.core.benchmark_gate import BenchmarkGate, BenchmarkResult
from evolution.core.config import EvolutionConfig


def test_disabled_runs_zero_gates():
    config = EvolutionConfig()
    config.run_tblite = False
    results = BenchmarkGate(config).run_all(
        baseline_skill_path=Path("/tmp/anything.md"),
        evolved_skill_text="evolved body",
    )
    assert results == []


def test_enabled_runs_tblite_stub():
    config = EvolutionConfig()
    config.run_tblite = True
    results = BenchmarkGate(config).run_all(
        baseline_skill_path=Path("/tmp/anything.md"),
        evolved_skill_text="evolved body",
    )
    assert len(results) == 1
    [r] = results
    assert isinstance(r, BenchmarkResult)
    assert r.gate_name == "tblite_regression"
    # Stub: skipped=True so the gate does not block the pipeline, but it is
    # passed=True so the run is not falsely failed either. The message must
    # explain that the runner is unimplemented.
    assert r.skipped is True
    assert r.passed is True
    assert "not implemented" in r.message.lower()
    assert r.threshold == config.tblite_regression_threshold


def test_display_message_for_skipped():
    r = BenchmarkResult(
        gate_name="tblite_regression",
        skipped=True,
        passed=True,
        threshold=0.02,
        message="not implemented",
    )
    assert "skipped" in r.display_message.lower()


def test_display_message_for_passed():
    r = BenchmarkResult(
        gate_name="tblite_regression",
        skipped=False,
        passed=True,
        regression=-0.005,
        threshold=0.02,
    )
    assert "passed" in r.display_message.lower()


def test_display_message_for_failed():
    r = BenchmarkResult(
        gate_name="tblite_regression",
        skipped=False,
        passed=False,
        message="regression of 5% exceeds 2% threshold",
    )
    assert "FAILED" in r.display_message

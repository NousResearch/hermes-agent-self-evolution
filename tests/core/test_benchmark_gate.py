"""Tests for benchmark gate (no actual TBLite runs)."""

from pathlib import Path
from evolution.core.benchmark_gate import (
    GateResult,
    compare,
    run_fast_subset,
    TBLITE_FAST_SUBSET,
)


def test_fast_subset_has_20_tasks():
    assert len(TBLITE_FAST_SUBSET) == 20


def test_compare_passes_on_equal():
    baseline = {"pass_rate": 0.8, "passed": 16, "failed": 4, "total": 20, "per_task": {}}
    evolved = {"pass_rate": 0.8, "passed": 16, "failed": 4, "total": 20, "per_task": {}}
    result = compare(baseline, evolved)
    assert result.passed


def test_compare_passes_on_improvement():
    baseline = {"pass_rate": 0.7, "passed": 14, "failed": 6, "total": 20, "per_task": {}}
    evolved = {"pass_rate": 0.85, "passed": 17, "failed": 3, "total": 20, "per_task": {}}
    result = compare(baseline, evolved)
    assert result.passed
    assert result.regression < 0


def test_compare_fails_on_large_regression():
    baseline = {"pass_rate": 0.9, "passed": 18, "failed": 2, "total": 20, "per_task": {}}
    evolved = {"pass_rate": 0.8, "passed": 16, "failed": 4, "total": 20, "per_task": {}}
    result = compare(baseline, evolved, regression_threshold=0.05)
    # 0.9 - 0.8 = 0.10 regression > 0.05 threshold → FAIL
    assert not result.passed


def test_compare_uses_custom_threshold():
    baseline = {"pass_rate": 0.9, "passed": 18, "failed": 2, "total": 20, "per_task": {}}
    evolved = {"pass_rate": 0.85, "passed": 17, "failed": 3, "total": 20, "per_task": {}}
    # 0.9 - 0.85 = 0.05 regression. Threshold 0.1 → pass
    result = compare(baseline, evolved, regression_threshold=0.1)
    assert result.passed
    # Threshold 0.03 → fail
    result2 = compare(baseline, evolved, regression_threshold=0.03)
    assert not result2.passed


def test_compare_handles_error():
    baseline = {"error": "batch_runner not found", "pass_rate": 0.0, "passed": 0, "failed": 0, "total": 0, "per_task": {}}
    evolved = {"pass_rate": 0.0, "passed": 0, "failed": 0, "total": 0, "per_task": {}}
    result = compare(baseline, evolved)
    assert not result.passed
    assert "error" in result.message.lower()


def test_run_fast_subset_nonexistent_repo():
    result = run_fast_subset(Path("/nonexistent"), task_list=["test_task"])
    assert result["error"] is not None
    assert result["pass_rate"] == 0.0

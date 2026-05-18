"""Tests for Phase 1 gate evaluation."""

from evolution.core.config import EvolutionConfig
from evolution.core.phase_gate import (
    REQUIRED_MANIFEST_FIELDS,
    evaluate_phase1_gate,
)


def _manifest_fixture() -> dict:
    return {field: "value" for field in REQUIRED_MANIFEST_FIELDS} | {
        "dataset_counts": {"train": 1, "val": 1, "holdout": 1},
        "baseline_score": 0.5,
        "evolved_score": 0.6,
        "improvement": 0.1,
        "iterations": 1,
        "elapsed_seconds": 1.0,
    }


def test_phase1_gate_passes_with_valid_improvement_and_manifest():
    config = EvolutionConfig()
    result = evaluate_phase1_gate(
        baseline_score=0.5,
        evolved_score=0.6,
        improvement=0.1,
        config=config,
        manifest=_manifest_fixture(),
    )
    assert result.passed
    assert result.failures == []


def test_phase1_gate_fails_without_improvement():
    config = EvolutionConfig()
    result = evaluate_phase1_gate(
        baseline_score=0.8,
        evolved_score=0.79,
        improvement=-0.01,
        config=config,
        manifest=_manifest_fixture(),
    )
    assert not result.passed
    assert any("Improvement gate failed" in failure for failure in result.failures)


def test_phase1_gate_fails_when_manifest_missing_required_fields():
    config = EvolutionConfig()
    bad_manifest = {"skill_name": "x"}
    result = evaluate_phase1_gate(
        baseline_score=0.5,
        evolved_score=0.6,
        improvement=0.1,
        config=config,
        manifest=bad_manifest,
    )
    assert not result.passed
    assert any("Reproducibility manifest" in failure for failure in result.failures)


def test_phase1_gate_fails_on_benchmark_regression():
    config = EvolutionConfig(phase1_max_benchmark_regression=0.01)
    result = evaluate_phase1_gate(
        baseline_score=0.5,
        evolved_score=0.55,
        improvement=0.05,
        config=config,
        manifest=_manifest_fixture(),
        benchmark_regression=0.02,
    )
    assert not result.passed
    assert any("Benchmark regression" in failure for failure in result.failures)

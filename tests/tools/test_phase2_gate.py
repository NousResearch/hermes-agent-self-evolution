"""Tests for the Phase 2 gate and the tool reproducibility manifest."""

from pathlib import Path

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.core.phase_gate import evaluate_phase2_gate
from evolution.core.reproducibility import build_tool_reproducibility_manifest


def _make_manifest(tmp_path: Path, baseline=0.5, evolved=0.65):
    dataset = EvalDataset(
        train=[EvalExample(task_input="a", expected_behavior="yes")],
        val=[EvalExample(task_input="b", expected_behavior="no")],
        holdout=[EvalExample(task_input="c", expected_behavior="yes")],
    )
    return build_tool_reproducibility_manifest(
        tool_name="read_file",
        tool_source_path=Path("/fake/tools/file_tools.py"),
        baseline_description="Read a text file by path.",
        dataset=dataset,
        config=EvolutionConfig(phase0_enforce=False),
        baseline_score=baseline,
        evolved_score=evolved,
        improvement=evolved - baseline,
        elapsed_seconds=12.0,
        evolution_repo_path=tmp_path,
    )


class TestEvaluatePhase2Gate:
    def test_passes_on_strong_lift_with_no_cross_tool_regression(self, tmp_path):
        manifest = _make_manifest(tmp_path, baseline=0.5, evolved=0.65)
        result = evaluate_phase2_gate(
            baseline_score=0.5,
            evolved_score=0.65,
            improvement=0.15,
            config=EvolutionConfig(phase0_enforce=False),
            manifest=manifest,
            per_tool_regression={"peer_a": 0.0, "peer_b": 0.01},
        )
        assert result.passed, result.failures
        assert result.checks["cross_tool_no_regression"] is True

    def test_fails_when_peer_regresses(self, tmp_path):
        manifest = _make_manifest(tmp_path, baseline=0.5, evolved=0.65)
        result = evaluate_phase2_gate(
            baseline_score=0.5,
            evolved_score=0.65,
            improvement=0.15,
            config=EvolutionConfig(phase0_enforce=False),
            manifest=manifest,
            per_tool_regression={"peer_a": -0.10},
        )
        assert not result.passed
        assert any("Cross-tool regression" in f for f in result.failures)

    def test_fails_without_required_manifest(self, tmp_path):
        result = evaluate_phase2_gate(
            baseline_score=0.5,
            evolved_score=0.65,
            improvement=0.15,
            config=EvolutionConfig(phase0_enforce=False),
            manifest={"some": "junk"},
            per_tool_regression={},
        )
        assert not result.passed
        assert any("manifest" in f.lower() for f in result.failures)

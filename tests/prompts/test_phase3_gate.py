"""Tests for the Phase 3 gate and the prompt-section reproducibility manifest."""

from pathlib import Path

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.core.phase_gate import evaluate_phase3_gate
from evolution.core.reproducibility import build_prompt_section_reproducibility_manifest


def _make_manifest(tmp_path: Path, baseline=0.4, evolved=0.55):
    dataset = EvalDataset(
        train=[EvalExample(task_input="a", expected_behavior="durable facts")],
        val=[EvalExample(task_input="b", expected_behavior="preferences")],
        holdout=[EvalExample(task_input="c", expected_behavior="memory")],
    )
    return build_prompt_section_reproducibility_manifest(
        section_name="MEMORY_GUIDANCE",
        section_source_path=Path("/fake/agent/prompt_builder.py"),
        baseline_text="Save durable facts.",
        dataset=dataset,
        config=EvolutionConfig(phase0_enforce=False),
        baseline_score=baseline,
        evolved_score=evolved,
        improvement=evolved - baseline,
        elapsed_seconds=5.0,
        evolution_repo_path=tmp_path,
    )


class TestEvaluatePhase3Gate:
    def test_passes_on_strong_lift(self, tmp_path):
        manifest = _make_manifest(tmp_path, baseline=0.4, evolved=0.55)
        result = evaluate_phase3_gate(
            baseline_score=0.4,
            evolved_score=0.55,
            improvement=0.15,
            config=EvolutionConfig(phase0_enforce=False),
            manifest=manifest,
        )
        assert result.passed, result.failures

    def test_fails_on_missing_identity_trait(self, tmp_path):
        manifest = _make_manifest(tmp_path, baseline=0.4, evolved=0.6)
        result = evaluate_phase3_gate(
            baseline_score=0.4,
            evolved_score=0.6,
            improvement=0.20,
            config=EvolutionConfig(phase0_enforce=False),
            manifest=manifest,
            identity_traits={"helpful": True, "direct": False, "admits_uncertainty": True},
        )
        assert not result.passed
        assert any("Identity traits" in f for f in result.failures)

    def test_identity_traits_optional_when_none(self, tmp_path):
        manifest = _make_manifest(tmp_path, baseline=0.4, evolved=0.55)
        result = evaluate_phase3_gate(
            baseline_score=0.4,
            evolved_score=0.55,
            improvement=0.15,
            config=EvolutionConfig(phase0_enforce=False),
            manifest=manifest,
            identity_traits=None,
        )
        assert result.passed
        assert result.checks["identity_traits_preserved"] is True

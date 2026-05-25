"""Unit tests for `build_run_inputs`."""

from __future__ import annotations

from evolution.core.config import EvolutionConfig
from evolution.core.hermes_provider import resolved_lms_dump
from evolution.core.run_inputs import build_run_inputs


def _fake_config() -> EvolutionConfig:
    return EvolutionConfig(
        seed=42,
        reflection_model="openai/gpt-4.1",
        eval_model="openai/gpt-4.1-mini",
        eval_dataset_size=150,
        holdout_ratio=0.5,
        enable_confusable_bucket=False,
    )


class TestBuildRunInputs:
    def test_skill_side_shape(self):
        config = _fake_config()
        result = build_run_inputs(
            config=config,
            iterations=10,
            optimizer_model="openai/gpt-4.1",
            quality_gate_preset="default",
            eval_source="synthetic",
            gepa_acceptance="improvement_or_equal",
            create_pr=False,
        )
        assert set(result.keys()) == {
            "seed",
            "iterations",
            "optimizer_model",
            "reflection_model",
            "eval_model",
            "resolved_lms",
            "eval_dataset_size",
            "holdout_ratio",
            "quality_gate_preset",
            "eval_source",
            "gepa_acceptance",
            "create_pr",
        }
        assert result["gepa_acceptance"] == "improvement_or_equal"
        assert result["create_pr"] is False

    def test_tool_side_adds_fitness_profile_and_confusable_bucket(self):
        config = _fake_config()
        config.enable_confusable_bucket = True
        result = build_run_inputs(
            config=config,
            iterations=10,
            optimizer_model="openai/gpt-4.1",
            quality_gate_preset="default",
            eval_source="synthetic",
            gepa_acceptance="strict_improvement",
            create_pr=True,
            fitness_profile="balanced",
            enable_confusable_bucket=True,
        )
        assert set(result.keys()) == {
            "seed",
            "iterations",
            "optimizer_model",
            "reflection_model",
            "eval_model",
            "resolved_lms",
            "eval_dataset_size",
            "holdout_ratio",
            "quality_gate_preset",
            "eval_source",
            "gepa_acceptance",
            "create_pr",
            "fitness_profile",
            "enable_confusable_bucket",
        }
        assert result["gepa_acceptance"] == "strict_improvement"
        assert result["fitness_profile"] == "balanced"
        assert result["enable_confusable_bucket"] is True
        assert result["create_pr"] is True

    def test_resolved_lms_matches_helper_output(self):
        config = _fake_config()
        result = build_run_inputs(
            config=config,
            iterations=10,
            optimizer_model="openai/gpt-4.1",
            quality_gate_preset="default",
            eval_source="synthetic",
            gepa_acceptance="improvement_or_equal",
            create_pr=False,
        )
        expected = resolved_lms_dump(
            optimizer="openai/gpt-4.1",
            reflection=config.reflection_model,
            eval_=config.eval_model,
        )
        assert result["resolved_lms"] == expected

    def test_enable_confusable_bucket_round_trips_when_passed(self):
        # Helper round-trip only. Call-site coverage that the deploy-gate
        # paths actually pass this kwarg lives in
        # `TestGateDecisionSchemaOnDeploy::test_gate_decision_schema_on_deploy`.
        config = _fake_config()
        config.enable_confusable_bucket = True
        result = build_run_inputs(
            config=config,
            iterations=10,
            optimizer_model="openai/gpt-4.1",
            quality_gate_preset="default",
            eval_source="synthetic",
            gepa_acceptance="improvement_or_equal",
            create_pr=False,
            fitness_profile="balanced",
            enable_confusable_bucket=config.enable_confusable_bucket,
        )
        assert "enable_confusable_bucket" in result
        assert result["enable_confusable_bucket"] is True

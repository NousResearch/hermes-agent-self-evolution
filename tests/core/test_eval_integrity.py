"""Tests for evaluation integrity utilities."""

from pathlib import Path

import pytest

from evolution.core.eval_integrity import (
    estimate_effect_size,
    frozen_holdout_manifest,
    model_family,
    validate_model_separation,
)


def test_model_family_parsing():
    assert model_family("openai/gpt-4.1") == "openai"
    assert model_family("anthropic:claude") == "anthropic"
    assert model_family("gemini-2.5-flash") == "gemini"


def test_validate_model_separation_requires_distinct_families():
    errors = validate_model_separation(
        generator_model="openai/gpt-4.1",
        judge_model="openai/gpt-4.1-mini",
        optimizer_model="anthropic/claude-3.5",
    )
    assert errors
    assert "at least 3 distinct families" in errors[0]


def test_validate_model_separation_can_allow_two_families():
    errors = validate_model_separation(
        generator_model="openai/gpt-4.1",
        judge_model="openai/gpt-4.1-mini",
        optimizer_model="anthropic/claude-3.5",
        minimum_distinct_families=2,
    )
    assert errors == []


def test_frozen_holdout_manifest_is_stable(tmp_path: Path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text("a\n")
    second.write_text("b\n")

    manifest = frozen_holdout_manifest([second, first])
    keys = list(manifest.keys())
    assert keys == [str(first), str(second)]
    assert len(manifest[str(first)]) == 64


def test_estimate_effect_size_significance():
    estimate = estimate_effect_size(
        baseline_scores=[0.2, 0.3, 0.4, 0.3],
        candidate_scores=[0.4, 0.5, 0.6, 0.5],
        minimum_detectable_effect=0.05,
    )
    assert estimate.delta > 0
    assert estimate.significant


def test_estimate_effect_size_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal length"):
        estimate_effect_size([0.1], [0.1, 0.2], minimum_detectable_effect=0.01)

"""Tests for guidance fitness (no API calls)."""

from evolution.prompts.guidance_fitness import (
    behavioral_score, token_efficiency_multiplier,
    contradiction_penalty, composite_fitness,
)


def test_token_efficiency_bonus_for_shorter():
    assert token_efficiency_multiplier("short", "very long baseline text") >= 1.0


def test_token_efficiency_penalty_for_longer():
    longer = "x" * 110
    baseline = "x" * 100
    mult = token_efficiency_multiplier(longer, baseline)
    assert mult < 1.0


def test_token_efficiency_raises_for_excessive_growth():
    import pytest
    with pytest.raises(ValueError, match="exceeds"):
        token_efficiency_multiplier("x" * 300, "x" * 100)


def test_token_efficiency_exact_match():
    assert token_efficiency_multiplier("same", "same") == 1.0


def test_contradiction_penalty_zero_for_identical():
    texts = {"A": "Always use tools for execution.", "B": "Always use tools for execution."}
    assert contradiction_penalty(texts) == 0.0


def test_contradiction_penalty_zero_for_low_overlap():
    texts = {"A": "Always use tools.", "B": "Never use tools, describe instead."}
    pen = contradiction_penalty(texts)
    assert pen == 0.0


def test_contradiction_penalty_positive_for_high_overlap_conflicting():
    # Two blocks with very similar wording (>85% word overlap) but opposing intent
    # Both share most words except the key action (use vs describe)
    texts = {
        "A": "Always use tools to execute commands directly when possible do not describe commands.",
        "B": "Always describe steps to execute commands directly when possible do not use tools.",
    }
    pen = contradiction_penalty(texts)
    assert pen > 0.0
    assert pen <= 1.0


def test_composite_fitness():
    scores = {"MEMORY_GUIDANCE": 0.8, "TASK_COMPLETION_GUIDANCE": 0.9}
    evolved = {"MEMORY_GUIDANCE": "short", "TASK_COMPLETION_GUIDANCE": "also short"}
    baseline = {"MEMORY_GUIDANCE": "very long baseline text", "TASK_COMPLETION_GUIDANCE": "also very long baseline"}
    fit = composite_fitness(scores, evolved, baseline)
    assert 0.0 <= fit <= 2.0

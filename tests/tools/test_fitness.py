"""Tests for evolution/tools/fitness.py — contrastive scoring."""

from __future__ import annotations

import dspy
import pytest

from evolution.tools.fitness import tool_fitness_metric


def _example(category: str) -> dspy.Example:
    return dspy.Example(
        task_input="some task",
        expected_behavior="some behavior",
        category=category,
    ).with_inputs("task_input")


def _prediction(decision: str, rationale: str = "") -> dspy.Prediction:
    return dspy.Prediction(output=decision, rationale=rationale)


# ── correct decisions ────────────────────────────────────────────────────


def test_yes_on_positive_scores_one():
    ex = _example("positive")
    pred = _prediction("yes")
    result = tool_fitness_metric(ex, pred)
    assert result.score == 1.0


def test_no_on_negative_scores_one():
    ex = _example("negative")
    pred = _prediction("no")
    result = tool_fitness_metric(ex, pred)
    assert result.score == 1.0


# ── wrong decisions ──────────────────────────────────────────────────────


def test_no_on_positive_scores_zero():
    ex = _example("positive")
    pred = _prediction("no", rationale="thought it was a different tool")
    result = tool_fitness_metric(ex, pred)
    assert result.score == 0.0
    assert "Wrong" in result.feedback
    assert "should have been" in result.feedback


def test_yes_on_negative_scores_zero():
    ex = _example("negative")
    pred = _prediction("yes")
    result = tool_fitness_metric(ex, pred)
    assert result.score == 0.0


# ── feedback content ─────────────────────────────────────────────────────


def test_feedback_includes_rationale_excerpt():
    ex = _example("positive")
    pred = _prediction("no", rationale="long rationale " * 20)
    result = tool_fitness_metric(ex, pred)
    # Feedback must mention rationale; long rationales are truncated
    assert "long rationale" in result.feedback


def test_correct_feedback_is_terse():
    ex = _example("positive")
    pred = _prediction("yes")
    result = tool_fitness_metric(ex, pred)
    assert "Correct" in result.feedback


# ── missing labels ───────────────────────────────────────────────────────


def test_missing_polarity_returns_zero_score():
    ex = dspy.Example(
        task_input="t", expected_behavior="b", category="medium"
    ).with_inputs("task_input")
    pred = _prediction("yes")
    result = tool_fitness_metric(ex, pred)
    assert result.score == 0.0
    assert "polarity" in result.feedback.lower()


def test_no_category_attribute_returns_zero_score():
    ex = dspy.Example(task_input="t", expected_behavior="b").with_inputs("task_input")
    pred = _prediction("yes")
    result = tool_fitness_metric(ex, pred)
    assert result.score == 0.0


# ── GEPA 5-arg signature ─────────────────────────────────────────────────


def test_accepts_gepa_5_arg_signature():
    """GEPA passes (gold, pred, trace, pred_name, pred_trace). Must not crash."""
    ex = _example("positive")
    pred = _prediction("yes")
    result = tool_fitness_metric(
        ex, pred,
        trace=[("step1", "result1")],
        pred_name="predictor",
        pred_trace={"k": "v"},
    )
    assert result.score == 1.0


# ── case insensitivity ───────────────────────────────────────────────────


def test_polarity_is_case_insensitive():
    ex = dspy.Example(
        task_input="t",
        expected_behavior="b",
        category="POSITIVE",
    ).with_inputs("task_input")
    pred = _prediction("yes")
    result = tool_fitness_metric(ex, pred)
    assert result.score == 1.0

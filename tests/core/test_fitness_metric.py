"""Tests for DSPy-compatible fitness metric behavior."""

import dspy

from evolution.core.fitness import skill_fitness_metric


def test_skill_fitness_metric_classic_shape_returns_float():
    example = dspy.Example(
        task_input="Debug a failing test",
        expected_behavior="Read the error, reproduce, and verify the fix.",
    ).with_inputs("task_input")
    prediction = dspy.Prediction(output="Read the error, reproduce the failure, then verify the fix.")

    score = skill_fitness_metric(example, prediction)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_skill_fitness_metric_gepa_feedback_shape_returns_score_and_feedback():
    example = dspy.Example(
        task_input="Debug a failing test",
        expected_behavior="Read the error, reproduce, and verify the fix.",
    ).with_inputs("task_input")
    prediction = dspy.Prediction(output="Read the error, reproduce the failure, then verify the fix.")

    result = skill_fitness_metric(
        example,
        prediction,
        trace=None,
        pred_name="predict",
        pred_trace=None,
    )

    assert isinstance(result, tuple)
    score, feedback = result
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert "Score" in feedback
    assert "Expected behavior" in feedback

"""Tests for fitness scoring helpers."""

import dspy

from evolution.core.fitness import skill_fitness_metric


def test_oversize_skill_prediction_gets_zero_fitness():
    example = dspy.Example(
        task_input="Summarize the procedure",
        expected_behavior="summary procedure",
    )
    prediction = dspy.Prediction(
        output="summary procedure",
        skill_text="x" * 15_001,
    )

    assert skill_fitness_metric(example, prediction) == 0.0


def test_near_limit_skill_prediction_gets_length_penalty():
    example = dspy.Example(
        task_input="Summarize the procedure",
        expected_behavior="summary procedure",
    )
    prediction = dspy.Prediction(
        output="summary procedure",
        skill_text="x" * 14_250,
    )

    assert skill_fitness_metric(example, prediction) < 1.0


def test_normal_size_skill_prediction_keeps_keyword_score():
    example = dspy.Example(
        task_input="Summarize the procedure",
        expected_behavior="summary procedure",
    )
    prediction = dspy.Prediction(
        output="summary procedure",
        skill_text="short skill",
    )

    assert skill_fitness_metric(example, prediction) == 1.0

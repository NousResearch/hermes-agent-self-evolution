"""Tests for evolution.core.fitness — fitness metrics and LLM judge."""

import dspy
import pytest
from evolution.core.fitness import (
    skill_fitness_metric,
    LLMJudge,
    FitnessScore,
    _parse_score,
)
from evolution.core.config import EvolutionConfig


class TestFitnessScore:
    def test_composite_calculation(self):
        """Composite = 0.5*correctness + 0.3*procedure + 0.2*conciseness."""
        score = FitnessScore(correctness=1.0, procedure_following=1.0, conciseness=1.0)
        assert abs(score.composite - 1.0) < 0.01

    def test_length_penalty_applied(self):
        """Length penalty reduces composite score."""
        base = FitnessScore(correctness=1.0, procedure_following=1.0, conciseness=1.0, length_penalty=0.0)
        penalized = FitnessScore(correctness=1.0, procedure_following=1.0, conciseness=1.0, length_penalty=0.5)
        assert penalized.composite < base.composite

    def test_zero_minimum(self):
        """Composite is never below 0."""
        score = FitnessScore(correctness=-1, procedure_following=-1, conciseness=-1, length_penalty=10)
        assert score.composite >= 0.0


class TestParseScore:
    def test_float_passthrough(self):
        assert _parse_score(0.75) == 0.75

    def test_int_conversion(self):
        assert _parse_score(1) == 1.0

    def test_string_conversion(self):
        assert _parse_score("0.85") == 0.85

    def test_clamps_above_1(self):
        assert _parse_score(5.0) == 1.0

    def test_clamps_below_0(self):
        assert _parse_score(-1.0) == 0.0

    def test_parse_failure_defaults_to_05(self):
        assert _parse_score("not-a-number") == 0.5


class TestSkillFitnessMetric:
    def test_empty_output_returns_zero(self):
        ex = dspy.Example(task_input="do something", expected_behavior="do it well").with_inputs("task_input", "expected_behavior")
        pred = dspy.Prediction(output="")
        score = skill_fitness_metric(ex, pred)
        assert score == 0.0

    def test_whitespace_output_returns_zero(self):
        ex = dspy.Example(task_input="do something", expected_behavior="do it well").with_inputs("task_input", "expected_behavior")
        pred = dspy.Prediction(output="   \n   ")
        score = skill_fitness_metric(ex, pred)
        assert score == 0.0

    def test_perfect_match_returns_high_score(self):
        ex = dspy.Example(
            task_input="review this python code for security issues",
            expected_behavior="identify the sql injection vulnerability in the query string concatenation",
        ).with_inputs("task_input", "expected_behavior")
        pred = dspy.Prediction(output="Found SQL injection vulnerability: string concatenation in query. Fix: use parameterized queries.")
        score = skill_fitness_metric(ex, pred)
        assert score >= 0.5  # Substantial keyword overlap

    def test_totally_wrong_answer_returns_low_score(self):
        ex = dspy.Example(
            task_input="review this code for security",
            expected_behavior="identify the sql injection vulnerability",
        ).with_inputs("task_input", "expected_behavior")
        pred = dspy.Prediction(output="The code looks clean. No issues found.")
        score = skill_fitness_metric(ex, pred)
        assert score < 0.8  # Poor overlap

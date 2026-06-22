"""Tests for evolution.core.fitness — fitness metrics and LLM judge."""

from unittest.mock import MagicMock

import dspy
import pytest
from evolution.core.fitness import (
    skill_fitness_metric,
    LLMJudge,
    FitnessScore,
    _parse_score,
    reset_judge_cache,
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


def _make_example():
    return dspy.Example(
        task_input="review this code",
        expected_behavior="find the bug",
        skill_text="1. Read code\n2. Find bugs",
    ).with_inputs("task_input", "expected_behavior")


class TestLLMJudgeMode:
    def setup_method(self):
        reset_judge_cache()

    def teardown_method(self):
        reset_judge_cache()

    def _judge_returning(self, correctness, procedure, conciseness, feedback="ok"):
        """Build an LLMJudge whose .score() returns a fixed FitnessScore."""
        judge = LLMJudge(EvolutionConfig())
        judge.score = MagicMock(return_value=FitnessScore(
            correctness=correctness,
            procedure_following=procedure,
            conciseness=conciseness,
            feedback=feedback,
        ))
        return judge

    def test_judge_mode_uses_composite_score(self):
        judge = self._judge_returning(1.0, 1.0, 1.0)
        ex = _make_example()
        pred = dspy.Prediction(output="Found the bug on line 5.")
        score = skill_fitness_metric(ex, pred, judge=judge)
        # composite of all-1.0 = 0.5 + 0.3 + 0.2 = 1.0
        assert score == pytest.approx(1.0)
        judge.score.assert_called_once()

    def test_judge_mode_empty_output_short_circuits(self):
        judge = self._judge_returning(1.0, 1.0, 1.0)
        ex = _make_example()
        pred = dspy.Prediction(output="   ")
        score = skill_fitness_metric(ex, pred, judge=judge)
        assert score == 0.0
        # Judge must not be called for empty output
        judge.score.assert_not_called()

    def test_judge_results_are_cached(self):
        judge = self._judge_returning(0.8, 0.6, 0.4)
        ex = _make_example()
        pred = dspy.Prediction(output="Same output every time.")

        s1 = skill_fitness_metric(ex, pred, judge=judge)
        s2 = skill_fitness_metric(ex, pred, judge=judge)

        assert s1 == s2
        # Second call should hit the cache, not the judge
        judge.score.assert_called_once()

    def test_different_outputs_not_cache_collision(self):
        judge = self._judge_returning(0.5, 0.5, 0.5)
        ex = _make_example()
        skill_fitness_metric(ex, dspy.Prediction(output="answer one"), judge=judge)
        skill_fitness_metric(ex, dspy.Prediction(output="answer two"), judge=judge)
        # Distinct outputs → two distinct judge calls
        assert judge.score.call_count == 2

"""Tests for the verifier registry and the DSPy metric adapter."""

import dspy
import pytest

from evolution.core.dataset_builder import EvalDataset
from evolution.core.fitness import FitnessScore
from evolution.core.verifier import (
    Verifier,
    get_verifier,
    register_verifier,
    registered_skills,
    verifier_metric,
)


class StubVerifier(Verifier):
    """Minimal verifier: the answer to everything is 42."""

    skill_name = "stub"

    def build_dataset(self, num_cases: int = 24, seed: int = 13) -> EvalDataset:
        return EvalDataset()

    def score(self, task_input: str, output: str) -> FitnessScore:
        correct = "42" in output
        return FitnessScore(
            correctness=1.0 if correct else 0.0,
            procedure_following=1.0 if correct else 0.0,
            conciseness=1.0 if correct else 0.0,
            feedback="Correct." if correct else "Expected the answer 42.",
        )


class TestRegistry:
    def test_arxiv_verifier_is_registered(self):
        verifier = get_verifier("arxiv")
        assert verifier is not None
        assert verifier.skill_name == "arxiv"

    def test_unknown_skill_returns_none(self):
        assert get_verifier("no-such-skill") is None

    def test_registered_skills_includes_arxiv(self):
        assert "arxiv" in registered_skills()

    def test_register_requires_skill_name(self):
        with pytest.raises(ValueError):
            @register_verifier
            class Nameless(Verifier):  # noqa: F811
                skill_name = ""

                def build_dataset(self, num_cases=24, seed=13):
                    return EvalDataset()

                def score(self, task_input, output):
                    return FitnessScore()


class TestVerifierMetric:
    def setup_method(self):
        self.metric = verifier_metric(StubVerifier())
        self.gold = dspy.Example(task_input="what is the answer?").with_inputs("task_input")

    def test_returns_float_for_two_arg_call(self):
        score = self.metric(self.gold, dspy.Prediction(output="the answer is 42"))
        assert isinstance(score, float)
        assert score == pytest.approx(1.0)

    def test_returns_float_for_three_arg_call(self):
        score = self.metric(self.gold, dspy.Prediction(output="I do not know"), None)
        assert isinstance(score, float)
        assert score == pytest.approx(0.0)

    def test_gepa_call_returns_score_with_feedback(self):
        result = self.metric(
            self.gold, dspy.Prediction(output="42"), None, "predictor", None
        )
        assert result.score == pytest.approx(1.0)
        assert "Correct" in result.feedback

    def test_gepa_feedback_explains_failure(self):
        result = self.metric(
            self.gold, dspy.Prediction(output="probably 7"), None, "predictor", None
        )
        assert result.score == pytest.approx(0.0)
        assert "42" in result.feedback

    def test_handles_missing_fields_gracefully(self):
        score = self.metric(dspy.Example().with_inputs(), dspy.Prediction())
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_metric_name_identifies_skill(self):
        assert self.metric.__name__ == "stub_verifier_metric"

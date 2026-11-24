"""Tests for the shared GEPA feedback-metric factory.

The wrapper must survive being called with 2 args (dspy evaluator), 3 args
(bootstrap few-shot) and 5 args (GEPA reflection), because dspy uses one
metric object for all three roles.
"""

import dspy
import pytest

from evolution.core.gepa_metric import make_feedback_metric


def _scorer(score):
    def score_fn(gold, pred, trace=None):
        assert gold == "gold"
        assert pred == "pred"
        return score

    return score_fn


@pytest.mark.parametrize("nargs", [2, 3, 5])
def test_high_score_returns_prediction_with_score(nargs):
    metric = make_feedback_metric(_scorer(0.8))
    args = ["gold", "pred", None, "predict", None][:nargs]

    result = metric(*args)

    assert isinstance(result, dspy.Prediction)
    assert result.score == pytest.approx(0.8)


@pytest.mark.parametrize("nargs", [2, 3, 5])
def test_low_score_feedback_mentions_missed(nargs):
    metric = make_feedback_metric(_scorer(0.2))
    args = ["gold", "pred", None, "predict", None][:nargs]

    result = metric(*args)

    assert result.score == pytest.approx(0.2)
    assert "missed" in result.feedback.lower()


def test_high_score_feedback_mentions_coverage():
    result = make_feedback_metric(_scorer(0.8))("gold", "pred")

    assert "coverage" in result.feedback.lower()


def test_score_at_half_is_neutral_not_success():
    """0.5 is skill_fitness_metric's no-signal base, not a pass."""
    result = make_feedback_metric(lambda g, p, t=None: 0.5)("gold", "pred")

    assert "coverage" not in result.feedback.lower()
    assert "missed" not in result.feedback.lower()
    assert "signal" in result.feedback.lower()


def test_trace_is_forwarded_to_score_fn():
    seen = {}

    def score_fn(gold, pred, trace=None):
        seen["trace"] = trace
        return 1.0

    make_feedback_metric(score_fn)("gold", "pred", ["step"])

    assert seen["trace"] == ["step"]

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


@pytest.mark.parametrize("score,expect", [
    (0.44, "missed"),
    (0.45, "signal"),
    (0.46, "signal"),
    (0.5, "signal"),
    (0.51, "coverage"),
])
def test_feedback_band_edges(score, expect):
    result = make_feedback_metric(lambda g, p, t=None: score)("gold", "pred")

    assert expect in result.feedback.lower()


def test_trace_is_forwarded_to_score_fn():
    seen = {}

    def score_fn(gold, pred, trace=None):
        seen["trace"] = trace
        return 1.0

    make_feedback_metric(score_fn)("gold", "pred", ["step"])

    assert seen["trace"] == ["step"]

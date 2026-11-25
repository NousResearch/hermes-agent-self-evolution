"""Shared GEPA feedback metric factory.

All five metric parameters carry defaults: dspy.GEPA's reflection step calls
the metric with five arguments, dspy's evaluator and bootstrap paths call the
same object with two or three.
"""

from typing import Any, Callable

import dspy

SUCCESS_FEEDBACK = "met expected behavior; tighten coverage"
NEUTRAL_FEEDBACK = "output carried no clear signal on the expected behavior; make the expected procedure explicit"
FAILURE_FEEDBACK = "missed key parts of the expected behavior"


def make_feedback_metric(score_fn: Callable[..., float]) -> Callable[..., dspy.Prediction]:
    """Wrap a ``score_fn(gold, pred, trace)`` scalar metric for GEPA.

    Returns a callable usable as a dspy evaluator metric *and* as a GEPA
    feedback metric: it always yields a ``dspy.Prediction`` carrying ``score``
    and a short ``feedback`` string the reflection LM can act on.
    """

    def metric(
        gold: Any,
        pred: Any,
        trace: Any = None,
        pred_name: Any = None,
        pred_trace: Any = None,
    ) -> dspy.Prediction:
        score = score_fn(gold, pred, trace)
        if score > 0.5:
            feedback = SUCCESS_FEEDBACK
        elif score < 0.45:
            feedback = FAILURE_FEEDBACK
        else:
            feedback = NEUTRAL_FEEDBACK
        return dspy.Prediction(score=score, feedback=feedback)

    return metric

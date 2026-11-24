"""Shared GEPA feedback metric factory.

dspy.GEPA's reflection step calls the metric with five arguments
``(gold, pred, trace, pred_name, pred_trace)``, while dspy's evaluator and
bootstrap paths call the same metric object with two or three. A strict
five-argument signature therefore raises TypeError for every evaluation and
shows up as a misleading "Failed to unpack prediction" warning with an
all-zero score, so all five parameters carry defaults and the extra GEPA
arguments are accepted but ignored.
"""

from typing import Any, Callable

import dspy

SUCCESS_FEEDBACK = "met expected behavior; tighten coverage"
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
        feedback = SUCCESS_FEEDBACK if score >= 0.5 else FAILURE_FEEDBACK
        return dspy.Prediction(score=score, feedback=feedback)

    return metric

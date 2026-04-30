"""Fitness metric for tool description evolution.

Tool descriptions are scored contrastively: positive examples (this tool
fits the task) should produce ``yes`` decisions, negative examples (this
tool does not fit) should produce ``no`` decisions. The metric returns
``1.0`` for a correct decision and ``0.0`` for an incorrect one, plus a
short feedback string GEPA can reflect on.
"""

from __future__ import annotations

import dspy


def tool_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> dspy.Prediction:
    """DSPy-compatible metric for tool description optimization.

    Accepts the 5-arg GEPA signature ``(gold, pred, trace, pred_name,
    pred_trace)`` so the same metric works under both GEPA and the
    older 3-arg MIPROv2 API.

    Returns a ``dspy.Prediction(score: float, feedback: str)``.
    """
    expected_polarity = _expected_polarity(example)
    decision = (getattr(prediction, "output", "") or "").strip().lower()
    rationale = (getattr(prediction, "rationale", "") or "").strip()

    if expected_polarity is None:
        # Example was not labelled with positive/negative; cannot score.
        return dspy.Prediction(
            score=0.0,
            feedback="Example missing polarity label (category must be 'positive' or 'negative').",
        )

    correct = (
        (expected_polarity == "positive" and decision == "yes")
        or (expected_polarity == "negative" and decision == "no")
    )
    score = 1.0 if correct else 0.0

    if correct:
        feedback = (
            f"Correct: tool was {decision} pick for a {expected_polarity} task."
        )
    else:
        wanted = "yes" if expected_polarity == "positive" else "no"
        feedback = (
            f"Wrong choice: returned {decision!r} on a {expected_polarity} task "
            f"(should have been {wanted!r}). "
            f"Description likely needs to clarify scope. Rationale was: "
            f"{rationale[:200] if rationale else 'none'}"
        )

    return dspy.Prediction(score=score, feedback=feedback)


def _expected_polarity(example) -> str | None:
    """Read the polarity from an EvalExample or DSPy Example.

    DSPy ``Example`` objects do not always preserve the ``category`` attribute;
    the dataset bridge sets it on the underlying example. We accept either
    attribute access or dict-like access so the metric works in both contexts.
    """
    polarity = getattr(example, "category", None)
    if polarity is None and hasattr(example, "get"):
        try:
            polarity = example.get("category")
        except Exception:
            polarity = None
    if polarity is None:
        # DSPy Examples may stash extra fields in `_store`
        store = getattr(example, "_store", None)
        if isinstance(store, dict):
            polarity = store.get("category")
    if isinstance(polarity, str):
        polarity = polarity.strip().lower()
        if polarity in ("positive", "negative"):
            return polarity
    return None

"""Fitness metric for tool-selection optimization.

Plain accuracy: does the picked tool name match the expected one? Cheap and
un-gameable — unlike the keyword-overlap heuristic used for skills.
"""

from typing import Optional

import dspy


def _normalize(name: str) -> str:
    cleaned = str(name).strip().strip("`'\"").rstrip(".,;:").strip().strip("`'\"")
    return cleaned.lower().replace("-", "_")


def tool_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name: str = "",
    pred_trace=None,
) -> float:
    """DSPy-compatible metric: 1.0 when the correct tool was chosen.

    Expected tool rides in `expected_behavior` (dataset reuse) or a
    dedicated `expected_tool` field.
    """
    expected = getattr(example, "expected_tool", None) or getattr(example, "expected_behavior", "") or ""
    picked = getattr(prediction, "tool_name", None) or getattr(prediction, "output", "") or ""

    # expected may be "Use grep to search..." — take the leading identifier
    expected_first = str(expected).split()[0].rstrip(".,;:") if str(expected).strip() else ""
    if not expected_first or not str(picked).strip():
        return 0.0
    return 1.0 if _normalize(picked) == _normalize(expected_first) else 0.0

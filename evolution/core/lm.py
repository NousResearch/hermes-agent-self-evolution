"""Bounded dspy.LM construction — every LM in the pipeline carries a timeout."""

from typing import Any

import dspy

DEFAULT_TIMEOUT = 120
DEFAULT_NUM_RETRIES = 2
REFLECTION_TIMEOUT = 180
REFLECTION_TEMPERATURE = 1.0
REFLECTION_MAX_TOKENS = 32000


def make_lm(model: str, reflection: bool = False) -> dspy.LM:
    """Build a bounded ``dspy.LM``.

    ``reflection=True`` returns the stronger sampling profile GEPA's reflection
    step expects (temperature 1.0, 32k tokens) with a longer timeout.
    """
    kwargs: dict[str, Any] = {"timeout": DEFAULT_TIMEOUT, "num_retries": DEFAULT_NUM_RETRIES}
    if reflection:
        kwargs.update(
            temperature=REFLECTION_TEMPERATURE,
            max_tokens=REFLECTION_MAX_TOKENS,
            timeout=REFLECTION_TIMEOUT,
        )
    return dspy.LM(model, **kwargs)

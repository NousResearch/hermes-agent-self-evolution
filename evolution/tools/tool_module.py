"""DSPy module wrapping a single tool description as the optimizable parameter.

The target tool's top-level `description` is stored as the predictor's
signature `instructions`, so prompt optimizers (MIPROv2, GEPA) mutate the
description text directly. The forward pass returns a yes/no decision on
whether the tool is the right choice for a given task, given the other
tools that are available.
"""

from __future__ import annotations

import dspy


class ToolDescriptionModule(dspy.Module):
    """Wrap one tool's description for selection-accuracy optimization."""

    class _ToolSelectionSignature(dspy.Signature):
        """Placeholder; instructions are overridden with the target tool's description."""

        task_input: str = dspy.InputField(desc="The user's request to the agent")
        tool_name: str = dspy.InputField(desc="The tool currently under evaluation")
        other_tools_summary: str = dspy.InputField(
            desc="One-line summaries of the other tools available to the agent",
        )
        decision: str = dspy.OutputField(
            desc="Reply exactly 'yes' if this tool is the best choice for the task, "
            "or 'no' if a different tool (or no tool) would be more appropriate.",
        )

    def __init__(self, tool_name: str, description: str, other_tools_summary: str = ""):
        super().__init__()
        self.tool_name = tool_name
        self.other_tools_summary = other_tools_summary or ""
        signature = self._ToolSelectionSignature.with_instructions(description or "")
        self.predictor = dspy.ChainOfThought(signature)

    def _inner_predict(self):
        return getattr(self.predictor, "predict", self.predictor)

    @property
    def tool_description(self) -> str:
        sig = self._inner_predict().signature
        return getattr(sig, "instructions", "") or ""

    @tool_description.setter
    def tool_description(self, value: str) -> None:
        inner = self._inner_predict()
        inner.signature = inner.signature.with_instructions(value or "")

    def forward(self, task_input: str, **_: object) -> dspy.Prediction:
        result = self.predictor(
            task_input=task_input,
            tool_name=self.tool_name,
            other_tools_summary=self.other_tools_summary,
        )
        return dspy.Prediction(decision=result.decision, output=result.decision)


_INLINE_EXAMPLE_MARKERS = (
    "\n\nExample:",
    "\nExample:",
    "\n\nTask Input:",
    "\nTask Input:",
    "Reasoning: Let's",
    "\nReasoning:",
    "\nDecision:",
)


def clean_evolved_description(instr: str, max_chars: int | None = None) -> str:
    """Strip MIPROv2-style inline few-shot demonstrations from the
    optimized instruction text so only the rewritten tool description
    survives. Tool-schema descriptions must stay short and prose-only;
    the optimizer's inline examples and reasoning templates are not
    part of the description we ship.

    If ``max_chars`` is provided and the trimmed description is still
    longer than the budget, drop trailing sentences (preserving the
    leading prose) until the result fits or only one sentence remains.
    """
    if not instr:
        return ""
    truncated = instr
    for marker in _INLINE_EXAMPLE_MARKERS:
        idx = truncated.find(marker)
        if idx > 0:
            truncated = truncated[:idx]
    truncated = truncated.strip()
    if max_chars is None or len(truncated) <= max_chars:
        return truncated

    import re

    sentences = re.split(r"(?<=[.!?])\s+", truncated)
    while len(sentences) > 1 and len(" ".join(sentences)) > max_chars:
        sentences.pop()
    fitted = " ".join(sentences).strip()
    if len(fitted) <= max_chars:
        return fitted
    return fitted[:max_chars].rstrip()


def normalize_decision(raw: str) -> str:
    text = (raw or "").strip().lower()
    if not text:
        return ""
    if text.startswith("yes") or "yes" == text:
        return "yes"
    if text.startswith("no") or "no" == text:
        return "no"
    if "yes" in text and "no" not in text:
        return "yes"
    if "no" in text and "yes" not in text:
        return "no"
    return ""


def tool_selection_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Binary score: 1.0 when the decision matches the labeled expected_behavior."""
    expected = (getattr(example, "expected_behavior", "") or "").strip().lower()
    predicted = normalize_decision(getattr(prediction, "decision", None) or getattr(prediction, "output", ""))
    if not expected or not predicted:
        return 0.0
    return 1.0 if predicted == expected else 0.0

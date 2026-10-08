"""Wraps a set of tool descriptions as a DSPy module for optimization.

The tool-selection problem is a classification task: given a user task and
the candidate tool descriptions, pick the right tool name. The concatenated
description block is set as the predictor's instructions — the exact search
space GEPA mutates.
"""

import re
from typing import Optional

import dspy


def format_descriptions(tools: dict[str, str]) -> str:
    """Render {name: description} as the instruction block GEPA mutates."""
    lines = ["You are a tool selector. Pick exactly one tool name for the task.", "", "## Tools"]
    lines += [f"{name}: {desc}" for name, desc in tools.items()]
    return "\n".join(lines)


def parse_descriptions(text: str) -> Optional[dict[str, str]]:
    """Parse the 'name: description' lines back into a dict.

    Returns None when the block does not round-trip (GEPA rewrote it into
    something unparseable) — callers treat that as a constraint failure.
    """
    tools: dict[str, str] = {}
    in_tools = False
    for line in text.splitlines():
        if line.strip() == "## Tools":
            in_tools = True
            continue
        if not in_tools or not line.strip():
            continue
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_\-]*):\s*(.+)$", line)
        if not m:
            return None
        tools[m.group(1)] = m.group(2)
    return tools or None


class ToolModule(dspy.Module):
    """A DSPy module that wraps tool descriptions for GEPA optimization."""

    class ToolChoice(dspy.Signature):
        """Choose the single best tool for the task."""
        task_input: str = dspy.InputField(desc="The task the user wants done")
        tool_name: str = dspy.OutputField(desc="Name of exactly one chosen tool")

    def __init__(self, descriptions_text: str):
        super().__init__()
        self.predictor = dspy.Predict(self.ToolChoice.with_instructions(descriptions_text))

    @property
    def descriptions_text(self) -> str:
        """Live view of the (possibly evolved) instructions GEPA optimizes."""
        return self.predictor.signature.instructions

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(tool_name=str(result.tool_name).strip())

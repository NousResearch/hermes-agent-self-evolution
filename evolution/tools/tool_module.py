"""DSPy module wrapping a tool manifest for description evolution.

ToolModule exposes one predictor (the `selector`). Its signature instructions
are the full rendered manifest with the target tool's description wrapped in
sentinel markers. GEPA mutates these instructions; BudgetAwareToolProposer
constrains the mutation to the sentinel-delimited region.
"""

from __future__ import annotations

import dspy

from evolution.tools.tool_source import (
    SentinelParseError,
    ToolManifest,
)


def _open_sentinel(target_name: str) -> str:
    return f"<!-- TARGET:{target_name} -->"


def _close_sentinel(target_name: str) -> str:
    return f"<!-- /TARGET:{target_name} -->"


def _render_manifest_for_prompt(
    manifest: ToolManifest,
    target_name: str,
    target_description: str,
) -> str:
    """Render the full manifest as markdown instructions for the selector.

    Tools appear in alphabetical-by-name order. The target tool's description
    is wrapped in sentinel markers; all other tool slots are byte-identical
    regardless of which target description is plugged in.
    """
    sorted_tools = sorted(manifest.tools, key=lambda t: t.name)
    blocks = []
    for tool in sorted_tools:
        if tool.name == target_name:
            description_block = (
                f"{_open_sentinel(target_name)}{target_description}{_close_sentinel(target_name)}"
            )
        else:
            description_block = tool.description
        blocks.append(f"## {tool.name}\n{description_block}")

    body = "\n\n".join(blocks)
    return (
        "You are picking the best tool for the user's task. Available tools:\n\n"
        f"{body}\n\n"
        "Pick exactly one tool name from the list above. Output the tool name only."
    )


def _extract_description_from_sentinels(
    instructions: str,
    target_name: str,
) -> str:
    """Inverse of _render_manifest_for_prompt's sentinel wrapping.

    Returns the text between the opening and closing markers for `target_name`.
    Raises SentinelParseError if the markers are missing, duplicated, or
    malformed.
    """
    open_marker = _open_sentinel(target_name)
    close_marker = _close_sentinel(target_name)

    open_count = instructions.count(open_marker)
    close_count = instructions.count(close_marker)

    if open_count == 0:
        raise SentinelParseError(
            f"opening sentinel {open_marker!r} not found in instructions"
        )
    if close_count == 0:
        raise SentinelParseError(
            f"closing sentinel {close_marker!r} not found in instructions"
        )
    if open_count > 1 or close_count > 1:
        raise SentinelParseError(
            f"sentinels for {target_name!r} appear multiple times "
            f"(open={open_count}, close={close_count})"
        )

    start = instructions.find(open_marker) + len(open_marker)
    end = instructions.find(close_marker)
    if end < start:
        raise SentinelParseError(
            f"closing sentinel {close_marker!r} precedes opening sentinel"
        )
    return instructions[start:end]


class ToolSelectionSignature(dspy.Signature):
    """Pick the best tool for the user's task.

    The signature instructions (installed via with_instructions per-instance)
    contain the full rendered manifest. Inputs and outputs declared here.
    """
    task: str = dspy.InputField(desc="The user task to pick a tool for")
    reasoning: str = dspy.OutputField(desc="Brief reasoning for the choice")
    chosen_tool: str = dspy.OutputField(desc="The chosen tool's name only")


class ToolModule(dspy.Module):
    """DSPy module exposing one selector predictor for tool description evolution.

    The selector's signature instructions are the full rendered manifest
    produced by `_render_manifest_for_prompt`. GEPA mutates these instructions
    via `named_predictors()`; `BudgetAwareToolProposer` (separate module)
    constrains mutations to the sentinel-delimited region.
    """

    def __init__(
        self,
        target_tool_name: str,
        manifest: ToolManifest,
        target_description: str,
    ):
        super().__init__()
        self.target_tool_name = target_tool_name
        self.manifest = manifest

        rendered = _render_manifest_for_prompt(
            manifest, target_tool_name, target_description
        )
        self.selector = dspy.ChainOfThought(ToolSelectionSignature)
        self.selector.predict.signature = (
            self.selector.predict.signature.with_instructions(rendered)
        )

    def forward(
        self,
        task: str,
        closed_loop_task_id: Optional[str] = None,
    ) -> dspy.Prediction:
        if closed_loop_task_id is not None:
            # Behavioral example: skip the selector LM call, stuff the
            # current candidate text into the Prediction so the metric's
            # behavioral branch can score via the closed-loop cache.
            # The metric reads these via getattr regardless of pred_trace,
            # so score is consistent across GEPA's Pareto-eval and
            # reflective-feedback paths.
            return dspy.Prediction(
                chosen_tool="",
                reasoning="",
                _closed_loop_task_id=closed_loop_task_id,
                _candidate_text=self.description_text,
            )
        result = self.selector(task=task)
        return dspy.Prediction(
            chosen_tool=result.chosen_tool,
            reasoning=result.reasoning,
        )

    @property
    def description_text(self) -> str:
        """Extract the current target description from the selector's instructions.

        Reads the sentinel-delimited region. Knee-point and the static
        validator use this to score / validate just the description rather
        than the full rendered manifest.
        """
        instructions = self.selector.predict.signature.instructions
        return _extract_description_from_sentinels(instructions, self.target_tool_name)

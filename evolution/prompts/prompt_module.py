"""PromptModule — DSPy module wrapping a prompt-section candidate.

Unlike ``ToolModule``, the predictor here is a passthrough: there is no
cheap "select a tool from the manifest" classification GEPA can score
without a real agent. Every meaningful eval requires a Hermes subprocess
(via closed-loop). The predictor exists only to give GEPA a place to hang
the candidate text via ``signature.instructions`` — GEPA mutates the
instructions, the framework extracts the candidate via ``section_text``,
and the closed-loop scorer runs it against the real agent.

Sentinel markers wrap the candidate region so ``section_text`` reads it
back unambiguously after GEPA's edits.

DO NOT "simplify" by dropping the predictor wrapper. GEPA discovers
optimization targets via ``dspy.Module.named_predictors()``, which only
returns objects with the predictor interface. A bare module with no
predictor child has nothing for GEPA to mutate.
"""

from __future__ import annotations

from typing import Optional

import dspy


class SentinelParseError(ValueError):
    """The candidate sentinels are missing, duplicated, or malformed."""


def _open_sentinel(section_name: str) -> str:
    return f"<!-- SECTION:{section_name} -->"


def _close_sentinel(section_name: str) -> str:
    return f"<!-- /SECTION:{section_name} -->"


def _render_instructions(section_name: str, candidate_text: str) -> str:
    return (
        f"The following is a candidate for the {section_name} section of an "
        f"agent's system prompt. Iteration mutates only the text between the "
        f"sentinel markers below.\n\n"
        f"{_open_sentinel(section_name)}{candidate_text}{_close_sentinel(section_name)}"
    )


def _extract_from_sentinels(instructions: str, section_name: str) -> str:
    open_marker = _open_sentinel(section_name)
    close_marker = _close_sentinel(section_name)
    open_count = instructions.count(open_marker)
    close_count = instructions.count(close_marker)
    if open_count == 0 or close_count == 0:
        raise SentinelParseError(
            f"sentinels for {section_name!r} not found in instructions "
            f"(open={open_count}, close={close_count})"
        )
    if open_count > 1 or close_count > 1:
        raise SentinelParseError(
            f"sentinels for {section_name!r} appear multiple times "
            f"(open={open_count}, close={close_count})"
        )
    start = instructions.find(open_marker) + len(open_marker)
    end = instructions.find(close_marker)
    if end < start:
        raise SentinelParseError(
            f"closing sentinel for {section_name!r} precedes opening sentinel"
        )
    return instructions[start:end]


class PromptPassthroughSignature(dspy.Signature):
    """Carrier for the candidate section text via signature.instructions.

    The input/output fields are placeholders; the real evaluation happens
    behaviorally via closed-loop, routed by the metric's behavioral branch.
    """

    task: str = dspy.InputField(desc="Placeholder; real evaluation is behavioral")
    response: str = dspy.OutputField(desc="Placeholder")


class PromptModule(dspy.Module):
    """DSPy module hosting a prompt-section candidate as predictor instructions."""

    def __init__(self, section_name: str, candidate_text: str):
        super().__init__()
        self.section_name = section_name
        self.passthrough = dspy.ChainOfThought(PromptPassthroughSignature)
        self.passthrough.predict.signature = (
            self.passthrough.predict.signature.with_instructions(
                _render_instructions(section_name, candidate_text)
            )
        )

    def forward(
        self,
        task: str,
        closed_loop_task_id: Optional[str] = None,
    ) -> dspy.Prediction:
        # Invoke the passthrough predictor so GEPA captures a trace for
        # ``passthrough.predict``. Without a traced predictor call, GEPA's
        # make_reflective_dataset finds "no valid predictions" and never
        # proposes a mutation (the tool path gets traces from synthetic
        # examples; prompt sections are pure-behavioral, so the trace has
        # to come from here). The predictor's output is a placeholder — the
        # real score comes from the metric's behavioral branch, which reads
        # the candidate text + task id attached below.
        result = self.passthrough(task=task)
        return dspy.Prediction(
            response=getattr(result, "response", ""),
            _closed_loop_task_id=closed_loop_task_id,
            _candidate_text=self.section_text,
        )

    @property
    def section_text(self) -> str:
        """Extract the current candidate text from the predictor's instructions."""
        instructions = self.passthrough.predict.signature.instructions
        return _extract_from_sentinels(instructions, self.section_name)

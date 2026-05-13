"""Tests for ToolModule.forward()'s behavioral-example branch.

When called with ``closed_loop_task_id`` set, the module must skip the
selector LM call entirely and return a Prediction carrying the task id
and the current sentinel-delimited description text. The metric's
behavioral branch then reads those fields and scores via the closed-loop
cache.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import dspy

from evolution.tools.tool_module import ToolModule
from evolution.tools.tool_source import ToolEntry, ToolManifest


def _build_manifest() -> ToolManifest:
    return ToolManifest(
        tools=(
            ToolEntry(
                name="write_file",
                description="Write content to a file.",
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
            ToolEntry(
                name="patch",
                description="Apply targeted edits.",
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
        ),
    )


class TestForwardBehavioralBranch:
    def test_selector_forward_not_invoked_when_task_id_set(self, monkeypatch):
        module = ToolModule(
            target_tool_name="write_file",
            manifest=_build_manifest(),
            target_description="Write content to a file.",
        )
        # Replace selector.forward so any LM call would error; leave the
        # selector itself intact so description_text can still read
        # selector.predict.signature.instructions.
        monkeypatch.setattr(
            module.selector,
            "forward",
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("selector LM path should not run on behavioral branch")
            ),
        )
        pred = module.forward(task="any task text", closed_loop_task_id="t1")
        # Marker fields prove the behavioral branch took (and the selector
        # forward wasn't invoked — otherwise the AssertionError above fires).
        assert pred._closed_loop_task_id == "t1"
        assert pred._candidate_text == "Write content to a file."

    def test_candidate_text_matches_description_text_property(self):
        module = ToolModule(
            target_tool_name="write_file",
            manifest=_build_manifest(),
            target_description="Write content to a file.",
        )
        pred = module.forward(task="placeholder", closed_loop_task_id="t1")
        assert pred._candidate_text == module.description_text
        # Sanity: candidate_text is the description, not the full manifest.
        assert pred._candidate_text == "Write content to a file."

    def test_non_behavioral_call_still_invokes_selector(self):
        module = ToolModule(
            target_tool_name="write_file",
            manifest=_build_manifest(),
            target_description="Write content to a file.",
        )
        module.selector = MagicMock(
            return_value=dspy.Prediction(chosen_tool="patch", reasoning="r")
        )
        pred = module.forward(task="some task")
        module.selector.assert_called_once_with(task="some task")
        assert pred.chosen_tool == "patch"
        assert not hasattr(pred, "_closed_loop_task_id")

    def test_pred_carries_empty_chosen_and_reasoning_on_behavioral(self):
        module = ToolModule(
            target_tool_name="write_file",
            manifest=_build_manifest(),
            target_description="Write content to a file.",
        )
        pred = module.forward(task="placeholder", closed_loop_task_id="t1")
        assert pred.chosen_tool == ""
        assert pred.reasoning == ""

"""Tests for evolution.core.behavioral_example."""

from __future__ import annotations

from pathlib import Path

import pytest

from evolution.core.behavioral_example import build_behavioral_examples
from evolution.validation.task import TaskSuite


def _write_suite(tmp_path: Path) -> TaskSuite:
    path = tmp_path / "suite.jsonl"
    path.write_text(
        '{"task_id": "b_task", "user_message": "do B"}\n'
        '{"task_id": "a_task", "user_message": "do A"}\n'
        '{"task_id": "c_task", "user_message": "do C"}\n'
    )
    return TaskSuite.from_jsonl(path)


class TestBuildBehavioralExamples:
    def test_one_example_per_task(self, tmp_path):
        suite = _write_suite(tmp_path)
        examples = build_behavioral_examples(suite)
        assert len(examples) == 3

    def test_stable_ordering_by_task_id(self, tmp_path):
        # Suite file has order b, a, c — examples should sort alphabetically.
        suite = _write_suite(tmp_path)
        examples = build_behavioral_examples(suite)
        ids = [ex.closed_loop_task_id for ex in examples]
        assert ids == ["a_task", "b_task", "c_task"]

    def test_task_value_carries_user_message(self, tmp_path):
        suite = _write_suite(tmp_path)
        examples = build_behavioral_examples(suite)
        by_id = {ex.closed_loop_task_id: ex for ex in examples}
        assert by_id["a_task"].task == "do A"
        assert by_id["b_task"].task == "do B"
        assert by_id["c_task"].task == "do C"

    def test_inputs_include_both_marker_and_task(self, tmp_path):
        # Critical: DSPy passes program(**example.inputs()), so both keys must
        # be in the input-key set or forward() never sees closed_loop_task_id.
        suite = _write_suite(tmp_path)
        examples = build_behavioral_examples(suite)
        inputs = examples[0].inputs()
        assert "task" in inputs
        assert "closed_loop_task_id" in inputs

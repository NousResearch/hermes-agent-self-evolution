"""Tests for the behavioral-example branch in the fitness metrics.

Verifies:
- The branch fires when ``pred._closed_loop_task_id`` is set.
- Score is binary (1.0 pass, 0.0 fail/abstain/miss) — preserves GEPA's
  byte-identity contract automatically (same candidate text → same cache
  → same verdict).
- The branch produces no LM call: the judge is never invoked.
- Feedback string includes the task_id and outcome.
- Both ``make_skill_fitness_metric`` and ``make_tool_fitness_metric``
  route consistently.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import dspy
import pytest

from evolution.core.fitness import (
    FitnessScore,
    _score_behavioral_example,
    make_skill_fitness_metric,
)
from evolution.tools.tool_judge import make_tool_fitness_metric
from evolution.tools.tool_source import ToolEntry, ToolManifest
from evolution.validation.report import TaskResult


def _behavioral_pred(*, task_id: str = "t1", candidate_text: str = "evolved desc"):
    return dspy.Prediction(
        chosen_tool="",
        reasoning="",
        _closed_loop_task_id=task_id,
        _candidate_text=candidate_text,
    )


def _make_manifest(tool_name: str = "patch") -> ToolManifest:
    return ToolManifest(
        tools=(
            ToolEntry(
                name=tool_name,
                description="targeted edits",
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
        ),
    )


class TestScoreBehavioralExampleHelper:
    def test_pass_returns_score_1(self):
        cache = MagicMock()
        cache.get_task_verdict.return_value = TaskResult(
            "t1", True, False, ["write_file"], 1.0, "stub", None
        )
        result = _score_behavioral_example(_behavioral_pred(), cache)
        assert result.score == 1.0
        assert "pass" in result.feedback
        assert "t1" in result.feedback

    def test_fail_returns_score_0(self):
        cache = MagicMock()
        cache.get_task_verdict.return_value = TaskResult(
            "t1", False, False, ["patch"], 1.0, "stub", None
        )
        result = _score_behavioral_example(_behavioral_pred(), cache)
        assert result.score == 0.0
        assert "fail" in result.feedback

    def test_abstain_returns_score_0(self):
        cache = MagicMock()
        cache.get_task_verdict.return_value = TaskResult(
            "t1", False, True, [], 1.0, "stub", "runner timeout"
        )
        result = _score_behavioral_example(_behavioral_pred(), cache)
        assert result.score == 0.0
        assert "abstain" in result.feedback
        assert "runner timeout" in result.feedback

    def test_cache_miss_returns_score_0(self):
        cache = MagicMock()
        cache.get_task_verdict.return_value = None
        result = _score_behavioral_example(_behavioral_pred(), cache)
        assert result.score == 0.0
        assert "no verdict" in result.feedback

    def test_no_cache_returns_score_0(self):
        result = _score_behavioral_example(_behavioral_pred(), None)
        assert result.score == 0.0
        assert "cache unavailable" in result.feedback

    def test_score_byte_identical_across_calls_for_same_candidate(self):
        # The cache is what guarantees determinism — same candidate text
        # → same TaskResult → same score. We just verify the helper itself
        # is a pure function of (verdict, prediction).
        cache = MagicMock()
        cache.get_task_verdict.return_value = TaskResult(
            "t1", True, False, ["write_file"], 1.0, "stub", None
        )
        r1 = _score_behavioral_example(_behavioral_pred(), cache)
        r2 = _score_behavioral_example(_behavioral_pred(), cache)
        assert r1.score == r2.score
        assert r1.feedback == r2.feedback


class TestSkillMetricBehavioralBranch:
    def test_behavioral_branch_short_circuits_judge(self):
        judge = MagicMock()
        cache = MagicMock()
        cache.get_task_verdict.return_value = TaskResult(
            "t1", True, False, ["write_file"], 1.0, "stub", None
        )
        metric = make_skill_fitness_metric(
            judge, "baseline", closed_loop_cache=cache
        )
        result = metric(
            dspy.Example(closed_loop_task_id="t1"),
            _behavioral_pred(),
        )
        assert result.score == 1.0
        # Judge MUST NOT have been called — behavioral path is LM-free.
        judge.score.assert_not_called()

    def test_non_behavioral_example_falls_through_to_judge(self):
        judge = MagicMock()
        judge.score.return_value = FitnessScore(
            correctness=0.5, procedure_following=0.5, conciseness=0.5,
            feedback="judge feedback", profile="balanced",
        )
        metric = make_skill_fitness_metric(judge, "baseline")
        result = metric(
            dspy.Example(task_input="t", expected_behavior="b"),
            dspy.Prediction(output="some output"),
        )
        # Standard v1 path — judge was called.
        judge.score.assert_called_once()
        assert result.score > 0


class TestToolMetricBehavioralBranch:
    def test_behavioral_branch_short_circuits_judge(self):
        judge = MagicMock()
        manifest = _make_manifest()
        cache = MagicMock()
        cache.get_task_verdict.return_value = TaskResult(
            "t1", True, False, ["patch"], 1.0, "stub", None
        )
        metric = make_tool_fitness_metric(
            judge,
            baseline_description="desc",
            manifest=manifest,
            target_tool_name="patch",
            max_growth=0.2,
            closed_loop_cache=cache,
        )
        result = metric(
            dspy.Example(closed_loop_task_id="t1"),
            _behavioral_pred(),
        )
        assert result.score == 1.0
        judge.score.assert_not_called()

    def test_non_behavioral_example_falls_through_to_judge(self):
        judge = MagicMock()
        judge.score.return_value = FitnessScore(
            correctness=0.7, procedure_following=0.7, conciseness=0.7,
            feedback="judge feedback", profile="balanced",
        )
        manifest = _make_manifest()
        metric = make_tool_fitness_metric(
            judge,
            baseline_description="desc",
            manifest=manifest,
            target_tool_name="patch",
            max_growth=0.2,
        )
        result = metric(
            dspy.Example(task_input="t", expected_behavior="patch"),
            dspy.Prediction(chosen_tool="patch", reasoning="r"),
        )
        judge.score.assert_called_once()
        assert result.score > 0

    def test_behavioral_branch_passes_candidate_text_to_cache(self):
        cache = MagicMock()
        cache.get_task_verdict.return_value = None
        manifest = _make_manifest()
        metric = make_tool_fitness_metric(
            judge=MagicMock(),
            baseline_description="desc",
            manifest=manifest,
            target_tool_name="patch",
            max_growth=0.2,
            closed_loop_cache=cache,
        )
        metric(
            dspy.Example(closed_loop_task_id="t1"),
            _behavioral_pred(candidate_text="EVOLVED_TEXT_HERE"),
        )
        cache.get_task_verdict.assert_called_once_with("EVOLVED_TEXT_HERE", "t1")

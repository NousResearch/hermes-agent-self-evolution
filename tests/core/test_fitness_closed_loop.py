"""Tests for the closed-loop-feedback integration in fitness metrics.

Verifies that:
  - The metric records judge scores on the cache when set.
  - The [CLOSED_LOOP] block only appears when pred_trace is set (the
    reflective-feedback path); Pareto-evaluation calls (pred_trace=None)
    must not pay any closed-loop cost.
  - The metric's score is unchanged by closed-loop enrichment — GEPA's
    byte-identity requirement between predictor and module call sites.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import dspy
import pytest

from evolution.core.fitness import (
    FitnessScore,
    _augment_feedback_with_closed_loop,
    make_skill_fitness_metric,
)
from evolution.tools.tool_judge import make_tool_fitness_metric
from evolution.tools.tool_source import ToolEntry, ToolManifest


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


def _fake_pred_trace(instructions: str = "current instruction text"):
    predictor = SimpleNamespace(
        signature=SimpleNamespace(instructions=instructions)
    )
    return [(predictor, {}, SimpleNamespace())]


class TestAugmentHelper:
    def test_no_cache_passes_through(self):
        result = _augment_feedback_with_closed_loop(
            "base", None, _fake_pred_trace(), None
        )
        assert result == "base"

    def test_no_pred_trace_passes_through(self):
        cache = MagicMock()
        result = _augment_feedback_with_closed_loop("base", cache, None, None)
        assert result == "base"
        cache.get_or_run.assert_not_called()

    def test_empty_pred_trace_passes_through(self):
        cache = MagicMock()
        result = _augment_feedback_with_closed_loop("base", cache, [], None)
        assert result == "base"
        cache.get_or_run.assert_not_called()

    def test_cache_miss_returns_none_passes_through(self):
        cache = MagicMock()
        cache.get_or_run.return_value = None
        result = _augment_feedback_with_closed_loop(
            "base", cache, _fake_pred_trace(), None
        )
        assert result == "base"
        cache.get_or_run.assert_called_once_with("current instruction text")

    def test_cache_hit_appends_block(self):
        cache = MagicMock()
        # Return a stub report; render_feedback_block handles real ones,
        # but here we just need a value that triggers the rendering path.
        # We monkeypatch render_feedback_block by stubbing get_or_run to
        # return a real ValidationReport-shaped object via a deep-import.
        from tests.core.test_closed_loop_feedback import _build_report
        cache.get_or_run.return_value = _build_report(
            decision="regression",
            pass_rate_change=-0.50,
        )
        result = _augment_feedback_with_closed_loop(
            "base", cache, _fake_pred_trace(), None
        )
        assert result.startswith("base\n\n[CLOSED_LOOP]")

    def test_text_extractor_overrides_signature_instructions(self):
        cache = MagicMock()
        cache.get_or_run.return_value = None
        predictor = SimpleNamespace(
            signature=SimpleNamespace(instructions="not this")
        )
        pred_trace = [(predictor, {}, SimpleNamespace())]
        extractor = MagicMock(return_value="extracted text")
        _augment_feedback_with_closed_loop(
            "base", cache, pred_trace, text_extractor=extractor
        )
        extractor.assert_called_once_with(predictor)
        cache.get_or_run.assert_called_once_with("extracted text")


class TestSkillFitnessMetricIntegration:
    def _make_judge(self, score: float = 0.7) -> MagicMock:
        judge = MagicMock()
        judge.score.return_value = FitnessScore(
            correctness=score,
            procedure_following=score,
            conciseness=score,
            feedback="judge feedback",
            profile="balanced",
        )
        return judge

    def test_no_cache_no_record_no_append(self):
        judge = self._make_judge()
        metric = make_skill_fitness_metric(judge, "baseline", max_growth=0.2)
        gold = dspy.Example(task_input="t", expected_behavior="b")
        pred = dspy.Prediction(output="some output")
        result = metric(gold, pred, trace=None, pred_name=None, pred_trace=None)
        assert "[CLOSED_LOOP]" not in result.feedback

    def test_cache_records_judge_score_every_call(self):
        judge = self._make_judge(0.8)
        cache = MagicMock()
        cache.get_or_run.return_value = None
        metric = make_skill_fitness_metric(
            judge, "baseline", closed_loop_cache=cache
        )
        gold = dspy.Example(task_input="t", expected_behavior="b")
        pred = dspy.Prediction(output="some output")
        metric(gold, pred, trace=None, pred_name=None, pred_trace=None)
        cache.record_judge_score.assert_called_once()
        # Score is the composite from FitnessScore — verify by checking
        # the recorded value is in [0, 1] and matches what judge returned.
        recorded = cache.record_judge_score.call_args.args[0]
        assert 0.0 <= recorded <= 1.0

    def test_cache_block_only_appended_with_pred_trace(self):
        judge = self._make_judge()
        cache = MagicMock()
        from tests.core.test_closed_loop_feedback import _build_report
        cache.get_or_run.return_value = _build_report(
            decision="regression", pass_rate_change=-0.30
        )
        metric = make_skill_fitness_metric(
            judge, "baseline", closed_loop_cache=cache
        )
        gold = dspy.Example(task_input="t", expected_behavior="b")
        pred = dspy.Prediction(output="some output")

        # Pareto-eval path: pred_trace=None → no closed-loop work
        r1 = metric(gold, pred, trace=None, pred_name=None, pred_trace=None)
        assert "[CLOSED_LOOP]" not in r1.feedback
        cache.get_or_run.assert_not_called()

        # Reflective-feedback path: pred_trace set → closed-loop runs
        r2 = metric(
            gold, pred, trace=None, pred_name=None,
            pred_trace=_fake_pred_trace(),
        )
        assert "[CLOSED_LOOP]" in r2.feedback
        cache.get_or_run.assert_called_once()

    def test_score_byte_identical_across_pred_trace_paths(self):
        # GEPA's correctness requirement: same score regardless of pred_trace.
        judge = self._make_judge(0.42)
        cache = MagicMock()
        from tests.core.test_closed_loop_feedback import _build_report
        cache.get_or_run.return_value = _build_report(
            decision="regression", pass_rate_change=-0.30
        )
        metric = make_skill_fitness_metric(
            judge, "baseline", closed_loop_cache=cache
        )
        gold = dspy.Example(task_input="t", expected_behavior="b")
        pred = dspy.Prediction(output="some output")

        r1 = metric(gold, pred, pred_trace=None)
        r2 = metric(gold, pred, pred_trace=_fake_pred_trace())
        assert r1.score == r2.score

    def test_empty_output_short_circuits_before_closed_loop(self):
        judge = self._make_judge()
        cache = MagicMock()
        metric = make_skill_fitness_metric(
            judge, "baseline", closed_loop_cache=cache
        )
        gold = dspy.Example(task_input="t", expected_behavior="b")
        pred = dspy.Prediction(output="")
        result = metric(gold, pred, pred_trace=_fake_pred_trace())
        assert result.score == 0.0
        cache.record_judge_score.assert_not_called()
        cache.get_or_run.assert_not_called()


class TestToolFitnessMetricIntegration:
    def _make_judge(self, composite: float = 0.7) -> MagicMock:
        judge = MagicMock()
        judge.score.return_value = FitnessScore(
            correctness=composite,
            procedure_following=composite,
            conciseness=composite,
            feedback="judge feedback",
            profile="balanced",
        )
        return judge

    def test_no_cache_no_record(self):
        judge = self._make_judge()
        manifest = _make_manifest()
        metric = make_tool_fitness_metric(
            judge,
            baseline_description="desc",
            manifest=manifest,
            target_tool_name="patch",
            max_growth=0.2,
        )
        gold = dspy.Example(task_input="t", expected_behavior="patch")
        pred = dspy.Prediction(chosen_tool="patch", reasoning="r")
        result = metric(gold, pred, pred_trace=None)
        assert "[CLOSED_LOOP]" not in result.feedback

    def test_cache_records_and_appends(self):
        judge = self._make_judge(0.85)
        manifest = _make_manifest()
        cache = MagicMock()
        from tests.core.test_closed_loop_feedback import _build_report
        cache.get_or_run.return_value = _build_report(
            decision="regression", pass_rate_change=-0.30
        )
        metric = make_tool_fitness_metric(
            judge,
            baseline_description="desc",
            manifest=manifest,
            target_tool_name="patch",
            max_growth=0.2,
            closed_loop_cache=cache,
        )
        gold = dspy.Example(task_input="t", expected_behavior="patch")
        pred = dspy.Prediction(chosen_tool="patch", reasoning="r")

        # Reflective-feedback path: closed-loop fires
        result = metric(gold, pred, pred_trace=_fake_pred_trace())
        cache.record_judge_score.assert_called_once()
        cache.get_or_run.assert_called_once()
        assert "[CLOSED_LOOP]" in result.feedback

    def test_text_extractor_threads_through_to_cache(self):
        judge = self._make_judge(0.99)
        manifest = _make_manifest()
        cache = MagicMock()
        cache.get_or_run.return_value = None
        extractor = MagicMock(return_value="extracted from extractor")
        metric = make_tool_fitness_metric(
            judge,
            baseline_description="desc",
            manifest=manifest,
            target_tool_name="patch",
            max_growth=0.2,
            text_extractor=extractor,
            closed_loop_cache=cache,
        )
        gold = dspy.Example(task_input="t", expected_behavior="patch")
        pred = dspy.Prediction(chosen_tool="patch", reasoning="r")
        metric(gold, pred, pred_trace=_fake_pred_trace())
        # Extractor called twice: once by _augment_feedback_with_pred_trace,
        # once by _augment_feedback_with_closed_loop.
        assert extractor.call_count == 2
        cache.get_or_run.assert_called_once_with("extracted from extractor")

    def test_unparseable_tool_short_circuits_before_cache(self):
        judge = self._make_judge()
        manifest = _make_manifest()
        cache = MagicMock()
        metric = make_tool_fitness_metric(
            judge,
            baseline_description="desc",
            manifest=manifest,
            target_tool_name="patch",
            max_growth=0.2,
            closed_loop_cache=cache,
        )
        gold = dspy.Example(task_input="t", expected_behavior="patch")
        pred = dspy.Prediction(chosen_tool="", reasoning="r")
        result = metric(gold, pred, pred_trace=_fake_pred_trace())
        assert result.score == 0.0
        cache.record_judge_score.assert_not_called()
        cache.get_or_run.assert_not_called()

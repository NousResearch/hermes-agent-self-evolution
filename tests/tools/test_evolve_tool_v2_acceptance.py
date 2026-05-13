"""Integration test: closed-loop trainset mode produces a non-baseline acceptance signal.

The v2 mechanism: behavioral examples in GEPA's trainset score float(passed)
from the closed-loop validator; that score contributes to GEPA's
``sum(minibatch_scores)`` acceptance rule (gepa/core/engine.py:491-493).
This test bypasses the full GEPA loop (no merge proposer, no Pareto sampler,
no real reflection LM) and exercises the load-bearing piece: invoke
``DspyAdapter.evaluate()`` twice — once with the baseline candidate, once
with an evolved candidate — against a hand-built minibatch that mixes
saturated-judge examples with behavioral examples. Behavioral verdicts are
served by a fake cache that flips a single task's outcome between the two
candidates.

The math GEPA would do (engine.py:491-493): ``sum_after > sum_before`` →
accept. This test asserts the per-example scores literally and the sum
inequality.

Zero LM cost: the selector LM is replaced by ``dspy.utils.DummyLM`` and
the judge by a ``MagicMock`` that always returns 1.0.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

import dspy
import pytest
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from dspy.utils.dummies import DummyLM

from evolution.core.fitness import FitnessScore
from evolution.tools.tool_judge import make_tool_fitness_metric
from evolution.tools.tool_module import (
    ToolModule,
    _render_manifest_for_prompt,
)
from evolution.tools.tool_source import ToolEntry, ToolManifest
from evolution.validation.report import TaskResult


BASELINE_DESC = "Write the entire file contents."
EVOLVED_DESC = (
    "Write the entire file contents. Always pass an absolute path; "
    "create parent directories implicitly."
)


# ---- Fakes ----


@dataclass
class _FakeCache:
    """Stand-in for ClosedLoopFeedbackCache. Verdicts come from a dict the
    test seeds — same (candidate_text, task_id) → same TaskResult, so the
    byte-identity contract from fitness.py:144-146 is automatic."""

    verdicts: dict[tuple[str, str], bool]
    judge_scores: list[float] = field(default_factory=list)

    def get_task_verdict(self, candidate_text: str, task_id: str) -> Optional[TaskResult]:
        if (candidate_text, task_id) not in self.verdicts:
            return None
        passed = self.verdicts[(candidate_text, task_id)]
        return TaskResult(
            task_id=task_id,
            passed=passed,
            abstained=False,
            tool_calls_seq=["write_file"] if passed else [],
            duration_seconds=0.0,
            model_name="stub",
            error=None,
        )

    def get_or_run(self, candidate_text: str):
        # The v1-style [CLOSED_LOOP] feedback augmentation short-circuits
        # when this returns None; that's the desired behavior here — we're
        # testing the score channel, not the feedback channel.
        return None

    def record_judge_score(self, score: float) -> None:
        self.judge_scores.append(score)


# ---- Fixtures ----


def _manifest() -> ToolManifest:
    return ToolManifest(
        tools=(
            ToolEntry(
                name="write_file",
                description=BASELINE_DESC,
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
            ToolEntry(
                name="read_file",
                description="Read a file's contents.",
                input_schema={"type": "object", "properties": {}, "required": []},
            ),
        ),
    )


def _module(desc: str) -> ToolModule:
    return ToolModule(
        target_tool_name="write_file",
        manifest=_manifest(),
        target_description=desc,
    )


def _saturated_judge() -> MagicMock:
    j = MagicMock()
    j.score.return_value = FitnessScore(
        correctness=1.0,
        procedure_following=1.0,
        conciseness=1.0,
        length_penalty=0.0,
        feedback="judge stub",
        profile="balanced",
    )
    return j


def _build_adapter(student: ToolModule, metric) -> DspyAdapter:
    return DspyAdapter(
        student_module=student,
        metric_fn=metric,
        # Feedback map is keyed by predictor name; the no-op feedback fn is
        # only invoked if make_reflective_dataset runs, which only runs
        # when capture_traces=True AND the example has a matching trace.
        feedback_map={"selector.predict": lambda **kw: {"score": 0.0, "feedback": ""}},
        failure_score=0.0,
        num_threads=1,
        add_format_failure_as_feedback=False,
        rng=random.Random(0),
        reflection_lm=None,
        custom_instruction_proposer=None,
        warn_on_score_mismatch=False,
    )


def _judge_example(task: str) -> dspy.Example:
    # Two input keys: `task` (consumed by ToolModule.forward) and
    # `task_input` (read by tool_judge.metric for the judge prompt).
    return dspy.Example(
        task=task,
        task_input=task,
        expected_behavior="write_file",
    ).with_inputs("task")


def _behavioral_example(task_id: str) -> dspy.Example:
    return dspy.Example(
        task="(behavioral placeholder)",
        closed_loop_task_id=task_id,
    ).with_inputs("task", "closed_loop_task_id")


def _minibatch() -> list[dspy.Example]:
    return [
        _judge_example("write a config file"),
        _judge_example("save the report"),
        _behavioral_example("t_loss"),
        _behavioral_example("t_neutral"),
    ]


def _candidate_dict_for(desc: str) -> dict[str, str]:
    """Build the candidate dict GEPA would pass to adapter.evaluate.

    Key matches predictor name from ToolModule.named_predictors() →
    `selector.predict`. Value is the rendered manifest with sentinels
    around the target tool's description, which is exactly what
    ToolModule.__init__ installs and what description_text reads back.
    """
    return {
        "selector.predict": _render_manifest_for_prompt(
            _manifest(), "write_file", desc
        )
    }


# ---- Warm-up unit test ----


class TestMinibatchSumBreaksJudgeTie:
    """Direct metric calls (no adapter) — verifies the per-example scores
    are what we expect and the minibatch isn't all-perfect, which is the
    condition that lets GEPA's reflection step fire instead of skipping.
    See gepa/strategies/reflective_mutation.py:204."""

    def test_baseline_minibatch_has_sub_perfect_score(self):
        cache = _FakeCache(
            verdicts={
                (BASELINE_DESC, "t_loss"): False,
                (BASELINE_DESC, "t_neutral"): True,
            }
        )
        metric = make_tool_fitness_metric(
            judge=_saturated_judge(),
            baseline_description=BASELINE_DESC,
            manifest=_manifest(),
            target_tool_name="write_file",
            max_growth=0.5,
            closed_loop_cache=cache,
        )

        # Hand-built (example, prediction) pairs — the judge examples
        # bypass the LM by feeding pre-built Predictions directly.
        judge_pred = dspy.Prediction(chosen_tool="write_file", reasoning="r")
        judge_score = metric(_judge_example("x"), judge_pred).score

        behavioral_pred_loss = dspy.Prediction(
            chosen_tool="",
            reasoning="",
            _closed_loop_task_id="t_loss",
            _candidate_text=BASELINE_DESC,
        )
        behav_loss_score = metric(_behavioral_example("t_loss"), behavioral_pred_loss).score

        behavioral_pred_neutral = dspy.Prediction(
            chosen_tool="",
            reasoning="",
            _closed_loop_task_id="t_neutral",
            _candidate_text=BASELINE_DESC,
        )
        behav_neutral_score = metric(
            _behavioral_example("t_neutral"), behavioral_pred_neutral
        ).score

        scores = [judge_score, behav_loss_score, behav_neutral_score]
        assert scores == [1.0, 0.0, 1.0]
        # GEPA's reflective_mutation.py:204 condition would NOT skip:
        assert not all(s >= 1.0 for s in scores)


# ---- Primary integration test ----


class TestV2BehavioralWinBreaksAcceptance:
    """End-to-end via DspyAdapter.evaluate: a candidate that flips one
    behavioral example from fail to pass produces a higher minibatch sum
    than the baseline. That is precisely the inequality GEPA's acceptance
    rule (engine.py:491-493) checks before accepting a new candidate."""

    def _run(self, capture_traces: bool):
        cache = _FakeCache(
            verdicts={
                (BASELINE_DESC, "t_loss"): False,
                (EVOLVED_DESC, "t_loss"): True,
                (BASELINE_DESC, "t_neutral"): True,
                (EVOLVED_DESC, "t_neutral"): True,
            }
        )
        judge = _saturated_judge()
        metric = make_tool_fitness_metric(
            judge=judge,
            baseline_description=BASELINE_DESC,
            manifest=_manifest(),
            target_tool_name="write_file",
            max_growth=0.5,
            closed_loop_cache=cache,
        )
        adapter = _build_adapter(_module(BASELINE_DESC), metric)

        baseline_candidate = _candidate_dict_for(BASELINE_DESC)
        evolved_candidate = _candidate_dict_for(EVOLVED_DESC)
        batch = _minibatch()

        # The selector LM is the only LM the test would otherwise call.
        # Provide enough deterministic responses to cover both adapter.evaluate
        # invocations × 2 judge examples × possible CoT retries.
        selector_responses = [
            {"reasoning": "stub", "chosen_tool": "write_file"} for _ in range(64)
        ]
        with dspy.context(lm=DummyLM(selector_responses)):
            eval_before = adapter.evaluate(
                batch, baseline_candidate, capture_traces=capture_traces
            )
            eval_after = adapter.evaluate(
                batch, evolved_candidate, capture_traces=capture_traces
            )

        return eval_before, eval_after, judge, cache

    def test_acceptance_math_capture_traces_false(self):
        eval_before, eval_after, judge, cache = self._run(capture_traces=False)

        # Per-example scores in batch order [judge, judge, behavioral, behavioral]
        assert eval_before.scores == [1.0, 1.0, 0.0, 1.0]
        assert eval_after.scores == [1.0, 1.0, 1.0, 1.0]

        # The acceptance arithmetic GEPA does:
        assert sum(eval_after.scores) > sum(eval_before.scores)
        assert sum(eval_after.scores) - sum(eval_before.scores) == pytest.approx(1.0)

        # Behavioral examples never reached the judge — only the 2 judge
        # examples × 2 candidate evals = 4 judge calls.
        assert judge.score.call_count == 4

        # Cache recorded a judge score on each judge metric call.
        assert len(cache.judge_scores) == 4

    def test_acceptance_math_capture_traces_true(self):
        # The reflective-feedback path (capture_traces=True) goes through
        # bootstrap_trace_data, but for behavioral examples the trace is
        # empty (selector LM never invoked). Scores must still be identical
        # to the no-trace path — fitness.py:144 byte-identity contract.
        eval_before, eval_after, _, _ = self._run(capture_traces=True)

        assert eval_before.scores == [1.0, 1.0, 0.0, 1.0]
        assert eval_after.scores == [1.0, 1.0, 1.0, 1.0]
        assert sum(eval_after.scores) > sum(eval_before.scores)


class TestScoreByteIdentityAcrossPredTracePaths:
    """The same (candidate, behavioral example) pair must produce identical
    scores whether GEPA captured traces or not — GEPA warns and overrides
    on byte divergence between predictor-level and module-level call sites
    (fitness.py:140-146)."""

    def test_behavioral_score_identical_with_and_without_traces(self):
        cache = _FakeCache(
            verdicts={
                (EVOLVED_DESC, "t_loss"): True,
            }
        )
        metric = make_tool_fitness_metric(
            judge=_saturated_judge(),
            baseline_description=BASELINE_DESC,
            manifest=_manifest(),
            target_tool_name="write_file",
            max_growth=0.5,
            closed_loop_cache=cache,
        )
        adapter = _build_adapter(_module(BASELINE_DESC), metric)
        candidate = _candidate_dict_for(EVOLVED_DESC)
        batch = [_behavioral_example("t_loss")]

        with dspy.context(lm=DummyLM([{"reasoning": "x", "chosen_tool": "write_file"}] * 8)):
            scores_no_traces = adapter.evaluate(
                batch, candidate, capture_traces=False
            ).scores
            scores_with_traces = adapter.evaluate(
                batch, candidate, capture_traces=True
            ).scores

        assert scores_no_traces == scores_with_traces == [1.0]

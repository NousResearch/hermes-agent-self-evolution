"""Tests for the skill fitness metric."""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import dspy
import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.fitness import (
    FitnessScore,
    LLMJudge,
    _clamp_to_unit,
    make_skill_fitness_metric,
)


def _stub_judge(score: float = 0.7, feedback: str = "explanatory feedback") -> Mock:
    """Build a Mock LLMJudge whose .score() returns a deterministic FitnessScore."""
    judge = Mock(spec=LLMJudge)
    judge.score.return_value = FitnessScore(
        correctness=score,
        procedure_following=score,
        conciseness=score,
        length_penalty=0.0,
        feedback=feedback,
    )
    return judge


class TestMetricArity:
    """Lock the metric signature against DSPy GEPA's runtime arity check.

    DSPy 3.2.0's dspy.GEPA.__init__ runs:
        inspect.signature(metric).bind(None, None, None, None, None)
    and raises TypeError if the metric can't accept five positional args.
    Required signature: (gold, pred, trace=None, pred_name=None, pred_trace=None).
    """

    def test_factory_metric_accepts_five_positional_args(self):
        metric = make_skill_fitness_metric(_stub_judge())
        inspect.signature(metric).bind(None, None, None, None, None)

    def test_factory_metric_accepts_two_positional_args(self):
        metric = make_skill_fitness_metric(_stub_judge())
        inspect.signature(metric).bind(None, None)


class TestFactoryMetricBehavior:
    def test_returns_prediction_with_score_and_feedback(self):
        judge = _stub_judge(score=0.6, feedback="agent missed step 3")
        metric = make_skill_fitness_metric(judge)

        example = SimpleNamespace(task_input="t", expected_behavior="b")
        prediction = SimpleNamespace(output="some non-empty agent response")

        result = metric(example, prediction)

        assert isinstance(result, dspy.Prediction)
        # FitnessScore.composite for all-0.6 inputs: 0.5*0.6 + 0.3*0.6 + 0.2*0.6 = 0.6
        assert result.score == 0.6
        assert result.feedback == "agent missed step 3"
        # Judge was actually invoked with the right inputs (no skill_text).
        judge.score.assert_called_once_with(
            task_input="t",
            expected_behavior="b",
            agent_output="some non-empty agent response",
        )

    def test_empty_output_returns_zero_without_calling_judge(self, caplog):
        judge = _stub_judge()
        metric = make_skill_fitness_metric(judge)

        example = SimpleNamespace(task_input="t", expected_behavior="b")
        prediction = SimpleNamespace(output="   ")  # whitespace-only

        with caplog.at_level("WARNING", logger="evolution.core.fitness"):
            result = metric(example, prediction)

        assert isinstance(result, dspy.Prediction)
        assert result.score == 0.0
        assert "empty" in result.feedback.lower()
        judge.score.assert_not_called()
        assert any("empty agent output" in rec.message for rec in caplog.records)


class TestJudgeSignatureFieldTypes:
    """LLM scores arrive as strings; declaring float OutputFields used to
    trigger typeguard warnings on every call. Lock the declared types as
    str so the warning suppression isn't load-bearing for this surface."""

    def test_score_outputs_are_str(self):
        sig = LLMJudge.JudgeSignature
        assert sig.output_fields["correctness"].annotation is str
        assert sig.output_fields["procedure_following"].annotation is str
        assert sig.output_fields["conciseness"].annotation is str

    def test_feedback_output_is_str(self):
        sig = LLMJudge.JudgeSignature
        assert sig.output_fields["feedback"].annotation is str

    def test_skill_text_input_was_dropped(self):
        # The judge no longer reads the skill body — that risked rewarding
        # outputs that quote the skill back instead of solving the task.
        sig = LLMJudge.JudgeSignature
        assert "skill_text" not in sig.input_fields


class TestPredTraceFeedback:
    """Lock the contract that the metric uses pred_trace to enrich feedback
    for GEPA's reflective loop, while keeping `score` byte-identical across
    the two GEPA call sites (predictor-level vs Evaluate-over-valset).
    """

    def _stub_pred_trace(self, instructions: str, reasoning: str = "step 1; step 2"):
        signature = SimpleNamespace(instructions=instructions)
        predictor = SimpleNamespace(signature=signature)
        output = SimpleNamespace(reasoning=reasoning, output="agent answer")
        return [(predictor, {"task_input": "t"}, output)]

    def test_budget_line_appears_when_pred_trace_set(self):
        judge = _stub_judge(score=0.6, feedback="judge says X")
        # Baseline 100 chars; +20% budget → target 120; current 250 → +150%.
        metric = make_skill_fitness_metric(
            judge, baseline_skill_text="x" * 100, max_growth=0.2
        )

        example = SimpleNamespace(task_input="t", expected_behavior="b")
        prediction = SimpleNamespace(output="agent answer")
        pred_trace = self._stub_pred_trace(instructions="y" * 250)

        result = metric(example, prediction, pred_trace=pred_trace)

        assert "[BUDGET]" in result.feedback
        assert "250 chars" in result.feedback
        assert "100 chars" in result.feedback
        assert "+150.0%" in result.feedback
        assert "120 chars" in result.feedback  # target
        # Original judge feedback preserved.
        assert "judge says X" in result.feedback

    def test_reasoning_line_appears_when_pred_trace_set(self):
        judge = _stub_judge(feedback="judge says X")
        metric = make_skill_fitness_metric(
            judge, baseline_skill_text="x" * 100, max_growth=0.2
        )

        prediction = SimpleNamespace(output="agent answer")
        pred_trace = self._stub_pred_trace(
            instructions="y" * 110,
            reasoning="thought process: I considered A, then B, then C.",
        )

        result = metric(
            SimpleNamespace(task_input="t", expected_behavior="b"),
            prediction,
            pred_trace=pred_trace,
        )

        assert "[REASONING]" in result.feedback
        assert "I considered A, then B, then C." in result.feedback

    def test_reasoning_truncated_at_500_chars(self):
        judge = _stub_judge()
        metric = make_skill_fitness_metric(judge, baseline_skill_text="x")
        long_reasoning = "Z" * 1000
        pred_trace = self._stub_pred_trace(instructions="y", reasoning=long_reasoning)

        result = metric(
            SimpleNamespace(task_input="t", expected_behavior="b"),
            SimpleNamespace(output="answer"),
            pred_trace=pred_trace,
        )

        assert "Z" * 500 in result.feedback
        assert "Z" * 501 not in result.feedback
        assert "…" in result.feedback

    def test_no_pred_trace_means_plain_judge_feedback(self):
        judge = _stub_judge(feedback="plain feedback")
        metric = make_skill_fitness_metric(judge, baseline_skill_text="x" * 100)

        result = metric(
            SimpleNamespace(task_input="t", expected_behavior="b"),
            SimpleNamespace(output="answer"),
            # pred_trace omitted → defaults to None (Evaluate-over-valset path).
        )

        assert result.feedback == "plain feedback"
        assert "[BUDGET]" not in result.feedback
        assert "[REASONING]" not in result.feedback

    def test_score_identical_across_pred_trace_call_sites(self):
        """GEPA warns and overrides if score differs between the predictor-level
        call (pred_trace set) and Evaluate-over-valset call (pred_trace None).
        Lock the contract."""
        judge = _stub_judge(score=0.42, feedback="judge")
        metric = make_skill_fitness_metric(judge, baseline_skill_text="x" * 100)

        example = SimpleNamespace(task_input="t", expected_behavior="b")
        prediction = SimpleNamespace(output="answer")

        # Module-level call (no pred_trace).
        score_no_trace = metric(example, prediction).score
        # Predictor-level call (pred_trace set).
        score_with_trace = metric(
            example,
            prediction,
            pred_trace=self._stub_pred_trace(instructions="y" * 200),
        ).score

        assert score_no_trace == score_with_trace

    def test_empty_baseline_skips_budget_line(self):
        # If the caller didn't pass a baseline, we can't compute growth %;
        # skip the BUDGET line rather than emit nonsense.
        judge = _stub_judge(feedback="judge")
        metric = make_skill_fitness_metric(judge, baseline_skill_text="")
        pred_trace = self._stub_pred_trace(instructions="y" * 200, reasoning="")

        result = metric(
            SimpleNamespace(task_input="t", expected_behavior="b"),
            SimpleNamespace(output="answer"),
            pred_trace=pred_trace,
        )

        assert "[BUDGET]" not in result.feedback


class TestFitnessProfile:
    """Profile-driven composite weighting for the LLM judge's score.

    'balanced' preserves the legacy (0.5/0.3/0.2) split. 'compression'
    upweights conciseness; 'growth' drops it so the optimizer doesn't
    punish necessary additions.
    """

    def test_balanced_profile_matches_legacy_formula(self):
        score = FitnessScore(
            correctness=1.0,
            procedure_following=1.0,
            conciseness=1.0,
            profile="balanced",
        )
        assert score.composite == pytest.approx(0.5 + 0.3 + 0.2)

    def test_compression_profile_upweights_conciseness(self):
        # All-1.0 still sums to 1.0 (weights sum to 1).
        all_ones = FitnessScore(
            correctness=1.0,
            procedure_following=1.0,
            conciseness=1.0,
            profile="compression",
        )
        assert all_ones.composite == pytest.approx(0.4 + 0.2 + 0.4)
        # Per-dimension probe: correctness=1.0 alone returns the
        # compression weight (0.4), not the balanced weight (0.5).
        only_correctness = FitnessScore(
            correctness=1.0,
            procedure_following=0.0,
            conciseness=0.0,
            profile="compression",
        )
        assert only_correctness.composite == pytest.approx(0.4)

    def test_growth_profile_drops_conciseness(self):
        score = FitnessScore(
            correctness=0.0,
            procedure_following=0.0,
            conciseness=1.0,
            profile="growth",
        )
        assert score.composite == 0.0

    def test_invalid_profile_raises(self):
        config = EvolutionConfig(fitness_profile="bogus")
        with pytest.raises(ValueError):
            LLMJudge(config)

    def test_default_profile_is_balanced(self):
        assert FitnessScore().profile == "balanced"


class TestClampToUnit:
    def test_parses_valid_floats(self):
        assert _clamp_to_unit("0.7") == 0.7
        assert _clamp_to_unit("0.0") == 0.0
        assert _clamp_to_unit("1.0") == 1.0

    def test_clamps_above_one(self):
        assert _clamp_to_unit("1.5") == 1.0
        assert _clamp_to_unit("99") == 1.0

    def test_clamps_below_zero(self):
        assert _clamp_to_unit("-0.3") == 0.0

    def test_handles_whitespace(self):
        assert _clamp_to_unit("  0.42  ") == 0.42

    def test_neutral_fallback_on_garbage(self):
        # LLMs occasionally emit explanations instead of bare floats.
        assert _clamp_to_unit("not a number") == 0.5
        assert _clamp_to_unit("") == 0.5
        assert _clamp_to_unit(None) == 0.5

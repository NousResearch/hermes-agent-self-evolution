"""Tests for the fitness metric.

Two audited defects lived in this path:

* the judge was handed the *baseline* skill text while scoring every
  candidate, so GEPA's reflective feedback described an artifact that was not
  under evaluation;
* the search used the judge composite while the reported holdout delta came
  from keyword overlap, so the headline number was not the quantity anyone
  optimized.

Both are pinned here.
"""

from __future__ import annotations

import dspy
import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.fitness import (
    FitnessScore,
    LLMJudge,
    _parse_score,
    candidate_text,
    heuristic_quality,
    make_fitness_metric,
)
from evolution.core.objectives import ObjectiveWeights


class RecordingJudge:
    """Stands in for the LLM judge and remembers what it was shown."""

    def __init__(self, quality=0.9):
        self.quality = quality
        self.seen: list[dict] = []

    def score(self, task_input, expected_behavior, agent_output, skill_text):
        self.seen.append(
            {
                "task_input": task_input,
                "agent_output": agent_output,
                "skill_text": skill_text,
            }
        )
        return FitnessScore(
            correctness=self.quality,
            procedure_following=self.quality,
            conciseness=self.quality,
            feedback="tighten section 2",
        )


class BrokenJudge:
    def score(self, **kwargs):
        raise RuntimeError("judge API down")


def example(task="do the thing", expected="mention rollback"):
    return dspy.Example(task_input=task, expected_behavior=expected)


def prediction(output="an answer", skill_text=None):
    kwargs = {"output": output}
    if skill_text is not None:
        kwargs["skill_text"] = skill_text
    return dspy.Prediction(**kwargs)


def config(**kw):
    return EvolutionConfig(hermes_agent_path=None, **kw)


class TestFitnessScore:
    def test_composite_is_weighted(self):
        score = FitnessScore(correctness=1.0, procedure_following=0.0, conciseness=0.0)
        assert score.composite == pytest.approx(0.5)

    def test_composite_is_clamped(self):
        assert FitnessScore(correctness=5, procedure_following=5, conciseness=5).composite == 1.0


class TestCandidateText:
    def test_reads_the_text_attached_by_the_module(self):
        assert candidate_text(prediction(skill_text="CANDIDATE")) == "CANDIDATE"

    def test_falls_back_when_absent(self):
        assert candidate_text(prediction(), fallback="BASE") == "BASE"

    def test_blank_text_falls_back(self):
        assert candidate_text(prediction(skill_text="   "), fallback="BASE") == "BASE"


class TestJudgeSeesTheCandidate:
    def test_the_judge_is_shown_the_candidate_not_the_baseline(self):
        judge = RecordingJudge()
        metric = make_fitness_metric(
            config(), baseline_text="BASELINE TEXT", size_budget=10_000, judge=judge
        )

        metric(example(), prediction(skill_text="CANDIDATE TEXT"))

        assert judge.seen[0]["skill_text"] == "CANDIDATE TEXT"

    def test_each_candidate_is_judged_on_its_own_text(self):
        judge = RecordingJudge()
        metric = make_fitness_metric(config(), baseline_text="BASE", size_budget=10_000, judge=judge)

        metric(example(), prediction(skill_text="VARIANT A"))
        metric(example(), prediction(skill_text="VARIANT B"))

        assert [s["skill_text"] for s in judge.seen] == ["VARIANT A", "VARIANT B"]


class TestSizePressureReachesTheOptimizer:
    def test_a_bloated_candidate_scores_lower(self):
        judge = RecordingJudge(quality=0.9)
        metric = make_fitness_metric(
            config(), baseline_text="x" * 10_000, size_budget=12_000, judge=judge
        )

        tight = metric(example(), prediction(skill_text="x" * 10_000))
        bloated = metric(example(), prediction(skill_text="x" * 20_000))

        assert bloated.score < tight.score

    def test_the_feedback_explains_the_size_cost(self):
        metric = make_fitness_metric(
            config(), baseline_text="x" * 10_000, size_budget=12_000, judge=RecordingJudge()
        )
        result = metric(example(), prediction(skill_text="x" * 20_000))

        assert "SIZE:" in result.feedback
        assert "budget" in result.feedback
        # The optimizer needs to be told what to do about it, not just scolded.
        assert "consolidat" in result.feedback or "Cut" in result.feedback

    def test_no_size_note_when_the_candidate_is_comfortably_small(self):
        metric = make_fitness_metric(
            config(), baseline_text="x" * 1_000, size_budget=100_000, judge=RecordingJudge()
        )
        assert "SIZE:" not in metric(example(), prediction(skill_text="x" * 1_000)).feedback

    def test_size_weight_zero_removes_the_pressure(self):
        metric = make_fitness_metric(
            config(),
            baseline_text="x" * 10_000,
            size_budget=12_000,
            weights=ObjectiveWeights(quality=1.0, size=0.0),
            judge=RecordingJudge(quality=0.8),
        )
        assert metric(example(), prediction(skill_text="x" * 40_000)).score == pytest.approx(0.8)


class TestMetricContract:
    def test_returns_a_prediction_with_score_and_feedback(self):
        metric = make_fitness_metric(config(), "base", 10_000, judge=RecordingJudge())
        result = metric(example(), prediction())
        assert hasattr(result, "score") and hasattr(result, "feedback")
        assert 0.0 <= result.score <= 1.0

    def test_accepts_gepas_five_argument_signature(self):
        metric = make_fitness_metric(config(), "base", 10_000, judge=RecordingJudge())
        result = metric(example(), prediction(), None, "pred_name", None)
        assert result.score > 0

    def test_vectors_are_streamed_to_the_sink(self):
        seen = []
        metric = make_fitness_metric(
            config(), "base", 10_000, judge=RecordingJudge(), on_vector=seen.append
        )
        metric(example(), prediction(skill_text="abc"))
        assert len(seen) == 1
        assert seen[0].size_chars == 3


class TestJudgeFailureFallback:
    def test_a_judge_outage_does_not_kill_the_run(self):
        metric = make_fitness_metric(config(), "base", 10_000, judge=BrokenJudge())
        result = metric(example(expected="mention rollback"), prediction(output="mention rollback"))
        assert 0.0 <= result.score <= 1.0

    def test_the_fallback_declares_itself_low_confidence(self):
        metric = make_fitness_metric(config(), "base", 10_000, judge=BrokenJudge())
        feedback = metric(example(), prediction()).feedback
        assert "Judge unavailable" in feedback
        assert "low-confidence" in feedback


class TestHeuristicQuality:
    def test_empty_output_scores_zero(self):
        assert heuristic_quality(example(), prediction(output="")) == 0.0

    def test_full_overlap_scores_high(self):
        score = heuristic_quality(example(expected="alpha beta"), prediction(output="alpha beta"))
        assert score == pytest.approx(1.0)

    def test_no_expectation_is_neutral(self):
        assert heuristic_quality(example(expected=""), prediction(output="x")) == 0.5


class TestParseScore:
    @pytest.mark.parametrize(
        "raw,expected",
        [(0.5, 0.5), ("0.75", 0.75), (" 1 ", 1.0), (5, 1.0), (-2, 0.0)],
    )
    def test_parses_and_clamps(self, raw, expected):
        assert _parse_score(raw) == expected

    def test_unparseable_is_neutral_not_zero(self):
        """Zero would punish a candidate for the judge's formatting slip."""
        assert _parse_score("very good") == 0.5

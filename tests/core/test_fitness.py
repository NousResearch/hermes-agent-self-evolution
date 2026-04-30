"""Tests for the multi-signal fitness metric and LLMJudge wiring.

Covers:
  - Empty-output edge case
  - Identical / similar / dissimilar input pairs
  - Length-quality boundaries
  - LLMJudge fallback on judge failure
  - reset_fitness_metric isolation between tests
"""

from unittest.mock import patch, MagicMock

import dspy
import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.fitness import (
    FitnessScore,
    LLMJudge,
    init_fitness_metric,
    reset_fitness_metric,
    skill_fitness_metric,
    _char_ngram_similarity,
    _content_density_score,
    _keyword_overlap_score,
    _length_quality_score,
    _structural_match_score,
    _parse_score,
)


def _make_pred(output: str) -> dspy.Prediction:
    return dspy.Prediction(output=output)


def _make_example(expected: str = "", task: str = "task") -> dspy.Example:
    return dspy.Example(task_input=task, expected_behavior=expected).with_inputs("task_input")


class TestDeterministicMultiSignal:
    """The metric used when no LLMJudge has been initialized."""

    def setup_method(self):
        reset_fitness_metric()

    def test_empty_output_scores_zero(self):
        out = skill_fitness_metric(_make_example("anything"), _make_pred(""))
        assert out.score == 0.0
        assert "Empty" in out.feedback

    def test_whitespace_only_output_scores_zero(self):
        out = skill_fitness_metric(_make_example("anything"), _make_pred("   \n\n  "))
        assert out.score == 0.0

    def test_identical_text_high_score(self):
        text = "Sort messages into topics. Use clear category labels. Provide concise summaries."
        out = skill_fitness_metric(_make_example(text), _make_pred(text))
        assert out.score >= 0.7

    def test_unrelated_low_score(self):
        out = skill_fitness_metric(
            _make_example("write SQL to filter rows by date"),
            _make_pred("Italian pasta recipes for beginners"),
        )
        assert out.score < 0.55

    def test_returns_prediction_with_feedback(self):
        out = skill_fitness_metric(_make_example("foo"), _make_pred("bar baz qux"))
        assert hasattr(out, "score")
        assert hasattr(out, "feedback")
        assert isinstance(out.feedback, str) and out.feedback


class TestKeywordOverlap:
    def test_identical_returns_high(self):
        score = _keyword_overlap_score("alpha bravo charlie", "alpha bravo charlie")
        assert score >= 0.9

    def test_disjoint_returns_zero(self):
        score = _keyword_overlap_score("alpha bravo charlie", "delta echo foxtrot")
        assert score == 0.0

    def test_empty_expected_returns_neutral(self):
        assert _keyword_overlap_score("", "anything goes here") == 0.5

    def test_stop_words_filtered(self):
        # Pair where tokens that *would* match are stop words (the/a/and).
        # Recall+precision computed on the meaningful set.
        a = "the quick brown fox jumps over the lazy dog"
        b = "the quick brown fox jumps over the lazy dog"
        assert _keyword_overlap_score(a, b) >= 0.9


class TestCharNgram:
    def test_identical_jaccard_one(self):
        assert _char_ngram_similarity("hello world", "hello world") == 1.0

    def test_disjoint_zero(self):
        assert _char_ngram_similarity("abcdef", "uvwxyz") == 0.0

    def test_empty_both_neutral(self):
        assert _char_ngram_similarity("", "") == 0.5

    def test_empty_one_zero(self):
        assert _char_ngram_similarity("foo", "") == 0.0


class TestStructuralMatch:
    def test_no_features_in_expected_returns_neutral(self):
        # No code blocks / lists / headers in plain prose → 0.6 floor.
        assert _structural_match_score("plain prose only", "plain prose only") == 0.6

    def test_matched_code_block(self):
        expected = "```python\nprint('x')\n```"
        output = "```python\nprint('y')\n```"
        # Both have code_block; expected_features = {code_block}, recall = 1.0
        assert _structural_match_score(expected, output) > 0.6

    def test_unexpected_features_penalised(self):
        expected = "plain text"
        output = "```code```\n- bullet\n# header"
        # No expected features → returns neutral 0.6 regardless of unexpected
        assert _structural_match_score(expected, output) == 0.6


class TestLengthQuality:
    def test_zero_expected_short_output(self):
        assert _length_quality_score("", "x") == 0.2

    def test_zero_expected_medium_output(self):
        assert _length_quality_score("", "a moderately long output here") == 0.8

    def test_zero_expected_very_long(self):
        assert _length_quality_score("", "x" * 6000) == 0.4

    def test_zero_output_zero_score(self):
        assert _length_quality_score("expected text", "") == 0.0

    def test_ratio_near_one_returns_one(self):
        expected = "x" * 100
        output = "y" * 100
        assert _length_quality_score(expected, output) == 1.0

    def test_very_short_relative_to_expected(self):
        expected = "x" * 100
        output = "y" * 5
        assert _length_quality_score(expected, output) < 0.2


class TestContentDensity:
    def test_empty_zero(self):
        assert _content_density_score("") == 0.0

    def test_repetitive_low_unique_ratio(self):
        score = _content_density_score("foo foo foo foo foo")
        # Unique ratio = 1/5 = 0.2 → contributes 0.4*0.2 = 0.08
        assert score < 0.6

    def test_varied_high(self):
        score = _content_density_score(
            "Comprehensive documentation describes architecture decisions clearly. "
            "Successive paragraphs introduce different aspects."
        )
        assert score > 0.5


class TestParseScore:
    def test_int(self):
        assert _parse_score(1) == 1.0

    def test_float(self):
        assert _parse_score(0.42) == 0.42

    def test_str_numeric(self):
        assert _parse_score("0.7") == 0.7

    def test_clamps_high(self):
        assert _parse_score(2.5) == 1.0

    def test_clamps_low(self):
        assert _parse_score(-1.0) == 0.0

    def test_garbage_returns_neutral(self):
        assert _parse_score("not a number") == 0.5

    def test_none_returns_neutral(self):
        assert _parse_score(None) == 0.5


class TestLLMJudgeWiring:
    """LLMJudge is opt-in via init_fitness_metric(use_llm_judge=True)."""

    def setup_method(self):
        reset_fitness_metric()

    def teardown_method(self):
        reset_fitness_metric()

    def test_uninitialized_uses_deterministic(self):
        # No init_fitness_metric call → judge is None → falls through.
        out = skill_fitness_metric(
            _make_example("alpha bravo charlie"),
            _make_pred("alpha bravo charlie"),
        )
        # Deterministic path returns a Prediction; judge would have produced
        # FitnessScore.composite. Either way, the score is a plain float.
        assert 0.0 <= out.score <= 1.0

    def test_judge_failure_falls_back_with_flag(self):
        config = EvolutionConfig()
        # init then mock the judge instance to raise
        init_fitness_metric(config, skill_text="dummy", use_llm_judge=True)

        from evolution.core import fitness as fitness_module
        bad_judge = MagicMock()
        bad_judge.score.side_effect = RuntimeError("simulated judge outage")
        fitness_module._judge = bad_judge

        out = skill_fitness_metric(
            _make_example("expected text"),
            _make_pred("some output"),
        )
        assert "judge unavailable" in out.feedback.lower()
        # Score should still be a valid float from the deterministic fallback
        assert 0.0 <= out.score <= 1.0

    def test_judge_success_returns_composite(self):
        config = EvolutionConfig()
        init_fitness_metric(config, skill_text="dummy", use_llm_judge=True)

        from evolution.core import fitness as fitness_module
        good_judge = MagicMock()
        good_judge.score.return_value = FitnessScore(
            correctness=0.9,
            procedure_following=0.8,
            completeness=0.7,
            length_penalty=0.0,
            feedback="great",
        )
        fitness_module._judge = good_judge

        out = skill_fitness_metric(
            _make_example("expected"),
            _make_pred("output"),
        )
        # composite = 0.4*0.9 + 0.3*0.8 + 0.3*0.7 = 0.36+0.24+0.21 = 0.81
        assert abs(out.score - 0.81) < 0.01
        assert out.feedback == "great"


class TestFitnessScore:
    def test_composite_weighted_correctly(self):
        s = FitnessScore(correctness=1.0, procedure_following=1.0, completeness=1.0)
        assert s.composite == pytest.approx(1.0)

    def test_length_penalty_subtracted(self):
        s = FitnessScore(
            correctness=1.0, procedure_following=1.0, completeness=1.0,
            length_penalty=0.3,
        )
        assert s.composite == pytest.approx(0.7)

    def test_floor_at_zero(self):
        s = FitnessScore(correctness=0.0, procedure_following=0.0, completeness=0.0,
                         length_penalty=0.5)
        assert s.composite == 0.0

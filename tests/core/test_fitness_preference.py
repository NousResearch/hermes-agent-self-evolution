"""Tests for the preference wiring inside FitnessScore (no LLM calls)."""

from types import SimpleNamespace

import pytest

from evolution.core.fitness import FitnessScore, _format_examples, make_skill_fitness_metric
from evolution.core.preference import PreferenceBook, PreferenceSignal, blend_preference


def _score(**kw) -> FitnessScore:
    base = dict(correctness=0.8, procedure_following=0.6, conciseness=0.5)
    base.update(kw)
    return FitnessScore(**base)


class TestBackwardCompatible:
    def test_no_preference_matches_plain_rubric(self):
        s = _score()
        # 0.5*0.8 + 0.3*0.6 + 0.2*0.5 = 0.68
        assert s.rubric_score == pytest.approx(0.68)
        assert s.composite == pytest.approx(0.68)

    def test_length_penalty_still_applies(self):
        s = _score(length_penalty=0.1)
        assert s.composite == pytest.approx(0.58)

    def test_zero_weight_is_inert(self):
        s = _score(preference_alignment=0.0, preference_weight=0.0)
        assert s.composite == pytest.approx(0.68)


class TestPreferenceBlend:
    def test_alignment_lifts_composite_within_cap(self):
        s = _score(preference_alignment=1.0, preference_weight=1.0, preference_influence=0.35)
        expected = blend_preference(0.68, 1.0, 1.0, 0.35)
        assert s.composite == pytest.approx(expected)
        assert s.composite > 0.68  # approval raised it

    def test_rejection_lowers_composite(self):
        s = _score(preference_alignment=0.0, preference_weight=1.0, preference_influence=0.35)
        assert s.composite < 0.68

    def test_partial_weight_scales_effect(self):
        strong = _score(preference_alignment=1.0, preference_weight=1.0).composite
        weak = _score(preference_alignment=1.0, preference_weight=0.3).composite
        assert strong > weak > 0.68

    def test_composite_stays_bounded(self):
        s = _score(correctness=1.0, procedure_following=1.0, conciseness=1.0,
                   preference_alignment=1.0, preference_weight=1.0)
        assert 0.0 <= s.composite <= 1.0


def test_format_examples():
    assert _format_examples([]) == "(none)"
    assert _format_examples(["  a  ", "", "b"]) == "- a\n- b"


class TestMetricFactory:
    def _ex(self, task, expected):
        return SimpleNamespace(task_input=task, expected_behavior=expected)

    def _pred(self, output):
        return SimpleNamespace(output=output)

    def test_no_book_equals_rubric_only(self):
        metric = make_skill_fitness_metric(book=None)
        ex = self._ex("write release notes", "concise bullet list")
        pred = self._pred("a concise bullet list")
        # Matches the historical skill_fitness_metric (rubric heuristic only).
        from evolution.core.fitness import skill_fitness_metric
        assert metric(ex, pred) == pytest.approx(skill_fitness_metric(ex, pred))

    def test_empty_book_is_inert(self):
        metric = make_skill_fitness_metric(book=PreferenceBook())
        ex = self._ex("write release notes", "concise bullet list")
        pred = self._pred("a concise bullet list")
        from evolution.core.fitness import skill_fitness_metric
        assert metric(ex, pred) == pytest.approx(skill_fitness_metric(ex, pred))

    def test_relevant_approval_lifts_score(self):
        book = PreferenceBook([
            PreferenceSignal(response="a concise bullet list of release notes", valence="up"),
            PreferenceSignal(response="a long rambling paragraph about release notes", valence="down"),
        ])
        metric = make_skill_fitness_metric(book, influence=0.35)
        plain = make_skill_fitness_metric(book=None)
        ex = self._ex("write the release notes", "release notes")
        approved_like = self._pred("a concise bullet list of release notes")
        # An approved-style output scores at least as high as the rubric alone.
        assert metric(ex, approved_like) >= plain(ex, approved_like)

    def test_offtopic_task_leaves_score_unchanged(self):
        book = PreferenceBook([PreferenceSignal(response="concise release notes bullet list", valence="up")])
        metric = make_skill_fitness_metric(book)
        plain = make_skill_fitness_metric(book=None)
        ex = self._ex("bake sourdough bread", "sourdough")
        pred = self._pred("mix flour water and salt then bake")
        assert metric(ex, pred) == pytest.approx(plain(ex, pred))

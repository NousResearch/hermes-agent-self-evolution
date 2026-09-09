"""Tests for multi-objective scoring and Pareto selection.

The headline case is a regression test for the defect that killed the only
real production run: the size penalty was computed from the baseline body, so
it evaluated to 0.000 for every candidate no matter how much the artifact
grew.
"""

from __future__ import annotations

import pytest

from evolution.core.objectives import (
    MAX_SIZE_PENALTY,
    SIZE_SOFT_START,
    ObjectiveVector,
    ObjectiveWeights,
    dominates,
    pareto_front,
    select_best,
    summarize_front,
)


def vec(quality=0.9, size=13_000, budget=17_500, baseline=13_218, **kw):
    return ObjectiveVector(
        quality=quality,
        size_chars=size,
        size_budget=budget,
        baseline_chars=baseline,
        **kw,
    )


class TestSizePenaltyTracksCandidate:
    """The penalty must respond to the candidate's size, not the baseline's."""

    def test_penalty_increases_with_candidate_size(self):
        penalties = [vec(size=s).size_penalty() for s in (13_000, 15_000, 16_500, 17_500)]
        assert penalties == sorted(penalties)
        assert penalties[0] < penalties[-1], "penalty must vary with candidate size"

    def test_the_aug21_variant_is_penalized(self):
        """13,218 -> 18,190 chars scored a 0.000 penalty and was then rejected."""
        bloated = ObjectiveVector(
            quality=0.93, size_chars=18_190, size_budget=15_000, baseline_chars=13_218
        )
        assert bloated.size_penalty() > 0
        assert bloated.growth_penalty() > 0
        assert bloated.scalarize() < 0.5

    def test_no_penalty_well_under_budget(self):
        assert vec(size=int(17_500 * SIZE_SOFT_START) - 1).size_penalty() == 0.0

    def test_penalty_is_capped(self):
        assert vec(size=10_000_000).size_penalty() == pytest.approx(MAX_SIZE_PENALTY)

    def test_tight_variant_beats_bloated_one_despite_lower_quality(self):
        tight = ObjectiveVector(quality=0.90, size_chars=13_100, size_budget=17_500, baseline_chars=13_218)
        bloated = ObjectiveVector(quality=0.93, size_chars=18_190, size_budget=17_500, baseline_chars=13_218)
        assert tight.scalarize() > bloated.scalarize()


class TestGrowthPenalty:
    def test_growth_within_allowance_is_free(self):
        assert vec(size=int(13_218 * 1.15)).growth_penalty() == 0.0

    def test_growth_beyond_allowance_costs(self):
        assert vec(size=int(13_218 * 1.40)).growth_penalty() > 0

    def test_no_baseline_means_no_growth_term(self):
        assert vec(size=50_000, baseline=0).growth_penalty() == 0.0


class TestScalarize:
    def test_quality_uses_the_full_range_when_alone(self):
        clean = dict(size=1_000, budget=17_500, baseline=1_000)
        assert vec(quality=1.0, **clean).scalarize() == pytest.approx(1.0)
        assert vec(quality=0.0, **clean).scalarize() == pytest.approx(0.0)
        assert vec(quality=0.5, **clean).scalarize() == pytest.approx(0.5)

    def test_size_weight_zero_disables_the_penalty(self):
        v = ObjectiveVector(quality=0.9, size_chars=30_000, size_budget=15_000, baseline_chars=10_000)
        assert v.scalarize(ObjectiveWeights(quality=1.0, size=0.0)) == pytest.approx(0.9)

    def test_efficiency_axes_need_budgets(self):
        v = vec(size=1_000, baseline=1_000, tokens=40_000, tool_calls=12)
        w = ObjectiveWeights(quality=1.0, size=1.0, tokens=1.0, tool_calls=1.0)
        assert v.scalarize(w) == pytest.approx(0.9)
        assert v.scalarize(w, token_budget=50_000, tool_call_budget=20) < 0.9

    def test_result_is_clamped(self):
        assert 0.0 <= vec(quality=5.0).scalarize() <= 1.0
        assert 0.0 <= vec(quality=-5.0, size=99_999).scalarize() <= 1.0

    def test_nan_quality_scores_zero(self):
        assert vec(quality=float("nan"), size=1_000, baseline=1_000).scalarize() == 0.0


class TestPareto:
    def test_dominance_requires_better_somewhere(self):
        a = ObjectiveVector(quality=0.9, size_chars=100)
        assert not dominates(a, a)

    def test_strictly_better_dominates(self):
        better = ObjectiveVector(quality=0.9, size_chars=100)
        worse = ObjectiveVector(quality=0.8, size_chars=200)
        assert dominates(better, worse)
        assert not dominates(worse, better)

    def test_trade_off_is_not_dominated(self):
        small_weak = ObjectiveVector(quality=0.7, size_chars=100)
        big_strong = ObjectiveVector(quality=0.95, size_chars=900)
        assert not dominates(small_weak, big_strong)
        assert not dominates(big_strong, small_weak)

    def test_axes_absent_on_both_sides_are_ignored(self):
        """Missing token telemetry must not make everything non-dominated."""
        a = ObjectiveVector(quality=0.9, size_chars=100, tokens=0)
        b = ObjectiveVector(quality=0.5, size_chars=100, tokens=0)
        assert dominates(a, b)

    def test_front_excludes_dominated_points(self):
        vectors = [
            ObjectiveVector(quality=0.9, size_chars=100),
            ObjectiveVector(quality=0.5, size_chars=200),  # dominated
            ObjectiveVector(quality=0.95, size_chars=900),
        ]
        assert pareto_front(vectors) == [0, 2]

    def test_select_best_never_picks_a_dominated_candidate(self):
        vectors = [
            ObjectiveVector(quality=0.9, size_chars=100, size_budget=1_000),
            ObjectiveVector(quality=0.5, size_chars=200, size_budget=1_000),
        ]
        assert select_best(vectors) == 0

    def test_select_best_on_empty_is_none(self):
        assert select_best([]) is None

    def test_summarize_front_returns_dicts(self):
        front = summarize_front([ObjectiveVector(quality=0.9, size_chars=100)])
        assert front and "scalar" in front[0] and "size_penalty" in front[0]

"""Tests for the statistics core.

Reference values are hand-computed or taken from standard tables and written as
literals, so these tests validate the implementation rather than re-deriving it
with the same code under test. Fully offline and deterministic: the bootstrap is
seeded, and no test depends on wall-clock time.
"""

import math

import pytest

from evolution.core.stats import (
    Interval,
    binomial_cdf,
    binomial_sf,
    chance_accuracy,
    cohens_d,
    cohens_h,
    holm_adjust,
    compare_paired_binary,
    compare_paired_continuous,
    mcnemar_exact,
    min_detectable_paired_shift,
    ols_trend,
    paired_bootstrap_ci,
    student_t_sf,
    wilcoxon_signed_rank,
    wilson_interval,
)


class TestStudentT:
    @pytest.mark.parametrize(
        "t,df,two_sided_p",
        [
            (2.228, 10, 0.050),   # standard table: t(0.025, 10)
            (1.812, 10, 0.100),   # t(0.05, 10)
            (3.169, 10, 0.010),   # t(0.005, 10)
            (2.086, 20, 0.050),   # t(0.025, 20)
            (12.706, 1, 0.050),   # t(0.025, 1)
        ],
    )
    def test_matches_published_tables(self, t, df, two_sided_p):
        assert 2 * student_t_sf(t, df) == pytest.approx(two_sided_p, abs=1e-3)

    def test_zero_is_the_median(self):
        assert student_t_sf(0.0, 7) == pytest.approx(0.5)

    def test_symmetric_about_zero(self):
        assert student_t_sf(-1.5, 9) == pytest.approx(1 - student_t_sf(1.5, 9))

    def test_converges_to_normal_at_high_df(self):
        assert 2 * student_t_sf(1.96, 5_000_000) == pytest.approx(0.05, abs=1e-3)

    def test_infinite_t_has_zero_tail(self):
        assert student_t_sf(math.inf, 5) == 0.0


class TestBinomial:
    def test_cdf_matches_hand_computation(self):
        # P(X <= 2 | n=12, p=0.5) = (1 + 12 + 66) / 4096
        assert binomial_cdf(2, 12) == pytest.approx(79 / 4096)

    def test_sf_matches_hand_computation(self):
        assert binomial_sf(10, 12) == pytest.approx((66 + 12 + 1) / 4096)

    def test_cdf_and_sf_are_complementary(self):
        assert binomial_cdf(4, 9) + binomial_sf(5, 9) == pytest.approx(1.0)

    def test_degenerate_inputs(self):
        assert binomial_cdf(5, 0) == 1.0
        assert binomial_sf(0, 10) == 1.0
        assert binomial_sf(11, 10) == 0.0


class TestMcNemar:
    def test_two_sided_matches_hand_computation(self):
        assert mcnemar_exact(10, 2) == pytest.approx(2 * 79 / 4096)

    def test_one_sided_worse_is_half_the_symmetric_case(self):
        assert mcnemar_exact(10, 2, "worse") == pytest.approx(79 / 4096)

    def test_no_discordant_pairs_is_no_evidence(self):
        assert mcnemar_exact(0, 0) == 1.0

    def test_concordant_count_is_irrelevant(self):
        """The whole point of the conditional test: only disagreements matter."""
        small = compare_paired_binary([True] * 2 + [False] * 3, [True] * 5)
        large = compare_paired_binary(
            [True] * 200 + [False] * 3, [True] * 203
        )
        assert small.p_two_sided == pytest.approx(large.p_two_sided)

    def test_direction_matters(self):
        assert mcnemar_exact(8, 1, "worse") < mcnemar_exact(1, 8, "worse")

    def test_symmetric_split_is_not_significant(self):
        assert mcnemar_exact(5, 5) == pytest.approx(1.0)


class TestWilson:
    def test_perfect_score_does_not_claim_certainty(self):
        interval = wilson_interval(10, 10)
        assert interval.low == pytest.approx(0.7225, abs=1e-3)
        assert interval.high == pytest.approx(1.0)

    def test_matches_published_value(self):
        interval = wilson_interval(50, 100)
        assert interval.low == pytest.approx(0.4038, abs=1e-3)
        assert interval.high == pytest.approx(0.5962, abs=1e-3)

    def test_narrows_as_n_grows(self):
        assert wilson_interval(80, 100).width < wilson_interval(8, 10).width

    def test_zero_successes_stays_in_range(self):
        interval = wilson_interval(0, 12)
        assert interval.low == 0.0
        assert 0.0 < interval.high < 1.0

    def test_empty_sample_is_maximally_uncertain(self):
        interval = wilson_interval(0, 0)
        assert (interval.low, interval.high) == (0.0, 1.0)

    def test_boundary_counts_hit_the_boundary_at_every_n(self):
        """The bound must be exactly 0 or 1, not merely near it.

        Checking a single n only samples the floating-point luck of that n.
        ``wilson_interval(0, 12).low`` is exactly 0.0, but ``(0, 10)`` lands on
        2.8e-17, which is a different answer to ``contains(0.0)`` for an eval
        set of a perfectly ordinary size.
        """
        for n in range(1, 201):
            assert wilson_interval(0, n).low == 0.0, f"n={n}"
            assert wilson_interval(n, n).high == 1.0, f"n={n}"

    def test_boundary_intervals_contain_the_boundary(self):
        """The property the exactness is for: 0 misses can still be a 0% rate."""
        for n in (10, 13, 34, 47):
            assert wilson_interval(0, n).contains(0.0), f"n={n}"
            assert wilson_interval(n, n).contains(1.0), f"n={n}"

    def test_interior_counts_stay_strictly_inside(self):
        """The clamps must not drag a real interval onto the boundary."""
        for n in range(2, 60):
            for successes in range(1, n):
                interval = wilson_interval(successes, n)
                assert 0.0 < interval.low <= interval.high < 1.0, f"{successes}/{n}"


class TestPower:
    def test_ten_examples_cannot_detect_less_than_fifty_points(self):
        assert min_detectable_paired_shift(10, 0.05) == pytest.approx(0.5)

    def test_hundred_examples_reach_five_points(self):
        assert min_detectable_paired_shift(100, 0.05) == pytest.approx(0.05)

    def test_stricter_alpha_needs_a_bigger_effect(self):
        assert min_detectable_paired_shift(50, 0.01) > min_detectable_paired_shift(50, 0.05)

    def test_underpowered_flag_is_honest_about_the_default_tolerance(self):
        comparison = compare_paired_binary([True] * 34 + [False] * 6,
                                           [True] * 30 + [False] * 10)
        assert comparison.n == 40
        assert comparison.underpowered_for(0.05) is True

    def test_rejects_nonsense_alpha(self):
        with pytest.raises(ValueError):
            min_detectable_paired_shift(10, 0.0)


class TestPairedBinary:
    def test_counts_the_four_cells(self):
        r = compare_paired_binary(
            [True, True, False, False], [True, False, True, False]
        )
        assert (r.both_correct, r.baseline_only, r.candidate_only, r.both_wrong) == (1, 1, 1, 1)

    def test_rates_and_delta(self):
        r = compare_paired_binary([True] * 8 + [False] * 2, [True] * 6 + [False] * 4)
        assert r.baseline_rate == pytest.approx(0.8)
        assert r.candidate_rate == pytest.approx(0.6)
        assert r.delta == pytest.approx(-0.2)

    def test_identical_outcomes_centre_on_zero_but_stay_uncertain(self):
        """No disagreement is not proof of no difference.

        Three examples that both versions handled the same way put the estimate
        at zero, but the rate of disagreement is unobserved rather than known to
        be zero, so the interval has to stay open.
        """
        r = compare_paired_binary([True, False, True], [True, False, True])
        interval = r.delta_interval()
        assert interval.point == 0.0
        assert interval.width > 0.0
        assert interval.contains(0.0)
        assert not r.significant_regression

    def test_a_clean_sweep_does_not_report_a_zero_width_interval(self):
        """The old Wald variance collapsed to zero exactly on the strongest result."""
        r = compare_paired_binary([False] * 10, [True] * 10)
        interval = r.delta_interval()
        assert interval.point == pytest.approx(1.0)
        assert interval.width > 0.1
        assert interval.low < 1.0

    def test_more_examples_narrow_the_no_disagreement_interval(self):
        few = compare_paired_binary([True] * 5, [True] * 5).delta_interval()
        many = compare_paired_binary([True] * 100, [True] * 100).delta_interval()
        assert many.width < few.width

    def test_large_consistent_regression_is_significant(self):
        r = compare_paired_binary([True] * 20, [False] * 8 + [True] * 12)
        assert r.significant_regression
        assert r.p_worse < 0.05

    def test_small_regression_on_few_examples_is_not_significant(self):
        r = compare_paired_binary([True] * 8 + [False] * 2, [True] * 7 + [False] * 3)
        assert not r.significant_regression

    def test_improvement_is_not_flagged_as_regression(self):
        r = compare_paired_binary([False] * 8 + [True] * 2, [True] * 10)
        assert r.significant_improvement
        assert not r.significant_regression

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="equal lengths"):
            compare_paired_binary([True], [True, False])

    def test_serialises(self):
        r = compare_paired_binary([True, False], [False, True])
        blob = r.to_dict()
        assert blob["n"] == 2
        assert "delta_ci" in blob and "p_worse" in blob


class TestPairedContinuous:
    def test_detects_a_consistent_shift(self):
        base = [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47]
        cand = [b + 0.20 for b in base]
        r = compare_paired_continuous(base, cand)
        assert r.delta == pytest.approx(0.20)
        assert r.significant_improvement
        assert not r.inconclusive

    def test_noise_is_inconclusive(self):
        base = [0.5, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65]
        cand = [0.55, 0.65, 0.35, 0.55, 0.45, 0.5, 0.5, 0.6]
        r = compare_paired_continuous(base, cand)
        assert r.inconclusive

    def test_empty_input_is_safe(self):
        r = compare_paired_continuous([], [])
        assert r.n == 0 and r.delta == 0.0

    def test_bootstrap_is_deterministic(self):
        base = [0.1, 0.4, 0.6, 0.2, 0.9, 0.3]
        cand = [0.2, 0.5, 0.5, 0.4, 0.8, 0.5]
        first = paired_bootstrap_ci(base, cand)
        second = paired_bootstrap_ci(base, cand)
        assert (first.low, first.high) == (second.low, second.high)

    def test_bootstrap_interval_brackets_the_point_estimate(self):
        base = [0.2, 0.4, 0.6, 0.8]
        cand = [0.3, 0.5, 0.7, 0.9]
        interval = paired_bootstrap_ci(base, cand)
        assert interval.low <= interval.point <= interval.high

    def test_constant_difference_has_no_spread(self):
        interval = paired_bootstrap_ci([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
        assert interval.low == pytest.approx(interval.high)


class TestWilcoxon:
    def test_all_positive_differences_are_significant(self):
        base = list(range(10))
        cand = [b + 5 for b in base]
        _, p = wilcoxon_signed_rank(base, cand)
        assert p < 0.05

    def test_identical_series_is_not_significant(self):
        _, p = wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])
        assert p == 1.0

    def test_zero_differences_are_dropped(self):
        stat_a, p_a = wilcoxon_signed_rank([1, 2, 3, 4], [2, 3, 4, 5])
        stat_b, p_b = wilcoxon_signed_rank([1, 2, 3, 4, 9], [2, 3, 4, 5, 9])
        assert (stat_a, p_a) == (stat_b, p_b)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            wilcoxon_signed_rank([1, 2], [1])


class TestOLSTrend:
    def test_recovers_a_known_line(self):
        t = ols_trend([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])
        assert t.slope == pytest.approx(2.0)
        assert t.intercept == pytest.approx(1.0)
        assert t.r_squared == pytest.approx(1.0)

    def test_pure_oscillation_is_not_a_trend(self):
        """The exact case the previous magnitude rule got wrong."""
        values = [0.90, 0.35, 0.85, 0.30, 0.88, 0.33, 0.60]
        t = ols_trend(list(range(len(values))), values)
        assert not t.significant
        assert t.p_value > 0.5
        assert t.r_squared < 0.2

    def test_real_decline_is_significant(self):
        values = [0.91, 0.86, 0.78, 0.71, 0.62, 0.55]
        t = ols_trend(list(range(len(values))), values)
        assert t.significant
        assert t.slope < 0
        assert t.r_squared > 0.95

    def test_two_points_are_never_a_trend(self):
        t = ols_trend([0, 1], [0.9, 0.1])
        assert t.n == 2
        assert not t.significant
        assert t.slope == 0.0

    def test_identical_timestamps_yield_no_slope(self):
        t = ols_trend([5, 5, 5], [0.1, 0.5, 0.9])
        assert t.slope == 0.0
        assert not t.significant

    def test_slope_interval_brackets_the_estimate(self):
        values = [0.9, 0.82, 0.79, 0.70, 0.66, 0.55]
        t = ols_trend(list(range(len(values))), values)
        assert t.slope_ci.low <= t.slope <= t.slope_ci.high

    def test_flat_data_is_not_significant(self):
        t = ols_trend([0, 1, 2, 3], [0.5, 0.5, 0.5, 0.5])
        assert not t.significant

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            ols_trend([1, 2], [1])

    def test_serialises(self):
        blob = ols_trend([0, 1, 2, 3], [1, 2, 3, 5]).to_dict()
        assert "p_value" in blob and "r_squared" in blob and "slope_ci" in blob


class TestEffectSizes:
    def test_cohens_h_is_zero_for_equal_proportions(self):
        assert cohens_h(0.5, 0.5) == pytest.approx(0.0)

    def test_cohens_h_weights_the_extremes_more(self):
        near_ceiling = abs(cohens_h(0.99, 0.95))
        mid_range = abs(cohens_h(0.54, 0.50))
        assert near_ceiling > mid_range

    def test_cohens_d_scales_by_variability(self):
        tight = cohens_d([0.5] * 6, [0.6, 0.6, 0.6, 0.6, 0.6, 0.61])
        loose = cohens_d([0.5] * 6, [0.1, 0.9, 0.3, 0.8, 0.2, 1.0])
        assert abs(tight) > abs(loose)

    def test_cohens_d_of_no_difference_is_zero(self):
        assert cohens_d([1, 2, 3], [1, 2, 3]) == 0.0

    def test_chance_accuracy(self):
        assert chance_accuracy(12) == pytest.approx(1 / 12)
        assert chance_accuracy(0) == 0.0


class TestInterval:
    def test_contains(self):
        interval = Interval(0.1, -0.05, 0.25)
        assert interval.contains(0.0)
        assert not interval.contains(0.5)

    def test_describe_renders_both_scales(self):
        interval = Interval(0.1, 0.05, 0.15)
        assert "%" in interval.describe(as_percent=True)
        assert "0.100" in interval.describe(as_percent=False)


class TestHolmAdjust:
    """Correcting a disjunction, and only a disjunction.

    Testing k candidates against one baseline and deploying any that clears is
    selection of the best of k. Four sections at alpha = 0.05 give a family-wise
    error of 1 - 0.95**4 = 18.6%, not 5%.
    """

    def test_matches_hand_computation(self):
        # Sorted p is [0.01, 0.03, 0.04, 0.20]; multipliers are 4, 3, 2, 1
        # with the running maximum enforced down the ladder.
        assert holm_adjust([0.01, 0.04, 0.03, 0.20]) == pytest.approx(
            [0.04, 0.09, 0.09, 0.20]
        )

    def test_a_single_test_is_left_alone(self):
        assert holm_adjust([0.03]) == pytest.approx([0.03])

    def test_empty_input(self):
        assert holm_adjust([]) == []

    def test_it_can_only_make_p_values_larger(self):
        raw = [0.001, 0.02, 0.049, 0.3]
        assert all(a >= r for r, a in zip(raw, holm_adjust(raw)))

    def test_it_is_monotone_in_the_raw_ordering(self):
        raw = [0.001, 0.02, 0.049, 0.3]
        adjusted = holm_adjust(raw)
        by_raw = [a for _, a in sorted(zip(raw, adjusted))]
        assert by_raw == sorted(by_raw)

    def test_borderline_results_stop_surviving(self):
        raw = [0.01, 0.04, 0.03, 0.20]
        assert sum(p < 0.05 for p in raw) == 3
        assert sum(p < 0.05 for p in holm_adjust(raw)) == 1

    def test_it_never_exceeds_one(self):
        assert all(a <= 1.0 for a in holm_adjust([0.6, 0.7, 0.9]))

    def test_it_is_more_powerful_than_plain_bonferroni(self):
        raw = [0.01, 0.02, 0.03]
        bonferroni = [min(1.0, len(raw) * p) for p in raw]
        assert holm_adjust(raw)[0] <= bonferroni[0]


class TestADegenerateIntervalSaysSo:
    """A zero-width bootstrap interval is a fact about the sample, not certainty.

    Interval's own contract reserves estimable=False for "a bootstrap over
    differences that are all identical, which has no spread to resample", and
    these are exactly those samples.
    """

    def test_an_empty_sample_is_not_estimable(self):
        assert paired_bootstrap_ci([], []).estimable is False

    def test_a_single_pair_is_not_estimable(self):
        assert paired_bootstrap_ci([0.0], [1.0]).estimable is False

    def test_identical_differences_are_not_estimable(self):
        assert paired_bootstrap_ci([0.0, 0.0], [1.0, 1.0]).estimable is False

    def test_differences_equal_within_float_noise_are_not_estimable(self):
        """0.3 - 0.1 and 0.4 - 0.2 differ in the last bit. That is not spread."""
        interval = paired_bootstrap_ci([0.1, 0.2], [0.3, 0.4])
        assert interval.estimable is False

    def test_a_sample_with_real_spread_is_still_estimable(self):
        interval = paired_bootstrap_ci([0.1, 0.9, 0.4], [0.5, 0.2, 0.8])
        assert interval.estimable is True
        assert interval.width > 0

    def test_the_empty_comparison_carries_a_non_estimable_interval(self):
        assert compare_paired_continuous([], []).delta_ci.estimable is False

    def test_a_non_estimable_interval_refuses_to_print_as_precision(self):
        interval = paired_bootstrap_ci([0.0], [1.0])
        assert "not estimable" in interval.describe()


class TestInconclusiveReadsTheEvidenceNotTheWidth:
    """A zero-width interval can mean no data, or it can mean total agreement.

    Both produce estimable=False, and they are opposite verdicts, so the
    signed-rank test decides which one it is.
    """

    def test_a_single_pair_settles_nothing(self):
        assert compare_paired_continuous([0.0], [1.0]).inconclusive is True

    def test_two_identical_pairs_settle_nothing(self):
        assert compare_paired_continuous([0.0, 0.0], [1.0, 1.0]).inconclusive is True

    def test_an_empty_comparison_settles_nothing(self):
        assert compare_paired_continuous([], []).inconclusive is True

    def test_perfect_consistency_is_evidence_not_the_absence_of_it(self):
        """Eight pairs that every one move +0.20 is the cleanest signal there is.

        The bootstrap has nothing to resample, so the interval is not
        estimable, but the signed-rank test reads p = 0.008. Treating a
        non-estimable interval as automatically inconclusive would throw this
        away along with the single-pair case.
        """
        base = [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47]
        result = compare_paired_continuous(base, [b + 0.20 for b in base])

        assert result.delta_ci.estimable is False
        assert result.wilcoxon_p < 0.01
        assert result.significant_improvement is True
        assert result.inconclusive is False

    def test_a_consistent_regression_is_also_evidence(self):
        base = [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47]
        result = compare_paired_continuous(base, [b - 0.20 for b in base])

        assert result.delta_ci.estimable is False
        assert result.significant_regression is True
        assert result.inconclusive is False

    def test_a_straddling_interval_is_still_inconclusive(self):
        base = [0.5, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45, 0.65]
        cand = [0.55, 0.65, 0.35, 0.55, 0.45, 0.5, 0.5, 0.6]
        result = compare_paired_continuous(base, cand)
        assert result.delta_ci.estimable is True
        assert result.inconclusive is True

    def test_the_serialised_form_agrees_with_the_property(self):
        result = compare_paired_continuous([0.0], [1.0])
        assert result.to_dict()["inconclusive"] is True
        assert result.to_dict()["delta_ci"]["estimable"] is False

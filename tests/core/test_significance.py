"""Tests for evolution.core.significance.

Pure standard library — no dspy/network needed, so these run in isolation.
"""

import math

import pytest

from evolution.core.significance import (
    SignificanceResult,
    bootstrap_delta_ci,
    compare,
    paired_permutation_test,
)


# ── decision logic ───────────────────────────────────────────────────────────

def test_clear_improvement_is_accepted():
    r = compare([0.2] * 6, [0.8] * 6)
    assert r.n == 6
    assert r.mean_delta == pytest.approx(0.6)
    assert r.win_rate == 1.0
    assert r.exact is True
    assert r.p_value < 0.05
    assert r.significant is True
    assert r.meets_min_effect is True
    assert r.accepted is True
    assert "significant improvement" in r.verdict


def test_no_difference_is_not_significant():
    scores = [0.5, 0.6, 0.7, 0.4]
    r = compare(scores, list(scores))
    assert r.mean_delta == pytest.approx(0.0)
    assert r.p_value == pytest.approx(1.0)
    assert r.significant is False
    assert r.accepted is False
    assert r.win_rate == 0.0
    assert "no change" in r.verdict


def test_tiny_noisy_delta_small_n_is_not_significant():
    # +0.033 average over 3 examples is well within noise.
    r = compare([0.5, 0.5, 0.5], [0.5, 0.5, 0.6])
    assert r.mean_delta > 0
    assert r.p_value >= 0.5  # exact sign-flip test: p == 0.5 here
    assert r.significant is False
    assert r.accepted is False


def test_significant_but_below_min_effect_is_rejected():
    # Perfectly consistent +0.03 shift: significant (p small) but only ~3.3%
    # relative — below the default 10% gate, so NOT accepted.
    r = compare([0.90] * 8, [0.93] * 8)
    assert r.significant is True
    assert r.p_value < 0.05
    assert r.meets_min_effect is False
    assert r.accepted is False
    assert "below" in r.verdict and "threshold" in r.verdict


def test_worse_variant_rejected():
    r = compare([0.8] * 5, [0.5] * 5)
    assert r.mean_delta < 0
    assert r.significant is False
    assert r.accepted is False
    assert "worse" in r.verdict


def test_partial_win_rate():
    r = compare([0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.3])
    assert r.win_rate == pytest.approx(0.75)


def test_low_power_note_for_n_below_3():
    r = compare([0.2, 0.2], [0.9, 0.9])
    assert r.n == 2
    assert any("power" in note for note in r.notes)


def test_empty_holdout():
    r = compare([], [])
    assert isinstance(r, SignificanceResult)
    assert r.n == 0
    assert r.accepted is False
    assert r.ci_method == "none"
    assert "no holdout examples" in r.verdict


# ── input validation ─────────────────────────────────────────────────────────

def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compare([0.1, 0.2], [0.3])
    with pytest.raises(ValueError):
        paired_permutation_test([0.1], [0.2, 0.3])
    with pytest.raises(ValueError):
        bootstrap_delta_ci([0.1], [0.2, 0.3])


def test_invalid_alternative_raises():
    with pytest.raises(ValueError):
        paired_permutation_test([0.1], [0.2], alternative="sideways")


def test_invalid_alpha_and_confidence_raise():
    with pytest.raises(ValueError):
        compare([0.1] * 3, [0.2] * 3, alpha=0.0)
    with pytest.raises(ValueError):
        compare([0.1] * 3, [0.2] * 3, confidence=1.0)


# ── permutation test ─────────────────────────────────────────────────────────

def test_two_sided_p_value_exact():
    p, exact = paired_permutation_test([0.2] * 5, [0.8] * 5, alternative="two-sided")
    assert exact is True
    # two-sided: both all-+1 and all--1 reach |0.6| → 2 / 2**5
    assert p == pytest.approx(2 / 32)


def test_less_alternative():
    # evolved clearly worse ⇒ one-sided "less" should be significant.
    p_less, _ = paired_permutation_test([0.8] * 6, [0.2] * 6, alternative="less")
    p_greater, _ = paired_permutation_test([0.8] * 6, [0.2] * 6, alternative="greater")
    assert p_less < 0.05
    assert p_greater == pytest.approx(1.0)


def test_monte_carlo_path_for_large_n():
    baseline = [0.3] * 30
    evolved = [0.6 if i % 2 == 0 else 0.55 for i in range(30)]  # varied ⇒ BCa runs
    r = compare(baseline, evolved, n_resamples=3000)
    assert r.exact is False  # n > exact enumeration cap
    assert r.p_value < 0.05
    assert r.accepted is True
    assert r.ci_method == "bca"


# ── confidence interval (BCa) ────────────────────────────────────────────────

def test_zero_variance_gives_point_interval_with_note():
    r = compare([0.2] * 6, [0.8] * 6)
    assert r.ci_method == "point"
    assert r.ci_low == pytest.approx(0.6)
    assert r.ci_high == pytest.approx(0.6)
    assert any("zero variance" in note for note in r.notes)


def test_varied_data_uses_bca_and_brackets_estimate():
    baseline = [0.40, 0.42, 0.38, 0.41, 0.39, 0.43]
    evolved = [0.55, 0.70, 0.50, 0.68, 0.60, 0.62]
    r = compare(baseline, evolved)
    assert r.ci_method == "bca"
    tol = 1e-9  # CI is bootstrapped from per-pair diffs; allow float epsilon
    assert r.ci_low - tol <= r.mean_delta <= r.ci_high + tol
    assert r.ci_low < r.ci_high  # genuine interval, not a point


def test_ci_endpoints_clamped_to_valid_range():
    # Differences span nearly the whole [-1, 1] range; CI must stay in-range.
    baseline = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    evolved = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    low, high = bootstrap_delta_ci(baseline, evolved)
    assert -1.0 <= low <= high <= 1.0


def test_zero_baseline_uses_infinite_relative_delta():
    r = compare([0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5])
    assert math.isinf(r.relative_delta)
    assert r.meets_min_effect is True  # any improvement clears the relative gate
    assert r.to_dict()["relative_delta"] is None  # non-finite ⇒ JSON-safe None


def test_effect_size_infinite_for_consistent_shift_serializes_to_none():
    r = compare([0.2] * 6, [0.8] * 6)
    assert math.isinf(r.effect_size)
    assert r.to_dict()["effect_size"] is None


# ── determinism ──────────────────────────────────────────────────────────────

def test_determinism():
    baseline = [0.4, 0.5, 0.45, 0.6, 0.55]
    evolved = [0.6, 0.55, 0.7, 0.65, 0.6]
    r1 = compare(baseline, evolved, seed=123)
    r2 = compare(baseline, evolved, seed=123)
    assert r1.p_value == r2.p_value
    assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)
    assert bootstrap_delta_ci(baseline, evolved, seed=7) == bootstrap_delta_ci(
        baseline, evolved, seed=7
    )


def test_decision_independent_of_ci_method():
    # Accept/reject is gated on the exact p-value + point effect, not the CI;
    # a zero-variance ("point" CI) case still accepts when warranted.
    r = compare([0.3] * 6, [0.7] * 6)
    assert r.ci_method == "point"
    assert r.accepted is True

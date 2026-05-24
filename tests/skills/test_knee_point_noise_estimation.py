"""Tests for noise-estimated knee-point ε via paired bootstrap.

Pure-Python, no LM. Synthetic val_subscores matrices exercise the helper's
degenerate paths (saturation, single candidate, all-zero diffs, partial
coverage) and pin its order-of-magnitude behavior against the analytical
binomial SE for a Bernoulli front.
"""

from __future__ import annotations

import math
import random

import pytest

from evolution.skills.knee_point import _estimate_val_noise


class TestEstimateValNoise:
    def test_estimate_val_noise_returns_zero_on_saturated_matrix(self):
        # Every candidate scores 1.0 everywhere → diff vector is all zeros →
        # bootstrap CI collapses to [0, 0]. No useful signal, no band.
        val_subscores = [[1.0] * 50 for _ in range(5)]
        eps = _estimate_val_noise(val_subscores, best_idx=0)
        assert eps == 0.0

    def test_estimate_val_noise_matches_analytical_se_on_bernoulli_p_half(self):
        # Independent Bernoulli(0.5) draws for best vs one competitor. The
        # paired diff has Var(X-Y) = 2·p(1-p) = 0.5 at p=0.5, so the SE of
        # the mean diff at n=50 is √(0.5/50) = 0.1. A 90% normal CI half-
        # width is ~1.645·SE ≈ 0.165. The helper's bootstrap CI half-width
        # should land in this neighborhood; a wide tolerance catches sign
        # errors and axis mistakes without overfitting to RNG quirks.
        rng = random.Random(123)
        n = 50
        best_scores = [float(rng.random() < 0.5) for _ in range(n)]
        other_scores = [float(rng.random() < 0.5) for _ in range(n)]
        val_subscores = [best_scores, other_scores]

        eps = _estimate_val_noise(val_subscores, best_idx=0)

        paired_se = math.sqrt(2.0 * 0.5 * 0.5 / n)
        analytical_ci_half = 1.645 * paired_se  # ≈ 0.165
        assert eps == pytest.approx(analytical_ci_half, rel=0.4, abs=0.05)

    def test_estimate_val_noise_widens_with_higher_variance(self):
        # Low-variance: diffs cluster tight (~0.01 spread).
        # High-variance: diffs span ±0.5. Bootstrap CI half-width must
        # be strictly larger on the high-variance matrix.
        n = 40
        best_low = [0.5] * n
        other_low = [0.5 + (0.01 if i % 2 == 0 else -0.01) for i in range(n)]
        low_var = [best_low, other_low]

        best_high = [0.5] * n
        other_high = [0.5 + (0.5 if i % 2 == 0 else -0.5) for i in range(n)]
        high_var = [best_high, other_high]

        eps_low = _estimate_val_noise(low_var, best_idx=0)
        eps_high = _estimate_val_noise(high_var, best_idx=0)

        assert eps_high > eps_low

    def test_estimate_val_noise_falls_back_on_single_candidate(self):
        # Only one candidate → no paired diffs possible. Degenerate path
        # returns the binomial-SE-ish floor 0.5 / √n_val.
        n_val = 64
        val_subscores = [[0.7] * n_val]
        eps = _estimate_val_noise(val_subscores, best_idx=0)
        assert eps == pytest.approx(0.5 / math.sqrt(n_val))
        assert eps == pytest.approx(0.0625)

    def test_estimate_val_noise_is_deterministic_with_seed(self):
        rng = random.Random(42)
        n = 30
        best_scores = [float(rng.random() < 0.6) for _ in range(n)]
        other_scores = [float(rng.random() < 0.4) for _ in range(n)]
        val_subscores = [best_scores, other_scores]

        eps_a = _estimate_val_noise(val_subscores, best_idx=0)
        eps_b = _estimate_val_noise(val_subscores, best_idx=0)
        assert eps_a == eps_b

    def test_estimate_val_noise_handles_partial_coverage(self):
        # Coverage policy under test: align by position; aggregate only over
        # indices present in both best and competitor (i.e., the first
        # min(len(best), len(k)) positions). This matches how DSPy stores
        # val_subscores positionally per-example; positions beyond the
        # shorter list are treated as un-evaluated, not as zeros.
        n_best = 50
        n_other = 30
        rng = random.Random(7)
        best_scores = [float(rng.random() < 0.5) for _ in range(n_best)]
        other_scores = [float(rng.random() < 0.5) for _ in range(n_other)]
        val_subscores = [best_scores, other_scores]

        eps = _estimate_val_noise(val_subscores, best_idx=0)
        # No crash; non-negative; finite.
        assert eps >= 0.0
        assert math.isfinite(eps)

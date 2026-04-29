"""Tests for MAD (Median Absolute Deviation) confidence scoring.

Ported from upstream pr/21. Covers pure-math functions without LLM calls.
"""

from __future__ import annotations

import pytest

from evolution.core.mad_scoring import (
    compute_mad,
    compute_confidence,
    is_better,
    find_baseline,
    find_best_kept,
    ConfidenceResult,
)


class TestComputeMAD:
    def test_empty_list(self):
        assert compute_mad([]) == 0.0

    def test_single_value(self):
        assert compute_mad([5.0]) == 0.0

    def test_identical_values(self):
        assert compute_mad([0.5, 0.5, 0.5]) == 0.0

    def test_simple_spread(self):
        # values: 1, 2, 3 → median 2 → deviations [1,0,1] → median 1
        assert compute_mad([1.0, 2.0, 3.0]) == 1.0

    def test_robust_to_outliers(self):
        """MAD should be unaffected by a single outlier."""
        clean = [0.5, 0.5, 0.5, 0.5, 0.5]
        with_outlier = [0.5, 0.5, 0.5, 0.5, 100.0]
        # MAD of clean is 0; with one outlier in 5 values, median stays 0.5
        # deviations: [0,0,0,0,99.5], median is 0
        assert compute_mad(clean) == 0.0
        assert compute_mad(with_outlier) == 0.0  # robust


class TestIsBetter:
    def test_higher_direction(self):
        assert is_better(0.7, 0.5, "higher") is True
        assert is_better(0.3, 0.5, "higher") is False
        assert is_better(0.5, 0.5, "higher") is False

    def test_lower_direction(self):
        assert is_better(0.3, 0.5, "lower") is True
        assert is_better(0.7, 0.5, "lower") is False


class TestFindBaseline:
    def test_first_value(self):
        assert find_baseline([0.7, 0.8, 0.9]) == 0.7

    def test_empty(self):
        assert find_baseline([]) == 0.0


class TestFindBestKept:
    def test_higher(self):
        assert find_best_kept([0.5, 0.7, 0.6], "higher") == 0.7

    def test_lower(self):
        assert find_best_kept([0.5, 0.7, 0.3], "lower") == 0.3

    def test_empty(self):
        assert find_best_kept([], "higher") == 0.0


class TestComputeConfidence:
    def test_empty_returns_discard(self):
        result = compute_confidence([])
        assert result.decision == "discard"
        assert result.label == "within noise"
        assert result.confidence == 0.0

    def test_real_improvement_likely_real(self):
        # Baseline 0.5, then [0.6, 0.7, 0.65] — clear positive trend, low MAD
        scores = [0.5, 0.7, 0.65, 0.68, 0.71, 0.69]
        result = compute_confidence(scores, direction="higher")
        assert result.confidence >= 2.0
        assert result.label == "likely real"
        assert result.decision == "keep"

    def test_within_noise(self):
        # Baseline higher than any subsequent score — no real improvement,
        # delta after best-kept logic is positive but small relative to MAD.
        scores = [0.5, 0.40, 0.42, 0.55, 0.45, 0.48]
        result = compute_confidence(scores, direction="higher")
        # Big spread, small delta from baseline → low confidence
        assert result.confidence < 2.0  # not "likely real"

    def test_lower_direction(self):
        scores = [0.8, 0.6, 0.55, 0.5, 0.45, 0.4]
        result = compute_confidence(scores, direction="lower")
        # Best (lowest) is 0.4, baseline 0.8 → delta -0.4 → big improvement
        assert result.delta < 0
        assert result.label == "likely real"

    def test_returns_confidence_result_dataclass(self):
        result = compute_confidence([0.5, 0.6, 0.7])
        assert isinstance(result, ConfidenceResult)
        assert hasattr(result, "decision")
        assert hasattr(result, "confidence")
        assert hasattr(result, "delta")
        assert hasattr(result, "label")


class TestEvolveSkillMADIntegration:
    """The post-hoc MAD calculation in evolve_skill.py should label correctly."""

    def test_labels_likely_real_on_clear_improvement(self):
        """Per-example deltas with low MAD should label 'likely real'."""
        baseline = [0.5, 0.5, 0.5, 0.5]
        evolved = [0.7, 0.71, 0.69, 0.7]
        deltas = [e - b for e, b in zip(evolved, baseline)]
        mad = compute_mad(deltas)
        mean_delta = sum(deltas) / len(deltas)
        confidence = abs(mean_delta) / mad if mad > 0 else float("inf")
        assert confidence >= 2.0  # likely real

    def test_labels_within_noise_when_inconsistent(self):
        """Mixed-direction deltas with large MAD should be within noise."""
        baseline = [0.5, 0.5, 0.5, 0.5]
        evolved = [0.55, 0.45, 0.50, 0.50]
        deltas = [e - b for e, b in zip(evolved, baseline)]
        mad = compute_mad(deltas)
        mean_delta = sum(deltas) / len(deltas)
        # mean_delta near 0, MAD nontrivial → low confidence
        confidence = abs(mean_delta) / mad if mad > 0 else 0.0
        assert confidence < 1.0

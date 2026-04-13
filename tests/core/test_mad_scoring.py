"""Tests for MAD confidence scoring and MAD-aware GEPA metric."""

import pytest
from evolution.core.mad_scoring import (
    compute_mad,
    compute_confidence,
    ConfidenceResult,
    mad_fitness_metric,
    ConfidenceScoredFitness,
)


# ---------------------------------------------------------------------------
# Unit tests: compute_mad
# ---------------------------------------------------------------------------

class TestComputeMad:
    def test_empty(self):
        assert compute_mad([]) == 0.0

    def test_single_value(self):
        assert compute_mad([5.0]) == 0.0

    def test_identical_values(self):
        assert compute_mad([3.0, 3.0, 3.0]) == 0.0

    def test_basic_spread(self):
        # median=4, deviations=[2,1,0,2,4], MAD=2
        assert compute_mad([2.0, 3.0, 4.0, 6.0, 8.0]) == 2.0

    def test_symmetric(self):
        assert compute_mad([1.0, 5.0]) == 2.0

    def test_robust_to_outlier(self):
        # Without outlier: MAD=0.5; with massive outlier, MAD stays small
        normal = [4.0, 5.0, 6.0]
        with_outlier = [4.0, 5.0, 6.0, 1000.0]
        mad_normal = compute_mad(normal)
        mad_outlier = compute_mad(with_outlier)
        # MAD should be robust — outlier increases it but not catastrophically
        assert mad_outlier < mad_normal * 5


# ---------------------------------------------------------------------------
# Unit tests: compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_empty_scores(self):
        result = compute_confidence([])
        assert result.decision == "discard"
        assert result.label == "within noise"

    def test_two_scores_fallback(self):
        # <3 scores triggers simple delta fallback
        result = compute_confidence([0.5, 0.8])
        assert result.delta == pytest.approx(0.3)
        assert result.n_trials == 2

    def test_high_confidence_keep(self):
        # Clear improvement: baseline=0.5, candidates=0.9,0.88,0.92
        # MAD is small relative to delta → high confidence
        scores = [0.5, 0.9, 0.88, 0.92]
        result = compute_confidence(scores)
        assert result.label == "likely real"
        assert result.decision == "keep"
        assert result.confidence >= 2.0

    def test_noisy_no_improvement(self):
        # Baseline is already the best — no candidate improved on it
        # scores[0]=0.7 is the max, so delta=0
        # median=0.5, deviations=[0.2,0.2,0.0,0.1,0.1], MAD=0.15
        # confidence=0/0.15=0 → within noise
        scores = [0.7, 0.3, 0.5, 0.4, 0.6]
        result = compute_confidence(scores)
        assert result.delta == 0.0
        assert result.label == "within noise"
        assert result.decision == "discard"

    def test_marginal_improvement(self):
        # Small improvement, moderate noise
        scores = [0.5, 0.6, 0.55, 0.58]
        result = compute_confidence(scores)
        # Should be marginal or within noise depending on MAD
        assert result.confidence > 0

    def test_zero_mad_identical_scores(self):
        # All identical except baseline different → confidence = abs(delta)
        scores = [0.5, 0.8, 0.8, 0.8]
        result = compute_confidence(scores)
        assert result.mad == 0.0
        assert result.confidence == pytest.approx(0.3)

    def test_direction_lower(self):
        # When direction="lower", best should be min
        scores = [10.0, 8.0, 7.5, 8.5]
        result = compute_confidence(scores, direction="lower")
        assert result.best == 7.5
        assert result.delta < 0  # improvement means lower
        assert result.label == "likely real"
        assert result.decision == "keep"

    def test_delta_pct(self):
        scores = [1.0, 1.5, 1.4, 1.6]
        result = compute_confidence(scores)
        assert result.delta_pct == pytest.approx(60.0, abs=1.0)

    def test_baseline_zero(self):
        # baseline=0 should not divide by zero
        scores = [0.0, 0.5, 0.6]
        result = compute_confidence(scores)
        assert result.delta_pct == 0.0  # guarded against div-by-zero


# ---------------------------------------------------------------------------
# Integration test: mad_fitness_metric with duck-typed mock objects
# ---------------------------------------------------------------------------

class _MockExample:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockPrediction:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestMadFitnessMetric:
    def test_empty_output_returns_zero(self):
        ex = _MockExample(task_input="do X", expected_behavior="do X well")
        pred = _MockPrediction(output="")
        assert mad_fitness_metric(ex, pred) == 0.0

    def test_good_overlap_returns_high(self):
        ex = _MockExample(
            task_input="search files by content using grep",
            expected_behavior="search files by content using grep recursively",
        )
        pred = _MockPrediction(output="search files by content using grep recursively in directory")
        score = mad_fitness_metric(ex, pred)
        assert score > 0.7

    def test_poor_overlap_returns_low(self):
        ex = _MockExample(
            task_input="deploy to production",
            expected_behavior="deploy the application to production server",
        )
        pred = _MockPrediction(output="banana apple orange fruit salad")
        score = mad_fitness_metric(ex, pred)
        assert score < 0.5

    def test_n_trials_parameter(self):
        """With n_trials > 1, the metric runs multiple evaluations and averages."""
        ex = _MockExample(
            task_input="search files",
            expected_behavior="search files recursively",
        )
        pred = _MockPrediction(output="search files recursively using find")
        score_multi = mad_fitness_metric(ex, pred, n_trials=3)
        score_single = mad_fitness_metric(ex, pred, n_trials=1)
        # Both should be in valid range
        assert 0.0 <= score_multi <= 1.0
        assert 0.0 <= score_single <= 1.0

    def test_confidence_returned(self):
        """When return_confidence=True, returns (score, ConfidenceResult) tuple."""
        ex = _MockExample(
            task_input="search files",
            expected_behavior="search files recursively",
        )
        pred = _MockPrediction(output="search files recursively using find")
        result = mad_fitness_metric(ex, pred, n_trials=3, return_confidence=True)
        assert isinstance(result, tuple)
        score, confidence = result
        assert 0.0 <= score <= 1.0
        assert isinstance(confidence, ConfidenceResult)

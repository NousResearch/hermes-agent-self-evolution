"""MAD confidence scoring for GEPA fitness evaluation.

Extracts Median Absolute Deviation (MAD) logic from factory-autoresearch-plugin.
Wraps LLMJudge with multi-trial noise-aware scoring so only statistically
significant improvements propagate through GEPA.

Usage:
    from evolution.core.mad_scoring import ConfidenceScoredFitness, compute_confidence

    # Wrap LLMJudge with 3-trial MAD scoring
    mad_judge = ConfidenceScoredFitness(judge, n_trials=3, confidence_threshold=2.0)
    score, confidence = mad_judge.score_with_confidence(task, expected, output, skill)

    # Or standalone confidence computation on a list of scores
    confidence = compute_confidence(scores, direction="higher")
"""

from dataclasses import dataclass
from typing import List, Optional, Literal
import statistics

ConfidenceLabel = Literal["likely real", "marginal", "within noise"]


@dataclass
class ConfidenceResult:
    """Result of a MAD confidence computation.

    confidence = |best - baseline| / MAD
    - confidence >= 2.0x  → "likely real" improvement
    - confidence >= 1.0x  → "marginal" (improved but within noise)
    - confidence <  1.0x  → "within noise" (no meaningful difference)
    """
    decision: str          # "keep" or "discard"
    confidence: float      # |best - baseline| / MAD
    delta: float           # best - baseline
    delta_pct: float       # (best - baseline) / baseline * 100
    label: ConfidenceLabel
    best: float
    baseline: float
    mad: float
    n_trials: int


def compute_mad(values: List[float]) -> float:
    """Compute Median Absolute Deviation (MAD).

    MAD = median(|x_i - median(values)|)
    This is a robust measure of variability that is less sensitive to
    outliers than standard deviation.

    Args:
        values: List of numeric values (e.g., repeated fitness scores)

    Returns:
        MAD as a float. Returns 0.0 for empty or single-value lists.
    """
    if not values:
        return 0.0
    if len(values) < 2:
        return 0.0  # No spread can be computed from 0-1 points

    med = statistics.median(values)
    deviations = [abs(v - med) for v in values]
    return statistics.median(deviations)


def is_better(
    current: float,
    best: float,
    direction: Literal["higher", "lower"] = "higher",
) -> bool:
    """Directional comparison."""
    if direction == "higher":
        return current > best
    else:
        return current < best


def find_baseline(results: List[float]) -> float:
    """First measurement is the baseline (before any changes)."""
    if not results:
        return 0.0
    return results[0]


def find_best_kept(
    results: List[float],
    direction: Literal["higher", "lower"] = "higher",
) -> float:
    """Best score from the kept/accepted measurements (excluding discarded outliers)."""
    if not results:
        return 0.0
    if direction == "higher":
        return max(results)
    else:
        return min(results)


def compute_confidence(
    scores: List[float],
    segment: int = 0,
    direction: Literal["higher", "lower"] = "higher",
) -> ConfidenceResult:
    """Compute MAD-based confidence that a change is a real improvement.

    confidence = |best - baseline| / MAD

    Args:
        scores:        List of scores ordered by evaluation index.
                       scores[0] = baseline (before change).
                       scores[1:] = candidate improvements.
        segment:       Not currently used; reserved for multi-segment analysis.
        direction:     "higher" = improvement means higher scores.

    Returns:
        ConfidenceResult with decision ("keep"/"discard"), confidence score,
        delta, delta_pct, and human-readable label.
    """
    if not scores:
        return ConfidenceResult(
            decision="discard",
            confidence=0.0,
            delta=0.0,
            delta_pct=0.0,
            label="within noise",
            best=0.0,
            baseline=0.0,
            mad=0.0,
            n_trials=0,
        )

    if len(scores) < 3:
        # Not enough trials for MAD — fall back to simple delta
        baseline = scores[0]
        best = max(scores) if direction == "higher" else min(scores)
        delta = best - baseline
        mad = 0.0
        confidence = abs(delta) if mad == 0.0 else abs(delta) / mad
        return ConfidenceResult(
            decision="keep" if abs(delta) > 0.05 else "discard",
            confidence=confidence,
            delta=delta,
            delta_pct=(delta / baseline * 100) if baseline != 0 else 0.0,
            label="within noise" if confidence < 1.0 else "marginal",
            best=best,
            baseline=baseline,
            mad=mad,
            n_trials=len(scores),
        )

    baseline = find_baseline(scores)
    best = find_best_kept(scores, direction)
    delta = best - baseline

    # Compute MAD over all measurements
    mad = compute_mad(scores)

    # Prevent division by zero: if MAD is 0, treat any non-zero delta as marginal
    if mad == 0.0:
        confidence = abs(delta) if delta != 0.0 else 0.0
    else:
        confidence = abs(delta) / mad

    # Label
    if confidence >= 2.0:
        label: ConfidenceLabel = "likely real"
    elif confidence >= 1.0:
        label = "marginal"
    else:
        label = "within noise"

    # Decision: keep only if improvement is real (confidence >= 2.0x)
    improvement = is_better(best, baseline, direction)
    decision = "keep" if (improvement and confidence >= 2.0) else "discard"

    delta_pct = (delta / baseline * 100) if baseline != 0.0 else 0.0

    return ConfidenceResult(
        decision=decision,
        confidence=confidence,
        delta=delta,
        delta_pct=delta_pct,
        label=label,
        best=best,
        baseline=baseline,
        mad=mad,
        n_trials=len(scores),
    )


# ---------------------------------------------------------------------------
# ConfidenceScoredFitness — wraps LLMJudge with multi-trial MAD scoring
# ---------------------------------------------------------------------------

# These imports are at the bottom of the file to allow the pure MAD functions
# (compute_mad, compute_confidence) to be used without dspy installed.

class ConfidenceScoredFitness:
    """Wraps LLMJudge with MAD-based confidence scoring.

    Instead of accepting a single LLM-as-judge evaluation at face value,
    this wrapper runs n_trials evaluations and computes MAD to determine
    whether observed differences are real signal or LLM noise.

    The confidence threshold (default 2.0x) means an improvement must be
    at least 2× the median absolute deviation of the measurement noise
    before it is trusted.

    Usage:
        judge = LLMJudge(config)
        mad_judge = ConfidenceScoredFitness(judge, n_trials=3, confidence_threshold=2.0)
        score, confidence = mad_judge.score_with_confidence(
            task_input="...",
            expected_behavior="...",
            agent_output="...",
            skill_text="...",
        )
    """

    def __init__(
        self,
        judge,
        n_trials: int = 3,
        confidence_threshold: float = 2.0,
        direction: Literal["higher", "lower"] = "higher",
    ):
        from evolution.core.fitness import LLMJudge, FitnessScore  # lazy
        self.judge = judge
        self.n_trials = n_trials
        self.confidence_threshold = confidence_threshold
        self.direction = direction

    def score_with_confidence(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> tuple:
        """Score agent output with MAD confidence over n_trials.

        Returns:
            Tuple of (best FitnessScore across trials, ConfidenceResult)
        """
        trial_fitness_scores: List[FitnessScore] = []

        for _ in range(self.n_trials):
            result = self.judge.score(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
                artifact_size=artifact_size,
                max_size=max_size,
            )
            trial_fitness_scores.append(result)

        trial_scores = [fs.composite for fs in trial_fitness_scores]

        # Compute confidence
        confidence_result = compute_confidence(trial_scores, direction=self.direction)

        # Return the best FitnessScore (by composite) across trials
        best_score = max(trial_fitness_scores, key=lambda s: s.composite)

        return best_score, confidence_result


# ---------------------------------------------------------------------------
# GEPA-compatible MAD metric
# ---------------------------------------------------------------------------

import random


def _heuristic_score_single(expected: str, output: str) -> float:
    """Fast deterministic heuristic — keyword overlap with noise injection point.

    This is a single-sample version of skill_fitness_metric that can be
    called N times with bootstrap resampling to produce synthetic variance
    for MAD computation without LLM cost.
    """
    if not output.strip():
        return 0.0

    expected_lower = expected.lower()
    output_lower = output.lower()

    expected_words = set(expected_lower.split())
    output_words = set(output_lower.split())

    if not expected_words:
        return 0.5

    overlap = len(expected_words & output_words) / len(expected_words)
    return min(1.0, max(0.0, 0.3 + (0.7 * overlap)))


def _heuristic_score_bootstrap(expected: str, output: str) -> float:
    """Bootstrap-resampled heuristic score for synthetic variance.

    Subsamples 70% of expected words per call to introduce variance
    suitable for MAD computation. This simulates the noise pattern
    of LLM-as-judge without API cost.
    """
    if not output.strip():
        return 0.0

    expected_lower = expected.lower()
    output_lower = output.lower()

    expected_words = list(set(expected_lower.split()))
    output_words = set(output_lower.split())

    if not expected_words:
        return 0.5

    # Bootstrap: subsample 70% of expected words
    sample_size = max(1, int(len(expected_words) * 0.7))
    sampled = random.sample(expected_words, sample_size)

    overlap = len(set(sampled) & output_words) / len(sampled)
    return min(1.0, max(0.0, 0.3 + (0.7 * overlap)))


def mad_fitness_metric(
    example,
    prediction,
    trace=None,
    n_trials: int = 1,
    return_confidence: bool = False,
):
    """MAD-aware DSPy metric for GEPA optimization.

    Drop-in replacement for skill_fitness_metric that adds multi-trial
    MAD confidence scoring. With n_trials=1, behaves identically to
    the baseline heuristic. With n_trials>=3, uses bootstrap resampling
    to estimate variance and returns a MAD-confidence-gated score.

    Args:
        example: DSPy example with task_input, expected_behavior fields.
        prediction: DSPy prediction with output field.
        trace: DSPy trace (unused, kept for interface compatibility).
        n_trials: Number of bootstrap trials. 1 = no MAD, >=3 = MAD scoring.
        return_confidence: If True, returns (score, ConfidenceResult) tuple.

    Returns:
        Float score (0-1) by default, or (score, ConfidenceResult) if requested.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""

    if not agent_output.strip():
        score = 0.0
        if return_confidence:
            return score, compute_confidence([])
        return score

    if n_trials < 3:
        # Fast path: single heuristic, no MAD overhead
        score = _heuristic_score_single(expected, agent_output)
        if return_confidence:
            return score, compute_confidence([score])
        return score

    # Multi-trial: bootstrap resampling for synthetic variance
    trial_scores = [_heuristic_score_bootstrap(expected, agent_output) for _ in range(n_trials)]

    # Use the mean of trials as the score (stable estimate)
    mean_score = sum(trial_scores) / len(trial_scores)

    # Compute MAD confidence
    confidence = compute_confidence(trial_scores)

    # Gate: if confidence says "within noise", discount the score
    # toward the baseline (first trial) to prevent GEPA from chasing noise
    if confidence.label == "within noise":
        gated_score = trial_scores[0]  # Use baseline, not mean
    else:
        gated_score = mean_score

    if return_confidence:
        return gated_score, confidence
    return gated_score

"""Evaluation integrity utilities for Phase 0 hardening."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import mean, stdev
import math


def model_family(model: str) -> str:
    """Extract a coarse model family from a model identifier."""
    if "/" in model:
        return model.split("/", 1)[0].strip().lower()
    if ":" in model:
        return model.split(":", 1)[0].strip().lower()
    return model.strip().lower().split("-", 1)[0]


def validate_model_separation(
    generator_model: str,
    judge_model: str,
    optimizer_model: str,
    minimum_distinct_families: int = 3,
) -> list[str]:
    """Return violations when generator/judge/optimizer families are not separated."""
    families = {
        "generator": model_family(generator_model),
        "judge": model_family(judge_model),
        "optimizer": model_family(optimizer_model),
    }
    errors: list[str] = []
    distinct = len(set(families.values()))
    if distinct < minimum_distinct_families:
        errors.append(
            "Model-family separation violated: generator/judge/optimizer must use "
            f"at least {minimum_distinct_families} distinct families "
            f"(got {distinct}: {families})",
        )
    return errors


def frozen_holdout_manifest(holdout_files: list[Path]) -> dict[str, str]:
    """Create a stable hash manifest for holdout files."""
    manifest: dict[str, str] = {}
    for path in sorted(holdout_files, key=lambda p: str(p)):
        digest = sha256(path.read_bytes()).hexdigest()
        manifest[str(path)] = digest
    return manifest


@dataclass(frozen=True)
class EffectEstimate:
    """Summary of baseline vs candidate metric deltas."""

    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    significant: bool


def estimate_effect_size(
    baseline_scores: list[float],
    candidate_scores: list[float],
    minimum_detectable_effect: float,
) -> EffectEstimate:
    """Estimate improvement with a 95% CI and significance decision."""
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError("Baseline and candidate score arrays must have equal length")
    if not baseline_scores:
        raise ValueError("Score arrays must be non-empty")

    deltas = [cand - base for base, cand in zip(baseline_scores, candidate_scores)]
    delta_mean = mean(deltas)
    if len(deltas) == 1:
        margin = 0.0
    else:
        margin = 1.96 * (stdev(deltas) / math.sqrt(len(deltas)))

    ci_low = delta_mean - margin
    ci_high = delta_mean + margin
    significant = ci_low >= minimum_detectable_effect

    return EffectEstimate(
        baseline_mean=mean(baseline_scores),
        candidate_mean=mean(candidate_scores),
        delta=delta_mean,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=significant,
    )

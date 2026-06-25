"""Statistical significance testing for evolution results.

The optimization loop (see ``PLAN.md`` → Architecture → step 5, "EVALUATE &
COMPARE / Statistical significance check") calls for deciding whether an
evolved variant is *actually* better than the baseline before proposing it.
With GEPA running on as few as 3 holdout examples, a raw average-score delta is
dominated by noise — a ``+0.02`` mean improvement over 5 examples is
meaningless on its own, yet the pipeline would otherwise treat any positive
delta as a win.

This module compares **paired** per-example scores (each holdout example scored
under both the baseline and the evolved variant) and reports whether the
improvement is real.

Method
------
*Decision (p-value):* an **exact paired randomization test**. Under the null
hypothesis that the evolved and baseline variants are exchangeable for each
example (no effect), relabelling which score is "baseline" and which is
"evolved" flips the sign of that example's difference, and all ``2**n`` sign
assignments are equally likely. We therefore compare the observed mean
difference against the distribution of mean differences over every sign
assignment — computed *exactly* for the small holdout sets that are the common
case (``n <= 14``), and by seeded Monte Carlo (with the standard ``+1``
correction) above that. This is distribution-free and exact for small ``n``;
it is the statistic the accept/reject decision is gated on.

*Magnitude (confidence interval):* a **BCa (bias-corrected and accelerated)
bootstrap** interval for the mean paired delta — the gold-standard bootstrap
interval, far better calibrated than a raw percentile interval on the small,
bounded, often-skewed score differences here. It is reported as a diagnostic;
it never affects the accept/reject decision, so its small-``n`` imprecision
cannot cause a wrong call.

Also reported: win rate (probability the evolved variant beats the baseline on
a random example) and a paired effect size (Cohen's ``d_z``).

Implementation: pure standard library (``math``, ``random``, ``statistics`` —
``NormalDist`` supplies the normal CDF/inverse-CDF the BCa adjustment needs).
No numpy/scipy, no API calls, no model inference — it runs on scores already
collected, so it adds zero runtime cost. Monte Carlo paths are seeded, so
results are deterministic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import product
from statistics import NormalDist, fmean
from typing import List, Sequence, Tuple

__all__ = [
    "SignificanceResult",
    "compare",
    "paired_permutation_test",
    "bootstrap_delta_ci",
]

# Holdout sets at or below this size get an exact randomization test (enumerate
# all 2**n sign flips). Above it, fall back to seeded Monte Carlo. 2**14 = 16384
# enumerations is instant; real holdout sets are typically 5–10 examples.
_EXACT_MAX_N = 14
_DEFAULT_RESAMPLES = 10_000

# Paired [0, 1] fitness scores ⇒ each per-example difference lies in [-1, 1], so
# the mean difference (the estimand) is bounded by these. CIs are clamped here.
_DELTA_MIN, _DELTA_MAX = -1.0, 1.0

_TIE_EPS = 1e-12  # tolerance for "as extreme as observed" float comparisons
_STD = NormalDist()  # standard normal: .cdf(z) = Φ(z), .inv_cdf(p) = Φ⁻¹(p)
_VALID_ALTERNATIVES = ("greater", "less", "two-sided")


@dataclass
class SignificanceResult:
    """Outcome of a baseline-vs-evolved significance comparison.

    ``accepted`` is the bottom-line decision the pipeline should gate on: the
    improvement is both statistically significant (``significant``) and large
    enough to matter (``meets_min_effect``). It depends only on the exact
    randomization p-value and the point effect — never on the bootstrap CI.
    """

    n: int
    mean_baseline: float
    mean_evolved: float
    mean_delta: float
    relative_delta: float  # mean_delta / mean_baseline (fraction; inf if baseline==0)
    win_rate: float  # fraction of examples where evolved > baseline
    ci_low: float
    ci_high: float
    ci_method: str  # "bca", "percentile" (fallback), "point", or "none"
    confidence: float
    p_value: float
    alpha: float
    alternative: str
    effect_size: float  # Cohen's d_z for paired samples
    exact: bool  # True when the permutation p-value was computed exactly
    significant: bool
    meets_min_effect: bool
    accepted: bool
    min_relative_effect: float
    verdict: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable view (handy for metrics.json / PR bodies)."""
        return {
            "n": self.n,
            "mean_baseline": self.mean_baseline,
            "mean_evolved": self.mean_evolved,
            "mean_delta": self.mean_delta,
            "relative_delta": _json_float(self.relative_delta),
            "win_rate": self.win_rate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_method": self.ci_method,
            "confidence": self.confidence,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "alternative": self.alternative,
            "effect_size": _json_float(self.effect_size),
            "exact": self.exact,
            "significant": self.significant,
            "meets_min_effect": self.meets_min_effect,
            "accepted": self.accepted,
            "min_relative_effect": self.min_relative_effect,
            "verdict": self.verdict,
            "notes": list(self.notes),
        }


def paired_permutation_test(
    baseline: Sequence[float],
    evolved: Sequence[float],
    *,
    alternative: str = "greater",
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Tuple[float, bool]:
    """Exact (or Monte Carlo) paired sign-flip randomization test.

    Tests whether the evolved scores differ from the baseline scores, treating
    the two labels as exchangeable per example under the null. Returns
    ``(p_value, exact)`` where ``exact`` is True when every sign assignment was
    enumerated rather than sampled.

    ``alternative``: ``"greater"`` (evolved > baseline; the default for "did it
    improve?"), ``"less"``, or ``"two-sided"``.
    """
    if alternative not in _VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_VALID_ALTERNATIVES}")
    if len(baseline) != len(evolved):
        raise ValueError("baseline and evolved must have the same length")
    diffs = [e - b for e, b in zip(evolved, baseline)]
    n = len(diffs)
    if n == 0:
        return 1.0, True

    observed = fmean(diffs)

    def _is_as_extreme(perm: float) -> bool:
        if alternative == "greater":
            return perm >= observed - _TIE_EPS
        if alternative == "less":
            return perm <= observed + _TIE_EPS
        return abs(perm) >= abs(observed) - _TIE_EPS  # two-sided

    if n <= _EXACT_MAX_N:
        total = 2 ** n
        hits = sum(
            1
            for signs in product((1.0, -1.0), repeat=n)
            if _is_as_extreme(fmean(s * d for s, d in zip(signs, diffs)))
        )
        return hits / total, True

    # Monte Carlo with the Phipson–Smyth (2010) +1 correction: the observed
    # assignment is counted in both numerator and denominator, so p is never 0.
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_resamples):
        perm = fmean((1.0 if rng.random() < 0.5 else -1.0) * d for d in diffs)
        if _is_as_extreme(perm):
            hits += 1
    return min(1.0, (hits + 1) / (n_resamples + 1)), False


def bootstrap_delta_ci(
    baseline: Sequence[float],
    evolved: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Tuple[float, float]:
    """BCa bootstrap confidence interval for the mean paired delta.

    Thin wrapper over :func:`_confidence_interval` that returns just the
    ``(low, high)`` endpoints.
    """
    low, high, _method = _confidence_interval(
        baseline, evolved, confidence=confidence, n_resamples=n_resamples, seed=seed
    )
    return low, high


def compare(
    baseline_scores: Sequence[float],
    evolved_scores: Sequence[float],
    *,
    alpha: float = 0.05,
    min_relative_effect: float = 0.10,
    confidence: float = 0.95,
    alternative: str = "greater",
    n_resamples: int = _DEFAULT_RESAMPLES,
    seed: int = 0,
) -> SignificanceResult:
    """Compare paired baseline vs evolved scores and decide if the win is real.

    Args:
        baseline_scores: per-example scores under the baseline variant.
        evolved_scores: per-example scores under the evolved variant. Must be
            the same length and in the same example order as ``baseline_scores``.
        alpha: significance level for the randomization test (default 0.05).
        min_relative_effect: minimum improvement relative to the baseline mean
            required to accept (default 0.10 → the plan's "≥10%" gate).
        confidence: confidence level for the BCa interval (default 0.95).
        alternative: directionality of the test (default ``"greater"``).

    Returns:
        A :class:`SignificanceResult`. Gate deployment / PR creation on
        ``result.accepted``.
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if len(baseline_scores) != len(evolved_scores):
        raise ValueError(
            f"score lists differ in length: "
            f"{len(baseline_scores)} baseline vs {len(evolved_scores)} evolved"
        )

    n = len(baseline_scores)
    notes: List[str] = []

    if n == 0:
        return SignificanceResult(
            n=0, mean_baseline=0.0, mean_evolved=0.0, mean_delta=0.0,
            relative_delta=0.0, win_rate=0.0, ci_low=0.0, ci_high=0.0,
            ci_method="none", confidence=confidence, p_value=1.0, alpha=alpha,
            alternative=alternative, effect_size=0.0, exact=True,
            significant=False, meets_min_effect=False, accepted=False,
            min_relative_effect=min_relative_effect,
            verdict="no holdout examples — cannot assess significance",
            notes=["empty holdout set"],
        )

    mean_baseline = fmean(baseline_scores)
    mean_evolved = fmean(evolved_scores)
    mean_delta = mean_evolved - mean_baseline
    diffs = [e - b for e, b in zip(evolved_scores, baseline_scores)]

    if mean_baseline > 0:
        relative_delta = mean_delta / mean_baseline
    else:
        relative_delta = math.inf if mean_delta > 0 else 0.0

    wins = sum(1 for d in diffs if d > 0)
    win_rate = wins / n

    p_value, exact = paired_permutation_test(
        baseline_scores, evolved_scores,
        alternative=alternative, n_resamples=n_resamples, seed=seed,
    )
    ci_low, ci_high, ci_method = _confidence_interval(
        baseline_scores, evolved_scores,
        confidence=confidence, n_resamples=n_resamples, seed=seed,
    )
    effect_size = _cohens_dz(diffs)

    if n < 3:
        notes.append(
            f"only {n} holdout example(s) — statistical power is very low; "
            "treat the verdict as indicative, not conclusive"
        )
    if ci_method == "point":
        notes.append(
            "all per-example differences are identical (zero variance) — the "
            "interval collapses to the point estimate; the p-value carries the "
            "uncertainty"
        )
    elif ci_method == "percentile":
        notes.append("BCa adjustment was undefined; fell back to a percentile interval")

    significant = (mean_delta > 0) and (p_value < alpha)
    meets_min_effect = relative_delta >= min_relative_effect
    accepted = significant and meets_min_effect

    verdict = _build_verdict(
        accepted=accepted, significant=significant,
        meets_min_effect=meets_min_effect, mean_delta=mean_delta,
        relative_delta=relative_delta, p_value=p_value, alpha=alpha,
        ci_low=ci_low, ci_high=ci_high, confidence=confidence,
        wins=wins, n=n, min_relative_effect=min_relative_effect,
    )

    return SignificanceResult(
        n=n, mean_baseline=mean_baseline, mean_evolved=mean_evolved,
        mean_delta=mean_delta, relative_delta=relative_delta, win_rate=win_rate,
        ci_low=ci_low, ci_high=ci_high, ci_method=ci_method, confidence=confidence,
        p_value=p_value, alpha=alpha, alternative=alternative,
        effect_size=effect_size, exact=exact, significant=significant,
        meets_min_effect=meets_min_effect, accepted=accepted,
        min_relative_effect=min_relative_effect, verdict=verdict, notes=notes,
    )


# ── internals ───────────────────────────────────────────────────────────────


def _confidence_interval(
    baseline: Sequence[float],
    evolved: Sequence[float],
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> Tuple[float, float, str]:
    """BCa bootstrap CI for the mean paired delta; returns (low, high, method).

    ``method`` is ``"bca"`` normally, ``"percentile"`` if the BCa adjustment is
    undefined, ``"point"`` for zero-variance data, or ``"none"`` for n == 0.
    """
    if len(baseline) != len(evolved):
        raise ValueError("baseline and evolved must have the same length")
    diffs = [e - b for e, b in zip(evolved, baseline)]
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, "none"

    observed = fmean(diffs)
    # Zero variance ⇒ every resample has the same mean; the interval is a point.
    if all(abs(d - diffs[0]) < _TIE_EPS for d in diffs):
        v = _clamp_delta(observed)
        return v, v, "point"

    rng = random.Random(seed)
    boot = sorted(
        fmean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(n_resamples)
    )

    lo_q = (1.0 - confidence) / 2.0
    hi_q = 1.0 - lo_q

    method = "bca"
    adj = _bca_quantiles(diffs, boot, observed, lo_q, hi_q)
    if adj is None:  # BCa undefined (degenerate jackknife / extreme bias) → percentile
        adj = (lo_q, hi_q)
        method = "percentile"

    low = _clamp_delta(_percentile(boot, adj[0]))
    high = _clamp_delta(_percentile(boot, adj[1]))
    if low > high:  # numerical safety
        low, high = high, low
    return low, high, method


def _bca_quantiles(
    diffs: Sequence[float],
    boot: List[float],
    observed: float,
    lo_q: float,
    hi_q: float,
) -> Tuple[float, float] | None:
    """Bias-corrected & accelerated quantile adjustment (Efron).

    Returns the adjusted (lower, upper) probabilities to read off the sorted
    bootstrap distribution, or ``None`` if the adjustment is undefined.
    """
    b = len(boot)
    # Bias-correction z0 from the share of bootstrap means below the observed.
    n_below = sum(1 for x in boot if x < observed - _TIE_EPS)
    prop = _clamp_open(n_below / b)
    z0 = _STD.inv_cdf(prop)
    if not math.isfinite(z0):
        return None

    # Acceleration from the jackknife skewness of the statistic.
    n = len(diffs)
    total = math.fsum(diffs)
    jack = [(total - d) / (n - 1) for d in diffs]  # leave-one-out means
    jbar = fmean(jack)
    d3 = math.fsum((jbar - j) ** 3 for j in jack)
    d2 = math.fsum((jbar - j) ** 2 for j in jack)
    if d2 <= _TIE_EPS:
        return None
    a = d3 / (6.0 * (d2 ** 1.5))
    if not math.isfinite(a):
        return None

    def _adjust(q: float) -> float:
        zq = _STD.inv_cdf(q)
        denom = 1.0 - a * (z0 + zq)
        if abs(denom) < _TIE_EPS:
            return q
        return _STD.cdf(z0 + (z0 + zq) / denom)

    return _clamp_open(_adjust(lo_q)), _clamp_open(_adjust(hi_q))


def _percentile(sorted_values: List[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list (q in [0,1])."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _cohens_dz(diffs: Sequence[float]) -> float:
    """Paired effect size: mean(diffs) / sample-stdev(diffs).

    Returns ``inf``/``-inf`` for a perfectly consistent nonzero shift (zero
    variance) and ``0.0`` when there is no shift at all.
    """
    n = len(diffs)
    if n < 2:
        return 0.0
    m = fmean(diffs)
    var = math.fsum((d - m) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    if sd < _TIE_EPS:
        if abs(m) < _TIE_EPS:
            return 0.0
        return math.inf if m > 0 else -math.inf
    return m / sd


def _clamp_delta(value: float) -> float:
    """Clamp a mean-difference estimate to the valid [-1, 1] range."""
    return max(_DELTA_MIN, min(_DELTA_MAX, value))


def _clamp_open(p: float) -> float:
    """Clamp a probability into the open interval (0, 1) for inv_cdf safety."""
    return min(1.0 - 1e-12, max(1e-12, p))


def _json_float(value: float) -> float | None:
    """Map non-finite floats to None so the value survives ``json.dumps``."""
    if value is None or math.isinf(value) or math.isnan(value):
        return None
    return value


def _fmt_pct(fraction: float) -> str:
    if math.isinf(fraction):
        return "+∞%" if fraction > 0 else "-∞%"
    return f"{fraction * 100:+.1f}%"


def _build_verdict(
    *, accepted: bool, significant: bool, meets_min_effect: bool,
    mean_delta: float, relative_delta: float, p_value: float, alpha: float,
    ci_low: float, ci_high: float, confidence: float, wins: int, n: int,
    min_relative_effect: float,
) -> str:
    conf_pct = int(round(confidence * 100))
    stats = (
        f"Δ={mean_delta:+.3f} ({_fmt_pct(relative_delta)}), p={p_value:.3f}, "
        f"{conf_pct}% CI [{ci_low:+.3f}, {ci_high:+.3f}], win rate {wins}/{n}"
    )
    if accepted:
        return f"significant improvement — {stats}"
    if significant and not meets_min_effect:
        return (
            f"statistically significant but below the {min_relative_effect:.0%} "
            f"effect threshold — {stats}"
        )
    if mean_delta > 0:
        return f"improvement not statistically significant (likely noise) — {stats}"
    if mean_delta == 0:
        return f"no change between baseline and evolved — {stats}"
    return f"evolved variant scored worse than baseline — {stats}"

"""Decide what to optimize next, and be able to say why.

PLAN.md Phase 5 gives the ranking rule in one line: rank candidates by
``potential improvement x usage frequency``. That product is the whole idea.
A skill that fails half the time but runs twice a year is not worth a GEPA run;
a tool that is only slightly misdescribed but sits on the hot path of every
session is. Neither factor alone gets that right.

Around the product sit three adjustments, each applied as a named multiplier so
it stays visible in the explanation rather than disappearing into a number:

    declining trend     a target that is getting worse outranks one that is
                        merely bad and stable, because the stable one has
                        already been priced in
    user corrections    "no, use X instead" is the highest-quality failure
                        label available, so corrections push a target up
    threshold trigger   PLAN.md's rule that a failure rate past X% auto-fires
                        an optimization, independent of ranking

Every entry carries its factors, so :meth:`TriageEntry.explain` can state the
arithmetic that put it where it is. That matters because the output of this
module becomes an unattended decision to spend money on an optimizer run, and a
human reviewing the resulting PR needs to be able to reconstruct the reasoning
without rerunning anything. Each entry also carries the p-value and R2 of its
trend, so a target promoted by a clean decline is visibly more certain than one
promoted by a noisy series that happened to end low.

Usage frequency is compressed, not linear. Normalizing sample counts by simple
division against the busiest target means one high-traffic target squashes
everything else toward zero: in a real ranking, a target that fell 0.91 to 0.55
placed *below* one that fell 0.82 to 0.74, purely because the second ran more
often. Volume is real evidence and should count, but it is evidence about how
much a fix is worth, not about how broken something is, and a linear weight
lets it overwhelm severity. A log1p curve keeps the ordering by volume while
flattening the gap between "busy" and "extremely busy" - see
:func:`_usage_weight`.

Honesty rule carried over from the gates: a benchmark score is a real signal but
there is no phase entry point that optimizes "tblite", so benchmark candidates
are ranked and reported as advisory and never marked actionable. Reporting a
target the loop cannot act on is fine. Pretending to act on it is not.

Pure computation over :mod:`evolution.monitor.metrics` points: no network, no
model, no subprocesses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

from evolution.core.stats import Interval, wilson_interval
from evolution.monitor.metrics import (
    BENCHMARK_SCORE,
    SECONDS_PER_DAY,
    SKILL_SUCCESS_RATE,
    TOOL_SELECTION_ACCURACY,
    USER_CORRECTION,
    Aggregate,
    MetricPoint,
    MetricStore,
    Trend,
    compute_trend,
    summarize,
    utc_now,
)

__all__ = [
    "TargetType",
    "METRIC_TARGET_TYPES",
    "TriageConfig",
    "ScoreFactor",
    "TriageEntry",
    "AutoTriage",
    "rank_points",
]


class TargetType(str, Enum):
    """What kind of artifact a candidate names.

    The loop maps these to phase entry points. ``BENCHMARK`` deliberately has no
    mapping: a low benchmark score tells you something is wrong without telling
    you which artifact to evolve.
    """

    SKILL = "skill"
    TOOL = "tool"
    PROMPT = "prompt"
    CODE = "code"
    BENCHMARK = "benchmark"


# Which artifact a metric is measuring, when nothing more specific is known.
METRIC_TARGET_TYPES: dict[str, TargetType] = {
    SKILL_SUCCESS_RATE: TargetType.SKILL,
    TOOL_SELECTION_ACCURACY: TargetType.TOOL,
    BENCHMARK_SCORE: TargetType.BENCHMARK,
    # A correction is usually "you should have used tool X". When the target
    # also appears under another metric, that metric's type wins instead.
    USER_CORRECTION: TargetType.TOOL,
}

# Types that a phase entry point can actually act on.
ACTIONABLE_TYPES = (
    TargetType.SKILL,
    TargetType.TOOL,
    TargetType.PROMPT,
    TargetType.CODE,
)


@dataclass
class TriageConfig:
    """Knobs for ranking. Defaults are the ones the CLI ships with."""

    # PLAN.md: "when a skill's failure rate exceeds X%, auto-trigger GEPA".
    # Crossing is inclusive: a failure rate exactly at the threshold fires.
    failure_threshold: float = 0.30

    # How far back to look. Older history stays on disk and still counts for
    # nothing here, because a problem fixed two months ago is not a problem.
    window_days: float = 30.0

    # A target needs this many underlying observations before its rate is
    # allowed to trigger an optimization run.
    min_samples: int = 5

    # Trend detection. ``significant_change`` is the practical floor a decline
    # must clear; ``trend_alpha`` is the level the t-test on the slope is read
    # at. A trend has to pass both before it counts as an actionable decline.
    min_points_for_trend: int = 3
    flat_tolerance: float = 0.02
    significant_change: float = 0.05
    trend_alpha: float = 0.05
    trend_confidence: float = 0.95

    # How hard to compress usage frequency. The weight is
    # ``log1p(k * samples) / log1p(k * busiest)``: as k approaches 0 the curve
    # straightens back into the old ``samples / busiest``, and as it grows the
    # curve flattens until volume barely separates targets at all. 1.0 keeps
    # order by volume while stopping one hot target from zeroing the rest.
    # Setting it to 0 restores exact linear normalization.
    usage_compression: float = 1.0

    # The best a metric can be. Headroom is measured against this.
    ceiling: float = 1.0

    # Multiplier weights for the two adjustments.
    decline_weight: float = 0.5
    correction_weight: float = 0.5

    # Corrections in the window that count as "as bad as it gets" for pressure.
    correction_saturation: float = 10.0
    # Corrections alone that are enough to fire a trigger.
    correction_trigger_count: int = 5

    # Entries scoring below this are dropped, unless they were triggered.
    min_score: float = 0.01

    # Extra metric name -> target type, for signals a deployment adds later.
    extra_metric_types: dict = field(default_factory=dict)

    def metric_types(self) -> dict:
        """The metric-to-target-type map with any caller overrides applied."""
        merged = dict(METRIC_TARGET_TYPES)
        merged.update(self.extra_metric_types)
        return merged


@dataclass
class ScoreFactor:
    """One named term in a candidate's score, kept for the explanation."""

    name: str
    value: float
    detail: str

    def to_dict(self) -> dict:
        """Serialise one scoring signal and its detail."""
        return {"name": self.name, "value": self.value, "detail": self.detail}


@dataclass
class TriageEntry:
    """A ranked optimization candidate and the reasoning behind its rank."""

    target: str
    target_type: TargetType
    metric: str
    score: float
    potential_improvement: float
    usage_weight: float
    usage_samples: int
    current_value: Optional[float]
    observations: int
    corrections: int
    trend: Optional[Trend] = None
    triggered: bool = False
    trigger_reason: str = ""
    factors: list[ScoreFactor] = field(default_factory=list)
    # Copied off the trend so the uncertainty survives serialization and shows
    # up in the table next to the score it moved. ``None`` means there was no
    # fittable series: a correction-only target, or too little history.
    trend_p_value: Optional[float] = None
    trend_r_squared: Optional[float] = None

    @property
    def actionable(self) -> bool:
        """True when a phase entry point exists for this kind of target."""
        return self.target_type in ACTIONABLE_TYPES

    @property
    def headroom(self) -> Optional[float]:
        """Distance from the current value up to the configured ceiling.

        This is what ``potential_improvement`` measures and what the ranking
        multiplies by usage. It equals the failure rate only when the ceiling
        is 1.0.
        """
        return None if self.current_value is None else self.potential_improvement

    @property
    def failure_rate(self) -> Optional[float]:
        """The share of observations that actually failed, ``1 - value``.

        Deliberately not the same number as :attr:`headroom`. This used to
        return headroom, which mislabelled a ceiling shortfall as a failure
        rate and misstated it in the trigger message whenever the ceiling was
        below 1.0.
        """
        if self.current_value is None:
            return None
        return max(0.0, min(1.0, 1.0 - self.current_value))

    def confidence_note(self) -> str:
        """Uncertainty of the trend behind this entry, or "" when there is none."""
        return self.trend.confidence_note() if self.trend is not None else ""

    def explain(self) -> str:
        """One line stating why this entry ranked where it did."""
        terms = " x ".join(f"{f.value:.2f} {f.name}" for f in self.factors)
        line = f"score {self.score:.3f} = {terms}" if terms else f"score {self.score:.3f}"
        confidence = self.confidence_note()
        if confidence:
            line += f" [trend {confidence}]"
        if self.triggered:
            line += f" [TRIGGERED: {self.trigger_reason}]"
        if not self.actionable:
            line += " [advisory: no phase entry point for this target type]"
        return line

    def details(self) -> list[str]:
        """The factor details, one per line, for verbose output."""
        return [f"{f.name}: {f.detail}" for f in self.factors]

    def to_dict(self) -> dict:
        """Serialise the triage entry, its score and its signals."""
        return {
            "target": self.target,
            "target_type": self.target_type.value,
            "metric": self.metric,
            "score": self.score,
            "potential_improvement": self.potential_improvement,
            "headroom": self.headroom,
            "failure_rate": self.failure_rate,
            "usage_weight": self.usage_weight,
            "usage_samples": self.usage_samples,
            "current_value": self.current_value,
            "observations": self.observations,
            "corrections": self.corrections,
            "trend": self.trend.to_dict() if self.trend else None,
            "trend_p_value": self.trend_p_value,
            "trend_r_squared": self.trend_r_squared,
            "triggered": self.triggered,
            "trigger_reason": self.trigger_reason,
            "actionable": self.actionable,
            "factors": [f.to_dict() for f in self.factors],
            "explanation": self.explain(),
        }


# ──────────────────────────────────────────────────────────────────────────
# Ranking
# ──────────────────────────────────────────────────────────────────────────


def _known_target_types(
    points: Sequence[MetricPoint], metric_types: dict
) -> dict[str, TargetType]:
    """What each target has been observed as, across the whole history.

    Built from every point handed to :func:`rank_points`, not just the windowed
    ones, because a target's *kind* does not expire the way its numbers do. A
    skill that was measured two months ago and has only user corrections this
    month is still a skill, and dispatching it to the tool phase would be wrong.
    Later observations win, and an explicit ``target_type`` in metadata beats
    anything inferred from a metric name.
    """
    known: dict[str, TargetType] = {}
    ordered = sorted(points, key=lambda p: p.timestamp)

    for point in ordered:
        if point.metric == USER_CORRECTION:
            continue
        inferred = metric_types.get(point.metric)
        if inferred is not None and inferred in ACTIONABLE_TYPES:
            known[point.target] = inferred

    for point in ordered:
        declared = point.metadata.get("target_type")
        if not declared:
            continue
        try:
            known[point.target] = TargetType(str(declared).lower())
        except ValueError:
            continue

    return known


def _resolve_target_type(
    target: str,
    metric: str,
    by_target: dict[str, list[MetricPoint]],
    known_types: dict[str, TargetType],
    metric_types: dict,
) -> TargetType:
    """Work out what kind of artifact *target* is.

    Priority: an explicit ``target_type`` in recent metadata, then the metric's
    own meaning, then what this target has been observed as elsewhere in the
    history, then the metric's default. That third rule is what stops a
    correction logged against a skill from being dispatched to the tool phase.
    """
    for point in reversed(by_target.get(target, [])):
        declared = point.metadata.get("target_type")
        if declared:
            try:
                return TargetType(str(declared).lower())
            except ValueError:
                continue

    if metric != USER_CORRECTION:
        inferred = metric_types.get(metric)
        if inferred is not None:
            return inferred

    if target in known_types:
        return known_types[target]

    return metric_types.get(metric, TargetType.SKILL)


def _usage_weight(samples: int, max_samples: int, config: TriageConfig) -> float:
    """How much weight this target's traffic earns, on a 0-1 scale.

    ``samples / busiest`` is the obvious answer and it is the wrong one. It is
    linear against a single reference point, so a target running ten thousand
    times a week drives every other target's weight toward zero, and the
    ranking stops being "improvement x usage" and becomes "usage". The observed
    failure was a target that fell 0.91 to 0.55 ranking below one that fell
    0.82 to 0.74, on sample count alone.

    ``log1p(k * samples) / log1p(k * busiest)`` fixes the compression without
    discarding the signal. It is monotone, so more traffic still means more
    weight and the busiest target still scores 1.0, but the curve is steep
    where the counts are small (10 vs 100 uses is a real difference) and flat
    where they are large (10,000 vs 100,000 uses barely changes what a fix is
    worth). *k* is ``config.usage_compression``: as it approaches 0 the
    expression converges on the old linear ratio, and 0 selects it exactly.
    """
    if max_samples <= 0:
        return 0.0
    samples = max(0, samples)
    k = config.usage_compression
    if k <= 0:
        return min(1.0, samples / max_samples)
    denominator = math.log1p(k * max_samples)
    if denominator <= 0:
        return 0.0
    return min(1.0, math.log1p(k * samples) / denominator)


def _correction_pressure(corrections: int, config: TriageConfig) -> float:
    if corrections <= 0 or config.correction_saturation <= 0:
        return 0.0
    return min(1.0, corrections / config.correction_saturation)


def rank_points(
    points: Sequence[MetricPoint],
    config: Optional[TriageConfig] = None,
    now: Optional[float] = None,
) -> list[TriageEntry]:
    """Rank optimization candidates found in *points*.

    Filters to the configured trailing window itself, so callers can hand over
    a whole history. Returns entries sorted by score, highest first; ties break
    on triggered-first and then target name so the order is reproducible.
    """
    config = config or TriageConfig()
    end = utc_now() if now is None else now
    start = end - config.window_days * SECONDS_PER_DAY

    windowed = [p for p in points if start <= p.timestamp <= end]
    metric_types = config.metric_types()
    known_types = _known_target_types(points, metric_types)

    by_target: dict[str, list[MetricPoint]] = {}
    by_series: dict[tuple[str, str], list[MetricPoint]] = {}
    corrections: dict[str, int] = {}
    for point in sorted(windowed, key=lambda p: p.timestamp):
        by_target.setdefault(point.target, []).append(point)
        by_series.setdefault((point.metric, point.target), []).append(point)
        if point.metric == USER_CORRECTION:
            corrections[point.target] = corrections.get(point.target, 0) + max(
                point.samples, 1
            )

    # Quality series that can carry a headroom measurement.
    quality_series = {
        key: pts
        for key, pts in by_series.items()
        if key[0] in metric_types and key[0] != USER_CORRECTION
    }
    aggregates = {
        key: summarize(key[0], key[1], pts) for key, pts in quality_series.items()
    }

    # Targets whose only signal is user corrections still deserve a rank.
    covered = {target for _, target in quality_series}
    correction_only = sorted(t for t in corrections if t not in covered)

    usage_pool = [agg.samples for agg in aggregates.values()]
    usage_pool += [corrections[t] for t in correction_only]
    max_samples = max(usage_pool) if usage_pool else 0

    entries: list[TriageEntry] = []

    for (metric, target), aggregate in aggregates.items():
        entries.append(
            _score_quality_series(
                metric=metric,
                target=target,
                aggregate=aggregate,
                series=quality_series[(metric, target)],
                corrections=corrections.get(target, 0),
                max_samples=max_samples,
                by_target=by_target,
                known_types=known_types,
                metric_types=metric_types,
                config=config,
            )
        )

    for target in correction_only:
        entries.append(
            _score_correction_only(
                target=target,
                corrections=corrections[target],
                max_samples=max_samples,
                by_target=by_target,
                known_types=known_types,
                metric_types=metric_types,
                config=config,
            )
        )

    kept = [e for e in entries if e.triggered or e.score >= config.min_score]
    kept.sort(key=lambda e: (-e.score, not e.triggered, e.target, e.metric))
    return kept


def _rate_interval(value: float, samples: int, config: "TriageConfig") -> Interval:
    """Wilson interval on a rate reconstructed from its mean and sample count.

    Metric points carry a rate and how many observations it summarizes rather
    than raw successes, so the success count is recovered as ``value *
    samples``. That is exact whenever the rate really was a proportion over
    those samples, which is what the four tracked metrics are.
    """
    if samples <= 0:
        return Interval(value, 0.0, 1.0, estimable=False)
    bounded = min(max(value, 0.0), 1.0)
    successes = int(round(bounded * samples))
    return wilson_interval(successes, samples)


def _score_quality_series(
    *,
    metric: str,
    target: str,
    aggregate: Aggregate,
    series: Sequence[MetricPoint],
    corrections: int,
    max_samples: int,
    by_target: dict[str, list[MetricPoint]],
    known_types: dict[str, TargetType],
    metric_types: dict,
    config: TriageConfig,
) -> TriageEntry:
    value = aggregate.weighted_mean
    headroom = max(0.0, min(1.0, config.ceiling - value))
    failure_rate = max(0.0, min(1.0, 1.0 - value))
    usage_weight = _usage_weight(aggregate.samples, max_samples, config)

    factors = [
        ScoreFactor(
            "potential improvement",
            headroom,
            f"{metric} sits at {value:.2f}, leaving {headroom:.2f} to {config.ceiling:.2f}",
        ),
        ScoreFactor(
            "usage frequency",
            usage_weight,
            f"{aggregate.samples} observations in window "
            f"({aggregate.count} reports, busiest target has {max_samples})",
        ),
    ]
    score = headroom * usage_weight

    trend = compute_trend(
        series,
        metric=metric,
        target=target,
        min_points=config.min_points_for_trend,
        flat_tolerance=config.flat_tolerance,
        significant_change=config.significant_change,
        alpha=config.trend_alpha,
        confidence=config.trend_confidence,
    )
    if trend.significant:
        multiplier = 1.0 + config.decline_weight
        score *= multiplier
        factors.append(ScoreFactor("declining trend", multiplier, trend.describe()))

    pressure = _correction_pressure(corrections, config)
    if pressure > 0:
        multiplier = 1.0 + config.correction_weight * pressure
        score *= multiplier
        factors.append(
            ScoreFactor(
                "user corrections",
                multiplier,
                f"{corrections} correction(s) logged against this target",
            )
        )

    triggered = False
    reason = ""
    # The trigger is the one decision in this module that spends money, so it
    # asks for evidence rather than a point estimate. A count floor
    # (min_samples) says how much data arrived, not how much of the shortfall
    # is real: 21 successes in 30 sits at 0.70 with a Wilson interval reaching
    # 0.83, so the "30% headroom" that clears a 30% threshold is equally
    # consistent with 17%. Requiring the interval's upper bound to stay bad
    # enough means the run only fires when the shortfall survives its own
    # uncertainty.
    rate_interval = _rate_interval(value, aggregate.samples, config)
    worst_tolerable = config.ceiling - config.failure_threshold
    if (
        aggregate.samples >= config.min_samples
        and headroom >= config.failure_threshold
        and rate_interval.high <= worst_tolerable + 1e-9
    ):
        triggered = True
        # The condition tests headroom, so the sentence says headroom. The two
        # are the same number only at a ceiling of 1.0: against a 0.8 ceiling,
        # calling the shortfall a "failure rate" understates the real one by
        # twenty points. Both are reported, each under its own name.
        reason = (
            f"headroom {headroom:.0%} to the {config.ceiling:.2f} ceiling is at or "
            f"above threshold {config.failure_threshold:.0%} over "
            f"{aggregate.samples} observations (failure rate {failure_rate:.0%}, "
            f"rate {rate_interval.describe()})"
        )
    elif trend.significant and aggregate.samples >= config.min_samples:
        triggered = True
        reason = f"significant decline: {trend.describe()}"

    return TriageEntry(
        target=target,
        target_type=_resolve_target_type(
            target, metric, by_target, known_types, metric_types
        ),
        metric=metric,
        score=score,
        potential_improvement=headroom,
        usage_weight=usage_weight,
        usage_samples=aggregate.samples,
        current_value=value,
        observations=aggregate.count,
        corrections=corrections,
        trend=trend,
        triggered=triggered,
        trigger_reason=reason,
        factors=factors,
        trend_p_value=trend.p_value if trend.fitted else None,
        trend_r_squared=trend.r_squared if trend.fitted else None,
    )


def _score_correction_only(
    *,
    target: str,
    corrections: int,
    max_samples: int,
    by_target: dict[str, list[MetricPoint]],
    known_types: dict[str, TargetType],
    metric_types: dict,
    config: TriageConfig,
) -> TriageEntry:
    """Rank a target whose only evidence is the user correcting the agent.

    There is no rate to measure headroom against, so the correction pressure
    stands in for potential improvement. It is a weaker signal than a measured
    success rate, which is why it saturates rather than scaling without bound.
    """
    pressure = _correction_pressure(corrections, config)
    usage_weight = _usage_weight(corrections, max_samples, config)
    score = pressure * usage_weight

    factors = [
        ScoreFactor(
            "potential improvement",
            pressure,
            f"{corrections} user correction(s), no measured success rate for this target",
        ),
        ScoreFactor(
            "usage frequency",
            usage_weight,
            f"{corrections} correction(s) against a busiest-target count of {max_samples}",
        ),
    ]

    triggered = corrections >= config.correction_trigger_count
    reason = (
        f"{corrections} user corrections at or above trigger count "
        f"{config.correction_trigger_count}"
        if triggered
        else ""
    )

    return TriageEntry(
        target=target,
        target_type=_resolve_target_type(
            target, USER_CORRECTION, by_target, known_types, metric_types
        ),
        metric=USER_CORRECTION,
        score=score,
        potential_improvement=pressure,
        usage_weight=usage_weight,
        usage_samples=corrections,
        current_value=None,
        observations=corrections,
        corrections=corrections,
        trend=None,
        triggered=triggered,
        trigger_reason=reason,
        factors=factors,
    )


@dataclass
class AutoTriage:
    """Ranks optimization targets out of a :class:`MetricStore`."""

    store: MetricStore
    config: TriageConfig = field(default_factory=TriageConfig)

    def rank(
        self,
        now: Optional[float] = None,
        limit: Optional[int] = None,
        actionable_only: bool = False,
    ) -> list[TriageEntry]:
        """Rank triage entries worst first."""
        end = self.store.now() if now is None else now
        entries = rank_points(self.store.load(), self.config, now=end)
        if actionable_only:
            entries = [e for e in entries if e.actionable]
        return entries[:limit] if limit else entries

    def triggers(self, now: Optional[float] = None) -> list[TriageEntry]:
        """Only the entries that crossed a threshold, in rank order."""
        return [e for e in self.rank(now=now) if e.triggered]

    def declining(self, now: Optional[float] = None) -> list[TriageEntry]:
        """Entries whose series is measurably getting worse."""
        return [
            e
            for e in self.rank(now=now)
            if e.trend is not None and e.trend.is_deterioration
        ]

"""Durable performance history for the continuous self-improvement loop.

Phases 1-4 are one-shot tools: a human picks a target, runs an optimizer, reads
a diff. Phase 5 only works if the pipeline can answer "which target is worth a
run this week?" on its own, and that question is unanswerable without a memory
of how things have been going. This module is that memory.

PLAN.md names four signals to track:

    skill_success_rate         per-skill success, mined from real sessions
    tool_selection_accuracy    did the agent reach for the right tool
    benchmark_score            periodic benchmark runs, scored over time
    user_correction            "no, use X instead" is a labelled failure

They share one shape - a value attached to a named target at a moment in time -
so they share one store: a JSONL file that is only ever appended to. Append-only
matters more than it sounds. A monitor that rewrites its own history can quietly
erase the evidence that a regression happened, and the loop's whole claim to
usefulness rests on that evidence being trustworthy. Rotation exists
(:meth:`MetricStore.archive_before`) but it moves old points to a sibling file
rather than dropping them.

Two design rules the tests depend on:

1. **The clock is injected.** Every function that needs "now" takes it as an
   argument, defaulting to the store's clock. Nothing calls ``datetime.now()``
   from inside logic worth testing, so trend detection is reproducible.
2. **Nothing is inferred that was not measured.** An absent benchmark records no
   point at all rather than a zero. A zero would read as "the agent failed every
   task", which is a very different claim from "we did not run".

**A trend is a claim about evidence, so it is tested like one.** This module
used to call a decline "significant" whenever the modelled change exceeded a
fixed magnitude. That is not a significance test, and it does not survive
contact with a noisy series: the oscillation [0.90, 0.35, 0.85, 0.30, 0.88,
0.33, 0.60] has no trend at all, yet a magnitude rule reports a significant
decline and fires an optimization run on nothing. :func:`compute_trend` now
fits the series with :func:`evolution.core.stats.ols_trend` and runs a real
t-test on the slope, so the p-value, R2, standard error, and the confidence
interval on the slope all travel with the answer. Direction stays
magnitude-based, because direction is a description of the movement rather than
a claim that the movement is real.

Pure local I/O throughout: no network, no model, no hermes-agent checkout.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence, Union

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None
try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - POSIX has no msvcrt
    msvcrt = None

from evolution.core.stats import Interval, ols_trend

__all__ = [
    "SECONDS_PER_DAY",
    "SKILL_SUCCESS_RATE",
    "TOOL_SELECTION_ACCURACY",
    "BENCHMARK_SCORE",
    "USER_CORRECTION",
    "OPTIMIZATION_RUN",
    "TRACKED_METRICS",
    "HIGHER_IS_BETTER",
    "utc_now",
    "MetricPoint",
    "MetricStore",
    "Aggregate",
    "Trend",
    "TrendDirection",
    "compute_trend",
]

SECONDS_PER_DAY = 86_400.0

# The four signals PLAN.md Phase 5 lists under "Performance monitor".
SKILL_SUCCESS_RATE = "skill_success_rate"
TOOL_SELECTION_ACCURACY = "tool_selection_accuracy"
BENCHMARK_SCORE = "benchmark_score"
USER_CORRECTION = "user_correction"

# Bookkeeping the loop writes about itself, so a later cycle can see that a
# target was already optimized and does not need optimizing again this week.
OPTIMIZATION_RUN = "optimization_run"

TRACKED_METRICS: tuple[str, ...] = (
    SKILL_SUCCESS_RATE,
    TOOL_SELECTION_ACCURACY,
    BENCHMARK_SCORE,
    USER_CORRECTION,
)

# Direction of goodness per metric. Corrections are the odd one out: more of
# them is worse, so a rising correction count is a deterioration, not progress.
HIGHER_IS_BETTER: dict[str, bool] = {
    SKILL_SUCCESS_RATE: True,
    TOOL_SELECTION_ACCURACY: True,
    BENCHMARK_SCORE: True,
    USER_CORRECTION: False,
    OPTIMIZATION_RUN: True,
}

Selector = Optional[Union[str, Iterable[str]]]


def utc_now() -> float:
    """Current UTC time as unix seconds.

    The only clock reader in this module. Everything else takes a timestamp or
    a clock callable, which is what keeps the tests deterministic.
    """
    return datetime.now(timezone.utc).timestamp()


def _as_set(value: Selector) -> Optional[set[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


@contextlib.contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Hold an exclusive lock over every writer of *path* for the block.

    Appending one line at a time is safe on its own, but rotation is not: it
    reads the whole file, writes the survivors somewhere else, and replaces the
    original. A point appended between that read and that replace would be
    thrown away by a snapshot taken before it existed. Every writer takes this
    lock so rotation and appends cannot interleave.

    The lock lives in a sibling ``<name>.lock`` file and is held by the kernel,
    so it is released even if the process is killed mid-rotation - no stale
    lock file can wedge the next run. ``fcntl`` covers POSIX and ``msvcrt``
    covers Windows; on a filesystem that supports neither, the block still runs
    unserialised rather than failing, because a monitor that cannot write its
    history is worse than one that races on it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    try:
        handle = lock_path.open("a+b")
    except OSError:
        yield
        return

    locked = False
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        elif msvcrt is not None:  # pragma: no cover - exercised on Windows
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            locked = True
    except OSError:
        locked = False

    try:
        yield
    finally:
        if locked:
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                else:  # pragma: no cover - exercised on Windows
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        handle.close()


# ──────────────────────────────────────────────────────────────────────────
# Points
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class MetricPoint:
    """One observation: a value for a named target at a moment in time.

    *samples* is how many underlying observations the value summarizes. A 50%
    success rate over two sessions and a 50% success rate over five hundred are
    the same number carrying wildly different weight, and triage ranks by usage
    frequency, so the count has to travel with the value.
    """

    metric: str
    target: str
    value: float
    timestamp: float
    source: str = "unknown"
    samples: int = 1
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metric = str(self.metric).strip()
        self.target = str(self.target).strip()
        if not self.metric:
            raise ValueError("MetricPoint.metric must be a non-empty name")
        if not self.target:
            raise ValueError("MetricPoint.target must be a non-empty name")
        self.value = float(self.value)
        self.timestamp = float(self.timestamp)
        # float() happily accepts NaN and the infinities, and every one of them
        # does damage further down. json.dumps writes a NaN as bare ``NaN``,
        # which no other reader of this JSONL file will accept; one NaN value
        # drags the mean, minimum and maximum of every window it lands in; and
        # an infinite timestamp raises inside ``when`` before the point can even
        # be serialised. A store's job is to refuse this at the door.
        if not math.isfinite(self.value):
            raise ValueError("MetricPoint.value must be a finite number")
        if not math.isfinite(self.timestamp):
            raise ValueError("MetricPoint.timestamp must be a finite number")
        # A fractional count is a caller bug and int() would round it away in
        # silence: int(1.9) is 1, and int(-0.5) is 0, which walks a negative
        # count straight past the check below. Numeric strings still coerce,
        # because reading "4" out of a stored record has always worked.
        counted = int(self.samples)
        if not isinstance(self.samples, str) and counted != self.samples:
            raise ValueError("MetricPoint.samples must be a whole number of observations")
        self.samples = counted
        if self.samples < 0:
            raise ValueError("MetricPoint.samples cannot be negative")
        if self.metadata is None:
            self.metadata = {}

    @property
    def when(self) -> datetime:
        """The timestamp as an aware UTC datetime."""
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

    @property
    def higher_is_better(self) -> bool:
        """Whether a rise in this metric is an improvement."""
        return HIGHER_IS_BETTER.get(self.metric, True)

    def to_dict(self) -> dict:
        """Serialise the point for the JSONL store."""
        record = {
            "metric": self.metric,
            "target": self.target,
            "value": self.value,
            "timestamp": self.timestamp,
            # Redundant with timestamp, kept because a human tailing this file
            # should be able to read it without doing epoch arithmetic.
            "at": self.when.isoformat(),
            "source": self.source,
            "samples": self.samples,
        }
        if self.metadata:
            record["metadata"] = self.metadata
        return record

    def to_json_line(self) -> str:
        """Serialise to one JSONL line, key-sorted so diffs stay stable."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, blob: dict) -> "MetricPoint":
        """Rebuild from a stored dict, accepting the older 'at' timestamp field."""
        # A line holding a valid JSON array, scalar or null parses without
        # complaint and then dies on .get with an AttributeError, which
        # MetricStore.load does not catch. One such line would abort the whole
        # read rather than cost a single skipped_lines increment, so the shape
        # is checked here where the error can be the documented kind.
        if not isinstance(blob, dict):
            raise TypeError("a metric record must be a JSON object")
        timestamp = blob.get("timestamp")
        if timestamp is None:
            at = blob.get("at")
            if not at:
                raise ValueError("record has neither 'timestamp' nor 'at'")
            timestamp = datetime.fromisoformat(at).timestamp()
        return cls(
            metric=blob["metric"],
            target=blob["target"],
            value=blob.get("value", 0.0),
            timestamp=float(timestamp),
            source=blob.get("source", "unknown"),
            # Passed through raw. int() here would truncate a fractional stored
            # count before __post_init__ ever got the chance to refuse it.
            samples=blob.get("samples", 1),
            metadata=dict(blob.get("metadata") or {}),
        )

    @classmethod
    def from_json_line(cls, line: str) -> "MetricPoint":
        """Rebuild from one JSONL line."""
        return cls.from_dict(json.loads(line))


# ──────────────────────────────────────────────────────────────────────────
# Trends
# ──────────────────────────────────────────────────────────────────────────


class TrendDirection(str, Enum):
    """Which way a metric is moving, or UNKNOWN when there is too little data."""
    RISING = "rising"
    FLAT = "flat"
    DECLINING = "declining"
    UNKNOWN = "unknown"


@dataclass
class Trend:
    """Where a series is heading, whether that is real, and whether it matters.

    ``direction`` describes the *value*: rising means the number went up, which
    is good news for a success rate and bad news for a correction count. It is
    magnitude-based on purpose. Direction is a description of the movement, not
    a claim that the movement is distinguishable from noise.

    ``significant`` is the claim, and it takes three things at once:

    1. the fitted slope differs from zero at *alpha* - a t-test on the OLS
       slope, carrying ``p_value``, ``stderr``, ``r_squared`` and ``slope_ci``,
    2. the movement is a deterioration for this metric, and
    3. the modelled ``change`` clears ``practical_threshold``.

    Statistical and practical significance answer different questions, and a
    monitor needs both answers. Firing on statistics alone chases a two-point
    drift that happens to be tidy; firing on magnitude alone chases noise, which
    is the defect this replaced. An optimization run costs money, so it takes
    evidence that the decline is real *and* that it is big enough to be worth a
    cycle.
    """

    metric: str
    target: str
    direction: TrendDirection
    slope_per_day: float
    change: float
    first_value: Optional[float]
    last_value: Optional[float]
    n: int
    span_days: float
    higher_is_better: bool = True
    significant: bool = False
    note: str = ""
    # Inference, from the least-squares fit. Defaults describe "no evidence":
    # a p-value of 1.0 and no explained variance, which is the right answer for
    # a series too short to fit.
    p_value: float = 1.0
    r_squared: float = 0.0
    stderr: float = 0.0
    slope_ci: Optional[Interval] = None
    alpha: float = 0.05
    # The practical floor the change must clear, i.e. the ``significant_change``
    # argument :func:`compute_trend` was called with.
    practical_threshold: float = 0.05

    @property
    def is_deterioration(self) -> bool:
        """True when the trend moves the wrong way for this metric."""
        if self.direction is TrendDirection.DECLINING:
            return self.higher_is_better
        if self.direction is TrendDirection.RISING:
            return not self.higher_is_better
        return False

    @property
    def statistically_significant(self) -> bool:
        """True when the slope is distinguishable from zero at *alpha*.

        Needs three points before it can be True at all: two points define a
        line exactly, leaving no residual scatter to test the slope against.
        """
        return self.n >= 3 and self.p_value < self.alpha

    @property
    def practically_significant(self) -> bool:
        """True when the modelled change is big enough to be worth acting on."""
        return abs(self.change) >= self.practical_threshold

    @property
    def fitted(self) -> bool:
        """True when there was enough history to fit a slope and test it.

        Below three points, or with every reading at the same instant, the
        p-value and R2 carry no information, so callers should show nothing
        rather than show 1.00 and let a reader mistake it for a measurement.
        """
        return (
            self.n >= 3
            and self.direction is not TrendDirection.UNKNOWN
            and self.slope_ci is not None
        )

    def confidence_note(self) -> str:
        """The uncertainty in one short phrase, for tables and one-liners."""
        if not self.fitted:
            return ""
        return f"p={self.p_value:.3f}, R²={self.r_squared:.2f}"

    def describe(self) -> str:
        """Direction, change, slope, sample size and fit, or why it is unknown."""
        if self.direction is TrendDirection.UNKNOWN:
            return self.note or f"not enough history ({self.n} points)"
        line = (
            f"{self.direction.value} {self.change:+.3f} over {self.span_days:.1f}d "
            f"({self.slope_per_day:+.4f}/day, n={self.n}, "
            f"R²={self.r_squared:.2f}, p={self.p_value:.3f})"
        )
        if self.slope_ci is not None:
            line += (
                f" {self.slope_ci.confidence:.0%} CI on slope "
                f"[{self.slope_ci.low:+.4f}, {self.slope_ci.high:+.4f}]/day"
            )
        return line

    def to_dict(self) -> dict:
        """Serialise the trend for the run artifacts."""
        return {
            "metric": self.metric,
            "target": self.target,
            "direction": self.direction.value,
            "slope_per_day": self.slope_per_day,
            "change": self.change,
            "first_value": self.first_value,
            "last_value": self.last_value,
            "n": self.n,
            "span_days": round(self.span_days, 4),
            "higher_is_better": self.higher_is_better,
            "significant": self.significant,
            "note": self.note,
            "p_value": round(self.p_value, 6),
            "r_squared": round(self.r_squared, 6),
            "stderr": round(self.stderr, 8),
            "slope_ci": self.slope_ci.to_dict() if self.slope_ci else None,
            "alpha": self.alpha,
            "practical_threshold": self.practical_threshold,
            "statistically_significant": self.statistically_significant,
            "practically_significant": self.practically_significant,
            "fitted": self.fitted,
        }


def compute_trend(
    points: Sequence[MetricPoint],
    *,
    metric: str = "",
    target: str = "",
    higher_is_better: Optional[bool] = None,
    min_points: int = 3,
    flat_tolerance: float = 0.02,
    significant_change: float = 0.05,
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> Trend:
    """Least-squares trend over *points*, expressed as change per day.

    The slope is fitted against time in days rather than sample index, so an
    irregular reporting cadence does not distort it. ``change`` is the modelled
    movement across the observed span, which is the number a human actually
    cares about: "this skill lost nine points of success rate in three weeks".

    Fewer than *min_points* observations yields ``UNKNOWN`` rather than a guess.
    Two noisy readings are not a trend, and acting on them would burn an
    optimization cycle on nothing.

    The fit comes from :func:`evolution.core.stats.ols_trend`, which attaches a
    t-test on the slope. *alpha* is the level that test is read at and
    *significant_change* is the practical floor the modelled change must clear;
    :attr:`Trend.significant` requires both, plus the movement being in the bad
    direction for this metric. *flat_tolerance* still governs the descriptive
    rising / flat / declining label only.
    """
    ordered = sorted(points, key=lambda p: p.timestamp)
    metric = metric or (ordered[0].metric if ordered else "")
    target = target or (ordered[0].target if ordered else "")
    if higher_is_better is None:
        higher_is_better = HIGHER_IS_BETTER.get(metric, True)

    if len(ordered) < max(2, min_points):
        return Trend(
            metric=metric,
            target=target,
            direction=TrendDirection.UNKNOWN,
            slope_per_day=0.0,
            change=0.0,
            first_value=ordered[0].value if ordered else None,
            last_value=ordered[-1].value if ordered else None,
            n=len(ordered),
            span_days=0.0,
            higher_is_better=higher_is_better,
            note=f"need {max(2, min_points)} points, have {len(ordered)}",
            alpha=alpha,
            practical_threshold=significant_change,
        )

    t0 = ordered[0].timestamp
    xs = [(p.timestamp - t0) / SECONDS_PER_DAY for p in ordered]
    ys = [p.value for p in ordered]
    span_days = xs[-1] - xs[0]

    # The descriptive slope is computed here rather than taken from the fit so
    # that the degenerate cases keep behaving as they always have: a two-point
    # series still reports the line through both points, and a series with one
    # shared timestamp still reports its raw movement. The fit below supplies
    # the uncertainty, and it declines to answer in exactly those cases.
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator > 0:
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
        change = slope * span_days
    else:
        # Every point shares a timestamp. There is no rate to fit, but the
        # values still moved, so report the raw difference and no slope.
        slope = 0.0
        change = ys[-1] - ys[0]

    if change > flat_tolerance:
        direction = TrendDirection.RISING
    elif change < -flat_tolerance:
        direction = TrendDirection.DECLINING
    else:
        direction = TrendDirection.FLAT

    fit = ols_trend(xs, ys, alpha=alpha, confidence=confidence)

    trend = Trend(
        metric=metric,
        target=target,
        direction=direction,
        slope_per_day=slope,
        change=change,
        first_value=ys[0],
        last_value=ys[-1],
        n=len(ordered),
        span_days=span_days,
        higher_is_better=higher_is_better,
        p_value=fit.p_value,
        r_squared=fit.r_squared,
        stderr=fit.stderr,
        # With no spread on the time axis there is no slope to put an interval
        # around, and a zero-width interval would read as perfect precision.
        slope_ci=fit.slope_ci if denominator > 0 else None,
        alpha=alpha,
        practical_threshold=significant_change,
    )
    # Three conditions, all required. Dropping the t-test is how a pure
    # oscillation gets reported as a significant decline; dropping the
    # magnitude floor is how a real but trivial drift burns a cycle.
    trend.significant = (
        trend.statistically_significant
        and trend.is_deterioration
        and trend.practically_significant
    )
    return trend


# ──────────────────────────────────────────────────────────────────────────
# Aggregates
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Aggregate:
    """Summary of one (metric, target) series over a window."""

    metric: str
    target: str
    count: int = 0
    samples: int = 0
    mean: float = 0.0
    weighted_mean: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    total: float = 0.0
    first_timestamp: Optional[float] = None
    last_timestamp: Optional[float] = None
    last_value: Optional[float] = None
    sources: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        """True when no points fell in the window."""
        return self.count == 0

    def to_dict(self) -> dict:
        """Serialise the aggregate."""
        return {
            "metric": self.metric,
            "target": self.target,
            "count": self.count,
            "samples": self.samples,
            "mean": self.mean,
            "weighted_mean": self.weighted_mean,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "total": self.total,
            # Without these two an artifact reader can see what the window said
            # but not which window it was, so the summary cannot be checked
            # against the history it came from.
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "last_value": self.last_value,
            "sources": list(self.sources),
        }


def summarize(metric: str, target: str, points: Sequence[MetricPoint]) -> Aggregate:
    """Fold *points* into an :class:`Aggregate`.

    ``weighted_mean`` weights each point by its sample count, so a rate computed
    over four hundred sessions is not outvoted by a rate computed over three.
    """
    if not points:
        return Aggregate(metric=metric, target=target)

    ordered = sorted(points, key=lambda p: p.timestamp)
    values = [p.value for p in ordered]
    samples = sum(p.samples for p in ordered)
    weight_total = sum(max(p.samples, 0) for p in ordered)
    if weight_total > 0:
        weighted = sum(p.value * max(p.samples, 0) for p in ordered) / weight_total
    else:
        weighted = sum(values) / len(values)

    return Aggregate(
        metric=metric,
        target=target,
        count=len(ordered),
        samples=samples,
        mean=sum(values) / len(values),
        weighted_mean=weighted,
        minimum=min(values),
        maximum=max(values),
        total=sum(values),
        first_timestamp=ordered[0].timestamp,
        last_timestamp=ordered[-1].timestamp,
        last_value=ordered[-1].value,
        sources=tuple(sorted({p.source for p in ordered})),
    )


# ──────────────────────────────────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────────────────────────────────


class MetricStore:
    """Append-only JSONL history of performance observations.

    One JSON object per line, appended and flushed immediately, so a cycle that
    dies halfway through still leaves every point it had already written. A line
    that fails to parse is skipped rather than fatal - a truncated final line
    from a killed process should not blind the monitor to the year of history
    above it - and the count lands in :attr:`skipped_lines` so the damage is
    visible instead of silent.
    """

    def __init__(
        self,
        path: Union[str, Path],
        clock: Callable[[], float] = utc_now,
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self.skipped_lines = 0

    # -- clock ------------------------------------------------------------

    def now(self) -> float:
        """Current time from the injected clock, so tests can control it."""
        return float(self._clock())

    # -- writing ----------------------------------------------------------

    def append(self, point: MetricPoint) -> MetricPoint:
        """Append one point and return it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(self.path):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(point.to_json_line() + "\n")
                handle.flush()
        return point

    def extend(self, points: Iterable[MetricPoint]) -> list[MetricPoint]:
        """Append points to the store and return the ones written."""
        written: list[MetricPoint] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(self.path):
            with self.path.open("a", encoding="utf-8") as handle:
                for point in points:
                    handle.write(point.to_json_line() + "\n")
                    written.append(point)
                handle.flush()
        return written

    def record(
        self,
        metric: str,
        target: str,
        value: float,
        *,
        source: str = "unknown",
        samples: int = 1,
        timestamp: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> MetricPoint:
        """Build a point from the store's clock and append it.

        An explicit *timestamp* always wins, which is how backfilled history and
        deterministic tests get written.
        """
        point = MetricPoint(
            metric=metric,
            target=target,
            value=value,
            timestamp=self.now() if timestamp is None else timestamp,
            source=source,
            samples=samples,
            metadata=dict(metadata or {}),
        )
        return self.append(point)

    # -- reading ----------------------------------------------------------

    def load(self) -> list[MetricPoint]:
        """Every point in file order. Missing file reads as empty history."""
        self.skipped_lines = 0
        if not self.path.exists():
            return []

        points: list[MetricPoint] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    points.append(MetricPoint.from_json_line(line))
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    self.skipped_lines += 1
        return points

    def query(
        self,
        metric: Selector = None,
        target: Selector = None,
        source: Selector = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> list[MetricPoint]:
        """Points matching every supplied filter, oldest first.

        *since* and *until* are both inclusive.
        """
        metrics = _as_set(metric)
        targets = _as_set(target)
        sources = _as_set(source)

        selected = []
        for point in self.load():
            if metrics is not None and point.metric not in metrics:
                continue
            if targets is not None and point.target not in targets:
                continue
            if sources is not None and point.source not in sources:
                continue
            if since is not None and point.timestamp < since:
                continue
            if until is not None and point.timestamp > until:
                continue
            selected.append(point)
        selected.sort(key=lambda p: p.timestamp)
        return selected

    def window(
        self,
        days: float,
        *,
        now: Optional[float] = None,
        metric: Selector = None,
        target: Selector = None,
        source: Selector = None,
    ) -> list[MetricPoint]:
        """Points inside the trailing *days*-long window ending at *now*."""
        end = self.now() if now is None else now
        return self.query(
            metric=metric,
            target=target,
            source=source,
            since=end - days * SECONDS_PER_DAY,
            until=end,
        )

    def pairs(self, points: Optional[Sequence[MetricPoint]] = None) -> list[tuple[str, str]]:
        """Sorted (metric, target) pairs present in the history."""
        source = self.load() if points is None else points
        return sorted({(p.metric, p.target) for p in source})

    def targets(
        self,
        metric: Selector = None,
        points: Optional[Sequence[MetricPoint]] = None,
    ) -> list[str]:
        """Every target seen, optionally restricted to *metric*."""
        metrics = _as_set(metric)
        source = self.load() if points is None else points
        return sorted(
            {p.target for p in source if metrics is None or p.metric in metrics}
        )

    def latest(self, metric: str, target: str) -> Optional[MetricPoint]:
        """The most recent point for a metric and target, or None."""
        found = self.query(metric=metric, target=target)
        return found[-1] if found else None

    def summarize(
        self,
        metric: str,
        target: str,
        *,
        window_days: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Aggregate:
        """Aggregate one metric for one target, optionally over a trailing window."""
        if window_days is None:
            points = self.query(metric=metric, target=target)
        else:
            points = self.window(window_days, now=now, metric=metric, target=target)
        return summarize(metric, target, points)

    def trend(
        self,
        metric: str,
        target: str,
        *,
        window_days: Optional[float] = None,
        now: Optional[float] = None,
        min_points: int = 3,
        flat_tolerance: float = 0.02,
        significant_change: float = 0.05,
        alpha: float = 0.05,
        confidence: float = 0.95,
    ) -> Trend:
        """Direction and slope for a metric, UNKNOWN below *min_points*."""
        if window_days is None:
            points = self.query(metric=metric, target=target)
        else:
            points = self.window(window_days, now=now, metric=metric, target=target)
        return compute_trend(
            points,
            metric=metric,
            target=target,
            min_points=min_points,
            flat_tolerance=flat_tolerance,
            significant_change=significant_change,
            alpha=alpha,
            confidence=confidence,
        )

    # -- rotation ---------------------------------------------------------

    @property
    def archive_path(self) -> Path:
        """Where rotated history is written when the store is compacted."""
        return self.path.with_suffix(".archive" + self.path.suffix)

    def archive_before(self, cutoff: float) -> int:
        """Move points older than *cutoff* into the sibling archive file.

        Nothing is deleted. An unattended weekly job writing to one JSONL file
        forever needs somewhere for the old lines to go, but a monitor that
        discards its own evidence is worse than one that grows, so the old
        points are appended to ``<name>.archive.jsonl`` and the live file is
        rewritten atomically with what remains. Returns the number moved.

        The whole rotation runs under the writer lock, because the read here
        and the replace at the end are far apart and anything appended between
        them would be replaced away by a snapshot that never contained it. The
        archive is written before the live file is replaced, so a rotation that
        dies in between loses nothing - it leaves both copies holding the same
        points, and the count below makes re-running it safe rather than
        doubling them.
        """
        with _exclusive(self.path):
            points = self.load()
            if not points:
                return 0

            old = [p for p in points if p.timestamp < cutoff]
            if not old:
                return 0
            kept = [p for p in points if p.timestamp >= cutoff]

            self.archive_path.parent.mkdir(parents=True, exist_ok=True)
            # Counted, not a set: the same value measured twice at the same
            # instant is two real observations and both belong in the archive.
            # What must not happen is a retry archiving a point some earlier,
            # interrupted attempt already wrote.
            archived: Counter[str] = Counter()
            if self.archive_path.exists():
                with self.archive_path.open("r", encoding="utf-8") as handle:
                    archived = Counter(
                        stripped for line in handle if (stripped := line.strip())
                    )

            with self.archive_path.open("a", encoding="utf-8") as handle:
                for point in old:
                    line = point.to_json_line()
                    if archived[line]:
                        archived[line] -= 1
                        continue
                    handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            fd, tmp_name = tempfile.mkstemp(
                dir=str(self.path.parent), prefix=self.path.name, suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    for point in kept:
                        handle.write(point.to_json_line() + "\n")
                os.replace(tmp_name, self.path)
            except BaseException:
                Path(tmp_name).unlink(missing_ok=True)
                raise
            return len(old)

"""Stop one tool's description from stealing another tool's selections.

This is the part PLAN.md singles out as the hard part of Phase 2, and the
reason is arithmetic. Suppose ``search_files`` is picked correctly 60% of the
time and ``read_file`` 90%. Rewrite ``search_files`` to sound like the answer to
every question about files and it might climb to 85% while dragging
``read_file`` down to 70%. Aggregate accuracy went up. The agent got worse: it
now grep-scans files a user asked it to read.

Aggregate accuracy cannot see that, so it is not what decides. Every candidate
is evaluated over the whole catalogue at once, per-tool selection rates are
computed side by side, and a candidate is rejected when *any* individual tool
regresses beyond tolerance, however good the average looks. When a tool does
regress the confusion matrix says where its selections went, which is the
difference between "search_files got worse" and "search_files lost eleven
selections to read_file".

Tolerance is configurable because zero is right for a large dataset and cruel
for a small one: with eight examples per tool, one flipped answer is a 12.5%
swing that may be noise. :class:`CrossToolGuard` defaults to zero - no
regression at all, which is what PLAN.md asks for - and lets a caller loosen it
deliberately rather than by accident.

**Why the per-example outcome vectors are kept.** Baseline and candidate are
always evaluated on the identical example set, so every example produces a
matched pair of outcomes. Counts throw that away. Two counts cannot tell the
difference between a candidate that flipped four answers against you and four
in your favour and a candidate that changed nothing, and those have very
different evidence behind them. So :class:`ToolRate` carries the aligned
per-example outcome vector and the guard runs an exact paired McNemar test per
tool through :mod:`evolution.core.stats`, rather than comparing two rates to a
number.

**Power is reported, not hidden.** A tolerance the sample size could never have
detected is not being enforced by evidence, it is being asserted. Each tool
comparison exposes ``min_detectable_shift`` and ``underpowered``, the verdict
carries the list of underpowered tools, and :meth:`CrossToolVerdict.summary`
says so in the same sentence that reports the pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from evolution.core.gates import GateResult, GateStatus
from evolution.core.stats import (
    Interval,
    PairedBinary,
    chance_accuracy,
    compare_paired_binary,
    wilson_interval,
)
from evolution.tools.selection_eval import (
    NO_TOOL,
    SelectionOutcome,
    SelectionReport,
)

__all__ = [
    "DEFAULT_TOLERANCE",
    "DEFAULT_ALPHA",
    "DEFAULT_CONFIDENCE",
    "ConfusionMatrix",
    "ToolRate",
    "CrossToolReport",
    "ToolComparison",
    "ToolRegression",
    "ToolImprovement",
    "CrossToolVerdict",
    "CrossToolGuard",
    "align_outcomes",
    "paired_for_tool",
]

# PLAN.md: "No individual tool's selection rate regresses."
DEFAULT_TOLERANCE = 0.0

# Significance level for the per-tool paired test, and the interval width that
# goes with it. One-sided at alpha for "did this tool regress", two-sided at
# ``confidence`` for the reported interval.
DEFAULT_ALPHA = 0.05
DEFAULT_CONFIDENCE = 0.95

# Rates are ratios of small integers; this only guards float representation.
_EPSILON = 1e-9


# ──────────────────────────────────────────────────────────────────────────
# Matrices
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ConfusionMatrix:
    """What got picked, for each tool that should have been picked.

    ``counts[expected][predicted]`` is a tally, so the diagonal is correct
    selections and everything off it is a misroute. :data:`NO_TOOL` is a row and
    a column like any other tool, because "answered directly when it should have
    called something" and its inverse are exactly the failures a description
    rewrite causes.
    """

    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, expected: str, predicted: str, n: int = 1) -> None:
        """Add *n* observations of *expected* being answered with *predicted*."""
        row = self.counts.setdefault(expected, {})
        row[predicted] = row.get(predicted, 0) + n

    def row(self, expected: str) -> dict[str, int]:
        """Predictions recorded for one expected tool."""
        return dict(self.counts.get(expected, {}))

    def column(self, predicted: str) -> dict[str, int]:
        """Which tools lost selections to *predicted*, and how many."""
        return {
            expected: row[predicted]
            for expected, row in self.counts.items()
            if row.get(predicted) and expected != predicted
        }

    def total(self) -> int:
        """Every observation in the matrix."""
        return sum(sum(row.values()) for row in self.counts.values())

    def correct(self, expected: str) -> int:
        """How often *expected* was answered with itself."""
        return self.counts.get(expected, {}).get(expected, 0)

    def opportunities(self, expected: str) -> int:
        """How often *expected* was the right answer, however it was answered."""
        return sum(self.counts.get(expected, {}).values())

    def misroutes(self, expected: str) -> list[tuple[str, int]]:
        """Where *expected*'s selections went instead, worst first."""
        row = self.row(expected)
        row.pop(expected, None)
        return sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))

    def top_confusions(self, limit: int = 5) -> list[tuple[str, str, int]]:
        """The worst ``(expected, predicted, count)`` mix-ups across the board."""
        pairs = [
            (expected, predicted, count)
            for expected, row in self.counts.items()
            for predicted, count in row.items()
            if predicted != expected and count > 0
        ]
        pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
        return pairs[:limit]

    def to_dict(self) -> dict:
        """Serialise the confusion matrix, expected tool first."""
        return {expected: dict(row) for expected, row in sorted(self.counts.items())}


@dataclass(frozen=True)
class ToolRate:
    """How often one tool was chosen when it was the right answer.

    ``outcomes`` is the per-example record behind the counts, in dataset order,
    and ``example_keys`` names the example each entry came from. Keeping both is
    what makes a paired comparison possible at all: ``opportunities`` and
    ``correct`` can say 3/4 twice over without ever revealing whether the two
    runs got the same three right.
    """

    tool: str
    opportunities: int
    correct: int
    outcomes: tuple[bool, ...] = ()
    example_keys: tuple[str, ...] = ()

    @property
    def rate(self) -> float:
        """Correct selections over opportunities, 0.0 when there were none."""
        if self.opportunities <= 0:
            return 0.0
        return self.correct / self.opportunities

    @property
    def paired_ready(self) -> bool:
        """True when this row carries the per-example detail a pairing needs."""
        return bool(self.outcomes)

    def to_dict(self) -> dict:
        """Serialise one tool's selection rate."""
        return {
            "tool": self.tool,
            "opportunities": self.opportunities,
            "correct": self.correct,
            "rate": round(self.rate, 4),
            # Serialised so a reader can recompute every p-value in this report
            # from the artifact alone, without rerunning the evaluation. That
            # takes both halves: the outcomes, and the keys align_outcomes
            # pairs them by when two runs are not in the same order.
            "outcomes": [int(flag) for flag in self.outcomes],
            "example_keys": list(self.example_keys),
        }


def _example_keys(outcomes: Sequence[SelectionOutcome]) -> list[str]:
    """A stable identifier per outcome, disambiguating repeated task text.

    The same dataset evaluated twice produces the same keys in the same order,
    which is what lets a candidate row be matched to its baseline row by example
    rather than by position alone.
    """
    seen: dict[str, int] = {}
    keys: list[str] = []
    for outcome in outcomes:
        task = outcome.example.task
        count = seen.get(task, 0)
        seen[task] = count + 1
        keys.append(task if count == 0 else f"{task}#{count + 1}")
    return keys


@dataclass
class CrossToolReport:
    """Per-tool selection rates plus the confusion matrix behind them."""

    rates: dict[str, ToolRate] = field(default_factory=dict)
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    n: int = 0
    overall_accuracy: float = 0.0
    param_accuracy: float = 0.0
    combined_score: float = 0.0

    @classmethod
    def from_outcomes(
        cls,
        outcomes: Sequence[SelectionOutcome],
        tools: Optional[Iterable[str]] = None,
    ) -> "CrossToolReport":
        """Build a report from scored outcomes.

        *tools* seeds the rate table so a tool with zero examples still appears,
        with zero opportunities. A silently absent row is how a regression hides.

        The order of *outcomes* is preserved inside each tool's outcome vector.
        Evaluating baseline and candidate over the same example list therefore
        yields two reports whose vectors line up example for example.
        """
        confusion = ConfusionMatrix()
        tallies: dict[str, list[int]] = {}
        vectors: dict[str, list[bool]] = {}
        keys: dict[str, list[str]] = {}
        for name in tools or []:
            tallies.setdefault(name, [0, 0])
        tallies.setdefault(NO_TOOL, [0, 0])

        example_keys = _example_keys(outcomes)
        for outcome, key in zip(outcomes, example_keys):
            expected = outcome.expected_tool
            predicted = outcome.predicted_tool
            confusion.record(expected, predicted)
            counts = tallies.setdefault(expected, [0, 0])
            counts[0] += 1
            if outcome.tool_correct:
                counts[1] += 1
            vectors.setdefault(expected, []).append(bool(outcome.tool_correct))
            keys.setdefault(expected, []).append(key)

        report = SelectionReport(outcomes=list(outcomes))
        return cls(
            rates={
                name: ToolRate(
                    tool=name,
                    opportunities=counts[0],
                    correct=counts[1],
                    outcomes=tuple(vectors.get(name, ())),
                    example_keys=tuple(keys.get(name, ())),
                )
                for name, counts in sorted(tallies.items())
            },
            confusion=confusion,
            n=len(outcomes),
            overall_accuracy=report.tool_accuracy,
            param_accuracy=report.param_accuracy,
            combined_score=report.score,
        )

    @classmethod
    def from_report(
        cls,
        report: SelectionReport,
        tools: Optional[Iterable[str]] = None,
    ) -> "CrossToolReport":
        """Build from a :class:`SelectionReport`, optionally restricted to *tools*."""
        return cls.from_outcomes(report.outcomes, tools=tools)

    def rate(self, tool: str) -> float:
        """Selection accuracy for *tool*, or 0.0 when it was never the answer."""
        entry = self.rates.get(tool)
        return entry.rate if entry else 0.0

    def opportunities(self, tool: str) -> int:
        """How many examples *tool* was the correct answer for."""
        entry = self.rates.get(tool)
        return entry.opportunities if entry else 0

    def outcome_vector(self, tool: str) -> tuple[bool, ...]:
        """This tool's per-example correctness record, in dataset order."""
        entry = self.rates.get(tool)
        return entry.outcomes if entry else ()

    def example_keys(self, tool: str) -> tuple[str, ...]:
        """Keys of the examples *tool* was the correct answer for.

        These are what let a baseline and a candidate report be compared pairwise
        rather than as two independent rates.
        """
        entry = self.rates.get(tool)
        return entry.example_keys if entry else ()

    @property
    def measured_tools(self) -> list[str]:
        """Tools that actually had at least one chance to be selected."""
        return [name for name, entry in self.rates.items() if entry.opportunities > 0]

    @property
    def correct(self) -> int:
        """Total correct selections across every tool."""
        return sum(entry.correct for entry in self.rates.values())

    @property
    def num_options(self) -> int:
        """How many answers the selector could have given, counting 'none'.

        Every row of the rate table is a choice the selector could make, and
        :data:`NO_TOOL` is one of them.
        """
        return len(self.rates)

    @property
    def chance_accuracy(self) -> float:
        """What a selector picking uniformly at random would score."""
        return chance_accuracy(self.num_options)

    def accuracy_interval(self, confidence: float = DEFAULT_CONFIDENCE) -> Interval:
        """Wilson interval on overall selection accuracy."""
        return wilson_interval(self.correct, self.n, confidence)

    def describe_accuracy(self, confidence: float = DEFAULT_CONFIDENCE) -> str:
        """Accuracy with the two numbers that make it readable.

        A bare accuracy is uninterpretable. 40% is poor against two tools and
        excellent against thirty, and 40% of five examples is not a measurement
        at all, so the chance baseline and the interval travel with it.
        """
        interval = self.accuracy_interval(confidence)
        return (
            f"{self.overall_accuracy:.1%} [{interval.low:.1%}, {interval.high:.1%}] "
            f"over {self.n} example(s) vs {self.chance_accuracy:.1%} chance "
            f"across {self.num_options} option(s)"
        )

    def to_dict(self) -> dict:
        """Serialise the report with its per-tool rates and accuracy interval."""
        interval = self.accuracy_interval()
        return {
            "n": self.n,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "correct": self.correct,
            "num_options": self.num_options,
            "chance_accuracy": round(self.chance_accuracy, 4),
            "accuracy_interval": interval.to_dict(),
            "param_accuracy": round(self.param_accuracy, 4),
            "combined_score": round(self.combined_score, 4),
            "rates": {name: entry.to_dict() for name, entry in self.rates.items()},
            "confusion": self.confusion.to_dict(),
        }


# ──────────────────────────────────────────────────────────────────────────
# Pairing
# ──────────────────────────────────────────────────────────────────────────


def align_outcomes(
    baseline: CrossToolReport,
    candidate: CrossToolReport,
    tool: str,
) -> Optional[tuple[list[bool], list[bool]]]:
    """Line one tool's baseline outcomes up with its candidate outcomes.

    Matching is by example key, so a report whose examples were shuffled still
    pairs correctly. Falling back to an unpaired comparison is never the answer
    when the pairing is recoverable: the pairing is the evidence.

    Returns ``None`` only when there is nothing to pair - one side never
    measured the tool, or the two sides share no example. A tool that vanished
    from the candidate is not silently treated as a wash; the caller reports it
    as a comparison with no paired evidence, and the point estimate still
    decides.
    """
    left = baseline.rates.get(tool)
    right = candidate.rates.get(tool)
    if left is None or right is None:
        return None
    if not left.outcomes or not right.outcomes:
        return None

    if left.example_keys and right.example_keys:
        if left.example_keys == right.example_keys:
            return list(left.outcomes), list(right.outcomes)
        index = dict(zip(right.example_keys, right.outcomes))
        pairs = [
            (was, index[key])
            for key, was in zip(left.example_keys, left.outcomes)
            if key in index
        ]
        if not pairs:
            return None
        return [p[0] for p in pairs], [p[1] for p in pairs]

    # No keys recorded, which means the vectors were supplied directly by a
    # caller who is asserting they are in the same example order. Equal lengths
    # are the only check available; unequal lengths are a real misalignment.
    if len(left.outcomes) != len(right.outcomes):
        return None
    return list(left.outcomes), list(right.outcomes)


def paired_for_tool(
    baseline: CrossToolReport,
    candidate: CrossToolReport,
    tool: str,
    alpha: float = DEFAULT_ALPHA,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Optional[PairedBinary]:
    """Exact paired comparison for one tool, or ``None`` when it cannot be built."""
    aligned = align_outcomes(baseline, candidate, tool)
    if aligned is None:
        return None
    return compare_paired_binary(
        aligned[0], aligned[1], alpha=alpha, confidence=confidence
    )


# ──────────────────────────────────────────────────────────────────────────
# The guard
# ──────────────────────────────────────────────────────────────────────────


class _PairedEvidence:
    """Statistics shared by every per-tool record.

    Each record holds the :class:`~evolution.core.stats.PairedBinary` for its
    tool, or ``None`` when no pairing could be built. Everything derived from it
    degrades to "no evidence" rather than to a confident default: a p-value of
    1.0 would read as "tested and found innocent", which is not what an absent
    test means.
    """

    tool: str
    baseline_rate: float
    candidate_rate: float
    opportunities: int
    paired: Optional[PairedBinary]
    tolerance: float

    @property
    def delta(self) -> float:
        """The change this record's verdict is about.

        Taken from the paired comparison when there is one, so the delta, the
        interval and the p-value all describe the same set of examples. The
        unpaired rates come from each report as a whole, and the two populations
        are not always the same: if the candidate only managed to evaluate 2 of
        the baseline's 20 examples for a tool, the whole-report delta was
        +50% while the paired evidence covering those 2 examples said +0% with
        p = 1.000, and the headline sat outside its own confidence interval.

        Falls back to the report rates only when nothing could be paired, where
        an unpaired estimate is the sole thing available and is labelled as such
        by :attr:`unpaired`.
        """
        if self.paired is not None:
            return self.paired.delta
        return self.candidate_rate - self.baseline_rate

    @property
    def unpaired(self) -> bool:
        """True when no paired comparison backs this record's delta."""
        return self.paired is None

    @property
    def population_mismatch(self) -> bool:
        """True when the pairing covers fewer examples than the report claims.

        Not an error in the shipped Phase 2 path, where ``evaluate_selection``
        scores every example and the two runs always align. It matters for any
        caller that evaluates partially, because the unpaired rates then
        describe a different population from the test.
        """
        return self.paired is not None and self.paired.n != self.opportunities

    @property
    def unpaired_delta(self) -> float:
        """The whole-report rate difference, reported alongside but never gated on."""
        return self.candidate_rate - self.baseline_rate

    @property
    def breaches_tolerance(self) -> bool:
        """True when the point estimate alone falls past the tolerance."""
        return self.delta < -abs(self.tolerance) - _EPSILON

    @property
    def significant_regression(self) -> bool:
        """True when the paired test finds a regression, False when unpaired."""
        return bool(self.paired and self.paired.significant_regression)

    @property
    def significant_improvement(self) -> bool:
        """True when the paired test finds an improvement, False when unpaired."""
        return bool(self.paired and self.paired.significant_improvement)

    @property
    def regressed(self) -> bool:
        """A safety gate rejects on either kind of bad news.

        The point estimate catches a large drop the sample is too small to prove,
        and the test catches a small drop that is real. Requiring both would make
        the gate more permissive than it was before statistics arrived, which is
        the wrong direction for a gate.
        """
        return self.breaches_tolerance or self.significant_regression

    @property
    def p_worse(self) -> Optional[float]:
        """One-sided p for 'this tool regressed', or ``None`` with no pairing."""
        return self.paired.p_worse if self.paired else None

    @property
    def min_detectable_shift(self) -> Optional[float]:
        """Smallest shift this sample could detect, or None when unpaired."""
        return self.paired.min_detectable_shift() if self.paired else None

    @property
    def underpowered(self) -> bool:
        """True when this sample could not have detected the tolerance in force.

        A zero tolerance needs no warning: any drop at all breaches it on the
        point estimate, so the gate is already as strict as it can be. The
        warning is for a positive tolerance that no result on this many examples
        could ever have shown to be exceeded.
        """
        if self.paired is None or self.tolerance <= 0:
            return False
        return self.paired.underpowered_for(self.tolerance)

    def delta_interval(self) -> Optional[Interval]:
        """Confidence interval on the change, or None when the runs were not paired."""
        return self.paired.delta_interval() if self.paired else None

    def _evidence(self) -> str:
        """The statistics half of a description line."""
        if self.paired is None:
            return " [no paired evidence: the two runs share no example for this tool]"
        interval = self.paired.delta_interval()
        text = (
            f" [{interval.confidence:.0%} CI {interval.low:+.1%}, {interval.high:+.1%}; "
            f"p={self.paired.p_worse:.3f}; discordant={self.paired.discordant}]"
        )
        if self.underpowered:
            text += (
                f" [underpowered: {self.paired.n} example(s) could only ever show a "
                f"{self.paired.min_detectable_shift():.1%} shift, so the "
                f"{self.tolerance:.1%} tolerance is not enforced by evidence here]"
            )
        return text

    def _headline(self) -> str:
        return (
            f"{self.tool}: {self.baseline_rate:.1%} -> {self.candidate_rate:.1%} "
            f"({self.delta:+.1%} over {self.opportunities} example(s))"
        )

    def _stats_dict(self) -> dict:
        interval = self.delta_interval()
        return {
            "tolerance": self.tolerance,
            "breaches_tolerance": self.breaches_tolerance,
            "significant_regression": self.significant_regression,
            "significant_improvement": self.significant_improvement,
            "underpowered": self.underpowered,
            "p_worse": None if self.p_worse is None else round(self.p_worse, 6),
            "min_detectable_shift": (
                None
                if self.min_detectable_shift is None
                else round(self.min_detectable_shift, 6)
            ),
            "delta_ci": interval.to_dict() if interval else None,
            "paired": self.paired.to_dict() if self.paired else None,
        }


@dataclass(frozen=True)
class ToolComparison(_PairedEvidence):
    """One tool's full baseline-vs-candidate story, regressed or not.

    Every measured tool gets one of these, including the ones that held steady,
    because a tool that held steady on four examples and a tool that held steady
    on four hundred are not the same finding.
    """

    tool: str
    baseline_rate: float
    candidate_rate: float
    opportunities: int
    paired: Optional[PairedBinary] = None
    tolerance: float = DEFAULT_TOLERANCE
    stolen_by: dict[str, int] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        """True when the tool gained beyond floating-point noise."""
        return self.delta > _EPSILON

    def describe(self) -> str:
        """The tool, its move, and the evidence behind it."""
        return self._headline() + self._evidence()

    def to_dict(self) -> dict:
        """Serialise the improvement with its rates and evidence."""
        blob = {
            "tool": self.tool,
            "baseline_rate": round(self.baseline_rate, 4),
            "candidate_rate": round(self.candidate_rate, 4),
            "delta": round(self.delta, 4),
            "opportunities": self.opportunities,
            "n_paired": self.paired.n if self.paired else 0,
            "regressed": self.regressed,
            "improved": self.improved,
            # ToolRegression serialises this too; dropping it here lost the
            # misroute detail for every tool that did not regress.
            "stolen_by": dict(self.stolen_by),
        }
        blob.update(self._stats_dict())
        return blob


@dataclass(frozen=True)
class ToolRegression(_PairedEvidence):
    """One tool that got worse, and where its selections went."""

    tool: str
    baseline_rate: float
    candidate_rate: float
    opportunities: int
    stolen_by: dict[str, int] = field(default_factory=dict)
    paired: Optional[PairedBinary] = None
    tolerance: float = DEFAULT_TOLERANCE

    def describe(self) -> str:
        """The regression, plus which tools took the selections it lost."""
        text = self._headline() + self._evidence()
        if self.stolen_by:
            thief = ", ".join(
                f"{name} +{count}"
                for name, count in sorted(self.stolen_by.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            text += f"; lost to {thief}"
        return text

    def to_dict(self) -> dict:
        """Serialise the regression, including where its selections went."""
        blob = {
            "tool": self.tool,
            "baseline_rate": round(self.baseline_rate, 4),
            "candidate_rate": round(self.candidate_rate, 4),
            "delta": round(self.delta, 4),
            "opportunities": self.opportunities,
            "stolen_by": dict(self.stolen_by),
        }
        blob.update(self._stats_dict())
        return blob


@dataclass(frozen=True)
class ToolImprovement(_PairedEvidence):
    """One tool that got better."""

    tool: str
    baseline_rate: float
    candidate_rate: float
    opportunities: int
    paired: Optional[PairedBinary] = None
    tolerance: float = DEFAULT_TOLERANCE

    def describe(self) -> str:
        """The tool, its move, and the evidence behind it."""
        return (
            f"{self.tool}: {self.baseline_rate:.1%} -> {self.candidate_rate:.1%} "
            f"({self.delta:+.1%})" + self._evidence()
        )

    def to_dict(self) -> dict:
        """Serialise the per-tool comparison."""
        blob = {
            "tool": self.tool,
            "baseline_rate": round(self.baseline_rate, 4),
            "candidate_rate": round(self.candidate_rate, 4),
            "delta": round(self.delta, 4),
            "opportunities": self.opportunities,
        }
        blob.update(self._stats_dict())
        return blob


@dataclass
class CrossToolVerdict:
    """Accept or reject, with the per-tool evidence for the decision."""

    accepted: bool
    baseline_accuracy: float
    candidate_accuracy: float
    regressions: list[ToolRegression] = field(default_factory=list)
    improvements: list[ToolImprovement] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    tolerance: float = DEFAULT_TOLERANCE
    reason: str = ""
    comparisons: list[ToolComparison] = field(default_factory=list)
    underpowered: list[str] = field(default_factory=list)
    unpaired: list[str] = field(default_factory=list)
    baseline_interval: Optional[Interval] = None
    candidate_interval: Optional[Interval] = None
    chance_accuracy: float = 0.0
    num_options: int = 0

    @property
    def overall_delta(self) -> float:
        """Candidate accuracy minus baseline accuracy across all tools."""
        return self.candidate_accuracy - self.baseline_accuracy

    @property
    def significant_regressions(self) -> list[ToolRegression]:
        """Regressions the paired test called real, not just large."""
        return [r for r in self.regressions if r.significant_regression]

    def comparison(self, tool: str) -> Optional[ToolComparison]:
        """The per-tool comparison for *tool*, or None."""
        for entry in self.comparisons:
            if entry.tool == tool:
                return entry
        return None

    def power_note(self) -> str:
        """The sentence that stops a pass from reading as more than it is."""
        if not self.underpowered:
            return ""
        return (
            f"{len(self.underpowered)} tool(s) had too few examples for any result "
            f"to reach significance at a {self.tolerance:.1%} tolerance, so that "
            f"tolerance is not enforced by evidence for them "
            f"({', '.join(self.underpowered)})"
        )

    def accuracy_note(self) -> str:
        """Overall accuracy with its interval and the chance baseline."""
        if self.baseline_interval is None or self.candidate_interval is None:
            return ""
        return (
            f"accuracy {self.baseline_interval.describe()} -> "
            f"{self.candidate_interval.describe()} vs "
            f"{self.chance_accuracy:.1%} chance across {self.num_options} option(s)"
        )

    def summary(self) -> str:
        """One line: the verdict, the overall move, and the reason for it."""
        head = "accepted" if self.accepted else "REJECTED"
        text = (
            f"cross-tool {head}: overall {self.baseline_accuracy:.1%} -> "
            f"{self.candidate_accuracy:.1%} ({self.overall_delta:+.1%}); {self.reason}"
        )
        power = self.power_note()
        if power:
            text += f"; {power}"
        if self.unpaired:
            text += (
                f"; no paired evidence for {len(self.unpaired)} tool(s) "
                f"({', '.join(self.unpaired)})"
            )
        return text

    def to_gate_result(self) -> GateResult:
        """Express the verdict as a gate so it can join a GateChain."""
        details = [r.describe() for r in self.regressions]
        power = self.power_note()
        if power:
            details.append(power)
        return GateResult(
            name="cross_tool",
            status=GateStatus.PASSED if self.accepted else GateStatus.FAILED,
            message=self.reason,
            score=self.candidate_accuracy,
            baseline=self.baseline_accuracy,
            details="\n".join(details),
        )

    def to_dict(self) -> dict:
        """Serialise the verdict with every regression, improvement and ignored tool."""
        return {
            "accepted": self.accepted,
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "candidate_accuracy": round(self.candidate_accuracy, 4),
            "overall_delta": round(self.overall_delta, 4),
            "baseline_accuracy_interval": (
                self.baseline_interval.to_dict() if self.baseline_interval else None
            ),
            "candidate_accuracy_interval": (
                self.candidate_interval.to_dict() if self.candidate_interval else None
            ),
            "chance_accuracy": round(self.chance_accuracy, 4),
            "num_options": self.num_options,
            "tolerance": self.tolerance,
            "reason": self.reason,
            "regressions": [r.to_dict() for r in self.regressions],
            "improvements": [i.to_dict() for i in self.improvements],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "underpowered": list(self.underpowered),
            "unpaired": list(self.unpaired),
            "ignored": list(self.ignored),
        }


@dataclass
class CrossToolGuard:
    """Compare two cross-tool reports and decide whether to accept the candidate.

    ``tolerance`` is how far a single tool may fall before the candidate is
    rejected, as an absolute rate difference. Zero means no regression at all.
    A tool is rejected when the point estimate breaches the tolerance *or* the
    paired McNemar test finds the regression significant at ``alpha``.

    ``min_opportunities`` skips tools with too few examples to say anything;
    those tools are listed in the verdict's ``ignored`` field rather than being
    silently dropped, so nobody reads a pass as coverage it did not have.

    ``require_overall_improvement`` additionally demands the aggregate move up.
    Off by default: holding every tool steady while shortening 3,896 chars of
    description is a legitimate win.

    ``alpha`` and ``confidence`` set the significance level of the per-tool test
    and the width of the reported intervals.
    """

    tolerance: float = DEFAULT_TOLERANCE
    min_opportunities: int = 1
    require_overall_improvement: bool = False
    alpha: float = DEFAULT_ALPHA
    confidence: float = DEFAULT_CONFIDENCE

    def compare(
        self,
        baseline: CrossToolReport,
        candidate: CrossToolReport,
    ) -> CrossToolVerdict:
        """Compare baseline against candidate and decide whether to accept.

        A tool that regressed past tolerance rejects the candidate outright. A
        tool with no measurable opportunities is reported as ignored rather than
        counted as unchanged, so an untested tool cannot look like a safe one.
        """
        regressions: list[ToolRegression] = []
        improvements: list[ToolImprovement] = []
        comparisons: list[ToolComparison] = []
        ignored: list[str] = []
        underpowered: list[str] = []
        unpaired: list[str] = []

        tools = sorted(set(baseline.rates) | set(candidate.rates))
        for tool in tools:
            opportunities = baseline.opportunities(tool) or candidate.opportunities(tool)
            if opportunities < max(1, self.min_opportunities):
                if opportunities == 0:
                    ignored.append(tool)
                else:
                    ignored.append(f"{tool} ({opportunities} example(s))")
                continue

            before = baseline.rate(tool)
            after = candidate.rate(tool)
            paired = paired_for_tool(
                baseline, candidate, tool, alpha=self.alpha, confidence=self.confidence
            )
            stolen_by = self._stolen_by(tool, baseline, candidate)

            comparison = ToolComparison(
                tool=tool,
                baseline_rate=before,
                candidate_rate=after,
                opportunities=opportunities,
                paired=paired,
                tolerance=self.tolerance,
                stolen_by=stolen_by,
            )
            comparisons.append(comparison)
            if paired is None:
                unpaired.append(tool)
            if comparison.underpowered:
                underpowered.append(tool)

            # Each tool is tested at alpha with no multiplicity correction, and
            # that is deliberate. Accepting a candidate asserts the conjunction
            # "no tool regressed", so this is an intersection-union test: a
            # conjunction of claims each tested at alpha is itself valid at
            # alpha. Bonferroni would raise each tool's bar as the catalogue
            # grows, making the gate looser the more tools it has to protect,
            # which is backwards for a safety gate.
            if comparison.regressed:
                regressions.append(
                    ToolRegression(
                        tool=tool,
                        baseline_rate=before,
                        candidate_rate=after,
                        opportunities=opportunities,
                        stolen_by=stolen_by,
                        paired=paired,
                        tolerance=self.tolerance,
                    )
                )
            elif comparison.improved:
                improvements.append(
                    ToolImprovement(
                        tool=tool,
                        baseline_rate=before,
                        candidate_rate=after,
                        opportunities=opportunities,
                        paired=paired,
                        tolerance=self.tolerance,
                    )
                )

        overall_delta = candidate.overall_accuracy - baseline.overall_accuracy
        accepted = not regressions
        if accepted and self.require_overall_improvement and overall_delta <= _EPSILON:
            accepted = False
            reason = (
                f"no per-tool regression, but overall accuracy did not improve "
                f"({overall_delta:+.1%})"
            )
        elif regressions:
            worst = min(regressions, key=lambda r: r.delta)
            significant = [r for r in regressions if r.significant_regression]
            reason = (
                f"{len(regressions)} tool(s) regressed beyond a "
                f"{self.tolerance:.1%} tolerance, worst {worst.describe()}"
            )
            if significant:
                reason += (
                    f" - {len(significant)} of them significant at alpha="
                    f"{self.alpha:g} ({', '.join(r.tool for r in significant)})"
                )
            if overall_delta > 0:
                reason += (
                    f" - rejected despite the aggregate improving {overall_delta:+.1%}"
                )
        else:
            reason = (
                f"no tool regressed beyond a {self.tolerance:.1%} tolerance "
                f"({len(improvements)} improved, {len(ignored)} not measurable)"
            )

        return CrossToolVerdict(
            accepted=accepted,
            baseline_accuracy=baseline.overall_accuracy,
            candidate_accuracy=candidate.overall_accuracy,
            regressions=regressions,
            improvements=improvements,
            ignored=ignored,
            tolerance=self.tolerance,
            reason=reason,
            comparisons=comparisons,
            underpowered=underpowered,
            unpaired=unpaired,
            baseline_interval=baseline.accuracy_interval(self.confidence),
            candidate_interval=candidate.accuracy_interval(self.confidence),
            # Derive the chance rate from the option count rather than taking
            # the max of both. chance = 1 / num_options, so max() of one is
            # min() of the other: two reports with 4 and 6 options used to
            # report "25% chance across 6 options", which is arithmetically
            # impossible. Take the wider catalogue and compute its chance rate.
            chance_accuracy=chance_accuracy(
                max(baseline.num_options, candidate.num_options)
            ),
            num_options=max(baseline.num_options, candidate.num_options),
        )

    def gate(
        self,
        baseline: CrossToolReport,
        candidate: CrossToolReport,
    ) -> GateResult:
        """The comparison as a :class:`GateResult`, ready for a GateChain."""
        return self.compare(baseline, candidate).to_gate_result()

    @staticmethod
    def _stolen_by(
        tool: str,
        baseline: CrossToolReport,
        candidate: CrossToolReport,
    ) -> dict[str, int]:
        """Which tools newly absorbed *tool*'s selections in the candidate."""
        before = baseline.confusion.row(tool)
        after = candidate.confusion.row(tool)
        stolen: dict[str, int] = {}
        for predicted, count in after.items():
            if predicted == tool:
                continue
            gained = count - before.get(predicted, 0)
            if gained > 0:
                stolen[predicted] = gained
        return stolen

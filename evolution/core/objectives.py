"""Multi-objective scoring and Pareto selection.

GEPA is *Genetic-Pareto*: it maintains a front of candidates that trade off
against each other. The original integration collapsed that to a single
LLM-judge float, which threw away the trade-off structure the algorithm exists
to exploit — and, worse, carried no size term at all.

The consequence was not theoretical. Because the size penalty was computed
from the baseline body rather than the candidate, it evaluated to exactly zero
for every variant, the search grew the skill 37.6% unopposed, and a post-hoc
constraint gate then rejected the winner. Size pressure has to live *inside*
the objective, where the optimizer can feel it, not in a gate that fires after
the budget is already spent.

This module keeps the axes separate:

    quality      0-1, maximize — judge or grader score
    size_chars   minimize — artifact length, soft-capped against a budget
    tokens       minimize — what the agent burned producing the answer
    tool_calls   minimize — how much flailing the skill caused

``scalarize`` folds them into the single float GEPA's metric must return, with
the size term computed from the candidate. ``pareto_front`` keeps the full
vectors for final selection among survivors.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

# Size pressure starts here, as a fraction of the budget, and reaches full
# strength at the budget itself. Starting below 1.0 means the optimizer feels
# the constraint while it still has room to respond, instead of discovering it
# at the gate.
SIZE_SOFT_START = 0.75

# Most a size overrun can subtract from a score. Large enough to outrank the
# judge's preference for thoroughness, small enough that a genuinely better
# but slightly longer variant can still win.
MAX_SIZE_PENALTY = 0.35

# Same idea for growth against the baseline.
MAX_GROWTH_PENALTY = 0.25


@dataclass(frozen=True)
class ObjectiveWeights:
    """Relative importance of each axis when collapsing to one number."""

    quality: float = 1.0
    size: float = 1.0
    tokens: float = 0.0
    tool_calls: float = 0.0

    def normalized(self) -> "ObjectiveWeights":
        total = self.quality + self.size + self.tokens + self.tool_calls
        if total <= 0:
            return ObjectiveWeights()
        return ObjectiveWeights(
            quality=self.quality / total,
            size=self.size / total,
            tokens=self.tokens / total,
            tool_calls=self.tool_calls / total,
        )


@dataclass
class ObjectiveVector:
    """One candidate's score on every axis we care about."""

    quality: float = 0.0
    size_chars: int = 0
    tokens: int = 0
    tool_calls: int = 0

    # Context needed to interpret the raw numbers.
    size_budget: int = 15_000
    baseline_chars: int = 0
    max_growth: float = 0.20

    feedback: str = ""
    detail: dict = field(default_factory=dict)

    # ── derived size terms ───────────────────────────────────────────────

    @property
    def size_ratio(self) -> float:
        """Candidate size as a fraction of its budget."""
        return self.size_chars / max(1, self.size_budget)

    @property
    def growth(self) -> float:
        """Growth against the baseline artifact, as a fraction."""
        if self.baseline_chars <= 0:
            return 0.0
        return (self.size_chars - self.baseline_chars) / self.baseline_chars

    def size_penalty(self) -> float:
        """Penalty for approaching or exceeding the size budget.

        Zero below the soft-start threshold, ramping linearly to
        ``MAX_SIZE_PENALTY`` at the budget and staying there beyond it (a
        variant that is 3x over is already disqualified; scoring it as
        progressively worse buys nothing).
        """
        ratio = self.size_ratio
        if ratio <= SIZE_SOFT_START:
            return 0.0
        span = max(1e-6, 1.0 - SIZE_SOFT_START)
        scaled = (ratio - SIZE_SOFT_START) / span
        return min(MAX_SIZE_PENALTY, scaled * MAX_SIZE_PENALTY)

    def growth_penalty(self) -> float:
        """Penalty for growing faster than the allowed rate against baseline."""
        if self.baseline_chars <= 0:
            return 0.0
        over = self.growth - self.max_growth
        if over <= 0:
            return 0.0
        # Full penalty once the candidate has doubled the allowance.
        scaled = min(1.0, over / max(1e-6, self.max_growth))
        return scaled * MAX_GROWTH_PENALTY

    # ── scalarization ────────────────────────────────────────────────────

    def scalarize(
        self,
        weights: Optional[ObjectiveWeights] = None,
        token_budget: int = 0,
        tool_call_budget: int = 0,
    ) -> float:
        """Collapse to the single 0-1 float GEPA's metric contract requires.

        Rewards are a weighted average over the axes that actually have data,
        so a run with no agent-level telemetry scores purely on quality and
        still uses the full 0-1 range. Size is not a reward — a short skill is
        not automatically a good one — so it subtracts from the result.
        """
        w = weights or ObjectiveWeights()

        # Weighted average over active reward axes. Averaging (rather than
        # summing normalized weights) keeps quality on the full 0-1 scale when
        # it is the only axis with data.
        rewards: list[tuple[float, float]] = [(max(0.0, w.quality), _clamp01(self.quality))]
        if token_budget > 0 and w.tokens > 0:
            rewards.append((w.tokens, 1.0 - min(1.0, self.tokens / token_budget)))
        if tool_call_budget > 0 and w.tool_calls > 0:
            rewards.append((w.tool_calls, 1.0 - min(1.0, self.tool_calls / tool_call_budget)))

        total_weight = sum(weight for weight, _ in rewards)
        if total_weight <= 0:
            return 0.0
        score = sum(weight * value for weight, value in rewards) / total_weight

        # Size pressure scales against the quality weight, so the default 1:1
        # split gives it full effect and ObjectiveWeights(size=0) removes it.
        size_scale = min(2.0, w.size / max(w.quality, 1e-6))
        score -= (self.size_penalty() + self.growth_penalty()) * size_scale

        return _clamp01(score)

    def with_quality(self, quality: float) -> "ObjectiveVector":
        return replace(self, quality=quality)

    def as_dict(self) -> dict:
        return {
            "quality": round(self.quality, 4),
            "size_chars": self.size_chars,
            "size_ratio": round(self.size_ratio, 3),
            "growth": round(self.growth, 4),
            "size_penalty": round(self.size_penalty(), 4),
            "growth_penalty": round(self.growth_penalty(), 4),
            "tokens": self.tokens,
            "tool_calls": self.tool_calls,
            "scalar": round(self.scalarize(), 4),
        }


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


# ── Pareto machinery ─────────────────────────────────────────────────────

# (attribute, higher_is_better)
_AXES: tuple[tuple[str, bool], ...] = (
    ("quality", True),
    ("size_chars", False),
    ("tokens", False),
    ("tool_calls", False),
)


def dominates(a: ObjectiveVector, b: ObjectiveVector, axes=_AXES) -> bool:
    """True when ``a`` is at least as good as ``b`` everywhere and better somewhere.

    Axes with no data on either side (both zero) are skipped, so a run without
    token telemetry does not make every candidate mutually non-dominated.
    """
    strictly_better = False
    for attr, higher_better in axes:
        av, bv = getattr(a, attr), getattr(b, attr)
        if av == 0 and bv == 0:
            continue
        if higher_better:
            if av < bv:
                return False
            if av > bv:
                strictly_better = True
        else:
            if av > bv:
                return False
            if av < bv:
                strictly_better = True
    return strictly_better


def pareto_front(vectors: Sequence[ObjectiveVector]) -> list[int]:
    """Indices of the non-dominated candidates, preserving input order."""
    front: list[int] = []
    for i, candidate in enumerate(vectors):
        if not any(
            dominates(other, candidate)
            for j, other in enumerate(vectors)
            if j != i
        ):
            front.append(i)
    return front


def select_best(
    vectors: Sequence[ObjectiveVector],
    weights: Optional[ObjectiveWeights] = None,
) -> Optional[int]:
    """Pick one winner: highest scalar score among the Pareto-optimal set.

    Restricting to the front first means the winner is never a candidate that
    another beats on every axis — which a pure scalar sort can otherwise pick
    when the weights happen to flatter it.
    """
    if not vectors:
        return None
    front = pareto_front(vectors)
    if not front:
        front = list(range(len(vectors)))
    return max(front, key=lambda i: vectors[i].scalarize(weights))


def summarize_front(vectors: Sequence[ObjectiveVector]) -> list[dict]:
    """Front members as plain dicts, for reports and metrics files."""
    return [vectors[i].as_dict() for i in pareto_front(vectors)]

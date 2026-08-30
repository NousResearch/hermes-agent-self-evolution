"""What a run actually spent.

PLAN.md requires the cost of the optimization run in every PR body, and puts a
number on the expectation: "GEPA optimization: ~$2-10 per run". A reviewer
deciding whether an evolved artifact was worth producing needs that figure, and
so does anyone deciding whether to run the pipeline again.

DSPy records every model call in ``dspy.clients.base_lm.GLOBAL_HISTORY`` with a
``usage`` dict and a ``cost``. This module reads that log rather than trying to
price calls itself, because a local price table goes stale the week a provider
changes rates and a stale price presented as a cost is worse than no cost.

Three honesty rules, because a cost report that quietly rounds down is worse
than useless:

* ``cost`` is ``None`` for any model DSPy has no pricing for. Those calls are
  counted separately and reported as unpriced, never summed as zero.
* DSPy caps its global history, so a long run can lose early entries. When the
  log looks truncated the report says the totals are a lower bound.
* A cached call costs nothing and is counted as cached, so a cheap rerun is not
  mistaken for a cheap pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

__all__ = [
    "LMCall",
    "CostReport",
    "UsageTracker",
    "read_history",
]


_FALLBACKS_NOTED: set = set()


def _note_fallback(message: str) -> None:
    """Log a degraded-capability path once, not once per call.

    The fallbacks below are silent by design - a run without cost accounting
    still works - but silent and invisible are different things. One debug
    line marks the moment a DSPy upgrade moved the log out from under us.
    """
    if message not in _FALLBACKS_NOTED:
        _FALLBACKS_NOTED.add(message)
        logging.getLogger(__name__).debug(message)


def _history() -> list:
    """The live DSPy call log, or an empty list if DSPy moves it."""
    try:
        from dspy.clients.base_lm import GLOBAL_HISTORY

        return GLOBAL_HISTORY
    except Exception:  # pragma: no cover - depends on the installed dspy
        _note_fallback(
            "dspy no longer exposes clients.base_lm.GLOBAL_HISTORY; "
            "cost reporting will show no calls"
        )
        return []


def _max_history_size() -> int:
    """How many entries DSPy keeps before it starts evicting. 0 if unknown."""
    try:
        from dspy.clients import base_lm

        size = getattr(base_lm, "MAX_HISTORY_SIZE", 0)
        return int(size) if isinstance(size, int) else 0
    except Exception:  # pragma: no cover
        _note_fallback(
            "dspy no longer exposes clients.base_lm.MAX_HISTORY_SIZE; "
            "truncation detection is off"
        )
        return 0


def _entry_identity(entry: Any) -> Any:
    """Something stable that identifies one history entry.

    DSPy stamps each entry with a uuid. Falling back to ``id()`` keeps the
    anchor working for hand-built entries in tests, where the objects are the
    same list members throughout.
    """
    if isinstance(entry, dict):
        marker = entry.get("uuid")
        if marker is not None:
            return ("uuid", marker)
    return ("id", id(entry))


@dataclass(frozen=True)
class LMCall:
    """One model call, as much of it as DSPy recorded."""

    model: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: Optional[float] = None
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens for this call."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def priced(self) -> bool:
        """True when a price was available for this call."""
        return self.cost is not None

    @classmethod
    def from_entry(cls, entry: Any) -> "LMCall":
        """Read one DSPy history entry defensively.

        Entry shape is not part of DSPy's public API, so every field is
        optional and a malformed entry degrades to an unpriced zero-token call
        rather than raising in the middle of a run.
        """
        if not isinstance(entry, dict):
            return cls()

        usage = entry.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}

        def _count(*names: str) -> int:
            for name in names:
                value = usage.get(name)
                if isinstance(value, (int, float)):
                    return int(value)
            return 0

        cost = entry.get("cost")
        if not isinstance(cost, (int, float)):
            cost = None

        return cls(
            model=str(entry.get("model") or "unknown"),
            prompt_tokens=_count("prompt_tokens", "input_tokens"),
            completion_tokens=_count("completion_tokens", "output_tokens"),
            cost=float(cost) if cost is not None else None,
            # A call DSPy served from cache reports no usage at all.
            cached=bool(entry.get("cached")) or (not usage and cost in (None, 0)),
        )


@dataclass
class CostReport:
    """Totals for a set of model calls, with the gaps named."""

    calls: list[LMCall] = field(default_factory=list)
    truncated: bool = False

    @property
    def n_calls(self) -> int:
        """How many model calls were recorded."""
        return len(self.calls)

    @property
    def prompt_tokens(self) -> int:
        """Prompt tokens summed across every call."""
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        """Completion tokens summed across every call."""
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens across every call."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def known_cost(self) -> float:
        """Summed cost of the calls that had one. Never includes a guess."""
        return sum(c.cost or 0.0 for c in self.calls if c.priced)

    @property
    def unpriced_calls(self) -> int:
        """Calls with no price available, which make the total a lower bound."""
        return sum(1 for c in self.calls if not c.priced and not c.cached)

    @property
    def cached_calls(self) -> int:
        """Calls served from cache, which are not counted as unpriced."""
        return sum(1 for c in self.calls if c.cached)

    @property
    def complete(self) -> bool:
        """True when every call was priced and nothing was lost from the log."""
        return not self.truncated and self.unpriced_calls == 0

    @property
    def models(self) -> dict[str, int]:
        """Call count per model, name-sorted."""
        counts: dict[str, int] = {}
        for call in self.calls:
            counts[call.model] = counts.get(call.model, 0) + 1
        return dict(sorted(counts.items()))

    def describe(self) -> str:
        """One line of cost, saying so plainly when the total is only a lower bound."""
        if not self.calls:
            return "no model calls recorded"

        parts = [f"{self.n_calls} call(s)"]
        if self.total_tokens:
            parts.append(
                f"{self.total_tokens:,} tokens "
                f"({self.prompt_tokens:,} in / {self.completion_tokens:,} out)"
            )
        prefix = "at least " if not self.complete else ""
        parts.append(f"{prefix}${self.known_cost:.4f}")
        if self.unpriced_calls:
            parts.append(f"{self.unpriced_calls} call(s) with no price available")
        if self.cached_calls:
            parts.append(f"{self.cached_calls} served from cache")
        if self.truncated:
            parts.append("history truncated, totals are a lower bound")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        """Serialise the cost report for the run artifacts."""
        return {
            "calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "known_cost_usd": round(self.known_cost, 6),
            "unpriced_calls": self.unpriced_calls,
            "cached_calls": self.cached_calls,
            "complete": self.complete,
            "truncated": self.truncated,
            "models": self.models,
        }


def read_history(entries: Iterable[Any]) -> CostReport:
    """Build a report from raw DSPy history entries."""
    return CostReport(calls=[LMCall.from_entry(e) for e in entries])


class UsageTracker:
    """Context manager that measures the model calls made inside its block.

    Records the length of DSPy's global log on entry and reads everything added
    by the time it exits. If the log is shorter on exit than it was on entry,
    DSPy evicted entries and the report says so rather than reporting a
    negative or silently partial total.

        with UsageTracker() as usage:
            run_the_optimizer()
        print(usage.report.describe())
    """

    def __init__(self) -> None:
        self._start = 0
        self._anchor: Any = None
        self.report = CostReport()

    def __enter__(self) -> "UsageTracker":
        history = _history()
        self._start = len(history)
        # Remember *which* entry was last, not just how many there were. DSPy
        # evicts from the front once the log reaches MAX_HISTORY_SIZE, so its
        # length stops growing and a length comparison alone cannot notice that
        # 25,000 calls happened. Anchoring on an identity can.
        self._anchor = _entry_identity(history[-1]) if history else None
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def stop(self) -> CostReport:
        """Stop tracking and return everything recorded since :meth:`start`.

        The anchor is searched from the end of the history, so a run whose anchor
        the history cap has already evicted reports the whole visible window
        rather than silently reporting nothing.
        """
        history = list(_history())
        cap = _max_history_size()

        if self._anchor is not None:
            index = None
            for i in range(len(history) - 1, -1, -1):
                if _entry_identity(history[i]) == self._anchor:
                    index = i
                    break
            if index is None:
                # The entry we anchored on has been evicted, so this run made
                # more calls than the log can hold. Everything left is a lower
                # bound on what happened.
                self.report = read_history(history)
                self.report.truncated = True
                return self.report
            new = history[index + 1:]
        elif len(history) < self._start:
            self.report = read_history(history)
            self.report.truncated = True
            return self.report
        else:
            new = history[self._start:]

        self.report = read_history(new)
        # Sitting exactly at the cap means eviction either happened or was one
        # call away, and the two are indistinguishable from here. Say so rather
        # than report a confident total that may be missing its first half.
        if cap and len(history) >= cap:
            self.report.truncated = True
        return self.report

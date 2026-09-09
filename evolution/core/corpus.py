"""Size budgets derived from the real skill corpus.

The hardcoded 15 KB cap was not derived from anything in Hermes. Measured
against the installed library it rejects 27 of 201 skills *at baseline* — the
largest, ``research-paper-writing``, is 72 KB. Those skills could never be
evolved: their unmodified text failed the gate before the optimizer ran.

A budget should describe the corpus it governs. This module measures the
skills that actually exist and sets the ceiling from their distribution, with
the individual skill's own current size as a floor so no skill is ever
disqualified for already being what it is.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Percentile of the corpus a skill may grow to. p90 leaves headroom for the
# genuinely large reference skills without licensing unbounded growth.
DEFAULT_PERCENTILE = 90

# Never let a derived budget fall below this — a tiny corpus (a handful of
# stub skills) would otherwise produce a budget that blocks normal writing.
MIN_BUDGET_CHARS = 8_000

# How much room above its current size a skill always gets, even when the
# corpus percentile sits below it. Without this, an already-large skill has a
# budget under its own baseline and cannot be edited at all.
BASELINE_HEADROOM = 0.15


@dataclass(frozen=True)
class CorpusStats:
    """Size distribution of a skill corpus, in characters."""

    count: int
    smallest: int
    median: int
    p75: int
    p90: int
    largest: int

    def percentile(self, pct: int) -> int:
        if pct <= 50:
            return self.median
        if pct <= 75:
            return self.p75
        if pct <= 90:
            return self.p90
        return self.largest

    def describe(self) -> str:
        return (
            f"{self.count} skills · median {self.median:,} · "
            f"p90 {self.p90:,} · max {self.largest:,} chars"
        )


def iter_skill_files(*roots: Path) -> Iterable[Path]:
    """Every ``SKILL.md`` under the given roots, de-duplicated by resolved path.

    Skill trees overlap in a real install (the repo ships skills that the user
    tree also carries), so the same file can be reached by two roots. Counting
    it twice would skew the distribution toward whatever is duplicated.
    """
    seen: set[Path] = set()
    for root in roots:
        if not root or not root.is_dir():
            continue
        for path in sorted(root.rglob("SKILL.md")):
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def measure_corpus(*roots: Path) -> Optional[CorpusStats]:
    """Measure skill sizes across the given trees, or None when none exist."""
    sizes: list[int] = []
    for path in iter_skill_files(*roots):
        try:
            sizes.append(len(path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue

    if not sizes:
        return None

    sizes.sort()
    return CorpusStats(
        count=len(sizes),
        smallest=sizes[0],
        median=int(statistics.median(sizes)),
        p75=_percentile(sizes, 75),
        p90=_percentile(sizes, 90),
        largest=sizes[-1],
    )


def _percentile(sorted_sizes: list[int], pct: int) -> int:
    if not sorted_sizes:
        return 0
    if len(sorted_sizes) == 1:
        return sorted_sizes[0]
    # Nearest-rank: the smallest value at or above the requested percentile.
    rank = max(1, min(len(sorted_sizes), round(pct / 100 * len(sorted_sizes))))
    return sorted_sizes[rank - 1]


def derive_size_budget(
    baseline_chars: int,
    stats: Optional[CorpusStats],
    percentile: int = DEFAULT_PERCENTILE,
    fallback: int = 15_000,
) -> tuple[int, str]:
    """Return (budget_chars, human-readable rationale).

    The budget is the larger of the corpus percentile and the skill's own size
    plus headroom, floored at ``MIN_BUDGET_CHARS``. Returning the rationale
    alongside the number keeps run logs self-explanatory — a rejected variant
    should never leave the reader guessing where the limit came from.
    """
    baseline_floor = int(baseline_chars * (1 + BASELINE_HEADROOM)) if baseline_chars else 0

    if stats is None:
        budget = max(fallback, baseline_floor, MIN_BUDGET_CHARS)
        source = f"no corpus found; fallback {fallback:,}"
    else:
        corpus_budget = stats.percentile(percentile)
        budget = max(corpus_budget, baseline_floor, MIN_BUDGET_CHARS)
        source = f"corpus p{percentile} = {corpus_budget:,} over {stats.count} skills"

    if baseline_floor and budget == baseline_floor:
        source += f"; raised to baseline +{BASELINE_HEADROOM:.0%}"

    return budget, source

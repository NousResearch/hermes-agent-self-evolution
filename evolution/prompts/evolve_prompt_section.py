"""Evolve a hermes-agent system prompt section with DSPy + GEPA.

Usage:
    python -m evolution.prompts.evolve_prompt_section --section MEMORY_GUIDANCE
    python -m evolution.prompts.evolve_prompt_section --all-sections --iterations 5 --strict-gates
    python -m evolution.prompts.evolve_prompt_section --section SKILLS_GUIDANCE --write
    python -m evolution.prompts.evolve_prompt_section --section MEMORY_GUIDANCE --write --push --open-pr

This is the riskiest tier in PLAN.md, and the code is shaped by that rather
than by the optimizer. A skill affects the sessions that load it. A tool
description affects the sessions that see the tool. A system prompt section is
in front of every turn of every session, so the interesting questions here are
all about what the run refuses to do:

* Only the four allowlisted string constants can be targeted. ``PLATFORM_HINTS``
  is reported and skipped, because in the real ``agent/prompt_builder.py`` it is
  a dict of per-platform strings with its own accuracy rules.
* A candidate that grows a section past +20%, drops a core identity trait, or
  blows the prompt caching budget is rejected before it is ever scored.
* Benchmark regression tolerance is 0.0, not the 2% Phases 2 and 3 use
  elsewhere. PLAN.md is explicit: "zero tolerance for regression here."
  ``--strict-gates`` goes one step further and treats a benchmark that could
  not be run, or a cache block boundary crossing, as blocking - which is the
  honest setting given that none of PLAN.md's benchmarks exist in hermes-agent
  today.
* Writing is off by default, and even with ``--write`` the run refuses while a
  Hermes session looks live. A running session has already assembled and cached
  its system prompt; an evolved section deploys on the NEXT session and is never
  hot-swapped into a running one.
* The holdout comparison is a paired test, not a subtraction. Baseline and
  candidate answer the same scenarios in the same order, so the run has matched
  pairs and uses them: a candidate is deployable only when the paired Wilcoxon
  test calls the improvement significant AND the point estimate clears
  PLAN.md's 10% bar. A +12% swing that the test cannot separate from noise is
  reported as inconclusive, with the number of extra scenarios it would take to
  settle the question.
* Categories are checked one at a time as well as in aggregate. A rewrite that
  lifts memory guidance while wrecking platform formatting is a regression, and
  the aggregate mean is exactly where that hides, so any category that drops
  past the tolerance - or drops significantly at all - blocks the write.

Nothing here mutates the hermes-agent working tree except the explicit
write-back step and the gate ladder's staged write, which restores the original
in a ``finally`` and leaves a backup in the run's output directory first.

A run that does write ends by building the pull request PLAN.md's constraint 5
asks for: a branch, a commit, and a body carrying the paired holdout evidence,
the gate results, the cost of the run, and every section that was refused along
the way. All of that is local. Pushing the branch and opening the PR are
separate steps that happen only when the operator asks for them by name, and
the checkout is put back on the ref it started on either way.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import Optional, Sequence

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.artifact_io import EVOLVABLE_PROMPT_SECTIONS
from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.cost import CostReport, UsageTracker
from evolution.core.gates import GateChain, run_benchmark_gate, run_pytest_gate
from evolution.core.pr_builder import (
    GitError,
    PullRequestPlan,
    RejectedCandidate,
    ScoreLine,
    build_pull_request,
)
from evolution.core.stats import (
    PairedContinuous,
    compare_paired_continuous,
    holm_adjust,
    min_detectable_paired_shift,
    wilcoxon_signed_rank,
)
from evolution.prompts.behavioral_eval import (
    SECTION_CATEGORIES,
    BehavioralJudge,
    BehavioralReport,
    BehavioralSuite,
    SectionBehaviorModule,
    make_behavioral_metric,
    scenarios_to_dspy_examples,
    select_harness,
)
from evolution.prompts.sections import (
    NEXT_SESSION_NOTICE,
    SectionValidation,
    UnknownSection,
    detect_active_session,
    load_sections,
    staged_prompt_write,
    validate_section_names,
    write_sections,
)

console = Console()

# PLAN.md: "Benchmarks hold or improve (zero tolerance for regression here)."
# Binary floating point makes 0.6 - 0.5 come out at 0.09999999999999998, so a
# strict comparison rejected an exactly-10% lift against a 10% bar and printed
# "+10.0% is under the 10% threshold". Compare with a hair of slack instead.
_FLOAT_SLACK = 1e-9

# How much of a holdout suite may go unmeasured before the run is refused
# rather than compared on what is left. A few dropped scenarios shrink the
# sample; a third of them means the harness, not the prompt, decided the
# result.
MAX_UNMEASURED_FRACTION = 0.2

ZERO_REGRESSION_TOLERANCE = 0.0

# Narrow slice of hermes-agent's suite. The full suite is 2550+ tests and none
# of the rest can be affected by prompt text, so the affordable subset is the
# one that reads prompt_builder.
DEFAULT_PYTEST_SUBSET = ("tests/", "-k", "prompt")

DEFAULT_BENCHMARKS = ("tblite", "yc_bench")

# PLAN.md: "Behavioral test scores improve (>=10% on targeted sections)."
# Judge scores live in [0, 1], so the practical bar is 0.10 in score units.
PRACTICAL_IMPROVEMENT = 0.10

# Significance level for every paired holdout test in this phase.
HOLDOUT_ALPHA = 0.05

# A category may drift, but not far. Past this the candidate is refused on the
# point estimate alone, without waiting for a significance test the holdout is
# usually too small to pass.
CATEGORY_REGRESSION_TOLERANCE = 0.05

# Fixed so a gate decision is reproducible. The bootstrap interval feeds an
# accept/reject call, and a verdict that could flip on a rerun is not auditable.
HOLDOUT_BOOTSTRAP_SEED = 20260731

# Baseline and candidate holdout runs must be labelled identically. The label
# reaches the direct harness's instruction scaffold, so letting it differ would
# change the prompt on one side of a paired comparison for no reason.
HOLDOUT_SECTION_LABEL = "SYSTEM_PROMPT"


# ──────────────────────────────────────────────────────────────────────────
# Holdout statistics
# ──────────────────────────────────────────────────────────────────────────


class UnpairedHoldout(ValueError):
    """The two holdout runs do not line up scenario for scenario.

    Raised rather than worked around. The pairing is what makes a holdout of
    this size worth anything: the uncertainty in a paired difference comes from
    the scenarios where the two prompts disagreed, not from the spread of the
    scores. Two means measured over different scenario sets answer a question
    nobody asked, so a misalignment is a bug in the run and gets treated as one.
    """


@lru_cache(maxsize=None)
def min_scenarios_for_significance(alpha: float = HOLDOUT_ALPHA) -> int:
    """Smallest holdout whose paired scores could ever reach significance.

    The most favourable evidence a paired suite can produce is every scenario
    moving the same way by the same amount: the signed-rank statistic bottoms
    out at zero, and the tie correction takes the variance as low as it goes.
    Feeding exactly that into the same Wilcoxon routine the gate uses gives a
    hard floor. Below this many scenarios the test cannot return ``p < alpha``
    for any candidate at all, so "not significant" there is a statement about
    the sample size rather than about the candidate, and saying which one it is
    is the whole point of reporting power.

    Note this is the floor, not a target. Real judge scores move by different
    amounts on different scenarios, which loses the tie correction and needs
    more scenarios than this.
    """
    for n in range(1, 201):
        _, p = wilcoxon_signed_rank([0.0] * n, [1.0] * n)
        if p < alpha:
            return n
    return 201


def align_holdout_scores(
    baseline: BehavioralReport, candidate: BehavioralReport
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    """Line two holdout reports up scenario by scenario.

    ``BehavioralJudge.score_all`` emits exactly one outcome per scenario, in the
    order it was handed them, so two runs over the same scenario list are
    already aligned. This checks that invariant rather than trusting it: a
    silent misalignment still produces a number, and the number looks fine.

    A scenario the harness failed to produce a transcript for is **dropped from
    both sides**, not scored as a behavioural failure. An unmeasured run is
    missing data: giving it 0.0 on the side that timed out and a real score on
    the side that completed fabricates a difference out of a flake, and six
    baseline timeouts against a clean candidate run read as a significant +32%
    improvement. Losing more than ``MAX_UNMEASURED_FRACTION`` of the suite that
    way means the run cannot be trusted at all and is refused outright.

    Returns ``(scenario_ids, categories, baseline_scores, candidate_scores)``.
    """
    base_ids = tuple(o.scenario_id for o in baseline.outcomes)
    cand_ids = tuple(o.scenario_id for o in candidate.outcomes)
    if len(base_ids) != len(cand_ids):
        raise UnpairedHoldout(
            f"holdout runs produced {len(base_ids)} and {len(cand_ids)} outcomes; "
            "a paired test needs the same scenarios on both sides"
        )
    if base_ids != cand_ids:
        drift = [
            f"index {i}: {b or '?'} vs {c or '?'}"
            for i, (b, c) in enumerate(zip(base_ids, cand_ids))
            if b != c
        ]
        raise UnpairedHoldout(
            "holdout runs are not in the same scenario order (" + "; ".join(drift[:3]) + ")"
        )

    kept = [
        (b, c)
        for b, c in zip(baseline.outcomes, candidate.outcomes)
        if getattr(b, "measured", True) and getattr(c, "measured", True)
    ]
    total = len(base_ids)
    dropped = total - len(kept)
    if total and dropped / total > MAX_UNMEASURED_FRACTION:
        raise UnpairedHoldout(
            f"{dropped} of {total} holdout scenarios produced no transcript on one "
            f"side or the other; over the {MAX_UNMEASURED_FRACTION:.0%} limit, so "
            "this run measured too little to compare"
        )

    return (
        tuple(b.scenario_id for b, _ in kept),
        tuple(b.category for b, _ in kept),
        tuple(b.score for b, _ in kept),
        tuple(c.score for _, c in kept),
    )


@dataclass
class HoldoutComparison:
    """Paired baseline-vs-candidate analysis of one holdout run.

    Three questions, kept separate because they fail separately:

    1. Did anything happen? ``overall.significant_improvement`` is a Wilcoxon
       signed-rank test on the matched differences, not a sign on a subtraction.
    2. Did enough happen? ``practical_threshold`` is PLAN.md's 10% bar. A
       significant +2% is real and still not worth deploying a system prompt for.
    3. Did anything break? ``by_category`` runs the same paired test inside each
       behavioural category, because an aggregate mean is exactly where one
       wrecked category hides behind four improved ones.
    """

    scenario_ids: tuple[str, ...]
    categories: tuple[str, ...]
    baseline_scores: tuple[float, ...]
    candidate_scores: tuple[float, ...]
    overall: PairedContinuous
    by_category: dict[str, PairedContinuous] = field(default_factory=dict)
    targeted_category: str = ""
    alpha: float = HOLDOUT_ALPHA
    practical_threshold: float = PRACTICAL_IMPROVEMENT
    category_tolerance: float = CATEGORY_REGRESSION_TOLERANCE

    @property
    def n(self) -> int:
        """Scenarios compared, which is what the pairing uses."""
        return self.overall.n

    @property
    def delta(self) -> float:
        """Overall change from baseline to candidate."""
        return self.overall.delta

    @property
    def targeted(self) -> Optional[PairedContinuous]:
        """The category owned by the section under test, when it has scenarios."""
        return self.by_category.get(self.targeted_category)

    @property
    def movement(self) -> tuple[int, int, int]:
        """How many scenarios went up, down, and nowhere."""
        pairs = list(zip(self.baseline_scores, self.candidate_scores))
        up = sum(1 for b, c in pairs if c > b)
        down = sum(1 for b, c in pairs if c < b)
        return up, down, len(pairs) - up - down

    def category_regressed(self, comparison: PairedContinuous) -> bool:
        """A category fails on either kind of evidence.

        Conservative by construction: a drop past the tolerance is refused even
        when the suite is too small to prove it, and a statistically significant
        drop is refused even when it is small. Requiring both would make this
        gate weaker than the fixed-tolerance check it replaces, which is not
        what adding statistics is for.
        """
        if comparison.n == 0:
            return False
        return (
            comparison.delta <= -abs(self.category_tolerance)
            or comparison.significant_regression
        )

    @property
    def regressed_categories(self) -> tuple[str, ...]:
        """Every category this candidate is refused over.

        Intersection-union test: accepting the candidate means accepting the
        conjunction "no category regressed". A conjunction of claims each tested
        at alpha is itself valid at alpha, so no Bonferroni or Benjamini-Hochberg
        adjustment is applied here and none is needed. Correcting would raise
        the bar for calling any single category regressed as the category count
        grew, making the gate more permissive the more places a rewrite could do
        damage - exactly backwards for a safety check.
        """
        return tuple(
            name
            for name, comparison in self.by_category.items()
            if self.category_regressed(comparison)
        )

    @property
    def min_detectable_shift(self) -> float:
        """Smallest shift an exact paired test could call significant here."""
        return min_detectable_paired_shift(self.n, self.alpha)

    @property
    def scenarios_needed(self) -> int:
        """How many more scenarios significance would take, at minimum."""
        return max(0, min_scenarios_for_significance(self.alpha) - self.n)

    @property
    def underpowered(self) -> bool:
        """True when no result on this many scenarios could reach significance."""
        return self.n < min_scenarios_for_significance(self.alpha)

    @property
    def underpowered_categories(self) -> tuple[str, ...]:
        """Categories with too few scenarios for a significant result."""
        floor = min_scenarios_for_significance(self.alpha)
        return tuple(
            name for name, comparison in self.by_category.items() if comparison.n < floor
        )

    @property
    def improved(self) -> bool:
        """Significant on the test AND large enough to be worth deploying."""
        if not self.overall.significant_improvement:
            return False
        if self.overall.delta < self.practical_threshold - _FLOAT_SLACK:
            return False
        targeted = self.targeted
        if targeted is not None and targeted.n and targeted.delta < self.practical_threshold - _FLOAT_SLACK:
            return False
        return True

    @property
    def accepted(self) -> bool:
        """True when the run improved and no category regressed."""
        return bool(self.n) and self.improved and not self.regressed_categories

    @property
    def power_note(self) -> str:
        """What shift this sample size could have detected, in plain words."""
        if self.n == 0:
            return "no holdout scenarios, so this run measured nothing"
        floor = min_scenarios_for_significance(self.alpha)
        head = (
            f"{self.n} scenario(s); the smallest shift an exact paired test could "
            f"call significant here is {self.min_detectable_shift:.0%} of them"
        )
        if self.n < floor:
            return (
                f"{head}. Under {floor} scenarios nothing can reach "
                f"p < {self.alpha:g}, so a 'not significant' verdict here is the "
                f"sample size talking, not the candidate"
            )
        return head

    @property
    def shortfall_note(self) -> str:
        """What it would take to settle an inconclusive result."""
        if self.scenarios_needed:
            return (
                f"needs at least {self.scenarios_needed} more holdout scenario(s) "
                f"before this test can reach p < {self.alpha:g}"
            )
        up, down, flat = self.movement
        return (
            f"{up} improved / {down} regressed / {flat} unchanged, so the direction "
            "is not consistent enough for more scenarios alone to settle it"
        )

    @property
    def headline(self) -> str:
        """The verdict in a few words, for a table cell.

        The full :attr:`reason` is printed under the Holdout banner and saved in
        metrics.json. A results table that wraps a paragraph into every row is
        not a results table.
        """
        if self.n == 0:
            return "no holdout evidence"
        if self.regressed_categories:
            return "regressed: " + ", ".join(self.regressed_categories)
        if not self.overall.significant_improvement:
            return f"inconclusive (p={self.overall.wilcoxon_p:.3f})"
        if self.overall.delta < self.practical_threshold - _FLOAT_SLACK:
            return f"under the {self.practical_threshold:.0%} bar"
        targeted = self.targeted
        if targeted is not None and targeted.n and targeted.delta < self.practical_threshold - _FLOAT_SLACK:
            return f"targeted {targeted.delta:+.1%}, under the bar"
        return "accepted"

    @property
    def reason(self) -> str:
        """One line, in the order the checks are allowed to fail."""
        if self.n == 0:
            return "no holdout evidence"
        if self.regressed_categories:
            details = ", ".join(
                f"{name} {self.by_category[name].delta:+.1%}"
                for name in self.regressed_categories
            )
            return f"category regression: {details}"
        if not self.overall.significant_improvement:
            return (
                f"inconclusive: {self.overall.delta:+.1%} is not distinguishable "
                f"from noise (p={self.overall.wilcoxon_p:.3f}); {self.shortfall_note}"
            )
        if self.overall.delta < self.practical_threshold - _FLOAT_SLACK:
            return (
                f"significant but small: {self.overall.delta:+.1%} is under the "
                f"{self.practical_threshold:.0%} practical threshold"
            )
        targeted = self.targeted
        if targeted is not None and targeted.n and targeted.delta < self.practical_threshold - _FLOAT_SLACK:
            return (
                f"targeted category {self.targeted_category} moved "
                f"{targeted.delta:+.1%}, under the {self.practical_threshold:.0%} "
                "threshold PLAN.md sets for the section being evolved"
            )
        return (
            f"holdout improved {self.overall.delta:+.1%} "
            f"(p={self.overall.wilcoxon_p:.3f}) with no category regression"
        )

    def describe(self) -> str:
        """The overall paired comparison as one line."""
        return self.overall.describe()

    def to_dict(self) -> dict:
        """Serialise the holdout analysis, movement and per-category detail."""
        up, down, flat = self.movement
        targeted = self.targeted
        return {
            "n": self.n,
            "alpha": self.alpha,
            "practical_threshold": self.practical_threshold,
            "category_tolerance": self.category_tolerance,
            "targeted_category": self.targeted_category,
            "scenario_ids": list(self.scenario_ids),
            "overall": self.overall.to_dict(),
            "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
            "targeted": targeted.to_dict() if targeted else None,
            "regressed_categories": list(self.regressed_categories),
            "underpowered": self.underpowered,
            "underpowered_categories": list(self.underpowered_categories),
            "min_scenarios_for_significance": min_scenarios_for_significance(self.alpha),
            "scenarios_needed": self.scenarios_needed,
            "min_detectable_shift": round(self.min_detectable_shift, 6),
            "movement": {"improved": up, "regressed": down, "unchanged": flat},
            "power_note": self.power_note,
            "headline": self.headline,
            "improved": self.improved,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def compare_holdout(
    baseline: BehavioralReport,
    candidate: BehavioralReport,
    targeted_category: str = "",
    alpha: float = HOLDOUT_ALPHA,
    practical_threshold: float = PRACTICAL_IMPROVEMENT,
    category_tolerance: float = CATEGORY_REGRESSION_TOLERANCE,
    seed: int = HOLDOUT_BOOTSTRAP_SEED,
) -> HoldoutComparison:
    """Paired comparison of two holdout runs, in aggregate and per category.

    Raises :class:`UnpairedHoldout` when the two runs did not cover the same
    scenarios in the same order. Everything downstream assumes index i means the
    same scenario on both sides.
    """
    ids, categories, base_scores, cand_scores = align_holdout_scores(baseline, candidate)

    buckets: dict[str, tuple[list[float], list[float]]] = {}
    for category, base, cand in zip(categories, base_scores, cand_scores):
        bucket = buckets.setdefault(category, ([], []))
        bucket[0].append(base)
        bucket[1].append(cand)

    by_category = {
        category: compare_paired_continuous(
            base, cand, alpha=alpha, confidence=1 - alpha, seed=seed
        )
        for category, (base, cand) in sorted(buckets.items())
    }

    return HoldoutComparison(
        scenario_ids=ids,
        categories=categories,
        baseline_scores=base_scores,
        candidate_scores=cand_scores,
        overall=compare_paired_continuous(
            list(base_scores), list(cand_scores), alpha=alpha, confidence=1 - alpha, seed=seed
        ),
        by_category=by_category,
        targeted_category=targeted_category,
        alpha=alpha,
        practical_threshold=practical_threshold,
        category_tolerance=category_tolerance,
    )


@dataclass
class SectionOutcome:
    """What happened to one section during a run."""

    name: str
    baseline_text: str
    evolved_text: str
    baseline_score: float = 0.0
    evolved_score: float = 0.0
    holdout_baseline: Optional[float] = None
    holdout_evolved: Optional[float] = None
    holdout: Optional[HoldoutComparison] = None
    validation: Optional[SectionValidation] = None
    accepted: bool = False
    reason: str = ""
    # Set only when several sections were tested against one baseline in the
    # same run; None for a single-section run, where no correction applies.
    adjusted_p: Optional[float] = None
    elapsed_s: float = 0.0
    optimizer: str = ""

    @property
    def improvement(self) -> float:
        """Descriptive only, and deliberately not a decision input.

        ``baseline_score`` is the validation-split mean under the fast judge and
        ``evolved_score`` is the holdout mean under the LLM judge, so this
        subtracts two different populations measured two different ways. It is
        kept because it has always been reported. Deployment reads
        :attr:`holdout`, which is paired, tested, and measured on one population.
        """
        return self.evolved_score - self.baseline_score

    @property
    def holdout_improvement(self) -> Optional[float]:
        """Raw paired mean difference on the holdout. Point estimate, no noise model.

        Useful for reading the direction at a glance. The accept/reject call
        belongs to :attr:`holdout`, which knows what this number's error bars are.
        """
        if self.holdout_baseline is None or self.holdout_evolved is None:
            return None
        return self.holdout_evolved - self.holdout_baseline

    @property
    def growth(self) -> float:
        """Size change as a fraction of the baseline."""
        base = max(1, len(self.baseline_text))
        return (len(self.evolved_text) - len(self.baseline_text)) / base

    def to_dict(self) -> dict:
        """Serialise the section change with its sizes and growth."""
        return {
            "name": self.name,
            "baseline_chars": len(self.baseline_text),
            "evolved_chars": len(self.evolved_text),
            "growth": round(self.growth, 4),
            "baseline_score": round(self.baseline_score, 4),
            "evolved_score": round(self.evolved_score, 4),
            "improvement": round(self.improvement, 4),
            "holdout_baseline": self.holdout_baseline,
            "holdout_evolved": self.holdout_evolved,
            "holdout": self.holdout.to_dict() if self.holdout else None,
            "accepted": self.accepted,
            "reason": self.reason,
            "elapsed_s": round(self.elapsed_s, 2),
            "optimizer": self.optimizer,
            "validation": self.validation.to_dict() if self.validation else None,
        }


def _effect_text(value: float) -> str:
    """Render a paired Cohen's d without letting a degenerate one fill a column.

    When every scenario moves by the same amount the standard deviation of the
    differences is zero and d is infinite. That is a real, readable fact - the
    change was perfectly consistent - but printed as a sixteen digit number it
    just looks like a bug.
    """
    if value != value:  # NaN
        return "-"
    if abs(value) > 99:
        return "+>99" if value > 0 else "->99"
    return f"{value:+.2f}"


def _banner(text: str) -> None:
    console.print(f"\n[bold]── {text} " + "─" * max(0, 58 - len(text)) + "[/bold]")


def run_gate_ladder(
    hermes_repo: Path,
    updates: dict[str, str],
    strict: bool = False,
    python: Optional[str] = None,
    benchmarks: Sequence[str] = DEFAULT_BENCHMARKS,
    regression_threshold: float = ZERO_REGRESSION_TOLERANCE,
    pytest_subset: Sequence[str] = DEFAULT_PYTEST_SUBSET,
    backup_path: Optional[Path] = None,
    fast: bool = True,
    run_pytest: bool = True,
) -> GateChain:
    """Run the validation ladder against the candidate sections.

    The candidate has to be on disk for pytest or a benchmark to see it - there
    is no ephemeral-prompt channel for either - so the sections are staged, the
    gates run, and the original file is restored no matter how the block exits.
    A copy of the untouched file is written to *backup_path* first.

    Benchmark baselines are measured before staging, then re-measured with the
    candidate in place and compared at *regression_threshold*, which this phase
    sets to 0.0. In today's hermes-agent every benchmark reports UNAVAILABLE;
    that is a pass in permissive mode and a blocker under ``strict``.
    """
    baselines: dict[str, Optional[float]] = {}
    for name in benchmarks:
        result = run_benchmark_gate(hermes_repo, name, fast=fast)
        baselines[name] = result.score

    chain = GateChain(strict=strict)
    with staged_prompt_write(
        hermes_repo, updates, enabled=bool(updates), backup_path=backup_path
    ):
        gates = []
        if run_pytest:
            gates.append(
                partial(
                    run_pytest_gate,
                    hermes_repo,
                    subset=list(pytest_subset),
                    python=python,
                )
            )
        for name in benchmarks:
            gates.append(
                partial(
                    run_benchmark_gate,
                    hermes_repo,
                    name,
                    baseline=baselines[name],
                    regression_threshold=regression_threshold,
                    fast=fast,
                )
            )
        chain.run(*gates)

    return chain


def _optimize_section(
    section_name: str,
    baseline_text: str,
    trainset,
    valset,
    iterations: int,
    optimizer_model: str,
) -> tuple[str, str]:
    """Run GEPA over one section, falling back to MIPROv2.

    Returns the evolved text and the name of the optimizer that produced it.
    The section lives in the signature's instructions, so instruction mutation
    is section mutation and the evolved text reads straight back out.
    """
    module = SectionBehaviorModule(baseline_text, section_name)
    metric = make_behavioral_metric()

    try:
        optimizer = dspy.GEPA(
            metric=metric,
            max_metric_calls=max(1, iterations) * max(1, len(trainset)),
            reflection_lm=dspy.LM(optimizer_model),
            track_stats=False,
        )
        optimized = optimizer.compile(module, trainset=trainset, valset=valset)
        return optimized.section_text, "GEPA"
    except Exception as exc:  # noqa: BLE001 - GEPA availability varies by dspy build
        console.print(f"  [yellow]GEPA unavailable ({exc}), falling back to MIPROv2[/yellow]")

    optimizer = dspy.MIPROv2(metric=metric, auto="light")
    optimized = optimizer.compile(module, trainset=trainset)
    return optimized.section_text, "MIPROv2"


# ──────────────────────────────────────────────────────────────────────────
# Deployment
# ──────────────────────────────────────────────────────────────────────────


PR_PHASE = "Phase 3 (system prompt sections)"

# Branch target when one run deployed several sections. A branch name has to be
# short and stable, and gluing four section names together is neither, so a run
# that ships more than one section ships them under one slug.
MULTI_SECTION_TARGET = "prompt-sections"

# The run wrote and branched, but a step the operator explicitly asked for -
# the push, or opening the PR - did not happen. Distinct from 1 (setup refused
# before anything ran) and 2 (a live session blocked the write).
EXIT_DEPLOYMENT_INCOMPLETE = 3


def pr_target(names: Sequence[str]) -> str:
    """What ``evolve/<target>-<timestamp>`` is named after."""
    unique = list(dict.fromkeys(names))
    if len(unique) == 1:
        return unique[0]
    return MULTI_SECTION_TARGET


def repo_relative(repo: Path, path: Path) -> str:
    """A path git can stage, relative to the repo root when it lives inside it."""
    try:
        return Path(path).relative_to(Path(repo)).as_posix()
    except ValueError:
        return Path(path).as_posix()


def holdout_score_lines(deployed: Sequence[SectionOutcome]) -> list[ScoreLine]:
    """One before/after row per deployed section, taken from the holdout.

    Only the holdout is reported. This phase's train and validation numbers are
    the fast judge's, measured over a different scenario set, so putting them in
    the same table as an LLM-judged holdout mean would invite exactly the
    comparison those numbers cannot support.
    """
    lines: list[ScoreLine] = []
    multiple = len(deployed) > 1
    for outcome in deployed:
        comparison = outcome.holdout
        if comparison is None or not comparison.n:
            continue
        up, down, unchanged = comparison.movement
        lines.append(
            ScoreLine(
                split=f"holdout / {outcome.name}" if multiple else "holdout",
                baseline=comparison.overall.baseline_mean,
                evolved=comparison.overall.candidate_mean,
                detail=(
                    f"paired, n={comparison.n}, "
                    f"p={comparison.overall.wilcoxon_p:.3f}, "
                    f"{up} up / {down} down / {unchanged} unchanged"
                ),
            )
        )
    return lines


def holdout_statistics(deployed: Sequence[SectionOutcome]) -> str:
    """The paired evidence for the PR body, in the comparison's own words.

    Nothing is recomputed here. :class:`HoldoutComparison` already renders the
    delta with its interval, the Wilcoxon p, the effect size, the per-category
    verdicts and the power note, and a second rendering of the same numbers is
    a second chance to disagree with the gate that made the decision.
    """
    blocks: list[str] = []
    for outcome in deployed:
        comparison = outcome.holdout
        if comparison is None or not comparison.n:
            continue
        lines = [
            f"**{outcome.name}** - paired Wilcoxon signed-rank over "
            f"{comparison.n} holdout scenario(s), alpha={comparison.alpha:g}",
            "",
            f"- Overall: {comparison.describe()}",
        ]
        for category, result in comparison.by_category.items():
            label = category
            if category == comparison.targeted_category:
                label = f"{category} (targeted)"
            verdict = "regressed" if comparison.category_regressed(result) else "held"
            lines.append(f"- {label}: {result.describe()} [{verdict}]")
        if comparison.underpowered_categories:
            lines.append(
                "- Too small to test on their own, so watched on the point "
                f"estimate only ({comparison.category_tolerance:.0%} tolerance): "
                + ", ".join(comparison.underpowered_categories)
            )
        lines.append(f"- Power: {comparison.power_note}")
        if outcome.adjusted_p is not None:
            lines.append(
                "- Holm-adjusted p across the sections tested against one "
                f"baseline: {outcome.adjusted_p:.3f}"
            )
        lines.append(f"- Verdict: {comparison.reason}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def rejected_candidates(
    outcomes: Sequence[SectionOutcome], deployed: Sequence[SectionOutcome]
) -> list[RejectedCandidate]:
    """Every section this run produced and refused, and what refused it.

    A reviewer looking at one deployed section cannot otherwise tell that three
    more were tried, that one lost a constraint, or that one cleared alpha on
    its own and was then dropped by the multiple-comparison correction. That
    last one is the most useful line in the list: it says the run was selecting
    from a family, and that the correction rather than the candidate made the
    call.
    """
    shipped = {o.name for o in deployed}
    rejected: list[RejectedCandidate] = []
    for outcome in outcomes:
        if outcome.name in shipped:
            continue
        reason = outcome.reason or "no holdout evidence"
        if outcome.adjusted_p is not None and outcome.adjusted_p >= HOLDOUT_ALPHA:
            reason = f"dropped by the Holm correction - {reason}"
        rejected.append(RejectedCandidate(label=outcome.name, reason=reason))
    return rejected


def gate_lines(chain: Optional[GateChain]) -> list[str]:
    """The gate ladder as flat text, including the gates that could not run."""
    if chain is None:
        return ["not reached - no candidate survived to be gated"]
    if not chain.results:
        return ["no gates ran"]
    return [f"{r.name} ({r.status.value}): {r.message}" for r in chain.results]


def emit_pull_request(
    *,
    repo: Path,
    files: Sequence[str],
    deployed: Sequence[SectionOutcome],
    outcomes: Sequence[SectionOutcome],
    timestamp: str,
    output_dir: Path,
    cost: Optional[CostReport] = None,
    chain: Optional[GateChain] = None,
    dataset: str = "",
    iterations: Optional[int] = None,
    notes: Sequence[str] = (),
    push: bool = False,
    open_pr: bool = False,
    base: str = "main",
) -> tuple[Optional[PullRequestPlan], int]:
    """Branch, commit and render the pull request for a run that just wrote.

    Local by default and local by design. The branch and ``PULL_REQUEST.md`` are
    made here; the network is touched only for the steps the operator named. The
    checkout is returned to the ref it started on in a ``finally``, so a failure
    part way through leaves the reviewer where they were rather than parked on a
    branch they did not ask for.

    Returns the plan, or ``None`` when no branch could be made, and a process
    exit code that is non-zero only when something explicitly requested did not
    happen. A repo that is not a git checkout is reported and shrugged off: the
    sections are already on disk, and failing the run after a successful write
    would say something untrue about the write.
    """
    target = pr_target([o.name for o in deployed])
    try:
        plan = build_pull_request(
            repo=repo,
            target=target,
            phase=PR_PHASE,
            timestamp=timestamp,
            files=list(files),
            scores=holdout_score_lines(deployed),
            cost=cost,
            rejected=rejected_candidates(outcomes, deployed),
            gates=gate_lines(chain),
            dataset=dataset,
            optimizer=", ".join(
                dict.fromkeys(o.optimizer for o in deployed if o.optimizer)
            ),
            iterations=iterations,
            statistics=holdout_statistics(deployed),
            notes=list(notes),
        )
    except GitError as exc:
        console.print(f"  [yellow]○ No pull request: {exc}[/yellow]")
        console.print(
            "  The evolved section(s) are on disk and the run's artifacts are "
            f"in {output_dir}/, so the change can still be reviewed by hand."
        )
        return None, 0

    code = 0
    try:
        body_path = plan.write_body(output_dir)
        console.print(f"  [green]✓ Branch {plan.branch}[/green] ({len(files)} file(s))")
        console.print(f"  PR body: {body_path}")

        if push:
            try:
                plan.push()
                console.print(f"  [green]✓ Pushed {plan.branch} to origin[/green]")
            except GitError as exc:
                console.print(f"  [red]✗ Push failed: {exc}[/red]")
                console.print(
                    "  The branch and the body are on this machine; nothing "
                    "reached the remote."
                )
                return plan, EXIT_DEPLOYMENT_INCOMPLETE
        else:
            console.print("  Not pushed (--no-push is the default).")

        if open_pr:
            try:
                plan.open(base=base)
                console.print(f"  [green]✓ Opened a pull request against {base}[/green]")
            except GitError as exc:
                console.print(f"  [red]✗ Could not open the pull request: {exc}[/red]")
                return plan, EXIT_DEPLOYMENT_INCOMPLETE
        else:
            console.print("  No pull request opened (--no-open-pr is the default).")
    finally:
        # Whatever happened above, the operator gets their checkout back. The
        # commit lives on the evolve branch; leaving someone on it after a
        # failed push is how an unrelated later commit ends up in this PR.
        try:
            plan.restore()
            if plan.original_ref:
                console.print(f"  Checkout restored to {plan.original_ref}.")
        except GitError as exc:
            console.print(
                f"  [red]✗ Could not restore the checkout to "
                f"{plan.original_ref}: {exc}[/red]"
            )
            console.print(f"  You are still on {plan.branch}.")
            code = EXIT_DEPLOYMENT_INCOMPLETE

    return plan, code


def _save_metrics(output_dir: Path, metrics: dict) -> Path:
    path = output_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def evolve(
    section_names: Sequence[str] = (),
    all_sections: bool = False,
    iterations: int = 5,
    hermes_repo: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    strict_gates: bool = False,
    dry_run: bool = False,
    write: bool = False,
    max_turns: int = 6,
    num_workers: int = 4,
    seed: int = 0,
    create_pr: bool = True,
    push: bool = False,
    open_pr: bool = False,
    pr_base: str = "main",
) -> int:
    """Optimize one or more system prompt sections. Returns a process exit code."""

    console.print(
        "\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] "
        "- Phase 3: system prompt sections\n"
    )

    # ── 0. Refuse impossible deployment combinations before spending ─────
    #
    # Checked here, at the top, because the alternative is discovering after a
    # paid optimization run that the thing the operator asked for at the end
    # was never going to happen.
    if (push or open_pr) and not create_pr:
        console.print(
            "[red]✗ --no-create-pr leaves no branch to push or open a pull "
            "request from[/red]"
        )
        return 1
    if open_pr and not push:
        console.print(
            "[red]✗ --open-pr needs --push: gh cannot open a pull request for a "
            "branch that exists only on this machine[/red]"
        )
        return 1
    if (push or open_pr) and not write:
        console.print(
            "[red]✗ --push and --open-pr need --write: a pull request is only "
            "built when a run actually deploys something[/red]"
        )
        return 1

    # ── 1. Resolve the repo and discover sections ───────────────────────
    _banner("Discovering sections")
    try:
        repo = resolve_hermes_agent_path(hermes_repo)
    except FileNotFoundError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 1

    config = EvolutionConfig(
        hermes_agent_path=repo,
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
    )

    try:
        inventory = load_sections(repo, max_growth=config.max_prompt_growth)
    except UnknownSection as exc:
        console.print(f"[red]✗ {exc}[/red]")
        return 1

    if not inventory.sections:
        console.print(
            f"[red]✗ No evolvable prompt sections found under {repo}[/red]\n"
            f"  Expected string constants "
            f"({', '.join(EVOLVABLE_PROMPT_SECTIONS)}) in agent/prompt_builder.py"
        )
        return 1

    console.print(f"  Repo: {repo}")
    console.print(f"  File: {inventory.prompt_builder}")
    for section in inventory:
        console.print(
            f"  [green]✓[/green] {section.name}: {section.baseline_size:,} chars "
            f"(~{section.baseline_tokens:,} tokens, ceiling {section.max_chars:,})"
        )
    for missing in inventory.missing:
        console.print(f"  [yellow]○ {missing}: not found in this checkout[/yellow]")
    for structured in inventory.structured:
        console.print(f"  [yellow]○ {structured.name}: {structured.reason}[/yellow]")

    console.print(
        f"  Assembled prefix: ~{inventory.estimated_tokens():,} tokens "
        f"in {inventory.cache_blocks()} cache block(s) of "
        f"{inventory.cache_budget_tokens:,} budgeted"
    )

    # ── 2. Select targets ────────────────────────────────────────────────
    requested = list(section_names)
    if all_sections:
        requested = list(inventory.names)
    if not requested:
        console.print(
            "[red]✗ Nothing to do: pass --section NAME (repeatable) or --all-sections[/red]"
        )
        return 1

    unknown = validate_section_names(requested)
    if unknown:
        for name in unknown:
            hint = ""
            if name in {s.name for s in inventory.structured}:
                hint = " (present in prompt_builder.py, but not a plain string)"
            console.print(f"[red]✗ {name} is not an evolvable prompt section{hint}[/red]")
        console.print(f"  Allowed: {', '.join(EVOLVABLE_PROMPT_SECTIONS)}")
        return 1

    targets = []
    for name in requested:
        try:
            targets.append(inventory.get(name))
        except UnknownSection as exc:
            console.print(f"[red]✗ {exc}[/red]")
            return 1

    console.print(f"\n  Targeting: {', '.join(t.name for t in targets)}")

    # ── 3. Behavioral scenarios ──────────────────────────────────────────
    _banner("Building behavioral suite")
    target_names = [t.name for t in targets]
    suite = BehavioralSuite.from_seeds(
        sections=target_names + ["PLATFORM_HINTS"],
        include_platform=True,
    )
    train, val, holdout = suite.split(
        train_ratio=config.train_ratio, val_ratio=config.val_ratio, seed=seed
    )
    console.print(f"  Scenarios: {len(suite)} across {len(suite.categories())} categories")
    for category, count in suite.categories().items():
        console.print(f"    {category}: {count}")
    console.print(f"  Split: {len(train)} train / {len(val)} val / {len(holdout)} holdout")

    harness, harness_reason = select_harness(
        repo, model=eval_model, max_turns=max_turns, num_workers=num_workers
    )
    console.print(f"  Harness: {harness_reason}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / "prompts" / timestamp

    if dry_run:
        _banner("Dry run")
        console.print("  Setup validated. Nothing was optimized and nothing was written.")
        console.print(f"  Would optimize: {', '.join(target_names)}")
        console.print(f"  Would run {iterations} iteration(s) with {optimizer_model}")
        console.print(
            f"  Would gate on pytest + {', '.join(DEFAULT_BENCHMARKS)} at "
            f"{ZERO_REGRESSION_TOLERANCE:.0%} regression tolerance"
            f"{' (strict)' if strict_gates else ''}"
        )
        console.print(
            f"  Would deploy only on a significant paired improvement of at least "
            f"{PRACTICAL_IMPROVEMENT:.0%} with no category regression past "
            f"{CATEGORY_REGRESSION_TOLERANCE:.0%}"
        )
        # Worth knowing before the money is spent, not after: a holdout this
        # size may not be able to support the claim the run is going to make.
        console.print(
            f"  Holdout power: {len(holdout)} scenario(s), smallest detectable shift "
            f"{min_detectable_paired_shift(len(holdout), HOLDOUT_ALPHA):.0%} of the suite"
        )
        if len(holdout) < min_scenarios_for_significance(HOLDOUT_ALPHA):
            console.print(
                f"  [yellow]○ Underpowered: fewer than "
                f"{min_scenarios_for_significance(HOLDOUT_ALPHA)} holdout scenarios, so "
                f"nothing this run measures could be called significant[/yellow]"
            )
        console.print(f"  Would write output to {output_dir}/")
        console.print(
            "  Would build a branch and a PR body only after an actual write "
            f"({'--create-pr' if create_pr else '--no-create-pr'}); a dry run "
            "builds neither, and nothing is ever pushed without --push."
        )
        console.print(f"  [dim]{NEXT_SESSION_NOTICE}[/dim]")
        return 0

    # ── 4. Baseline behaviour ────────────────────────────────────────────
    #
    # The cost meter opens here and stops once the last holdout comparison is
    # in. That is exactly the stretch of the run that calls a model - the
    # baseline sweep, the optimizer, and the holdout - and PLAN.md wants the
    # figure in the PR body. An ExitStack rather than a `with` block because
    # the measured stretch covers five numbered stages, and wrapping all of
    # them in another level of indentation would bury the structure.
    cost_meter = ExitStack()
    usage = cost_meter.enter_context(UsageTracker())

    _banner("Baseline behaviour")
    lm = dspy.LM(eval_model, temperature=0.0)
    dspy.configure(lm=lm)
    fast_judge = BehavioralJudge(use_llm=False)

    baseline_prompt = inventory.assembled_prompt()
    baseline_reports: dict[str, BehavioralReport] = {}
    for section in targets:
        scenarios = [s for s in val if s.section_under_test == section.name] or [
            s for s in suite.for_section(section.name)
        ]
        report = suite.evaluate(
            baseline_prompt,
            harness,
            judge=fast_judge,
            section_name=section.name,
            run_name=f"hase-baseline-{section.name.lower()}-{timestamp}",
            scenarios=scenarios,
        )
        baseline_reports[section.name] = report
        console.print(
            f"  {section.name}: {report.mean_score:.3f} mean over {len(report)} scenario(s) "
            f"({report.pass_rate:.0%} pass)"
        )

    # ── 5. Optimize ──────────────────────────────────────────────────────
    _banner(f"Optimizing ({iterations} iteration(s))")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    outcomes: list[SectionOutcome] = []
    for section in targets:
        section_train = [s for s in train if s.section_under_test == section.name]
        section_val = [s for s in val if s.section_under_test == section.name]
        if not section_train:
            console.print(
                f"  [yellow]○ {section.name}: no training scenarios, skipping[/yellow]"
            )
            continue

        console.print(
            f"\n  [cyan]{section.name}[/cyan] "
            f"({len(section_train)} train / {len(section_val)} val)"
        )
        started = time.time()
        evolved_text, optimizer_name = _optimize_section(
            section_name=section.name,
            baseline_text=section.baseline_text,
            trainset=scenarios_to_dspy_examples(section_train),
            valset=scenarios_to_dspy_examples(section_val or section_train),
            iterations=iterations,
            optimizer_model=optimizer_model,
        )
        elapsed = time.time() - started

        outcome = SectionOutcome(
            name=section.name,
            baseline_text=section.baseline_text,
            evolved_text=evolved_text,
            baseline_score=baseline_reports[section.name].mean_score,
            elapsed_s=elapsed,
            optimizer=optimizer_name,
        )
        console.print(
            f"  {optimizer_name} finished in {elapsed:.1f}s "
            f"({len(section.baseline_text):,} -> {len(evolved_text):,} chars)"
        )
        outcomes.append(outcome)

    if not outcomes:
        cost_meter.close()
        console.print("[red]✗ No section had scenarios to optimize against[/red]")
        console.print(f"  Spent getting here: {usage.report.describe()}")
        return 1

    # ── 6. Constraints ───────────────────────────────────────────────────
    _banner("Constraints")
    for outcome in outcomes:
        validation = inventory.validate(outcome.name, outcome.evolved_text)
        outcome.validation = validation
        console.print(f"  [cyan]{outcome.name}[/cyan]")
        for check in validation.checks:
            colour = "green" if check.passed else ("yellow" if check.is_warning else "red")
            console.print(f"    [{colour}]{'✓' if check.passed else '✗'} {check.name}[/{colour}]: {check.message}")

        if not validation.passed:
            outcome.accepted = False
            outcome.reason = "failed constraints: " + ", ".join(
                c.name for c in validation.errors
            )
        elif strict_gates and not validation.passed_strict:
            outcome.accepted = False
            outcome.reason = "strict mode: " + ", ".join(
                c.name for c in validation.warnings
            )
        else:
            outcome.accepted = True
            outcome.reason = "constraints passed"

    survivors = [o for o in outcomes if o.accepted]
    for rejected in (o for o in outcomes if not o.accepted):
        console.print(f"  [red]✗ {rejected.name} rejected: {rejected.reason}[/red]")

    # ── 7. Gate ladder ───────────────────────────────────────────────────
    _banner("Gate ladder")
    chain: Optional[GateChain] = None
    if not survivors:
        console.print("  [yellow]○ Skipped: no candidate survived the constraints[/yellow]")
    else:
        updates = {o.name: o.evolved_text for o in survivors}
        console.print(
            f"  Staging {len(updates)} section(s), regression tolerance "
            f"{ZERO_REGRESSION_TOLERANCE:.0%}{' , strict' if strict_gates else ''}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        backup_path = output_dir / "prompt_builder.py.bak"
        chain = run_gate_ladder(
            hermes_repo=repo,
            updates=updates,
            strict=strict_gates,
            python=sys.executable,
            backup_path=backup_path,
        )
        console.print(f"  Backup of the original: {backup_path}")
        for result in chain.results:
            colour = {
                "passed": "green",
                "failed": "red",
                "unavailable": "yellow",
                "skipped": "dim",
            }[result.status.value]
            console.print(f"  [{colour}]{result.name}[/{colour}]: {result.message}")

        if not chain.passed:
            blockers = ", ".join(r.name for r in chain.blockers)
            console.print(f"  [red]✗ Gate ladder blocked on: {blockers}[/red]")
            for outcome in survivors:
                outcome.accepted = False
                outcome.reason = f"gate blocked: {blockers}"
            survivors = []

    # ── 8. Holdout ───────────────────────────────────────────────────────
    #
    # Every survivor is measured against the SAME baseline run over the SAME
    # holdout scenarios in the SAME order. That is what makes the comparison
    # paired, and the pairing is where the statistical power comes from: with a
    # holdout this small, an unpaired comparison of two means could not detect
    # anything short of a collapse.
    #
    # The whole holdout is used, not just the scenarios that belong to the
    # section being edited. A section rewrite goes into the one prompt every
    # category is answered under, so the categories it was not aimed at are the
    # ones worth watching - they are where collateral damage shows up.
    _banner("Holdout")
    deployable: list[SectionOutcome] = []
    if not survivors:
        console.print("  [yellow]○ Skipped: nothing survived to evaluate[/yellow]")
    elif not holdout:
        console.print("  [yellow]○ Skipped: the split left no holdout scenarios[/yellow]")
        for outcome in survivors:
            outcome.accepted = False
            outcome.reason = "no holdout evidence"
    else:
        holdout_judge = BehavioralJudge(model=eval_model, use_llm=True)
        base_report = suite.evaluate(
            baseline_prompt,
            harness,
            judge=holdout_judge,
            section_name=HOLDOUT_SECTION_LABEL,
            run_name=f"hase-holdout-base-{timestamp}",
            scenarios=holdout,
        )
        holdout_categories = sorted({s.category for s in holdout})
        console.print(
            f"  Baseline: {base_report.mean_score:.3f} over {len(holdout)} scenario(s) "
            f"in {len(holdout_categories)} categor{'y' if len(holdout_categories) == 1 else 'ies'} "
            f"({', '.join(holdout_categories)})"
        )
        console.print(
            f"  Power: {min_detectable_paired_shift(len(holdout), HOLDOUT_ALPHA):.0%} of the "
            f"suite is the smallest shift an exact paired test could call significant; "
            f"significance needs at least "
            f"{min_scenarios_for_significance(HOLDOUT_ALPHA)} scenario(s)"
        )

        for outcome in survivors:
            evolved_prompt = inventory.assembled_prompt({outcome.name: outcome.evolved_text})
            evolved_report = suite.evaluate(
                evolved_prompt,
                harness,
                judge=holdout_judge,
                section_name=HOLDOUT_SECTION_LABEL,
                run_name=f"hase-holdout-evolved-{outcome.name.lower()}-{timestamp}",
                scenarios=holdout,
            )
            try:
                comparison = compare_holdout(
                    base_report,
                    evolved_report,
                    targeted_category=SECTION_CATEGORIES.get(outcome.name, ""),
                )
            except UnpairedHoldout as exc:
                outcome.accepted = False
                outcome.reason = f"holdout pairing broken: {exc}"
                console.print(f"  [red]✗ {outcome.name}: {outcome.reason}[/red]")
                continue

            outcome.holdout = comparison
            outcome.holdout_baseline = comparison.overall.baseline_mean
            outcome.holdout_evolved = comparison.overall.candidate_mean
            outcome.evolved_score = comparison.overall.candidate_mean

            console.print(f"\n  [cyan]{outcome.name}[/cyan]: {comparison.describe()}")
            for category, result in comparison.by_category.items():
                regressed = comparison.category_regressed(result)
                colour = "red" if regressed else "green"
                icon = "✗" if regressed else "✓"
                target = " (targeted)" if category == comparison.targeted_category else ""
                console.print(
                    f"    [{colour}]{icon} {category}{target}[/{colour}]: {result.describe()}"
                )
            if comparison.underpowered_categories:
                console.print(
                    f"    [yellow]○ too small to test: "
                    f"{', '.join(comparison.underpowered_categories)} - a category "
                    f"regression there can only be caught on the point estimate "
                    f"({CATEGORY_REGRESSION_TOLERANCE:.0%} tolerance)[/yellow]"
                )
            console.print(f"    Power: {comparison.power_note}")

            if comparison.accepted:
                outcome.accepted = True
                outcome.reason = comparison.reason
                deployable.append(outcome)
            else:
                outcome.accepted = False
                outcome.reason = comparison.reason
                console.print(f"    [yellow]○ not deployable: {comparison.reason}[/yellow]")

    # ── 8b. Correct for testing several sections against one baseline ────
    #
    # Each section above was tested at alpha on its own. Deploying every section
    # that clears is a disjunction - "any of these worked" - and selecting the
    # best of k inflates the family-wise error: four sections at 0.05 give
    # 1 - 0.95**4 = 18.5%, not 5%. Holm-adjust the surviving p-values and drop
    # any that no longer clear.
    #
    # This is the opposite call from the per-category guard a few lines above,
    # deliberately. That one is a conjunction - accepting means every category
    # held - which is an intersection-union test and is already valid at alpha.
    # Correcting a conjunction would loosen the gate as the catalogue grows.
    if len(deployable) > 1:
        raw = [o.holdout.overall.wilcoxon_p for o in deployable]
        adjusted = holm_adjust(raw)
        # Not named `survivors`: that list still means "passed the constraints
        # and the gate ladder", and the write-back report walks it to announce
        # which of those never became deployable. Rebinding it here would
        # silence that announcement whenever more than one section got this far.
        corrected: list = []
        for outcome, raw_p, adj_p in zip(list(deployable), raw, adjusted):
            outcome.adjusted_p = adj_p
            if adj_p < HOLDOUT_ALPHA:
                corrected.append(outcome)
                continue
            outcome.accepted = False
            outcome.reason = (
                f"p={raw_p:.3f} alone, but {adj_p:.3f} after correcting for "
                f"{len(raw)} sections tested against one baseline"
            )
            console.print(
                f"  [yellow]○ {outcome.name}: dropped by the "
                f"multiple-comparison correction ({outcome.reason})[/yellow]"
            )
        if len(corrected) != len(deployable):
            console.print(
                f"  [dim]Holm correction over {len(raw)} sections: "
                f"{len(corrected)} of {len(raw)} survive at alpha={HOLDOUT_ALPHA}[/dim]"
            )
        deployable = corrected

    # Everything that could call a model has run.
    cost_meter.close()
    cost = usage.report

    # ── 9. Results ───────────────────────────────────────────────────────
    table = Table(title="Phase 3 - System Prompt Evolution")
    table.add_column("Section", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Growth", justify="right")
    table.add_column("Base", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Δ (95% CI)", justify="right")
    table.add_column("p", justify="right")
    table.add_column("d", justify="right")
    table.add_column("Verdict")

    for outcome in outcomes:
        comparison = outcome.holdout
        change_text = "-"
        p_text = "-"
        d_text = "-"
        if comparison is not None and comparison.n:
            delta = comparison.overall.delta
            ci = comparison.overall.delta_ci
            colour = "green" if delta > 0 else ("dim" if delta == 0 else "red")
            change_text = (
                f"[{colour}]{delta:+.3f}[/{colour}] "
                f"[{ci.low:+.3f}, {ci.high:+.3f}]"
            )
            p_text = f"{comparison.overall.wilcoxon_p:.3f}"
            d_text = _effect_text(comparison.overall.effect_size)
        if outcome.accepted:
            verdict = "[green]accepted[/green]"
        elif comparison is not None:
            verdict = f"[red]{comparison.headline}[/red]"
        else:
            verdict = f"[red]{outcome.reason}[/red]"
        table.add_row(
            outcome.name,
            f"{len(outcome.baseline_text):,} → {len(outcome.evolved_text):,}",
            f"{outcome.growth:+.1%}",
            "-" if outcome.holdout_baseline is None else f"{outcome.holdout_baseline:.3f}",
            "-" if outcome.holdout_evolved is None else f"{outcome.holdout_evolved:.3f}",
            change_text,
            p_text,
            d_text,
            verdict,
        )

    console.print()
    console.print(table)

    # Per-category verdicts get their own table. An aggregate row cannot show
    # "everything improved except platform formatting, which fell off a cliff",
    # and that is the failure this phase most needs to see.
    measured = [o for o in outcomes if o.holdout is not None and o.holdout.n]
    if measured:
        categories = Table(title="Holdout by category (paired, per section)")
        categories.add_column("Section", style="bold")
        categories.add_column("Category")
        categories.add_column("n", justify="right")
        categories.add_column("Base", justify="right")
        categories.add_column("Evolved", justify="right")
        categories.add_column("Δ", justify="right")
        categories.add_column("p", justify="right")
        categories.add_column("Verdict")

        for outcome in measured:
            comparison = outcome.holdout
            for category, result in comparison.by_category.items():
                regressed = comparison.category_regressed(result)
                if regressed:
                    note = "[red]✗ regressed[/red]"
                elif result.n < min_scenarios_for_significance(comparison.alpha):
                    note = "[yellow]○ held (too small to test)[/yellow]"
                else:
                    note = "[green]✓ held[/green]"
                label = category
                if category == comparison.targeted_category:
                    label = f"{category} (targeted)"
                categories.add_row(
                    outcome.name,
                    label,
                    str(result.n),
                    f"{result.baseline_mean:.3f}",
                    f"{result.candidate_mean:.3f}",
                    f"{result.delta:+.3f}",
                    f"{result.wilcoxon_p:.3f}",
                    note,
                )

        console.print()
        console.print(categories)

        underpowered = [o for o in measured if o.holdout.underpowered]
        if underpowered:
            console.print(
                f"\n  [yellow]○ Underpowered: {', '.join(o.name for o in underpowered)} "
                f"ran on fewer than {min_scenarios_for_significance(HOLDOUT_ALPHA)} holdout "
                f"scenarios. No verdict from this run can be called significant, whatever "
                f"the point estimates say.[/yellow]"
            )

    # ── 9c. Cost ─────────────────────────────────────────────────────────
    #
    # PLAN.md wants this figure in the PR body, and a reviewer deciding whether
    # the run was worth repeating wants it on the console. It is reported
    # whether or not anything is deployable: a run that spent money and shipped
    # nothing is exactly the run whose cost is worth knowing.
    _banner("Cost")
    console.print(f"  {cost.describe()}")
    for model, count in cost.models.items():
        console.print(f"    {model}: {count} call(s)")
    if not cost.complete:
        console.print(
            "  [yellow]○ Incomplete: the figure above is a lower bound, not the "
            "full price of this run[/yellow]"
        )

    # ── 10. Save ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    for outcome in outcomes:
        (output_dir / f"baseline_{outcome.name}.txt").write_text(
            outcome.baseline_text, encoding="utf-8"
        )
        (output_dir / f"evolved_{outcome.name}.txt").write_text(
            outcome.evolved_text, encoding="utf-8"
        )
    suite.save(output_dir / "scenarios.jsonl")
    metrics = {
        "timestamp": timestamp,
        "hermes_repo": str(repo),
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "strict_gates": strict_gates,
        "regression_threshold": ZERO_REGRESSION_TOLERANCE,
        "harness": harness_reason,
        "holdout_test": {
            "test": "paired Wilcoxon signed-rank with a seeded paired bootstrap CI",
            "alpha": HOLDOUT_ALPHA,
            "practical_threshold": PRACTICAL_IMPROVEMENT,
            "category_tolerance": CATEGORY_REGRESSION_TOLERANCE,
            "bootstrap_seed": HOLDOUT_BOOTSTRAP_SEED,
            "min_scenarios_for_significance": min_scenarios_for_significance(HOLDOUT_ALPHA),
            "min_detectable_shift": round(
                min_detectable_paired_shift(len(holdout), HOLDOUT_ALPHA), 6
            ),
            "multiplicity_correction": (
                "none - per-category non-regression is an intersection-union "
                "conjunction, valid at alpha without adjustment"
            ),
        },
        "scenarios": {
            "total": len(suite),
            "train": len(train),
            "val": len(val),
            "holdout": len(holdout),
            "by_category": suite.categories(),
        },
        "inventory": inventory.to_dict(),
        "gates": chain.to_dict() if chain else None,
        "cost": cost.to_dict(),
        "sections": [o.to_dict() for o in outcomes],
        # Filled in below when a write actually produced a branch.
        "pull_request": None,
    }
    _save_metrics(output_dir, metrics)
    console.print(f"\n  Output saved to {output_dir}/")

    # ── 11. Write-back ───────────────────────────────────────────────────
    _banner("Write-back")
    console.print(f"  [dim]{NEXT_SESSION_NOTICE}[/dim]")

    # Holdout evidence is the price of admission for a write, and "evidence"
    # means the paired test, not the sign of a subtraction. This tier is too
    # wide to deploy on a training-set improvement, and a delta the test cannot
    # separate from noise is not evidence of anything either.
    improved = list(deployable)
    deployable_names = {o.name for o in improved}
    for outcome in survivors:
        if outcome.name in deployable_names:
            continue
        console.print(
            f"  [yellow]○ {outcome.name}: {outcome.reason or 'no holdout evidence'}, "
            f"not deployable[/yellow]"
        )

    if not write:
        console.print("  No write requested (--no-write is the default).")
        if improved:
            console.print(
                f"  Review first: diff {output_dir}/baseline_<SECTION>.txt "
                f"{output_dir}/evolved_<SECTION>.txt"
            )
            console.print("  Then re-run with --write to apply.")
        return 0

    if not improved:
        console.print(
            "  [yellow]Nothing to write: no section cleared the paired holdout test."
            "[/yellow]"
        )
        return 0

    session = detect_active_session()
    if session.active:
        console.print(f"  [red]✗ Refusing to write: {session.summary}[/red]")
        console.print(
            "  A live session has already assembled and cached its system prompt. "
            "Evolved sections deploy on the next session, so finish or stop the "
            "current one and re-run with --write."
        )
        return 2

    updates = {o.name: o.evolved_text for o in improved}
    result = write_sections(repo, updates)
    console.print(
        f"  [green]✓ Wrote {', '.join(result.updated)} to {result.path}[/green] "
        f"({result.before_chars:,} -> {result.after_chars:,} chars)"
    )
    console.print("  Neighbouring constants verified unchanged.")
    console.print(f"  [bold]{NEXT_SESSION_NOTICE}[/bold]")

    # ── 12. Pull request ─────────────────────────────────────────────────
    #
    # PLAN.md constraint 5: evolved content reaches hermes-agent as a branch
    # and a reviewable PR, never as a direct commit. Only reached because the
    # write above happened - no write, no branch, and a dry run never gets
    # here at all.
    if not create_pr:
        console.print("\n  No pull request requested (--no-create-pr).")
        console.print(
            f"  The write is in your working tree at {result.path}; commit it "
            "yourself or re-run with --create-pr."
        )
        return 0

    _banner("Pull request")
    plan, code = emit_pull_request(
        repo=repo,
        files=[repo_relative(repo, result.path)],
        deployed=improved,
        outcomes=outcomes,
        timestamp=timestamp,
        output_dir=output_dir,
        cost=cost,
        chain=chain,
        dataset=(
            f"behavioural suite, {len(suite)} scenario(s) "
            f"({len(train)} train / {len(val)} val / {len(holdout)} holdout), "
            f"judged by {eval_model}"
        ),
        iterations=iterations,
        notes=[
            NEXT_SESSION_NOTICE,
            f"Evaluation harness: {harness_reason}",
            f"Gate ladder regression tolerance: {ZERO_REGRESSION_TOLERANCE:.0%}"
            f"{', strict' if strict_gates else ''}",
            "Run artifacts, including the baseline of every section and the "
            f"scenarios used: {output_dir}/",
        ],
        push=push,
        open_pr=open_pr,
        base=pr_base,
    )
    if plan is not None:
        metrics["pull_request"] = plan.to_dict()
        _save_metrics(output_dir, metrics)
    return code


@click.command()
@click.option(
    "--section",
    "sections",
    multiple=True,
    help="Prompt section to evolve; repeatable. Must be on the evolvable allowlist.",
)
@click.option("--all-sections", is_flag=True, help="Evolve every discovered section")
@click.option("--iterations", default=5, help="Optimizer iterations per section")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluation and judging")
@click.option(
    "--strict-gates",
    is_flag=True,
    help="Treat an unavailable benchmark or a cache block crossing as blocking",
)
@click.option("--dry-run", is_flag=True, help="Validate setup without optimizing or writing")
@click.option(
    "--write/--no-write",
    default=False,
    help="Write accepted sections back to prompt_builder.py (default: no-write)",
)
@click.option(
    "--create-pr/--no-create-pr",
    default=True,
    help=(
        "On a write, branch and render PULL_REQUEST.md locally "
        "(default: create-pr). Nothing leaves this machine either way."
    ),
)
@click.option(
    "--push/--no-push",
    default=False,
    help="Push the evolve branch to origin (default: no-push)",
)
@click.option(
    "--open-pr/--no-open-pr",
    default=False,
    help="Open the pull request with gh; needs --push (default: no-open-pr)",
)
@click.option("--pr-base", default="main", help="Base branch for the pull request")
def main(
    sections,
    all_sections,
    iterations,
    hermes_repo,
    optimizer_model,
    eval_model,
    strict_gates,
    dry_run,
    write,
    create_pr,
    push,
    open_pr,
    pr_base,
):
    """Evolve hermes-agent system prompt sections with DSPy + GEPA."""
    code = evolve(
        section_names=list(sections),
        all_sections=all_sections,
        iterations=iterations,
        hermes_repo=hermes_repo,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        strict_gates=strict_gates,
        dry_run=dry_run,
        write=write,
        create_pr=create_pr,
        push=push,
        open_pr=open_pr,
        pr_base=pr_base,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()

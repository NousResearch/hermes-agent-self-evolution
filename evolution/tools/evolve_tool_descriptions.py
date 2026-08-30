"""Evolve hermes-agent tool descriptions with DSPy + GEPA.

PLAN.md Phase 2. The pipeline, in order:

    catalogue -> selection dataset -> baseline -> GEPA -> constraints ->
    cross-tool regression check -> gate ladder -> holdout -> write-back ->
    pull request

Five rules shape the whole thing:

1. **All tools are always evaluated together.** Even when ``--tool`` narrows
   what may be rewritten, the selector sees the entire catalogue and the
   cross-tool guard scores every tool. Optimizing one description in isolation
   is how one tool's gain quietly becomes another's loss.
2. **Writing to a real checkout is opt-in.** ``--no-write`` is the default and
   still runs the complete rewrite through ``artifact_io`` in dry-run mode, so a
   run that reports a clean write really would have written cleanly.
3. **A gate that could not run says so.** hermes-agent ships no benchmarks
   today, so the TBLite gate reports UNAVAILABLE. ``--strict-gates`` turns that
   into a blocker for anyone who needs the gate to have actually run.
4. **A write is deployed as a branch, never as a commit on your ref.** When the
   run actually writes, it builds ``evolve/<target>-<timestamp>``, commits the
   modified hermes-agent files onto it, drops PULL_REQUEST.md next to the run's
   other artifacts, and puts the checkout back on the ref it started from. No
   write means no branch, and a dry run builds nothing.
5. **Nothing reaches the network unless it was asked for.** Building the branch
   and writing the body are local. ``--push`` and ``--open-pr`` are separate
   flags, both off by default, so a run cannot phone out to GitHub by accident.

Usage:
    python -m evolution.tools.evolve_tool_descriptions --dry-run
    python -m evolution.tools.evolve_tool_descriptions --toolset file --iterations 8
    python -m evolution.tools.evolve_tool_descriptions --tool read_file --tool search_files --write
    python -m evolution.tools.evolve_tool_descriptions --write --push --open-pr
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.constraints import ConstraintValidator
from evolution.core.cost import UsageTracker
from evolution.core.gates import (
    GateChain,
    GateResult,
    GateStatus,
    run_benchmark_gate,
    run_pytest_gate,
)
from evolution.core.pr_builder import (
    GitError,
    require_clean_worktree,
    PullRequestPlan,
    RejectedCandidate,
    ScoreLine,
    build_pull_request,
)
from evolution.tools.accuracy import (
    DescriptionEntailment,
    FactualAccuracyChecker,
    facts_from_catalog,
)
from evolution.tools.cross_tool import (
    DEFAULT_TOLERANCE,
    CrossToolGuard,
    CrossToolReport,
    CrossToolVerdict,
)
from evolution.tools.selection_eval import (
    NO_TOOL,
    ToolSelectionDataset,
    ToolSelectionDatasetBuilder,
    ToolSelector,
    catalog_signatures,
    evaluate_selection,
    extract_bundle,
    gepa_selection_metric,
    selector_predict_fn,
    tool_selection_metric,
)
from evolution.tools.tool_catalog import (
    ToolCatalog,
    ToolDescriptions,
    UnknownTool,
    bundle_to_dict,
    diff_bundles,
    load_catalog,
    write_bundle,
)

console = Console()

__all__ = [
    "EXIT_DEPLOYMENT_INCOMPLETE",
    "ConstraintOutcome",
    "benchmark_candidate",
    "build_accuracy_checker",
    "collect_rejections",
    "deployment_incomplete",
    "enforce_constraints",
    "freeze_unselected",
    "pr_target_slug",
    "score_lines",
    "evolve_tool_descriptions",
    "main",
]


def _banner(text: str) -> None:
    console.print(f"\n[bold]── {text} ─[/bold]")


# ──────────────────────────────────────────────────────────────────────────
# Candidate hygiene
# ──────────────────────────────────────────────────────────────────────────


def freeze_unselected(
    candidate: dict[str, ToolDescriptions],
    baseline: dict[str, ToolDescriptions],
    allowed: Sequence[str],
) -> dict[str, ToolDescriptions]:
    """Keep only the tools the run was asked to touch; restore the rest.

    An optimizer handed the whole catalogue will happily rewrite a tool nobody
    asked about. That is out of scope for the run, so it is reverted before the
    candidate is scored: the cross-tool comparison then measures the effect of
    the requested change and nothing else.
    """
    permitted = set(allowed)
    merged: dict[str, ToolDescriptions] = {}
    for tool_name, base in baseline.items():
        proposed = candidate.get(tool_name)
        if proposed is None or tool_name not in permitted:
            merged[tool_name] = base.copy()
        else:
            merged[tool_name] = proposed.copy()
    return merged


@dataclass
class ConstraintOutcome:
    """What the constraint validator said about one description.

    ``messages`` is every check that ran, passing or not, because a reader of
    the artifacts wants to see what was looked at. ``failures`` is the subset
    that actually refused the rewrite, which is what the PR body quotes: a
    reviewer reading "rejected because size_limit: Size OK" would rightly stop
    trusting the report.
    """

    target: str
    kind: str  # "tool_description" or "param_description"
    passed: bool
    reverted: bool
    messages: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def reason(self) -> str:
        """Why this description was refused. Empty when it was not."""
        return "; ".join(self.failures)

    def to_dict(self) -> dict:
        """Serialise one constraint outcome, including whether it was reverted."""
        return {
            "target": self.target,
            "kind": self.kind,
            "passed": self.passed,
            "reverted": self.reverted,
            "messages": list(self.messages),
            "failures": list(self.failures),
        }


def build_accuracy_checker(
    catalog: ToolCatalog,
    lm: object = None,
    entailment: object = None,
) -> FactualAccuracyChecker:
    """A factual-accuracy checker for this catalogue.

    The entailment predictor is built by default but only ever called when an
    LM is configured, so this is safe to construct in an offline run. Pass
    ``entailment`` to inject a stub, or ``entailment=False`` to run the
    deterministic checks alone.
    """
    if entailment is False:
        predictor = None
    elif entailment is not None:
        predictor = entailment
    else:
        predictor = dspy.ChainOfThought(DescriptionEntailment)
    return FactualAccuracyChecker(
        facts=facts_from_catalog(catalog), entailment=predictor, lm=lm
    )


def enforce_constraints(
    candidate: dict[str, ToolDescriptions],
    baseline: dict[str, ToolDescriptions],
    validator: ConstraintValidator,
    allowed: Optional[Sequence[str]] = None,
    accuracy: Optional[FactualAccuracyChecker] = None,
) -> tuple[dict[str, ToolDescriptions], list[ConstraintOutcome]]:
    """Revert any evolved description that busts its budget or its schema.

    Checked against the 500 / 200 char budgets and the growth limit from
    EvolutionConfig, and - when *accuracy* is supplied - against PLAN.md's
    remaining Phase 2 constraint, that a description "must remain factually
    accurate (can't claim a tool does something it doesn't)". A factual finding
    reverts the description exactly like a budget failure does: an inaccurate
    description is not a smaller problem than a long one, it is a larger one.

    A failure reverts that single description to baseline rather than throwing
    away the whole candidate, so one greedy rewrite does not cost the run every
    other improvement it found.

    Unchanged text is not re-validated. hermes-agent's ``read_file``
    description is already 539 chars and its ``write_file.cross_profile``
    parameter is already 302; failing the run on a violation that was there
    before evolution started would make the tool unusable on the real repo.
    """
    permitted = set(allowed) if allowed is not None else set(baseline)
    result: dict[str, ToolDescriptions] = {}
    outcomes: list[ConstraintOutcome] = []

    for tool_name, base in baseline.items():
        proposed = candidate.get(tool_name)
        if proposed is None or tool_name not in permitted:
            result[tool_name] = base.copy()
            continue

        kept = proposed.copy()

        if kept.description != base.description:
            checks = validator.validate_all(
                kept.description, "tool_description", baseline_text=base.description
            )
            failures = [c for c in checks if not c.passed]
            messages = [f"{c.constraint_name}: {c.message}" for c in checks]
            findings = (
                accuracy.check_tool(tool_name, kept.description, base.description)
                if accuracy
                else []
            )
            messages.extend(f"factual_accuracy: {f.describe()}" for f in findings)
            reasons = [f"{c.constraint_name}: {c.message}" for c in failures]
            reasons.extend(f"factual_accuracy: {f.describe()}" for f in findings)
            rejected = bool(failures or findings)
            if rejected:
                kept.description = base.description
            outcomes.append(
                ConstraintOutcome(
                    target=tool_name,
                    kind="tool_description",
                    passed=not rejected,
                    reverted=rejected,
                    messages=messages,
                    failures=reasons,
                )
            )

        for param, base_text in base.params.items():
            new_text = kept.params.get(param)
            if new_text is None or new_text == base_text:
                kept.params[param] = base_text
                continue
            checks = validator.validate_all(
                new_text, "param_description", baseline_text=base_text
            )
            failures = [c for c in checks if not c.passed]
            messages = [f"{c.constraint_name}: {c.message}" for c in checks]
            findings = (
                accuracy.check_param(tool_name, param, new_text, base_text)
                if accuracy
                else []
            )
            messages.extend(f"factual_accuracy: {f.describe()}" for f in findings)
            reasons = [f"{c.constraint_name}: {c.message}" for c in failures]
            reasons.extend(f"factual_accuracy: {f.describe()}" for f in findings)
            rejected = bool(failures or findings)
            if rejected:
                kept.params[param] = base_text
            outcomes.append(
                ConstraintOutcome(
                    target=f"{tool_name}.{param}",
                    kind="param_description",
                    passed=not rejected,
                    reverted=rejected,
                    messages=messages,
                    failures=reasons,
                )
            )

        # Parameters the optimizer invented are dropped: the schema is frozen.
        kept.params = {name: kept.params[name] for name in base.params}
        result[tool_name] = kept

    return result, outcomes


# ──────────────────────────────────────────────────────────────────────────
# Deployment helpers
# ──────────────────────────────────────────────────────────────────────────

_UNSAFE_IN_A_BRANCH = re.compile(r"[^A-Za-z0-9._]+")

# Restated from Phase 3 rather than imported, so Phase 2 does not depend on a
# sibling phase for a number. docs/ARCHITECTURE.md documents it as the status
# for "you asked for this and it did not happen", which is a different claim
# from "the run failed".
EXIT_DEPLOYMENT_INCOMPLETE = 3


def deployment_incomplete(metrics: object) -> bool:
    """Whether a deployment step the caller asked for did not happen.

    A GitError on its own is not enough. A checkout that is not a git repo
    fails to build a branch, but the evolved descriptions are already on disk
    and failing the run over it would say something untrue about the write. It
    is the unmet *request* - ``--push`` that did not push, ``--open-pr`` that
    did not open - that the exit status has to carry.
    """
    if not isinstance(metrics, dict):
        return False
    deployment = metrics.get("deployment") or {}
    requested = deployment.get("requested") or {}
    return bool(
        (requested.get("push") and not deployment.get("pushed"))
        or (requested.get("open_pr") and not deployment.get("opened"))
    )


def benchmark_candidate(
    repo: Path,
    bundle: dict[str, ToolDescriptions],
    baseline: GateResult,
    *,
    name: str = "tblite",
    regression_threshold: float = 0.02,
    fast: bool = True,
) -> GateResult:
    """Measure *bundle* against *baseline* on a throwaway copy of *repo*.

    A benchmark can only score what is on disk, and the candidate is not: the
    write-back happens after this gate, and has to, because this gate is one of
    the things that decides whether it happens at all. Running both
    measurements against the unmodified checkout is worse than not running the
    second one, because it reports PASSED on evidence that describes the
    baseline twice and can never show a regression.

    Staging into *repo* would fix the measurement and break something more
    important: a candidate that has not passed its gates would be sitting in
    the operator's working tree, and any crash in between would leave it there.
    So the candidate is applied to a copy and the copy is thrown away.

    Returns *baseline* unchanged when there is no benchmark to run, which is
    every hermes-agent checkout today - copying a repo to measure nothing is a
    cost with no answer attached.
    """
    if baseline.status is GateStatus.UNAVAILABLE:
        return baseline

    repo = Path(repo)
    with tempfile.TemporaryDirectory(prefix="hermes-candidate-") as tmp:
        mirror = Path(tmp) / repo.name
        # .git is the bulk of a checkout and a benchmark never reads it.
        shutil.copytree(repo, mirror, ignore=shutil.ignore_patterns(".git"))
        write_bundle(mirror, bundle)
        return run_benchmark_gate(
            mirror,
            name,
            baseline=baseline.score,
            regression_threshold=regression_threshold,
            fast=fast,
        )


def pr_target_slug(tools: Sequence[str], toolset: Optional[str] = None) -> str:
    """A stable, branch-safe name for whatever this run evolved.

    It lands in ``evolve/<target>-<timestamp>``, so it carries no spaces and no
    slashes. One tool is named outright, a toolset run is named after its
    toolset, and a whole-catalogue run says so rather than listing thirty tools
    in a branch name.
    """
    names = [name for name in tools if name]
    if len(names) == 1:
        return _slugify(names[0])
    if toolset:
        return _slugify(f"{toolset}-toolset")
    return "all-tools"


def _slugify(text: str) -> str:
    slug = _UNSAFE_IN_A_BRANCH.sub("-", text).strip("-.")
    return slug or "tools"


def score_lines(
    baseline_report: CrossToolReport,
    candidate_report: CrossToolReport,
    val_examples: int,
    baseline_holdout: object = None,
    candidate_holdout: object = None,
    holdout_examples: int = 0,
) -> list[ScoreLine]:
    """The before/after rows for the PR body, for the splits that were measured.

    There is no train row: this phase never scores the train split, and an
    invented number is worse than a missing one. Holdout goes last because
    ``build_pull_request`` headlines the final row, and the split that was held
    back is the one a reviewer should read first.
    """
    lines = [
        ScoreLine(
            split="val",
            baseline=baseline_report.overall_accuracy,
            evolved=candidate_report.overall_accuracy,
            detail=(
                f"{val_examples} examples, "
                f"{candidate_report.chance_accuracy:.1%} chance across "
                f"{candidate_report.num_options} options"
            ),
        )
    ]
    if baseline_holdout is not None and candidate_holdout is not None:
        lines.append(
            ScoreLine(
                split="holdout",
                baseline=baseline_holdout.tool_accuracy,
                evolved=candidate_holdout.tool_accuracy,
                detail=f"{holdout_examples} examples, never optimized against",
            )
        )
    return lines


def collect_rejections(
    outcomes: Sequence[ConstraintOutcome],
    verdict: Optional[CrossToolVerdict] = None,
    holdout_verdict: Optional[CrossToolVerdict] = None,
) -> list[RejectedCandidate]:
    """Everything the run produced and refused, with the real reason attached.

    PLAN.md wants these in the PR body. A body that shows only the surviving
    rewrite hides how hard the constraint stage and the cross-tool guard were
    working, and a reviewer cannot tell a careful run from a lucky one. Factual
    reverts are in here on the same footing as budget failures, because that is
    how ``enforce_constraints`` treats them.
    """
    rejected: list[RejectedCandidate] = []
    for outcome in outcomes:
        if outcome.reverted:
            rejected.append(
                RejectedCandidate(
                    label=outcome.target,
                    reason=outcome.reason() or "reverted to baseline",
                )
            )

    for split, split_verdict in (("val", verdict), ("holdout", holdout_verdict)):
        if split_verdict is None:
            continue
        for regression in split_verdict.regressions:
            rejected.append(
                RejectedCandidate(
                    label=f"{regression.tool} ({split} cross-tool)",
                    reason=regression.describe(),
                )
            )
        # A verdict can refuse a candidate without naming a single regressed
        # tool, for instance when an overall improvement was required and did
        # not arrive. That refusal still belongs in the body.
        if not split_verdict.accepted and not split_verdict.regressions:
            rejected.append(
                RejectedCandidate(
                    label=f"whole candidate ({split} cross-tool)",
                    reason=split_verdict.reason,
                )
            )
    return rejected


def _repo_relative(path: Path, repo: Path) -> str:
    """A path git can stage, relative to the repo root when it lives there."""
    try:
        return str(Path(path).resolve().relative_to(Path(repo).resolve()))
    except ValueError:
        return str(path)


def _current_git_ref(repo: Path) -> str:
    """The ref *repo* is on, or an empty string when that cannot be read.

    Best effort on purpose: a hermes-agent checkout that is not a git
    repository is a reason to skip the PR, not a reason to end the run.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _restore_checkout(
    repo: Path, plan: Optional[PullRequestPlan], original_ref: str
) -> None:
    """Put *repo* back on the ref the run found it on. Never raises.

    ``build_pull_request`` creates the branch before it can fail and has no
    plan to hand back when it does, so the ref read before the call is the
    fallback. Leaving an operator parked on an ``evolve/`` branch they did not
    ask to be on is exactly the kind of surprise this pipeline is supposed to
    avoid.
    """
    try:
        if plan is not None:
            plan.restore()
        elif original_ref and original_ref != "HEAD":
            subprocess.run(
                ["git", "checkout", original_ref],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=60,
            )
    except Exception as exc:  # noqa: BLE001 - a cleanup path must not raise
        console.print(
            f"  [red]⚠ Could not restore {repo} to {original_ref or 'its original ref'}: "
            f"{exc}[/red]"
        )


# ──────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────────


def _catalogue_table(catalog: ToolCatalog, selected: ToolCatalog) -> Table:
    chosen = set(selected.names)
    table = Table(title="Tool catalogue")
    table.add_column("Tool", style="bold")
    table.add_column("Toolset")
    table.add_column("Module")
    table.add_column("Desc", justify="right")
    table.add_column("Params", justify="right")
    table.add_column("Budget")
    table.add_column("In run", justify="center")

    for entry in catalog:
        findings = entry.budget_findings()
        if findings:
            budget = "[red]" + "; ".join(f.describe() for f in findings) + "[/red]"
        else:
            budget = "[green]ok[/green]"
        table.add_row(
            entry.tool_name,
            f"{entry.toolset}" + ("" if entry.toolset_source == "registry" else " (inferred)"),
            entry.module,
            f"{entry.description_size}",
            f"{len(entry.param_names)}",
            budget,
            "✓" if entry.tool_name in chosen else "",
        )
    return table


def _rates_table(
    baseline: CrossToolReport,
    candidate: CrossToolReport,
    verdict: Optional[CrossToolVerdict] = None,
) -> Table:
    """Per-tool rates with the uncertainty that makes them readable.

    A rate change with no interval, no p-value and no power marker invites the
    reader to treat a one-example flip and a forty-example collapse as the same
    finding. The last three columns are what stop that.
    """
    table = Table(title="Per-tool selection rate")
    table.add_column("Tool", style="bold")
    table.add_column("Examples", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("95% CI on change", justify="right")
    table.add_column("p(worse)", justify="right")
    table.add_column("Power", justify="left")

    for tool in sorted(set(baseline.rates) | set(candidate.rates)):
        opportunities = baseline.opportunities(tool) or candidate.opportunities(tool)
        if opportunities == 0:
            continue
        before = baseline.rate(tool)
        after = candidate.rate(tool)
        delta = after - before
        colour = "green" if delta > 0 else ("red" if delta < 0 else "white")

        comparison = verdict.comparison(tool) if verdict else None
        interval = comparison.delta_interval() if comparison else None
        ci_text = (
            f"[{interval.low:+.1%}, {interval.high:+.1%}]" if interval else "[dim]-[/dim]"
        )
        p_worse = comparison.p_worse if comparison else None
        p_text = f"{p_worse:.3f}" if p_worse is not None else "[dim]-[/dim]"
        if comparison is None or comparison.paired is None:
            power = "[yellow]no pairing[/yellow]"
        elif comparison.underpowered:
            power = (
                f"[yellow]⚠ needs {comparison.min_detectable_shift:.0%}[/yellow]"
            )
        elif comparison.significant_regression:
            power = "[red]✗ significant[/red]"
        elif comparison.significant_improvement:
            power = "[green]✓ significant[/green]"
        else:
            power = "[green]✓[/green]"

        table.add_row(
            tool,
            str(opportunities),
            f"{before:.1%}",
            f"{after:.1%}",
            f"[{colour}]{delta:+.1%}[/{colour}]",
            ci_text,
            p_text,
            power,
        )
    return table


def _print_confusions(report: CrossToolReport) -> None:
    confusions = report.confusion.top_confusions(limit=5)
    if not confusions:
        console.print("  No misselections recorded.")
        return
    console.print("  Most common misselections (expected -> picked):")
    for expected, predicted, count in confusions:
        console.print(f"    {expected} -> {predicted}: {count}")


# ──────────────────────────────────────────────────────────────────────────
# The run
# ──────────────────────────────────────────────────────────────────────────


def evolve_tool_descriptions(
    tools: Sequence[str] = (),
    toolset: Optional[str] = None,
    iterations: int = 10,
    dataset_path: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    run_tests: bool = False,
    strict_gates: bool = False,
    dry_run: bool = False,
    write: bool = False,
    regression_tolerance: float = DEFAULT_TOLERANCE,
    output_root: Optional[Path] = None,
    create_pr: Optional[bool] = None,
    push: bool = False,
    open_pr: bool = False,
    pr_base: str = "main",
    allow_dirty: bool = False,
) -> Optional[dict]:
    """Run the full Phase 2 optimization. Returns the metrics it saved.

    ``create_pr`` defaults to :attr:`EvolutionConfig.create_pr`. It only ever
    does anything after a real write: no write, no branch, and a dry run builds
    nothing at all. ``push`` and ``open_pr`` are the only two things here that
    touch a network, and both are off unless the caller asks.
    """

    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo),
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=optimizer_model,  # Dataset generation deserves the strong model
        run_pytest=run_tests,
    )
    repo = Path(config.hermes_agent_path)
    # A dead field until now. An explicit flag wins; otherwise the config says.
    create_pr = config.create_pr if create_pr is None else create_pr

    # ── 1. Load the catalogue ───────────────────────────────────────────
    console.print(
        "\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] "
        "- Phase 2: tool descriptions\n"
    )
    _banner("1. Tool catalogue")
    console.print(f"  Repo: {repo}")

    catalog = load_catalog(repo, config)
    if not len(catalog):
        console.print(f"[red]✗ No literal tool schemas found under {repo / 'tools'}[/red]")
        sys.exit(1)

    try:
        selected = catalog.select(tools=list(tools), toolset=toolset)
    except UnknownTool as exc:
        console.print(f"[red]✗ {exc}[/red]")
        sys.exit(1)

    if not len(selected):
        console.print(f"[red]✗ No tools matched (toolset={toolset!r}, tools={list(tools)})[/red]")
        sys.exit(1)

    console.print(_catalogue_table(catalog, selected))
    console.print(
        f"  {len(catalog)} tool(s) in {len(catalog.by_toolset())} toolset(s), "
        f"{catalog.total_description_chars:,} chars of description total"
    )
    console.print(f"  Evolving {len(selected)}: {', '.join(selected.names)}")

    over_budget = catalog.budget_findings()
    if over_budget:
        console.print(
            f"[yellow]⚠ {len(over_budget)} description(s) already over budget "
            f"before evolution:[/yellow]"
        )
        for finding in over_budget:
            console.print(f"    {finding.describe()}")

    if dry_run:
        console.print("\n[bold green]DRY RUN - setup validated successfully.[/bold green]")
        console.print(f"  Would build a tool-selection dataset over all {len(catalog)} tools")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print("  Would check constraints, cross-tool regressions, and the gate ladder")
        console.print(
            f"  Would reject any tool regressing past a {regression_tolerance:.1%} "
            f"tolerance, or any regression significant at alpha=0.05"
        )
        console.print(
            "  Would " + ("write results back to the repo" if write else "leave the repo untouched")
        )
        if write and create_pr:
            console.print(
                f"  Would then commit the modified files onto "
                f"evolve/{pr_target_slug(selected.names, toolset)}-<timestamp> "
                f"and write PULL_REQUEST.md beside the run artifacts"
            )
            console.print(
                "  Would "
                + ("push that branch" if push else "not push")
                + " and would "
                + (f"open a pull request against {pr_base}" if open_pr else "not open a pull request")
            )
        elif write:
            console.print("  Would not build a branch (--no-create-pr)")
        console.print("  A dry run builds no branch and sends nothing.")
        return None

    # Refuse before spending anything if the operator has uncommitted work in a
    # file this run would overwrite. It has to happen here, before the write:
    # afterwards the run's own edits make the same files dirty and there is no
    # way left to tell whose changes they are. Without it, `git checkout -b`
    # carries their work onto the evolve branch, the commit absorbs it, and
    # restoring their ref makes it look deleted. Phase 4 has always refused this.
    if write and create_pr is not False:
        try:
            require_clean_worktree(
                repo,
                sorted({
                    str(e.descriptor.path.relative_to(repo))
                    for e in catalog.select()
                }),
                allow_dirty=allow_dirty,
            )
        except GitError as exc:
            console.print(f"[red]✗ {exc}[/red]")
            return 1

    baseline_bundle = catalog.bundle()
    signatures = catalog_signatures(catalog)

    # Everything that can reach a language model happens inside this block:
    # dataset generation, the baseline, the optimizer, and every evaluation.
    # PLAN.md wants the cost of the run in the PR body, and a figure that
    # skipped the optimizer would be the wrong figure.
    with UsageTracker() as usage:
        # ── 2. Selection dataset ────────────────────────────────────────────
        _banner("2. Tool-selection dataset")
        dataset_dir = Path(dataset_path) if dataset_path else Path("datasets") / "tools" / (toolset or "all")

        if (dataset_dir / "train.jsonl").exists():
            dataset = ToolSelectionDataset.load(dataset_dir)
            console.print(f"  Loaded {len(dataset)} examples from {dataset_dir}/")
        else:
            console.print(f"  Generating with {config.judge_model} (all {len(catalog)} tools)")
            builder = ToolSelectionDatasetBuilder(
                catalog=catalog,
                config=config,
                lm=dspy.LM(config.judge_model, temperature=0.0),
            )
            dataset = builder.generate()
            if not dataset.all_examples:
                console.print("[red]✗ Dataset generation produced no usable examples[/red]")
                sys.exit(1)
            dataset.save(dataset_dir)
            console.print(f"  Generated {len(dataset)} examples, saved to {dataset_dir}/")
            if builder.rejected:
                console.print(f"  [yellow]Rejected {len(builder.rejected)} generated case(s)[/yellow]")
                for target, reason in builder.rejected[:5]:
                    console.print(f"    {target}: {reason}")

        console.print(
            f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / "
            f"{len(dataset.holdout)} holdout"
        )
        console.print(f"  Categories: {dataset.category_counts()}")

        if not dataset.val:
            console.print("[red]✗ The val split is empty; cannot measure a regression[/red]")
            sys.exit(1)

        # ── 3. Baseline measurement ─────────────────────────────────────────
        _banner("3. Baseline measurement")
        lm = dspy.LM(config.eval_model, temperature=0.0)
        dspy.configure(lm=lm)

        baseline_module = ToolSelector(baseline_bundle, signatures)
        all_tools = list(catalog.names) + [NO_TOOL]

        with dspy.context(lm=lm):
            baseline_val = evaluate_selection(dataset.val, selector_predict_fn(baseline_module))
        baseline_report = CrossToolReport.from_report(baseline_val, tools=all_tools)

        console.print(f"  Selection accuracy: {baseline_report.describe_accuracy()}")
        console.print(f"  Parameter correctness: {baseline_val.param_accuracy:.1%}")
        _print_confusions(baseline_report)

        # ── 4. Optimize ─────────────────────────────────────────────────────
        _banner("4. GEPA optimization")
        console.print(f"  Optimizer model: {optimizer_model}")
        console.print(f"  Eval model: {eval_model}")
        console.print(f"  Iterations: {iterations}")

        trainset = dataset.to_dspy_examples("train")
        valset = dataset.to_dspy_examples("val")
        start_time = time.time()
        optimizer_used = "GEPA"

        try:
            optimizer = dspy.GEPA(
                metric=gepa_selection_metric,
                max_full_evals=iterations,
                reflection_lm=dspy.LM(optimizer_model),
            )
            optimized_module = optimizer.compile(
                baseline_module,
                trainset=trainset,
                valset=valset,
            )
        except Exception as exc:
            # Fall back to MIPROv2 if GEPA isn't available in this DSPy version
            console.print(f"[yellow]GEPA not available ({exc}), falling back to MIPROv2[/yellow]")
            optimizer_used = "MIPROv2"
            optimizer = dspy.MIPROv2(
                metric=tool_selection_metric,
                auto="light",
            )
            optimized_module = optimizer.compile(
                baseline_module,
                trainset=trainset,
            )

        elapsed = time.time() - start_time
        console.print(f"\n  Optimization completed in {elapsed:.1f}s using {optimizer_used}")

        raw_bundle = extract_bundle(optimized_module, baseline_bundle)
        candidate_bundle = freeze_unselected(raw_bundle, baseline_bundle, selected.names)

        # ── 5. Constraints ──────────────────────────────────────────────────
        _banner("5. Constraint validation")
        validator = ConstraintValidator(config)
        accuracy = build_accuracy_checker(catalog, lm=lm)
        candidate_bundle, constraint_outcomes = enforce_constraints(
            candidate_bundle,
            baseline_bundle,
            validator,
            allowed=selected.names,
            accuracy=accuracy,
        )
        # Counted off `failures`, not `messages`. The two carry the same
        # factual_accuracy entries today, because a finding is only ever raised
        # for a problem and a problem always reverts - but `messages` is
        # defined as every check that ran, passing ones included, so a count of
        # reverts taken from it is only correct by accident and would start
        # lying the moment a passing factual check wanted to announce itself.
        factual_reverts = sum(
            1
            for outcome in constraint_outcomes
            if outcome.reverted
            and any(m.startswith("factual_accuracy:") for m in outcome.failures)
        )
        console.print(
            "  Factual accuracy: schema-structural checks"
            + (
                " plus LLM entailment"
                if accuracy.entailment_ran
                else f" only ({accuracy.skipped_reason or 'entailment not run'})"
            )
        )

        if not constraint_outcomes:
            console.print("  No description changed, so nothing to validate.")
        for outcome in constraint_outcomes:
            icon = "✓" if outcome.passed else "✗"
            colour = "green" if outcome.passed else "red"
            note = "" if outcome.passed else " [reverted to baseline]"
            console.print(f"  [{colour}]{icon} {outcome.target}[/{colour}]{note}")
            for message in outcome.messages:
                console.print(f"      {message}")

        changes = diff_bundles(baseline_bundle, candidate_bundle)
        console.print(f"  {len(changes)} description(s) changed after validation")

        # ── 6. Cross-tool regression check ──────────────────────────────────
        _banner("6. Cross-tool regression check")
        candidate_module = ToolSelector(candidate_bundle, signatures)
        with dspy.context(lm=lm):
            candidate_val = evaluate_selection(dataset.val, selector_predict_fn(candidate_module))
        candidate_report = CrossToolReport.from_report(candidate_val, tools=all_tools)

        guard = CrossToolGuard(tolerance=regression_tolerance)
        verdict = guard.compare(baseline_report, candidate_report)

        console.print(_rates_table(baseline_report, candidate_report, verdict))
        console.print(f"  Overall: {candidate_report.describe_accuracy()}")
        icon = "✓" if verdict.accepted else "✗"
        colour = "green" if verdict.accepted else "red"
        console.print(f"  [{colour}]{icon} {verdict.summary()}[/{colour}]")
        for regression in verdict.regressions:
            console.print(f"    [red]{regression.describe()}[/red]")
        if verdict.underpowered:
            console.print(f"  [yellow]⚠ {verdict.power_note()}[/yellow]")
        _print_confusions(candidate_report)

        # ── 7. Gate ladder ──────────────────────────────────────────────────
        _banner("7. Gate ladder")
        chain = GateChain(strict=strict_gates)
        gates = [lambda: verdict.to_gate_result()]
        if run_tests:
            gates.append(lambda: run_pytest_gate(repo))

        # Measure the benchmark BEFORE the write-back so the post-write score has
        # something to be compared against. Passing baseline=None made the gate
        # structurally incapable of detecting a regression: gates.py returns
        # PASSED with "no baseline to compare" for any score at all, so
        # tblite_regression_threshold was read and then unreachable, and PLAN.md's
        # "Benchmarks hold (TBLite within 2%)" was not enforceable. Phase 3
        # already measures a baseline this way.
        tblite_baseline = run_benchmark_gate(repo, "tblite", fast=True)
        if tblite_baseline.status is GateStatus.PASSED and tblite_baseline.score is not None:
            console.print(
                f"  tblite baseline: {tblite_baseline.score:.1%} "
                f"(candidates must hold within "
                f"{config.tblite_regression_threshold:.1%})"
            )
        # Measured on a copy with the candidate applied, not on `repo`. Both
        # measurements used to run against the unmodified checkout, which made
        # the gate structurally incapable of failing: it was comparing the
        # baseline against itself.
        gates.append(
            lambda: benchmark_candidate(
                repo,
                candidate_bundle,
                tblite_baseline,
                regression_threshold=config.tblite_regression_threshold,
                fast=True,
            )
        )
        chain.run(*gates)
        console.print(chain.summary())

        # ── 8. Holdout ──────────────────────────────────────────────────────
        _banner(f"8. Holdout evaluation ({len(dataset.holdout)} examples)")
        if dataset.holdout:
            with dspy.context(lm=lm):
                baseline_holdout = evaluate_selection(
                    dataset.holdout, selector_predict_fn(baseline_module)
                )
                candidate_holdout = evaluate_selection(
                    dataset.holdout, selector_predict_fn(candidate_module)
                )
            holdout_baseline_report = CrossToolReport.from_report(baseline_holdout, tools=all_tools)
            holdout_candidate_report = CrossToolReport.from_report(candidate_holdout, tools=all_tools)
            holdout_verdict = guard.compare(holdout_baseline_report, holdout_candidate_report)
            console.print(f"  {holdout_verdict.summary()}")
        else:
            baseline_holdout = candidate_holdout = None
            holdout_verdict = None
            console.print("  [yellow]No holdout examples; skipping[/yellow]")

    cost = usage.report

    # ── 9. Results ──────────────────────────────────────────────────────
    _banner("9. Results")
    baseline_chars = sum(d.total_chars for d in baseline_bundle.values())
    candidate_chars = sum(d.total_chars for d in candidate_bundle.values())

    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    def _row(label: str, before: float, after: float, note: str = "") -> None:
        delta = after - before
        colour = "green" if delta > 0 else ("red" if delta < 0 else "white")
        table.add_row(
            label if not note else f"{label}\n[dim]{note}[/dim]",
            f"{before:.1%}",
            f"{after:.1%}",
            f"[{colour}]{delta:+.1%}[/{colour}]",
        )

    _row("Selection accuracy (val)", baseline_report.overall_accuracy, candidate_report.overall_accuracy)
    baseline_ci = baseline_report.accuracy_interval()
    candidate_ci = candidate_report.accuracy_interval()
    table.add_row(
        "  95% CI on accuracy",
        f"[{baseline_ci.low:.1%}, {baseline_ci.high:.1%}]",
        f"[{candidate_ci.low:.1%}, {candidate_ci.high:.1%}]",
        "",
    )
    # Raw accuracy is not interpretable on its own: 40% is poor against two
    # tools and excellent against thirty.
    table.add_row(
        f"  Chance ({candidate_report.num_options} options)",
        f"{baseline_report.chance_accuracy:.1%}",
        f"{candidate_report.chance_accuracy:.1%}",
        "",
    )
    _row(
        "Parameter correctness (val)",
        baseline_val.param_accuracy,
        candidate_val.param_accuracy,
        note=(
            f"over {sum(1 for o in baseline_val.outcomes if o.tool_correct)} "
            f"then {sum(1 for o in candidate_val.outcomes if o.tool_correct)} "
            f"correctly selected"
        ),
    )
    if baseline_holdout is not None and candidate_holdout is not None:
        _row("Selection accuracy (holdout)", baseline_holdout.tool_accuracy, candidate_holdout.tool_accuracy)
    table.add_row(
        "Description chars",
        f"{baseline_chars:,}",
        f"{candidate_chars:,}",
        f"{candidate_chars - baseline_chars:+,}",
    )
    table.add_row("Descriptions changed", "", str(len(changes)), "")
    table.add_row("Factual reverts", "", str(factual_reverts), "")
    table.add_row(
        "Underpowered tools",
        "",
        str(len(verdict.underpowered)),
        ", ".join(verdict.underpowered),
    )
    table.add_row("Optimizer", "", optimizer_used, "")
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    # describe() says "at least $X" when a model had no published price or the
    # DSPy log lost entries. Print it verbatim; a cost report that rounds an
    # unknown down to zero is worse than no cost report.
    table.add_row("Cost", "", cost.describe(), "")

    console.print()
    console.print(table)

    # ── 10. Write-back ──────────────────────────────────────────────────
    _banner("10. Write-back")
    # The holdout verdict gates too. It was previously computed, printed, and
    # then ignored, so a candidate that held on validation and collapsed on
    # holdout shipped anyway - and because collect_rejections walks both
    # verdicts, the winner's own holdout regression was then filed in the PR
    # body under "Rejected along the way". A reviewer skimming that read the
    # exact opposite of what happened. Holdout is the split nothing optimized
    # against, which is precisely why it is the one worth obeying.
    holdout_blocked = holdout_verdict is not None and not holdout_verdict.accepted
    may_write = (
        write
        and verdict.accepted
        and not holdout_blocked
        and chain.passed
        and bool(changes)
    )

    if not changes:
        console.print("  Nothing to write: no description survived validation unchanged.")
        write_report = None
    else:
        # Always exercise the rewrite. A dry run that verifies is evidence.
        write_report = write_bundle(
            repo, candidate_bundle, dry_run=not may_write, baseline=baseline_bundle
        )
        console.print(f"  {write_report.summary()}")
        for change in write_report.changes:
            console.print(f"    {change.target} ({change.delta_chars:+d} chars)")
        for target, reason in write_report.skipped:
            console.print(f"    [yellow]skipped {target}: {reason}[/yellow]")
        if not may_write:
            if not write:
                console.print("  [yellow]--no-write is the default; re-run with --write to apply[/yellow]")
            elif not verdict.accepted:
                console.print("  [red]Not written: the cross-tool guard rejected this candidate[/red]")
            elif holdout_blocked:
                console.print(
                    "  [red]Not written: the candidate held on validation and "
                    "regressed on holdout, the split it never optimized against[/red]"
                )
                console.print(f"    {holdout_verdict.summary()}")
            elif not chain.passed:
                console.print("  [red]Not written: a gate blocked this candidate[/red]")

    # ── 11. Save artifacts ──────────────────────────────────────────────
    _banner("11. Artifacts")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root or config.output_dir) / "tools" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "baseline_descriptions.json").write_text(
        json.dumps(bundle_to_dict(baseline_bundle), indent=2)
    )
    (output_dir / "evolved_descriptions.json").write_text(
        json.dumps(bundle_to_dict(candidate_bundle), indent=2)
    )
    (output_dir / "cross_tool_report.json").write_text(
        json.dumps(
            {
                "baseline": baseline_report.to_dict(),
                "candidate": candidate_report.to_dict(),
                "verdict": verdict.to_dict(),
                "holdout_verdict": holdout_verdict.to_dict() if holdout_verdict else None,
            },
            indent=2,
        )
    )
    (output_dir / "gates.json").write_text(json.dumps(chain.to_dict(), indent=2))
    (output_dir / "changes.json").write_text(
        json.dumps([c.to_dict() for c in changes], indent=2)
    )

    metrics = {
        "phase": "tool_descriptions",
        "timestamp": timestamp,
        "hermes_repo": str(repo),
        "tools_evolved": selected.names,
        "toolset": toolset,
        "iterations": iterations,
        "optimizer": optimizer_used,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_accuracy": baseline_report.overall_accuracy,
        "candidate_accuracy": candidate_report.overall_accuracy,
        # Both figures average only over the examples whose tool was chosen
        # correctly, so the denominator moves between baseline and candidate. A
        # candidate that selects fewer tools correctly but gets the arguments
        # right on the ones it did can show a parameter-accuracy "improvement"
        # that is purely a denominator artifact. The counts ship alongside the
        # rates so a reader can see the shift instead of inferring it.
        "baseline_param_accuracy": baseline_val.param_accuracy,
        "candidate_param_accuracy": candidate_val.param_accuracy,
        "baseline_param_accuracy_n": sum(1 for o in baseline_val.outcomes if o.tool_correct),
        "candidate_param_accuracy_n": sum(1 for o in candidate_val.outcomes if o.tool_correct),
        "holdout_baseline_accuracy": baseline_holdout.tool_accuracy if baseline_holdout else None,
        "holdout_candidate_accuracy": candidate_holdout.tool_accuracy if candidate_holdout else None,
        "baseline_accuracy_ci": baseline_report.accuracy_interval().to_dict(),
        "candidate_accuracy_ci": candidate_report.accuracy_interval().to_dict(),
        "chance_accuracy": candidate_report.chance_accuracy,
        "num_options": candidate_report.num_options,
        "cross_tool_accepted": verdict.accepted,
        "regression_tolerance": regression_tolerance,
        # Per tool: baseline rate, candidate rate, delta with its interval, the
        # one-sided p-value, and whether this many examples could ever have
        # detected the tolerance being enforced.
        "per_tool": [comparison.to_dict() for comparison in verdict.comparisons],
        "underpowered_tools": list(verdict.underpowered),
        "unpaired_tools": list(verdict.unpaired),
        "significant_regressions": [r.tool for r in verdict.significant_regressions],
        "gates_passed": chain.passed,
        "descriptions_changed": len(changes),
        "baseline_chars": baseline_chars,
        "candidate_chars": candidate_chars,
        "constraint_reverts": sum(1 for o in constraint_outcomes if o.reverted),
        "factual_reverts": factual_reverts,
        "entailment_ran": accuracy.entailment_ran,
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "cost": cost.to_dict(),
        "written": bool(may_write and write_report and write_report.files_written),
        "pull_request": None,
        # Present on every run like pull_request is, so a reader never has to
        # tell "deployment did not happen" apart from "the key is missing".
        "deployment": None,
    }
    metrics_path = output_dir / "metrics.json"

    def save_metrics() -> None:
        """Write the metrics file now, so a later crash still leaves one behind."""
        metrics_path.write_text(json.dumps(metrics, indent=2))

    save_metrics()
    console.print(f"\n  Output saved to {output_dir}/")

    # ── 12. Deployment ──────────────────────────────────────────────────
    _banner("12. Deployment")
    written_files = [
        _repo_relative(path, repo)
        for path in (write_report.files_written if write_report else [])
    ]

    if not metrics["written"] or not written_files:
        # PLAN.md constraint 5 is about how a change reaches hermes-agent. A run
        # that changed nothing has nothing to deploy, and inventing an empty
        # branch for it would only add noise to the review queue.
        console.print(
            "  Nothing was written, so there is no branch to build and nothing "
            "was pushed."
        )
    elif not create_pr:
        console.print(
            "  [yellow]--no-create-pr: the files were written in place and left "
            "uncommitted. Nothing was pushed.[/yellow]"
        )
    else:
        original_ref = _current_git_ref(repo)
        plan: Optional[PullRequestPlan] = None
        # What deployment actually did, step by step. A branch can be built
        # locally and still never reach a remote, so "there is a pull_request
        # record" is not the same claim as "it was pushed" - each step marks
        # itself only once it has returned, and a GitError leaves the reason
        # behind in the artifact instead of only on the terminal.
        deployment = {
            "requested": {"push": push, "open_pr": open_pr},
            "status": "pending",
            "pushed": False,
            "opened": False,
            "error": None,
        }
        metrics["deployment"] = deployment
        try:
            plan = build_pull_request(
                repo=repo,
                target=pr_target_slug(selected.names, toolset),
                phase="Phase 2 (tool descriptions)",
                timestamp=timestamp,
                files=written_files,
                scores=score_lines(
                    baseline_report,
                    candidate_report,
                    val_examples=len(dataset.val),
                    baseline_holdout=baseline_holdout,
                    candidate_holdout=candidate_holdout,
                    holdout_examples=len(dataset.holdout),
                ),
                cost=cost,
                rejected=collect_rejections(constraint_outcomes, verdict, holdout_verdict),
                gates=chain.summary().splitlines(),
                dataset=(
                    f"{dataset_dir} - {len(dataset.train)} train / "
                    f"{len(dataset.val)} val / {len(dataset.holdout)} holdout"
                ),
                optimizer=optimizer_used,
                iterations=iterations,
                statistics=verdict.summary(),
                notes=[
                    f"Reflection model: {optimizer_model}",
                    f"Eval model: {eval_model}",
                    f"Descriptions changed: {len(changes)}",
                    f"Description size: {baseline_chars:,} to {candidate_chars:,} chars "
                    f"({candidate_chars - baseline_chars:+,})",
                    f"Optimization wall clock: {elapsed:.1f}s",
                    f"Run artifacts: {output_dir}",
                ],
            )
            body_path = plan.write_body(output_dir)
            metrics["pull_request"] = plan.to_dict()
            save_metrics()

            console.print(f"  Branch: {plan.branch}")
            console.print(f"  Commit: {len(written_files)} file(s) - {', '.join(written_files)}")
            console.print(f"  PR body: {body_path}")

            if push:
                plan.push()
                deployment["pushed"] = True
                console.print("  Pushed the branch to origin, as --push asked.")
            if open_pr:
                if not push:
                    console.print(
                        "  [yellow]--open-pr without --push: gh can only open a "
                        "pull request for a branch the remote already has.[/yellow]"
                    )
                plan.open(base=pr_base)
                deployment["opened"] = True
                console.print(f"  Opened a pull request against {pr_base}, as --open-pr asked.")
            deployment["status"] = "ok"
            if not push and not open_pr:
                console.print(
                    "  Nothing was pushed and no pull request was opened. The branch "
                    "and PULL_REQUEST.md are local only."
                )
                console.print("  Re-run with --push --open-pr, or use them by hand.")
            elif not open_pr:
                console.print("  No pull request was opened.")
        except GitError as exc:
            deployment["status"] = "failed"
            deployment["error"] = str(exc)
            console.print(f"  [red]✗ {exc}[/red]")
            console.print(
                "  The evolved descriptions are still in the run artifacts above."
            )
        finally:
            save_metrics()
            _restore_checkout(repo, plan, original_ref)
            if plan is not None:
                console.print(
                    f"  {repo} is back on {plan.original_ref or original_ref}; the "
                    f"change lives on the branch, not on your checkout."
                )

    improvement = candidate_report.overall_accuracy - baseline_report.overall_accuracy
    if verdict.accepted and improvement > 0:
        console.print(
            f"\n[bold green]✓ Selection accuracy improved {improvement:+.1%} "
            f"with no per-tool regression[/bold green]"
        )
    elif verdict.accepted:
        console.print(
            f"\n[yellow]⚠ No accuracy improvement ({improvement:+.1%}), "
            f"but nothing regressed[/yellow]"
        )
    else:
        console.print(
            "\n[red]✗ Candidate rejected by the cross-tool guard - "
            "one tool's gain came out of another's[/red]"
        )

    return metrics


@click.command()
@click.option("--tool", "tools", multiple=True, help="Tool to evolve (repeatable, default all)")
@click.option("--toolset", default=None, help="Limit to one toolset, e.g. 'file'")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--dataset-path", default=None, help="Directory holding train/val/holdout JSONL")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--run-tests", is_flag=True, help="Run the hermes-agent pytest suite as a gate")
@click.option("--strict-gates", is_flag=True, help="Treat an unavailable gate as a failure")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--write/--no-write", default=False, help="Write evolved descriptions into the repo")
@click.option(
    "--regression-tolerance",
    default=DEFAULT_TOLERANCE,
    type=float,
    help="How far one tool's selection rate may fall before rejection (0 = not at all)",
)
@click.option(
    "--create-pr/--no-create-pr",
    "create_pr",
    default=None,
    help=(
        "After a write, commit the modified files onto evolve/<target>-<timestamp> "
        "and write PULL_REQUEST.md beside the run artifacts "
        "(default: EvolutionConfig.create_pr). Local only; see --push"
    ),
)
@click.option(
    "--push/--no-push",
    default=False,
    help="Push the deployment branch to origin. Off by default: nothing leaves this machine unasked",
)
@click.option(
    "--open-pr/--no-open-pr",
    default=False,
    help="Open the pull request with gh. Off by default, and needs the branch pushed",
)
@click.option("--pr-base", default="main", help="Base branch for the pull request")
@click.option("--allow-dirty", is_flag=True,
              help="Proceed even if the target files have uncommitted changes")
def main(
    tools,
    toolset,
    iterations,
    dataset_path,
    hermes_repo,
    optimizer_model,
    eval_model,
    run_tests,
    strict_gates,
    dry_run,
    write,
    regression_tolerance,
    create_pr,
    push,
    open_pr,
    pr_base,
    allow_dirty,
):
    """Evolve hermes-agent tool descriptions using DSPy + GEPA optimization."""
    result = evolve_tool_descriptions(
        tools=tools,
        toolset=toolset,
        iterations=iterations,
        dataset_path=dataset_path,
        hermes_repo=hermes_repo,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        run_tests=run_tests,
        strict_gates=strict_gates,
        dry_run=dry_run,
        write=write,
        regression_tolerance=regression_tolerance,
        create_pr=create_pr,
        push=push,
        open_pr=open_pr,
        pr_base=pr_base,
        allow_dirty=allow_dirty,
    )
    # evolve_tool_descriptions returns the metrics it saved, or a bare exit
    # code when it refused to start at all. Both have to become a status here,
    # and neither reading is optional: handing the metrics dict itself to
    # SystemExit printed the whole dict and exited 1 on every successful run,
    # and swallowing a failed push made it look like a clean one. Phase 5 reads
    # exactly this status to decide whether an optimization was proposed.
    if isinstance(result, int):
        code = result
    elif deployment_incomplete(result):
        code = EXIT_DEPLOYMENT_INCOMPLETE
    else:
        code = 0
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()

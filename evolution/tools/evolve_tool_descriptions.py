"""Evolve a Hermes Agent tool description using DSPy + GEPA.

The artifact is the tool's ``description`` string — the text the model reads
when choosing a tool — not the full selection prompt. Real (task -> tool) pairs
mined from the session store provide the positives and negatives.

Usage:
    python -m evolution.tools.evolve_tool_descriptions --tool search_files --days 90
    python -m evolution.tools.evolve_tool_descriptions --tool read_file --dry-run
"""

import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.constraints import ConstraintValidator
from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.core.gepa_metric import make_feedback_metric
from evolution.core.lm import make_lm
from evolution.core.optimize import build_gepa, print_constraints
from evolution.core.state_db import DEFAULT_DB
from evolution.tools.tool_doc_module import ToolDocModule, extract_description
from evolution.tools.tool_registry_reader import read_tool_descriptions
from evolution.tools.tool_usage_miner import mine_tool_usage

console = Console()

OTHER = "other"
MIN_POSITIVES = 10
NEGATIVE_MULTIPLIER = 2
TRAIN_RATIO = 0.5
VAL_RATIO = 0.25


def selection_metric(gold, pred, trace=None) -> float:
    """Score a tool-choice answer: 1.0 when it names the right tool."""
    expected = str(getattr(gold, "expected_behavior", "") or "").strip().lower()
    answer = str(getattr(pred, "output", "") or "").strip().lower()
    if not expected or not answer:
        return 0.0
    if answer == expected:
        return 1.0
    # A gold tool name inside a sentence ("use search_files here") still counts;
    # token-boundary match so "close_terminal" never scores for "terminal".
    if expected != OTHER and re.search(
        rf"(?<![\w-]){re.escape(expected)}(?![\w-])", answer
    ):
        return 1.0
    return 0.0


def build_selection_dataset(
    pairs: Sequence[dict],
    tool_name: str,
    negative_multiplier: int = NEGATIVE_MULTIPLIER,
    seed: int = 0,
) -> EvalDataset:
    """Turn mined (task, tool) pairs into a balanced train/val/holdout dataset.

    A task that produced both the target tool and another tool is contradictory
    evidence and is kept only as a positive.
    """
    rng = random.Random(seed)
    positives: list[str] = []
    positive_tasks: set[str] = set()
    negatives: list[str] = []

    for pair in pairs:
        task = str(pair.get("task_input") or "").strip()
        if not task:
            continue
        if pair.get("tool") == tool_name:
            if task not in positive_tasks:
                positive_tasks.add(task)
                positives.append(task)
        else:
            negatives.append(task)

    seen = set(positive_tasks)
    unique_negatives = []
    for task in negatives:
        if task in seen:
            continue
        seen.add(task)
        unique_negatives.append(task)

    rng.shuffle(positives)
    rng.shuffle(unique_negatives)
    unique_negatives = unique_negatives[: len(positives) * negative_multiplier]

    examples = [
        EvalExample(
            task_input=task,
            expected_behavior=tool_name,
            difficulty="medium",
            category="selection",
            source="sessiondb",
        )
        for task in positives
    ]
    examples += [
        EvalExample(
            task_input=task,
            expected_behavior=OTHER,
            difficulty="medium",
            category="selection",
            source="sessiondb",
        )
        for task in unique_negatives
    ]
    rng.shuffle(examples)
    return _split(examples)


def _split(examples: list[EvalExample]) -> EvalDataset:
    n = len(examples)
    if n == 0:
        return EvalDataset()
    n_train = max(1, int(n * TRAIN_RATIO))
    n_val = max(1, int(n * VAL_RATIO))
    if n_train + n_val > n:
        n_train = max(1, n - 1)
        n_val = min(n_val, n - 1)
    return EvalDataset(
        train=examples[:n_train],
        val=examples[n_train:n_train + n_val],
        holdout=examples[n_train + n_val:],
    )


def evolve(
    tool_name: str,
    hermes_repo: Optional[str] = None,
    days: int = 90,
    iterations: int = 10,
    eval_model: str = "openai/gpt-4.1-mini",
    optimizer_model: str = "openai/gpt-4.1",
    dry_run: bool = False,
):
    """Optimize one tool's description against mined tool-selection pairs."""
    hermes_path: Path = resolve_hermes_agent_path(hermes_repo)
    config = EvolutionConfig(
        hermes_agent_path=hermes_path,
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
        run_pytest=False,
    )

    console.print(
        f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — "
        f"Evolving tool description: [bold]{tool_name}[/bold]\n"
    )

    # ── 1. Read the current description ─────────────────────────────────
    try:
        descriptions = read_tool_descriptions(hermes_path)
    except FileNotFoundError as e:
        console.print(f"[red]✗ {e}[/red]")
        sys.exit(1)

    description = descriptions.get(tool_name)
    if description is None:
        console.print(
            f"[red]✗ Tool '{tool_name}' has no statically readable description in "
            f"{hermes_path / 'tools'}[/red]"
        )
        readable = ", ".join(sorted(descriptions)) or "(none)"
        console.print(f"  Readable tools: {readable}")
        sys.exit(1)

    console.print(f"  Description: {len(description)} chars")
    console.print(f"  {description[:120]}{'...' if len(description) > 120 else ''}")

    # ── 2. Mined tool-selection dataset ─────────────────────────────────
    console.print(f"\n[bold]Mining tool usage[/bold] (last {days} days)")
    try:
        pairs = mine_tool_usage(days)
    except (RuntimeError, ValueError) as e:
        console.print(f"[red]✗ {e}[/red]")
        sys.exit(1)
    if not pairs:
        console.print(
            f"[red]✗ No usage pairs mined — no session store at {DEFAULT_DB}, "
            f"or the window is empty[/red]"
        )
        sys.exit(1)
    dataset = build_selection_dataset(pairs, tool_name)
    positives = sum(
        1 for ex in dataset.all_examples if ex.expected_behavior == tool_name
    )
    negatives = len(dataset.all_examples) - positives

    console.print(f"  Mined pairs: {len(pairs)}")
    console.print(f"  Positives ({tool_name}): {positives}")
    console.print(f"  Negatives (other): {negatives}")

    if positives < MIN_POSITIVES:
        console.print(
            f"[red]✗ Insufficient usage data for '{tool_name}': {positives} positive "
            f"example(s), need at least {MIN_POSITIVES}[/red]"
        )
        console.print("  Try a larger --days window or a tool with more traffic.")
        sys.exit(1)

    console.print(
        f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / "
        f"{len(dataset.holdout)} holdout"
    )

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Would validate description (max 500 chars) and markers")
        console.print(f"  Would write output/tools/{tool_name}/<timestamp>/")
        return

    # ── 3. Baseline constraints ─────────────────────────────────────────
    console.print("\n[bold]Validating baseline description[/bold]")
    validator = ConstraintValidator(config)
    baseline_instructions = ToolDocModule(tool_name, description).extract()
    baseline_constraints = validator.validate_tool_description(
        description, baseline_instructions
    )
    if not print_constraints(console, baseline_constraints):
        console.print(
            "[yellow]⚠ Baseline description has constraint violations — proceeding anyway[/yellow]"
        )

    # ── 4. GEPA ─────────────────────────────────────────────────────────
    console.print("\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    lm = make_lm(eval_model)
    dspy.configure(lm=lm)

    baseline_module = ToolDocModule(tool_name, description)
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()
    # Template overhead is ~250 chars; capping instructions at 750 keeps the
    # description inside its 500-char limit.
    optimizer = build_gepa(
        make_feedback_metric(selection_metric),
        iterations,
        optimizer_model,
        max_instruction_chars=750,
        proposer_guidance=(
            "Always preserve the <<DESCRIPTION>> and <</DESCRIPTION>> markers verbatim. "
            "Keep the description under 500 characters."
        ),
    )
    optimized_module = optimizer.compile(
        baseline_module,
        trainset=trainset,
        valset=valset,
    )
    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 5. Extract + validate the evolved description ───────────────────
    evolved_instructions = optimized_module.extract()
    console.print("\n[bold]Validating evolved description[/bold]")
    try:
        evolved_description = extract_description(evolved_instructions)
    except ValueError as e:
        console.print(f"[red]✗ {e}[/red]")
        console.print("[red]✗ Evolved variant is unusable — not deploying[/red]")
        return

    evolved_constraints = validator.validate_tool_description(
        evolved_description, evolved_instructions
    )
    constraints_passed = print_constraints(console, evolved_constraints)
    if not constraints_passed:
        console.print("[red]✗ Evolved description FAILED constraints — not deploying[/red]")
        failed_dir = Path("output") / "tools" / tool_name
        failed_dir.mkdir(parents=True, exist_ok=True)
        (failed_dir / "evolved_FAILED.txt").write_text(evolved_description)
        console.print(f"  Saved failed variant to {failed_dir}/evolved_FAILED.txt")
        return

    # ── 6. Holdout comparison ───────────────────────────────────────────
    # Score the deployable artifact — the extracted description inside a fresh
    # frozen template — not the co-optimized prompt GEPA produced.
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    try:
        evolved_module = ToolDocModule(tool_name, evolved_description)
    except ValueError as e:
        console.print(f"[red]✗ {e}[/red]")
        console.print("[red]✗ Evolved variant is unusable — not deploying[/red]")
        return
    baseline_scores = []
    evolved_scores = []
    for ex in dataset.to_dspy_examples("holdout"):
        with dspy.context(lm=lm):
            baseline_scores.append(
                selection_metric(ex, baseline_module(task_input=ex.task_input))
            )
            evolved_scores.append(
                selection_metric(ex, evolved_module(task_input=ex.task_input))
            )

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Accuracy",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Description Size",
        f"{len(description)} chars",
        f"{len(evolved_description)} chars",
        f"{len(evolved_description) - len(description):+} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 7. Save output ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "tools" / tool_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "baseline_description.txt").write_text(description)
    (output_dir / "evolved_description.txt").write_text(evolved_description)

    metrics = {
        "tool_name": tool_name,
        "timestamp": timestamp,
        "days": days,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_accuracy": avg_baseline,
        "evolved_accuracy": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(description),
        "evolved_size": len(evolved_description),
        "positives": positives,
        "negatives": negatives,
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": constraints_passed,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(
            f"\n[bold green]✓ Evolution improved '{tool_name}' selection by "
            f"{improvement:+.3f}[/bold green]"
        )
        console.print(
            f"  Review the diff: diff {output_dir}/baseline_description.txt "
            f"{output_dir}/evolved_description.txt"
        )
    else:
        console.print(
            f"\n[yellow]⚠ Evolution did not improve selection (change: {improvement:+.3f})[/yellow]"
        )
        console.print("  Try: more iterations, more usage data, or a different optimizer model")


@click.command()
@click.option("--tool", "tool_name", required=True, help="Name of the tool to evolve")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--days", default=90, show_default=True, type=click.IntRange(min=1),
              help="Session-history window in days")
@click.option("--iterations", default=10, show_default=True, help="Number of GEPA iterations")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
def main(tool_name, hermes_repo, days, iterations, eval_model, optimizer_model, dry_run):
    """Evolve a Hermes Agent tool description using DSPy + GEPA."""
    evolve(
        tool_name=tool_name,
        hermes_repo=hermes_repo,
        days=days,
        iterations=iterations,
        eval_model=eval_model,
        optimizer_model=optimizer_model,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()

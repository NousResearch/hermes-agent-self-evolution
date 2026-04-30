"""Evolve a tool description using DSPy + GEPA.

Phase 2 of self-evolution. Mirrors the Phase 1 skill pipeline but optimizes a
single tool description (max 500 chars by default) using contrastive
synthetic eval data: positive tasks (where the tool fits) and negative tasks
(where it does not).

Usage:
    python -m evolution.tools.evolve_tool --tool-def tools/search_files.json
    python -m evolution.tools.evolve_tool --tool-def tools/search_files.json \\
        --iterations 10 --use-llm-judge --create-pr
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import dspy
import litellm
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig, MINIMAX_MODELS, validate_model_string
from evolution.core.constraints import ConstraintValidator
from evolution.core.external_importers import scrub_secrets
from evolution.core.mad_scoring import compute_mad
from evolution.monitor.progress import start_run, log_event, complete_run, fail_run
from evolution.tools.dataset import (
    ToolDatasetBuilder,
    to_dspy_examples_with_polarity,
)
from evolution.tools.fitness import tool_fitness_metric
from evolution.tools.tool_module import (
    ToolModule,
    extract_evolved_description,
    load_tool_definition,
    save_tool_definition,
)


console = Console()

# Same litellm timeout convention as evolve_skill so a hung connection
# cannot wedge the optimizer.
litellm.request_timeout = float(os.environ.get("LITELLM_REQUEST_TIMEOUT", "90"))


def _score_value(pred) -> float:
    if isinstance(pred, (int, float)):
        return float(pred)
    val = getattr(pred, "score", None)
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def evolve_tool(
    tool_def_path: str,
    iterations: int = 10,
    optimizer_model: Optional[str] = None,
    eval_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    use_minimax: bool = False,
    dry_run: bool = False,
    create_pr: bool = False,
    output_dir: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
) -> None:
    # ── Validate user-supplied model strings ──────────────────────────
    for label, m in [("optimizer", optimizer_model), ("eval", eval_model), ("judge", judge_model)]:
        if m is not None:
            try:
                validate_model_string(m)
            except ValueError as exc:
                console.print(f"[red]✗ Invalid --{label}-model: {exc}[/red]")
                sys.exit(1)

    # ── Resolve model defaults ──────────────────────────────────────────
    cfg_defaults = EvolutionConfig()
    minimax_default = f"minimax/{MINIMAX_MODELS[0]}"
    if use_minimax:
        optimizer_model = optimizer_model or minimax_default
        eval_model = eval_model or minimax_default
        judge_model = judge_model or minimax_default
    else:
        optimizer_model = optimizer_model or cfg_defaults.optimizer_model
        eval_model = eval_model or cfg_defaults.eval_model
        judge_model = judge_model or eval_model

    # ── Build config ────────────────────────────────────────────────────
    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
        create_pr=create_pr,
        api_base=api_base,
        api_key=api_key,
    )
    if output_dir:
        config.output_dir = Path(output_dir).expanduser()

    # ── Load the tool definition ───────────────────────────────────────
    tool_path = Path(tool_def_path).expanduser()
    try:
        tool = load_tool_definition(tool_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]✗ {exc}[/red]")
        sys.exit(1)

    tool_name = tool["name"]
    baseline_desc = tool["description"]

    # Register the run for cross-process observability.
    run_meta = start_run(f"tool:{tool_name}", config)
    run_id = run_meta["run_id"]

    console.print(
        f"\n[bold cyan]🔧 Hermes Tool Evolution[/bold cyan] — "
        f"Evolving description for: [bold]{tool_name}[/bold]\n"
    )
    console.print(f"  Source: {tool_path}")
    console.print(f"  Description: {baseline_desc[:80]}{'...' if len(baseline_desc) > 80 else ''}")
    console.print(f"  Size: {len(baseline_desc):,} / {config.max_tool_desc_size} chars")
    log_event(run_id, "loading", f"Loaded {tool_path.name} ({len(baseline_desc):,} chars)")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        complete_run(run_id, {"scoring_method": "dry_run", "constraints_passed": 0})
        return

    # ── Validate baseline against tool_description constraints ─────────
    validator = ConstraintValidator(config)
    baseline_results = validator.validate_all(baseline_desc, "tool_description")
    for r in baseline_results:
        icon = "✓" if r.passed else "✗"
        color = "green" if r.passed else "red"
        console.print(f"  [{color}]{icon} {r.constraint_name}[/{color}]: {r.message}")
    if not all(r.passed for r in baseline_results):
        console.print(
            "[yellow]⚠ Baseline tool description has constraint violations — "
            "proceeding anyway, but evolved-vs-baseline comparison will be unreliable[/yellow]"
        )

    # ── Build contrastive synthetic dataset ────────────────────────────
    console.print(f"\n[bold]Building contrastive eval dataset[/bold] (synthetic)")
    builder = ToolDatasetBuilder(config)
    try:
        dataset = builder.generate(tool_name=tool_name, description=baseline_desc)
    except ValueError as exc:
        console.print(f"[red]✗ Dataset generation failed: {exc}[/red]")
        fail_run(run_id, str(exc))
        sys.exit(1)

    save_path = config.output_dir / "datasets" / "tools" / tool_name
    dataset.save(save_path)
    pos = sum(1 for e in dataset.all_examples if e.category == "positive")
    neg = sum(1 for e in dataset.all_examples if e.category == "negative")
    console.print(
        f"  Generated {len(dataset.all_examples)} examples ({pos} positive / {neg} negative)"
    )
    console.print(
        f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout"
    )
    log_event(
        run_id,
        "dataset_built",
        f"positive={pos} negative={neg} train={len(dataset.train)} val={len(dataset.val)} holdout={len(dataset.holdout)}",
    )

    # ── Configure optimizer and run GEPA ───────────────────────────────
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    lm = config.make_lm(eval_model)
    dspy.configure(lm=lm)

    baseline_module = ToolModule(baseline_desc)
    trainset = to_dspy_examples_with_polarity(dataset, "train")
    valset = to_dspy_examples_with_polarity(dataset, "val")

    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")
    start_time = time.time()
    try:
        optimizer = dspy.GEPA(
            metric=tool_fitness_metric,
            max_metric_calls=iterations * 10,
        )
        optimized_module = optimizer.compile(
            baseline_module, trainset=trainset, valset=valset
        )
        optimizer_used = "GEPA"
    except (TypeError, AttributeError, ImportError) as exc:
        console.print(
            f"[yellow]GEPA not available ({exc.__class__.__name__}: {exc}); "
            "falling back to MIPROv2[/yellow]"
        )
        optimizer = dspy.MIPROv2(metric=tool_fitness_metric, auto="light")
        optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
        optimizer_used = "MIPROv2"

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s using {optimizer_used}")
    log_event(run_id, "optimization_complete", f"completed in {elapsed:.1f}s using {optimizer_used}")

    # ── Extract evolved description ────────────────────────────────────
    evolved_desc = extract_evolved_description(optimized_module, baseline=baseline_desc)
    scrubbed = scrub_secrets(evolved_desc)
    if scrubbed != evolved_desc:
        console.print(
            "[yellow]⚠ Evolved description contained secret-shaped substrings — "
            "redacted before persisting[/yellow]"
        )
        evolved_desc = scrubbed

    # ── Validate evolved against constraints ───────────────────────────
    console.print("\n[bold]Validating evolved description[/bold]")
    evolved_results = validator.validate_all(
        evolved_desc, "tool_description", baseline_text=baseline_desc
    )
    all_pass = True
    for r in evolved_results:
        icon = "✓" if r.passed else "✗"
        color = "green" if r.passed else "red"
        console.print(f"  [{color}]{icon} {r.constraint_name}[/{color}]: {r.message}")
        if not r.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved description FAILED gates — not deploying[/red]")
        failed_dir = config.output_dir / f"tool:{tool_name}" / "FAILED"
        failed_dir.mkdir(parents=True, exist_ok=True)
        save_tool_definition(tool, evolved_desc, failed_dir / f"{tool_name}.json")
        fail_run(run_id, "Evolved description failed gates")
        return

    # ── Holdout eval with MAD confidence ──────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")
    holdout = to_dspy_examples_with_polarity(dataset, "holdout")
    optimized_module_for_eval = optimized_module
    baseline_for_eval = ToolModule(baseline_desc)

    baseline_scores: list[float] = []
    evolved_scores: list[float] = []
    for ex in holdout:
        with dspy.context(lm=lm):
            b_pred = baseline_for_eval(task_input=ex.task_input)
            baseline_scores.append(_score_value(tool_fitness_metric(ex, b_pred)))
            e_pred = optimized_module_for_eval(task_input=ex.task_input)
            evolved_scores.append(_score_value(tool_fitness_metric(ex, e_pred)))

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    mad_label = ""
    mad_confidence = None
    if len(baseline_scores) >= 2:
        deltas = [e - b for e, b in zip(evolved_scores, baseline_scores)]
        mad_value = compute_mad(deltas)
        if mad_value > 0:
            mad_confidence = abs(improvement) / mad_value
        else:
            mad_confidence = float("inf") if abs(improvement) > 0 else 0.0
        if mad_confidence >= 2.0:
            mad_label = "likely real"
        elif mad_confidence >= 1.0:
            mad_label = "marginal"
        else:
            mad_label = "within noise"

    # ── Report ────────────────────────────────────────────────────────
    table = Table(title="Tool Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")
    color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Accuracy",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{color}]{improvement:+.3f}[/{color}]",
    )
    table.add_row(
        "Description Size",
        f"{len(baseline_desc):,} chars",
        f"{len(evolved_desc):,} chars",
        f"{len(evolved_desc) - len(baseline_desc):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    if mad_confidence is not None:
        cc = "green" if mad_label == "likely real" else "yellow" if mad_label == "marginal" else "red"
        cd = f"{mad_confidence:.2f}x" if mad_confidence != float("inf") else "∞"
        table.add_row("MAD Confidence", "", f"[{cc}]{cd} ({mad_label})[/{cc}]", "")
    console.print()
    console.print(table)

    successful = (evolved_desc != baseline_desc) and (improvement > 0)

    # ── Save run output ───────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.output_dir / f"tool:{tool_name}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    save_tool_definition(tool, baseline_desc, run_dir / f"{tool_name}.baseline.json")
    save_tool_definition(tool, evolved_desc, run_dir / f"{tool_name}.evolved.json")

    metrics = {
        "tool_name": tool_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "judge_model": judge_model,
        "optimizer_used": optimizer_used,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "successful_improvement": successful,
        "baseline_size": len(baseline_desc),
        "evolved_size": len(evolved_desc),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "mad_confidence": mad_confidence if mad_confidence not in (None, float("inf")) else None,
        "mad_label": mad_label or None,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    console.print(f"\n  Output saved to {run_dir}/")

    # ── Optional proposal bundle ──────────────────────────────────────
    if config.create_pr and successful:
        proposal_dir = config.output_dir / "proposals" / f"tool:{tool_name}" / timestamp
        proposal_dir.mkdir(parents=True, exist_ok=True)
        save_tool_definition(tool, baseline_desc, proposal_dir / f"{tool_name}.baseline.json")
        save_tool_definition(tool, evolved_desc, proposal_dir / f"{tool_name}.evolved.json")
        (proposal_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        decision = {
            "tool_name": tool_name,
            "source_path": str(tool_path),
            "constraint_results": [
                {"name": c.constraint_name, "passed": c.passed, "message": c.message}
                for c in evolved_results
            ],
            "improvement": improvement,
            "successful_improvement": successful,
        }
        (proposal_dir / "decision.json").write_text(json.dumps(decision, indent=2))
        console.print(f"\n[bold green]✓ Proposal bundle written to {proposal_dir}/[/bold green]")

    complete_run(run_id, {
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(baseline_desc),
        "evolved_size": len(evolved_desc),
        "constraints_passed": 1 if all_pass else 0,
    })

    if successful:
        console.print(
            f"\n[bold green]✓ Evolution improved description by "
            f"{improvement:+.3f} ({improvement / max(0.001, avg_baseline) * 100:+.1f}%)[/bold green]"
        )
    elif evolved_desc == baseline_desc:
        console.print("\n[yellow]⚠ Evolution produced no description change[/yellow]")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve description (change: {improvement:+.3f})[/yellow]")


@click.command()
@click.option("--tool-def", required=True, help="Path to a tool definition JSON file")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--optimizer-model", default=None, help="Model for GEPA reflections")
@click.option("--eval-model", default=None, help="Model for evaluations")
@click.option("--judge-model", default=None, help="Model for synthetic data + holdout judging")
@click.option("--use-minimax", is_flag=True, help="Default models to MiniMax (requires MINIMAX_API_KEY)")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--create-pr", is_flag=True, help="On success, write a proposal bundle")
@click.option("--output-dir", default=None, help="Override output directory (default: ./output)")
@click.option("--api-base", default=None, help="Custom API base URL (vLLM, Ollama, ...)")
@click.option("--api-key", default=None, help="API key for the custom --api-base endpoint")
def main(**kwargs):
    """Evolve a tool description using DSPy + GEPA optimization."""
    evolve_tool(tool_def_path=kwargs.pop("tool_def"), **kwargs)


if __name__ == "__main__":
    main()

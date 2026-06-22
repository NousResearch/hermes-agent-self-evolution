"""CLI orchestrator for Phase 2 tool-description evolution."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console

from evolution.core.config import EvolutionConfig, resolve_model
from evolution.core.constraints import ConstraintValidator
from evolution.tools.targets import TARGET_TOOLS
from evolution.tools.tool_loader import (
    read_target_descriptions, list_all_tools, write_description_to_source,
)
from evolution.tools.tool_module import ToolSelectorModule
from evolution.tools.tool_dataset import ToolDatasetBuilder
from evolution.tools.tool_fitness import (
    selection_score, per_tool_accuracy, overall_accuracy, fitness_with_penalty,
)

console = Console()


def evolve_tools(
    iterations: int = 10,
    optimizer_model: str = "openai/gpt-4o-mini",
    eval_model: str = "openai/gpt-4o-mini",
    hermes_repo: Optional[str] = None,
    dry_run: bool = False,
    write_back: bool = False,
    skip_semantic: bool = False,
    evolve_params: bool = False,
):
    """Evolve the confusable-cluster tool descriptions. Returns None on dry-run/abort.

    When evolve_params=True, also evolves per-parameter descriptions (Phase 4).
    """
    config = EvolutionConfig(
        iterations=iterations, optimizer_model=optimizer_model,
        eval_model=eval_model, judge_model=eval_model,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)
    if skip_semantic:
        config.enable_semantic_check = False

    eval_model = resolve_model(eval_model)
    optimizer_model = resolve_model(optimizer_model)

    repo = config.hermes_agent_path
    if not (repo / "tools").exists():
        console.print(f"[red]✗ hermes-agent tools/ not found at {repo}[/red]")
        sys.exit(1)

    baseline = read_target_descriptions(repo)
    if not baseline:
        console.print(f"[red]✗ No target tool descriptions found in {repo}/tools[/red]")
        sys.exit(1)
    all_tools = list_all_tools(repo)
    console.print(f"  Found {len(baseline)}/{len(TARGET_TOOLS)} target tools; "
                  f"{len(all_tools)} tools total")

    # Phase 4: load parameter descriptions
    target_params = {}
    if evolve_params:
        from evolution.tools.tool_loader import read_all_param_descriptions
        all_params = read_all_param_descriptions(repo)
        target_params = {t: all_params[t] for t in TARGET_TOOLS if t in all_params and all_params[t]}
        console.print(f"  Loaded param descriptions for {len(target_params)} target tools "
                      f"({sum(len(v) for v in target_params.values())} params)")

    if dry_run:
        msg = (f"Would generate dataset, run GEPA ({iterations} iters), "
               f"validate constraints{', write back' if write_back else ''}"
               f"{', evolve params' if evolve_params else ''}.")
        console.print("[bold green]DRY RUN — setup validated.[/bold green]")
        console.print(f"  {msg}")
        return None

    # Pre-flight
    console.print("[bold]Pre-flight check[/bold]")
    try:
        dspy.LM(optimizer_model)("respond with OK")
        console.print(f"  [green]✓ Optimizer model '{optimizer_model}' OK[/green]")
    except Exception as e:
        console.print(f"[red]✗ Optimizer model '{optimizer_model}' unavailable: {e}[/red]")
        sys.exit(1)

    # Dataset
    builder = ToolDatasetBuilder(config)
    dataset = builder.generate(all_tools)
    dataset.save(Path("datasets") / "tools")
    console.print(f"  Dataset: {len(dataset.train)} train / {len(dataset.val)} val / "
                  f"{len(dataset.holdout)} holdout")

    # Modules
    eval_lm = dspy.LM(eval_model)
    dspy.configure(lm=eval_lm)
    reflection_lm = dspy.LM(optimizer_model, temperature=1.0)
    baseline_module = ToolSelectorModule(baseline, all_tools)

    # Baseline per-tool accuracy on holdout
    holdout = dataset.to_dspy_examples("holdout")
    with dspy.context(lm=eval_lm):
        base_pairs = [(ex, baseline_module(task=ex.task)) for ex in holdout]
    baseline_acc = per_tool_accuracy(base_pairs)
    baseline_overall = overall_accuracy(base_pairs)

    # GEPA
    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        return selection_score(example, prediction)

    start = time.time()
    optimizer = dspy.GEPA(metric=metric, max_full_evals=iterations, reflection_lm=reflection_lm)
    optimized = optimizer.compile(
        baseline_module,
        trainset=dataset.to_dspy_examples("train"),
        valset=dataset.to_dspy_examples("val"),
    )
    elapsed = time.time() - start

    # Parse-back + per-tool constraint validation (revert failures to baseline)
    evolved = optimized.descriptions
    validator = ConstraintValidator(config)
    accepted, reverted = {}, []
    for name in TARGET_TOOLS:
        if name not in baseline:
            continue
        cand = evolved.get(name, baseline[name])
        checks = [
            validator._check_size(cand, "tool_description"),
            validator._check_non_empty(cand),
            validator._check_semantic_preservation(cand, baseline[name]),
        ]
        if all(c.passed for c in checks):
            accepted[name] = cand
        else:
            accepted[name] = baseline[name]
            reverted.append(name)
    if reverted:
        console.print(f"[yellow]⚠ Reverted to baseline (constraint fail): {reverted}[/yellow]")

    # Evolved holdout accuracy
    with dspy.context(lm=eval_lm):
        evolved_pairs = [(ex, optimized(task=ex.task)) for ex in holdout]
    evolved_acc = per_tool_accuracy(evolved_pairs)
    evolved_overall = overall_accuracy(evolved_pairs)
    fitness = fitness_with_penalty(evolved_pairs, baseline_acc, config.tblite_regression_threshold)

    # Output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("output") / "tools" / ts
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_descriptions.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    (out / "evolved_descriptions.json").write_text(json.dumps(accepted, indent=2), encoding="utf-8")
    metrics = {
        "timestamp": ts, "iterations": iterations,
        "baseline_accuracy": baseline_overall, "evolved_accuracy": evolved_overall,
        "fitness_with_penalty": fitness,
        "baseline_per_tool": baseline_acc, "evolved_per_tool": evolved_acc,
        "reverted_tools": reverted, "elapsed_seconds": elapsed,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    console.print(f"  Output saved to {out}/")
    console.print(f"  Accuracy: {baseline_overall:.3f} → {evolved_overall:.3f} "
                  f"(fitness {fitness:+.3f})")

    # Optional write-back
    if write_back:
        for name, desc in accepted.items():
            if desc == baseline[name]:
                continue
            for src in sorted((repo / "tools").glob("*.py")):
                res = write_description_to_source(src, baseline[name], desc)
                if res.status == "written":
                    console.print(f"  [green]✓ wrote {name} -> {res.message}[/green]")
                    break
                if res.status == "ambiguous":
                    console.print(f"  [yellow]⚠ {name}: {res.message} (skipped)[/yellow]")
                    break

    return metrics


@click.command()
@click.option("--iterations", default=10, help="GEPA iterations")
@click.option("--optimizer-model", default="openai/gpt-4o-mini")
@click.option("--eval-model", default="openai/gpt-4o-mini")
@click.option("--hermes-repo", default=None)
@click.option("--dry-run", is_flag=True, help="Validate setup without running")
@click.option("--write-back", is_flag=True, default=False,
              help="Apply evolved descriptions to tools/*.py (default: report-only)")
@click.option("--skip-semantic", is_flag=True, default=False)
@click.option("--evolve-params", is_flag=True, default=False,
              help="Also evolve per-parameter descriptions (Phase 4)")
def main(iterations, optimizer_model, eval_model, hermes_repo, dry_run, write_back, skip_semantic, evolve_params):
    evolve_tools(
        iterations=iterations, optimizer_model=optimizer_model, eval_model=eval_model,
        hermes_repo=hermes_repo, dry_run=dry_run, write_back=write_back, skip_semantic=skip_semantic,
        evolve_params=evolve_params,
    )


if __name__ == "__main__":
    main()

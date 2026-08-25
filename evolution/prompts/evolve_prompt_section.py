"""Evolve a named Hermes prompt-builder section using DSPy + GEPA.

Usage:
    python -m evolution.prompts.evolve_prompt_section \
        --section MEMORY_GUIDANCE --iterations 5 --eval-source synthetic
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.constraints import ConstraintValidator
from evolution.core.dataset_builder import EvalDataset, GoldenDatasetLoader, SyntheticDatasetBuilder
from evolution.core.fitness import skill_fitness_metric
from evolution.prompts.prompt_module import PromptSectionModule, list_prompt_sections, load_prompt_section

console = Console()


def evolve(
    section_name: str,
    iterations: int = 5,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    dry_run: bool = False,
):
    """Run prompt-section evolution and save artifacts under output/prompts/."""
    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo),
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
    )

    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving prompt section: [bold]{section_name}[/bold]\n")
    section = load_prompt_section(config.hermes_agent_path, section_name)
    baseline_text = str(section["text"])
    console.print(f"  Loaded: {section['path']}")
    console.print(f"  Size: {len(baseline_text):,} chars")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Available sections: {', '.join(sorted(list_prompt_sections(config.hermes_agent_path)))}")
        return

    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")
    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
    elif eval_source == "synthetic":
        dataset = SyntheticDatasetBuilder(config).generate(
            artifact_text=baseline_text,
            artifact_type="prompt_section",
        )
        save_path = Path("datasets") / "prompts" / section_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)
    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(baseline_text, "prompt_section")
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "yellow"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")

    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)

    baseline_module = PromptSectionModule(baseline_text)
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")
    start_time = time.time()
    optimizer = dspy.GEPA(
        metric=skill_fitness_metric,
        max_full_evals=iterations,
        reflection_lm=dspy.LM(optimizer_model),
    )
    optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    evolved_text = optimized_module.section_text
    evolved_constraints = validator.validate_all(evolved_text, "prompt_section", baseline_text=baseline_text)
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        all_pass = all_pass and c.passed

    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")
    baseline_scores = []
    evolved_scores = []
    with dspy.context(lm=lm):
        for ex in dataset.to_dspy_examples("holdout"):
            baseline_scores.append(skill_fitness_metric(ex, baseline_module(task_input=ex.task_input)).score)
            evolved_scores.append(skill_fitness_metric(ex, optimized_module(task_input=ex.task_input)).score)

    n = max(1, len(baseline_scores))
    baseline_score = sum(baseline_scores) / n
    evolved_score = sum(evolved_scores) / n
    change = evolved_score - baseline_score

    table = Table(title="Prompt Section Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")
    table.add_row("Holdout Score", f"{baseline_score:.3f}", f"{evolved_score:.3f}", f"{change:+.3f}")
    console.print(table)

    output_dir = Path("output") / "prompts" / section_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_section.md").write_text(baseline_text)
    (output_dir / "evolved_section.md").write_text(evolved_text)
    (output_dir / "metrics.json").write_text(json.dumps({
        "section": section_name,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": baseline_score,
        "evolved_score": evolved_score,
        "improvement": change,
        "constraints_passed": all_pass,
        "elapsed_seconds": elapsed,
    }, indent=2))
    console.print(f"\n  Output saved to {output_dir}/")


@click.command()
@click.option("--section", required=True, help="Prompt-builder constant to evolve, e.g. MEMORY_GUIDANCE")
@click.option("--iterations", default=5, help="Number of GEPA iterations")
@click.option("--eval-source", type=click.Choice(["synthetic", "golden"]), default="synthetic")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--dry-run", is_flag=True, help="Validate setup without optimization")
def main(section, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, dry_run):
    evolve(section, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, dry_run)


if __name__ == "__main__":
    main()

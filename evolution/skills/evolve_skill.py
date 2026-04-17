"""Evolve a Hermes Agent skill using native evolutionary algorithm.

Usage:
    python -m evolution.skills.evolve_skill --skill systematic-debugging --num-generations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import FitnessScore
from evolution.core.constraints import ConstraintValidator
from evolution.ea.engine import EvolutionEngine
from evolution.skills.skill_module import load_skill, find_skill, reassemble_skill

console = Console()


def evolve(
    skill_name: str,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    num_islands: int = 3,
    num_generations: int = 10,
    population_size: int = 8,
    dry_run: bool = False,
):
    """Main evolution function — orchestrates the full optimization loop."""

    config = EvolutionConfig(
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
        api_base=api_base,
        api_key=api_key,
        num_islands=num_islands,
        num_generations=num_generations,
        population_size=population_size,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]Hermes Agent Self-Evolution[/bold cyan] — "
                  f"Evolving skill: [bold]{skill_name}[/bold]\n")

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        console.print(f"[red]Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}[/red]")
        sys.exit(1)

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path.relative_to(config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run EA: {num_islands} islands x {num_generations} generations x {population_size} pop")
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

    if eval_source == "golden" and dataset_path:
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
        console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
    elif eval_source == "sessiondb":
        save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        dataset = build_dataset_from_external(
            skill_name=skill_name,
            skill_text=skill["raw"],
            sources=["claude-code", "copilot", "hermes"],
            output_path=save_path,
            model=eval_model,
        )
        if not dataset.all_examples:
            console.print("[red]No relevant examples found from session history[/red]")
            sys.exit(1)
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(artifact_text=skill["raw"], artifact_type="skill")
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    for c in validator.validate_all(skill["raw"], "skill"):
        icon = "+" if c.passed else "x"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")

    # ── 4. Run evolutionary optimization ────────────────────────────────
    console.print(f"\n[bold]Configuring EA[/bold]")
    console.print(f"  Islands: {num_islands}")
    console.print(f"  Generations: {num_generations}")
    console.print(f"  Population/island: {population_size}")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    def on_generation(**kwargs):
        gen = kwargs["generation"]
        best = kwargs["best_score"]
        scores = kwargs["island_scores"]
        migrated = kwargs.get("migrated", False)
        elapsed = kwargs.get("elapsed", 0)
        island_strs = "  ".join(f"I{i}={s:.3f}" for i, s in enumerate(scores))
        mig_str = " [migrated]" if migrated else ""
        console.print(f"  Gen {gen:>3d}  best={best:.3f}  {island_strs}  [{elapsed:.1f}s]{mig_str}")

    console.print(f"\n[bold cyan]Running evolution...[/bold cyan]\n")
    start_time = time.time()

    engine = EvolutionEngine(
        config=config,
        dataset=dataset,
        baseline_text=skill["body"],
        num_islands=num_islands,
        num_generations=num_generations,
        population_size=population_size,
        migration_interval=config.migration_interval,
        stagnation_limit=config.stagnation_limit,
        on_generation=on_generation,
    )
    best = engine.run()

    elapsed = time.time() - start_time
    evolved_body = best.genome
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)
    console.print(f"\n  Evolution completed in {elapsed:.1f}s")

    # ── 5. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validator.validate_all(evolved_full, "skill", baseline_text=skill["raw"])
    all_pass = True
    for c in evolved_constraints:
        icon = "+" if c.passed else "x"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]Evolved skill FAILED constraints — not deploying[/red]")
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 6. Report results ───────────────────────────────────────────────
    baseline_score = 0.0  # Baseline was the seed genome with score from initial eval
    evolved_score = best.score
    fitness = best.fitness or FitnessScore()

    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Composite Score", f"{evolved_score:.3f}")
    table.add_row("Correctness", f"{fitness.correctness:.3f}")
    table.add_row("Procedure Following", f"{fitness.procedure_following:.3f}")
    table.add_row("Conciseness", f"{fitness.conciseness:.3f}")
    table.add_row("Skill Size", f"{len(evolved_body):,} chars ({len(evolved_body) - len(skill['body']):+,})")
    table.add_row("Time", f"{elapsed:.1f}s")
    table.add_row("Islands x Generations", f"{num_islands} x {num_generations}")
    table.add_row("Mutation Type", best.mutation_type)

    console.print()
    console.print(table)

    # ── 7. Save output ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "evolved_skill.md").write_text(evolved_full)
    (output_dir / "baseline_skill.md").write_text(skill["raw"])
    (output_dir / "metrics.json").write_text(json.dumps({
        "skill_name": skill_name,
        "timestamp": timestamp,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "num_islands": num_islands,
        "num_generations": num_generations,
        "population_size": population_size,
        "evolved_score": evolved_score,
        "correctness": fitness.correctness,
        "procedure_following": fitness.procedure_following,
        "conciseness": fitness.conciseness,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "mutation_type": best.mutation_type,
        "generation": best.generation,
    }, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if evolved_score > 0.5:
        console.print(f"\n[bold green]Evolution complete — score {evolved_score:.3f}[/bold green]")
        console.print(f"  Review: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]Low score ({evolved_score:.3f}) — try more generations or better eval data[/yellow]")


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for mutations/crossover")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluation")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--api-base", default=None, envvar="OPENAI_API_BASE", help="Custom OpenAI-compatible API base URL")
@click.option("--api-key", default=None, envvar="OPENAI_API_KEY", help="API key (or set OPENAI_API_KEY env var)")
@click.option("--num-islands", default=3, help="Number of evolution islands")
@click.option("--num-generations", default=10, help="Number of generations")
@click.option("--population-size", default=8, help="Population size per island")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
def main(skill, eval_source, dataset_path, optimizer_model, eval_model,
         hermes_repo, api_base, api_key, num_islands, num_generations,
         population_size, dry_run):
    """Evolve a Hermes Agent skill using evolutionary optimization."""
    evolve(
        skill_name=skill,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        api_base=api_base,
        api_key=api_key,
        num_islands=num_islands,
        num_generations=num_generations,
        population_size=population_size,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()

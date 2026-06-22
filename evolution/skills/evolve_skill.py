"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import EvolutionConfig, get_hermes_agent_path, resolve_model, resolve_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import skill_fitness_metric, LLMJudge, FitnessScore
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4o-mini",
    eval_model: str = "openai/gpt-4o-mini",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    create_pr: bool = False,
    benchmark_gate: bool = False,
    skip_semantic: bool = False,
):
    """Main evolution function — orchestrates the full optimization loop."""

    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo),
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
    )

    # Resolve model aliases
    eval_model = resolve_model(eval_model)
    optimizer_model = resolve_model(optimizer_model)

    if not dry_run:
        # Pre-flight: verify optimizer model responds
        console.print(f"\n[bold]Pre-flight check[/bold]")
        try:
            test_lm = dspy.LM(optimizer_model)
            test_lm("respond with OK")
            console.print(f"  [green]✓ Optimizer model '{optimizer_model}' OK[/green]")
        except Exception as e:
            console.print(f"[red]✗ Optimizer model '{optimizer_model}' not available: {e}[/red]")
            console.print("[yellow]  Check your LITELLM_MODEL / OPENAI_API_KEY / OPENROUTER_API_KEY settings[/yellow]")
            sys.exit(1)

        if optimizer_model == eval_model:
            console.print("[yellow]⚠ Optimizer and eval model are the same — fitness scores may be biased[/yellow]")

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]** Hermes Agent Self-Evolution **[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        console.print(f"[red]✗ Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}[/red]")
        sys.exit(1)

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path.relative_to(config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if skip_semantic:
        config.enable_semantic_check = False

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Would validate constraints and create PR")
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
            console.print("[red]✗ No relevant examples found from session history[/red]")
            sys.exit(1)
        console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
    elif eval_source == "synthetic":
        builder = SyntheticDatasetBuilder(config)
        dataset = builder.generate(
            artifact_text=skill["raw"],
            artifact_type="skill",
        )
        # Save for reuse
        save_path = Path("datasets") / "skills" / skill_name
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
        console.print(f"  Saved to {save_path}/")
    elif dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
        sys.exit(1)

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate constraints on baseline ─────────────────────────────
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(skill["body"], "skill")
    all_pass = True
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]")

    # ── 4. Set up DSPy + GEPA optimizer ─────────────────────────────────
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    # Configure DSPy
    eval_lm = dspy.LM(eval_model)
    dspy.configure(lm=eval_lm)

    # GEPA requires a reflection LM — use the optimizer_model with higher temperature
    reflection_lm = dspy.LM(optimizer_model, temperature=1.0)

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    # Create LLM judge for fitness scoring
    llm_judge = LLMJudge(config)

    # Create metric with judge wired in
    def metric_with_judge(example, prediction, trace=None, pred_name=None, pred_trace=None):
        return skill_fitness_metric(
            example, prediction,
            trace=trace, pred_name=pred_name, pred_trace=pred_trace,
            judge=llm_judge,
        )

    optimizer = dspy.GEPA(
        metric=metric_with_judge,
        max_full_evals=iterations,
        reflection_lm=reflection_lm,
    )

    optimized_module = optimizer.compile(
        baseline_module,
        trainset=trainset,
        valset=valset,
    )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ───────────────────────────────────
    # skill_text reads from the optimized module's predictor signature instructions
    evolved_body = optimized_module.skill_text
    if not evolved_body or evolved_body.isspace():
        console.print("[red]✗ Evolved body is empty — using baseline as fallback[/red]")
        evolved_body = skill["body"]
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")

    # Validate size, growth, and non-empty on the evolved BODY (not the full
    # file with frontmatter — that would inflate size and growth stats).
    # Use "skill_body" type to skip frontmatter structure checks on the body.
    evolved_constraints = validator.validate_all(evolved_body, "skill_body", baseline_text=skill["body"])
    # Also verify the reassembled file preserves valid skill structure
    structure_result = validator._check_skill_structure(evolved_full)
    evolved_constraints.append(structure_result)

    # Check semantic preservation (compares evolved body to original body)
    semantic_result = validator._check_semantic_preservation(evolved_body, skill["body"])
    evolved_constraints.append(semantic_result)

    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        # Still save for inspection
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full, encoding="utf-8")
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        # Score baseline
        with dspy.context(lm=eval_lm):
            baseline_pred = baseline_module(task_input=ex.task_input)
            baseline_score = skill_fitness_metric(ex, baseline_pred)
            baseline_scores.append(baseline_score)

            evolved_pred = optimized_module(task_input=ex.task_input)
            evolved_score = skill_fitness_metric(ex, evolved_pred)
            evolved_scores.append(evolved_score)

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 9. Report results ───────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Skill Size",
        f"{len(skill['body']):,} chars",
        f"{len(evolved_body):,} chars",
        f"{len(evolved_body) - len(skill['body']):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 10. Save output ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save evolved skill
    (output_dir / "evolved_skill.md").write_text(evolved_full, encoding="utf-8")

    # Save baseline for comparison
    (output_dir / "baseline_skill.md").write_text(skill["raw"], encoding="utf-8")

    # Save metrics
    metrics = {
        "skill_name": skill_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    # ── 11. Create PR (optional) ─────────────────────────────────────
    if create_pr and improvement > 0 and all_pass:
        try:
            from evolution.core.pr_builder import create_evolution_branch
            branch_name = create_evolution_branch(
                skill_name=skill_name,
                evolved_skill_path=output_dir / "evolved_skill.md",
                baseline_skill_path=output_dir / "baseline_skill.md",
                metrics_path=output_dir / "metrics.json",
                hermes_repo=config.hermes_agent_path,
            )
            console.print(f"\n[bold green]✓ Created branch: {branch_name}[/bold green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to create PR: {e}[/red]")
            console.print("[yellow]  Ensure gh CLI is authenticated and HERMES_AGENT_REPO is clean[/yellow]")

    # ── 12. Benchmark gate (optional) ─────────────────────────────────
    if benchmark_gate and improvement > 0:
        try:
            from evolution.core.benchmark_gate import run_benchmark_gate
            gate_result = run_benchmark_gate(
                skill_name=skill_name,
                hermes_repo=config.hermes_agent_path,
                baseline_module=baseline_module,
                evolved_module=optimized_module,
            )
            if not gate_result.passed:
                console.print(f"[red]✗ Benchmark gate FAILED: {gate_result.message}[/red]")
            else:
                console.print(f"[green]✓ Benchmark gate passed: {gate_result.message}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Benchmark gate error (non-fatal): {e}[/yellow]")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve skill (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4o-mini", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4o-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--pr", "create_pr", is_flag=True, default=False,
              help="Create a git branch + PR with the evolved skill")
@click.option("--benchmark-gate", is_flag=True, default=False,
              help="Run TBLite fast subset as regression gate (expensive)")
@click.option("--skip-semantic", is_flag=True, default=False,
              help="Skip semantic preservation check")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests, dry_run, create_pr, benchmark_gate, skip_semantic):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
        create_pr=create_pr,
        benchmark_gate=benchmark_gate,
        skip_semantic=skip_semantic,
    )


if __name__ == "__main__":
    main()

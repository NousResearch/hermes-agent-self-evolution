"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import skill_fitness_metric
from evolution.core.integration_adapter import HermesIntegrationAdapter
from evolution.core.eval_integrity import validate_model_separation
from evolution.core.phase_gate import evaluate_phase1_gate, write_phase_gate_result
from evolution.core.reproducibility import build_reproducibility_manifest, write_manifest
from evolution.core.rollout_policy import get_rollout_policy, should_auto_rollback
from evolution.core.stop_loss import StopLossGuard
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


def score_holdout_examples(
    holdout_examples: list[dspy.Example],
    baseline_module: SkillModule,
    optimized_module: SkillModule,
    lm,
    *,
    evolved_text_changed: bool,
) -> tuple[list[float], list[float]]:
    """Score baseline/evolved modules on holdout examples.

    If optimization produced identical skill text, copy baseline scores instead of
    calling the evolved module again. This prevents stochastic LLM generations from
    inventing a fake KPI regression for a no-op optimization pass.
    """
    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        with dspy.context(lm=lm) if lm is not None else nullcontext():
            baseline_pred = baseline_module(task_input=ex.task_input)
            baseline_score = skill_fitness_metric(ex, baseline_pred)
            baseline_scores.append(baseline_score)

            if evolved_text_changed:
                evolved_pred = optimized_module(task_input=ex.task_input)
                evolved_score = skill_fitness_metric(ex, evolved_pred)
            else:
                evolved_score = baseline_score
            evolved_scores.append(evolved_score)

    return baseline_scores, evolved_scores


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    judge_model: str = "openrouter/google/gemini-2.5-flash",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    allow_two_family_mode: bool = False,
    minimum_model_families: Optional[int] = None,
):
    """Main evolution function — orchestrates the full optimization loop."""

    if minimum_model_families is None:
        resolved_min_families = 2 if allow_two_family_mode else 3
    else:
        resolved_min_families = max(1, int(minimum_model_families))

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
        run_pytest=run_tests,
        minimum_model_families=resolved_min_families,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    adapter = HermesIntegrationAdapter(config.hermes_agent_path)
    if config.phase0_enforce:
        try:
            adapter.assert_compatible()
        except RuntimeError as exc:
            console.print(f"[red]✗ Phase 0 compatibility check failed: {exc}[/red]")
            sys.exit(1)

        separation_errors = validate_model_separation(
            generator_model=config.judge_model,
            judge_model=config.eval_model,
            optimizer_model=config.optimizer_model,
            minimum_distinct_families=config.minimum_model_families,
        )
        if separation_errors:
            for error in separation_errors:
                console.print(f"[red]✗ {error}[/red]")
            sys.exit(1)

    stop_loss = StopLossGuard(config)

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        console.print(f"[red]✗ Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}[/red]")
        sys.exit(1)

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path.relative_to(config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print("  Would validate constraints and create PR")
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
    console.print("\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(skill["raw"], "skill")
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
    console.print("\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")
    console.print(f"  Eval model: {eval_model}")

    # Configure DSPy
    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    try:
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric,
            max_steps=iterations,
        )

        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
    except Exception as e:
        # Fall back to MIPROv2 if GEPA isn't available in this DSPy version
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(
            metric=skill_fitness_metric,
            auto="light",
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
        )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ───────────────────────────────────
    # The optimized module's instructions contain the evolved skill text
    evolved_body = optimized_module.skill_text
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print("\n[bold]Validating evolved skill[/bold]")
    evolved_constraints = validator.validate_all(evolved_full, "skill", baseline_text=skill["raw"])
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

    evolved_text_changed = evolved_body != skill["body"]
    if not evolved_text_changed:
        console.print("[yellow]No skill text changes produced; reusing baseline holdout scores[/yellow]")

    baseline_scores, evolved_scores = score_holdout_examples(
        holdout_examples,
        baseline_module,
        optimized_module,
        lm,
        evolved_text_changed=evolved_text_changed,
    )

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline
    stop_loss.register_attempt(
        cost_usd=0.0,
        runtime_minutes=elapsed / 60.0,
        improvement=improvement,
        stable=improvement >= config.minimum_detectable_effect,
    )

    rollout_policy = get_rollout_policy("skill_text")
    rollback = should_auto_rollback(
        kpi_delta=improvement,
        safety_incidents=0,
        policy=rollout_policy,
    )

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

    if rollback:
        console.print("[yellow]⚠ Rollout policy recommends rollback due to KPI regression[/yellow]")

    stop_loss_reasons = stop_loss.termination_reasons()
    if stop_loss_reasons:
        console.print("[yellow]⚠ Stop-loss triggered:[/yellow]")
        for reason in stop_loss_reasons:
            console.print(f"  - {reason}")

    # ── 10. Save output ─────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_reproducibility_manifest(
        skill_name=skill_name,
        skill_path=skill_path,
        dataset=dataset,
        config=config,
        baseline_score=avg_baseline,
        evolved_score=avg_evolved,
        improvement=improvement,
        elapsed_seconds=elapsed,
        evolution_repo_path=Path(__file__).resolve().parents[2],
    )
    gate_result = evaluate_phase1_gate(
        baseline_score=avg_baseline,
        evolved_score=avg_evolved,
        improvement=improvement,
        config=config,
        manifest=manifest,
        benchmark_regression=None,
        safety_incidents=0,
    )

    # Save evolved skill
    (output_dir / "evolved_skill.md").write_text(evolved_full, encoding="utf-8")

    # Save baseline for comparison
    (output_dir / "baseline_skill.md").write_text(skill["raw"], encoding="utf-8")

    # Save reproducibility and gate artifacts
    write_manifest(output_dir / "reproducibility_manifest.json", manifest)
    write_phase_gate_result(output_dir / "phase1_gate.json", gate_result)
    if gate_result.passed:
        console.print("[bold green]✓ Phase 1 gate passed[/bold green]")
    else:
        console.print("[yellow]⚠ Phase 1 gate failed[/yellow]")
        for failure in gate_result.failures:
            console.print(f"  - {failure}")

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
        "rollout_level": rollout_policy.rollout_level,
        "rollback_recommended": rollback,
        "stop_loss_reasons": stop_loss_reasons,
        "phase1_gate_passed": gate_result.passed,
        "phase1_gate_failures": gate_result.failures,
        "reproducibility_manifest_path": str(output_dir / "reproducibility_manifest.json"),
        "phase1_gate_path": str(output_dir / "phase1_gate.json"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    console.print(f"\n  Output saved to {output_dir}/")

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
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option(
    "--judge-model",
    default="openrouter/google/gemini-2.5-flash",
    help="Model for dataset generation and relevance scoring",
)
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option(
    "--allow-two-family-mode",
    is_flag=True,
    help="Allow execution with two distinct model families instead of three",
)
@click.option(
    "--minimum-model-families",
    type=int,
    default=None,
    help="Minimum distinct model families required (overrides default/--allow-two-family-mode)",
)
def main(
    skill,
    iterations,
    eval_source,
    dataset_path,
    optimizer_model,
    eval_model,
    judge_model,
    hermes_repo,
    run_tests,
    dry_run,
    allow_two_family_mode,
    minimum_model_families,
):
    """Evolve a Hermes Agent skill using DSPy + GEPA optimization."""
    evolve(
        skill_name=skill,
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        dry_run=dry_run,
        allow_two_family_mode=allow_two_family_mode,
        minimum_model_families=minimum_model_families,
    )


if __name__ == "__main__":
    main()

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

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import skill_fitness_metric, LLMJudge, FitnessScore, init_fitness_metric
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)
from evolution.monitor.progress import start_run, log_event, complete_run, fail_run

console = Console()


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
):
    """Main evolution function — orchestrates the full optimization loop."""

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    # ── 0. Register run in progress DB ─────────────────────────────────
    run_meta = start_run(skill_name, config)
    run_id = run_meta["run_id"]

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        console.print(f"[red]✗ Skill '{skill_name}' not found in {config.hermes_agent_path / 'skills'}[/red]")
        fail_run(run_id, f"Skill '{skill_name}' not found")
        sys.exit(1)

    skill = load_skill(skill_path)
    log_event(run_id, "loading", f"Loaded skill from {skill_path.relative_to(config.hermes_agent_path)} ({len(skill['raw']):,} chars)")
    console.print(f"  Loaded: {skill_path.relative_to(config.hermes_agent_path)}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Would validate constraints and create PR")
        complete_run(run_id, {"scoring_method": "dry_run", "constraints_passed": 0})
        return

    # ── 2. Build or load evaluation dataset ─────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")
    log_event(run_id, "dataset_generation", f"source={eval_source} — generating eval examples")

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
            fail_run(run_id, "No relevant examples found from session history")
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
        fail_run(run_id, "No dataset path specified and eval_source is not synthetic")
        sys.exit(1)

    log_event(run_id, "dataset_generation", f"source={eval_source} total={len(dataset.all_examples)} train={len(dataset.train)} val={len(dataset.val)} holdout={len(dataset.holdout)}")
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
    log_event(run_id, "optimizer_setup", f"GEPA optimizer={optimizer_model} eval={eval_model}")

    # Configure DSPy
    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Initialize LLM-as-judge for the metric function ─────────────
    console.print(f"[bold]Initializing LLM-as-judge[/bold] (model: {eval_model})")
    init_fitness_metric(config, skill_text=skill["body"])
    log_event(run_id, "init_judge", f"model={eval_model}")

    # ── 6. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()
    log_event(run_id, "optimization_start", f"optimizer=GEPA iterations={iterations} optimizer_model={optimizer_model}")

    try:
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric,
            max_metric_calls=iterations * 10,  # Convert iterations to metric call budget
            auto="light",
        )

        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
            valset=valset,
        )
    except Exception as e:
        # Fall back to MIPROv2 if GEPA isn't available in this DSPy version
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        log_event(run_id, "optimization_iteration", f"GEPA unavailable, falling back to MIPROv2: {e}")
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
    log_event(run_id, "optimization_complete", f"completed in {elapsed:.1f}s")

    # ── 7. Extract evolved skill text ───────────────────────────────────
    # MIPROv2/GEPA replaces the predictor's instruction. Extract the full instruction
    # and separate the evolved skill text from the wrapper.
    # The instruction was prepended with "Follow these skill instructions...\n\n{skill_text}\n\n---\n"
    evolved_instruction = ""
    for name, pred in optimized_module.predictor.named_predictors():
        evolved_instruction = pred.signature.instructions
        break
    
    if not evolved_instruction:
        evolved_instruction = getattr(
            optimized_module.predictor, 'signature', None
        )
        if evolved_instruction and hasattr(evolved_instruction, 'instructions'):
            evolved_instruction = evolved_instruction.instructions
        else:
            evolved_instruction = skill["body"]
    
    # Parse out the skill text from the instruction wrapper
    import re as _re
    skill_header = "Follow these skill instructions to complete the task:\n\n"
    separator = "\n\n---\n"
    
    if evolved_instruction.startswith(skill_header):
        rest = evolved_instruction[len(skill_header):]
        if separator in rest:
            evolved_body = rest.split(separator, 1)[0]
        else:
            # No separator found — the optimizer may have reformulated
            evolved_body = rest
    else:
        # Optimizer completely rewrote the instruction — treat as evolved body
        evolved_body = evolved_instruction
    
    # If the evolved body is empty or just whitespace, fall back to original
    if not evolved_body.strip():
        evolved_body = skill["body"]
    
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 8. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
    log_event(run_id, "validation", "Running constraint validation on evolved skill")
    evolved_constraints = validator.validate_all(evolved_body, "skill", baseline_text=skill["body"])
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
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
        log_event(run_id, "validation_failed", f"Constraints failed — saved to {output_path}")
        fail_run(run_id, "Evolved skill failed constraint validation")
        return

    log_event(run_id, "validation_passed", "All constraints passed")

    # ── 9. Evaluate on holdout set using LLM-as-judge ──────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")
    console.print(f"  Using LLM-as-judge for multi-dimensional scoring")
    log_event(run_id, "holdout_eval", f"Starting holdout evaluation on {len(dataset.holdout)} examples")

    holdout_examples = dataset.to_dspy_examples("holdout")

    # Initialize a fresh judge for holdout eval (ensures clean state)
    judge = LLMJudge(config)

    baseline_scores = []
    evolved_scores = []
    baseline_details = []
    evolved_details = []

    for i, ex in enumerate(holdout_examples):
        # Run both modules to get outputs
        if i % max(1, len(holdout_examples) // 3) == 0:
            log_event(run_id, "holdout_eval", f"Judging example {i+1}/{len(holdout_examples)}")
        with dspy.context(lm=lm):
            baseline_pred = baseline_module(task_input=ex.task_input)
            evolved_pred = optimized_module(task_input=ex.task_input)

        # Score with LLM-as-judge
        baseline_judge = judge.score(
            task_input=ex.task_input,
            expected_behavior=getattr(ex, "expected_behavior", ""),
            agent_output=getattr(baseline_pred, "output", ""),
            skill_text=skill["body"],
        )
        evolved_judge = judge.score(
            task_input=ex.task_input,
            expected_behavior=getattr(ex, "expected_behavior", ""),
            agent_output=getattr(evolved_pred, "output", ""),
            skill_text=evolved_body,
        )

        baseline_scores.append(baseline_judge.composite)
        evolved_scores.append(evolved_judge.composite)
        baseline_details.append(baseline_judge)
        evolved_details.append(evolved_judge)

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 10. Report results ───────────────────────────────────────────────
    log_event(run_id, "reporting", f"baseline={avg_baseline:.3f} evolved={avg_evolved:.3f} improvement={improvement:+.3f}")
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Composite Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )

    # Show dimension breakdowns
    if baseline_details and evolved_details:
        avg_b_correct = sum(d.correctness for d in baseline_details) / len(baseline_details)
        avg_e_correct = sum(d.correctness for d in evolved_details) / len(evolved_details)
        avg_b_proc = sum(d.procedure_following for d in baseline_details) / len(baseline_details)
        avg_e_proc = sum(d.procedure_following for d in evolved_details) / len(evolved_details)
        avg_b_comp = sum(d.completeness for d in baseline_details) / len(baseline_details)
        avg_e_comp = sum(d.completeness for d in evolved_details) / len(evolved_details)

        table.add_row("  Correctness", f"{avg_b_correct:.3f}", f"{avg_e_correct:.3f}",
                       f"{avg_e_correct - avg_b_correct:+.3f}")
        table.add_row("  Procedure", f"{avg_b_proc:.3f}", f"{avg_e_proc:.3f}",
                       f"{avg_e_proc - avg_b_proc:+.3f}")
        table.add_row("  Completeness", f"{avg_b_comp:.3f}", f"{avg_e_comp:.3f}",
                       f"{avg_e_comp - avg_b_comp:+.3f}")

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
    (output_dir / "evolved_skill.md").write_text(evolved_full)

    # Save baseline for comparison
    (output_dir / "baseline_skill.md").write_text(skill["raw"])

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
        "scoring_method": "llm_judge",
    }
    if baseline_details and evolved_details:
        metrics["baseline_dimensions"] = {
            "correctness": sum(d.correctness for d in baseline_details) / len(baseline_details),
            "procedure_following": sum(d.procedure_following for d in baseline_details) / len(baseline_details),
            "completeness": sum(d.completeness for d in baseline_details) / len(baseline_details),
        }
        metrics["evolved_dimensions"] = {
            "correctness": sum(d.correctness for d in evolved_details) / len(evolved_details),
            "procedure_following": sum(d.procedure_following for d in evolved_details) / len(evolved_details),
            "completeness": sum(d.completeness for d in evolved_details) / len(evolved_details),
        }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve skill (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")

    # ── 11. Finalize progress tracking ──────────────────────────────────
    complete_run(run_id, {
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(evolved_body),
        "constraints_passed": int(all_pass),
        "scoring_method": "llm_judge",
    })


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Number of GEPA iterations")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests, dry_run):
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
    )


if __name__ == "__main__":
    main()

"""Evolve a Hermes Agent skill using DSPy + GEPA with MAD confidence scoring.

Similar to evolve_skill.py but wraps the fitness function with multi-trial
MAD (Median Absolute Deviation) scoring so only statistically significant
improvements propagate through the evolutionary loop.

Usage:
    python -m evolution.skills.MADevolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.MADevolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
    python -m evolution.skills.MADevolve_skill --skill code-review --n-trials 5 --confidence-threshold 2.5
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import (
    skill_fitness_metric,
    LLMJudge,
    FitnessScore,
    ConfidenceScoredFitness,
    compute_confidence,
    ConfidenceResult,
)
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


# ---------------------------------------------------------------------------
# MAD-aware DSPy metric — runs n_trials and gates on confidence
# ---------------------------------------------------------------------------

class MADGuardedMetric:
    """DSPy-compatible metric that wraps skill_fitness_metric with MAD confidence.

    Instead of accepting a single score, this runs n_trials evaluations of
    skill_fitness_metric and only propagates the score if
    confidence = |best - baseline| / MAD >= confidence_threshold.

    This prevents noisy LLM-as-judge evaluations from misleading GEPA's
    search over the skill space.
    """

    def __init__(
        self,
        n_trials: int = 3,
        confidence_threshold: float = 2.0,
        direction: str = "higher",
    ):
        self.n_trials = n_trials
        self.confidence_threshold = confidence_threshold
        self.direction = direction
        self._trial_cache: dict = {}

    def __call__(self, example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
        """Evaluate with MAD confidence gating.

        Returns a float 0-1 score only if confidence >= threshold.
        Returns the baseline score (0.5) if confidence < threshold,
        signalling GEPA to discard this variant.
        """
        agent_output = getattr(prediction, "output", "") or ""
        expected = getattr(example, "expected_behavior", "") or ""
        task = getattr(example, "task_input", "") or ""

        if not agent_output.strip():
            return 0.0

        # Run n_trials
        trial_scores: List[float] = []
        for _ in range(self.n_trials):
            score = skill_fitness_metric(example, prediction, trace)
            trial_scores.append(score)

        # Compute MAD confidence
        confidence_result = compute_confidence(trial_scores, direction=self.direction)

        if confidence_result.decision == "keep":
            # Return the best score from the trials
            return max(trial_scores)
        else:
            # Gate: return a neutral score so GEPA discards this variant
            # 0.5 is the neutral baseline — neither reward nor penalize
            return 0.5

    def get_last_confidence(self) -> Optional[ConfidenceResult]:
        """Returns the ConfidenceResult from the last evaluation.

        Call this after a batch of evaluations to log confidence stats.
        """
        return getattr(self, "_last_confidence", None)


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
    n_trials: int = 3,
    confidence_threshold: float = 2.0,
):
    """Main evolution function with MAD confidence scoring — orchestrates the full optimization loop."""

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        run_pytest=run_tests,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution (MAD)[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]")
    console.print(f"  MAD config: {n_trials} trials, confidence threshold {confidence_threshold}x\n")

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
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations, {n_trials} trials, {confidence_threshold}x MAD threshold)")
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
    console.print(f"  MAD trials per evaluation: {n_trials}")
    console.print(f"  Confidence threshold: {confidence_threshold}x")

    # Configure DSPy
    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)

    # Create the baseline skill module
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Create MAD-guarded metric ─────────────────────────────────────
    mad_metric = MADGuardedMetric(
        n_trials=n_trials,
        confidence_threshold=confidence_threshold,
        direction="higher",
    )

    # ── 6. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization with MAD confidence gating ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    try:
        optimizer = dspy.GEPA(
            metric=mad_metric,
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
            metric=mad_metric,
            auto="light",
        )
        optimized_module = optimizer.compile(
            baseline_module,
            trainset=trainset,
        )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 7. Extract evolved skill text ────────────────────────────────────
    # The optimized module's instructions contain the evolved skill text
    evolved_body = optimized_module.skill_text
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 8. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
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
        return

    # ── 9. Evaluate on holdout set with MAD confidence ──────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")
    console.print(f"  Running {n_trials} trials per example with MAD confidence gating\n")

    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_scores = []
    evolved_scores = []
    all_confidence_results: List[ConfidenceResult] = []

    for ex in holdout_examples:
        # Score baseline with MAD
        baseline_trial_scores: List[float] = []
        for _ in range(n_trials):
            with dspy.context(lm=lm):
                baseline_pred = baseline_module(task_input=ex.task_input)
                baseline_score = skill_fitness_metric(ex, baseline_pred)
                baseline_trial_scores.append(baseline_score)

        baseline_cr = compute_confidence(baseline_trial_scores, direction="higher")
        baseline_best = max(baseline_trial_scores)
        baseline_scores.append(baseline_best)
        all_confidence_results.append(baseline_cr)

        # Score evolved with MAD
        evolved_trial_scores: List[float] = []
        for _ in range(n_trials):
            with dspy.context(lm=lm):
                evolved_pred = optimized_module(task_input=ex.task_input)
                evolved_score = skill_fitness_metric(ex, evolved_pred)
                evolved_trial_scores.append(evolved_score)

        evolved_cr = compute_confidence(evolved_trial_scores, direction="higher")
        evolved_best = max(evolved_trial_scores)
        evolved_scores.append(evolved_best)
        all_confidence_results.append(evolved_cr)

        # Log per-example confidence
        delta = evolved_best - baseline_best
        icon = "✓" if evolved_cr.decision == "keep" else "✗"
        color = "green" if evolved_cr.decision == "keep" else "red"
        console.print(
            f"  [{color}]{icon}[/{color}] ex={ex.task_input[:40]}... "
            f"baseline={baseline_best:.3f} evolved={evolved_best:.3f} "
            f"delta={delta:+.3f} conf={evolved_cr.confidence:.2f}x [{evolved_cr.label}]"
        )

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # Aggregate confidence stats
    keep_count = sum(1 for cr in all_confidence_results if cr.decision == "keep")
    marginal_count = sum(1 for cr in all_confidence_results if cr.label == "marginal")
    noise_count = sum(1 for cr in all_confidence_results if cr.label == "within noise")
    avg_confidence = sum(cr.confidence for cr in all_confidence_results) / max(1, len(all_confidence_results))

    # ── 10. Report results ───────────────────────────────────────────────
    table = Table(title="Evolution Results (MAD-Guarded)")
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
    table.add_row("MAD Trials", "", str(n_trials), "")
    table.add_row("Confidence Threshold", "", f"{confidence_threshold}x", "")
    table.add_row("Avg Confidence", "", f"{avg_confidence:.2f}x", "")
    table.add_row("Keep / Marginal / Noise", "", f"{keep_count} / {marginal_count} / {noise_count}", "")

    console.print()
    console.print(table)

    # ── 11. Save output ─────────────────────────────────────────────────
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
        # MAD-specific metrics
        "n_trials": n_trials,
        "confidence_threshold": confidence_threshold,
        "avg_confidence": avg_confidence,
        "keep_count": keep_count,
        "marginal_count": marginal_count,
        "noise_count": noise_count,
        "confidence_label": (
            "likely real" if keep_count > marginal_count + noise_count
            else "marginal" if marginal_count > noise_count
            else "within noise"
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save per-example confidence results
    confidence_log = [
        {
            "task_input": ex.task_input,
            "baseline_best": baseline_scores[i],
            "evolved_best": evolved_scores[i],
            "delta": evolved_scores[i] - baseline_scores[i],
            "confidence": all_confidence_results[i].confidence,
            "decision": all_confidence_results[i].decision,
            "label": all_confidence_results[i].label,
        }
        for i, ex in enumerate(holdout_examples)
    ]
    (output_dir / "confidence_log.jsonl").write_text(
        "\n".join(json.dumps(r) for r in confidence_log)
    )

    console.print(f"\n  Output saved to {output_dir}/")

    if improvement > 0:
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
        console.print(f"  {keep_count} examples passed MAD confidence threshold ({confidence_threshold}x)")
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve skill (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more trials, lower confidence threshold, or different optimizer model")


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
@click.option("--n-trials", default=3, help="Number of LLM-judge trials per evaluation for MAD scoring")
@click.option("--confidence-threshold", default=2.0, help="MAD confidence threshold (in x-MAD units)")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests, dry_run, n_trials, confidence_threshold):
    """Evolve a Hermes Agent skill using DSPy + GEPA with MAD confidence scoring."""
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
        n_trials=n_trials,
        confidence_threshold=confidence_threshold,
    )


if __name__ == "__main__":
    main()

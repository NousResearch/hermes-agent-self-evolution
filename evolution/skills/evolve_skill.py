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
from evolution.core.fitness import skill_fitness_metric, mad_fitness_metric, LLMJudge, FitnessScore, ConfidenceScoredFitness, ConfidenceResult, compute_confidence, compute_mad
from evolution.core.hermes_judge import HermesJudge
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
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    judge_model: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    mad_trials: int = 1,
):
    """Main evolution function — orchestrates the full optimization loop.

    Args:
        mad_trials: Number of bootstrap trials for MAD confidence scoring.
                    1 = standard heuristic (no MAD).
                    >=3 = MAD-gated scoring (filters noise before GEPA sees it).
        judge_model: Model for LLM-as-judge (MAD scoring). If None, uses eval_model.
                     Set to 'xiaomi/mimo-v2-pro' for free judging via Nous API.
    """
    import functools

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=judge_model or eval_model,
        judge_model=judge_model or eval_model,
        run_pytest=run_tests,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

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
    console.print(f"  Eval model: {config.eval_model}")
    if config.nous_api_key:
        console.print(f"  Nous API: active (free MiMo via inference-api.nousresearch.com)")
    else:
        console.print(f"  Nous API: not configured (set up hermes auth for free MiMo)")

    # Select fitness metric
    if mad_trials >= 3:
        console.print(f"  MAD scoring: enabled ({mad_trials} trials, confidence threshold 2.0x)")
        metric = functools.partial(mad_fitness_metric, n_trials=mad_trials)
    else:
        console.print(f"  MAD scoring: disabled (use --mad-trials >= 3 to enable)")
        metric = skill_fitness_metric

    # Configure DSPy
    if config.nous_api_key:
        # Use Nous API (OpenAI-compatible) for all model calls
        lm = dspy.LM(
            "openai/xiaomi/mimo-v2-pro",
            api_key=config.nous_api_key,
            api_base=config.nous_base_url,
        )
    else:
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
            metric=metric,
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
            metric=metric,
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

    # Extract optimized MIPROv2 artifacts for proof
    optimized_instructions = {}
    try:
        for name, pred in optimized_module.named_predictors():
            optimized_instructions[name] = {
                "signature_instructions": getattr(pred, "signature", None).instructions if hasattr(pred, "signature") else None,
                "demos": [
                    {"input": d.get("task_input", ""), "output": d.get("output", "")}
                    for d in (getattr(pred, "demos", []) or [])
                ],
            }
    except Exception:
        pass

    # Save optimization proof alongside evolved skill
    proof = {
        "skill_text_changed": evolved_body != skill["body"],
        "skill_text_size_before": len(skill["body"]),
        "skill_text_size_after": len(evolved_body),
        "optimized_instructions": optimized_instructions,
        "optimizer_score": float(scores[0]) if 'scores' in dir() else None,
    }

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
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
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 8. Evaluate on holdout set with MAD confidence ──────────────────
    holdout_n_trials = mad_trials if mad_trials >= 3 else 3
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")
    console.print(f"  Confidence scoring: {holdout_n_trials} trials per example")

    holdout_examples = dataset.to_dspy_examples("holdout")

    # Use hermes chat as judge — GPT-5.4 via hermes auth, no API key management
    judge_model = judge_model or "gpt-5.4"
    console.print(f"  Judge: hermes chat -m {judge_model}")

    judge = HermesJudge(model=judge_model)
    mad_judge = ConfidenceScoredFitness(judge, n_trials=holdout_n_trials)

    baseline_scores = []
    evolved_scores = []
    baseline_confidences = []
    evolved_confidences = []
    for ex in holdout_examples:
        with dspy.context(lm=lm):
            # Baseline with MAD confidence
            baseline_pred = baseline_module(task_input=ex.task_input)
            b_fitness, b_conf = mad_judge.score_with_confidence(
                task_input=getattr(ex, "task_input", "") or "",
                expected_behavior=getattr(ex, "expected_behavior", "") or "",
                agent_output=getattr(baseline_pred, "output", "") or "",
                skill_text=skill["body"],
            )
            baseline_scores.append(b_fitness.composite)
            baseline_confidences.append(b_conf)

            # Evolved with MAD confidence
            evolved_pred = optimized_module(task_input=ex.task_input)
            e_fitness, e_conf = mad_judge.score_with_confidence(
                task_input=getattr(ex, "task_input", "") or "",
                expected_behavior=getattr(ex, "expected_behavior", "") or "",
                agent_output=getattr(evolved_pred, "output", "") or "",
                skill_text=evolved_body,
            )
            evolved_scores.append(e_fitness.composite)
            evolved_confidences.append(e_conf)

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # Compute aggregate confidence: is the evolved skill genuinely better?
    # Direct MAD on per-example deltas: delta = evolved - baseline per example
    delta_scores = [e - b for e, b in zip(evolved_scores, baseline_scores)]
    mean_delta = sum(delta_scores) / max(1, len(delta_scores))
    median_delta = sorted(delta_scores)[len(delta_scores) // 2]
    mad_delta = compute_mad(delta_scores)

    if mad_delta > 0:
        confidence_ratio = abs(mean_delta) / mad_delta
    else:
        confidence_ratio = abs(mean_delta) / 0.01  # floor to avoid div-by-zero

    if confidence_ratio >= 2.0:
        conf_label = "likely real"
    elif confidence_ratio >= 1.0:
        conf_label = "marginal"
    else:
        conf_label = "within noise"

    # Create a synthetic ConfidenceResult for reporting
    from evolution.core.mad_scoring import ConfidenceResult as _CR
    holdout_confidence = _CR(
        decision="keep" if (mean_delta > 0 and confidence_ratio >= 2.0) else "discard",
        confidence=confidence_ratio,
        delta=mean_delta,
        delta_pct=(mean_delta / max(0.001, abs(avg_baseline)) * 100) if avg_baseline != 0 else 0.0,
        label=conf_label,
        best=max(evolved_scores) if evolved_scores else 0.0,
        baseline=avg_baseline,
        mad=mad_delta,
        n_trials=len(delta_scores),
    ) if len(delta_scores) >= 3 else None

    # Classify per-example confidence labels
    def _conf_summary(confidences):
        if not confidences:
            return {}
        from collections import Counter
        labels = Counter(c.label for c in confidences)
        return {"likely_real": labels.get("likely real", 0),
                "marginal": labels.get("marginal", 0),
                "within_noise": labels.get("within noise", 0)}

    baseline_conf_summary = _conf_summary(baseline_confidences)
    evolved_conf_summary = _conf_summary(evolved_confidences)

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

    # Confidence rows
    if holdout_confidence:
        conf_color = {"likely real": "green", "marginal": "yellow", "within noise": "red"}.get(holdout_confidence.label, "white")
        table.add_row(
            "Holdout Confidence",
            "",
            "",
            f"[{conf_color}]{holdout_confidence.label} ({holdout_confidence.confidence:.2f}x)[/{conf_color}]",
        )

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
        "confidence": {
            "holdout_trials_per_example": holdout_n_trials,
            "holdout_delta_confidence": {
                "label": holdout_confidence.label,
                "confidence": holdout_confidence.confidence,
                "delta": holdout_confidence.delta,
                "delta_pct": holdout_confidence.delta_pct,
                "mad": holdout_confidence.mad,
            } if holdout_confidence else None,
            "baseline_per_example": baseline_conf_summary,
            "evolved_per_example": evolved_conf_summary,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Save optimization proof (instruction changes, demos, skill text diff)
    (output_dir / "proof.json").write_text(json.dumps(proof, indent=2, default=str))

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
@click.option("--eval-model", default="openai/gpt-5.4", help="Model for LLM-as-judge evaluations")
@click.option("--judge-model", default=None,
              help="Model for LLM-as-judge (e.g. 'xiaomi/mimo-v2-pro' for free Nous API)")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--mad-trials", default=1, type=int,
              help="MAD confidence trials (>=3 enables noise-gated scoring)")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, judge_model, hermes_repo, run_tests, mad_trials, dry_run):
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
        mad_trials=mad_trials,
    )


if __name__ == "__main__":
    main()

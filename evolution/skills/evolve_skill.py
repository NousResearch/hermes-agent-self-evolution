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

from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.fitness import (
    skill_fitness_metric,
    make_gepa_metric,
    make_llm_judge_metric,
    LLMJudge,
    FitnessScore,
)
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


def build_gepa_optimizer(metric, iterations: int, reflection_lm=None):
    """Build a GEPA optimizer using the budget knob the installed DSPy exposes.

    DSPy 3.x renamed the iteration budget: `max_steps` no longer exists and
    passing it raises `GEPA.__init__() got an unexpected keyword argument`.
    The current knobs are `max_metric_calls` (budget in metric invocations,
    the unit that actually bounds cost) and `max_full_evals`.

    `iterations` is translated into full-eval equivalents so the CLI flag keeps
    its documented meaning while the enforced unit becomes metric calls.
    """
    import inspect

    # Inspect the CLASS, not `__init__` of an instance: budget knobs such as
    # max_metric_calls are keyword-only args declared on __init__, so reading
    # a bound/unbound method can miss them depending on how it is wrapped.
    params = set(inspect.signature(dspy.GEPA).parameters) | set(
        inspect.signature(dspy.GEPA.__init__).parameters
    )
    kwargs = {"metric": metric}

    if reflection_lm is not None:
        kwargs["reflection_lm"] = reflection_lm

    if "max_metric_calls" in params:
        # 1 iteration ~ a few metric calls; use the call budget directly.
        kwargs["max_metric_calls"] = max(1, iterations) * 4
    elif "max_full_evals" in params:
        kwargs["max_full_evals"] = max(1, iterations)
    elif "max_steps" in params:  # pragma: no cover - legacy DSPy
        kwargs["max_steps"] = max(1, iterations)
    else:  # pragma: no cover - unknown DSPy
        raise TypeError(
            "dspy.GEPA exposes no budget knob (max_metric_calls / max_full_evals / "
            "max_steps); refusing to start an unbounded optimization"
        )

    return dspy.GEPA(**kwargs)


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
    use_llm_judge: bool = True,
    min_improvement: float = 0.10,  # 10% relative improvement required (PLAN.md)
    max_tokens: int = 20000,  # headroom for reasoning models' hidden tokens
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
    baseline_constraints = validator.validate_all(skill["body"], "skill", full_text=skill["raw"])
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

    # Configure DSPy.
    # max_tokens is not optional for reasoning models: they spend most of the
    # budget on hidden `reasoning` tokens, so a small cap truncates the reply
    # at finish_reason=length and the structured output comes back empty, which
    # DSPy reports as "empty or null response" and the optimizer silently skips
    # the candidate. Default 8000 leaves room for reasoning plus the payload.
    lm = dspy.LM(eval_model, max_tokens=max_tokens)
    dspy.configure(lm=lm)

    # Fitness signal. The legacy default is a keyword-overlap heuristic that
    # scores word frequency, not behaviour — optimising it produces skills that
    # merely echo the rubric. LLM-as-judge scores correctness /
    # procedure_following / conciseness and returns feedback GEPA can reflect on.
    if use_llm_judge:
        base_metric = make_llm_judge_metric(config, judge_lm=lm)
        console.print("  Fitness: LLM-as-judge (correctness/procedure/conciseness)")
    else:
        base_metric = skill_fitness_metric
        console.print("  Fitness: keyword-overlap heuristic (legacy)")

    # DSPy 3.x calls the metric with five positional args.
    metric = make_gepa_metric(base_metric)

    # Create the baseline skill module (skill text lives in its instructions)
    baseline_module = SkillModule(skill["body"])

    # Prepare DSPy examples
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run GEPA optimization ────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()

    optimizer = build_gepa_optimizer(metric, iterations, reflection_lm=lm)

    optimized_module = optimizer.compile(
        baseline_module,
        trainset=trainset,
        valset=valset,
    )

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Extract evolved skill text ───────────────────────────────────
    # skill_text is now a property reading live instructions — no stale attr.
    evolved_body = optimized_module.skill_text
    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

    # ── 7. Validate evolved skill ───────────────────────────────────────
    console.print(f"\n[bold]Validating evolved skill[/bold]")
    # Use validate_skill: it reassembles frontmatter+body internally so the
    # structure check runs against the full artifact (not bare body).
    evolved_constraints = validator.validate_skill(
        evolved_body, frontmatter=skill["frontmatter"], baseline_body=skill["body"]
    )
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved skill FAILED constraints — not deploying[/red]")
        # Still save for inspection with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("output") / skill_name / f"evolved_FAILED_{timestamp}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_scores = []
    evolved_scores = []
    skipped = 0
    for ex in holdout_examples:
        # A provider hiccup on ONE example must not discard an optimization that
        # already ran: an agent asked to "execute the change" with no tools can
        # return an empty response, and that would otherwise take down the whole
        # run after the expensive part is already paid for.
        try:
            with dspy.context(lm=lm):
                baseline_pred = baseline_module(task_input=ex.task_input)
                baseline_score = skill_fitness_metric(ex, baseline_pred)
                baseline_scores.append(baseline_score)

                evolved_pred = optimized_module(task_input=ex.task_input)
                evolved_score = skill_fitness_metric(ex, evolved_pred)
                evolved_scores.append(evolved_score)
        except Exception as e:  # noqa: BLE001 - transient provider/adapter failure
            skipped += 1
            console.print(
                f"  [yellow]⚠ holdout example skipped ({type(e).__name__}: "
                f"{str(e)[:100]})[/yellow]"
            )

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # Report N/A rather than a score computed from a partial holdout: a delta
    # over 2 of 5 examples is not the same claim as one over 5.
    holdout_complete = skipped == 0 and len(evolved_scores) == len(holdout_examples)

    # Statistical significance: require min_improvement (relative) over baseline
    rel_improvement = improvement / max(0.001, avg_baseline)
    significant = holdout_complete and rel_improvement >= min_improvement

    # ── 9. Report results ───────────────────────────────────────────────
    table = Table(title="Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if significant else "red"
    table.add_row(
        "Holdout Score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    if skipped:
        table.add_row(
            "Holdout Coverage",
            f"{len(holdout_examples)} examples",
            f"{len(evolved_scores)} scored, {skipped} skipped",
            "[yellow]incomplete[/yellow]",
        )
    table.add_row(
        "Relative Δ",
        "—",
        f"{rel_improvement:+.1%}",
        f"[{'green' if significant else 'red'}]{'≥ threshold' if significant else 'below threshold'}[/]",
    )
    table.add_row(
        "Skill Size",
        f"{len(skill['body']):,} chars",
        f"{len(optimized_module.skill_text):,} chars",
        f"{len(optimized_module.skill_text) - len(skill['body']):+,} chars",
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
        "rel_improvement": rel_improvement,
        "significant": significant,
        "min_improvement_threshold": min_improvement,
        "baseline_size": len(skill["body"]),
        "evolved_size": len(optimized_module.skill_text),
        "train_examples": len(dataset.train),
        "val_examples": len(dataset.val),
        "holdout_examples": len(dataset.holdout),
        "holdout_scored": len(evolved_scores),
        "holdout_skipped": skipped,
        "holdout_complete": holdout_complete,
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "fitness": "llm_as_judge" if use_llm_judge else "keyword_overlap_heuristic",
        "optimizer_backend": "GEPA",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if not holdout_complete:
        console.print(
            f"\n[yellow]⚠ Holdout incomplete — {len(evolved_scores)}/{len(holdout_examples)} "
            f"examples scored, {skipped} skipped by provider/adapter errors. "
            f"No win is claimed.[/yellow]"
        )
    elif significant:
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} ({rel_improvement:+.1%} ≥ {min_improvement:.0%})[/bold green]")
        console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not meet significance threshold ({rel_improvement:+.1%} < {min_improvement:.0%})[/yellow]")
        console.print("  Try: more iterations, better eval dataset, or different optimizer model")


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
@click.option("--legacy-metric", is_flag=True,
              help="Use the keyword-overlap fitness instead of LLM-as-judge")
@click.option("--min-improvement", default=0.10, type=float,
              help="Minimum relative improvement (default 10%, per PLAN.md)")
@click.option("--max-tokens", default=20000, type=int,
              help="Completion token cap (default 20000; reasoning models need headroom)")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests, dry_run, legacy_metric, min_improvement, max_tokens):
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
        use_llm_judge=not legacy_metric,
        min_improvement=min_improvement,
        max_tokens=max_tokens,
    )


if __name__ == "__main__":
    main()
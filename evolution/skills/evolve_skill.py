"""Evolve a Hermes Agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/

Provider selection:
    The --provider flag chooses the LLM backend. Three values are supported:

    - "openai" (default): use OPENAI_API_KEY / OPENAI_BASE_URL env vars, model
      names prefixed with "openai/". Works against any OpenAI-compatible
      endpoint.
    - "minimax": use MINIMAX_API_KEY from env (falls back to OPENAI_API_KEY),
      set the base URL to https://api.minimax.io/v1, and pass through
      MiniMax-flavored model names like "MiniMax-M2.7-highspeed". This is the
      provider wired into the project owner's setup.
    - Anything else: used verbatim as the dspy.LM model string (e.g.
      "anthropic/claude-sonnet-4"), assuming the user has configured the
      relevant litellm provider keys.

    The MiniMax branch exists because MiniMax's v1 endpoint is
    OpenAI-compatible but ships only its own model names, not generic ones —
    so we can't just rename MINIMAX_API_KEY to OPENAI_API_KEY and call it a
    day. We also have to translate model strings.
"""

import json
import os
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
from evolution.core.fitness import skill_fitness_metric, LLMJudge, FitnessScore
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)

console = Console()


# Mapping from generic "openai/<name>" strings (as used in
# EvolutionConfig defaults) to MiniMax's actual model IDs. We need this
# because the v1 endpoint only accepts MiniMax's own model names — passing
# "openai/gpt-4.1" returns a 404 even though the wire format is OpenAI-
# compatible.
_OPENAI_TO_MINIMAX_MODEL = {
    "openai/gpt-4.1": "openai/MiniMax-M3",
    "openai/gpt-4.1-mini": "openai/MiniMax-M2.7-highspeed",
    "openai/gpt-4.1-nano": "openai/MiniMax-M2.5",
}


def _resolve_model_string(model: str, provider: str) -> str:
    """Translate an EvolutionConfig-style model string for the chosen provider.

    For provider="minimax", rewrite the OpenAI default names to MiniMax
    equivalents. Pass through anything else verbatim so users can still
    pin an exact MiniMax model if they want (e.g. "MiniMax-M3" or
    "MiniMax-Text-01" — but note the latter is only chat, not reasoning).
    """
    if provider != "minimax":
        return model
    if model in _OPENAI_TO_MINIMAX_MODEL:
        return _OPENAI_TO_MINIMAX_MODEL[model]
    # If the caller already wrote "openai/MiniMax-M2.7" or "MiniMax-M2.7",
    # pass through — we just need to make sure the base URL is set.
    return model


def _configure_provider_env(provider: str) -> None:
    """Set OPENAI_API_KEY / OPENAI_BASE_URL for the chosen provider.

    DSPy + litellm read these env vars when constructing the LM client.
    Setting them here (not in the caller) keeps the evolver's surface
    simple — callers just pick --provider and the env wiring follows.

    For MiniMax we read MINIMAX_API_KEY and override the base URL to the
    v1 endpoint (which is OpenAI-compatible; the anthropic endpoint at
    /anthropic is NOT compatible with dspy.LM's OpenAI-format adapter).
    """
    if provider == "minimax":
        key = os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            console.print(
                "[red]✗ provider=minimax requires MINIMAX_API_KEY (or "
                "OPENAI_API_KEY) in env[/red]"
            )
            sys.exit(2)
        os.environ["OPENAI_API_KEY"] = key
        # /v1 is the OpenAI-compatible surface; /anthropic is anthropic-format.
        os.environ["OPENAI_BASE_URL"] = os.getenv(
            "MINIMAX_BASE_URL", "https://api.minimax.io/v1"
        )
    # For provider="openai" (default) we assume the caller already set
    # OPENAI_API_KEY / OPENAI_BASE_URL — no override needed.
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            console.print(
                "[red]✗ provider=openai requires OPENAI_API_KEY in env[/red]"
            )
            sys.exit(2)
    # Custom string providers: trust the caller.


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
    provider: str = "openai",
):
    """Main evolution function — orchestrates the full optimization loop.

    The ``provider`` argument selects the LLM backend. See module docstring
    for the three supported values. We translate the EvolutionConfig's
    OpenAI-flavored default model strings into provider-appropriate names
    here, so callers can keep using the natural ``--optimizer-model`` /
    ``--eval-model`` flags without having to know which provider is wired
    up.
    """

    # Configure provider env FIRST, before constructing EvolutionConfig, so
    # any LM objects created downstream pick up the right base URL / key.
    _configure_provider_env(provider)

    # Translate model strings for the chosen provider (e.g. gpt-4.1 →
    # MiniMax-M3 when provider=minimax). We resolve against the user's
    # explicit --optimizer-model / --eval-model args first so they can
    # still pin a specific MiniMax model.
    resolved_optimizer = _resolve_model_string(optimizer_model, provider)
    resolved_eval = _resolve_model_string(eval_model, provider)

    if resolved_optimizer != optimizer_model or resolved_eval != eval_model:
        console.print(
            f"  [dim]Provider '{provider}': rewrote model strings[/dim]\n"
            f"    optimizer: {optimizer_model} → {resolved_optimizer}\n"
            f"    eval:      {eval_model} → {resolved_eval}"
        )

    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo),
        iterations=iterations,
        optimizer_model=resolved_optimizer,
        eval_model=resolved_eval,
        judge_model=resolved_eval,  # Use same model for dataset generation
        run_pytest=run_tests,
    )

    # ── 1. Find and load the skill ──────────────────────────────────────
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")
    console.print(f"  Provider: {provider} (base URL: {os.getenv('OPENAI_BASE_URL', '<default>')})")

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
    # Validate the full raw skill (frontmatter + body), not just the body.
    # _check_skill_structure looks for the YAML frontmatter at the top of
    # the text; passing only the body always fails the frontmatter check
    # even when the baseline itself is structurally correct.
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
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {config.optimizer_model}")
    console.print(f"  Eval model: {config.eval_model}")

    # Configure DSPy with the eval model as the global default. Any
    # unconfigured LM downstream (e.g. MIPROv2's data-aware proposer) will
    # pick this up rather than falling back to a hardcoded "gpt-4.1-mini"
    # that doesn't exist on MiniMax.
    lm = dspy.LM(config.eval_model)
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
        # Pass a reflection_lm to GEPA so it uses our chosen model for
        # the reflective mutation prompts. Without this, GEPA falls back
        # to dspy.settings.lm (already configured) but also defaults to
        # "gpt-4.1-mini" for some internal calls — which 404s on MiniMax.
        # Constructing an explicit reflection_lm makes the wiring obvious
        # and provider-portable.
        #
        # GEPA's budget arg is `max_full_evals` (not `max_steps` — that was
        # the older GEPA API). Each "full eval" runs the metric across the
        # whole valset, which gives a real budget ≈ `iterations * valset_size`
        # LM calls. We also cap with `max_metric_calls` as a hard ceiling
        # so the run can't run away even if valset is large.
        reflection_lm = dspy.LM(config.optimizer_model)
        # GEPA accepts exactly one of {max_full_evals, max_metric_calls, auto}.
        # Pass max_full_evals so a tiny iterations budget (e.g. 3) doesn't
        # blow up into thousands of metric calls. Drop max_metric_calls
        # entirely — it causes GEPA to refuse to start with a config error.
        optimizer = dspy.GEPA(
            metric=skill_fitness_metric,
            max_full_evals=iterations,
            reflection_lm=reflection_lm,
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
    console.print(f"\n[bold]Validating evolved skill[/bold]")
    # Use the reassembled full skill (frontmatter + body) for validation —
    # _check_skill_structure looks for the YAML frontmatter block at the
    # top of the text, and that lives in frontmatter, not body. Without
    # this the check always reports the frontmatter as "missing" and
    # gates every deploy, even when the artifact is structurally fine.
    evolved_constraints = validator.validate_all(
        evolved_full, "skill", baseline_text=skill["raw"]
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
        # Still save for inspection
        output_path = Path("output") / skill_name / "evolved_FAILED.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_full)
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 8. Evaluate on holdout set ──────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")

    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_scores = []
    evolved_scores = []
    for ex in holdout_examples:
        # Score baseline
        with dspy.context(lm=lm):
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
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

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
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run full pytest suite as constraint gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--provider", default="openai", type=click.Choice(["openai", "minimax"], case_sensitive=False),
              help="LLM backend. 'openai' uses OPENAI_API_KEY/OPENAI_BASE_URL env vars; "
                   "'minimax' uses MINIMAX_API_KEY and rewrites model strings to MiniMax equivalents.")
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo, run_tests, dry_run, provider):
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
        provider=provider,
    )


if __name__ == "__main__":
    main()

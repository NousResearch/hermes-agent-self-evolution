"""Evolve the Hermes Agent persona using LLM-guided text mutation.

Unlike skills (where MIPROv2 optimizes instructions), persona is a text artifact
that must be directly mutated and evaluated. We use a genetic algorithm approach:
1. Generate N candidate variations of the persona
2. Score each against evaluation scenarios
3. Keep the best, optionally iterate

Usage:
    python -m evolution.artifacts.persona_evolver --iterations 3
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import SyntheticDatasetBuilder
from evolution.core.constraints import ConstraintValidator

console = Console()

PERSONA_PATH = Path.home() / ".hermes" / "artifacts" / "persona" / "persona.md"

def load_persona() -> str:
    if PERSONA_PATH.exists():
        return PERSONA_PATH.read_text()
    console.print(f"[red]✗ Persona not found at {PERSONA_PATH}[/red]")
    sys.exit(1)


def save_persona(text: str):
    PERSONA_PATH.write_text(text)


class PersonaVariationGenerator(dspy.Signature):
    """Generate improved variations of a persona text.

    Given a current persona and feedback on what works/doesn't work,
    produce 3 improved variations. Each variation should preserve the
    core identity while strengthening weak areas.
    """

    current_persona = dspy.InputField(desc="The current persona text")
    evaluation_scenarios = dspy.InputField(desc="Scenarios where the persona performed well or poorly")
    weaknesses = dspy.InputField(desc="Identified weaknesses to address")
    variation_1 = dspy.OutputField(desc="First improved persona variation (full text)")
    variation_2 = dspy.OutputField(desc="Second improved persona variation (full text)")
    variation_3 = dspy.OutputField(desc="Third improved persona variation (full text)")
    changes_summary = dspy.OutputField(desc="Brief summary of key changes made")


class PersonaEvaluator(dspy.Signature):
    """Evaluate how well a persona guides an agent's response to a scenario.

    Score 0-10 on: directness, conciseness, correctness, proactive_action, tone_match.
    """

    persona = dspy.InputField(desc="The persona text")
    scenario = dspy.InputField(desc="The scenario to evaluate")
    expected_behavior = dspy.InputField(desc="The expected ideal behavior")
    score = dspy.OutputField(desc="Score 0-10")
    reasoning = dspy.OutputField(desc="Brief reasoning")


def score_persona(persona_text: str, scenario: str, expected: str, lm: dspy.LM) -> float:
    """Score a persona variant against a single scenario."""
    with dspy.context(lm=lm):
        evaluator = dspy.Predict(PersonaEvaluator)
        result = evaluator(persona=persona_text, scenario=scenario, expected_behavior=expected)
    try:
        return float(result.score)
    except (ValueError, TypeError):
        return 5.0


def evaluate_persona(persona_text: str, examples: list, lm: dspy.LM) -> float:
    """Average score across all evaluation examples."""
    scores = []
    for ex in examples:
        # Handle different field names in dataset
        scenario = ex.task_input if hasattr(ex, "task_input") else ex.get("task_input", "")
        expected = ex.expected_output if hasattr(ex, "expected_output") else \
                   ex.expected_behavior if hasattr(ex, "expected_behavior") else \
                   ex.get("expected_behavior", ex.get("expected_output", ""))
        s = score_persona(persona_text, scenario, expected, lm)
        scores.append(s)
    return sum(scores) / max(1, len(scores))


def generate_variations(current_persona: str, examples: list, lm: dspy.LM) -> List[str]:
    """Generate 3 improved variations of the persona."""
    # Build weaknesses summary from lowest-scoring examples
    scores = []
    for ex in examples:
        scenario = ex.task_input if hasattr(ex, "task_input") else ex.get("task_input", "")
        expected = ex.expected_output if hasattr(ex, "expected_output") else \
                   ex.expected_behavior if hasattr(ex, "expected_behavior") else \
                   ex.get("expected_behavior", ex.get("expected_output", ""))
        s = score_persona(current_persona, scenario, expected, lm)
        scores.append((s, scenario, expected))

    # Sort by score, lowest first
    scores.sort(key=lambda x: x[0])
    weak_scenarios = scores[:3]
    strong_scenarios = scores[-2:]

    eval_text = "WEAK PERFORMANCES (need improvement):\n"
    for s, sc, exp in weak_scenarios:
        eval_text += f"- Score {s}/10 | Scenario: {sc[:120]}... | Expected: {exp[:120]}...\n"

    eval_text += "\nSTRONG PERFORMANCES (preserve these):\n"
    for s, sc, exp in strong_scenarios:
        eval_text += f"- Score {s}/10 | Scenario: {sc[:120]}... | Expected: {exp[:120]}...\n"

    weaknesses = "; ".join([f"{sc[:60]}... (scored {s})" for s, sc, _ in weak_scenarios])

    with dspy.context(lm=lm):
        generator = dspy.Predict(PersonaVariationGenerator)
        result = generator(
            current_persona=current_persona,
            evaluation_scenarios=eval_text,
            weaknesses=weaknesses,
        )

    variations = []
    for v in [result.variation_1, result.variation_2, result.variation_3]:
        if v and len(v) > 200:
            variations.append(v)

    # Always include current as baseline
    variations.append(current_persona)
    return variations


def evolve_persona(
    iterations: int = 120,
    optimizer_model: str = "openai/glm-5.1",
    eval_model: str = "openai/glm-5.1",
    dry_run: bool = False,
):
    """Main persona evolution function using text mutation."""

    console.print(f"\n[bold cyan]🧬 Persona Self-Evolution[/bold cyan] — Evolving agent identity\n")

    persona_text = load_persona()
    console.print(f"  Loaded: {PERSONA_PATH}")
    console.print(f"  Size: {len(persona_text):,} chars")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated.[/bold green]")
        return

    # ── 1. Build evaluation dataset ──────────────────────────────────────
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: synthetic)")

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
    )
    builder = SyntheticDatasetBuilder(config)
    dataset = builder.generate(
        artifact_text=persona_text,
        artifact_type="persona",
    )
    save_path = Path("datasets") / "artifacts" / "persona"
    dataset.save(save_path)
    console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 2. Configure DSPy ────────────────────────────────────────────────
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    lm_kwargs = {}
    if api_base:
        lm_kwargs["api_base"] = api_base
    if api_key:
        lm_kwargs["api_key"] = api_key
    lm = dspy.LM(eval_model, **lm_kwargs)
    dspy.configure(lm=lm)

    # ── 3. Baseline evaluation ───────────────────────────────────────────
    console.print(f"\n[bold]Evaluating baseline persona[/bold]")
    val_examples = dataset.to_dspy_examples("val")
    baseline_score = evaluate_persona(persona_text, val_examples, lm)
    console.print(f"  Baseline score: {baseline_score:.2f}/10")

    # ── 4. Evolution loop ────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Running evolution ({iterations} iterations)...[/bold cyan]\n")

    start_time = time.time()
    best_persona = persona_text
    best_score = baseline_score
    history = [("baseline", baseline_score)]

    for i in range(iterations):
        console.print(f"  Iteration {i+1}/{iterations}")
        variations = generate_variations(best_persona, val_examples, lm)

        best_var_score = best_score
        best_var_persona = best_persona

        for j, var in enumerate(variations[:-1]):  # Exclude baseline (last)
            score = evaluate_persona(var, val_examples, lm)
            console.print(f"    Variant {j+1}: {score:.2f}/10")
            if score > best_var_score:
                best_var_score = score
                best_var_persona = var

        if best_var_score > best_score:
            improvement = best_var_score - best_score
            console.print(f"  [green]→ Improved by +{improvement:.2f}[/green]")
            best_score = best_var_score
            best_persona = best_var_persona
            history.append((f"iter_{i+1}", best_score))
        else:
            console.print(f"  [yellow]→ No improvement[/yellow]")
            history.append((f"iter_{i+1}", best_score))

    elapsed = time.time() - start_time
    console.print(f"\n  Evolution completed in {elapsed:.1f}s")

    # ── 5. Validate constraints ──────────────────────────────────────────
    console.print(f"\n[bold]Validating constraints[/bold]")
    validator = ConstraintValidator(config)
    constraints = validator.validate_all(best_persona, "skill", baseline_text=persona_text)

    # For persona, skill_structure constraint may fail (no YAML frontmatter) — that's OK
    all_pass = True
    for c in constraints:
        if c.constraint_name == "skill_structure":
            # Persona doesn't need YAML frontmatter
            console.print(f"  [yellow]⚑ {c.constraint_name}[/yellow]: {c.message} (skipped for persona)")
            continue
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False

    # ── 6. Holdout evaluation ────────────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")
    holdout_examples = dataset.to_dspy_examples("holdout")
    holdout_baseline = evaluate_persona(persona_text, holdout_examples, lm)
    holdout_evolved = evaluate_persona(best_persona, holdout_examples, lm)
    improvement = holdout_evolved - holdout_baseline

    # ── 7. Report ────────────────────────────────────────────────────────
    table = Table(title="Persona Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Holdout Score",
        f"{holdout_baseline:.2f}",
        f"{holdout_evolved:.2f}",
        f"[{change_color}]{improvement:+.2f}[/{change_color}]",
    )
    table.add_row(
        "Persona Size",
        f"{len(persona_text):,} chars",
        f"{len(best_persona):,} chars",
        f"{len(best_persona) - len(persona_text):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")

    console.print()
    console.print(table)

    # ── 8. Save output ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "persona" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "evolved_persona.md").write_text(best_persona)
    (output_dir / "baseline_persona.md").write_text(persona_text)

    metrics = {
        "artifact_type": "persona",
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": holdout_baseline,
        "evolved_score": holdout_evolved,
        "improvement": improvement,
        "baseline_size": len(persona_text),
        "evolved_size": len(best_persona),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    # ── 9. Deploy if improved ────────────────────────────────────────────
    if improvement > 0:
        console.print(f"\n[bold green]✓ Persona improved by {improvement:+.2f} on holdout[/bold green]")
        save_persona(best_persona)
        console.print(f"[bold green]✓ Deployed to {PERSONA_PATH}[/bold green]")
    elif holdout_evolved >= holdout_baseline:
        console.print(f"\n[yellow]⚠ No holdout improvement ({improvement:+.2f}). Keeping current.[/yellow]")
    else:
        console.print(f"\n[red]✗ Holdout degraded ({improvement:+.2f}). Keeping baseline.[/red]")


@click.command()
@click.option("--iterations", default=120, help="Number of evolution iterations")
@click.option("--optimizer-model", default="openai/glm-5.1", help="Model for generating variations")
@click.option("--eval-model", default="openai/glm-5.1", help="Model for evaluation")
@click.option("--dry-run", is_flag=True, help="Validate setup without running")
def main(iterations, optimizer_model, eval_model, dry_run):
    """Evolve the Hermes Agent persona."""
    evolve_persona(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()

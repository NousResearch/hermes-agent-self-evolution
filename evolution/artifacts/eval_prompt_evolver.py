"""Evolve LLM judge / evaluator prompts using LLM-guided mutation.

Eval prompts are critical because they determine how we score everything else.
We evolve them by testing their discrimination power on a set of scenarios.

Usage:
    python -m evolution.artifacts.eval_prompt_evolver --prompt-name persona_evaluator --iterations 3
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig

console = Console()

PROMPTS_DIR = Path.home() / ".hermes" / "artifacts" / "eval_prompts"

def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    if path.exists():
        return path.read_text()
    console.print(f"[red]✗ Prompt not found at {path}[/red]")
    sys.exit(1)


def save_prompt(name: str, text: str):
    path = PROMPTS_DIR / f"{name}.txt"
    path.write_text(text)


class EvalPromptVariationGenerator(dspy.Signature):
    """Generate improved variations of an evaluation prompt.

    Given a current evaluator prompt and examples where it performed
    well or poorly, produce 3 improved variations.
    """

    current_prompt = dspy.InputField(desc="The current evaluator prompt text")
    good_examples = dspy.InputField(desc="Scenarios where the prompt evaluated correctly")
    bad_examples = dspy.InputField(desc="Scenarios where the prompt evaluated incorrectly")
    variation_1 = dspy.OutputField(desc="First improved prompt variation (full text)")
    variation_2 = dspy.OutputField(desc="Second improved prompt variation (full text)")
    variation_3 = dspy.OutputField(desc="Third improved prompt variation (full text)")
    changes_summary = dspy.OutputField(desc="Brief summary of key changes")


class ScenarioGenerator(dspy.Signature):
    """Generate evaluation scenarios for testing an evaluator prompt."""

    prompt_purpose = dspy.InputField(desc="What this evaluator judges (e.g., persona quality)")
    num_scenarios = dspy.InputField(desc="Number of scenarios to generate")
    scenarios = dspy.OutputField(desc="JSON array of scenarios, each with: description, good_response, bad_response")


def generate_test_scenarios(purpose: str, n: int, lm: dspy.LM) -> List[dict]:
    """Generate scenarios with known good/bad responses."""
    with dspy.context(lm=lm):
        gen = dspy.Predict(ScenarioGenerator)
        result = gen(prompt_purpose=purpose, num_scenarios=n)
    import json, re
    from json_repair import repair_json
    raw = result.scenarios or "[]"
    try:
        return json.loads(repair_json(raw))
    except Exception:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(repair_json(match.group()))
            except Exception:
                return []
        return []


def score_with_prompt(prompt_text: str, scenario: dict, lm: dspy.LM) -> tuple:
    """Use the prompt to score good vs bad response. Returns (good_score, bad_score, discrimination)."""
    # Fill in the prompt template
    good_prompt = prompt_text.replace("{{scenario}}", scenario.get("description", ""))
    good_prompt = good_prompt.replace("{{expected}}", scenario.get("good_response", ""))
    good_prompt = good_prompt.replace("{{actual}}", scenario.get("good_response", ""))

    bad_prompt = prompt_text.replace("{{scenario}}", scenario.get("description", ""))
    bad_prompt = bad_prompt.replace("{{expected}}", scenario.get("good_response", ""))
    bad_prompt = bad_prompt.replace("{{actual}}", scenario.get("bad_response", ""))

    with dspy.context(lm=lm):
        good_result = lm(good_prompt)
        bad_result = lm(bad_prompt)

    # Extract scores
    import re
    good_score = 5.0
    bad_score = 5.0
    if good_result:
        m = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', str(good_result[0]) if isinstance(good_result, list) else str(good_result))
        if m:
            good_score = float(m.group(1))
    if bad_result:
        m = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', str(bad_result[0]) if isinstance(bad_result, list) else str(bad_result))
        if m:
            bad_score = float(m.group(1))

    discrimination = good_score - bad_score
    return good_score, bad_score, discrimination


def evaluate_prompt(prompt_text: str, scenarios: List[dict], lm: dspy.LM) -> float:
    """Average discrimination power across scenarios."""
    discriminations = []
    for sc in scenarios:
        _, _, disc = score_with_prompt(prompt_text, sc, lm)
        discriminations.append(disc)
    return sum(discriminations) / max(1, len(discriminations))


def generate_variations(current_prompt: str, scenarios: List[dict], lm: dspy.LM) -> List[str]:
    """Generate improved variations of the eval prompt."""
    # Score current on all scenarios
    results = []
    for sc in scenarios:
        g, b, d = score_with_prompt(current_prompt, sc, lm)
        results.append((d, sc))

    results.sort(key=lambda x: x[0])
    bad_examples = results[:3]
    good_examples = results[-2:]

    bad_text = "\n".join([f"- Discrimination {d:.1f} on: {sc.get('description', '')[:100]}" for d, sc in bad_examples])
    good_text = "\n".join([f"- Discrimination {d:.1f} on: {sc.get('description', '')[:100]}" for d, sc in good_examples])

    with dspy.context(lm=lm):
        gen = dspy.Predict(EvalPromptVariationGenerator)
        result = gen(
            current_prompt=current_prompt,
            good_examples=good_text,
            bad_examples=bad_text,
        )

    variations = []
    for v in [result.variation_1, result.variation_2, result.variation_3]:
        if v and len(v) > 100:
            variations.append(v)
    variations.append(current_prompt)
    return variations


def evolve_eval_prompt(
    prompt_name: str = "persona_evaluator",
    iterations: int = 120,
    num_scenarios: int = 8,
    optimizer_model: str = "openai/glm-5.1",
    eval_model: str = "openai/glm-5.1",
    dry_run: bool = False,
):
    """Evolve an evaluation prompt."""

    console.print(f"\n[bold cyan]🧬 Eval Prompt Self-Evolution[/bold cyan] — Evolving: {prompt_name}\n")

    prompt_text = load_prompt(prompt_name)
    console.print(f"  Loaded: {PROMPTS_DIR / (prompt_name + '.txt')}")
    console.print(f"  Size: {len(prompt_text):,} chars")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated.[/bold green]")
        return

    # Configure DSPy
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    lm_kwargs = {}
    if api_base:
        lm_kwargs["api_base"] = api_base
    if api_key:
        lm_kwargs["api_key"] = api_key
    lm = dspy.LM(eval_model, **lm_kwargs)
    dspy.configure(lm=lm)

    # Generate test scenarios
    console.print(f"\n[bold]Generating {num_scenarios} test scenarios[/bold]")
    scenarios = generate_test_scenarios(
        f"Evaluate agent responses based on {prompt_name}",
        num_scenarios,
        lm,
    )
    console.print(f"  Generated {len(scenarios)} scenarios")

    # Baseline
    console.print(f"\n[bold]Evaluating baseline prompt[/bold]")
    baseline_score = evaluate_prompt(prompt_text, scenarios, lm)
    console.print(f"  Baseline discrimination: {baseline_score:.2f}")

    # Evolution loop
    console.print(f"\n[bold cyan]Running evolution ({iterations} iterations)...[/bold cyan]\n")
    start_time = time.time()

    best_prompt = prompt_text
    best_score = baseline_score

    for i in range(iterations):
        console.print(f"  Iteration {i+1}/{iterations}")
        variations = generate_variations(best_prompt, scenarios, lm)

        best_var_score = best_score
        best_var_prompt = best_prompt

        for j, var in enumerate(variations[:-1]):
            score = evaluate_prompt(var, scenarios, lm)
            console.print(f"    Variant {j+1}: discrimination {score:.2f}")
            if score > best_var_score:
                best_var_score = score
                best_var_prompt = var

        if best_var_score > best_score:
            improvement = best_var_score - best_score
            console.print(f"  [green]→ Improved by +{improvement:.2f}[/green]")
            best_score = best_var_score
            best_prompt = best_var_prompt
        else:
            console.print(f"  [yellow]→ No improvement[/yellow]")

    elapsed = time.time() - start_time
    console.print(f"\n  Evolution completed in {elapsed:.1f}s")

    # Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "eval_prompts" / prompt_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "evolved_prompt.txt").write_text(best_prompt)
    (output_dir / "baseline_prompt.txt").write_text(prompt_text)

    metrics = {
        "artifact_type": "eval_prompt",
        "prompt_name": prompt_name,
        "timestamp": timestamp,
        "iterations": iterations,
        "baseline_score": baseline_score,
        "evolved_score": best_score,
        "improvement": best_score - baseline_score,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if best_score > baseline_score:
        console.print(f"\n[bold green]✓ Prompt improved by {best_score - baseline_score:+.2f}[/bold green]")
        save_prompt(prompt_name, best_prompt)
        console.print(f"[bold green]✓ Deployed to {PROMPTS_DIR / (prompt_name + '.txt')}[/bold green]")
    else:
        console.print(f"\n[yellow]⚠ No improvement. Keeping baseline.[/yellow]")


@click.command()
@click.option("--prompt-name", default="persona_evaluator", help="Name of the eval prompt to evolve")
@click.option("--iterations", default=120, help="Number of evolution iterations")
@click.option("--num-scenarios", default=8, help="Number of test scenarios")
@click.option("--optimizer-model", default="openai/glm-5.1", help="Model for generating variations")
@click.option("--eval-model", default="openai/glm-5.1", help="Model for evaluation")
@click.option("--dry-run", is_flag=True, help="Validate setup without running")
def main(prompt_name, iterations, num_scenarios, optimizer_model, eval_model, dry_run):
    """Evolve an evaluation prompt."""
    evolve_eval_prompt(
        prompt_name=prompt_name,
        iterations=iterations,
        num_scenarios=num_scenarios,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()

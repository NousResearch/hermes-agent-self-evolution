"""Evolve hermes-agent tool descriptions using DSPy + GEPA (Phase 2).

Unlike skill evolution, tool descriptions are optimized as a *group*: all
target tools' descriptions live in one router instructions block (see
tool_module.build_router_instructions) so GEPA sees cross-tool competition —
mutating one tool's description to steal selections from another shows up as
a fitness regression on the stolen-from tool, not just a gain on the mutated one.

Usage:
    python -m evolution.tools.evolve_tool_descriptions \\
        --tools read_file,write_file,search_files,terminal --iterations 5
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.dataset_builder import EvalExample, EvalDataset
from evolution.core.constraints import ConstraintValidator
from evolution.tools.tool_module import ToolRouterModule, load_tool_descriptions

console = Console()


class GenerateToolSelectionCases(dspy.Signature):
    """Generate a tool-selection evaluation dataset for a set of agent tools.

    Given several tool descriptions, produce realistic tasks that test whether
    an agent would pick the right tool. For EACH tool, include a mix of:
    - Clear-cut tasks where that tool is obviously correct
    - Confusing tasks where a DIFFERENT tool in the list could plausibly be
      picked instead, to test whether descriptions clearly disambiguate
    Do not include tasks for tools outside the given list.
    """
    tool_descriptions_block: str = dspy.InputField(desc="All candidate tools and their current descriptions")
    tool_names: str = dspy.InputField(desc="Comma-separated list of valid tool names — every correct_tool value must be one of these, verbatim")
    cases_per_tool: int = dspy.InputField(desc="How many test cases to generate per tool")
    test_cases: str = dspy.OutputField(desc="JSON array of objects: {task_input, correct_tool, difficulty}")


def _generate_tool_selection_dataset(
    descriptions: dict[str, str],
    config: EvolutionConfig,
    cases_per_tool: int = 6,
) -> EvalDataset:
    from evolution.tools.tool_module import build_router_instructions

    generator = dspy.ChainOfThought(GenerateToolSelectionCases)
    lm = dspy.LM(config.judge_model)

    with dspy.context(lm=lm):
        result = generator(
            tool_descriptions_block=build_router_instructions(descriptions),
            tool_names=", ".join(descriptions.keys()),
            cases_per_tool=cases_per_tool,
        )

    try:
        cases_raw = json.loads(result.test_cases)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\[.*\]", result.test_cases, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse test cases from LLM output: {result.test_cases[:200]}")
        cases_raw = json.loads(match.group())

    valid_names = set(descriptions.keys())
    examples = [
        EvalExample(
            task_input=c["task_input"],
            expected_behavior=c["correct_tool"],
            difficulty=c.get("difficulty", "medium"),
            category=c["correct_tool"],
            source="synthetic",
        )
        for c in cases_raw
        if c.get("task_input") and c.get("correct_tool") in valid_names
    ]

    dropped = len(cases_raw) - len(examples)
    if dropped:
        console.print(f"  [yellow]Dropped {dropped}/{len(cases_raw)} generated cases (missing fields or unknown tool name)[/yellow]")

    import random
    random.shuffle(examples)
    n = len(examples)
    n_train = max(1, int(n * config.train_ratio))
    n_val = max(1, int(n * config.val_ratio))

    return EvalDataset(
        train=examples[:n_train],
        val=examples[n_train:n_train + n_val],
        holdout=examples[n_train + n_val:],
    )


def tool_selection_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> dspy.Prediction:
    """GEPA-compatible metric: did the router pick the gold tool name?"""
    predicted = (getattr(prediction, "tool_name", "") or "").strip()
    gold = (getattr(example, "expected_behavior", "") or "").strip()

    if predicted == gold:
        return dspy.Prediction(score=1.0, feedback=f"Correctly routed to '{gold}'.")

    return dspy.Prediction(
        score=0.0,
        feedback=(
            f"Task: {example.task_input!r} — expected tool '{gold}' but the router "
            f"picked '{predicted}'. The descriptions for these two tools don't "
            f"clearly disambiguate this case; sharpen what distinguishes them."
        ),
    )


def evolve(
    tool_names: list[str],
    iterations: int = 5,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    hermes_repo: Optional[str] = None,
    cases_per_tool: int = 6,
    dry_run: bool = False,
):
    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo),
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
    )

    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — Evolving tool descriptions: [bold]{', '.join(tool_names)}[/bold]\n")

    descriptions = load_tool_descriptions(config.hermes_agent_path, tool_names)
    for name, desc in descriptions.items():
        console.print(f"  {name}: {len(desc):,} chars")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate ~{cases_per_tool * len(tool_names)} synthetic selection cases")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        return

    console.print("\n[bold]Generating tool-selection eval dataset[/bold]")
    dataset = _generate_tool_selection_dataset(descriptions, config, cases_per_tool)
    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    validator = ConstraintValidator(config)
    console.print("\n[bold]Baseline description sizes[/bold]")
    for name, desc in descriptions.items():
        result = validator._check_size(desc, "tool_description")
        icon = "✓" if result.passed else "✗"
        console.print(f"  [{'green' if result.passed else 'yellow'}]{icon} {name}[/]: {result.message}")

    dspy.configure(lm=dspy.LM(eval_model))

    baseline_module = ToolRouterModule(descriptions)
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")
    start_time = time.time()

    optimizer = dspy.GEPA(
        metric=tool_selection_metric,
        max_full_evals=iterations,
        reflection_lm=dspy.LM(optimizer_model),
    )
    optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    evolved_descriptions = optimized_module.descriptions
    missing = set(tool_names) - evolved_descriptions.keys()
    if missing:
        console.print(f"[red]✗ Lost marker for tool(s) {sorted(missing)} during optimization — structure did not survive[/red]")

    console.print("\n[bold]Evolved description sizes[/bold]")
    per_tool_ok = {}
    for name in tool_names:
        desc = evolved_descriptions.get(name, "")
        result = validator._check_size(desc, "tool_description") if desc else None
        ok = bool(desc) and result.passed
        per_tool_ok[name] = ok
        icon = "✓" if ok else "✗"
        msg = result.message if result else "missing after optimization"
        console.print(f"  [{'green' if ok else 'red'}]{icon} {name}[/]: {msg}")

    console.print(f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]")
    holdout_examples = dataset.to_dspy_examples("holdout")

    baseline_correct = evolved_correct = 0
    lm = dspy.LM(eval_model)
    with dspy.context(lm=lm):
        for ex in holdout_examples:
            baseline_correct += int(baseline_module(task_input=ex.task_input).tool_name.strip() == ex.expected_behavior)
            evolved_correct += int(optimized_module(task_input=ex.task_input).tool_name.strip() == ex.expected_behavior)

    n_holdout = max(1, len(holdout_examples))
    baseline_acc = baseline_correct / n_holdout
    evolved_acc = evolved_correct / n_holdout

    table = Table(title="Tool Description Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")
    change = evolved_acc - baseline_acc
    table.add_row(
        "Holdout tool-selection accuracy",
        f"{baseline_acc:.3f}",
        f"{evolved_acc:.3f}",
        f"[{'green' if change > 0 else 'red'}]{change:+.3f}[/]",
    )
    console.print()
    console.print(table)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "tools" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_descriptions.json").write_text(json.dumps(descriptions, indent=2))
    (output_dir / "evolved_descriptions.json").write_text(json.dumps(evolved_descriptions, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps({
        "tools": tool_names,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_accuracy": baseline_acc,
        "evolved_accuracy": evolved_acc,
        "improvement": change,
        "per_tool_constraints_passed": per_tool_ok,
        "elapsed_seconds": elapsed,
    }, indent=2))
    console.print(f"\n  Output saved to {output_dir}/")


@click.command()
@click.option("--tools", required=True, help="Comma-separated tool names to evolve together")
@click.option("--iterations", default=5, help="Number of GEPA iterations")
@click.option("--cases-per-tool", default=6, help="Synthetic eval cases to generate per tool")
@click.option("--optimizer-model", default="openai/gpt-4.1", help="Model for GEPA reflections")
@click.option("--eval-model", default="openai/gpt-4.1-mini", help="Model for evaluations")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
def main(tools, iterations, cases_per_tool, optimizer_model, eval_model, hermes_repo, dry_run):
    """Evolve a group of hermes-agent tool descriptions using DSPy + GEPA."""
    evolve(
        tool_names=[t.strip() for t in tools.split(",") if t.strip()],
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        cases_per_tool=cases_per_tool,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()

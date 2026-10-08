"""Evolve a set of tool descriptions using DSPy + GEPA.

Phase 2 of the self-evolution plan: tool selection is a classification
problem — for each task, does the model pick the right tool?

Usage:
    python -m evolution.tools.evolve_tool --tools tools.json --iterations 5
    python -m evolution.tools.evolve_tool --tools tools.json --dry-run

tools.json format: {"tool_name": "description", ...} — names must match the
live tool names so the evolved descriptions can be pasted back.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig, make_lm
from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.tools.tool_module import ToolModule, format_descriptions, parse_descriptions
from evolution.tools.constraints_tools import validate_tool_descriptions
from evolution.tools.fitness_tools import tool_fitness_metric

console = Console()


class ToolDatasetBuilder:
    """Generate (task, expected tool) pairs for the candidate tools."""

    class GenerateCases(dspy.Signature):
        """Generate realistic tool-selection test cases.

        Each case: a realistic user task and the name of the ONE tool that
        should handle it. Vary difficulty; make tasks where the wrong-but-
        similar tools exist (e.g. grep vs read).
        """
        tools: str = dspy.InputField(desc="Tool descriptions block, one per line")
        num_cases: int = dspy.InputField(desc="Number of test cases")
        cases: str = dspy.OutputField(desc="JSON array of {task_input, expected_tool}")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.GenerateCases)

    def generate(self, tools: dict[str, str], num_cases: int) -> EvalDataset:
        import random

        lm = make_lm(self.config.judge_model)
        with dspy.context(lm=lm):
            result = self.generator(
                tools="\n".join(f"{n}: {d}" for n, d in tools.items()),
                num_cases=num_cases,
            )
        raw = result.cases
        try:
            cases = json.loads(raw)
        except json.JSONDecodeError:
            import re
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                raise ValueError(f"Could not parse cases: {raw[:200]}")
            cases = json.loads(m.group())

        examples = [
            EvalExample(
                task_input=c["task_input"],
                expected_behavior=c["expected_tool"],  # metric reads first token
                category="tool_selection",
            )
            for c in cases
            if c.get("task_input") and c.get("expected_tool") in tools
        ]
        random.shuffle(examples)
        n = len(examples)
        n_train = max(1, int(n * self.config.train_ratio))
        n_val = max(1, int(n * self.config.val_ratio))
        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )


def _score_split(module: ToolModule, examples: list[EvalExample]) -> float:
    scores = []
    for ex in examples:
        pred = module(task_input=ex.task_input)
        scores.append(tool_fitness_metric(
            dspy.Example(task_input=ex.task_input, expected_tool=ex.expected_behavior), pred))
    return sum(scores) / max(1, len(scores))


def evolve_tools(
    tools_path: str,
    iterations: int = 5,
    dataset_size: int = 20,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    dataset_path: Optional[str] = None,
    dry_run: bool = False,
):
    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
        eval_dataset_size=dataset_size,
    )

    tools = json.loads(Path(tools_path).read_text())
    if not isinstance(tools, dict) or not tools:
        console.print(f"[red]✗ {tools_path} must be a non-empty {{name: description}} object[/red]")
        sys.exit(1)

    console.print(f"\n[bold cyan]🧬 Tool Description Evolution[/bold cyan] — {len(tools)} tools\n")
    for name, desc in tools.items():
        console.print(f"  {name}: {len(desc)} chars")

    baseline_text = format_descriptions(tools)

    if dry_run:
        # Round-trip check proves the format survives parse without any LLM call
        reparsed = parse_descriptions(baseline_text)
        ok = reparsed is not None and set(reparsed) == set(tools)
        console.print(f"\n[bold green]DRY RUN[/bold green] — round-trip: {'OK' if ok else 'FAILED'}")
        if not ok:
            sys.exit(1)
        return

    # Dataset
    if dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        console.print(f"\n[bold]Generating {dataset_size} selection cases[/bold]")
        dataset = ToolDatasetBuilder(config).generate(tools, dataset_size)
        save = Path("datasets") / "tools" / Path(tools_path).stem
        dataset.save(save)
        console.print(f"  Saved to {save}/")
    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # Optimize
    dspy.configure(lm=make_lm(eval_model))
    module = ToolModule(baseline_text)

    console.print(f"\n[bold cyan]Running GEPA ({iterations} full evals)...[/bold cyan]")
    start = time.time()
    optimizer = dspy.GEPA(
        metric=tool_fitness_metric,
        max_full_evals=iterations,
        reflection_lm=make_lm(optimizer_model),
    )
    optimized = optimizer.compile(
        module,
        trainset=[dspy.Example(task_input=e.task_input, expected_tool=e.expected_behavior).with_inputs("task_input")
                  for e in dataset.train],
        valset=[dspy.Example(task_input=e.task_input, expected_tool=e.expected_behavior).with_inputs("task_input")
                for e in dataset.val],
    )
    elapsed = time.time() - start
    evolved_text = optimized.descriptions_text

    # Constraints
    console.print("\n[bold]Validating evolved descriptions[/bold]")
    results = validate_tool_descriptions(evolved_text, tools, config)
    for r in results:
        icon, color = ("✓", "green") if r.passed else ("✗", "red")
        console.print(f"  [{color}]{icon} {r.constraint_name}[/{color}]: {r.message}")
    if not any(r.passed for r in results) or not all(r.passed for r in results):
        out = Path("output") / "tools" / "evolved_FAILED.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(evolved_text)
        console.print(f"[red]✗ Constraints FAILED — not deploying. Saved to {out}[/red]")
        return

    # Holdout comparison
    base_score = _score_split(module, dataset.holdout)
    evo_score = _score_split(optimized, dataset.holdout)

    table = Table(title="Tool Evolution Results")
    for col in ("Metric", "Baseline", "Evolved", "Change"):
        table.add_column(col, style="bold" if col == "Metric" else "")
    delta = evo_score - base_score
    color = "green" if delta > 0 else "red"
    table.add_row("Holdout accuracy", f"{base_score:.3f}", f"{evo_score:.3f}", f"[{color}]{delta:+.3f}[/{color}]")
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    console.print(table)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("output") / "tools" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evolved_descriptions.md").write_text(evolved_text)
    (out_dir / "baseline_descriptions.md").write_text(baseline_text)
    (out_dir / "metrics.json").write_text(json.dumps({
        "tools": list(tools),
        "baseline_accuracy": base_score,
        "evolved_accuracy": evo_score,
        "improvement": delta,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
    }, indent=2))
    console.print(f"\n  Output saved to {out_dir}/")
    if delta > 0:
        console.print(f"[bold green]✓ Improved tool selection by {delta:+.3f}[/bold green]")
    else:
        console.print("[yellow]⚠ No improvement — try more iterations or a stronger optimizer model[/yellow]")


@click.command()
@click.option("--tools", "tools_path", required=True, help="JSON file: {tool_name: description}")
@click.option("--iterations", default=5, help="GEPA full-eval budget")
@click.option("--dataset-size", default=20, type=int, help="Selection cases to generate")
@click.option("--dataset-path", default=None, help="Reuse an existing dataset directory")
@click.option("--optimizer-model", default="openai/gpt-4.1")
@click.option("--eval-model", default="openai/gpt-4.1-mini")
@click.option("--dry-run", is_flag=True, help="Round-trip check only, no LLM calls")
def main(tools_path, iterations, dataset_size, dataset_path, optimizer_model, eval_model, dry_run):
    """Evolve tool descriptions so the model picks the right tool more often."""
    evolve_tools(
        tools_path=tools_path,
        iterations=iterations,
        dataset_size=dataset_size,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        dataset_path=dataset_path,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()

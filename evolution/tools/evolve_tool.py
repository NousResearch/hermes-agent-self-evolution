"""Phase 2 — evolve tool descriptions against tool-selection accuracy.

Unlike skill evolution, this phase does not need an LLM judge. The objective
is a classification: given a task and the catalog, does the model pick the
tool the agent actually used? That is gradeable exactly, which makes it both
cheaper and more trustworthy than a rubric score.

The cost being attacked is concrete. On the reference install ``tool_search``
and ``tool_describe`` account for 1,594 calls in a single profile — turns
spent locating a tool instead of using one. Descriptions that discriminate
better should reduce both the misses and the hunting.

Usage:
    python -m evolution.tools.evolve_tool --catalog tools.json --iterations 6
    hermes tools list --json > tools.json   # to produce the catalog
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.dspy_lm import make_dspy_lm
from evolution.core.hermes_paths import try_find_hermes_install
from evolution.core.objectives import ObjectiveVector, ObjectiveWeights
from evolution.core.report import ABReport, arm_from_scores
from evolution.tools.tool_catalog import (
    DISCOVERY_TOOLS,
    ToolCatalog,
    ToolChoiceExample,
    discovery_overhead,
    mine_tool_choices,
)

console = Console()

# How many distractor tools accompany the correct one in a selection prompt.
# Large enough that guessing is not viable, small enough to stay cheap.
DEFAULT_DISTRACTORS = 11


class SelectTool(dspy.Signature):
    """Pick the single best tool for a task, given only the catalog.

    Answer with the tool name exactly as it appears in the catalog and
    nothing else.
    """

    catalog: str = dspy.InputField(desc="Available tools, one per line, as 'name: description'")
    task: str = dspy.InputField(desc="What the user asked for")
    tool_name: str = dspy.OutputField(desc="Name of the single best tool")


class ToolSelectionModule(dspy.Module):
    """Tool descriptions as the optimizable parameter.

    The catalog text is embedded in the signature instructions, which is what
    DSPy optimizers mutate — so GEPA rewrites the descriptions themselves.
    """

    def __init__(self, catalog: ToolCatalog, distractors: int = DEFAULT_DISTRACTORS):
        super().__init__()
        self.catalog = catalog
        self.distractors = distractors
        sig = SelectTool.with_instructions(_instructions_for(catalog))
        self.predictor = dspy.Predict(sig)

    def current_instructions(self) -> str:
        return self.predictor.signature.instructions

    def evolved_catalog(self) -> ToolCatalog:
        """Parse the mutated instructions back into a catalog.

        Descriptions the optimizer dropped or mangled fall back to the
        originals, so a malformed rewrite degrades to the baseline for that
        tool rather than emptying its description.
        """
        parsed = _parse_catalog_block(self.current_instructions())
        return self.catalog.with_descriptions(parsed)

    def forward(self, catalog: str, task: str) -> dspy.Prediction:
        result = self.predictor(catalog=catalog, task=task)
        return dspy.Prediction(
            tool_name=_clean_name(result.tool_name, self.catalog.names()),
            catalog_text=self.current_instructions(),
        )


_CATALOG_HEADER = "TOOL CATALOG"


def _instructions_for(catalog: ToolCatalog) -> str:
    return (
        "You select the single best tool for a task.\n\n"
        "Answer with the tool name exactly as written below, and nothing else.\n\n"
        f"{_CATALOG_HEADER}\n{catalog.render()}\n"
    )


def _parse_catalog_block(instructions: str) -> dict[str, str]:
    if _CATALOG_HEADER not in instructions:
        return {}
    block = instructions.split(_CATALOG_HEADER, 1)[1]
    out: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        name, _, desc = line[2:].partition(":")
        name, desc = name.strip(), desc.strip()
        if name and desc:
            out[name] = desc
    return out


def _clean_name(raw: str, known: Optional[Iterable[str]] = None) -> str:
    """Extract a tool name from a model answer.

    Models answer "use the `read_file` tool" despite being told not to, so
    taking the first word would score a correct choice as wrong and teach the
    optimizer to fix a problem the descriptions do not have. Preference order:
    a token that is an actual tool name, then a backticked token, then the
    most name-shaped token.
    """
    text = str(raw or "").strip()
    tokens = [
        t.strip("`'\".,():;[]").strip()
        for t in text.replace(",", " ").replace("\n", " ").split()
    ]
    tokens = [t for t in tokens if t and all(c.isalnum() or c in "_-" for c in t)]
    if not tokens:
        return text.strip("`'\"").strip()

    if known:
        known_set = set(known)
        for token in tokens:
            if token in known_set:
                return token

    # Backticked spans are the model's own emphasis on the answer.
    backticked = re.findall(r"`([A-Za-z0-9_-]+)`", text)
    if backticked:
        return backticked[0]

    underscored = [t for t in tokens if "_" in t]
    if underscored:
        return max(underscored, key=len)

    return tokens[0] if len(tokens) == 1 else max(tokens, key=len)


def build_selection_examples(
    choices: list[ToolChoiceExample],
    catalog: ToolCatalog,
    distractors: int = DEFAULT_DISTRACTORS,
    seed: int = 17,
) -> list[dspy.Example]:
    """Turn mined tool choices into selection problems.

    The candidate list for each example is fixed at build time with a seeded
    shuffle, so both arms of an A/B see identical prompts. Re-sampling
    distractors per arm would make any measured difference partly an artifact
    of the sampling.
    """
    rng = random.Random(seed)
    known = set(catalog.names())
    examples: list[dspy.Example] = []

    for choice in choices:
        if choice.chosen_tool not in known:
            continue
        pool = [n for n in catalog.names() if n != choice.chosen_tool and n not in DISCOVERY_TOOLS]
        rng.shuffle(pool)
        candidates = [choice.chosen_tool] + pool[:distractors]
        rng.shuffle(candidates)

        examples.append(
            dspy.Example(
                task=choice.task_input,
                candidates=candidates,
                expected_tool=choice.chosen_tool,
                discovery_calls=choice.discovery_calls,
            ).with_inputs("task", "catalog")
        )
    return examples


def make_selection_metric(
    catalog: ToolCatalog,
    size_budget: int,
    weights: Optional[ObjectiveWeights] = None,
    on_vector=None,
):
    """Exact-match accuracy on tool choice, with catalog size as a cost.

    Descriptions can always be made more discriminating by making them
    longer, and every tool description is paid for in every request's context.
    Size therefore has to be an objective here, not an afterthought.
    """
    _weights = weights or ObjectiveWeights()
    baseline_chars = catalog.total_chars()

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        expected = getattr(gold, "expected_tool", "")
        actual = getattr(pred, "tool_name", "")
        correct = bool(expected) and actual == expected

        catalog_text = getattr(pred, "catalog_text", "") or ""
        parsed = _parse_catalog_block(catalog_text)
        size_chars = sum(len(v) for v in parsed.values()) or baseline_chars

        vector = ObjectiveVector(
            quality=1.0 if correct else 0.0,
            size_chars=size_chars,
            size_budget=size_budget,
            baseline_chars=baseline_chars,
            max_growth=0.25,
        )
        if on_vector is not None:
            on_vector(vector)

        if correct:
            feedback = f"Correct: {expected}."
        else:
            expected_desc = (catalog.get(expected).description if catalog.get(expected) else "")
            chosen_desc = (catalog.get(actual).description if catalog.get(actual) else "")
            feedback = (
                f"Wrong tool. Task: {getattr(gold, 'task', '')[:200]}\n"
                f"Expected '{expected}' (described as: {expected_desc[:200]})\n"
                f"Model chose '{actual}' (described as: {chosen_desc[:200]})\n"
                "Rewrite these two descriptions so the boundary between them is "
                "unambiguous for this task. State what each tool is for and, "
                "where they overlap, what makes one the wrong choice."
            )

        if vector.size_penalty() or vector.growth_penalty():
            feedback += (
                f"\nSIZE: the catalog is {size_chars:,} chars against a "
                f"{size_budget:,} budget. Sharpen wording rather than adding to it."
            )

        return dspy.Prediction(score=vector.scalarize(_weights), feedback=feedback)

    return metric


@dataclass
class ToolEvolutionResult:
    baseline_accuracy: float
    evolved_accuracy: float
    baseline_chars: int
    evolved_chars: int
    verdict: str
    output_dir: Path


def evolve_tools(
    catalog_path: Optional[str] = None,
    hermes_data_dir: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    iterations: int = 6,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    max_examples: int = 120,
    distractors: int = DEFAULT_DISTRACTORS,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    output_root: str = "./output",
    dry_run: bool = False,
) -> ToolEvolutionResult:
    """Optimize tool descriptions for selection accuracy."""
    console.print("\n[bold cyan]🔧 Tool description evolution[/bold cyan]\n")

    install = try_find_hermes_install(hermes_data_dir)
    if install is None:
        raise RuntimeError(
            "No Hermes data directory found — tool choices are mined from "
            "state.db. Set HERMES_DATA_DIR."
        )
    console.print(f"  Hermes data: {install.root}")

    # Catalog
    if catalog_path:
        catalog = ToolCatalog.from_json_file(Path(catalog_path))
        console.print(f"  Catalog: {len(catalog)} tools from {catalog_path}")
    else:
        from evolution.tools.tool_catalog import extract_catalog_from_repo

        if not hermes_repo:
            raise RuntimeError(
                "Provide --catalog (from `hermes tools list --json`) or --hermes-repo"
            )
        catalog = extract_catalog_from_repo(Path(hermes_repo))
        console.print(f"  Catalog: {len(catalog)} tools scanned from {hermes_repo}")

    if len(catalog) < 4:
        raise RuntimeError(f"Catalog has only {len(catalog)} tools — nothing to disambiguate")

    # The number this phase exists to move.
    overhead = discovery_overhead(install)
    console.print(
        f"  Discovery overhead: {overhead['discovery_calls']:,} of "
        f"{overhead['total_tool_calls']:,} tool calls "
        f"({overhead['discovery_share']:.1%}) spent finding tools"
    )

    choices = mine_tool_choices(install, limit=max_examples * 3)
    console.print(f"  Mined {len(choices)} real tool choices")
    examples = build_selection_examples(choices, catalog, distractors=distractors)
    console.print(f"  Built {len(examples)} selection examples")

    if len(examples) < 8:
        raise RuntimeError(
            f"Only {len(examples)} usable examples — the catalog and the mined "
            "tool names may not overlap. Check that --catalog matches this install."
        )

    random.Random(11).shuffle(examples)
    examples = examples[:max_examples]
    split = max(1, int(len(examples) * 0.5))
    val_split = max(split + 1, int(len(examples) * 0.75))
    trainset, valset, holdout = examples[:split], examples[split:val_split], examples[val_split:]
    console.print(f"  Split: {len(trainset)} train / {len(valset)} val / {len(holdout)} holdout")

    baseline_chars = catalog.total_chars()
    size_budget = int(baseline_chars * 1.15)
    console.print(f"  Catalog size: {baseline_chars:,} chars (budget {size_budget:,})")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated.[/bold green]")
        return ToolEvolutionResult(0.0, 0.0, baseline_chars, baseline_chars, "DRY_RUN", Path(output_root))

    lm = make_dspy_lm(eval_model, api_base=api_base, api_key=api_key)
    reflection_lm = make_dspy_lm(
        optimizer_model, temperature=1.0, max_tokens=4000, api_base=api_base, api_key=api_key
    )
    dspy.configure(lm=lm)

    baseline_module = ToolSelectionModule(catalog, distractors=distractors)
    metric = make_selection_metric(catalog, size_budget=size_budget)

    console.print(f"\n[bold cyan]Running GEPA ({iterations} full evals)...[/bold cyan]\n")
    start = time.time()
    optimizer = dspy.GEPA(metric=metric, max_full_evals=iterations, reflection_lm=reflection_lm)
    evolved_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
    elapsed = time.time() - start

    # Holdout: same examples, same prompts, both arms.
    baseline_scores, evolved_scores = [], []
    for ex in holdout:
        catalog_text = _catalog_for(baseline_module.catalog, ex)
        with dspy.context(lm=lm):
            baseline_scores.append(_accuracy(metric, ex, baseline_module, catalog_text))
            evolved_scores.append(_accuracy(metric, ex, evolved_module, catalog_text))

    evolved_catalog = evolved_module.evolved_catalog()

    report = ABReport(
        subject="tool descriptions",
        baseline=arm_from_scores("baseline", baseline_scores, baseline_chars),
        evolved=arm_from_scores("evolved", evolved_scores, evolved_catalog.total_chars()),
        metric_name="tool-selection accuracy × catalog size",
        extra={
            "tools": len(catalog),
            "distractors per example": distractors,
            "discovery overhead before": f"{overhead['discovery_share']:.1%}",
            "elapsed": f"{elapsed:.0f}s",
        },
    )
    verdict, reason = report.verdict()

    table = Table(title="Tool Description Evolution")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_row("Selection accuracy", f"{report.baseline.mean:.1%}", f"{report.evolved.mean:.1%}")
    table.add_row("Catalog size", f"{baseline_chars:,}", f"{evolved_catalog.total_chars():,}")
    console.print()
    console.print(table)
    console.print(f"\n[bold]Verdict: {verdict}[/bold] — {reason}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / "tools" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_catalog.json").write_text(catalog.to_json())
    (output_dir / "evolved_catalog.json").write_text(evolved_catalog.to_json())
    (output_dir / "discovery_overhead.json").write_text(json.dumps(overhead, indent=2))
    report.write(output_dir)
    console.print(f"\n  Output saved to {output_dir}/")

    return ToolEvolutionResult(
        baseline_accuracy=report.baseline.mean,
        evolved_accuracy=report.evolved.mean,
        baseline_chars=baseline_chars,
        evolved_chars=evolved_catalog.total_chars(),
        verdict=verdict,
        output_dir=output_dir,
    )


def _catalog_for(catalog: ToolCatalog, example) -> str:
    return catalog.restricted_to(getattr(example, "candidates", catalog.names())).render()


def _accuracy(metric, example, module, catalog_text: str) -> float:
    try:
        pred = module(catalog=catalog_text, task=example.task)
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [yellow]example failed: {exc}[/yellow]")
        return 0.0
    result = metric(example, pred)
    return float(getattr(result, "score", result) or 0.0)


@click.command()
@click.option("--catalog", "catalog_path", default=None,
              help="Tool catalog JSON (from `hermes tools list --json`)")
@click.option("--hermes-data-dir", default=None, help="Hermes data directory (state.db, profiles)")
@click.option("--hermes-repo", default=None, help="hermes-agent repo, for fallback catalog extraction")
@click.option("--iterations", default=6, help="GEPA full evaluations")
@click.option("--optimizer-model", default="openai/gpt-4.1")
@click.option("--eval-model", default="openai/gpt-4.1-mini")
@click.option("--max-examples", default=120, help="Cap on mined selection examples")
@click.option("--distractors", default=DEFAULT_DISTRACTORS, help="Wrong tools shown per example")
@click.option("--output-root", default="./output")
@click.option("--dry-run", is_flag=True)
@click.option("--api-base", default=None)
@click.option("--api-key", default=None)
def main(catalog_path, hermes_data_dir, hermes_repo, iterations, optimizer_model,
         eval_model, max_examples, distractors, output_root, dry_run, api_base, api_key):
    """Evolve Hermes tool descriptions against real tool-selection accuracy."""
    try:
        evolve_tools(
            catalog_path=catalog_path,
            hermes_data_dir=hermes_data_dir,
            hermes_repo=hermes_repo,
            iterations=iterations,
            optimizer_model=optimizer_model,
            eval_model=eval_model,
            max_examples=max_examples,
            distractors=distractors,
            api_base=api_base,
            api_key=api_key,
            output_root=output_root,
            dry_run=dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"\n[red]✗ {exc}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

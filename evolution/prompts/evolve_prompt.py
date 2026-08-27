"""Phase 3 — evolve a section of the live system prompt.

The prompt is read from ``state.db`` rather than reconstructed from source, so
what gets optimized is what Hermes actually ran, and how many sessions ran it
is known rather than assumed.

One section is optimized at a time. The system prompt is a single artifact
shared by every skill and every request, so rewriting all of it at once
produces a diff no reviewer can evaluate and a result no measurement can
attribute. Section-at-a-time keeps both tractable.

Usage:
    python -m evolution.prompts.evolve_prompt --list
    python -m evolution.prompts.evolve_prompt --section "Tool use" --iterations 6
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.dspy_lm import make_dspy_lm
from evolution.core.dataset_builder import SyntheticDatasetBuilder
from evolution.core.external_importers import NoSessionDataError, build_dataset_from_external
from evolution.core.fitness import LLMJudge, make_fitness_metric
from evolution.core.hermes_paths import try_find_hermes_install
from evolution.core.objectives import ObjectiveWeights
from evolution.core.report import ABReport, arm_from_scores
from evolution.core.state_db import prompt_variant_outcomes
from evolution.prompts.prompt_sections import (
    DEFAULT_PROMPT_GROWTH,
    load_live_prompts,
    load_prompt_file,
    prompt_cost_note,
)
from evolution.skills.evolve_skill import EvolutionError, _score_arms
from evolution.skills.skill_module import SkillModule

console = Console()


def list_prompts(hermes_data_dir: Optional[str], profile: Optional[str]) -> int:
    """Show the live prompt variants and their sections."""
    install = try_find_hermes_install(hermes_data_dir)
    if install is None:
        console.print("[red]✗ No Hermes data directory found. Set HERMES_DATA_DIR.[/red]")
        return 1

    prompts = load_live_prompts(install, profile=profile)
    if not prompts:
        console.print("[yellow]No system prompts recorded in state.db.[/yellow]")
        return 1

    console.print(f"\n[bold]System prompt variants in {install.root}[/bold]\n")
    outcomes = prompt_variant_outcomes(install)

    table = Table()
    table.add_column("hash", style="dim")
    table.add_column("sessions", justify="right")
    table.add_column("chars", justify="right")
    table.add_column("sections", justify="right")
    table.add_column("success", justify="right")
    for prompt in prompts[:12]:
        stats = outcomes.get(prompt.prompt_hash, {})
        rate = stats.get("success_rate")
        table.add_row(
            prompt.prompt_hash[:12],
            str(prompt.sessions),
            f"{prompt.size:,}",
            str(len(prompt.sections)),
            f"{rate:.0%}" if rate is not None else "—",
        )
    console.print(table)

    top = prompts[0]
    console.print(f"\n[bold]Sections of the most-used variant[/bold] ({top.describe()})\n")
    section_table = Table()
    section_table.add_column("section")
    section_table.add_column("chars", justify="right")
    section_table.add_column("share", justify="right")
    for sec in top.sections:
        section_table.add_row(sec.title, f"{sec.size:,}", f"{sec.size / max(1, top.size):.0%}")
    console.print(section_table)
    return 0


def evolve_prompt_section(
    section_name: str,
    hermes_data_dir: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    profile: Optional[str] = None,
    prompt_file: Optional[str] = None,
    iterations: int = 6,
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    eval_source: str = "sessiondb",
    dataset_path: Optional[str] = None,
    max_growth: float = DEFAULT_PROMPT_GROWTH,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    output_root: str = "./output",
    dry_run: bool = False,
) -> dict:
    """Optimize one named section of the system prompt."""
    console.print(f"\n[bold cyan]🧠 System prompt evolution[/bold cyan] — section: [bold]{section_name}[/bold]\n")

    install = try_find_hermes_install(hermes_data_dir)

    # Source the prompt
    if prompt_file:
        prompt = load_prompt_file(Path(prompt_file))
        console.print(f"  Prompt: {prompt.describe()} (from {prompt_file})")
    else:
        if install is None:
            raise EvolutionError(
                "No Hermes data directory found and no --prompt-file given. "
                "Set HERMES_DATA_DIR or pass a prompt file."
            )
        prompts = load_live_prompts(install, profile=profile)
        if not prompts:
            raise EvolutionError(
                f"No system prompts recorded in {install.root}. Run --list to check."
            )
        prompt = prompts[0]
        console.print(f"  Prompt: {prompt.describe()} (most-used variant)")

    section = prompt.section(section_name)
    if section is None:
        available = ", ".join(s.title for s in prompt.sections)
        raise EvolutionError(f"Section '{section_name}' not found. Available: {available}")

    console.print(
        f"  Section: '{section.title}' — {section.size:,} chars "
        f"({section.size / max(1, prompt.size):.0%} of the prompt)"
    )

    # Every request pays for the whole prompt, so the budget is tight.
    size_budget = int(section.size * (1 + max_growth))
    console.print(f"  Size budget: {size_budget:,} chars (max growth {max_growth:.0%} — paid per request)")

    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo) if hermes_repo else None,
        hermes_data_dir=hermes_data_dir,
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=eval_model,
        api_base=api_base,
        api_key=api_key,
        max_prompt_growth=max_growth,
    )

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated.[/bold green]")
        return {"dry_run": True, "section": section.title, "size_budget": size_budget}

    # Dataset
    console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")
    if eval_source == "sessiondb":
        if install is None:
            raise EvolutionError("--eval-source sessiondb requires a Hermes data directory")
        try:
            dataset = build_dataset_from_external(
                skill_name=section.slug,
                skill_text=section.body,
                sources=["hermes", "claude-code", "copilot"],
                output_path=Path("datasets") / "prompts" / section.slug,
                model=eval_model,
                api_base=api_base,
                api_key=api_key,
                install=install,
                profiles=[profile] if profile else None,
            )
        except NoSessionDataError as exc:
            raise EvolutionError(str(exc)) from exc
    elif eval_source == "golden":
        from evolution.core.dataset_builder import GoldenDatasetLoader

        if not dataset_path:
            raise EvolutionError("--eval-source golden requires --dataset-path")
        dataset = GoldenDatasetLoader.load(Path(dataset_path))
    else:
        dataset = SyntheticDatasetBuilder(config).generate(
            artifact_text=section.body, artifact_type="prompt_section"
        )
    console.print(
        f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout"
    )
    if not dataset.train:
        raise EvolutionError("Dataset has no training examples")

    # Optimize
    lm = make_dspy_lm(eval_model, api_base=api_base, api_key=api_key)
    reflection_lm = make_dspy_lm(
        optimizer_model, temperature=1.0, max_tokens=3000, api_base=api_base, api_key=api_key
    )
    dspy.configure(lm=lm)

    baseline_module = SkillModule(section.body)
    metric = make_fitness_metric(
        config=config,
        baseline_text=section.body,
        size_budget=size_budget,
        weights=ObjectiveWeights(quality=1.0, size=1.5),  # prompt bytes cost more
        judge=LLMJudge(config),
    )

    console.print(f"\n[bold cyan]Running GEPA ({iterations} full evals)...[/bold cyan]\n")
    start = time.time()
    optimizer = dspy.GEPA(metric=metric, max_full_evals=iterations, reflection_lm=reflection_lm)
    evolved_module = optimizer.compile(
        baseline_module,
        trainset=dataset.to_dspy_examples("train"),
        valset=dataset.to_dspy_examples("val"),
    )
    elapsed = time.time() - start

    evolved_body = evolved_module.get_evolved_text()
    evolved_prompt_text = prompt.replace_section(section, evolved_body)

    # Holdout — same metric, same examples, both arms
    holdout = dataset.to_dspy_examples("holdout") or dataset.to_dspy_examples("val")
    baseline_scores, evolved_scores = _score_arms(
        holdout, baseline_module, evolved_module, metric, lm
    )

    delta_chars = len(evolved_body) - section.size
    observed_requests = sum(
        s.get("sessions", 0) for s in (prompt_variant_outcomes(install).values() if install else [])
    )

    report = ABReport(
        subject=f"system prompt — {section.title}",
        baseline=arm_from_scores("baseline", baseline_scores, section.size),
        evolved=arm_from_scores("evolved", evolved_scores, len(evolved_body)),
        metric_name="judge composite × prompt-size objective",
        extra={
            "prompt variant": prompt.prompt_hash[:12] or "(file)",
            "sessions on this variant": prompt.sessions,
            "section share of prompt": f"{section.size / max(1, prompt.size):.0%}",
            "cost": prompt_cost_note(section, delta_chars, max(1, observed_requests)),
            "elapsed": f"{elapsed:.0f}s",
        },
    )
    if len(evolved_body) > size_budget:
        report.constraints_passed = False
        report.constraint_failures.append(
            f"size_limit: {len(evolved_body):,}/{size_budget:,} chars"
        )

    verdict, reason = report.verdict()

    table = Table(title=f"Prompt Section: {section.title}")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_row("Holdout score", f"{report.baseline.mean:.3f}", f"{report.evolved.mean:.3f}")
    table.add_row("Section size", f"{section.size:,}", f"{len(evolved_body):,}")
    table.add_row("Whole prompt", f"{prompt.size:,}", f"{len(evolved_prompt_text):,}")
    console.print()
    console.print(table)
    console.print(f"\n[bold]Verdict: {verdict}[/bold] — {reason}")
    console.print(f"  [dim]{report.extra['cost']}[/dim]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / "prompts" / section.slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_section.md").write_text(section.body)
    (output_dir / "evolved_section.md").write_text(evolved_body)
    (output_dir / "evolved_full_prompt.md").write_text(evolved_prompt_text)
    report.write(output_dir)
    console.print(f"\n  Output saved to {output_dir}/")

    return {
        "section": section.title,
        "verdict": verdict,
        "reason": reason,
        "baseline_score": report.baseline.mean,
        "evolved_score": report.evolved.mean,
        "delta_chars": delta_chars,
        "output_dir": str(output_dir),
    }


@click.command()
@click.option("--section", "section_name", default=None, help="Section title or slug to evolve")
@click.option("--list", "do_list", is_flag=True, help="List prompt variants and their sections")
@click.option("--hermes-data-dir", default=None, help="Hermes data directory (state.db, profiles)")
@click.option("--hermes-repo", default=None, help="hermes-agent repo path")
@click.option("--profile", default=None, help="Restrict to one profile")
@click.option("--prompt-file", default=None, help="Optimize a prompt from a file instead of state.db")
@click.option("--iterations", default=6, help="GEPA full evaluations")
@click.option("--optimizer-model", default="openai/gpt-4.1")
@click.option("--eval-model", default="openai/gpt-4.1-mini")
@click.option("--eval-source", default="sessiondb",
              type=click.Choice(["sessiondb", "synthetic", "golden"]))
@click.option("--dataset-path", default=None)
@click.option("--max-growth", default=DEFAULT_PROMPT_GROWTH, type=float,
              help="Max section growth. Tight by default: every request pays for it.")
@click.option("--output-root", default="./output")
@click.option("--dry-run", is_flag=True)
@click.option("--api-base", default=None)
@click.option("--api-key", default=None)
def main(section_name, do_list, hermes_data_dir, hermes_repo, profile, prompt_file,
         iterations, optimizer_model, eval_model, eval_source, dataset_path,
         max_growth, output_root, dry_run, api_base, api_key):
    """Evolve a section of the Hermes system prompt."""
    if do_list:
        sys.exit(list_prompts(hermes_data_dir, profile))

    if not section_name:
        console.print("[red]✗ Provide --section, or --list to see what is available.[/red]")
        sys.exit(1)

    try:
        evolve_prompt_section(
            section_name=section_name,
            hermes_data_dir=hermes_data_dir,
            hermes_repo=hermes_repo,
            profile=profile,
            prompt_file=prompt_file,
            iterations=iterations,
            optimizer_model=optimizer_model,
            eval_model=eval_model,
            eval_source=eval_source,
            dataset_path=dataset_path,
            max_growth=max_growth,
            api_base=api_base,
            api_key=api_key,
            output_root=output_root,
            dry_run=dry_run,
        )
    except EvolutionError as exc:
        console.print(f"\n[red]✗ {exc}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()

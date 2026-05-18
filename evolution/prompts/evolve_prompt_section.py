"""Evolve a single system-prompt section for behavioral fitness.

Usage:
    python -m evolution.prompts.evolve_prompt_section \
        --section MEMORY_GUIDANCE --iterations 1 --eval-source synthetic \
        --optimizer-model 'ollama/qwen2.5:7b'
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.core.eval_integrity import validate_model_separation
from evolution.core.integration_adapter import HermesIntegrationAdapter
from evolution.core.phase_gate import evaluate_phase3_gate, write_phase_gate_result
from evolution.core.reproducibility import (
    build_prompt_section_reproducibility_manifest,
    write_manifest,
)
from evolution.core.rollout_policy import get_rollout_policy, should_auto_rollback
from evolution.core.stop_loss import StopLossGuard
from evolution.prompts.prompt_dataset import SyntheticPromptScenarioBuilder
from evolution.prompts.prompt_loader import (
    EVOLVABLE_SECTION_NAMES,
    find_prompt_section,
    identity_traits_present,
)
from evolution.prompts.prompt_module import (
    PromptSectionModule,
    behavioral_fitness_metric,
    clean_evolved_section,
    make_llm_judge_metric,
)

console = Console()


def _score_holdout(
    holdout_examples: list[dspy.Example],
    baseline_module: PromptSectionModule,
    optimized_module: PromptSectionModule,
    lm,
    metric,
    *,
    section_text_changed: bool,
) -> tuple[list[float], list[float]]:
    baseline_scores: list[float] = []
    evolved_scores: list[float] = []
    for ex in holdout_examples:
        ctx = dspy.context(lm=lm) if lm is not None else nullcontext()
        with ctx:
            baseline_pred = baseline_module(scenario=ex.task_input)
            baseline_scores.append(metric(ex, baseline_pred))
            if section_text_changed:
                evolved_pred = optimized_module(scenario=ex.task_input)
            else:
                evolved_pred = baseline_pred
            evolved_scores.append(metric(ex, evolved_pred))
    return baseline_scores, evolved_scores


@click.command()
@click.option("--section", "section_name", required=True,
              help=f"Section name. One of: {', '.join(EVOLVABLE_SECTION_NAMES)} "
                   "or PLATFORM_HINTS:<platform>")
@click.option("--iterations", default=1, show_default=True, type=int)
@click.option("--eval-source", default="synthetic", show_default=True,
              type=click.Choice(["synthetic"]))
@click.option("--optimizer-model", default="openai/gpt-4.1", show_default=True)
@click.option("--eval-model", default="openai/gpt-4.1-mini", show_default=True)
@click.option("--judge-model", default="openrouter/google/gemini-2.5-flash", show_default=True)
@click.option("--hermes-repo", default=None, help="Override hermes-agent repo path")
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--allow-two-family-mode", is_flag=True, default=False)
@click.option("--minimum-model-families", type=int, default=None)
@click.option("--metric", "metric_name", default="overlap", show_default=True,
              type=click.Choice(["overlap", "judge", "hybrid"]),
              help="Fitness metric: 'overlap' (fast keyword), 'judge' (LLM-as-judge), "
                   "'hybrid' (judge with overlap fallback weight 0.3)")
@click.option("--eval-dataset-size", type=int, default=None,
              help="Override EvolutionConfig.eval_dataset_size (number of synthetic scenarios)")
@click.option("--optimizer-auto", default="light", show_default=True,
              type=click.Choice(["light", "medium", "heavy"]),
              help="MIPROv2 'auto' budget level — light/medium/heavy ≈ 10/25/50 trials")
def main(
    section_name: str,
    iterations: int,
    eval_source: str,
    optimizer_model: str,
    eval_model: str,
    judge_model: str,
    hermes_repo: Optional[str],
    dry_run: bool,
    allow_two_family_mode: bool,
    minimum_model_families: Optional[int],
    metric_name: str,
    eval_dataset_size: Optional[int],
    optimizer_auto: str,
):
    evolve(
        section_name=section_name,
        iterations=iterations,
        eval_source=eval_source,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
        hermes_repo=hermes_repo,
        dry_run=dry_run,
        allow_two_family_mode=allow_two_family_mode,
        minimum_model_families=minimum_model_families,
        metric_name=metric_name,
        eval_dataset_size=eval_dataset_size,
        optimizer_auto=optimizer_auto,
    )


def evolve(
    *,
    section_name: str,
    iterations: int = 1,
    eval_source: str = "synthetic",
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    judge_model: str = "openrouter/google/gemini-2.5-flash",
    hermes_repo: Optional[str] = None,
    dry_run: bool = False,
    allow_two_family_mode: bool = False,
    minimum_model_families: Optional[int] = None,
    metric_name: str = "overlap",
    eval_dataset_size: Optional[int] = None,
    optimizer_auto: str = "light",
):
    if minimum_model_families is None:
        resolved_min_families = 2 if allow_two_family_mode else 3
    else:
        resolved_min_families = max(1, int(minimum_model_families))

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
        minimum_model_families=resolved_min_families,
    )
    if eval_dataset_size is not None:
        config.eval_dataset_size = max(4, int(eval_dataset_size))
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    adapter = HermesIntegrationAdapter(config.hermes_agent_path)
    if config.phase0_enforce:
        try:
            adapter.assert_compatible()
        except RuntimeError as exc:
            console.print(f"[red]✗ Phase 0 compatibility check failed: {exc}[/red]")
            sys.exit(1)

        separation_errors = validate_model_separation(
            generator_model=config.judge_model,
            judge_model=config.eval_model,
            optimizer_model=config.optimizer_model,
            minimum_distinct_families=config.minimum_model_families,
        )
        if separation_errors:
            for error in separation_errors:
                console.print(f"[red]✗ {error}[/red]")
            sys.exit(1)

    stop_loss = StopLossGuard(config)

    console.print(f"\n[bold cyan]Phase 3 — System-Prompt Section Evolution[/bold cyan] → [bold]{section_name}[/bold]\n")

    section = find_prompt_section(section_name, config.hermes_agent_path)
    if not section:
        console.print(f"[red]✗ Section '{section_name}' not found in agent/prompt_builder.py[/red]")
        sys.exit(1)

    console.print(f"  Source: {section.source_path.relative_to(config.hermes_agent_path)} ({section.var_name})")
    if section.platform_key:
        console.print(f"  Platform: {section.platform_key}")
    console.print(f"  Section size: {len(section.text):,} chars")

    is_identity_section = section.name == "DEFAULT_AGENT_IDENTITY"
    if is_identity_section:
        baseline_traits = identity_traits_present(section.text)
        console.print(f"  Baseline identity traits: {baseline_traits}")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated.[/bold green]")
        return

    # ── 2. Build behavioral dataset ────────────────────────────────────
    console.print(f"\n[bold]Generating behavioral scenarios[/bold] (source: {eval_source})")
    builder = SyntheticPromptScenarioBuilder(config)
    dataset = builder.generate(section_name=section.name, section_text=section.text)
    save_path = Path("datasets") / "prompts" / section.name.replace(":", "__")
    dataset.save(save_path)
    console.print(f"  Generated {len(dataset.all_examples)} scenarios → {save_path}/")
    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate baseline constraints ───────────────────────────────
    console.print("\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    for c in validator.validate_all(section.text, "prompt_section"):
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")

    # ── 4. Configure optimizer + run ───────────────────────────────────
    console.print("\n[bold]Configuring optimizer[/bold]")
    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)
    baseline_module = PromptSectionModule(section_name=section.name, text=section.text)
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    judge_lm = dspy.LM(judge_model)
    if metric_name == "overlap":
        metric_fn = behavioral_fitness_metric
        console.print("  Metric: keyword overlap (fast)")
    elif metric_name == "judge":
        metric_fn = make_llm_judge_metric(judge_lm)
        console.print(f"  Metric: LLM-as-judge ({judge_model})")
    else:
        metric_fn = make_llm_judge_metric(judge_lm, fallback_weight=0.3)
        console.print(f"  Metric: hybrid (judge {judge_model} + 0.3 overlap fallback)")

    console.print(f"\n[bold cyan]Running optimization (auto={optimizer_auto}, iterations={iterations})...[/bold cyan]\n")
    start_time = time.time()
    try:
        optimizer = dspy.GEPA(metric=metric_fn, max_steps=iterations)
        optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
    except Exception as exc:
        console.print(f"[yellow]GEPA not available ({exc}); falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(metric=metric_fn, auto=optimizer_auto, track_stats=True)
        optimized_module = optimizer.compile(baseline_module, trainset=trainset)
    elapsed = time.time() - start_time
    console.print(f"  Optimization completed in {elapsed:.1f}s")

    if optimized_module.section_text == section.text:
        candidates = getattr(optimized_module, "candidate_programs", []) or []
        for entry in candidates:
            cand_program = entry.get("program") if isinstance(entry, dict) else None
            if cand_program is None:
                continue
            try:
                cand_text = getattr(cand_program, "section_text", None)
            except Exception:
                cand_text = None
            if cand_text and cand_text != section.text:
                console.print(
                    f"  [yellow]Best MIPROv2 winner == baseline; switching to next-best candidate "
                    f"(score {entry.get('score'):.3f})[/yellow]",
                )
                optimized_module = cand_program
                break

    raw_evolved = optimized_module.section_text
    max_budget = int(len(section.text) * (1.0 + config.max_prompt_growth))
    evolved_text = clean_evolved_section(raw_evolved, max_chars=max_budget)
    optimized_module.section_text = evolved_text
    section_text_changed = evolved_text != section.text
    if raw_evolved != evolved_text:
        console.print(
            f"  [yellow]Trimmed optimizer-inlined examples and enforced budget: "
            f"{len(raw_evolved):,} → {len(evolved_text):,} chars (budget {max_budget})[/yellow]",
        )

    evolved_traits: dict[str, bool] | None = None
    if is_identity_section:
        evolved_traits = identity_traits_present(evolved_text)
        console.print(f"  Evolved identity traits: {evolved_traits}")

    console.print("\n[bold]Validating evolved section[/bold]")
    evolved_results = validator.validate_all(evolved_text, "prompt_section", baseline_text=section.text)
    all_pass = True
    for c in evolved_results:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False
    if not all_pass:
        console.print("[red]✗ Evolved section FAILED constraints — not deploying[/red]")
        out_dir = Path("output") / "prompts" / section.name.replace(":", "__") / "FAILED"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "evolved_section.txt").write_text(evolved_text, encoding="utf-8")
        return

    # ── 5. Holdout evaluation ──────────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout ({len(dataset.holdout)} examples)[/bold]")
    if not section_text_changed:
        console.print("[yellow]No section text changes produced; reusing baseline scores[/yellow]")
    holdout_examples = dataset.to_dspy_examples("holdout")
    baseline_scores, evolved_scores = _score_holdout(
        holdout_examples, baseline_module, optimized_module, lm, metric_fn,
        section_text_changed=section_text_changed,
    )
    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    stop_loss.register_attempt(
        cost_usd=0.0,
        runtime_minutes=elapsed / 60.0,
        improvement=improvement,
        stable=improvement >= config.minimum_detectable_effect,
    )

    rollout_policy = get_rollout_policy("system_prompt")
    rollback = should_auto_rollback(
        kpi_delta=improvement,
        safety_incidents=0,
        policy=rollout_policy,
    )

    # ── 6. Report + artifacts ──────────────────────────────────────────
    table = Table(title="Prompt Section Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")
    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Behavioral score",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Section size",
        f"{len(section.text):,} chars",
        f"{len(evolved_text):,} chars",
        f"{len(evolved_text) - len(section.text):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")
    console.print()
    console.print(table)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "prompts" / section.name.replace(":", "__") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_prompt_section_reproducibility_manifest(
        section_name=section.name,
        section_source_path=section.source_path,
        baseline_text=section.text,
        dataset=dataset,
        config=config,
        baseline_score=avg_baseline,
        evolved_score=avg_evolved,
        improvement=improvement,
        elapsed_seconds=elapsed,
        evolution_repo_path=Path(__file__).resolve().parents[2],
    )
    write_manifest(output_dir / "reproducibility_manifest.json", manifest)

    (output_dir / "baseline_section.txt").write_text(section.text, encoding="utf-8")
    (output_dir / "evolved_section.txt").write_text(evolved_text, encoding="utf-8")

    metrics = {
        "section_name": section.name,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "elapsed_seconds": elapsed,
        "section_text_changed": section_text_changed,
        "baseline_chars": len(section.text),
        "evolved_chars": len(evolved_text),
        "evolved_identity_traits": evolved_traits,
        "metric": metric_name,
        "optimizer_auto": optimizer_auto,
        "eval_dataset_size": len(dataset.all_examples),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8",
    )

    gate_result = evaluate_phase3_gate(
        baseline_score=avg_baseline,
        evolved_score=avg_evolved,
        improvement=improvement,
        config=config,
        manifest=manifest,
        identity_traits=evolved_traits,
    )
    write_phase_gate_result(output_dir / "phase3_gate.json", gate_result)

    if gate_result.passed:
        console.print("[green]✓ Phase 3 gate passed[/green]")
    else:
        console.print("[red]✗ Phase 3 gate failed[/red]")
        for f in gate_result.failures:
            console.print(f"  - {f}")

    if rollback:
        console.print("[yellow]⚠ Rollout policy recommends rollback due to KPI regression[/yellow]")
    for reason in stop_loss.termination_reasons():
        console.print(f"[yellow]⚠ Stop-loss: {reason}[/yellow]")

    console.print(f"\n  Artifacts saved to {output_dir}/")


if __name__ == "__main__":
    main()

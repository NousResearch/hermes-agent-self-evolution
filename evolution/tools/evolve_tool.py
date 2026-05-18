"""Evolve a single tool's top-level description for selection accuracy.

Usage:
    python -m evolution.tools.evolve_tool --tool read_file --iterations 1 \
        --eval-source synthetic --optimizer-model 'ollama/qwen2.5:7b'
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
from evolution.core.phase_gate import evaluate_phase2_gate, write_phase_gate_result
from evolution.core.reproducibility import (
    build_tool_reproducibility_manifest,
    write_manifest,
)
from evolution.core.rollout_policy import get_rollout_policy, should_auto_rollback
from evolution.core.stop_loss import StopLossGuard
from evolution.tools.tool_dataset import SyntheticToolSelectionBuilder
from evolution.tools.tool_loader import (
    discover_tool_schemas,
    find_tool_schema,
    reassemble_schema,
    summarize_other_tools,
)
from evolution.tools.tool_module import (
    ToolDescriptionModule,
    clean_evolved_description,
    normalize_decision,
    tool_selection_metric,
)

console = Console()


def _score_holdout(
    holdout_examples: list[dspy.Example],
    baseline_module: ToolDescriptionModule,
    optimized_module: ToolDescriptionModule,
    lm,
    *,
    description_changed: bool,
) -> tuple[list[float], list[float], dict[str, float]]:
    baseline_scores: list[float] = []
    evolved_scores: list[float] = []
    selection_rate = {"yes": [0, 0], "no": [0, 0]}  # [baseline_selects_yes, evolved_selects_yes]

    for ex in holdout_examples:
        ctx = dspy.context(lm=lm) if lm is not None else nullcontext()
        with ctx:
            baseline_pred = baseline_module(task_input=ex.task_input)
            baseline_scores.append(tool_selection_metric(ex, baseline_pred))
            if description_changed:
                evolved_pred = optimized_module(task_input=ex.task_input)
            else:
                evolved_pred = baseline_pred
            evolved_scores.append(tool_selection_metric(ex, evolved_pred))

        label = (getattr(ex, "expected_behavior", "") or "").strip().lower()
        if label in selection_rate:
            b_dec = normalize_decision(getattr(baseline_pred, "decision", "") or "")
            e_dec = normalize_decision(getattr(evolved_pred, "decision", "") or "")
            selection_rate[label][0] += 1 if b_dec == "yes" else 0
            selection_rate[label][1] += 1 if e_dec == "yes" else 0

    # Cross-tool proxy regression: among negative examples (label="no"), how often
    # did the evolved description still say "yes"? An increase = stealing from peers.
    neg_count = sum(1 for ex in holdout_examples
                    if (getattr(ex, "expected_behavior", "") or "").strip().lower() == "no")
    if neg_count > 0:
        baseline_steal = selection_rate["no"][0] / neg_count
        evolved_steal = selection_rate["no"][1] / neg_count
        cross_tool_regression = baseline_steal - evolved_steal
    else:
        cross_tool_regression = 0.0

    per_tool = {"_peer_selection_rate_delta": cross_tool_regression}
    return baseline_scores, evolved_scores, per_tool


@click.command()
@click.option("--tool", "tool_name", required=True, help="Tool name (top-level schema 'name' field)")
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
def main(
    tool_name: str,
    iterations: int,
    eval_source: str,
    optimizer_model: str,
    eval_model: str,
    judge_model: str,
    hermes_repo: Optional[str],
    dry_run: bool,
    allow_two_family_mode: bool,
    minimum_model_families: Optional[int],
):
    evolve(
        tool_name=tool_name,
        iterations=iterations,
        eval_source=eval_source,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=judge_model,
        hermes_repo=hermes_repo,
        dry_run=dry_run,
        allow_two_family_mode=allow_two_family_mode,
        minimum_model_families=minimum_model_families,
    )


def evolve(
    *,
    tool_name: str,
    iterations: int = 1,
    eval_source: str = "synthetic",
    optimizer_model: str = "openai/gpt-4.1",
    eval_model: str = "openai/gpt-4.1-mini",
    judge_model: str = "openrouter/google/gemini-2.5-flash",
    hermes_repo: Optional[str] = None,
    dry_run: bool = False,
    allow_two_family_mode: bool = False,
    minimum_model_families: Optional[int] = None,
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

    console.print(f"\n[bold cyan]Phase 2 — Tool Description Optimization[/bold cyan] → [bold]{tool_name}[/bold]\n")

    # ── 1. Discover the tool + peer registry ────────────────────────────
    all_tools = discover_tool_schemas(config.hermes_agent_path)
    target = find_tool_schema(tool_name, config.hermes_agent_path)
    if not target:
        console.print(f"[red]✗ Tool '{tool_name}' not found under {config.hermes_agent_path / 'tools'}[/red]")
        sys.exit(1)

    other_tools_summary = summarize_other_tools(all_tools, exclude_name=tool_name)
    console.print(f"  Source: {target.source_path.relative_to(config.hermes_agent_path)} ({target.var_name})")
    console.print(f"  Description size: {len(target.description):,} chars")
    console.print(f"  Peer tools indexed: {sum(1 for t in all_tools if t.name != tool_name)}")

    if dry_run:
        console.print("\n[bold green]DRY RUN — setup validated.[/bold green]")
        return

    # ── 2. Build dataset ────────────────────────────────────────────────
    console.print(f"\n[bold]Building tool-selection dataset[/bold] (source: {eval_source})")
    builder = SyntheticToolSelectionBuilder(config)
    dataset = builder.generate(
        tool_name=tool_name,
        tool_description=target.description,
        other_tools_summary=other_tools_summary,
    )
    save_path = Path("datasets") / "tools" / tool_name
    dataset.save(save_path)
    console.print(f"  Generated {len(dataset.all_examples)} examples → {save_path}/")
    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate baseline constraints ────────────────────────────────
    console.print("\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    for c in validator.validate_all(target.description, "tool_description"):
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")

    # ── 4. Configure DSPy + optimizer ───────────────────────────────────
    console.print("\n[bold]Configuring optimizer[/bold]")
    lm = dspy.LM(eval_model)
    dspy.configure(lm=lm)

    baseline_module = ToolDescriptionModule(
        tool_name=tool_name,
        description=target.description,
        other_tools_summary=other_tools_summary,
    )
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    # ── 5. Run optimization ─────────────────────────────────────────────
    console.print(f"\n[bold cyan]Running optimization ({iterations} iterations)...[/bold cyan]\n")
    start_time = time.time()
    try:
        optimizer = dspy.GEPA(metric=tool_selection_metric, max_steps=iterations)
        optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
    except Exception as exc:
        console.print(f"[yellow]GEPA not available ({exc}); falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(metric=tool_selection_metric, auto="light")
        optimized_module = optimizer.compile(baseline_module, trainset=trainset)
    elapsed = time.time() - start_time
    console.print(f"  Optimization completed in {elapsed:.1f}s")

    raw_evolved = optimized_module.tool_description
    max_budget = min(config.max_tool_desc_size,
                     int(len(target.description) * (1.0 + config.max_prompt_growth)))
    evolved_description = clean_evolved_description(raw_evolved, max_chars=max_budget)
    optimized_module.tool_description = evolved_description
    description_changed = evolved_description != target.description
    if raw_evolved != evolved_description:
        console.print(
            f"  [yellow]Trimmed optimizer-inlined examples and enforced budget: "
            f"{len(raw_evolved):,} → {len(evolved_description):,} chars "
            f"(budget {max_budget})[/yellow]",
        )

    # ── 6. Validate evolved description constraints ─────────────────────
    console.print("\n[bold]Validating evolved description[/bold]")
    evolved_results = validator.validate_all(
        evolved_description,
        "tool_description",
        baseline_text=target.description,
    )
    all_pass = True
    for c in evolved_results:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not c.passed:
            all_pass = False
    if not all_pass:
        console.print("[red]✗ Evolved description FAILED constraints — not deploying[/red]")
        out_dir = Path("output") / "tools" / tool_name / "FAILED"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "evolved_description.txt").write_text(evolved_description, encoding="utf-8")
        return

    # ── 7. Holdout evaluation ───────────────────────────────────────────
    console.print(f"\n[bold]Evaluating on holdout ({len(dataset.holdout)} examples)[/bold]")
    if not description_changed:
        console.print("[yellow]No description changes produced; reusing baseline scores[/yellow]")
    holdout_examples = dataset.to_dspy_examples("holdout")
    baseline_scores, evolved_scores, per_peer_delta = _score_holdout(
        holdout_examples, baseline_module, optimized_module, lm,
        description_changed=description_changed,
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

    rollout_policy = get_rollout_policy("tool_description")
    rollback = should_auto_rollback(
        kpi_delta=improvement,
        safety_incidents=0,
        policy=rollout_policy,
    )

    # ── 8. Report ───────────────────────────────────────────────────────
    table = Table(title="Tool Description Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")
    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Selection accuracy",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    table.add_row(
        "Description size",
        f"{len(target.description):,} chars",
        f"{len(evolved_description):,} chars",
        f"{len(evolved_description) - len(target.description):+,} chars",
    )
    table.add_row("Time", "", f"{elapsed:.1f}s", "")
    table.add_row("Iterations", "", str(iterations), "")
    console.print()
    console.print(table)

    # ── 9. Save artifacts + phase gate ──────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "tools" / tool_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_tool_reproducibility_manifest(
        tool_name=tool_name,
        tool_source_path=target.source_path,
        baseline_description=target.description,
        dataset=dataset,
        config=config,
        baseline_score=avg_baseline,
        evolved_score=avg_evolved,
        improvement=improvement,
        elapsed_seconds=elapsed,
        evolution_repo_path=Path(__file__).resolve().parents[2],
    )
    write_manifest(output_dir / "reproducibility_manifest.json", manifest)

    (output_dir / "baseline_description.txt").write_text(target.description, encoding="utf-8")
    (output_dir / "evolved_description.txt").write_text(evolved_description, encoding="utf-8")

    evolved_schema = reassemble_schema(target.schema, evolved_description)
    (output_dir / "evolved_schema.json").write_text(
        json.dumps(evolved_schema, indent=2), encoding="utf-8",
    )

    metrics = {
        "tool_name": tool_name,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "elapsed_seconds": elapsed,
        "description_changed": description_changed,
        "baseline_description_chars": len(target.description),
        "evolved_description_chars": len(evolved_description),
        "peer_selection_rate_delta": per_peer_delta["_peer_selection_rate_delta"],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8",
    )

    cross_tool_input = {"_peer_proxy": per_peer_delta["_peer_selection_rate_delta"]}
    gate_result = evaluate_phase2_gate(
        baseline_score=avg_baseline,
        evolved_score=avg_evolved,
        improvement=improvement,
        config=config,
        manifest=manifest,
        per_tool_regression=cross_tool_input,
    )
    write_phase_gate_result(output_dir / "phase2_gate.json", gate_result)

    if gate_result.passed:
        console.print("[green]✓ Phase 2 gate passed[/green]")
    else:
        console.print("[red]✗ Phase 2 gate failed[/red]")
        for f in gate_result.failures:
            console.print(f"  - {f}")

    if rollback:
        console.print("[yellow]⚠ Rollout policy recommends rollback due to KPI regression[/yellow]")
    for reason in stop_loss.termination_reasons():
        console.print(f"[yellow]⚠ Stop-loss: {reason}[/yellow]")

    console.print(f"\n  Artifacts saved to {output_dir}/")


if __name__ == "__main__":
    main()

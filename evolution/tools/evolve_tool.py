"""Evolve a single tool description in an MCP manifest using DSPy + GEPA.

Usage:
    python -m evolution.tools.evolve_tool --tool search_files --manifest manifest.json
    python -m evolution.tools.evolve_tool --tool search_files --manifest manifest.json --apply
"""

from __future__ import annotations

import difflib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click
import dspy
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import (
    ConstraintValidator,
    effective_absolute_char_ceiling,
    resolve_decision_rule,
)
from evolution.core.dataset_builder import (
    EvalDataset,
    SyntheticDatasetBuilder,
    split_examples,
)
from evolution.core.fitness import LLMJudge
from evolution.core.lm_timing_callback import (
    COST_LEDGER,
    LMTimingCallback,
    register_litellm_cost_callback,
    register_litellm_failure_callback,
)
from evolution.core.quality_gate import (
    QUALITY_GATE_PRESETS,
    resolve_proposer_mode,
    write_gate_decision,
)
from evolution.core.stats import paired_bootstrap
from evolution.skills.knee_point import CandidatePick, select_knee_point
from evolution.tools.tool_judge import make_tool_fitness_metric
from evolution.tools.tool_module import (
    ToolModule,
    _extract_description_from_sentinels,
)
from evolution.tools.tool_proposer import BudgetAwareToolProposer
from evolution.tools.tool_source import SentinelParseError, ToolManifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)

console = Console()


def _description_from_predictor(predictor: Any, target_tool_name: str) -> str:
    """Read the sentinel-delimited region of a predictor's instructions.

    Used both by the knee-point parsimony measurement and by the metric's
    [BUDGET] reflection feedback so both axes track description length
    (not full rendered-manifest length).
    """
    instructions = predictor.signature.instructions or ""
    try:
        return _extract_description_from_sentinels(instructions, target_tool_name)
    except SentinelParseError as e:
        logging.getLogger(__name__).warning(
            "could not extract description from predictor instructions "
            "for target %r: %s; reporting empty for budget/parsimony purposes",
            target_tool_name, e,
        )
        return ""


def _candidate_description(candidate: Any, target_tool_name: str) -> str:
    """Pull the description text from a GEPA-built candidate module."""
    return _description_from_predictor(candidate.selector.predict, target_tool_name)


def _build_examples(eval_examples: list, *, for_module: bool) -> list[dspy.Example]:
    """Convert EvalExamples to dspy.Examples.

    `task_input` and `expected_behavior` are always populated so the metric
    can read them. `for_module=True` flips the input field to `task` so the
    ToolModule.forward signature receives the right kwarg.
    """
    input_field = "task" if for_module else "task_input"
    out = []
    for ex in eval_examples:
        out.append(
            dspy.Example(
                task=ex.task_input,
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
            ).with_inputs(input_field)
        )
    return out


def _dataset_payload(dataset: EvalDataset) -> dict[str, Any]:
    sources: dict[str, int] = {}
    categories: dict[str, int] = {}
    for ex in dataset.all_examples:
        sources[ex.source] = sources.get(ex.source, 0) + 1
        categories[ex.category] = categories.get(ex.category, 0) + 1
    return {
        "size_total": len(dataset.all_examples),
        "size_train": len(dataset.train),
        "size_val": len(dataset.val),
        "size_holdout": len(dataset.holdout),
        "sources": sources,
        "categories": categories,
    }


def _compute_win_loss(
    baseline_per_example: list[float],
    evolved_per_example: list[float],
) -> dict[str, Any]:
    deltas = [e - b for b, e in zip(baseline_per_example, evolved_per_example)]
    return {
        "n_wins": sum(1 for d in deltas if d > 0),
        "n_losses": sum(1 for d in deltas if d < 0),
        "n_ties": sum(1 for d in deltas if d == 0),
        "worst_regression": min(deltas) if deltas else 0.0,
        "worst_improvement": max(deltas) if deltas else 0.0,
    }


def _knee_point_payload(knee_pick: Optional[CandidatePick]) -> dict[str, Any]:
    if knee_pick is None:
        return {"applied": False, "reason": "no_detailed_results"}
    return {
        "applied": True,
        "fallback": knee_pick.fallback,
        "epsilon": knee_pick.epsilon,
        "band_size": knee_pick.band_size,
        "picked_idx": knee_pick.picked_idx,
        "picked_val_score": knee_pick.val_score,
        "picked_val_rank_in_band": knee_pick.val_rank_in_band,
        "picked_body_chars": knee_pick.body_chars,
        "gepa_default_idx": knee_pick.gepa_default_idx,
        "gepa_default_body_chars": knee_pick.gepa_default_body_chars,
        "band_roster": knee_pick.band_roster,
    }


def _holdout_evaluate_with_metric(
    module: dspy.Module,
    holdout_examples: list,
    metric,
    lm,
) -> tuple[float, list[float]]:
    """Run dspy.Evaluate against `module`, returning (mean, per_example_scores)."""
    def two_arg_metric(example, prediction, *_args, **_kwargs):
        result = metric(example, prediction)
        return float(getattr(result, "score", result))

    evaluator = dspy.Evaluate(
        devset=holdout_examples,
        metric=two_arg_metric,
        num_threads=4,
        provide_traceback=True,
        max_errors=len(holdout_examples) * 100,
    )
    with dspy.context(lm=lm):
        result = evaluator(module)
    mean = float(result.score) / 100.0
    per_example = [float(s) for _, _, s in result.results]
    return mean, per_example


def _emit_patch(baseline_text: str, evolved_text: str, path: Path) -> str:
    """Return a unified diff of (baseline -> evolved) labelled with `path`."""
    label = str(path)
    diff_lines = difflib.unified_diff(
        baseline_text.splitlines(keepends=True),
        evolved_text.splitlines(keepends=True),
        fromfile=f"a/{label}",
        tofile=f"b/{label}",
    )
    return "".join(diff_lines)


_CLAUDE_CODE_PLUGIN_CACHE_MARKER = (".claude", "plugins", "cache")


def _is_claude_code_plugin_cache_path(path: Path) -> bool:
    parts = path.resolve().parts
    marker = _CLAUDE_CODE_PLUGIN_CACHE_MARKER
    for i in range(len(parts) - len(marker) + 1):
        if parts[i:i + len(marker)] == marker:
            return True
    return False


def _apply_in_place(
    manifest_path: Path,
    manifest: ToolManifest,
    target_tool_name: str,
    evolved_description: str,
) -> bool:
    """Rewrite `manifest_path` in place with the evolved description.

    Preserves any `_evolution_metadata` block in the source file. Refuses
    to write into Claude Code's plugin cache.
    """
    if _is_claude_code_plugin_cache_path(manifest_path):
        logging.getLogger(__name__).warning(
            "--apply skipped: %s is under a Claude Code plugin cache "
            "(~/.claude/plugins/cache); plugin caches are managed by "
            "Claude Code and writing to them is unsafe.",
            manifest_path,
        )
        return False

    source = json.loads(manifest_path.read_text())
    for entry in source.get("tools", []):
        if entry.get("name") == target_tool_name:
            entry["description"] = evolved_description
            break
    manifest_path.write_text(json.dumps(source, indent=2) + "\n")
    return True


def _manifest_to_dict(manifest: ToolManifest) -> dict[str, Any]:
    """Serialize a ToolManifest back to MCP-list_tools shape (plus metadata)."""
    out: dict[str, Any] = {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in manifest.tools
        ],
    }
    if manifest.confusable_neighbors:
        out["_evolution_metadata"] = {
            "confusable_neighbors": dict(manifest.confusable_neighbors),
        }
    return out


def evolve(
    tool_name: str,
    manifest_path: Path,
    *,
    iterations: int = 5,
    fitness_profile: str = "balanced",
    quality_gate: str = "default",
    max_absolute_chars: Optional[int] = None,
    apply: bool = False,
    patch: bool = False,
    seed: int = 42,
    eval_dataset_size: int = 150,
    holdout_ratio: float = 0.5,
    output_dir: Optional[Path] = None,
    optimizer_model: str = "openai/gpt-4.1",
    reflection_model: Optional[str] = "openai/gpt-5-mini",
    eval_model: str = "openai/gpt-4.1-mini",
) -> dict[str, Any]:
    """Evolve one tool description inside a manifest.

    Returns a dict summary of the run (mirrors metrics.json).
    """
    manifest_path = Path(manifest_path)
    manifest = ToolManifest.from_json_file(manifest_path)
    target = manifest.find_tool(tool_name)
    baseline_description = target.description

    preset = QUALITY_GATE_PRESETS[quality_gate]
    if quality_gate == "off":
        logging.getLogger(__name__).warning(
            '--quality-gate off still enforces a regression check (mean ≥ 0). '
            'For "deploy if not significantly worse than baseline" semantics, '
            'use --quality-gate non-inferiority --inferiority-tolerance 0.02.'
        )
    resolved_abs = max_absolute_chars if max_absolute_chars is not None else preset["max_absolute_chars"]
    resolved_gate_mode = preset.get("gate_mode", "no_regression")
    resolved_tolerance = preset.get("inferiority_tolerance", 0.0)

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        reflection_model=reflection_model,
        eval_model=eval_model,
        judge_model=eval_model,
        seed=seed,
        growth_free_threshold=preset["growth_free_threshold"],
        growth_quality_slope=preset["growth_quality_slope"],
        max_absolute_chars=int(resolved_abs),
        gate_mode=resolved_gate_mode,
        inferiority_tolerance=float(resolved_tolerance),
        fitness_profile=fitness_profile,
        eval_dataset_size=eval_dataset_size,
        holdout_ratio=holdout_ratio,
    )

    console.print(
        f"\n[bold cyan]Tool Description Self-Evolution[/bold cyan] — "
        f"Evolving tool: [bold]{tool_name}[/bold]\n"
    )
    console.print(f"  Manifest: {manifest_path}")
    console.print(f"  Tools in manifest: {len(manifest.tools)}")
    console.print(f"  Baseline description ({len(baseline_description)} chars): {baseline_description[:80]}…")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path("output") / "tools" / tool_name / timestamp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = output_dir / "run.log"
    file_handler = logging.FileHandler(run_log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    ))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    try:
        register_litellm_failure_callback()
        register_litellm_cost_callback()
        COST_LEDGER.reset()
        console.print(f"  Run log: {run_log_path}")

        console.print(f"\n[bold]Building tool-selection eval dataset[/bold] (synthetic, three buckets)")
        builder = SyntheticDatasetBuilder(config)
        raw_examples = builder.generate_tool_selection(
            manifest=manifest,
            target_tool=tool_name,
            num_cases=config.eval_dataset_size,
        )
        dataset = split_examples(
            raw_examples,
            seed=config.seed,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            holdout_ratio=config.holdout_ratio,
        )
        console.print(
            f"  Generated {len(dataset.all_examples)} examples — "
            f"{len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout"
        )

        if len(dataset.holdout) < config.min_holdout_size:
            console.print(
                f"[red]✗ Holdout has only {len(dataset.holdout)} examples; need ≥{config.min_holdout_size} "
                f"to gate on improvement signal. Increase eval_dataset_size or holdout_ratio.[/red]"
            )
            sys.exit(1)

        console.print(f"\n[bold]Validating baseline description[/bold]")
        validator = ConstraintValidator(config)
        baseline_constraints = validator.validate_static(baseline_description, "tool_description")
        for c in baseline_constraints:
            icon = "✓" if c.passed else "✗"
            color = "green" if c.passed else "red"
            console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
        if not all(c.passed for c in baseline_constraints):
            console.print("[yellow]⚠ Baseline description has constraint violations — proceeding anyway[/yellow]")

        lm = dspy.LM(eval_model, request_timeout=60, num_retries=5)
        dspy.configure(
            lm=lm,
            warn_on_type_mismatch=False,
            callbacks=[LMTimingCallback()],
        )

        judge = LLMJudge(config)
        metric = make_tool_fitness_metric(
            judge=judge,
            baseline_description=baseline_description,
            manifest=manifest,
            target_tool_name=tool_name,
            max_growth=config.growth_free_threshold,
            text_extractor=lambda predictor: _description_from_predictor(predictor, tool_name),
        )

        baseline_module = ToolModule(
            target_tool_name=tool_name,
            manifest=manifest,
            target_description=baseline_description,
        )

        proposer_mode = resolve_proposer_mode(config.fitness_profile)
        proposer = BudgetAwareToolProposer(
            target_tool_name=tool_name,
            manifest=manifest,
            target_description=baseline_description,
            baseline_chars=len(baseline_description),
            max_growth=config.bap_max_growth,
        )

        trainset = _build_examples(dataset.train, for_module=True)
        valset = _build_examples(dataset.val, for_module=True)

        console.print(f"\n[bold cyan]Running GEPA optimization (max_full_evals={iterations})[/bold cyan]\n")
        start_time = time.time()

        reflection_lm = dspy.LM(
            reflection_model or optimizer_model,
            temperature=1.0,
            max_tokens=32000,
            cache=False,
            request_timeout=300,
            num_retries=2,
        )
        optimizer = dspy.GEPA(
            metric=metric,
            max_full_evals=iterations,
            reflection_lm=reflection_lm,
            seed=config.seed,
            track_stats=True,
            instruction_proposer=proposer,
        )
        optimized_module = optimizer.compile(
            baseline_module, trainset=trainset, valset=valset,
        )

        elapsed = time.time() - start_time
        console.print(f"\n  GEPA optimization completed in {elapsed:.1f}s")

        knee_pick: Optional[CandidatePick] = None
        if hasattr(optimized_module, "detailed_results"):
            details = optimized_module.detailed_results
            knee_pick = select_knee_point(
                candidates=details.candidates,
                val_aggregate_scores=details.val_aggregate_scores,
                n_val=len(valset),
                static_validator=lambda txt: validator.validate_static(txt, "tool_description"),
                gepa_default_idx=details.best_idx,
                text_extractor=lambda c: _candidate_description(c, tool_name),
            )
            evolved_description = _candidate_description(knee_pick.module, tool_name)
            optimized_module = ToolModule(
                target_tool_name=tool_name,
                manifest=manifest,
                target_description=evolved_description,
            )
            console.print(
                f"\n[bold]Knee-point selection[/bold]: picked candidate "
                f"{knee_pick.picked_idx} (val={knee_pick.val_score:.3f}, "
                f"rank {knee_pick.val_rank_in_band} of {knee_pick.band_size} in band, "
                f"{knee_pick.body_chars} chars vs GEPA default "
                f"{knee_pick.gepa_default_body_chars}; ε={knee_pick.epsilon:.3f}; "
                f"fallback={knee_pick.fallback})"
            )
        else:
            evolved_description = optimized_module.description_text

        console.print(f"\n[bold]Validating evolved description (static checks)[/bold]")
        static_constraints = validator.validate_static(evolved_description, "tool_description")
        static_pass = True
        for c in static_constraints:
            icon = "✓" if c.passed else "✗"
            color = "green" if c.passed else "red"
            console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
            if not c.passed:
                static_pass = False

        run_inputs = {
            "seed": config.seed,
            "iterations": iterations,
            "optimizer_model": optimizer_model,
            "reflection_model": config.reflection_model,
            "eval_model": config.eval_model,
            "eval_dataset_size": config.eval_dataset_size,
            "holdout_ratio": config.holdout_ratio,
            "quality_gate_preset": quality_gate,
            "fitness_profile": fitness_profile,
        }
        tool_payload_fields = {
            "artifact_type": "tool_description",
            "target_tool": tool_name,
            "manifest_neighbor_count": len(manifest.tools) - 1,
            "sentinel_failures": proposer.sentinel_failures,
        }

        if not static_pass:
            console.print("[red]✗ Evolved description FAILED static constraints — not deploying[/red]")
            failed_path = output_dir / "evolved_FAILED.json"
            evolved_manifest = manifest.replace_description(tool_name, evolved_description)
            failed_path.write_text(json.dumps(_manifest_to_dict(evolved_manifest), indent=2) + "\n")
            write_gate_decision(output_dir, {
                "schema_version": "4",
                "decision": "reject",
                "reason": "static_constraint_failure",
                "failed_constraints": [c.constraint_name for c in static_constraints if not c.passed],
                "messages": [c.message for c in static_constraints if not c.passed],
                "knee_point": _knee_point_payload(knee_pick),
                "dataset": _dataset_payload(dataset),
                "run_inputs": run_inputs,
                **tool_payload_fields,
            })
            console.print(f"  Saved failed variant to {failed_path}")
            return {"decision": "reject", "reason": "static_constraint_failure"}

        console.print(
            f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]"
        )
        holdout_examples = _build_examples(dataset.holdout, for_module=True)
        avg_baseline, baseline_per_example = _holdout_evaluate_with_metric(
            baseline_module, holdout_examples, metric, lm,
        )
        avg_evolved, evolved_per_example = _holdout_evaluate_with_metric(
            optimized_module, holdout_examples, metric, lm,
        )
        improvement = avg_evolved - avg_baseline

        console.print(f"\n[bold]Validating growth against holdout improvement[/bold]")
        bootstrap = paired_bootstrap(
            baseline_per_example,
            evolved_per_example,
            confidence=config.bootstrap_confidence,
            n_resamples=config.bootstrap_n_resamples,
            seed=config.seed,
        )
        # Growth + ceiling check on the description, not the rendered manifest —
        # the gate's curve has to apply to the artifact the user actually evolves.
        growth_constraints = validator.validate_growth_with_quality(
            evolved_description, baseline_description, bootstrap,
        )
        growth_pass = True
        for c in growth_constraints:
            icon = "✓" if c.passed else "✗"
            color = "green" if c.passed else "red"
            console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
            if not c.passed:
                growth_pass = False

        baseline_chars = len(baseline_description)
        evolved_chars = len(evolved_description)
        growth_pct = (evolved_chars - baseline_chars) / max(1, baseline_chars)
        required_improvement = max(
            0.0,
            config.growth_quality_slope * (growth_pct - config.growth_free_threshold),
        )
        decision_rule_used = resolve_decision_rule(config, growth_pct)
        decision_payload = {
            "schema_version": "4",
            "decision": "deploy" if growth_pass else "reject",
            "reason": "passed" if growth_pass else "growth_quality_gate",
            "decision_rule_used": decision_rule_used,
            "gate_mode": config.gate_mode,
            "inferiority_tolerance": config.inferiority_tolerance,
            "growth_pct": growth_pct,
            "required_improvement": required_improvement,
            "baseline_chars": baseline_chars,
            "evolved_chars": evolved_chars,
            "absolute_char_ceiling": config.max_absolute_chars,
            "effective_absolute_char_ceiling": effective_absolute_char_ceiling(
                config.max_absolute_chars, baseline_chars,
            ),
            "growth_free_threshold": config.growth_free_threshold,
            "fitness_profile": config.fitness_profile,
            "proposer_mode": proposer_mode,
            "growth_quality_slope": config.growth_quality_slope,
            "baseline_per_example": baseline_per_example,
            "evolved_per_example": evolved_per_example,
            "avg_baseline": avg_baseline,
            "avg_evolved": avg_evolved,
            "bootstrap": bootstrap,
            "win_loss": _compute_win_loss(baseline_per_example, evolved_per_example),
            "failed_constraints": [c.constraint_name for c in growth_constraints if not c.passed],
            "messages": [c.message for c in growth_constraints if not c.passed],
            "knee_point": _knee_point_payload(knee_pick),
            "dataset": _dataset_payload(dataset),
            "run_inputs": run_inputs,
            **tool_payload_fields,
        }
        gate_path = write_gate_decision(output_dir, decision_payload)
        console.print(f"  [dim]Gate decision logged to {gate_path}[/dim]")

        if not growth_pass:
            console.print("[red]✗ Evolved description REJECTED by quality gate — not deploying[/red]")
            evolved_manifest = manifest.replace_description(tool_name, evolved_description)
            failed_path = output_dir / "evolved_FAILED.json"
            failed_path.write_text(json.dumps(_manifest_to_dict(evolved_manifest), indent=2) + "\n")
            console.print(f"  Saved failed variant to {failed_path}")
            reject_reason = decision_payload.get("reason", "growth_quality_gate")
            if apply:
                print(
                    f"--apply skipped: gate rejected (decision: reject, reason: {reject_reason})",
                    file=sys.stderr,
                )
            if patch:
                print(
                    f"--patch skipped: gate rejected (decision: reject, reason: {reject_reason})",
                    file=sys.stderr,
                )
            return {"decision": "reject", "reason": reject_reason}

        table = Table(title="Tool Description Evolution Results")
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
            "Description Size",
            f"{baseline_chars:,} chars",
            f"{evolved_chars:,} chars",
            f"{evolved_chars - baseline_chars:+,} chars",
        )
        table.add_row("Time", "", f"{elapsed:.1f}s", "")
        console.print()
        console.print(table)

        evolved_manifest = manifest.replace_description(tool_name, evolved_description)
        (output_dir / "baseline_manifest.json").write_text(
            json.dumps(_manifest_to_dict(manifest), indent=2) + "\n"
        )
        (output_dir / "evolved_manifest.json").write_text(
            json.dumps(_manifest_to_dict(evolved_manifest), indent=2) + "\n"
        )
        metrics = {
            "tool_name": tool_name,
            "manifest_path": str(manifest_path),
            "timestamp": timestamp,
            "iterations": iterations,
            "optimizer_model": optimizer_model,
            "eval_model": eval_model,
            "baseline_score": avg_baseline,
            "evolved_score": avg_evolved,
            "improvement": improvement,
            "baseline_chars": baseline_chars,
            "evolved_chars": evolved_chars,
            "train_examples": len(dataset.train),
            "val_examples": len(dataset.val),
            "holdout_examples": len(dataset.holdout),
            "elapsed_seconds": elapsed,
            "sentinel_failures": proposer.sentinel_failures,
            "cost": COST_LEDGER.summary(),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        console.print(f"\n  Output saved to {output_dir}/")

        if patch:
            baseline_text = json.dumps(_manifest_to_dict(manifest), indent=2) + "\n"
            evolved_text = json.dumps(_manifest_to_dict(evolved_manifest), indent=2) + "\n"
            patch_text = _emit_patch(baseline_text, evolved_text, manifest_path)
            sys.stdout.write(patch_text)
            if patch_text and not patch_text.endswith("\n"):
                sys.stdout.write("\n")

        if apply:
            applied = _apply_in_place(manifest_path, manifest, tool_name, evolved_description)
            if applied:
                console.print(f"  --apply: wrote evolved description to {manifest_path}")

        if improvement > 0:
            console.print(
                f"\n[bold green]✓ Evolution improved tool description by "
                f"{improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]"
            )
        else:
            console.print(
                f"\n[yellow]⚠ Evolution did not improve tool description (change: {improvement:+.3f})[/yellow]"
            )

        return metrics
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


@click.command()
@click.option("--tool", "tool_name", required=True, help="Name of the tool to evolve")
@click.option(
    "--manifest",
    "manifest_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the MCP-list_tools-shaped manifest JSON file",
)
@click.option("--iterations", default=5, type=int, help="GEPA max_full_evals")
@click.option(
    "--fitness-profile",
    default="balanced",
    type=click.Choice(["compression", "balanced", "growth"]),
    help="Composite fitness weighting profile.",
)
@click.option(
    "--quality-gate",
    default="default",
    type=click.Choice(list(QUALITY_GATE_PRESETS.keys())),
    help="Preset for the deploy gate's growth-vs-improvement curve.",
)
@click.option(
    "--max-absolute-chars",
    default=None,
    type=int,
    help="Override the preset's max_absolute_chars (hard ceiling on description size).",
)
@click.option("--seed", default=42, type=int, help="RNG seed.")
@click.option(
    "--apply",
    "apply_flag",
    is_flag=True,
    default=False,
    help="On deploy, rewrite the manifest in place with the evolved description.",
)
@click.option(
    "--patch",
    "patch_flag",
    is_flag=True,
    default=False,
    help="On deploy, emit a unified diff of the manifest changes to stdout.",
)
def main(
    tool_name: str,
    manifest_path: Path,
    iterations: int,
    fitness_profile: str,
    quality_gate: str,
    max_absolute_chars: Optional[int],
    seed: int,
    apply_flag: bool,
    patch_flag: bool,
) -> None:
    """Evolve one tool description in an MCP manifest using DSPy + GEPA."""
    if apply_flag and patch_flag:
        raise click.UsageError("--apply and --patch are mutually exclusive")
    evolve(
        tool_name=tool_name,
        manifest_path=manifest_path,
        iterations=iterations,
        fitness_profile=fitness_profile,
        quality_gate=quality_gate,
        max_absolute_chars=max_absolute_chars,
        apply=apply_flag,
        patch=patch_flag,
        seed=seed,
    )


if __name__ == "__main__":
    main()

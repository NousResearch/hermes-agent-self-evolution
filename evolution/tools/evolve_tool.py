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
from evolution.core.auth_check import preflight as _preflight_lm_credentials
from evolution.core.saturation_check import (
    saturation_preflight,
    render_saturation_panel,
    interactive_confirm,
    is_non_interactive,
)
from evolution.core.cost_advisor import (
    find_cheaper_alternative as _find_cheaper_alternative,
    render_suggestion_panel as _render_cost_suggestion_panel,
)
from evolution.core.hermes_provider import (
    HermesProviderError,
    instantiate_lm,
    resolve_default_lm,
    resolved_lms_dump,
)
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
from evolution.core.lm_timing_callback import (
    COST_LEDGER,
    CostCeilingExceeded,
    LMTimingCallback,
    register_litellm_cost_callback,
    register_litellm_failure_callback,
)
from evolution.core.quality_gate import (
    QUALITY_GATE_PRESETS,
    resolve_proposer_mode,
    run_benchmark_hook,
    write_cost_ceiling_abort,
    write_gate_decision,
)
from evolution.core.stats import paired_bootstrap
from evolution.skills.knee_point import CandidatePick, select_knee_point
from evolution.tools.session_mining import (
    HermesToolImporter,
    build_tool_dataset_from_sessions,
)
from evolution.tools.tool_judge import ToolJudge, make_tool_fitness_metric
from evolution.tools.tool_module import (
    ToolModule,
    _extract_description_from_sentinels,
)
from evolution.tools.tool_proposer import BudgetAwareToolProposer
from evolution.tools.tool_source import (
    SentinelParseError,
    ToolManifest,
    ToolSource,
    discover_tool_sources,
)

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


def _dataset_payload(
    dataset: EvalDataset,
    *,
    dropped_tools: tuple[tuple[str, str], ...] = (),
    sessiondb_drops: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    sources: dict[str, int] = {}
    categories: dict[str, int] = {}
    for ex in dataset.all_examples:
        sources[ex.source] = sources.get(ex.source, 0) + 1
        categories[ex.category] = categories.get(ex.category, 0) + 1
    payload: dict[str, Any] = {
        "size_total": len(dataset.all_examples),
        "size_train": len(dataset.train),
        "size_val": len(dataset.val),
        "size_holdout": len(dataset.holdout),
        "sources": sources,
        "categories": categories,
        # Surface adapter-dropped tools (e.g., function-built schemas the
        # Hermes adapter couldn't parse statically). Empty list on the MCP
        # path. Serialized as a list of 2-lists for JSON compatibility.
        "dropped_tools": [list(pair) for pair in dropped_tools],
    }
    if sessiondb_drops is not None:
        # Surface session-mining drop reasons (importer + judge stages) so an
        # auditor can see why N candidates became M examples. Pulled out as a
        # top-level field so calibration scripts don't have to know the
        # internal key set.
        payload["sessiondb_drops"] = dict(sessiondb_drops)
        payload["dropped_non_manifest_count"] = int(sessiondb_drops.get("non_manifest", 0))
    return payload


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


def _resolve_source(manifest_path: Path) -> ToolSource:
    """Pick the first ToolSource adapter that claims support for ``manifest_path``.

    Discovery returns adapters per directory; we hand each its own dir
    (the file's parent when path is a file, the path itself when it's a
    directory) so each adapter has a stable root.
    """
    root = manifest_path.parent if manifest_path.is_file() else manifest_path
    sources = discover_tool_sources([root])
    for source in sources:
        if source.supports(manifest_path):
            return source
    raise ValueError(
        f"no ToolSource supports {manifest_path}; "
        f"expected a .json file (MCP manifest) or a directory of .py files (Hermes tools)"
    )


def _maybe_build_closed_loop_cache(
    *,
    tool_name: str,
    baseline_description: str,
    suite_path: Optional[Path],
    hermes_repo: Optional[Path],
    saturation_threshold: float,
    min_iters: int,
    window_size: int,
    gate_mode: str = "sampled",
    agent_model: Optional[str] = None,
    agent_timeout_seconds: Optional[int] = None,
):
    """Build a ClosedLoopFeedbackCache when the user opted in, else None.

    Local imports keep the validation stack out of the cold path — this
    module's tests + CLI smoke don't need the hermes_runner / validator
    imports unless the flag was set.
    """
    if suite_path is None:
        return None
    if hermes_repo is None:
        # main() guards this; assert in case evolve() is called from code
        # bypassing the CLI.
        raise ValueError(
            "closed_loop_suite_path set without closed_loop_hermes_repo"
        )
    from evolution.core.closed_loop_feedback import ClosedLoopFeedbackCache
    from evolution.validation.artifact_installer import (
        HermesToolDescriptionInstaller,
    )
    from evolution.validation.hermes_runner import HermesAgentRunner
    from evolution.validation.task import TaskSuite
    from evolution.validation.validator import ClosedLoopValidator

    installer = HermesToolDescriptionInstaller(
        hermes_repo=hermes_repo, tool_name=tool_name
    )
    runner_kwargs: dict = {"model": agent_model}
    if agent_timeout_seconds is not None:
        runner_kwargs["timeout_seconds"] = agent_timeout_seconds
    runner = HermesAgentRunner(**runner_kwargs)
    validator = ClosedLoopValidator(installer=installer, runner=runner)
    suite = TaskSuite.from_jsonl(suite_path)
    return ClosedLoopFeedbackCache(
        validator=validator,
        suite=suite,
        artifact_name=tool_name,
        baseline_artifact_text=baseline_description,
        saturation_threshold=saturation_threshold,
        min_iters=min_iters,
        window_size=window_size,
        gate_mode=gate_mode,
    )


def _load_behavioral_examples_from_suite(suite_path: Path) -> list:
    """Build behavioral dspy.Examples from a suite file. Local-import path
    so the validation stack isn't pulled in unless the user opted in."""
    from evolution.core.behavioral_example import build_behavioral_examples
    from evolution.validation.task import TaskSuite

    return build_behavioral_examples(TaskSuite.from_jsonl(suite_path))


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
    eval_source: str = "synthetic",
    eval_dataset_size: int = 150,
    holdout_ratio: float = 0.5,
    enable_confusable_bucket: bool = False,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
    optimizer_model: Optional[str] = None,
    reflection_model: Optional[str] = None,
    eval_model: Optional[str] = None,
    max_total_cost_usd: Optional[float] = None,
    benchmark_cmd: Optional[str] = None,
    benchmark_timeout_seconds: int = 600,
    closed_loop_suite_path: Optional[Path] = None,
    closed_loop_hermes_repo: Optional[Path] = None,
    closed_loop_saturation_threshold: float = 0.95,
    closed_loop_min_iters: int = 3,
    closed_loop_window_size: int = 8,
    closed_loop_mode: str = "feedback",
    closed_loop_in_valset: bool = False,
    closed_loop_agent_model: Optional[str] = None,
    closed_loop_task_timeout_seconds: Optional[int] = None,
    skip_preflight: bool = False,
    skip_cost_suggest: bool = False,
    skip_saturation_check: bool = False,
    force_saturation_check: bool = False,
    gepa_minibatch_size: int = 3,
) -> dict[str, Any]:
    """Evolve one tool description inside a manifest.

    Returns a dict summary of the run (mirrors metrics.json).
    """
    # Absolute: downstream sources join relative paths against their root,
    # which here is ``manifest_path.parent`` and would double-prefix.
    manifest_path = Path(manifest_path).resolve()
    source = _resolve_source(manifest_path)
    manifest = source.find_manifest(manifest_path)
    if manifest is None:
        raise ValueError(
            f"ToolSource {source.name!r} accepted {manifest_path} via supports() "
            f"but find_manifest returned None"
        )
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
        enable_confusable_bucket=enable_confusable_bucket,
        reflection_minibatch_size=gepa_minibatch_size,
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
        try:
            register_litellm_failure_callback()
            register_litellm_cost_callback()
            COST_LEDGER.reset()
            COST_LEDGER.set_ceiling(max_total_cost_usd)
            console.print(f"  Run log: {run_log_path}")
            if max_total_cost_usd is not None:
                console.print(f"  Cost ceiling: ${max_total_cost_usd:.4f}")

            # Validate credentials before any LM work — dataset gen alone
            # can spend $0.50+ before we'd otherwise discover a stale token.
            # Resolve up front so the cost advisor below can reuse the same
            # ResolvedLM without re-walking config + auth.json.
            if not dry_run:
                _preflight_optimizer = resolve_default_lm(role="optimizer", explicit_model=optimizer_model)
                _preflight_eval = resolve_default_lm(role="eval", explicit_model=eval_model)
                if not skip_preflight:
                    _preflight_lm_credentials([_preflight_optimizer, _preflight_eval])
                # Cost advisor: only fire when the user inherited the eval
                # model from Hermes (eval_model is None) AND the resolver
                # returned a stock LM. Custom factory paths (Codex) route
                # to a closed ChatGPT-subscription endpoint where suggesting
                # "use openai/gpt-5-nano" implies a different auth setup
                # the user didn't opt in to.
                if (
                    not skip_cost_suggest
                    and eval_model is None
                    and _preflight_eval.lm_factory is None
                ):
                    _alt = _find_cheaper_alternative(_preflight_eval.model)
                    if _alt is not None:
                        console.print(_render_cost_suggestion_panel("eval", _alt))

            sessiondb_drops: Optional[dict[str, int]] = None
            if eval_source == "synthetic":
                console.print(f"\n[bold]Building tool-selection eval dataset[/bold] (synthetic, three buckets)")
                if dry_run:
                    # Synthetic dataset gen is itself an LM call; --dry-run skips it.
                    # The "would generate N" line mirrors the skill-path dry-run shape.
                    console.print(
                        f"\n[bold green]DRY RUN — would generate {config.eval_dataset_size} "
                        f"synthetic examples; skipping LM dataset gen + GEPA.[/bold green]"
                    )
                    return {"decision": "dry-run", "eval_source": "synthetic"}
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
            elif eval_source == "sessiondb":
                console.print(
                    f"\n[bold]Building tool-selection eval dataset[/bold] (sessiondb, Hermes only)"
                )
                console.print(
                    "  [dim]Claude Code and Copilot logs don't carry tool-call data — only Hermes "
                    "session JSON is mined.[/dim]"
                )
                if dry_run:
                    # Importer is free (no LM calls); judge is the LM-spending stage.
                    # Run only the importer so the operator sees real candidate counts
                    # and the per-invoked-tool distribution before paying for the judge.
                    candidates, importer_drops = HermesToolImporter.extract_candidates(
                        manifest=manifest,
                        limit=config.eval_dataset_size * 2,
                    )
                    tool_counts: dict[str, int] = {}
                    for cand in candidates:
                        name = cand["invoked_tool"]
                        tool_counts[name] = tool_counts.get(name, 0) + 1
                    console.print(
                        f"\n[bold green]DRY RUN — importer surfaced {len(candidates)} candidates; "
                        f"skipping judge + GEPA.[/bold green]"
                    )
                    console.print(f"  Importer drops: {importer_drops}")
                    if tool_counts:
                        console.print(f"  Invoked-tool distribution: {tool_counts}")
                    return {
                        "decision": "dry-run",
                        "eval_source": "sessiondb",
                        "candidate_count": len(candidates),
                        "importer_drops": importer_drops,
                        "invoked_tool_distribution": tool_counts,
                    }
                dataset, sessiondb_drops = build_tool_dataset_from_sessions(
                    manifest=manifest,
                    target_tool=tool_name,
                    output_path=Path("datasets") / "tools" / tool_name,
                    model=eval_model,
                    max_examples=config.eval_dataset_size,
                    seed=config.seed,
                )
                non_manifest = sessiondb_drops.get("non_manifest", 0)
                if non_manifest:
                    console.print(
                        f"  [yellow]Dropped {non_manifest} session invocations of tools "
                        f"not in this manifest.[/yellow]"
                    )
                if not dataset.all_examples:
                    console.print(
                        "[red]✗ Session mining produced 0 usable examples. "
                        "Drop breakdown: " + ", ".join(f"{k}={v}" for k, v in sessiondb_drops.items() if v) +
                        ". Try --eval-source synthetic.[/red]"
                    )
                    sys.exit(1)
            else:
                raise ValueError(f"unknown eval_source: {eval_source!r}")

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

            # Guard: GEPA's reflective batch sampler asserts
            # len(trainset) >= reflection_minibatch_size mid-optimization
            # (gepa/strategies/batch_sampler.py). Catch the misconfiguration
            # at startup with an actionable message instead.
            if config.reflection_minibatch_size > len(dataset.train):
                console.print(
                    f"[red]✗ --gepa-minibatch-size={config.reflection_minibatch_size} "
                    f"exceeds trainset size {len(dataset.train)}. Pick a value ≤ "
                    f"{len(dataset.train)} or increase --eval-dataset-size.[/red]"
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

            _eval_lm = resolve_default_lm(role="eval", explicit_model=eval_model)
            lm = instantiate_lm(_eval_lm, request_timeout=60, num_retries=5)
            dspy.configure(
                lm=lm,
                warn_on_type_mismatch=False,
                callbacks=[LMTimingCallback()],
            )

            judge = ToolJudge(config)
            # In modes that route behavioral examples through the metric for
            # selection-affecting scoring, the saturation gate would defeat
            # the purpose — every novel candidate must score every time.
            cache_gate_mode = (
                "always" if closed_loop_mode in ("trainset", "both") else "sampled"
            )
            closed_loop_cache = _maybe_build_closed_loop_cache(
                tool_name=tool_name,
                baseline_description=baseline_description,
                suite_path=closed_loop_suite_path,
                hermes_repo=closed_loop_hermes_repo,
                saturation_threshold=closed_loop_saturation_threshold,
                min_iters=closed_loop_min_iters,
                window_size=closed_loop_window_size,
                gate_mode=cache_gate_mode,
                agent_model=closed_loop_agent_model,
                agent_timeout_seconds=closed_loop_task_timeout_seconds,
            )
            metric = make_tool_fitness_metric(
                judge=judge,
                baseline_description=baseline_description,
                manifest=manifest,
                target_tool_name=tool_name,
                max_growth=config.growth_free_threshold,
                text_extractor=lambda predictor: _description_from_predictor(predictor, tool_name),
                closed_loop_cache=closed_loop_cache,
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

            # Behavioral-example injection: each closed-loop task becomes an
            # additional dspy.Example whose score contributes to GEPA's
            # sum(minibatch_scores) acceptance — behavioral wins can break
            # judge ties on saturated baselines.
            if closed_loop_mode in ("trainset", "both"):
                if closed_loop_suite_path is None:
                    raise ValueError(
                        f"--closed-loop-mode={closed_loop_mode} requires "
                        "--closed-loop-during-evolution to be set"
                    )
                behavioral_examples = _load_behavioral_examples_from_suite(
                    closed_loop_suite_path
                )
                trainset = trainset + behavioral_examples
                if closed_loop_in_valset:
                    valset = valset + behavioral_examples

            cached_baseline_holdout_per_example = None
            if not skip_saturation_check:
                holdout_examples_for_preflight = _build_examples(
                    dataset.holdout, for_module=True
                )
                sat_report = saturation_preflight(
                    baseline_module=baseline_module,
                    holdout_examples=holdout_examples_for_preflight,
                    metric=metric,
                    lm=lm,
                    closed_loop_cache=closed_loop_cache,
                    baseline_artifact_text=baseline_description,
                )
                if sat_report.band != "healthy":
                    render_saturation_panel(sat_report, console=console)
                    if not force_saturation_check:
                        if is_non_interactive():
                            console.print(
                                "[yellow]Non-interactive context; refusing to "
                                "proceed. Pass --force-saturation-check to "
                                "override.[/yellow]"
                            )
                            # Exit code 3 distinguishes "refused to run for
                            # lack of a TTY to confirm against" from clean
                            # success (0) or hard user errors (1). Lets a
                            # wrapping CI / cron / scheduled runner detect
                            # silent denial.
                            sys.exit(3)
                        if not interactive_confirm():
                            console.print("[yellow]Aborted by user.[/yellow]")
                            sys.exit(0)
                else:
                    render_saturation_panel(sat_report, console=console)
                cached_baseline_holdout_per_example = sat_report.holdout_per_example

            console.print(f"\n[bold cyan]Running GEPA optimization (max_full_evals={iterations})[/bold cyan]\n")
            start_time = time.time()

            _reflection_lm = resolve_default_lm(
                role="reflection",
                explicit_model=reflection_model or optimizer_model,
            )
            reflection_lm = instantiate_lm(
                _reflection_lm,
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
                reflection_minibatch_size=config.reflection_minibatch_size,
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
                "resolved_lms": resolved_lms_dump(
                    optimizer=optimizer_model,
                    reflection=config.reflection_model,
                    eval_=config.eval_model,
                ),
                "eval_dataset_size": config.eval_dataset_size,
                "holdout_ratio": config.holdout_ratio,
                "quality_gate_preset": quality_gate,
                "fitness_profile": fitness_profile,
                "enable_confusable_bucket": config.enable_confusable_bucket,
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
                    "dataset": _dataset_payload(dataset, dropped_tools=manifest.dropped_tools, sessiondb_drops=sessiondb_drops),
                    "run_inputs": run_inputs,
                    **tool_payload_fields,
                })
                console.print(f"  Saved failed variant to {failed_path}")
                return {"decision": "reject", "reason": "static_constraint_failure"}

            console.print(
                f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]"
            )
            holdout_examples = _build_examples(dataset.holdout, for_module=True)
            if cached_baseline_holdout_per_example is not None:
                baseline_per_example = cached_baseline_holdout_per_example
                avg_baseline = sum(baseline_per_example) / len(baseline_per_example)
            else:
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

            # Write manifests before the hook so it can reference them via
            # $EVOLVED_PATH / $BASELINE_PATH.
            benchmark_block: Optional[dict[str, Any]] = None
            if growth_pass and benchmark_cmd is not None:
                evolved_manifest_for_hook = manifest.replace_description(tool_name, evolved_description)
                evolved_manifest_path = output_dir / "evolved_manifest.json"
                baseline_manifest_path = output_dir / "baseline_manifest.json"
                evolved_manifest_path.write_text(
                    json.dumps(_manifest_to_dict(evolved_manifest_for_hook), indent=2) + "\n"
                )
                baseline_manifest_path.write_text(
                    json.dumps(_manifest_to_dict(manifest), indent=2) + "\n"
                )
                benchmark_block = run_benchmark_hook(
                    benchmark_cmd,
                    timeout_seconds=benchmark_timeout_seconds,
                    evolved_path=evolved_manifest_path,
                    baseline_path=baseline_manifest_path,
                    output_dir=output_dir,
                    target_name=tool_name,
                    artifact_type="tool_description",
                )
                if not benchmark_block["passed"]:
                    growth_pass = False
                    evolved_manifest_path.unlink(missing_ok=True)
                    baseline_manifest_path.unlink(missing_ok=True)

            baseline_chars = len(baseline_description)
            evolved_chars = len(evolved_description)
            growth_pct = (evolved_chars - baseline_chars) / max(1, baseline_chars)
            required_improvement = max(
                0.0,
                config.growth_quality_slope * (growth_pct - config.growth_free_threshold),
            )
            decision_rule_used = resolve_decision_rule(config, growth_pct)
            if growth_pass:
                decision_reason = "passed"
            elif benchmark_block is not None and not benchmark_block["passed"]:
                decision_reason = "benchmark_failed"
            else:
                decision_reason = "growth_quality_gate"
            decision_payload = {
                "schema_version": "4",
                "decision": "deploy" if growth_pass else "reject",
                "reason": decision_reason,
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
                "dataset": _dataset_payload(dataset, dropped_tools=manifest.dropped_tools, sessiondb_drops=sessiondb_drops),
                "run_inputs": run_inputs,
                **tool_payload_fields,
            }
            if benchmark_block is not None:
                decision_payload["benchmark"] = benchmark_block
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
                "resolved_lms": resolved_lms_dump(
                    optimizer=optimizer_model, eval_=eval_model
                ),
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
                source.apply_evolved(
                    source_path=manifest_path,
                    evolved_manifest=evolved_manifest,
                    target_tool=tool_name,
                    new_description=evolved_description,
                )
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
        except CostCeilingExceeded as exc:
            # Abort may fire before `run_inputs` is built in the deploy path;
            # fall back to a minimal equivalent so gate_decision.json is still useful.
            run_inputs_for_abort = locals().get("run_inputs") or {
                "seed": config.seed,
                "iterations": iterations,
                "optimizer_model": optimizer_model,
                "reflection_model": config.reflection_model,
                "eval_model": config.eval_model,
                "resolved_lms": resolved_lms_dump(
                    optimizer=optimizer_model,
                    reflection=config.reflection_model,
                    eval_=config.eval_model,
                ),
                "eval_dataset_size": config.eval_dataset_size,
                "holdout_ratio": config.holdout_ratio,
                "quality_gate_preset": quality_gate,
                "fitness_profile": fitness_profile,
                "eval_source": eval_source,
            }
            write_cost_ceiling_abort(
                exc,
                output_dir=output_dir,
                run_inputs=run_inputs_for_abort,
                extra_fields={
                    "artifact_type": "tool_description",
                    "target_tool": tool_name,
                },
            )
            return {"decision": "aborted", "reason": "cost_ceiling_exceeded"}
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


@click.command()
@click.option("--tool", "tool_name", required=True, help="Name of the tool to evolve")
@click.option(
    "--manifest",
    "manifest_path",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=Path),
    help=(
        "Path to either an MCP-list_tools-shaped manifest JSON file or a "
        "directory of Python source files containing Hermes-style "
        "``*_SCHEMA`` declarations."
    ),
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
@click.option(
    "--enable-confusable-bucket",
    "enable_confusable_bucket",
    is_flag=True,
    default=False,
    help=(
        "Allocate 30% of the synthetic eval dataset to the "
        "confusable_neighbor bucket. Off by default; when off, the share "
        "rolls into target_correct."
    ),
)
@click.option(
    "--eval-source",
    default="synthetic",
    type=click.Choice(["synthetic", "sessiondb"]),
    help=(
        "Where the eval dataset comes from. 'synthetic' (default) generates "
        "tasks via the three-bucket synthetic generator. 'sessiondb' mines "
        "real Hermes session logs (~/.hermes/sessions/) for "
        "(task, invoked_tool) pairs and re-judges each pair against the "
        "current manifest. Claude Code and Copilot logs don't carry tool-call "
        "data and aren't mined."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help=(
        "Build the eval dataset and stop. Useful for confirming sessiondb "
        "discovery before spending judge + GEPA budget on a full run."
    ),
)
@click.option(
    "--max-total-cost-usd",
    default=None,
    type=click.FloatRange(min=0.0),
    help="Safety net: abort the run cleanly when cumulative LM cost (across "
         "dataset gen, GEPA, holdout eval, and any sessiondb judge calls) "
         "exceeds this dollar amount. Worst-case overshoot is one LM call "
         "past the ceiling. 0 is accepted (aborts on first call, useful for "
         "testing). Negative values rejected. Off by default.",
)
@click.option(
    "--benchmark-cmd",
    default=None,
    type=str,
    help="Deploy-gate hook: shell command run AFTER the framework's own "
         "deploy gate passes; nonzero exit flips the decision to reject "
         "with reason='benchmark_failed'. Receives EVOLVED_PATH, "
         "BASELINE_PATH, RUN_DIR, TARGET_NAME, ARTIFACT_TYPE via env. Runs "
         "under /bin/sh -c with shell=True (your shell, your command — "
         "interactive aliases are not available). Trust boundary: the "
         "command string is yours; do not pass strings you didn't write.",
)
@click.option(
    "--benchmark-timeout-seconds",
    default=600,
    type=click.IntRange(min=1),
    help="Wall-clock cap for the --benchmark-cmd hook (default 600s).",
)
@click.option(
    "--closed-loop-during-evolution",
    "closed_loop_suite_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a JSONL task suite (same shape as evolution/validation/suites/*.jsonl). "
         "When set, the framework runs the closed-loop validator on saturating GEPA "
         "iterations and appends a [CLOSED_LOOP] block to the reflection LM's feedback "
         "input. Held-out from training tasks (no overlap-detection enforcement). "
         "Requires --closed-loop-hermes-repo.",
)
@click.option(
    "--closed-loop-hermes-repo",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to the hermes-agent checkout the closed-loop validator should mutate "
         "in place. Required iff --closed-loop-during-evolution is set.",
)
@click.option(
    "--closed-loop-saturation-threshold",
    default=0.95,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Min judge score over the recent window for the saturation gate to open "
         "(default 0.95).",
)
@click.option(
    "--closed-loop-min-iters",
    default=3,
    type=click.IntRange(min=1),
    help="Periodic-fire floor: fire closed-loop at least every N reflective "
         "iterations even when the judge isn't saturating (default 3).",
)
@click.option(
    "--closed-loop-window-size",
    default=8,
    type=click.IntRange(min=1),
    help="Number of recent judge scores the saturation gate inspects (default 8).",
)
@click.option(
    "--closed-loop-mode",
    default="feedback",
    type=click.Choice(["feedback", "trainset", "both"]),
    help="How closed-loop signal participates in GEPA. 'feedback' (default) "
         "appends a [CLOSED_LOOP] block to the reflection LM's input — "
         "proposal-prompt signal only, no acceptance change. 'trainset' adds "
         "behavioral dspy.Examples to the training set whose score (binary "
         "pass/fail from the closed-loop validator) contributes to GEPA's "
         "sum(minibatch_scores) acceptance — lets behavioral wins break judge "
         "ties on saturated baselines. 'both' does trainset + the [CLOSED_LOOP] "
         "feedback block on non-behavioral examples (most expensive).",
)
@click.option(
    "--no-preflight",
    "skip_preflight",
    is_flag=True,
    default=False,
    help="Skip the LM credential preflight probe. By default, the framework "
         "makes one tiny ~$0.0001 litellm.completion call per unique LM "
         "before GEPA setup to validate credentials work — this catches "
         "expired tokens up front rather than 5 minutes into a run. Pass "
         "this flag to skip when you know your creds are good.",
)
@click.option(
    "--no-cost-suggest",
    "skip_cost_suggest",
    is_flag=True,
    default=False,
    help="Skip the post-preflight cost-suggestion panel. By default, when "
         "--eval-model is unset, the framework checks litellm.model_cost "
         "for a cheaper same-provider model with sufficient context window "
         "and prints a Rich panel with a paste-ready --eval-model flag. "
         "Pass this to suppress the panel.",
)
@click.option(
    "--no-saturation-check",
    "skip_saturation_check",
    is_flag=True,
    default=False,
    help="Skip the saturation pre-flight. By default, the framework "
         "scores the baseline on the holdout (and the closed-loop suite, "
         "if --closed-loop-during-evolution is set) BEFORE GEPA starts "
         "and refuses to spend on a saturated target. Pass this to skip "
         "(useful when you've already validated headroom externally).",
)
@click.option(
    "--force-saturation-check",
    "force_saturation_check",
    is_flag=True,
    default=False,
    help="Run the saturation pre-flight, render the panel, but proceed "
         "regardless of band. Required to override a non-healthy verdict "
         "in non-interactive contexts (no TTY). Without this in such a "
         "context, the framework exits cleanly without spending GEPA budget.",
)
@click.option(
    "--gepa-minibatch-size",
    "gepa_minibatch_size",
    default=3,
    type=click.IntRange(min=1),
    help="GEPA's reflective minibatch size — number of training examples "
         "sampled per reflective step for the sum() acceptance gate. "
         "Default 3 matches GEPA's own default. Bump to ~8 when the "
         "saturation pre-flight flags the weak_signal band: the wider "
         "sampling window makes discriminating examples appear in "
         "~68% of minibatches vs ~34% at default. Trade-off: larger "
         "minibatch means each accepted proposal consumes more of the "
         "metric-call budget, so consider also bumping --iterations to "
         "~10 to preserve the proposal count. Aborts at startup if the "
         "value exceeds the trainset size.",
)
@click.option(
    "--closed-loop-in-valset/--no-closed-loop-in-valset",
    "closed_loop_in_valset",
    default=False,
    help="When --closed-loop-mode is trainset or both, also include behavioral "
         "examples in the valset (adds them to the Pareto frontier + holdout "
         "scoring). Costs more — each accepted candidate triggers another full "
         "eval pass over the behavioral examples. Default off.",
)
@click.option(
    "--closed-loop-agent-model",
    "closed_loop_agent_model",
    default=None,
    type=str,
    help="Override the agent model the closed-loop validator runs `hermes -z` "
         "with (passed as `hermes -m MODEL -z ...`). When unset, the validator "
         "uses whatever's in your ~/.hermes/config.yaml. Useful when your "
         "daily-driver Hermes model saturates the planted-bug suite at 100%, "
         "hiding the behavioral signal — run validation against a weaker model "
         "without touching your config.",
)
@click.option(
    "--closed-loop-task-timeout-seconds",
    "closed_loop_task_timeout_seconds",
    default=None,
    type=click.IntRange(min=1),
    help="Per-task wall-clock budget for the closed-loop validator's `hermes -z` "
         "subprocess (default 120s). Bump when --closed-loop-agent-model selects "
         "a slow reasoning model that doesn't finish within the default — most "
         "OpenAI reasoning models (o1-family, o3-family) take 60-180s per "
         "debugging task. Hitting the timeout abstains the task verdict rather "
         "than failing it, so over-tight values silently produce no-signal runs.",
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
    enable_confusable_bucket: bool,
    eval_source: str,
    dry_run: bool,
    max_total_cost_usd: Optional[float],
    benchmark_cmd: Optional[str],
    benchmark_timeout_seconds: int,
    skip_preflight: bool,
    skip_cost_suggest: bool,
    skip_saturation_check: bool,
    force_saturation_check: bool,
    gepa_minibatch_size: int,
    closed_loop_suite_path: Optional[Path],
    closed_loop_hermes_repo: Optional[Path],
    closed_loop_saturation_threshold: float,
    closed_loop_min_iters: int,
    closed_loop_window_size: int,
    closed_loop_mode: str,
    closed_loop_in_valset: bool,
    closed_loop_agent_model: Optional[str],
    closed_loop_task_timeout_seconds: Optional[int],
) -> None:
    """Evolve one tool description in an MCP manifest using DSPy + GEPA."""
    if apply_flag and patch_flag:
        raise click.UsageError("--apply and --patch are mutually exclusive")
    if closed_loop_suite_path is not None and closed_loop_hermes_repo is None:
        raise click.UsageError(
            "--closed-loop-during-evolution requires --closed-loop-hermes-repo"
        )
    if closed_loop_mode != "feedback" and closed_loop_suite_path is None:
        raise click.UsageError(
            f"--closed-loop-mode={closed_loop_mode} requires "
            "--closed-loop-during-evolution to be set"
        )
    try:
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
            enable_confusable_bucket=enable_confusable_bucket,
            eval_source=eval_source,
            dry_run=dry_run,
            max_total_cost_usd=max_total_cost_usd,
            benchmark_cmd=benchmark_cmd,
            benchmark_timeout_seconds=benchmark_timeout_seconds,
            closed_loop_suite_path=closed_loop_suite_path,
            closed_loop_hermes_repo=closed_loop_hermes_repo,
            closed_loop_saturation_threshold=closed_loop_saturation_threshold,
            closed_loop_min_iters=closed_loop_min_iters,
            closed_loop_window_size=closed_loop_window_size,
            closed_loop_mode=closed_loop_mode,
            closed_loop_in_valset=closed_loop_in_valset,
            closed_loop_agent_model=closed_loop_agent_model,
            closed_loop_task_timeout_seconds=closed_loop_task_timeout_seconds,
            skip_preflight=skip_preflight,
            skip_cost_suggest=skip_cost_suggest,
            skip_saturation_check=skip_saturation_check,
            force_saturation_check=force_saturation_check,
            gepa_minibatch_size=gepa_minibatch_size,
        )
    except HermesProviderError as exc:
        # Render a clean error panel instead of dumping a Python traceback —
        # auth failures contain actionable per-provider recovery commands.
        console.print(Panel(str(exc), title="[bold]Authentication[/bold]", border_style="red"))
        sys.exit(2)


if __name__ == "__main__":
    main()

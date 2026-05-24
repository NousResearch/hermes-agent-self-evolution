"""Evolve an agent skill using DSPy + GEPA.

Usage:
    python -m evolution.skills.evolve_skill --skill github-code-review --iterations 10
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset datasets/skills/arxiv/
"""

import difflib
import json
import logging
import random
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
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
from evolution.core.quality_gate import (
    QUALITY_GATE_PRESETS,
    _check_cl_primary_gate,
    append_cl_decision_fields,
    resolve_proposer_mode,
    run_benchmark_hook,
    write_cost_ceiling_abort,
    write_gate_decision,
)
from evolution.core.run_inputs import build_run_inputs
from evolution.core.skill_sources import discover_skill_sources

# Without this, the BudgetAwareProposer + LMTimingCallback logs stay
# invisible: Python's root logger defaults to WARNING when unconfigured.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, GoldenDatasetLoader
from evolution.core.external_importers import build_dataset_from_external
from evolution.core.stats import paired_bootstrap
from evolution.core.fitness import LLMJudge, make_skill_fitness_metric
from evolution.core.constraints import (
    ConstraintResult,
    ConstraintValidator,
    effective_absolute_char_ceiling,
    resolve_decision_rule,
)
from evolution.core.lm_timing_callback import (
    COST_LEDGER,
    CostCeilingExceeded,
    LMTimingCallback,
    register_litellm_cost_callback,
    register_litellm_failure_callback,
)
from evolution.skills.budget_aware_proposer import BudgetAwareProposer, ProposerMode
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)
from evolution.skills.knee_point import select_knee_point, CandidatePick

console = Console()


_BUDGET_BY_ITERATIONS = {1: "light", 2: "medium", 3: "heavy"}


# Back-compat aliases so existing tests and external imports that reference
# the underscored names continue to work after the move to evolution.core.
_QUALITY_GATE_PRESETS = QUALITY_GATE_PRESETS
_resolve_proposer_mode = resolve_proposer_mode
_write_gate_decision = write_gate_decision


def _dataset_payload(dataset: EvalDataset) -> dict[str, Any]:
    """Serialize dataset composition for gate_decision.json.

    Records per-source counts (e.g. synthetic, sessiondb_*, golden) so a
    future calibration script can ask "is mined-source dominance correlated
    with deploy rate?" without re-running every PR. Source field is on
    each EvalExample; we just bucket by it.
    """
    sources: dict[str, int] = {}
    for ex in dataset.all_examples:
        src = ex.source or "unknown"
        sources[src] = sources.get(src, 0) + 1
    return {
        "size_total": len(dataset.all_examples),
        "size_train": len(dataset.train),
        "size_val": len(dataset.val),
        "size_holdout": len(dataset.holdout),
        "sources": sources,
    }


def _compute_win_loss(
    baseline_per_example: list[float],
    evolved_per_example: list[float],
) -> dict[str, Any]:
    """Decompose per-example deltas into a win/loss summary.

    The aggregate mean lift hides operational risk: a variant scoring +0.10
    on 60% of examples and -0.04 on 40% nets the same mean as a variant
    scoring +0.04 on every example, but the first carries a 40% regression
    rate. This block surfaces that distinction in gate_decision.json.
    """
    deltas = [e - b for b, e in zip(baseline_per_example, evolved_per_example)]
    return {
        "n_wins": sum(1 for d in deltas if d > 0),
        "n_losses": sum(1 for d in deltas if d < 0),
        "n_ties": sum(1 for d in deltas if d == 0),
        "worst_regression": min(deltas) if deltas else 0.0,
        "worst_improvement": max(deltas) if deltas else 0.0,
    }


def _knee_point_payload(knee_pick: Optional[CandidatePick]) -> dict[str, Any]:
    """Serialize a CandidatePick (or its absence) for gate_decision.json.

    `applied: false` lands when MIPROv2 fallback fired (no detailed_results)
    so a calibration script can distinguish "knee-point ran" from "knee-point
    skipped" without checking field presence.
    """
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


def _deferred_knee_point_payload(
    *, best_idx: int, val_score: float, body_chars: int,
) -> dict[str, Any]:
    """Payload for the val-best path that defers to GEPA's best_idx.

    Regenerated calibration showed the epsilon-band selector picked
    GEPA's default in every run across five epsilon modes; the val-best
    short-circuit skips the band walk entirely. `band_roster` stays a
    list so downstream calibration scripts that access it via
    ``.get("band_roster", [])`` keep working.
    """
    return {
        "applied": False,
        "fallback": "gepa_default",
        "picked_idx": best_idx,
        "gepa_default_idx": best_idx,
        "picked_val_score": val_score,
        "picked_body_chars": body_chars,
        "gepa_default_body_chars": body_chars,
        "band_roster": [],
    }


def _holdout_evaluate_with_metric(module, holdout_examples, metric, lm) -> tuple[float, list[float]]:
    """Score `module` on the holdout via dspy.Evaluate.

    The GEPA-shaped metric takes 5 positional args; dspy.Evaluate calls
    metric(example, prediction). Wrap it.

    Returns (mean_score, per_example_scores). Per-example scores feed
    the bootstrap CI in the deploy gate.
    """
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
    # dspy.Evaluate returns EvaluationResult(score=mean*100, results=[(ex,
    # pred, score), ...]) — per-example scores in devset order
    # (evaluate.py:179: zip(devset, results, strict=False)).
    mean = float(result.score) / 100.0
    per_example = [float(s) for _, _, s in result.results]
    return mean, per_example


_BAND_HOLDOUT_SUBSAMPLE_CAP = 100


def _evaluate_band_on_holdout(
    *,
    knee_pick: CandidatePick,
    candidates: list,
    holdout_examples: list,
    metric: Any,
    lm: Any,
    output_dir: Path,
    seed: int,
    subsample_cap: int = _BAND_HOLDOUT_SUBSAMPLE_CAP,
) -> Path:
    """Re-evaluate every candidate in the knee-point band on the holdout.

    Inline because GEPA's candidate programs are not persisted: a true
    post-run script can read `band_roster` from `gate_decision.json` but
    cannot reach the candidate programs once `evolve()` returns. This
    runs while the GEPA `details.candidates` list is still alive.

    Caps the holdout slice at `subsample_cap` examples (deterministic via
    `seed`) to bound cost when callers crank `--eval-dataset-size` to 400.
    Each candidate sees the same subsample so per-candidate scores are
    directly comparable.
    """
    if len(holdout_examples) > subsample_cap:
        rng = random.Random(seed)
        eval_examples = rng.sample(holdout_examples, subsample_cap)
    else:
        eval_examples = holdout_examples

    candidate_results: list[dict[str, Any]] = []
    for entry in knee_pick.band_roster:
        idx = entry["idx"]
        candidate_module = candidates[idx]
        holdout_score, holdout_per_example = _holdout_evaluate_with_metric(
            candidate_module, eval_examples, metric, lm,
        )
        candidate_results.append({
            "idx": idx,
            "val_score": entry["val_score"],
            "body_chars": entry["body_chars"],
            "holdout_score": holdout_score,
            "holdout_per_example": holdout_per_example,
        })

    payload = {
        "epsilon": knee_pick.epsilon,
        "holdout_subsample_size": len(eval_examples),
        "candidates": candidate_results,
    }
    path = output_dir / "band_holdout.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _resolve_budget(iterations: int, budget: Optional[str]) -> str:
    """Pick the GEPA budget. Explicit `budget` always wins.

    `iterations` is the legacy CLI knob: only the values 1/2/3 carry a
    meaningful mapping; anything else collapses to "light". Callers
    should prefer `--budget` and treat `--iterations` as deprecated.
    """
    if budget is not None:
        return budget
    return _BUDGET_BY_ITERATIONS.get(iterations, "light")


def _default_gepa_runner(
    *,
    baseline_module: SkillModule,
    trainset: list,
    valset: list,
    metric,
    gepa_budget: str,
    optimizer_model: str,
    seed: int,
    instruction_proposer=None,
    reflection_model: Optional[str] = None,
    reflection_minibatch_size: int = 3,
):
    # max_tokens=32000 satisfies DSPy's reasoning-model floor of 16000
    # (DSPy raises ValueError below that).
    reflection_lm_model = reflection_model or optimizer_model
    _reflection_lm = resolve_default_lm(role="reflection", explicit_model=reflection_lm_model)
    optimizer = dspy.GEPA(
        metric=metric,
        auto=gepa_budget,
        # cache=False because at temperature=1.0 the disk cache would
        # replay stale mutations across runs and shrink candidate diversity.
        reflection_lm=instantiate_lm(
            _reflection_lm,
            temperature=1.0,
            max_tokens=32000,
            cache=False,
            # 300s ≈ 3x the longest observed legitimate gpt-5-mini call.
            # num_retries=2 caps worst-case wall at 10min — preferable to
            # the silent 30-80min stalls we saw without bounded retries.
            # The TimeoutError surfaces at _build_optimizer_and_compile
            # and triggers the MIPROv2 fallback.
            request_timeout=300,
            num_retries=2,
        ),
        seed=seed,
        # Required for knee-point selection: exposes DspyGEPAResult
        # (.candidates, .val_aggregate_scores) on the returned module.
        track_stats=True,
        instruction_proposer=instruction_proposer,
        reflection_minibatch_size=reflection_minibatch_size,
    )
    return optimizer.compile(baseline_module, trainset=trainset, valset=valset)


def _default_mipro_runner(
    *,
    baseline_module: SkillModule,
    trainset: list,
    metric,
    seed: int,
):
    # MIPROv2 expects float-returning metrics; the GEPA-shaped one returns
    # dspy.Prediction(score, feedback).
    def float_metric(*args, **kwargs):
        result = metric(*args, **kwargs)
        return float(getattr(result, "score", result))

    optimizer = dspy.MIPROv2(
        metric=float_metric,
        auto="light",
        init_temperature=0.5,
        seed=seed,
    )
    return optimizer.compile(baseline_module, trainset=trainset)


def _print_fallback_banner(exc: Exception, failure_log_path: Optional[Path]) -> None:
    tb = traceback.format_exc()
    if failure_log_path is not None:
        failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        failure_log_path.write_text(f"{type(exc).__name__}: {exc}\n\n{tb}")
        location_line = f"Full traceback: {failure_log_path}"
    else:
        location_line = "Re-run with --no-fallback to surface GEPA's traceback."

    console.print(Panel(
        f"[bold]GEPA failed:[/bold] {type(exc).__name__}: {exc}\n\n"
        f"Falling back to MIPROv2.\n"
        f"{location_line}",
        title="[bold yellow]GEPA fallback[/bold yellow]",
        border_style="red",
    ))


def _build_optimizer_and_compile(
    *,
    baseline_module: SkillModule,
    trainset: list,
    valset: list,
    metric,
    gepa_budget: str,
    optimizer_model: str,
    seed: int,
    no_fallback: bool,
    failure_log_path: Optional[Path] = None,
    instruction_proposer=None,
    reflection_model: Optional[str] = None,
    reflection_minibatch_size: int = 3,
    _gepa_runner=_default_gepa_runner,
    _mipro_runner=_default_mipro_runner,
):
    """Run GEPA; fall back to MIPROv2 on failure unless `no_fallback`.

    Returns `(optimized_module, optimizer_name)`. ImportError from the
    MIPROv2 path (raised lazily inside MIPROv2.compile when optuna is
    missing) is re-raised with the GEPA failure preserved as `__cause__`
    so the user keeps both diagnostics.
    """
    try:
        optimized = _gepa_runner(
            baseline_module=baseline_module,
            trainset=trainset,
            valset=valset,
            metric=metric,
            gepa_budget=gepa_budget,
            optimizer_model=optimizer_model,
            seed=seed,
            instruction_proposer=instruction_proposer,
            reflection_model=reflection_model,
            reflection_minibatch_size=reflection_minibatch_size,
        )
        return optimized, "GEPA"
    except CostCeilingExceeded:
        # Don't fall back to MIPROv2 — that would re-incur cost and defeat the kill switch.
        raise
    except HermesProviderError:
        # Defensive: HermesProviderError is BaseException-derived so the
        # broad `except Exception` below already wouldn't catch it; this
        # explicit re-raise documents intent and guards against someone
        # changing the inheritance back to RuntimeError. Auth failures
        # don't get fixed by switching to MIPROv2 — same creds, same fail.
        raise
    except Exception as gepa_exc:
        if no_fallback:
            raise
        _print_fallback_banner(gepa_exc, failure_log_path)
        try:
            optimized = _mipro_runner(
                baseline_module=baseline_module,
                trainset=trainset,
                metric=metric,
                seed=seed,
            )
            return optimized, "MIPROv2"
        except ImportError as ie:
            console.print(
                "[red]✗ MIPROv2 fallback requires the [miprov2] extra. "
                "Install with: uv pip install 'agent-self-evolution[miprov2]'[/red]"
            )
            raise ie from gepa_exc


_BAP_SAFETY_MARGIN_DEFAULT = 0.10


def _resolve_bap_safety_margin(value: Optional[float]) -> float:
    """Resolve `--bap-safety-margin` to BudgetAwareProposer's `safety_margin`.

    `None` (sentinel: "user didn't set the flag") maps to the documented
    default. A user-provided `0.0` is preserved verbatim — without this
    helper the constructor's own default would silently re-apply 0.10.
    """
    return _BAP_SAFETY_MARGIN_DEFAULT if value is None else value


def _resolve_bap_max_growth(value: Optional[float], fallback: float) -> float:
    """Resolve `--bap-max-growth` to BudgetAwareProposer's `max_growth`.

    `None` falls back to `EvolutionConfig.bap_max_growth` (passed by the
    caller). A user-provided `0.0` is preserved — it's a legitimate
    "no headroom" target; the proposer's
    `prompt_growth = max(0.0, max_growth - safety_margin)` handles it.
    """
    return fallback if value is None else value


_CLAUDE_CODE_PLUGIN_CACHE_MARKER = (".claude", "plugins", "cache")


def _is_claude_code_plugin_cache_path(path: Path) -> bool:
    """Detect paths under ``~/.claude/plugins/cache`` (Claude Code's plugin cache).

    Plugin caches are managed externally by Claude Code; writing into them
    silently is wrong. Match by looking for the three-segment marker
    anywhere in the resolved path so the check works for both the user's
    home directory and tmp_path-rooted test layouts.
    """
    parts = path.resolve().parts
    marker = _CLAUDE_CODE_PLUGIN_CACHE_MARKER
    for i in range(len(parts) - len(marker) + 1):
        if parts[i:i + len(marker)] == marker:
            return True
    return False


def _emit_patch(baseline_text: str, evolved_text: str, path: Path) -> str:
    """Return a unified diff of (baseline -> evolved) labelled with `path`.

    The labels are ``a/<path>`` and ``b/<path>`` so the output is consumable
    by ``patch -p1`` or ``git apply``.
    """
    label = str(path)
    diff_lines = difflib.unified_diff(
        baseline_text.splitlines(keepends=True),
        evolved_text.splitlines(keepends=True),
        fromfile=f"a/{label}",
        tofile=f"b/{label}",
    )
    return "".join(diff_lines)


def _maybe_build_closed_loop_cache_skill(
    *,
    skill_name: str,
    skill_path: Path,
    baseline_skill_body: str,
    suite_path: Optional[Path],
    saturation_threshold: float,
    min_iters: int,
    window_size: int,
    gate_mode: str = "sampled",
    agent_model: Optional[str] = None,
    agent_timeout_seconds: Optional[int] = None,
):
    """Build a ClosedLoopFeedbackCache for the skill path; return None when disabled.

    Mirrors evolve_tool's _maybe_build_closed_loop_cache. Local imports keep
    the validation stack out of the cold path — most evolve_skill runs don't
    set the flag and shouldn't pay the import cost.

    Constructs:
      - a per-process workdir under /tmp where the installer maintains a
        writable copy of the baseline skill (decoupled from the user's
        real HERMES_HOME / plugin cache)
      - SkillFileInstaller pointing at that workdir
      - ClosedLoopValidator + HermesAgentRunner
      - ClosedLoopFeedbackCache wired with write_text_artifact (skill bodies
        are raw text, not MCP manifests) and .md suffix
    """
    if suite_path is None:
        return None
    import tempfile

    from evolution.core.closed_loop_feedback import (
        ClosedLoopFeedbackCache,
        write_text_artifact,
    )
    from evolution.validation.artifact_installer import SkillFileInstaller
    from evolution.validation.hermes_runner import HermesAgentRunner
    from evolution.validation.task import TaskSuite
    from evolution.validation.validator import ClosedLoopValidator

    workdir = Path(tempfile.mkdtemp(prefix="cl_skill_workdir_"))
    installer = SkillFileInstaller(
        skill_source_path=skill_path,
        skill_name=skill_name,
        workdir=workdir,
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
        artifact_name=skill_name,
        baseline_artifact_text=baseline_skill_body,
        saturation_threshold=saturation_threshold,
        min_iters=min_iters,
        window_size=window_size,
        gate_mode=gate_mode,
        artifact_writer=write_text_artifact,
        artifact_suffix=".md",
    )


def _load_behavioral_examples_from_suite(suite_path: Path) -> list:
    """Build behavioral dspy.Examples from a suite file.

    Uses ``task_field="task_input"`` so the examples are shape-compatible
    with ``SkillModule.forward(task_input=...)`` — tool-side uses ``"task"``.
    Local-import path keeps the validation stack out of the cold path.
    """
    from evolution.core.behavioral_example import build_behavioral_examples
    from evolution.validation.task import TaskSuite

    return build_behavioral_examples(
        TaskSuite.from_jsonl(suite_path), task_field="task_input"
    )


def _apply_in_place(skill_path: Path, evolved_full: str) -> bool:
    """Overwrite ``skill_path`` with ``evolved_full``.

    Returns True on success, False when the destination is under
    ``~/.claude/plugins/cache`` (Claude Code's plugin cache, which is
    externally managed) — in that case the file is left untouched and a
    warning is logged.
    """
    if _is_claude_code_plugin_cache_path(skill_path):
        logging.getLogger(__name__).warning(
            "--apply skipped: %s is under a Claude Code plugin cache "
            "(~/.claude/plugins/cache); plugin caches are managed by "
            "Claude Code and writing to them is unsafe.",
            skill_path,
        )
        return False
    skill_path.write_text(evolved_full)
    return True


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: Optional[str] = None,
    eval_model: Optional[str] = None,
    skill_source_dirs: Optional[list[str]] = None,
    dry_run: bool = False,
    seed: int = 42,
    budget: Optional[str] = None,
    no_fallback: bool = False,
    reflection_model: Optional[str] = None,
    quality_gate: str = "default",
    growth_free_threshold: Optional[float] = None,
    growth_quality_slope: Optional[float] = None,
    max_absolute_chars: Optional[int] = None,
    inferiority_tolerance: Optional[float] = None,
    bootstrap_confidence: Optional[float] = None,
    bootstrap_n_resamples: Optional[int] = None,
    knee_point_epsilon: Optional[float] = None,
    knee_point_strategy: str = "val-best",
    bap_safety_margin: Optional[float] = None,
    bap_max_growth: Optional[float] = None,
    eval_dataset_size: Optional[int] = None,
    holdout_ratio: Optional[float] = None,
    evaluate_band_on_holdout: bool = False,
    fitness_profile: str = "balanced",
    apply_in_place: bool = False,
    emit_patch: bool = False,
    max_total_cost_usd: Optional[float] = None,
    benchmark_cmd: Optional[str] = None,
    benchmark_timeout_seconds: int = 600,
    skip_preflight: bool = False,
    skip_cost_suggest: bool = False,
    skip_saturation_check: bool = False,
    force_saturation_check: bool = False,
    gepa_minibatch_size: int = 3,
    closed_loop_suite_path: Optional[Path] = None,
    closed_loop_saturation_threshold: float = 0.95,
    closed_loop_min_iters: int = 3,
    closed_loop_window_size: int = 8,
    closed_loop_mode: str = "feedback",
    closed_loop_in_valset: bool = False,
    closed_loop_agent_model: Optional[str] = None,
    closed_loop_task_timeout_seconds: Optional[int] = None,
):
    """Main evolution function — orchestrates the full optimization loop."""

    preset = QUALITY_GATE_PRESETS[quality_gate]
    if quality_gate == "off":
        logging.getLogger(__name__).warning(
            '--quality-gate off still enforces a regression check (mean ≥ 0). '
            'For "deploy if not significantly worse than baseline" semantics, '
            'use --quality-gate non-inferiority --inferiority-tolerance 0.02.'
        )
    resolved_free = growth_free_threshold if growth_free_threshold is not None else preset["growth_free_threshold"]
    resolved_slope = growth_quality_slope if growth_quality_slope is not None else preset["growth_quality_slope"]
    resolved_abs = max_absolute_chars if max_absolute_chars is not None else preset["max_absolute_chars"]
    resolved_gate_mode = preset.get("gate_mode", "no_regression")
    resolved_tolerance = (
        inferiority_tolerance
        if inferiority_tolerance is not None
        else preset.get("inferiority_tolerance", 0.0)
    )

    config_kwargs = dict(
        iterations=iterations,
        optimizer_model=optimizer_model,
        reflection_model=reflection_model,
        eval_model=eval_model,
        judge_model=eval_model,  # Use same model for dataset generation
        seed=seed,
        growth_free_threshold=resolved_free,
        growth_quality_slope=resolved_slope,
        max_absolute_chars=int(resolved_abs),
        gate_mode=resolved_gate_mode,
        inferiority_tolerance=float(resolved_tolerance),
        fitness_profile=fitness_profile,
    )
    if bootstrap_confidence is not None:
        config_kwargs["bootstrap_confidence"] = bootstrap_confidence
    if bootstrap_n_resamples is not None:
        config_kwargs["bootstrap_n_resamples"] = bootstrap_n_resamples
    if eval_dataset_size is not None:
        config_kwargs["eval_dataset_size"] = eval_dataset_size
    if holdout_ratio is not None:
        config_kwargs["holdout_ratio"] = holdout_ratio
    config_kwargs["reflection_minibatch_size"] = gepa_minibatch_size
    config = EvolutionConfig(**config_kwargs)
    explicit_dirs = [Path(d) for d in (skill_source_dirs or [])]
    if explicit_dirs:
        # Without explicit dirs, EvolutionConfig's default_factory already
        # ran discovery — don't double-walk.
        config.skill_sources = discover_skill_sources(explicit_dirs=explicit_dirs)

    console.print(f"\n[bold cyan]🧬 Agent Skill Self-Evolution[/bold cyan] — Evolving skill: [bold]{skill_name}[/bold]\n")

    skill_path = find_skill(skill_name, config.skill_sources)
    if not skill_path:
        searched = ", ".join(s.name for s in config.skill_sources) or "(no sources discovered)"
        console.print(f"[red]✗ Skill '{skill_name}' not found across sources: {searched}[/red]")
        for source in config.skill_sources:
            available = source.list_skills()
            if available:
                preview = ", ".join(available[:8]) + (" …" if len(available) > 8 else "")
                console.print(f"  [dim]{source.name}: {len(available)} skills available — {preview}[/dim]")
        sys.exit(1)

    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path}")
    console.print(f"  Name: {skill['name']}")
    console.print(f"  Size: {len(skill['raw']):,} chars")
    console.print(f"  Description: {skill['description'][:80]}...")

    if dry_run:
        resolved_budget = _resolve_budget(iterations, budget)
        console.print(f"\n[bold green]DRY RUN — setup validated successfully.[/bold green]")
        console.print(f"  Would generate eval dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization (budget={resolved_budget})")
        console.print(f"  Would validate constraints and create PR")
        return

    # Created up-front (not after GEPA) so the FileHandler captures
    # dataset-gen LM calls + GEPA reflection + holdout eval. Reused later
    # for evolved_skill.md and gate_decision.json.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / skill_name / timestamp
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
            # Module-level singleton: reset between runs so metrics.json reflects
            # this run only, not whatever previous evolve() call(s) accumulated.
            COST_LEDGER.reset()
            COST_LEDGER.set_ceiling(max_total_cost_usd)
            console.print(f"  Run log: {run_log_path}")

            # Validate credentials before doing ANY LM work — dataset
            # generation alone can spend $0.50+ before we'd otherwise
            # discover the eval LM has a stale token. Preflight is one
            # tiny call per unique LM, raises HermesProviderError with
            # provider-specific recovery guidance.
            # Resolve up front so both preflight (if enabled) and the cost
            # advisor (if enabled and eval_model wasn't explicit) share the
            # same ResolvedLM. The downstream LM-configure path (~line 700)
            # re-resolves; small duplicate cost (no network — just file I/O).
            _preflight_optimizer = resolve_default_lm(role="optimizer", explicit_model=optimizer_model)
            _preflight_eval = resolve_default_lm(role="eval", explicit_model=eval_model)
            if not skip_preflight:
                _preflight_lm_credentials([_preflight_optimizer, _preflight_eval])
            # Cost advisor: only fire when the user inherited the eval model
            # from Hermes (eval_model is None) AND the resolver returned a
            # stock LM. Custom factory paths (Codex) route to a closed
            # ChatGPT-subscription endpoint where suggesting "use openai/
            # gpt-5-nano" implies a different auth setup the user didn't
            # opt in to.
            if (
                not skip_cost_suggest
                and eval_model is None
                and _preflight_eval.lm_factory is None
            ):
                _alt = _find_cheaper_alternative(_preflight_eval.model)
                if _alt is not None:
                    console.print(_render_cost_suggestion_panel("eval", _alt))
            if max_total_cost_usd is not None:
                console.print(f"  Cost ceiling: ${max_total_cost_usd:.4f}")

            console.print(f"\n[bold]Building evaluation dataset[/bold] (source: {eval_source})")

            if eval_source == "golden" and dataset_path:
                dataset = GoldenDatasetLoader.load(Path(dataset_path), seed=config.seed)
                console.print(f"  Loaded golden dataset: {len(dataset.all_examples)} examples")
            elif eval_source == "sessiondb":
                save_path = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
                dataset = build_dataset_from_external(
                    skill_name=skill_name,
                    skill_text=skill["raw"],
                    sources=["claude-code", "copilot", "hermes"],
                    output_path=save_path,
                    model=eval_model,
                    seed=config.seed,
                )
                if not dataset.all_examples:
                    console.print("[red]✗ No relevant examples found from session history[/red]")
                    sys.exit(1)
                console.print(f"  Mined {len(dataset.all_examples)} examples from session history")
            elif eval_source == "synthetic":
                builder = SyntheticDatasetBuilder(config)
                dataset = builder.generate(
                    artifact_text=skill["raw"],
                    artifact_type="skill",
                )
                save_path = Path("datasets") / "skills" / skill_name
                dataset.save(save_path)
                console.print(f"  Generated {len(dataset.all_examples)} synthetic examples")
                console.print(f"  Saved to {save_path}/")
            elif dataset_path:
                dataset = EvalDataset.load(Path(dataset_path))
                console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
            else:
                console.print("[red]✗ Specify --dataset-path or use --eval-source synthetic[/red]")
                sys.exit(1)

            console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

            # A 1-2 example holdout has stdev ~0.2 — the bootstrap CI swamps any
            # real lift signal. Raise eval_dataset_size or holdout_ratio rather
            # than override min_holdout_size.
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

            # Static checks only — the growth-with-quality gate runs later on
            # the evolved artifact once there's a holdout improvement signal.
            console.print(f"\n[bold]Validating baseline constraints[/bold]")
            validator = ConstraintValidator(config)
            baseline_constraints = validator.validate_static(skill["raw"], "skill")
            all_pass = True
            for c in baseline_constraints:
                icon = "✓" if c.passed else "✗"
                color = "green" if c.passed else "red"
                console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
                if not c.passed:
                    all_pass = False

            if not all_pass:
                console.print("[yellow]⚠ Baseline skill has constraint violations — proceeding anyway[/yellow]")

            gepa_budget = _resolve_budget(iterations, budget)
            # Resolve up-front so the banner reflects the model the run will
            # actually call — printing the raw CLI flag value showed "None"
            # whenever Hermes was doing the resolving.
            _optimizer_lm = resolve_default_lm(role="optimizer", explicit_model=optimizer_model)
            _eval_lm = resolve_default_lm(role="eval", explicit_model=eval_model)
            console.print(f"\n[bold]Configuring optimizer[/bold]")
            console.print(f"  Optimizer: GEPA (budget={gepa_budget})")
            console.print(f"  Optimizer model: {_optimizer_lm.model} ({_optimizer_lm.source})")
            console.print(f"  Eval model: {_eval_lm.model} ({_eval_lm.source})")

            # request_timeout=60 ≈ 6x P99 of slowest observed gpt-4.1-mini call.
            lm = instantiate_lm(_eval_lm, request_timeout=60, num_retries=5)
            # warn_on_type_mismatch=False silences spam from signatures that pass
            # empty/None into `str` inputs (e.g. RelevanceFilter.assistant_response
            # before any assistant turn).
            dspy.configure(
                lm=lm,
                warn_on_type_mismatch=False,
                callbacks=[LMTimingCallback()],
            )

            baseline_module = SkillModule(skill["body"])

            # In behavioral-trainset modes the saturation gate would defeat
            # the purpose — every novel candidate must score every time it's
            # sampled. Otherwise default to "sampled" to keep cost bounded;
            # skill bodies mutate heavily, so cache hit rate on the validator
            # is lower than tool-path.
            _cache_gate_mode = (
                "always" if closed_loop_mode in ("trainset", "both") else "sampled"
            )
            closed_loop_cache = _maybe_build_closed_loop_cache_skill(
                skill_name=skill_name,
                skill_path=skill_path,
                baseline_skill_body=skill["body"],
                suite_path=closed_loop_suite_path,
                saturation_threshold=closed_loop_saturation_threshold,
                min_iters=closed_loop_min_iters,
                window_size=closed_loop_window_size,
                gate_mode=_cache_gate_mode,
                agent_model=closed_loop_agent_model,
                agent_timeout_seconds=closed_loop_task_timeout_seconds,
            )

            # Build the metric once: DSPy's LM cache lines up across GEPA's
            # per-iteration scoring and the holdout eval below. The [BUDGET]
            # feedback line targets growth_free_threshold (the zone where the
            # deploy gate doesn't require quality justification) so the optimizer
            # learns to land there.
            judge = LLMJudge(config)
            metric = make_skill_fitness_metric(
                judge,
                baseline_skill_text=skill["body"],
                max_growth=config.growth_free_threshold,
                closed_loop_cache=closed_loop_cache,
            )

            trainset = dataset.to_dspy_examples("train")
            valset = dataset.to_dspy_examples("val")

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

            cached_baseline_holdout_per_example: Optional[list[float]] = None
            preflight_band: Optional[str] = None
            cached_baseline_cl_per_example: Optional[list[float]] = None
            preflight_holdout_score: Optional[float] = None
            preflight_cl_score: Optional[float] = None
            if not skip_saturation_check:
                holdout_examples_for_preflight = dataset.to_dspy_examples("holdout")
                sat_report = saturation_preflight(
                    baseline_module=baseline_module,
                    holdout_examples=holdout_examples_for_preflight,
                    metric=metric,
                    lm=lm,
                    closed_loop_cache=closed_loop_cache,
                    baseline_artifact_text=skill["body"],
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
                # Preserve preflight outputs for the deploy gate's CL-primary
                # path. All None on the --no-saturation-check path (initialized
                # above the preflight branch).
                preflight_band = sat_report.band
                cached_baseline_cl_per_example = sat_report.closed_loop_per_example
                preflight_holdout_score = sat_report.holdout_score
                preflight_cl_score = sat_report.closed_loop_score

            console.print(f"\n[bold cyan]Running GEPA optimization (budget={gepa_budget})...[/bold cyan]\n")

            start_time = time.time()
            failure_log_path = Path("output") / skill_name / "gepa_failure.log"

            # gepa_kwargs={"reflection_prompt_template": ...} is the simpler path
            # but gepa.api rejects it whenever DspyAdapter (always) provides its
            # own propose_new_texts (gepa/api.py:317-321). instruction_proposer
            # is DSPy's documented extension point.
            resolved_bap_max_growth = _resolve_bap_max_growth(
                bap_max_growth, config.bap_max_growth,
            )
            resolved_bap_safety_margin = _resolve_bap_safety_margin(bap_safety_margin)
            proposer_mode = resolve_proposer_mode(config.fitness_profile)
            proposer = BudgetAwareProposer(
                baseline_chars=len(skill["body"]),
                max_growth=resolved_bap_max_growth,
                safety_margin=resolved_bap_safety_margin,
                mode=proposer_mode,
            )

            optimized_module, optimizer_name = _build_optimizer_and_compile(
                baseline_module=baseline_module,
                trainset=trainset,
                valset=valset,
                metric=metric,
                gepa_budget=gepa_budget,
                optimizer_model=optimizer_model,
                seed=config.seed,
                no_fallback=no_fallback,
                failure_log_path=failure_log_path,
                instruction_proposer=proposer,
                reflection_model=config.reflection_model,
                reflection_minibatch_size=config.reflection_minibatch_size,
            )

            elapsed = time.time() - start_time
            console.print(f"\n  {optimizer_name} optimization completed in {elapsed:.1f}s")

            # The val-best path defers to GEPA's argmax (details.best_idx).
            # Regenerated calibration showed the epsilon-band selector picked
            # GEPA's default 10/10 across five epsilon modes; see
            # reports/calibration_findings.md Finding 3. The --knee-point-strategy
            # smallest path still routes through select_knee_point for users
            # explicitly chasing compression.
            # Skipped cleanly when MIPROv2 fallback fired (no detailed_results).
            knee_pick: Optional[CandidatePick] = None
            knee_payload: dict[str, Any] = {
                "applied": False, "reason": "no_detailed_results",
            }
            if hasattr(optimized_module, "detailed_results"):
                details = optimized_module.detailed_results
                if knee_point_strategy == "smallest":
                    knee_pick = select_knee_point(
                        candidates=details.candidates,
                        val_aggregate_scores=details.val_aggregate_scores,
                        n_val=len(valset),
                        static_validator=lambda txt: validator.validate_static(
                            reassemble_skill(skill["frontmatter"], txt), "skill",
                        ),
                        gepa_default_idx=details.best_idx,
                        epsilon=knee_point_epsilon,
                        strategy=knee_point_strategy,
                    )
                    optimized_module = SkillModule(knee_pick.skill_text)
                    knee_payload = _knee_point_payload(knee_pick)
                    console.print(
                        f"\n[bold]Knee-point selection[/bold]: picked candidate "
                        f"{knee_pick.picked_idx} (val={knee_pick.val_score:.3f}, "
                        f"rank {knee_pick.val_rank_in_band} of {knee_pick.band_size} "
                        f"in band, {knee_pick.body_chars} chars vs GEPA default "
                        f"{knee_pick.gepa_default_body_chars} chars; ε={knee_pick.epsilon:.3f}; "
                        f"fallback={knee_pick.fallback})"
                    )
                else:
                    # val-best no longer walks the band on static failure;
                    # --knee-point-strategy smallest preserves that behavior.
                    best_text = details.candidates[details.best_idx].skill_text
                    optimized_module = SkillModule(best_text)
                    knee_payload = _deferred_knee_point_payload(
                        best_idx=details.best_idx,
                        val_score=float(details.val_aggregate_scores[details.best_idx]),
                        body_chars=len(best_text),
                    )
                    console.print(
                        f"\n[bold]Candidate selection[/bold]: GEPA val-argmax "
                        f"(candidate {details.best_idx}, val="
                        f"{details.val_aggregate_scores[details.best_idx]:.3f}, "
                        f"{len(best_text)} chars)"
                    )

            evolved_body = optimized_module.skill_text
            evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)

            # Fail-fast on broken artifacts before spending judge-call budget on
            # the holdout. Growth-with-quality is checked after the holdout.
            console.print(f"\n[bold]Validating evolved skill (static checks)[/bold]")
            static_constraints = validator.validate_static(evolved_full, "skill")
            static_pass = True
            for c in static_constraints:
                icon = "✓" if c.passed else "✗"
                color = "green" if c.passed else "red"
                console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
                if not c.passed:
                    static_pass = False

            if not static_pass:
                console.print("[red]✗ Evolved skill FAILED static constraints — not deploying[/red]")
                failed_path = output_dir / "evolved_FAILED.md"
                failed_path.write_text(evolved_full)
                write_gate_decision(output_dir, {
                    "schema_version": "5",
                    "decision": "reject",
                    "reason": "static_constraint_failure",
                    "decision_signal": "synthetic",
                    "failed_constraints": [c.constraint_name for c in static_constraints if not c.passed],
                    "messages": [c.message for c in static_constraints if not c.passed],
                    "knee_point": knee_payload,
                    "dataset": _dataset_payload(dataset),
                    "run_inputs": build_run_inputs(
                        config=config,
                        iterations=iterations,
                        optimizer_model=optimizer_model,
                        quality_gate_preset=quality_gate,
                        eval_source=eval_source,
                    ),
                })
                console.print(f"  Saved failed variant to {failed_path}")
                return

            console.print(
                f"\n[bold]Evaluating on holdout set ({len(dataset.holdout)} examples)[/bold]"
            )
            console.print(
                "  [dim]Holdout uses the same LLM-as-judge metric as GEPA — expect ~"
                f"{2 * len(dataset.holdout)} judge calls.[/dim]"
            )

            holdout_examples = dataset.to_dspy_examples("holdout")
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

            # Decide which deploy-gate path applies. CL-primary fires when
            # the preflight saw weak_signal AND CL data is present. All
            # other cases (no preflight, healthy/no_headroom/uniform_failure
            # bands, missing CL data) use the synthetic-only path.
            baseline_chars = len(skill["raw"])
            evolved_chars = len(evolved_full)
            growth_pct = (evolved_chars - baseline_chars) / max(1, baseline_chars)

            run_inputs = build_run_inputs(
                config=config,
                iterations=iterations,
                optimizer_model=optimizer_model,
                quality_gate_preset=quality_gate,
                eval_source=eval_source,
            )

            use_cl_primary = (
                preflight_band == "weak_signal"
                and cached_baseline_cl_per_example is not None
                and len(cached_baseline_cl_per_example) > 0
                and closed_loop_cache is not None
            )

            evolved_cl_report = None
            evolved_cl_per_example: Optional[list[float]] = None
            evolved_cl_errored_task_ids: list[str] = []
            cl_eval_cost_before: float = 0.0
            cl_eval_cost_usd: Optional[float] = None
            cl_constraint: Optional[ConstraintResult] = None

            if use_cl_primary:
                console.print(
                    f"\n[bold]Evaluating evolved skill body on closed-loop suite[/bold] "
                    "(weak_signal band → CL-primary gate)"
                )
                cl_eval_cost_before = COST_LEDGER.summary().get("total_usd", 0.0)
                try:
                    # force_run takes the BODY (no YAML frontmatter); the cache
                    # key was set up with skill["body"] during preflight, so we
                    # must match that to avoid silently double-spending on the
                    # evolved eval.
                    evolved_cl_report = closed_loop_cache.force_run(evolved_body)
                except Exception as exc:  # ValidatorError or downstream
                    cl_eval_cost_usd = COST_LEDGER.summary().get("total_usd", 0.0) - cl_eval_cost_before
                    console.print(
                        f"[red]✗ Evolved closed-loop eval failed: {exc}[/red] — writing aborted decision"
                    )
                    failed_path = output_dir / "evolved_FAILED.md"
                    failed_path.write_text(evolved_full)
                    console.print(f"  Saved failed variant to {failed_path}")
                    write_gate_decision(output_dir, {
                        "schema_version": "5",
                        "decision": "aborted",
                        "reason": "cl_eval_failed",
                        "decision_signal": "closed_loop",
                        "cl_eval_exception": str(exc),
                        "evolved_cl_eval_cost_usd": cl_eval_cost_usd,
                        "band_trigger_score": {
                            "holdout": preflight_holdout_score,
                            "closed_loop": preflight_cl_score,
                        },
                        "validator_agent_model": closed_loop_agent_model,
                        "baseline_chars": baseline_chars,
                        "evolved_chars": evolved_chars,
                        "growth_pct": growth_pct,
                        "knee_point": knee_payload,
                        "dataset": _dataset_payload(dataset),
                        "run_inputs": run_inputs,
                    })
                    return
                cl_eval_cost_usd = COST_LEDGER.summary().get("total_usd", 0.0) - cl_eval_cost_before

                # Detect abstained tasks (TaskResult.abstained == True means
                # the runner errored — see validation/report.py:score_task).
                # An infrastructure flake on an evolved task is NOT a quality
                # regression; conflating them would falsely reject good
                # candidates. Hard-fail with a written diagnostic instead.
                evolved_cl_errored_task_ids = [
                    t.task_id for t in evolved_cl_report.evolved.tasks if t.abstained
                ]
                evolved_cl_per_example = [
                    1.0 if t.passed else 0.0 for t in evolved_cl_report.evolved.tasks
                ]
                if evolved_cl_errored_task_ids:
                    console.print(
                        f"[red]✗ {len(evolved_cl_errored_task_ids)} evolved CL task(s) errored "
                        f"({', '.join(evolved_cl_errored_task_ids)}) — writing aborted decision[/red]"
                    )
                    failed_path = output_dir / "evolved_FAILED.md"
                    failed_path.write_text(evolved_full)
                    console.print(f"  Saved failed variant to {failed_path}")
                    write_gate_decision(output_dir, {
                        "schema_version": "5",
                        "decision": "aborted",
                        "reason": "cl_eval_incomplete",
                        "decision_signal": "closed_loop",
                        "evolved_closed_loop_errored_tasks": evolved_cl_errored_task_ids,
                        "evolved_closed_loop_per_example": evolved_cl_per_example,
                        "baseline_closed_loop_per_example": cached_baseline_cl_per_example,
                        "evolved_cl_eval_cost_usd": cl_eval_cost_usd,
                        "band_trigger_score": {
                            "holdout": preflight_holdout_score,
                            "closed_loop": preflight_cl_score,
                        },
                        "validator_agent_model": closed_loop_agent_model,
                        "baseline_chars": baseline_chars,
                        "evolved_chars": evolved_chars,
                        "growth_pct": growth_pct,
                        "knee_point": knee_payload,
                        "dataset": _dataset_payload(dataset),
                        "run_inputs": run_inputs,
                    })
                    return

                baseline_cl_passes = int(sum(cached_baseline_cl_per_example))
                evolved_cl_passes = int(sum(evolved_cl_per_example))
                cl_constraint = _check_cl_primary_gate(
                    baseline_cl_passes=baseline_cl_passes,
                    evolved_cl_passes=evolved_cl_passes,
                    baseline_synth_mean=avg_baseline,
                    evolved_synth_mean=avg_evolved,
                    growth_pct=growth_pct,
                )
                icon = "✓" if cl_constraint.passed else "✗"
                color = "green" if cl_constraint.passed else "red"
                console.print(
                    f"  [{color}]{icon} cl_primary_gate[/{color}]: {cl_constraint.message}"
                )

            if evaluate_band_on_holdout and knee_pick is not None:
                console.print(
                    f"\n[bold]Re-evaluating {knee_pick.band_size} band candidate(s) on holdout[/bold] "
                    "[dim](calibration telemetry; only enabled with --evaluate-band-on-holdout)[/dim]"
                )
                band_path = _evaluate_band_on_holdout(
                    knee_pick=knee_pick,
                    candidates=details.candidates,
                    holdout_examples=holdout_examples,
                    metric=metric,
                    lm=lm,
                    output_dir=output_dir,
                    seed=config.seed,
                )
                console.print(f"  Wrote {band_path.name}")

            console.print(f"\n[bold]Validating growth against holdout improvement[/bold]")
            bootstrap = paired_bootstrap(
                baseline_per_example,
                evolved_per_example,
                confidence=config.bootstrap_confidence,
                n_resamples=config.bootstrap_n_resamples,
                seed=config.seed,
            )
            if use_cl_primary:
                # CL-primary path: skip the synthetic growth_quality_gate
                # (it would always reject when synth is saturated and growth > 0).
                # But still enforce the absolute_char_ceiling — that's an
                # orthogonal wallpaper-protection backstop that must hold
                # regardless of which signal we're gating on.
                # cl_constraint was bound in the earlier `if use_cl_primary:` block;
                # the assert narrows Optional[ConstraintResult] so growth_constraints
                # types as list[ConstraintResult], not list[Optional[ConstraintResult]].
                assert cl_constraint is not None
                ceiling_constraint = validator._check_absolute_chars(
                    evolved_full, baseline_chars,
                )
                growth_constraints = [cl_constraint, ceiling_constraint]
            else:
                # Synthetic-only path (unchanged): growth_quality_gate runs both
                # the growth curve and the absolute-char ceiling internally.
                growth_constraints = validator.validate_growth_with_quality(
                    evolved_full, skill["raw"], bootstrap,
                )
            growth_pass = True
            for c in growth_constraints:
                icon = "✓" if c.passed else "✗"
                color = "green" if c.passed else "red"
                console.print(f"  [{color}]{icon} {c.constraint_name}[/{color}]: {c.message}")
                if not c.passed:
                    growth_pass = False

            # Write artifacts before the hook so it can reference them via
            # $EVOLVED_PATH / $BASELINE_PATH. On benchmark failure the deploy
            # path's identical write becomes a no-op overwrite of the reject
            # path's evolved_FAILED.md.
            benchmark_block: Optional[dict[str, Any]] = None
            if growth_pass and benchmark_cmd is not None:
                evolved_path = output_dir / "evolved_skill.md"
                baseline_path = output_dir / "baseline_skill.md"
                evolved_path.write_text(evolved_full)
                baseline_path.write_text(skill["raw"])
                benchmark_block = run_benchmark_hook(
                    benchmark_cmd,
                    timeout_seconds=benchmark_timeout_seconds,
                    evolved_path=evolved_path,
                    baseline_path=baseline_path,
                    output_dir=output_dir,
                    target_name=skill_name,
                    artifact_type="skill",
                )
                if not benchmark_block["passed"]:
                    growth_pass = False
                    evolved_path.unlink(missing_ok=True)
                    baseline_path.unlink(missing_ok=True)

            # baseline_chars / evolved_chars / growth_pct are bound earlier
            # (before the use_cl_primary branch) so the CL-primary path can
            # use them in its abort payloads. Don't recompute here.
            required_improvement = max(
                0.0,
                config.growth_quality_slope * (growth_pct - config.growth_free_threshold),
            )
            # Single source of truth for the rule string — same helper the constraint uses.
            decision_rule_used = resolve_decision_rule(config, growth_pct)
            if growth_pass:
                decision_reason = "passed"
            elif benchmark_block is not None and not benchmark_block["passed"]:
                decision_reason = "benchmark_failed"
            else:
                decision_reason = "growth_quality_gate"
            decision_payload = {
                "schema_version": "5",
                "decision": "deploy" if growth_pass else "reject",
                "reason": decision_reason,
                "decision_signal": "closed_loop" if use_cl_primary else "synthetic",
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
                "bap_max_growth": resolved_bap_max_growth,
                "bap_safety_margin": resolved_bap_safety_margin,
                "baseline_per_example": baseline_per_example,
                "evolved_per_example": evolved_per_example,
                "avg_baseline": avg_baseline,
                "avg_evolved": avg_evolved,
                "bootstrap": bootstrap,
                "win_loss": _compute_win_loss(baseline_per_example, evolved_per_example),
                "failed_constraints": [c.constraint_name for c in growth_constraints if not c.passed],
                "messages": [c.message for c in growth_constraints if not c.passed],
                "knee_point": knee_payload,
                "dataset": _dataset_payload(dataset),
                "run_inputs": run_inputs,
            }
            if benchmark_block is not None:
                decision_payload["benchmark"] = benchmark_block

            if use_cl_primary:
                append_cl_decision_fields(
                    decision_payload,
                    cached_baseline_cl_per_example=cached_baseline_cl_per_example,
                    evolved_cl_per_example=evolved_cl_per_example,
                    avg_baseline=avg_baseline,
                    avg_evolved=avg_evolved,
                    growth_pct=growth_pct,
                    cl_eval_cost_usd=cl_eval_cost_usd,
                    preflight_holdout_score=preflight_holdout_score,
                    preflight_cl_score=preflight_cl_score,
                    closed_loop_agent_model=closed_loop_agent_model,
                )

            if not use_cl_primary and preflight_band is None:
                # User passed --no-saturation-check; record why CL-primary
                # didn't fire even though CL may be configured.
                decision_payload["reason_synthetic"] = "preflight_skipped"

            gate_path = write_gate_decision(output_dir, decision_payload)
            console.print(f"  [dim]Gate decision logged to {gate_path}[/dim]")

            if not growth_pass:
                console.print("[red]✗ Evolved skill REJECTED by quality gate — not deploying[/red]")
                if use_cl_primary:
                    console.print(
                        f"[yellow]⚠ Evolution rejected: "
                        f"CL gain {decision_payload['cl_tasks_gained']} < "
                        f"required {decision_payload['cl_required_gain']}[/yellow]"
                    )
                failed_path = output_dir / "evolved_FAILED.md"
                failed_path.write_text(evolved_full)
                console.print(f"  Saved failed variant to {failed_path}")
                reject_reason = decision_payload.get("reason", "growth_quality_gate")
                if apply_in_place:
                    print(
                        f"--apply skipped: gate rejected (decision: reject, reason: {reject_reason})",
                        file=sys.stderr,
                    )
                if emit_patch:
                    print(
                        f"--patch skipped: gate rejected (decision: reject, reason: {reject_reason})",
                        file=sys.stderr,
                    )
                return

            table = Table(title="Evolution Results")
            table.add_column("Metric", style="bold")
            table.add_column("Baseline", justify="right")
            table.add_column("Evolved", justify="right")
            table.add_column("Change", justify="right")

            # Under CL-primary, the gate verdict — not the synthetic delta —
            # decides the row color; the synthetic delta is informational.
            row_color = (
                ("green" if growth_pass else "yellow")
                if use_cl_primary
                else ("green" if improvement > 0 else "red")
            )
            table.add_row(
                "Holdout Score",
                f"{avg_baseline:.3f}",
                f"{avg_evolved:.3f}",
                f"[{row_color}]{improvement:+.3f}[/{row_color}]",
            )
            if use_cl_primary:
                baseline_cl = int(sum(cached_baseline_cl_per_example))
                evolved_cl = int(sum(evolved_cl_per_example))
                table.add_row(
                    "Closed-loop (behavioral)",
                    f"{baseline_cl} tasks",
                    f"{evolved_cl} tasks",
                    f"[{row_color}]{evolved_cl - baseline_cl:+d} tasks[/{row_color}]",
                )
            table.add_row(
                "Skill Size",
                f"{len(skill['body']):,} chars",
                f"{len(evolved_body):,} chars",
                f"{len(evolved_body) - len(skill['body']):+,} chars",
            )
            table.add_row("Time", "", f"{elapsed:.1f}s", "")
            table.add_row("Iterations", "", str(iterations), "")

            console.print()
            console.print(table)

            evolved_skill_path = output_dir / "evolved_skill.md"
            evolved_skill_path.write_text(evolved_full)
            (output_dir / "baseline_skill.md").write_text(skill["raw"])
            metrics = {
                "skill_name": skill_name,
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
                "baseline_size": len(skill["body"]),
                "evolved_size": len(evolved_body),
                "train_examples": len(dataset.train),
                "val_examples": len(dataset.val),
                "holdout_examples": len(dataset.holdout),
                "elapsed_seconds": elapsed,
                "constraints_passed": all_pass,
                "cost": COST_LEDGER.summary(),
            }
            (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

            console.print(f"\n  Output saved to {output_dir}/")

            if growth_pass:
                if emit_patch:
                    patch_text = _emit_patch(skill["raw"], evolved_full, skill_path)
                    sys.stdout.write(patch_text)
                    if patch_text and not patch_text.endswith("\n"):
                        sys.stdout.write("\n")
                if apply_in_place:
                    applied = _apply_in_place(skill_path, evolved_full)
                    if applied:
                        console.print(f"  --apply: wrote evolved skill to {skill_path}")

            if use_cl_primary:
                console.print(
                    f"\n[bold green]✓ Evolution improved skill "
                    f"(CL gained +{decision_payload['cl_tasks_gained']} tasks)[/bold green]"
                )
                console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
            elif improvement > 0:
                console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} ({improvement/max(0.001, avg_baseline)*100:+.1f}%)[/bold green]")
                console.print(f"  Review the diff: diff {output_dir}/baseline_skill.md {output_dir}/evolved_skill.md")
            else:
                console.print(f"\n[yellow]⚠ Evolution did not improve skill (change: {improvement:+.3f})[/yellow]")
                console.print("  Try: more iterations, better eval dataset, or different optimizer model")
        except CostCeilingExceeded as exc:
            write_cost_ceiling_abort(
                exc,
                output_dir=output_dir,
                run_inputs=build_run_inputs(
                    config=config,
                    iterations=iterations,
                    optimizer_model=optimizer_model,
                    quality_gate_preset=quality_gate,
                    eval_source=eval_source,
                ),
                schema_version="5",
            )
            return
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option(
    "--iterations",
    default=10,
    help="DEPRECATED. Maps 1→light, 2→medium, 3→heavy GEPA budget; any other value falls through to light. Prefer --budget.",
)
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic", "golden", "sessiondb"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset (JSONL)")
@click.option(
    "--optimizer-model",
    default=None,
    help="LiteLLM model string for the optimizer LM (e.g. anthropic/claude-opus-4-5, "
    "openai/gpt-4.1, openrouter/anthropic/claude-opus-4-5). When unset, defaults "
    "to the model resolved from ~/.hermes/config.yaml + auth.json + provider env "
    "vars; if neither is configured, exits with an actionable error. "
    "Reflection LM is controlled separately by --reflection-model.",
)
@click.option(
    "--reflection-model",
    default=None,
    help="Model for the GEPA reflection LM (the LM the instruction proposer "
    "calls). DSPy's GEPA docstring recommends gpt-5-class reasoning models. "
    "Reasoning models require max_tokens >= 16000 (we set 32000). "
    "When unset, falls back to --optimizer-model.",
)
@click.option(
    "--eval-model",
    default=None,
    help="Model for evaluation + judge LMs. When unset, defaults to the "
    "model resolved from Hermes (same as --optimizer-model). On Hermes setups "
    "with one model only, all roles collapse onto it.",
)
@click.option(
    "--skill-source-dir",
    "skill_source_dir",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Additional skill-source root: <dir>/<name>/SKILL.md. Repeatable; "
    "explicit dirs take priority over auto-discovered Hermes/Claude Code "
    "sources. Use for Codex, openclaw, or any custom layout.",
)
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--seed", default=42, type=int, help="RNG seed for dataset shuffles and DSPy optimizer")
@click.option(
    "--budget",
    default=None,
    type=click.Choice(["light", "medium", "heavy"]),
    help="GEPA optimization budget. Overrides --iterations mapping.",
)
@click.option(
    "--no-fallback",
    is_flag=True,
    help="Re-raise GEPA failures instead of falling back to MIPROv2 (for debugging)",
)
@click.option(
    "--quality-gate",
    default="default",
    type=click.Choice(["strict", "default", "lenient", "off", "non-inferiority"]),
    help="Preset for the deploy gate's growth-vs-improvement curve. "
    "strict=(0.10/0.50/3000), default=(0.20/0.30/5000), "
    "lenient=(0.30/0.20/8000), off=slope/ceiling disabled but mean ≥ 0 still "
    "enforced (misnamed; see deprecation warning), "
    "non-inferiority=ships variants statistically not-worse-than-baseline by "
    "more than --inferiority-tolerance (recommended for compression runs).",
)
@click.option(
    "--growth-free-threshold",
    default=None,
    type=float,
    help="Advanced: override the preset's growth_free_threshold (growth "
    "below which no improvement justification is required).",
)
@click.option(
    "--growth-quality-slope",
    default=None,
    type=float,
    help="Advanced: override the preset's growth_quality_slope (linear "
    "rate at which required holdout improvement scales with growth above "
    "the free threshold).",
)
@click.option(
    "--max-absolute-chars",
    default=None,
    type=int,
    help="Advanced: override the preset's max_absolute_chars (hard char "
    "ceiling on the evolved artifact, independent of growth %).",
)
@click.option(
    "--inferiority-tolerance",
    default=None,
    type=float,
    help="Tolerance for the non-inferiority gate: pass when bootstrap "
    "lower bound > -tolerance. Only meaningful with "
    "--quality-gate non-inferiority (default tolerance there: 0.05).",
)
@click.option(
    "--bootstrap-confidence",
    default=None,
    type=float,
    help="Advanced: confidence level for the paired-bootstrap CI on the "
    "holdout improvement (default 0.90).",
)
@click.option(
    "--bootstrap-resamples",
    default=None,
    type=int,
    help="Advanced: number of bootstrap resamples (default 2000).",
)
@click.option(
    "--knee-point-epsilon",
    default=None,
    type=float,
    help="Advanced: ε tolerance for the knee-point band. Only used by "
    "--knee-point-strategy=smallest; the default val-best path defers to "
    "GEPA's val-argmax and ignores ε. Default = 1/n_val (one valset "
    "example's worth of disagreement).",
)
@click.option(
    "--knee-point-strategy",
    default="val-best",
    type=click.Choice(["val-best", "smallest"]),
    help="How to pick the deployed candidate from GEPA's output. val-best "
    "(default): defer to GEPA's val-argmax (best_idx) — does not walk an "
    "ε-band. smallest: walk the ε-band and pick the smallest body, "
    "accepting val cost for compression.",
)
@click.option(
    "--bap-safety-margin",
    default=None,
    type=float,
    help="Advanced: override BudgetAwareProposer's safety_margin (default "
    "0.10). The proposer asks the reflection LM for a target tighter than "
    "the validator's bar to absorb the LM's observed +8-9% overshoot. "
    "Setting to 0.0 disables the cushion — useful for calibration runs that "
    "want the LM to push toward the actual gate.",
)
@click.option(
    "--bap-max-growth",
    default=None,
    type=float,
    help="Advanced: override BudgetAwareProposer's max_growth — the growth "
    "target the proposer prompts the reflection LM toward. Decoupled from "
    "the gate's growth_free_threshold so calibration runs can test proposer "
    "behavior independently. Default (None): falls back to "
    "EvolutionConfig.bap_max_growth (default 0.20).",
)
@click.option(
    "--eval-dataset-size",
    default=None,
    type=int,
    help="Advanced: override EvolutionConfig.eval_dataset_size (default "
    "150). Total examples generated; train/val/holdout splits are derived "
    "via the configured ratios.",
)
@click.option(
    "--holdout-ratio",
    default=None,
    type=float,
    help="Advanced: override EvolutionConfig.holdout_ratio (default 0.50). "
    "Fraction of the dataset reserved for the deploy-gate's holdout "
    "evaluation, after train/val are taken.",
)
@click.option(
    "--evaluate-band-on-holdout/--no-evaluate-band-on-holdout",
    default=False,
    help="Calibration telemetry: after the picked candidate is selected, "
    "re-evaluate every candidate in the knee-point band on the holdout "
    "and write band_holdout.json. Off by default — adds judge calls "
    "proportional to band size × holdout examples (capped at 100).",
)
@click.option(
    "--fitness-profile",
    default="balanced",
    type=click.Choice(["balanced", "compression", "growth"]),
    help="Composite fitness weighting profile. 'balanced' (default) is "
    "general-purpose. 'compression' upweights conciseness for shrinking "
    "skills. 'growth' drops conciseness so the optimizer doesn't punish "
    "necessary additions.",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="On a deploy decision, copy evolved_skill.md over the source SKILL.md "
         "in place. No git operations — leaves workflow to the user. No-op on "
         "reject. No-op (with warning) when the skill source is read-only "
         "(e.g., Claude Code plugin cache).",
)
@click.option(
    "--patch",
    is_flag=True,
    default=False,
    help="On a deploy decision, emit a unified diff of (baseline → evolved) to "
         "stdout, labeled with the source path. Pipe to `patch`, `git apply`, "
         "or a code review tool. No-op on reject.",
)
@click.option(
    "--max-total-cost-usd",
    default=None,
    type=click.FloatRange(min=0.0),
    help="Safety net: abort the run cleanly when cumulative LM cost (across "
         "dataset gen, GEPA, holdout eval, and any sessiondb judge calls) "
         "exceeds this dollar amount. Worst-case overshoot is one LM call "
         "past the ceiling — the cost callback fires AFTER each call returns, "
         "and the next call aborts at start. 0 is accepted (aborts on first "
         "call, useful for testing). Negative values rejected. Off by default.",
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
         "in non-interactive contexts (no TTY).",
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
         "metric-call budget. The skill pipeline uses --budget (not "
         "--iterations) for its budget knob, so consider --budget heavy "
         "to preserve the proposal count. Aborts at startup if the "
         "value exceeds the trainset size.",
)
@click.option(
    "--closed-loop-during-evolution",
    "closed_loop_suite_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a JSONL task suite (e.g. evolution/validation/suites/"
         "systematic_debugging.jsonl). When set, the framework runs the "
         "closed-loop validator on saturating GEPA iterations and surfaces "
         "verdicts into the reflection LM's feedback. Held-out from training "
         "tasks (no overlap-detection enforcement).",
)
@click.option(
    "--closed-loop-saturation-threshold",
    default=0.95,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Min judge score over the recent window for the saturation gate to "
         "open (default 0.95).",
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
         "pass/fail from the validator) contributes to GEPA's "
         "sum(minibatch_scores) acceptance — lets behavioral wins break judge "
         "ties on saturated baselines. 'both' does trainset + the [CLOSED_LOOP] "
         "feedback block (most expensive). Skill bodies mutate heavily so "
         "trainset/both fires the validator on every novel candidate; default "
         "stays 'feedback' to keep cost bounded.",
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
         "daily-driver Hermes model is so capable it saturates the planted-bug "
         "suite at 100%, hiding the behavioral signal closed-loop is supposed "
         "to surface — run validation against a weaker model without touching "
         "your config.",
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
def main(skill, iterations, eval_source, dataset_path, optimizer_model, reflection_model,
         eval_model, skill_source_dir, dry_run, seed, budget, no_fallback,
         quality_gate, growth_free_threshold,
         growth_quality_slope, max_absolute_chars, inferiority_tolerance,
         bootstrap_confidence, bootstrap_resamples, knee_point_epsilon,
         knee_point_strategy, bap_safety_margin, bap_max_growth,
         eval_dataset_size, holdout_ratio, evaluate_band_on_holdout,
         fitness_profile, apply, patch, max_total_cost_usd,
         benchmark_cmd, benchmark_timeout_seconds,
         skip_preflight,
         skip_cost_suggest,
         skip_saturation_check,
         force_saturation_check,
         gepa_minibatch_size,
         closed_loop_suite_path,
         closed_loop_saturation_threshold,
         closed_loop_min_iters,
         closed_loop_window_size,
         closed_loop_mode,
         closed_loop_in_valset,
         closed_loop_agent_model,
         closed_loop_task_timeout_seconds):
    """Evolve an agent skill using DSPy + GEPA optimization."""
    try:
        evolve(
            skill_name=skill,
            iterations=iterations,
            eval_source=eval_source,
            dataset_path=dataset_path,
            optimizer_model=optimizer_model,
            reflection_model=reflection_model,
            eval_model=eval_model,
            skill_source_dirs=list(skill_source_dir) if skill_source_dir else None,
            dry_run=dry_run,
            seed=seed,
            budget=budget,
            no_fallback=no_fallback,
            quality_gate=quality_gate,
            growth_free_threshold=growth_free_threshold,
            growth_quality_slope=growth_quality_slope,
            max_absolute_chars=max_absolute_chars,
            inferiority_tolerance=inferiority_tolerance,
            bootstrap_confidence=bootstrap_confidence,
            bootstrap_n_resamples=bootstrap_resamples,
            knee_point_epsilon=knee_point_epsilon,
            knee_point_strategy=knee_point_strategy,
            bap_safety_margin=bap_safety_margin,
            bap_max_growth=bap_max_growth,
            eval_dataset_size=eval_dataset_size,
            holdout_ratio=holdout_ratio,
            evaluate_band_on_holdout=evaluate_band_on_holdout,
            fitness_profile=fitness_profile,
            apply_in_place=apply,
            emit_patch=patch,
            max_total_cost_usd=max_total_cost_usd,
            benchmark_cmd=benchmark_cmd,
            benchmark_timeout_seconds=benchmark_timeout_seconds,
            skip_preflight=skip_preflight,
            skip_cost_suggest=skip_cost_suggest,
            skip_saturation_check=skip_saturation_check,
            force_saturation_check=force_saturation_check,
            gepa_minibatch_size=gepa_minibatch_size,
            closed_loop_suite_path=closed_loop_suite_path,
            closed_loop_saturation_threshold=closed_loop_saturation_threshold,
            closed_loop_min_iters=closed_loop_min_iters,
            closed_loop_window_size=closed_loop_window_size,
            closed_loop_mode=closed_loop_mode,
            closed_loop_in_valset=closed_loop_in_valset,
            closed_loop_agent_model=closed_loop_agent_model,
            closed_loop_task_timeout_seconds=closed_loop_task_timeout_seconds,
        )
    except HermesProviderError as exc:
        # Render a clean error panel instead of dumping a Python traceback
        # — this is the failure mode for stale Hermes credentials and the
        # message contains actionable per-provider recovery commands.
        console.print(Panel(str(exc), title="[bold]Authentication[/bold]", border_style="red"))
        sys.exit(2)


if __name__ == "__main__":
    main()

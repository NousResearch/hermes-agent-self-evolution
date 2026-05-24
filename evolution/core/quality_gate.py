"""Artifact-agnostic quality-gate helpers extracted from ``evolve_skill``.

These helpers — preset table, proposer-mode resolution, and gate-decision
persistence — were never skill-specific. Hosting them in ``evolution.core``
lets the upcoming tool description pipeline reuse the deploy-gate scaffolding
without depending on the skill pipeline.
"""

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from evolution.core.constraints import ConstraintResult
from evolution.core.lm_timing_callback import COST_LEDGER, CostCeilingExceeded
from evolution.skills.budget_aware_proposer import ProposerMode

_console = Console()

_BENCHMARK_OUTPUT_TAIL_BYTES = 4096


# CL-primary deploy-gate formula constants. Mirrors the synthetic
# growth_quality_gate's free-threshold-then-slope shape (constraints.py
# _check_growth_with_quality_gate) but adapted to integer CL task gains.
#
# free_threshold matches EvolutionConfig.growth_free_threshold so both
# gates agree on the "free growth" boundary. slope=1.0 means "one extra
# task required per +100% growth above the free threshold."
CL_PRIMARY_GROWTH_FREE_THRESHOLD = 0.20
CL_PRIMARY_GROWTH_SLOPE = 1.0
CL_PRIMARY_SYNTH_TOLERANCE = 0.05


def _check_cl_primary_gate(
    *,
    baseline_cl_passes: int,
    evolved_cl_passes: int,
    baseline_synth_mean: float,
    evolved_synth_mean: float,
    growth_pct: float,
    synth_tolerance: float = CL_PRIMARY_SYNTH_TOLERANCE,
) -> ConstraintResult:
    """Deploy-gate decision rule used when the saturation pre-flight
    classifies the run as ``weak_signal`` (synthetic judge saturated,
    closed-loop signal has a gradient).

    ACCEPT iff (gain >= required_gain) AND (synthetic not catastrophically
    collapsed). ``required_gain`` scales with description growth so a
    +1 task win can't deploy +400% wallpaper.

    Parameters are scalars (not SaturationReport) so this helper is
    independent of the preflight subsystem and trivially unit-testable.
    Returns the standard ``ConstraintResult`` so the deploy gate's
    existing aggregation code works without changes.
    """
    cl_gain = evolved_cl_passes - baseline_cl_passes
    required_gain = max(
        1,
        math.ceil(
            max(0.0, CL_PRIMARY_GROWTH_SLOPE * (growth_pct - CL_PRIMARY_GROWTH_FREE_THRESHOLD))
        ),
    )
    synth_delta = evolved_synth_mean - baseline_synth_mean
    synth_passed = synth_delta >= -synth_tolerance

    if cl_gain < required_gain:
        return ConstraintResult(
            passed=False,
            constraint_name="cl_primary_gate",
            message=(
                f"CL gained {cl_gain:+d} tasks but required {required_gain} "
                f"for {growth_pct:+.2%} growth"
            ),
        )
    if not synth_passed:
        return ConstraintResult(
            passed=False,
            constraint_name="cl_primary_gate",
            message=(
                f"CL gained {cl_gain:+d} tasks but synthetic regressed "
                f"{synth_delta:+.3f} > tolerance {synth_tolerance:.3f}"
            ),
        )
    return ConstraintResult(
        passed=True,
        constraint_name="cl_primary_gate",
        message=(
            f"CL gained +{cl_gain} tasks (required {required_gain}); "
            f"synth Δ {synth_delta:+.3f} within ±{synth_tolerance:.3f}"
        ),
    )


def append_cl_decision_fields(
    decision_payload: dict,
    *,
    cached_baseline_cl_per_example: list[float],
    evolved_cl_per_example: list[float],
    avg_baseline: float,
    avg_evolved: float,
    growth_pct: float,
    cl_eval_cost_usd: float,
    preflight_holdout_score: Optional[float],
    preflight_cl_score: Optional[float],
    closed_loop_agent_model: str,
) -> None:
    """Append the closed-loop deploy-gate decision fields to ``decision_payload``."""
    decision_payload["baseline_closed_loop_per_example"] = cached_baseline_cl_per_example
    decision_payload["evolved_closed_loop_per_example"] = evolved_cl_per_example
    decision_payload["evolved_closed_loop_errored_tasks"] = []
    decision_payload["cl_tasks_gained"] = (
        int(sum(evolved_cl_per_example)) - int(sum(cached_baseline_cl_per_example))
    )
    decision_payload["cl_required_gain"] = max(
        1,
        math.ceil(
            max(0.0, CL_PRIMARY_GROWTH_SLOPE * (growth_pct - CL_PRIMARY_GROWTH_FREE_THRESHOLD))
        ),
    )
    decision_payload["synthetic_sanity_check"] = {
        "tolerance": CL_PRIMARY_SYNTH_TOLERANCE,
        "baseline_mean": avg_baseline,
        "evolved_mean": avg_evolved,
        "passed": (avg_evolved - avg_baseline) >= -CL_PRIMARY_SYNTH_TOLERANCE,
    }
    decision_payload["evolved_cl_eval_cost_usd"] = cl_eval_cost_usd
    decision_payload["band_trigger_score"] = {
        "holdout": preflight_holdout_score,
        "closed_loop": preflight_cl_score,
    }
    decision_payload["validator_agent_model"] = closed_loop_agent_model


# `default` is calibrated against the obsidian deploy (+24.2% growth,
# ~+0.07 expected improvement). `off` disables the slope/ceiling checks
# but still enforces bootstrap.mean ≥ 0 — see deprecation warning when
# users select it. `non-inferiority` is the recommended preset for
# compression-focused runs: it ships variants statistically not-worse-
# than-baseline by more than ``inferiority_tolerance``.
#
# Type widens to ``Any`` because ``gate_mode`` is a string and
# ``inferiority_tolerance`` is a float — no longer a uniform float dict.
QUALITY_GATE_PRESETS: dict[str, dict[str, Any]] = {
    "strict": {
        "growth_free_threshold": 0.10,
        "growth_quality_slope": 0.50,
        "max_absolute_chars": 3000,
    },
    "default": {
        "growth_free_threshold": 0.20,
        "growth_quality_slope": 0.30,
        "max_absolute_chars": 5000,
    },
    "lenient": {
        "growth_free_threshold": 0.30,
        "growth_quality_slope": 0.20,
        "max_absolute_chars": 8000,
    },
    "off": {
        "growth_free_threshold": 100.0,
        "growth_quality_slope": 0.0,
        "max_absolute_chars": 100_000,
    },
    "non-inferiority": {
        "growth_free_threshold": 100.0,
        "growth_quality_slope": 0.0,
        "max_absolute_chars": 100_000,
        "gate_mode": "non_inferiority",
        # See reports/calibration_findings.md for the tolerance sweep.
        "inferiority_tolerance": 0.05,
    },
}


def write_gate_decision(output_dir: Path, decision: dict[str, Any]) -> Path:
    """Persist the deploy-gate's structured decision for future calibration.

    Each run writes one of these regardless of outcome (deploy or reject).
    Recalibrating the curve is then `jq -s '...' output/*/*/gate_decision.json`
    rather than parsing free-form failure notes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gate_decision.json"
    path.write_text(json.dumps(decision, indent=2))
    return path


def write_cost_ceiling_abort(
    exc: CostCeilingExceeded,
    *,
    output_dir: Path,
    run_inputs: dict[str, Any],
    extra_fields: dict[str, Any] | None = None,
    schema_version: str = "4",
) -> Path:
    """Write a ``decision="aborted"`` gate_decision for a cost-ceiling trip.

    ``extra_fields`` lets callers add path-specific keys (e.g.,
    ``artifact_type``, ``target_tool``). ``schema_version`` defaults to
    ``"4"`` so skill-side callers (which haven't bumped past v4 yet) keep
    working unchanged; tool-side callers pass ``"5"`` to stay consistent
    with the rest of the gate_decision write sites in that ``output_dir``.
    """
    cost_summary = COST_LEDGER.summary()
    _console.print(
        f"\n[bold red]✗ Aborting: cost ${exc.total_usd:.4f} exceeded "
        f"ceiling ${exc.ceiling_usd:.4f}[/bold red]"
    )
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "decision": "aborted",
        "reason": "cost_ceiling_exceeded",
        "cost_ceiling_usd": exc.ceiling_usd,
        "cost_at_abort_usd": exc.total_usd,
        "cost_summary": cost_summary,
        "run_inputs": run_inputs,
    }
    if extra_fields:
        payload.update(extra_fields)
    return write_gate_decision(output_dir, payload)


def run_benchmark_hook(
    cmd: str,
    *,
    timeout_seconds: int,
    evolved_path: Path,
    baseline_path: Path,
    output_dir: Path,
    target_name: str,
    artifact_type: str,
) -> dict[str, Any]:
    """Run the user's ``--benchmark-cmd`` and report the outcome.

    ``shell=True`` because the user wrote the command and runs the CLI
    on their own machine — no untrusted-input pipeline; forcing
    ``shlex.split`` for the user's own shell pipelines is friction for
    no security gain.
    """
    env = {
        **os.environ,
        "EVOLVED_PATH": str(evolved_path),
        "BASELINE_PATH": str(baseline_path),
        "RUN_DIR": str(output_dir),
        "TARGET_NAME": target_name,
        "ARTIFACT_TYPE": artifact_type,
    }

    _console.print(f"\n[bold]Running benchmark hook[/bold] (timeout {timeout_seconds}s)")
    _console.print(f"  [dim]$ {cmd}[/dim]")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(Path.cwd()),
            env=env,
        )
        duration = time.time() - start
        passed = result.returncode == 0
        block = {
            "command": cmd,
            "exit_code": result.returncode,
            "duration_seconds": round(duration, 3),
            "stdout_tail": (result.stdout or "")[-_BENCHMARK_OUTPUT_TAIL_BYTES:],
            "stderr_tail": (result.stderr or "")[-_BENCHMARK_OUTPUT_TAIL_BYTES:],
            "passed": passed,
            "reason": "ok" if passed else "exit_nonzero",
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - start
        block = {
            "command": cmd,
            "exit_code": None,
            "duration_seconds": round(duration, 3),
            "stdout_tail": (exc.stdout or "")[-_BENCHMARK_OUTPUT_TAIL_BYTES:] if exc.stdout else "",
            "stderr_tail": (exc.stderr or "")[-_BENCHMARK_OUTPUT_TAIL_BYTES:] if exc.stderr else "",
            "passed": False,
            "reason": "timeout",
        }
    except OSError as exc:
        # shell=True normally suppresses FileNotFoundError (sh handles it
        # as exit 127), but other OSErrors (permission denied, etc.) can
        # still surface. Treat any spawn-side failure as a benchmark fail.
        duration = time.time() - start
        block = {
            "command": cmd,
            "exit_code": None,
            "duration_seconds": round(duration, 3),
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "passed": False,
            "reason": "command_error",
        }

    if block["passed"]:
        _console.print(
            f"  [green]✓ benchmark passed[/green] "
            f"(exit 0, {block['duration_seconds']:.1f}s)"
        )
    else:
        _console.print(
            f"  [red]✗ benchmark failed: {block['reason']}[/red] "
            f"(exit_code={block['exit_code']}, duration={block['duration_seconds']:.1f}s)"
        )
    return block


def resolve_proposer_mode(fitness_profile: str) -> ProposerMode:
    """Map the user's fitness profile to a proposer mode.

    Each profile selects its own template: `growth` swings the proposer toward
    additions, `compression` toward cuts, and `balanced` toward direction-agnostic
    revisions. Unknown values fall back to compression-mode defensively.
    """
    if fitness_profile == "growth":
        return "growth"
    if fitness_profile == "balanced":
        return "balanced"
    return "compression"  # compression profile, plus defensive fallback for unknown

"""Saturation pre-flight: detect doomed evolve_* runs before GEPA spends budget.

Mirrors the shape of evolution.core.auth_check: a pure helper that
returns a structured report. Call sites in evolve_skill / evolve_tool
render a Rich panel and decide whether to prompt or default-deny.

See reports/calibration_findings.md for the calibration data behind DEFAULT_THRESHOLDS.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, TypeAlias

import dspy
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

SaturationBand: TypeAlias = Literal[
    "healthy", "no_headroom", "weak_signal", "uniform_failure"
]

DEFAULT_THRESHOLDS: dict[str, float] = {
    "no_headroom_synthetic": 0.99,
    "weak_signal_synthetic": 0.95,
    "no_headroom_closed_loop": 0.95,
    "uniform_failure_closed_loop": 0.15,
}


@dataclass
class SaturationReport:
    band: SaturationBand
    holdout_score: float
    holdout_n: int
    holdout_per_example: list[float]
    closed_loop_score: Optional[float] = None
    closed_loop_n: Optional[int] = None
    closed_loop_per_example: Optional[list[float]] = None
    suggestions: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    # A/A noise floor (loaded from <suite>.noise.json if a calibration was run),
    # so the gate's measured intrinsic noise surfaces in every pre-flight.
    noise: Optional[dict] = None


def _classify_band(
    *,
    holdout_score: float,
    closed_loop_score: Optional[float],
    thresholds: dict[str, float],
) -> tuple[SaturationBand, list[str]]:
    """Categorize a (synthetic, closed-loop) score pair into a band.

    Returns (band, suggestions_to_show_user).
    """
    no_head_syn = thresholds["no_headroom_synthetic"]
    weak_syn = thresholds["weak_signal_synthetic"]
    no_head_cl = thresholds["no_headroom_closed_loop"]
    uniform_cl = thresholds["uniform_failure_closed_loop"]

    if closed_loop_score is not None and closed_loop_score <= uniform_cl:
        return "uniform_failure", [
            "Baseline scored 0 on every behavioral task — GEPA has nothing to optimize for.",
            "First check the validator actually ran: look in run.log for a 'Stripped LiteLLM provider prefix' line confirming --closed-loop-agent-model routed correctly, and for a non-zero number of subprocess LM calls.",
            "If the validator did run: try a stronger --closed-loop-agent-model, or harden the suite tasks so failure modes are interesting rather than 'model can't execute the task.'",
        ]

    synthetic_saturated = holdout_score >= no_head_syn
    synthetic_close = holdout_score >= weak_syn
    cl_saturated = (
        closed_loop_score is not None and closed_loop_score >= no_head_cl
    )
    no_cl_signal = closed_loop_score is None

    # no_headroom triggers when:
    #   - synthetic alone is saturated and there's no closed-loop signal
    #     (only signal available is judge, and it's pegged), OR
    #   - closed-loop is saturated AND synthetic is close enough (≥ weak
    #     threshold) that the judge isn't producing a useful gradient either.
    # CL-saturated alone with a low synthetic (< weak_syn) does NOT trigger:
    # there's real judge signal to optimize over even when behavioral is
    # pegged, and that scenario usually means a misconfigured eval rather
    # than true saturation.
    if (synthetic_saturated and no_cl_signal) or (
        cl_saturated and synthetic_close
    ):
        return "no_headroom", [
            "Baseline already saturates the eval. No measurable headroom to evolve into.",
            "Try a harder closed-loop suite, or pick a different optimization target.",
            "Sanity check: is the synthetic generator producing trivially-correct tasks?",
        ]

    if (
        holdout_score >= weak_syn
        and closed_loop_score is not None
        and uniform_cl < closed_loop_score < no_head_cl
    ):
        return "weak_signal", [
            "Judge saturating but closed-loop has signal; GEPA's small-minibatch acceptance will struggle.",
            "Try --gepa-minibatch-size 8 (default 3) — widens the sampling window so discriminating examples appear in ~68% of minibatches vs ~34% at default.",
            "Larger minibatch means fewer proposals per budget: on evolve_tool bump --iterations to ~10, on evolve_skill use --budget heavy.",
        ]

    return "healthy", []


def _score_baseline_on_holdout(
    *,
    baseline_module,
    holdout_examples: list,
    metric,
    lm,
) -> tuple[float, list[float]]:
    """Run dspy.Evaluate on the baseline, return (mean, per_example_scores).

    Carved out as its own helper so tests can patch it without touching DSPy
    plumbing. Shape matches _holdout_evaluate_with_metric in evolve_*.py.
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
        result = evaluator(baseline_module)
    mean = float(result.score) / 100.0
    per_example = [float(s) for _, _, s in result.results]
    return mean, per_example


def saturation_preflight(
    *,
    baseline_module,
    holdout_examples: list,
    metric,
    lm,
    closed_loop_cache=None,
    baseline_artifact_text: Optional[str] = None,
    thresholds: Optional[dict[str, float]] = None,
    suite_path: Optional[Path] = None,
) -> SaturationReport:
    """Score baseline on holdout (and closed-loop suite if cache provided),
    classify into a band, return a report.

    Does the work — LM eval of the baseline, optional closed-loop validator
    fire via ``force_run``, cache mutation, possibly a subprocess. The
    "purity" we care about is at a higher layer: this function doesn't
    render panels, prompt for confirmation, or exit. Call sites own those.
    """
    if not holdout_examples:
        raise ValueError("holdout_examples is empty; nothing to score")
    thresholds = thresholds if thresholds is not None else dict(DEFAULT_THRESHOLDS)

    holdout_mean, holdout_per_example = _score_baseline_on_holdout(
        baseline_module=baseline_module,
        holdout_examples=holdout_examples,
        metric=metric,
        lm=lm,
    )

    closed_loop_mean: Optional[float] = None
    closed_loop_n: Optional[int] = None
    closed_loop_per_example: Optional[list[float]] = None
    if closed_loop_cache is not None:
        if baseline_artifact_text is None:
            raise ValueError(
                "baseline_artifact_text is required when closed_loop_cache is provided"
            )
        report = closed_loop_cache.force_run(baseline_artifact_text)
        per_example = [1.0 if t.passed else 0.0 for t in report.evolved.tasks]
        closed_loop_per_example = per_example
        closed_loop_n = len(per_example)
        closed_loop_mean = sum(per_example) / len(per_example) if per_example else 0.0

    band, suggestions = _classify_band(
        holdout_score=holdout_mean,
        closed_loop_score=closed_loop_mean,
        thresholds=thresholds,
    )

    noise: Optional[dict] = None
    if suite_path is not None:
        # Lazy import: keeps the validation package out of core's import graph
        # except when a noise sidecar is actually consulted.
        from evolution.validation.noise_calibration import load_noise_sidecar

        noise = load_noise_sidecar(suite_path)

    return SaturationReport(
        band=band,
        holdout_score=holdout_mean,
        holdout_n=len(holdout_per_example),
        holdout_per_example=holdout_per_example,
        closed_loop_score=closed_loop_mean,
        closed_loop_n=closed_loop_n,
        closed_loop_per_example=closed_loop_per_example,
        suggestions=suggestions,
        thresholds=dict(thresholds),
        noise=noise,
    )


_BAND_TITLES: dict[SaturationBand, str] = {
    "healthy": "Saturation check passed",
    "no_headroom": "No measurable headroom",
    "weak_signal": "Weak signal — expect a hard run",
    "uniform_failure": "Uniform failure — closed-loop scored zero on every task",
}

_BAND_STYLES: dict[SaturationBand, str] = {
    "healthy": "green",
    "no_headroom": "yellow",
    "weak_signal": "yellow",
    "uniform_failure": "yellow",
}


def _noise_line(noise: Optional[dict]) -> Optional[str]:
    """One-line A/A noise-floor summary, or None when no sidecar was loaded."""
    if not noise:
        return None
    return (
        f"Noise floor: spurious strict-win {noise['spurious_strict_win_rate']:.0%}, "
        f"mean per-task flip {noise['mean_per_task_flip']:.0%} "
        f"({noise['runs']} A/A run(s), reps={noise['reps']})"
    )


def render_saturation_panel(
    report: SaturationReport, *, console: Optional[Console] = None,
) -> None:
    """Print a Rich panel to ``console`` (or default stdout) summarizing the report.

    Healthy band: one-line acknowledgement. Warn bands: full panel with
    scores + band-specific suggestions.
    """
    if console is None:
        console = Console()

    if report.band == "healthy":
        console.print(
            f"[dim]Saturation check passed (holdout={report.holdout_score:.3f}"
            + (
                f", closed-loop={report.closed_loop_score:.3f}"
                if report.closed_loop_score is not None
                else ""
            )
            + ").[/dim]"
        )
        noise_line = _noise_line(report.noise)
        if noise_line is not None:
            console.print(f"[dim]{noise_line}[/dim]")
        return

    body = Text()
    body.append(f"Band: {report.band}\n", style="bold")
    body.append(f"Holdout (synthetic): {report.holdout_score:.3f} over {report.holdout_n} examples\n")
    if report.closed_loop_score is not None:
        body.append(
            f"Closed-loop (behavioral): {report.closed_loop_score:.3f} over {report.closed_loop_n} tasks\n"
        )
    noise_line = _noise_line(report.noise)
    if noise_line is not None:
        body.append(f"{noise_line}\n")
    body.append("\nSuggestions:\n", style="bold")
    for s in report.suggestions:
        body.append(f"  • {s}\n")

    console.print(
        Panel(
            body,
            title=_BAND_TITLES[report.band],
            border_style=_BAND_STYLES[report.band],
        )
    )


def is_non_interactive() -> bool:
    """True when stdin isn't a TTY. Used by call sites to decide between
    prompting for y/N and printing the override-flag hint."""
    return not sys.stdin.isatty()


def interactive_confirm(prompt: str = "Continue anyway? [y/N] ") -> bool:
    """Read one line from stdin; return True only for {y, yes} case-insensitive.

    Ctrl-C and stdin EOF both → False (treat like 'n', no traceback noise).
    The EOF branch matters in practice when stdin is piped from ``/dev/null``
    or a closed pipe.
    """
    try:
        answer = input(prompt)
    except (KeyboardInterrupt, EOFError):
        return False
    return answer.strip().lower() in {"y", "yes"}

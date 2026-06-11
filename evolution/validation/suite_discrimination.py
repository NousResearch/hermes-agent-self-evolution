"""Per-task discrimination labeler — a suite-authoring diagnostic.

The binding constraint on this framework is authoring suites whose tasks
actually discriminate (headroom is a property of the suite, not the artifact).
The saturation pre-flight's only advice today is the dead-end "try a harder
suite"; this turns that into a per-task verdict.

Given a baseline artifact (and optionally a strong "ceiling" artifact) it scores
each task across reps and labels it. Discrimination is inherently a
baseline-vs-ceiling question: baseline pass-rate alone cannot tell an
*unfillable* task (no prompt lifts it) from a *fillable-but-baseline-fails* one
(a strong artifact reaches it) — only the ceiling separates them. The A/A noise
floor (per-task flip) further flags tasks whose achievable movement is within
noise, so the deploy gate could never see the gain.

Reuses the A/A machinery: one ClosedLoopValidator.validate(baseline==baseline)
run pools both phases for per-task pass-rate + flip (exactly as aggregate_noise
pools), at no extra agent cost.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from evolution.validation.report import ValidationReport
from evolution.validation.task import TaskSuite
from evolution.validation.validator import ValidationInputs

DISCRIMINATION_SIDECAR_SUFFIX = ".discrimination.json"

# Labels. too_easy / discriminative are usable with baseline alone; the rest
# require a ceiling artifact to be assigned.
TOO_EASY = "too_easy"
DISCRIMINATIVE = "discriminative"
UNFILLABLE = "unfillable"
NOISE_LIMITED = "noise_limited"
BASELINE_FAILS = "baseline_fails"  # baseline-only: low baseline, headroom unknown


def classify_task(
    baseline_rate: float,
    *,
    ceiling_rate: Optional[float] = None,
    flip: Optional[float] = None,
    high: float = 0.9,
    low: float = 0.15,
) -> str:
    """Label one task from its baseline pass-rate (+ optional ceiling, flip).

    Without a ceiling the verdict is honest-but-coarse: ``too_easy`` when the
    baseline already saturates, ``baseline_fails`` when it's at/below the floor
    (headroom unknown — could be unfillable or fillable), else ``discriminative``.

    With a ceiling we can separate ``unfillable`` (nothing lifts it) from a real
    ``discriminative`` task (a strong artifact reaches it), and downgrade to
    ``noise_limited`` when the fillable gain is within the A/A noise floor.
    """
    if baseline_rate >= high:
        return TOO_EASY
    if ceiling_rate is None:
        return BASELINE_FAILS if baseline_rate <= low else DISCRIMINATIVE
    if ceiling_rate <= low:
        return UNFILLABLE
    gain = ceiling_rate - baseline_rate
    if flip is not None and gain <= flip:
        return NOISE_LIMITED
    return DISCRIMINATIVE if ceiling_rate >= high or gain > 0 else UNFILLABLE


@dataclass(frozen=True)
class DiscriminationReport:
    labels: dict[str, str]
    baseline_rates: dict[str, float]
    ceiling_rates: Optional[dict[str, float]]
    flips: Optional[dict[str, float]]
    summary: dict[str, int]
    reps: int
    suite_sha256: str
    agent_model: Optional[str] = None
    recommendation: str = ""
    per_task: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discrimination_report(
    baseline_rates: dict[str, float],
    *,
    ceiling_rates: Optional[dict[str, float]] = None,
    flips: Optional[dict[str, float]] = None,
    reps: int = 1,
    suite_sha256: str = "",
    agent_model: Optional[str] = None,
    high: float = 0.9,
    low: float = 0.15,
) -> DiscriminationReport:
    """Classify every task and summarize the suite's discriminating power."""
    labels: dict[str, str] = {}
    per_task: dict[str, dict[str, Any]] = {}
    for tid, b in baseline_rates.items():
        c = ceiling_rates.get(tid) if ceiling_rates else None
        f = flips.get(tid) if flips else None
        label = classify_task(b, ceiling_rate=c, flip=f, high=high, low=low)
        labels[tid] = label
        per_task[tid] = {"baseline_rate": b, "ceiling_rate": c, "flip": f, "label": label}

    summary: dict[str, int] = {}
    for label in labels.values():
        summary[label] = summary.get(label, 0) + 1
    n = len(labels)
    n_disc = summary.get(DISCRIMINATIVE, 0)
    if n == 0:
        rec = "Empty suite."
    elif n_disc == 0:
        rec = (
            f"0/{n} tasks discriminate — every task is saturated, unfillable, or "
            f"noise-limited. This suite cannot justify search spend; author "
            f"discriminating tasks (baseline fails, a strong artifact succeeds, "
            f"gain above the noise floor)."
        )
    else:
        rec = f"{n_disc}/{n} tasks discriminate; the rest add no usable signal."

    return DiscriminationReport(
        labels=labels,
        baseline_rates=baseline_rates,
        ceiling_rates=ceiling_rates,
        flips=flips,
        summary=summary,
        reps=reps,
        suite_sha256=suite_sha256,
        agent_model=agent_model,
        recommendation=rec,
        per_task=per_task,
    )


def _pooled_rates_and_flips(
    report: ValidationReport,
) -> tuple[dict[str, float], dict[str, float]]:
    """Pool an A/A report's two phases per task → (pass_rate, flip).

    Baseline and evolved are independent samples of the same artifact, so
    pooling doubles the per-task sample for free. Abstentions are excluded.
    """
    verdicts: dict[str, list[bool]] = {}
    for t in list(report.baseline.tasks) + list(report.evolved.tasks):
        if t.abstained:
            continue
        verdicts.setdefault(t.task_id, []).append(bool(t.passed))
    rates: dict[str, float] = {}
    flips: dict[str, float] = {}
    for tid, vs in verdicts.items():
        p = sum(vs) / len(vs)
        rates[tid] = p
        flips[tid] = min(p, 1.0 - p)
    return rates, flips


def probe_discrimination(
    validator,
    suite: TaskSuite,
    baseline_artifact: Path,
    *,
    ceiling_artifact: Optional[Path] = None,
    reps: int = 1,
    agent_model: Optional[str] = None,
    tool_name: str = "suite_discrimination",
    high: float = 0.95,
    low: float = 0.15,
) -> DiscriminationReport:
    """Score the suite on the baseline (A/A) and optional ceiling, then label.

    ``validator`` is a ClosedLoopValidator (reps preset). The baseline run is an
    A/A no-op splice; its two phases pool for per-task baseline-rate + flip. The
    ceiling run (if any) is the same on the ceiling artifact, taking its rate.
    """
    base_report = validator.validate(
        ValidationInputs(
            tool_name=tool_name, suite=suite,
            baseline_artifact=baseline_artifact, evolved_artifact=baseline_artifact,
        )
    )
    baseline_rates, flips = _pooled_rates_and_flips(base_report)

    ceiling_rates: Optional[dict[str, float]] = None
    if ceiling_artifact is not None:
        ceil_report = validator.validate(
            ValidationInputs(
                tool_name=tool_name, suite=suite,
                baseline_artifact=ceiling_artifact, evolved_artifact=ceiling_artifact,
            )
        )
        ceiling_rates, _ = _pooled_rates_and_flips(ceil_report)

    return discrimination_report(
        baseline_rates, ceiling_rates=ceiling_rates, flips=flips,
        reps=validator.reps, suite_sha256=suite.sha256, agent_model=agent_model,
        high=high, low=low,
    )


def discrimination_sidecar_path(suite_path: Path) -> Path:
    suite_path = Path(suite_path)
    return suite_path.with_name(suite_path.name + DISCRIMINATION_SIDECAR_SUFFIX)


def write_discrimination_sidecar(report: DiscriminationReport, suite_path: Path) -> Path:
    path = discrimination_sidecar_path(suite_path)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _summary_text(report: DiscriminationReport) -> str:
    lines = [
        f"Suite discrimination ({len(report.labels)} tasks, reps={report.reps}"
        + (f", model={report.agent_model}" if report.agent_model else "") + ")",
        f"{'task':28}{'baseline':>10}{'ceiling':>10}{'flip':>8}  label",
    ]
    for tid in sorted(report.per_task):
        r = report.per_task[tid]
        c = "—" if r["ceiling_rate"] is None else f"{r['ceiling_rate']:.2f}"
        f = "—" if r["flip"] is None else f"{r['flip']:.2f}"
        lines.append(f"{tid:28}{r['baseline_rate']:>10.2f}{c:>10}{f:>8}  {r['label']}")
    lines.append("")
    lines.append("counts: " + ", ".join(f"{k}={v}" for k, v in sorted(report.summary.items())))
    lines.append(report.recommendation)
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from evolution.core.lm_timing_callback import COST_LEDGER, CostCeilingExceeded
    from evolution.prompts.backend import build_backend
    from evolution.validation.validator import ClosedLoopValidator

    parser = argparse.ArgumentParser(
        description="Label each suite task too_easy/discriminative/unfillable/noise_limited."
    )
    parser.add_argument("--target", choices=["hermes", "claude"], required=True)
    parser.add_argument("--section", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--hermes-repo", type=Path, default=None)
    parser.add_argument("--claude-md", type=Path, default=None)
    parser.add_argument("--baseline-override-file", type=Path, default=None)
    parser.add_argument(
        "--ceiling-artifact-file", type=Path, default=None,
        help="A strong artifact; separates fillable (discriminative) from unfillable tasks.",
    )
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--task-timeout-seconds", type=int, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output") / "suite_discrimination",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = build_backend(
        args.target, section_name=args.section, hermes_repo=args.hermes_repo,
        claude_md=args.claude_md, output_dir=args.output_dir, agent_model=args.agent_model,
        task_timeout_seconds=args.task_timeout_seconds,
        baseline_override_file=args.baseline_override_file,
    )
    suite = TaskSuite.from_jsonl(args.tasks)
    baseline_artifact = args.output_dir / "baseline.txt"
    baseline_artifact.write_text(backend.baseline_text, encoding="utf-8")
    ceiling_artifact = args.ceiling_artifact_file

    validator = ClosedLoopValidator(backend.installer, backend.runner, reps=args.reps)
    if args.max_cost_usd is not None:
        COST_LEDGER.set_ceiling(args.max_cost_usd)

    try:
        report = probe_discrimination(
            validator, suite, baseline_artifact,
            ceiling_artifact=ceiling_artifact, reps=args.reps, agent_model=args.agent_model,
        )
    except CostCeilingExceeded as exc:
        print(f"Cost ceiling hit: {exc}")
        return 2

    print(_summary_text(report))
    sidecar = write_discrimination_sidecar(report, args.tasks)
    print(f"\nWrote {sidecar}")
    print(f"Cost: {json.dumps(COST_LEDGER.summary())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

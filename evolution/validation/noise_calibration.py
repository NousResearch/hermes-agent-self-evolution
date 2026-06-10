"""A/A noise-floor probe for the closed-loop deploy gate.

The deploy gate (``compute_win_loss`` + ``decide``) scores *any* upward
per-task pass_rate movement as a strict win, with no within-noise tolerance.
On a tiny suite at a few reps, agent stochasticity alone can therefore
manufacture "wins" and "regressions" between two runs of the *same* artifact.
This probe measures that floor: it runs the suite with baseline and evolved
pointed at one identical artifact (a no-op splice), ``runs`` times, and reports

  - ``spurious_strict_win_rate``  — fraction of A/A runs that recorded ≥1 win,
  - ``spurious_regression_rate``  — fraction the gate would have called a regression,
  - ``mean_per_task_flip``        — mean over tasks of the minority-verdict
                                    fraction (0 = perfectly stable, 0.5 = a coin
                                    flip), pooled over runs and both A/A phases.

These are the calibration constants every downstream deploy threshold silently
assumes. Pass-rate "false-positive rate" is meaningless here (ties pass by
design), so we measure strict-wins / regressions / flips, not pass-rate FP.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from evolution.validation.report import ValidationReport
from evolution.validation.task import TaskSuite
from evolution.validation.validator import ValidationInputs

NOISE_SIDECAR_SUFFIX = ".noise.json"


@dataclass(frozen=True)
class NoiseReport:
    spurious_strict_win_rate: float
    spurious_regression_rate: float
    mean_per_task_flip: float
    per_task_flip: dict[str, float]
    runs: int
    reps: int
    suite_sha256: str
    agent_model: Optional[str] = None
    aborted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _SupportsValidate(Protocol):
    reps: int

    def validate(self, inputs: ValidationInputs) -> ValidationReport: ...


def aggregate_noise(
    reports: list[ValidationReport],
    *,
    reps: int,
    suite_sha256: str,
    agent_model: Optional[str] = None,
    aborted: bool = False,
) -> NoiseReport:
    """Reduce a list of A/A ValidationReports to a NoiseReport.

    Pure — no agent or I/O. ``mean_per_task_flip`` pools each task's per-phase
    ``passed`` verdicts across every run and both phases (baseline and evolved
    are independent A/A samples of the same artifact); abstentions are excluded.
    """
    runs = len(reports)
    n_strict_win = sum(1 for r in reports if r.delta.n_wins > 0)
    n_regression = sum(1 for r in reports if r.decision == "regression")

    verdicts: dict[str, list[bool]] = {}
    for r in reports:
        for t in list(r.baseline.tasks) + list(r.evolved.tasks):
            if t.abstained:
                continue
            verdicts.setdefault(t.task_id, []).append(bool(t.passed))

    per_task_flip: dict[str, float] = {}
    for tid, vs in verdicts.items():
        p = sum(vs) / len(vs)
        per_task_flip[tid] = min(p, 1.0 - p)
    mean_flip = statistics.mean(per_task_flip.values()) if per_task_flip else 0.0

    return NoiseReport(
        spurious_strict_win_rate=(n_strict_win / runs) if runs else 0.0,
        spurious_regression_rate=(n_regression / runs) if runs else 0.0,
        mean_per_task_flip=mean_flip,
        per_task_flip=per_task_flip,
        runs=runs,
        reps=reps,
        suite_sha256=suite_sha256,
        agent_model=agent_model,
        aborted=aborted,
    )


def calibrate_noise(
    validator: _SupportsValidate,
    suite: TaskSuite,
    artifact: Path,
    *,
    runs: int,
    agent_model: Optional[str] = None,
    tool_name: str = "noise_calibration",
) -> NoiseReport:
    """Run the A/A probe ``runs`` times and aggregate.

    ``artifact`` is installed into BOTH the baseline and evolved slots — a legal
    no-op splice, so any win/loss/flip the gate records is pure noise.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")
    reports: list[ValidationReport] = []
    for _ in range(runs):
        reports.append(
            validator.validate(
                ValidationInputs(
                    tool_name=tool_name,
                    suite=suite,
                    baseline_artifact=artifact,
                    evolved_artifact=artifact,
                )
            )
        )
    return aggregate_noise(
        reports,
        reps=validator.reps,
        suite_sha256=suite.sha256,
        agent_model=agent_model,
    )


def noise_sidecar_path(suite_path: Path) -> Path:
    """``<suite>.noise.json`` next to the suite file."""
    suite_path = Path(suite_path)
    return suite_path.with_name(suite_path.name + NOISE_SIDECAR_SUFFIX)


def write_noise_sidecar(report: NoiseReport, suite_path: Path) -> Path:
    path = noise_sidecar_path(suite_path)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_noise_sidecar(suite_path: Path) -> Optional[dict[str, Any]]:
    """Read ``<suite>.noise.json`` if present, else None."""
    path = noise_sidecar_path(suite_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_text(report: NoiseReport) -> str:
    lines = [
        f"A/A noise floor over {report.runs} run(s), reps={report.reps}"
        + (f", model={report.agent_model}" if report.agent_model else ""),
        f"  spurious strict-win rate: {report.spurious_strict_win_rate:.1%}",
        f"  spurious regression rate: {report.spurious_regression_rate:.1%}",
        f"  mean per-task flip:       {report.mean_per_task_flip:.1%}",
    ]
    if report.aborted:
        lines.append("  (ABORTED on cost ceiling — rates are over completed runs only)")
    flips = sorted(report.per_task_flip.items(), key=lambda kv: kv[1], reverse=True)
    if flips:
        lines.append("  per-task flip:")
        for tid, f in flips:
            lines.append(f"    {tid}: {f:.1%}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from evolution.core.lm_timing_callback import COST_LEDGER, CostCeilingExceeded
    from evolution.prompts.backend import build_backend
    from evolution.validation.validator import ClosedLoopValidator

    parser = argparse.ArgumentParser(
        description="A/A noise-floor probe for the closed-loop deploy gate."
    )
    parser.add_argument("--target", choices=["hermes", "claude"], required=True)
    parser.add_argument("--section", required=True, help="Prompt-section name.")
    parser.add_argument("--tasks", type=Path, required=True, help="Suite JSONL path.")
    parser.add_argument("--hermes-repo", type=Path, default=None)
    parser.add_argument("--claude-md", type=Path, default=None)
    parser.add_argument(
        "--baseline-override-file", type=Path, default=None,
        help="Seed text for the artifact (required for a brand-new CLAUDE.md region).",
    )
    parser.add_argument("--agent-model", default=None)
    parser.add_argument("--reps", type=int, default=1, help="Inner per-task reps.")
    parser.add_argument("--runs", type=int, default=10, help="Outer A/A repetitions (k).")
    parser.add_argument("--task-timeout-seconds", type=int, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output") / "noise_calibration",
        help="Scratch dir for the backend workdir + temp artifact.",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = build_backend(
        args.target,
        section_name=args.section,
        hermes_repo=args.hermes_repo,
        claude_md=args.claude_md,
        output_dir=args.output_dir,
        agent_model=args.agent_model,
        task_timeout_seconds=args.task_timeout_seconds,
        baseline_override_file=args.baseline_override_file,
    )

    suite = TaskSuite.from_jsonl(args.tasks)
    artifact = args.output_dir / "aa_artifact.txt"
    artifact.write_text(backend.baseline_text, encoding="utf-8")

    validator = ClosedLoopValidator(backend.installer, backend.runner, reps=args.reps)

    if args.max_cost_usd is not None:
        COST_LEDGER.set_ceiling(args.max_cost_usd)

    # On a cost-ceiling trip mid-probe, report the floor over whatever runs
    # completed rather than discarding the spend — a partial floor is still a
    # signal, and the report records aborted=True so it isn't read as final.
    reports: list[ValidationReport] = []
    aborted = False
    try:
        for _ in range(args.runs):
            reports.append(
                validator.validate(
                    ValidationInputs(
                        tool_name="noise_calibration",
                        suite=suite,
                        baseline_artifact=artifact,
                        evolved_artifact=artifact,
                    )
                )
            )
    except CostCeilingExceeded as exc:
        aborted = True
        print(f"Cost ceiling hit after {len(reports)} run(s): {exc}")

    if not reports:
        print("No runs completed — nothing to report.")
        return 1

    report = aggregate_noise(
        reports,
        reps=args.reps,
        suite_sha256=suite.sha256,
        agent_model=args.agent_model,
        aborted=aborted,
    )
    sidecar = write_noise_sidecar(report, args.tasks)
    print(_summary_text(report))
    print(f"\nWrote {sidecar}")
    print(f"Cost: {json.dumps(COST_LEDGER.summary())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

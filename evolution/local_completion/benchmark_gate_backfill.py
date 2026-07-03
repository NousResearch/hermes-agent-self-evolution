"""Benchmark gate backfill report writer for HSE strict PLAN promotion.

This module records the baseline/current subjects that need real benchmark
comparison before Phase 1/2 local active evidence can be promoted to strict
PLAN completion. It is intentionally non-executing: it does not run TBLite,
YC-Bench, TerminalBench, provider calls, GitHub writes, active applies, cron,
or gateway restart/reload.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags

B0_GATE_ID = "B0"
B0_PHASE = "Phase 1/2 Benchmark Gate Backfill"
B0_TARGET = "strict-plan-benchmark-regression-gate"
BENCHMARK_BACKFILL_SCHEMA_VERSION = "hse-benchmark-gate-backfill-v1"
BLOCKED_BY_BENCHMARK_APPROVAL = "BLOCKED_BY_BENCHMARK_APPROVAL"
READY_FOR_REAL_BENCHMARK_EXECUTION = "READY_FOR_REAL_BENCHMARK_EXECUTION"


def write_benchmark_gate_backfill(
    *,
    baseline_subject: Mapping[str, Any],
    current_subject: Mapping[str, Any],
    output_dir: str | Path,
    generated_at: str,
    real_benchmark_execution_approved: bool = False,
    approved_budget_usd: int | float = 0,
    approved_runtime_minutes: int | None = None,
    benchmark_plan: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Write a fail-closed benchmark backfill report and Markdown companion.

    The produced report is a readiness/blocker artifact, not benchmark evidence.
    It snapshots the intended baseline/current subjects and records whether real
    benchmark execution is approved. When approval is absent, the strict PLAN
    benchmark gate remains open and the status is
    ``BLOCKED_BY_BENCHMARK_APPROVAL``.
    """

    _validate_subject("baseline_subject", baseline_subject)
    _validate_subject("current_subject", current_subject)
    _require_non_empty("generated_at", generated_at)

    out = Path(output_dir).expanduser()
    inputs_dir = out / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    baseline_snapshot_path = inputs_dir / "baseline_subject.json"
    current_snapshot_path = inputs_dir / "current_subject.json"
    baseline_snapshot_path.write_text(json.dumps(dict(baseline_subject), indent=2, sort_keys=True) + "\n")
    current_snapshot_path.write_text(json.dumps(dict(current_subject), indent=2, sort_keys=True) + "\n")

    status = READY_FOR_REAL_BENCHMARK_EXECUTION if real_benchmark_execution_approved else BLOCKED_BY_BENCHMARK_APPROVAL
    blocked_reason = None if real_benchmark_execution_approved else "real benchmark execution approval is required before strict PLAN gate closure"

    report = base_decision_payload(
        gate_id=B0_GATE_ID,
        phase=B0_PHASE,
        target=B0_TARGET,
        generated_at=generated_at,
    )
    report["schema_version"] = BENCHMARK_BACKFILL_SCHEMA_VERSION
    report.update(
        {
            "status": status,
            "summary": _summary(status),
            "strict_plan_gate_closed": False,
            "benchmark_gate_passed": None,
            "real_benchmarks_executed": False,
            "real_benchmark_execution_approved": bool(real_benchmark_execution_approved),
            "current_authorized_budget_usd": approved_budget_usd,
            "approved_runtime_minutes": approved_runtime_minutes,
            "blocked_reason": blocked_reason,
            "required_next_action": _required_next_action(status),
            "benchmark_subjects": {
                "baseline": dict(baseline_subject),
                "current": dict(current_subject),
            },
            "benchmark_plan": dict(benchmark_plan or _default_benchmark_plan()),
            "input_snapshots": {
                "baseline_subject": {
                    "path": "inputs/baseline_subject.json",
                    "sha256": _sha256_path(baseline_snapshot_path),
                    "bytes": baseline_snapshot_path.stat().st_size,
                },
                "current_subject": {
                    "path": "inputs/current_subject.json",
                    "sha256": _sha256_path(current_snapshot_path),
                    "bytes": current_snapshot_path.stat().st_size,
                },
            },
            "artifacts": {
                "report": "benchmark_gate_backfill.json",
                "markdown": "benchmark_gate_backfill.md",
                "baseline_snapshot": "inputs/baseline_subject.json",
                "current_snapshot": "inputs/current_subject.json",
            },
            "boundaries": {
                "no_github_write": True,
                "github_policy": "NO_GITHUB_WRITE",
                "active_apply_performed": False,
                "runtime_restart_or_reload_performed": False,
                "cron_mutation_performed": False,
                "provider_or_model_spend_performed": False,
            },
            "required_before_strict_plan_promotion": [
                "obtain explicit real benchmark budget/runtime approval",
                "materialize or reference comparable baseline/current subjects",
                "run approved real benchmark suite with captured logs and artifacts",
                "compare baseline/current metrics for no regression or documented improvement",
                "write benchmark result manifest with immutable input/output hashes",
                "rerun focused HSE/Hermes regression checks after any source or evidence changes",
            ],
        }
    )
    reject_github_or_active_apply_flags(report)

    report_path = out / "benchmark_gate_backfill.json"
    markdown_path = out / "benchmark_gate_backfill.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_markdown(report))

    return {
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
        "baseline_snapshot_path": str(baseline_snapshot_path),
        "current_snapshot_path": str(current_snapshot_path),
    }


def _validate_subject(label: str, subject: Mapping[str, Any]) -> None:
    if not isinstance(subject, Mapping):
        raise ValueError(f"{label} must be an object")
    _require_non_empty(f"{label}.subject_id", subject.get("subject_id"))
    hermes_source = subject.get("hermes_source")
    if not isinstance(hermes_source, Mapping):
        raise ValueError(f"{label}.hermes_source must be an object")
    _require_non_empty(f"{label}.hermes_source.commit", hermes_source.get("commit"))


def _require_non_empty(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _summary(status: str) -> str:
    if status == READY_FOR_REAL_BENCHMARK_EXECUTION:
        return "Benchmark subjects are recorded; real benchmark execution is approved but has not run yet."
    return "Benchmark subjects are recorded; strict PLAN gate remains blocked until real benchmark execution is approved and completed."


def _required_next_action(status: str) -> str:
    if status == READY_FOR_REAL_BENCHMARK_EXECUTION:
        return "run_real_benchmarks_under_recorded_budget"
    return "request_real_benchmark_budget_runtime_approval"


def _default_benchmark_plan() -> dict[str, Any]:
    return {
        "suites": [
            {
                "name": "TBLite",
                "required_for": "Phase 1/2 no-regression evidence when available",
                "executed_now": False,
            },
            {
                "name": "YC-Bench",
                "required_for": "Phase 1/2 strict PLAN benchmark comparison when available",
                "executed_now": False,
            },
            {
                "name": "Hermes focused regression",
                "required_for": "local source health, not a substitute for real external benchmarks",
                "executed_now": False,
            },
        ],
        "comparison_policy": "baseline/current metrics must show no regression or documented improvement before strict PLAN promotion",
        "dry_run_or_fixture_results_count_as_strict_benchmark": False,
    }


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _render_markdown(report: Mapping[str, Any]) -> str:
    boundaries = report.get("boundaries", {}) if isinstance(report.get("boundaries"), Mapping) else {}
    subjects = report.get("benchmark_subjects", {}) if isinstance(report.get("benchmark_subjects"), Mapping) else {}
    baseline = subjects.get("baseline", {}) if isinstance(subjects.get("baseline"), Mapping) else {}
    current = subjects.get("current", {}) if isinstance(subjects.get("current"), Mapping) else {}
    baseline_source = baseline.get("hermes_source", {}) if isinstance(baseline.get("hermes_source"), Mapping) else {}
    current_source = current.get("hermes_source", {}) if isinstance(current.get("hermes_source"), Mapping) else {}
    required = report.get("required_before_strict_plan_promotion", [])
    required_lines = [f"- {item}" for item in required] if isinstance(required, list) else []
    return "\n".join(
        [
            "# HSE Benchmark Gate Backfill",
            "",
            f"Status: `{report.get('status')}`",
            "",
            "## Strict Gate State",
            "",
            f"- strict_plan_gate_closed={str(report.get('strict_plan_gate_closed')).lower()}",
            f"- benchmark_gate_passed={report.get('benchmark_gate_passed')}",
            f"- real_benchmarks_executed={str(report.get('real_benchmarks_executed')).lower()}",
            f"- real_benchmark_execution_approved={str(report.get('real_benchmark_execution_approved')).lower()}",
            f"- current_authorized_budget_usd={report.get('current_authorized_budget_usd')}",
            "",
            "## Subjects",
            "",
            f"- Baseline: `{baseline.get('subject_id')}` / Hermes commit `{baseline_source.get('commit')}`",
            f"- Current: `{current.get('subject_id')}` / Hermes commit `{current_source.get('commit')}`",
            "",
            "## Boundaries",
            "",
            f"- GitHub policy: `{boundaries.get('github_policy', 'NO_GITHUB_WRITE')}`",
            "- NO_GITHUB_WRITE",
            f"- Active apply performed: `{boundaries.get('active_apply_performed')}`",
            f"- Runtime restart/reload performed: `{boundaries.get('runtime_restart_or_reload_performed')}`",
            f"- Provider/model spend performed: `{boundaries.get('provider_or_model_spend_performed')}`",
            "",
            "## Required Before Strict PLAN Promotion",
            "",
            *required_lines,
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an HSE benchmark gate backfill report.")
    parser.add_argument("--baseline-subject", required=True, type=Path)
    parser.add_argument("--current-subject", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--approve-real-benchmark", action="store_true")
    parser.add_argument("--approved-budget-usd", type=float, default=0)
    parser.add_argument("--approved-runtime-minutes", type=int)
    args = parser.parse_args(argv)

    baseline = json.loads(args.baseline_subject.read_text())
    current = json.loads(args.current_subject.read_text())
    result = write_benchmark_gate_backfill(
        baseline_subject=baseline,
        current_subject=current,
        output_dir=args.output_dir,
        generated_at=args.generated_at,
        real_benchmark_execution_approved=args.approve_real_benchmark,
        approved_budget_usd=args.approved_budget_usd,
        approved_runtime_minutes=args.approved_runtime_minutes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

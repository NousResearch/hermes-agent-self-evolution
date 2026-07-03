"""Real benchmark approval packet writer for HSE strict benchmark execution.

This module prepares the human approval contract that must exist before HSE
runs real benchmark suites such as TBLite or YC-Bench. It is intentionally
non-executing: it records approval requirements and a fail-closed safety review,
but it does not start benchmarks, spend provider/model/API budget, query/write
GitHub, mutate active runtime state, or restart/reload Hermes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

from evolution.local_completion.benchmark_gate_backfill import BLOCKED_BY_BENCHMARK_APPROVAL
from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags

APPROVAL_PACKET_SCHEMA_VERSION = "hse-real-benchmark-approval-packet-v1"
SAFETY_REVIEW_SCHEMA_VERSION = "hse-real-benchmark-safety-review-v1"
RBA_GATE_ID = "B0-RBA"
RBA_PHASE = "Real Benchmark Approval Packet"
RBA_TARGET = "strict-plan-real-benchmark-approval"
AWAITING_EXPLICIT_BENCHMARK_APPROVAL = "AWAITING_EXPLICIT_BENCHMARK_APPROVAL"
APPROVAL_RECORDED_NOT_EXECUTED = "APPROVAL_RECORDED_NOT_EXECUTED"
SAFETY_REVIEW_PASS_FAIL_CLOSED = "PASS_PACKET_FAIL_CLOSED"
SAFETY_REVIEW_PASS_APPROVAL_RECORDED = "PASS_APPROVAL_RECORDED_NOT_EXECUTED"

REQUIRED_APPROVAL_FIELDS = [
    "benchmark_suites",
    "max_budget_usd_or_krw",
    "max_runtime_minutes",
    "network_provider_api_spend_allowed",
    "baseline_materialization_allowed",
    "current_materialization_allowed",
    "regression_thresholds",
    "allowed_write_roots",
    "rollback_plan",
    "human_approval_source",
]

DEFAULT_BENCHMARK_SUITES = ["TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples"]
DEFAULT_REGRESSION_THRESHOLDS = {
    "TBLite": "within_2_percent_or_better",
    "YC-Bench": "no_material_regression",
    "Phase2 PLAN-scale tool-selection triples": "no_aggregate_or_per_tool_regression_beyond_gate",
}


def write_real_benchmark_approval_packet(
    *,
    backfill_report_path: str | Path,
    output_dir: str | Path,
    generated_at: str,
    benchmark_suites: Sequence[str] | None = None,
    max_budget_usd: int | float | None = None,
    max_budget_krw: int | float | None = None,
    max_runtime_minutes: int | None = None,
    network_provider_api_spend_allowed: bool | None = None,
    baseline_materialization_allowed: bool | None = None,
    current_materialization_allowed: bool | None = None,
    human_approval_source: str | None = None,
    allowed_write_roots: Sequence[str] | None = None,
    rollback_plan: Mapping[str, Any] | None = None,
    regression_thresholds: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Write a non-executing real benchmark approval packet.

    If every approval field is present, the packet records explicit approval but
    still does not start execution. If any approval field is missing, the packet
    remains fail-closed with status ``AWAITING_EXPLICIT_BENCHMARK_APPROVAL``.
    """

    _require_non_empty("generated_at", generated_at)
    _validate_budget_limits(max_budget_usd=max_budget_usd, max_budget_krw=max_budget_krw)
    source_path = Path(backfill_report_path).expanduser()
    backfill = _load_backfill(source_path)
    _validate_backfill(backfill)

    out = Path(output_dir).expanduser()
    inputs_dir = out / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = inputs_dir / "benchmark_gate_backfill.json"
    snapshot_path.write_text(json.dumps(backfill, indent=2, sort_keys=True) + "\n")

    suites = list(benchmark_suites or DEFAULT_BENCHMARK_SUITES)
    thresholds = dict(DEFAULT_REGRESSION_THRESHOLDS)
    if regression_thresholds:
        thresholds.update({str(k): str(v) for k, v in regression_thresholds.items()})
    approval_complete = _approval_complete(
        benchmark_suites=suites,
        max_budget_usd=max_budget_usd,
        max_budget_krw=max_budget_krw,
        max_runtime_minutes=max_runtime_minutes,
        network_provider_api_spend_allowed=network_provider_api_spend_allowed,
        baseline_materialization_allowed=baseline_materialization_allowed,
        current_materialization_allowed=current_materialization_allowed,
        human_approval_source=human_approval_source,
        allowed_write_roots=allowed_write_roots,
        rollback_plan=rollback_plan,
    )
    status = APPROVAL_RECORDED_NOT_EXECUTED if approval_complete else AWAITING_EXPLICIT_BENCHMARK_APPROVAL

    packet = base_decision_payload(
        gate_id=RBA_GATE_ID,
        phase=RBA_PHASE,
        target=RBA_TARGET,
        generated_at=generated_at,
    )
    packet["schema_version"] = APPROVAL_PACKET_SCHEMA_VERSION
    packet.update(
        {
            "status": status,
            "summary": _summary(status),
            "approval_complete": approval_complete,
            "required_approval_fields": list(REQUIRED_APPROVAL_FIELDS),
            "approval_form": {
                "benchmark_suites": suites,
                "max_budget_usd": max_budget_usd if approval_complete else None,
                "max_budget_krw": max_budget_krw if approval_complete else None,
                "max_runtime_minutes": max_runtime_minutes if approval_complete else None,
                "network_provider_api_spend_allowed": bool(network_provider_api_spend_allowed and approval_complete),
                "baseline_materialization_allowed": bool(baseline_materialization_allowed and approval_complete),
                "current_materialization_allowed": bool(current_materialization_allowed and approval_complete),
                "regression_thresholds": thresholds,
                "allowed_write_roots": list(allowed_write_roots or []),
                "rollback_plan": dict(rollback_plan) if approval_complete and rollback_plan else None,
                "human_approval_source": human_approval_source if approval_complete else None,
                "approver_metadata": None,
            },
            "missing_approval_fields": _missing_approval_fields(
                benchmark_suites=suites,
                max_budget_usd=max_budget_usd,
                max_budget_krw=max_budget_krw,
                max_runtime_minutes=max_runtime_minutes,
                network_provider_api_spend_allowed=network_provider_api_spend_allowed,
                baseline_materialization_allowed=baseline_materialization_allowed,
                current_materialization_allowed=current_materialization_allowed,
                human_approval_source=human_approval_source,
                allowed_write_roots=allowed_write_roots,
                rollback_plan=rollback_plan,
            ),
            "source_backfill": {
                "path": str(source_path),
                "sha256": _sha256_path(source_path),
                "status": backfill.get("status"),
                "strict_plan_gate_closed": backfill.get("strict_plan_gate_closed"),
                "benchmark_gate_passed": backfill.get("benchmark_gate_passed"),
            },
            "execution_started": False,
            "real_benchmarks_executed": False,
            "real_benchmark_execution_approved": approval_complete,
            "current_authorized_budget_usd": max_budget_usd if approval_complete and max_budget_usd is not None else 0,
            "current_authorized_budget_krw": max_budget_krw if approval_complete and max_budget_krw is not None else 0,
            "approved_runtime_minutes": max_runtime_minutes if approval_complete else None,
            "network_provider_spend_allowed": bool(network_provider_api_spend_allowed and approval_complete),
            "baseline_materialization_allowed": bool(baseline_materialization_allowed and approval_complete),
            "current_materialization_allowed": bool(current_materialization_allowed and approval_complete),
            "human_approval_source": human_approval_source if approval_complete else None,
            "requested_benchmark_suites": _suite_records(suites),
            "regression_thresholds": thresholds,
            "allowed_write_roots": list(allowed_write_roots or []),
            "rollback_plan": dict(rollback_plan or _default_rollback_plan()),
            "required_next_action": _required_next_action(approval_complete),
            "execution_boundaries": {
                "benchmark_process_started": False,
                "provider_or_model_spend_performed": False,
                "network_calls_performed": False,
                "github_write_performed": False,
                "active_apply_performed": False,
                "gateway_restart_or_reload_performed": False,
                "cron_mutation_performed": False,
                "credential_or_secret_access_performed": False,
            },
            "artifacts": {
                "approval_packet": "real_benchmark_approval_packet.json",
                "approval_markdown": "real_benchmark_approval_packet.md",
                "safety_review": "real_benchmark_safety_review.json",
                "safety_review_markdown": "real_benchmark_safety_review.md",
                "backfill_snapshot": "inputs/benchmark_gate_backfill.json",
            },
        }
    )
    reject_github_or_active_apply_flags(packet)

    safety = _safety_review(packet)
    packet_path = out / "real_benchmark_approval_packet.json"
    markdown_path = out / "real_benchmark_approval_packet.md"
    safety_path = out / "real_benchmark_safety_review.json"
    safety_markdown_path = out / "real_benchmark_safety_review.md"
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_packet_markdown(packet))
    safety_path.write_text(json.dumps(safety, indent=2, sort_keys=True) + "\n")
    safety_markdown_path.write_text(_render_safety_markdown(safety))

    return {
        "approval_packet_path": str(packet_path),
        "approval_markdown_path": str(markdown_path),
        "safety_review_path": str(safety_path),
        "safety_review_markdown_path": str(safety_markdown_path),
        "backfill_snapshot_path": str(snapshot_path),
    }


def _load_backfill(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"backfill report not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("backfill report JSON root must be an object")
    return data


def _validate_backfill(backfill: Mapping[str, Any]) -> None:
    if backfill.get("status") != BLOCKED_BY_BENCHMARK_APPROVAL:
        raise ValueError("source backfill must be BLOCKED_BY_BENCHMARK_APPROVAL")
    if backfill.get("real_benchmarks_executed") is not False:
        raise ValueError("source backfill must not have executed real benchmarks")
    if backfill.get("real_benchmark_execution_approved") is not False:
        raise ValueError("source backfill must not already approve real benchmark execution")


def _validate_budget_limits(*, max_budget_usd: int | float | None, max_budget_krw: int | float | None) -> None:
    for label, value in (
        ("max_budget_usd", max_budget_usd),
        ("max_budget_krw", max_budget_krw),
    ):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{label} must be a finite numeric budget limit")
        if not isfinite(float(value)):
            raise ValueError("budget limits must be finite")



def _approval_complete(
    *,
    benchmark_suites: Sequence[str],
    max_budget_usd: int | float | None,
    max_budget_krw: int | float | None,
    max_runtime_minutes: int | None,
    network_provider_api_spend_allowed: bool | None,
    baseline_materialization_allowed: bool | None,
    current_materialization_allowed: bool | None,
    human_approval_source: str | None,
    allowed_write_roots: Sequence[str] | None,
    rollback_plan: Mapping[str, Any] | None,
) -> bool:
    return not _missing_approval_fields(
        benchmark_suites=benchmark_suites,
        max_budget_usd=max_budget_usd,
        max_budget_krw=max_budget_krw,
        max_runtime_minutes=max_runtime_minutes,
        network_provider_api_spend_allowed=network_provider_api_spend_allowed,
        baseline_materialization_allowed=baseline_materialization_allowed,
        current_materialization_allowed=current_materialization_allowed,
        human_approval_source=human_approval_source,
        allowed_write_roots=allowed_write_roots,
        rollback_plan=rollback_plan,
    )


def _missing_approval_fields(
    *,
    benchmark_suites: Sequence[str],
    max_budget_usd: int | float | None,
    max_budget_krw: int | float | None,
    max_runtime_minutes: int | None,
    network_provider_api_spend_allowed: bool | None,
    baseline_materialization_allowed: bool | None,
    current_materialization_allowed: bool | None,
    human_approval_source: str | None,
    allowed_write_roots: Sequence[str] | None,
    rollback_plan: Mapping[str, Any] | None,
) -> list[str]:
    missing: list[str] = []
    if not benchmark_suites:
        missing.append("benchmark_suites")
    if max_budget_usd is None and max_budget_krw is None:
        missing.append("max_budget_usd_or_krw")
    if max_runtime_minutes is None or max_runtime_minutes <= 0:
        missing.append("max_runtime_minutes")
    if network_provider_api_spend_allowed is None:
        missing.append("network_provider_api_spend_allowed")
    if baseline_materialization_allowed is not True:
        missing.append("baseline_materialization_allowed")
    if current_materialization_allowed is not True:
        missing.append("current_materialization_allowed")
    if not human_approval_source or not human_approval_source.strip():
        missing.append("human_approval_source")
    if not allowed_write_roots:
        missing.append("allowed_write_roots")
    if not rollback_plan:
        missing.append("rollback_plan")
    return missing


def _suite_records(suites: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(name),
            "execution_started": False,
            "real_result_required_for_strict_plan_gate": True,
        }
        for name in suites
    ]


def _default_rollback_plan() -> dict[str, Any]:
    return {
        "required_before_execution": True,
        "handles": [
            "git commit/hash for HSE packet state",
            "disposable baseline/current worktree cleanup plan",
            "benchmark output root manifest with SHA-256 hashes",
        ],
        "restore_mode": "no active runtime mutation expected; remove generated benchmark outputs if execution is cancelled before strict promotion",
    }


def _summary(status: str) -> str:
    if status == APPROVAL_RECORDED_NOT_EXECUTED:
        return "Explicit real benchmark approval is recorded; execution has not started."
    return "Real benchmark approval packet is fail-closed and awaiting explicit human approval."


def _required_next_action(approval_complete: bool) -> str:
    if approval_complete:
        return "run_real_benchmark_preflight_then_execute_under_packet"
    return "collect_missing_human_approval_fields_before_execution"


def _safety_review(packet: Mapping[str, Any]) -> dict[str, Any]:
    approval_complete = packet.get("approval_complete") is True
    blockers = [] if approval_complete else ["awaiting_explicit_human_benchmark_approval"]
    return {
        "schema_version": SAFETY_REVIEW_SCHEMA_VERSION,
        "status": SAFETY_REVIEW_PASS_APPROVAL_RECORDED if approval_complete else SAFETY_REVIEW_PASS_FAIL_CLOSED,
        "packet_status": packet.get("status"),
        "packet_approved_for_execution": approval_complete,
        "execution_started": False,
        "real_benchmarks_executed": False,
        "no_github_write_preserved": True,
        "blockers": blockers,
        "review_checks": {
            "fail_closed_booleans_present": True,
            "source_backfill_is_blocked_by_approval": packet.get("source_backfill", {}).get("status") == BLOCKED_BY_BENCHMARK_APPROVAL,
            "real_execution_not_started": packet.get("execution_started") is False,
            "provider_spend_not_performed": packet.get("execution_boundaries", {}).get("provider_or_model_spend_performed") is False,
            "github_write_not_performed": packet.get("execution_boundaries", {}).get("github_write_performed") is False,
            "active_apply_not_performed": packet.get("execution_boundaries", {}).get("active_apply_performed") is False,
        },
        "next_gate": _required_next_action(approval_complete),
    }


def _render_packet_markdown(packet: Mapping[str, Any]) -> str:
    suites = packet.get("requested_benchmark_suites", [])
    suite_lines = [f"- {suite.get('name')}" for suite in suites if isinstance(suite, Mapping)]
    missing = packet.get("missing_approval_fields", [])
    missing_lines = [f"- {field}" for field in missing] if isinstance(missing, list) else []
    return "\n".join(
        [
            "# HSE Real Benchmark Approval Packet",
            "",
            f"Status: `{packet.get('status')}`",
            "",
            "This packet is not approval to execute unless `approval_complete=true` and `execution_started=false` remains separately verified before running.",
            "",
            "## Execution State",
            "",
            f"- approval_complete={str(packet.get('approval_complete')).lower()}",
            f"- execution_started={str(packet.get('execution_started')).lower()}",
            f"- real_benchmarks_executed={str(packet.get('real_benchmarks_executed')).lower()}",
            f"- real_benchmark_execution_approved={str(packet.get('real_benchmark_execution_approved')).lower()}",
            f"- current_authorized_budget_usd={packet.get('current_authorized_budget_usd')}",
            f"- approved_runtime_minutes={packet.get('approved_runtime_minutes')}",
            "",
            "## Requested Suites",
            "",
            *suite_lines,
            "",
            "## Missing Approval Fields",
            "",
            *(missing_lines or ["- none"]),
            "",
            "## Boundaries",
            "",
            "- NO_GITHUB_WRITE",
            "- provider/model/API spend performed: false",
            "- benchmark process started: false",
            "- active apply performed: false",
            "- gateway restart/reload performed: false",
            "",
        ]
    )


def _render_safety_markdown(safety: Mapping[str, Any]) -> str:
    blockers = safety.get("blockers", [])
    blocker_lines = [f"- {blocker}" for blocker in blockers] if isinstance(blockers, list) else []
    checks = safety.get("review_checks", {})
    check_lines = [f"- {key}: `{value}`" for key, value in checks.items()] if isinstance(checks, Mapping) else []
    return "\n".join(
        [
            "# HSE Real Benchmark Safety Review",
            "",
            f"Status: `{safety.get('status')}`",
            "",
            f"- packet_approved_for_execution={str(safety.get('packet_approved_for_execution')).lower()}",
            f"- execution_started={str(safety.get('execution_started')).lower()}",
            f"- real_benchmarks_executed={str(safety.get('real_benchmarks_executed')).lower()}",
            "",
            "## Blockers",
            "",
            *(blocker_lines or ["- none"]),
            "",
            "## Checks",
            "",
            *check_lines,
            "",
        ]
    )


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_non_empty(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a non-executing HSE real benchmark approval packet.")
    parser.add_argument("--backfill-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args(argv)
    result = write_real_benchmark_approval_packet(
        backfill_report_path=args.backfill_report,
        output_dir=args.output_dir,
        generated_at=args.generated_at,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

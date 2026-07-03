"""Approval fields draft writer for HSE real benchmark approval.

This module prepares a conservative, non-executing draft of the human approval
fields that are still missing before any real benchmark execution can be
considered. It is intentionally only a draft: it does not approve execution,
start benchmark commands, materialize worktrees, spend provider/model/API
budget, query/write GitHub, mutate active runtime state, or close the strict
PLAN gate.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.local_completion.real_benchmark_approval_packet import (
    APPROVAL_PACKET_SCHEMA_VERSION,
    AWAITING_EXPLICIT_BENCHMARK_APPROVAL,
    REQUIRED_APPROVAL_FIELDS,
)
from evolution.local_completion.real_benchmark_preflight import (
    PREFLIGHT_RECORDED_NOT_EXECUTABLE,
    PREFLIGHT_SCHEMA_VERSION,
)
from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags

APPROVAL_FIELDS_DRAFT_SCHEMA_VERSION = "hse-real-benchmark-approval-fields-draft-v1"
APPROVAL_FIELDS_DRAFT_GATE_ID = "B0-AFD"
APPROVAL_FIELDS_DRAFT_PHASE = "Real Benchmark Approval Fields Draft"
APPROVAL_FIELDS_DRAFT_TARGET = "strict-plan-real-benchmark-approval-fields-draft"
APPROVAL_FIELDS_DRAFT_RECORDED_NOT_EXECUTABLE = "APPROVAL_FIELDS_DRAFT_RECORDED_NOT_EXECUTABLE"


def write_real_benchmark_approval_fields_draft(
    *,
    approval_packet_path: str | Path,
    preflight_report_path: str | Path,
    output_dir: str | Path,
    generated_at: str,
) -> dict[str, str]:
    """Write a conservative non-executing approval-fields draft."""

    _require_non_empty("generated_at", generated_at)
    approval_path = Path(approval_packet_path).expanduser()
    preflight_path = Path(preflight_report_path).expanduser()
    approval = _load_json_object(approval_path, "approval packet")
    preflight = _load_json_object(preflight_path, "preflight report")
    _validate_approval_packet(approval)
    _validate_preflight(preflight, approval_path=approval_path)

    out = Path(output_dir).expanduser()
    inputs_dir = out / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    approval_snapshot_path = inputs_dir / "real_benchmark_approval_packet.json"
    preflight_snapshot_path = inputs_dir / "real_benchmark_preflight.json"
    approval_snapshot_path.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n")
    preflight_snapshot_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")

    fields = _draft_fields(approval, preflight)
    blocked_by = _blocked_by(approval, preflight)
    report = base_decision_payload(
        gate_id=APPROVAL_FIELDS_DRAFT_GATE_ID,
        phase=APPROVAL_FIELDS_DRAFT_PHASE,
        target=APPROVAL_FIELDS_DRAFT_TARGET,
        generated_at=generated_at,
    )
    report["schema_version"] = APPROVAL_FIELDS_DRAFT_SCHEMA_VERSION
    report.update(
        {
            "status": APPROVAL_FIELDS_DRAFT_RECORDED_NOT_EXECUTABLE,
            "summary": "Conservative approval fields draft recorded; execution remains unapproved and blocked.",
            "draft_notice": "This is not approval to execute. It is a conservative human-review draft only.",
            "draft_only": True,
            "approval_draft_complete": True,
            "approval_complete": False,
            "real_benchmark_execution_approved": False,
            "execution_ready": False,
            "strict_plan_gate_closed": False,
            "execution_started": False,
            "real_benchmarks_executed": False,
            "current_authorized_budget_usd": 0,
            "current_authorized_budget_krw": 0,
            "approved_runtime_minutes": 0,
            "network_provider_spend_allowed": False,
            "baseline_materialization_allowed": False,
            "current_materialization_allowed": False,
            "github_policy": "NO_GITHUB_WRITE",
            "future_execution_requires_explicit_human_go": True,
            "required_approval_fields": list(REQUIRED_APPROVAL_FIELDS),
            "blocked_by": blocked_by,
            "draft_approval_fields": fields,
            "source_approval_packet": {
                "path": str(approval_path),
                "sha256": _sha256_path(approval_path),
                "status": approval.get("status"),
                "approval_complete": approval.get("approval_complete"),
                "missing_approval_fields": list(approval.get("missing_approval_fields", [])),
                "execution_started": approval.get("execution_started"),
                "real_benchmarks_executed": approval.get("real_benchmarks_executed"),
            },
            "source_preflight": {
                "path": str(preflight_path),
                "sha256": _sha256_path(preflight_path),
                "status": preflight.get("status"),
                "preflight_passed": preflight.get("preflight_passed"),
                "strict_plan_gate_closed": preflight.get("strict_plan_gate_closed"),
                "execution_ready": preflight.get("execution_ready"),
                "execution_started": preflight.get("execution_started"),
                "real_benchmarks_executed": preflight.get("real_benchmarks_executed"),
            },
            "human_review_decision_template": {
                "decision": "do_not_execute_yet",
                "if_approving_later_required_statement": "Sunwoo explicitly approves the named benchmark suites, budget, runtime, network/provider spend, baseline/current materialization, allowed write roots, rollback plan, and human approval source for a separate later execution gate.",
                "approval_source_placeholder": None,
            },
            "go_no_go_state": {
                "go_now": False,
                "no_go_reasons": blocked_by,
                "next_gate_required_before_execution": "explicit_benchmark_approval_then_execution_preflight_or_run",
            },
            "execution_boundaries": {
                "benchmark_process_started": False,
                "provider_or_model_spend_performed": False,
                "network_calls_performed": False,
                "github_write_performed": False,
                "active_apply_performed": False,
                "gateway_restart_or_reload_performed": False,
                "cron_mutation_performed": False,
                "credential_or_secret_access_performed": False,
                "worktree_materialization_performed": False,
                "benchmark_output_written": False,
            },
            "artifacts": {
                "approval_fields_draft": "real_benchmark_approval_fields_draft.json",
                "approval_fields_markdown": "real_benchmark_approval_fields_draft.md",
                "approval_packet_snapshot": "inputs/real_benchmark_approval_packet.json",
                "preflight_snapshot": "inputs/real_benchmark_preflight.json",
            },
        }
    )
    reject_github_or_active_apply_flags(report)

    draft_path = out / "real_benchmark_approval_fields_draft.json"
    markdown_path = out / "real_benchmark_approval_fields_draft.md"
    draft_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_markdown(report))
    return {
        "approval_fields_draft_path": str(draft_path),
        "approval_fields_markdown_path": str(markdown_path),
        "approval_packet_snapshot_path": str(approval_snapshot_path),
        "preflight_snapshot_path": str(preflight_snapshot_path),
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        data = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} JSON root must be an object: {path}")
    return data


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _validate_approval_packet(approval: Mapping[str, Any]) -> None:
    if approval.get("schema_version") != APPROVAL_PACKET_SCHEMA_VERSION:
        raise ValueError("approval packet schema_version mismatch")
    if approval.get("status") != AWAITING_EXPLICIT_BENCHMARK_APPROVAL or approval.get("approval_complete") is not False:
        raise ValueError("approval packet must still be awaiting explicit benchmark approval")
    if approval.get("execution_started") is not False or approval.get("real_benchmarks_executed") is not False:
        raise ValueError("approval packet must not have started execution")
    if approval.get("real_benchmark_execution_approved") is not False:
        raise ValueError("approval packet must not already approve real benchmark execution")
    if approval.get("current_authorized_budget_usd") != 0 or approval.get("current_authorized_budget_krw") != 0:
        raise ValueError("approval packet must have zero authorized budget for draft generation")
    if approval.get("network_provider_spend_allowed") is not False:
        raise ValueError("approval packet must not allow provider spend")
    github = approval.get("github")
    if isinstance(github, Mapping) and any(github.get(key) is True for key in ("queried", "pr_created", "push_performed", "merge_performed")):
        raise ValueError("approval packet must preserve NO_GITHUB_WRITE")


def _validate_preflight(preflight: Mapping[str, Any], *, approval_path: Path) -> None:
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("preflight report schema_version mismatch")
    if preflight.get("status") != PREFLIGHT_RECORDED_NOT_EXECUTABLE:
        raise ValueError("preflight report must still be recorded-not-executable")
    if preflight.get("preflight_passed") is not True:
        raise ValueError("preflight report must have passed before drafting approval fields")
    if preflight.get("strict_plan_gate_closed") is not False:
        raise ValueError("preflight report must keep strict PLAN gate open")
    if preflight.get("execution_ready") is not False:
        raise ValueError("preflight report must not be execution-ready for approval-fields draft")
    if preflight.get("execution_started") is not False or preflight.get("real_benchmarks_executed") is not False:
        raise ValueError("preflight report must not have started execution")
    if preflight.get("github_policy") != "NO_GITHUB_WRITE":
        raise ValueError("preflight report must preserve NO_GITHUB_WRITE")
    source = preflight.get("source_approval_packet")
    if isinstance(source, Mapping):
        source_path = source.get("path")
        if isinstance(source_path, str) and Path(source_path).expanduser().resolve(strict=False) != approval_path.resolve(strict=False):
            raise ValueError("preflight source approval packet path does not match approval_packet_path")


def _draft_fields(approval: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    suites = _benchmark_suites(approval, preflight)
    thresholds = dict(approval.get("regression_thresholds", preflight.get("regression_thresholds", {})))
    future_write_roots = _future_write_roots(preflight)
    rollback_candidate = preflight.get("rollback_cleanup_plan") if isinstance(preflight.get("rollback_cleanup_plan"), Mapping) else None
    return {
        "benchmark_suites": _field(
            conservative_default=suites,
            candidate_for_human_review=suites,
            source="approval_packet.requested_benchmark_suites",
            risk_notes=[
                "Changing suites changes benchmark scope, runtime, and comparability.",
                "Keeping suites from the packet does not authorize execution by itself.",
            ],
        ),
        "max_budget_usd_or_krw": _field(
            conservative_default={"max_budget_usd": 0, "max_budget_krw": 0},
            candidate_for_human_review=None,
            source="conservative_default_zero_spend",
            risk_notes=[
                "Any nonzero budget can permit provider/model/API spend.",
                "Sunwoo must supply an explicit capped amount before execution.",
            ],
        ),
        "max_runtime_minutes": _field(
            conservative_default=0,
            candidate_for_human_review=None,
            source="conservative_default_zero_runtime",
            risk_notes=[
                "A positive runtime limit bounds local resource use and hung benchmark risk.",
                "Zero minutes means execution remains impossible.",
            ],
        ),
        "network_provider_api_spend_allowed": _field(
            conservative_default=False,
            candidate_for_human_review=False,
            source="fail_closed_network_spend_boundary",
            risk_notes=[
                "True would allow network/provider/API spend surfaces.",
                "False preserves local-only no-spend behavior.",
            ],
        ),
        "baseline_materialization_allowed": _field(
            conservative_default=False,
            candidate_for_human_review=False,
            source="preflight.baseline_materialization_plan",
            risk_notes=[
                "True would allow disposable baseline worktree creation.",
                "False means the baseline worktree plan remains only a plan.",
            ],
        ),
        "current_materialization_allowed": _field(
            conservative_default=False,
            candidate_for_human_review=False,
            source="preflight.current_materialization_plan",
            risk_notes=[
                "True would allow disposable current worktree creation.",
                "False means the current worktree plan remains only a plan.",
            ],
        ),
        "regression_thresholds": _field(
            conservative_default=thresholds,
            candidate_for_human_review=thresholds,
            source="approval_packet.regression_thresholds",
            risk_notes=[
                "Thresholds define what counts as benchmark regression.",
                "They should not be loosened to manufacture a pass.",
            ],
        ),
        "allowed_write_roots": _field(
            conservative_default=[],
            candidate_for_human_review=future_write_roots,
            source="preflight.write_root_guard.allowed_write_roots",
            risk_notes=[
                "An approved write root permits benchmark outputs to be created there in a later run.",
                "The conservative default is an empty list, which blocks execution.",
            ],
        ),
        "rollback_plan": _field(
            conservative_default=None,
            candidate_for_human_review=dict(rollback_candidate) if rollback_candidate is not None else None,
            source="preflight.rollback_cleanup_plan",
            risk_notes=[
                "Rollback must remove disposable worktrees and any future benchmark output root if created.",
                "A rollback candidate is not accepted until Sunwoo approves it explicitly.",
            ],
        ),
        "human_approval_source": _field(
            conservative_default=None,
            candidate_for_human_review=None,
            source="future_explicit_sunwoo_message_required",
            risk_notes=[
                "A future approval source must cite an explicit Sunwoo message or approved artifact label.",
                "This draft and rec-action message are not execution approval.",
            ],
        ),
    }


def _field(
    *,
    conservative_default: Any,
    candidate_for_human_review: Any,
    source: str,
    risk_notes: list[str],
) -> dict[str, Any]:
    return {
        "conservative_default": conservative_default,
        "candidate_for_human_review": candidate_for_human_review,
        "approved_for_execution": False,
        "requires_explicit_human_approval": True,
        "source": source,
        "risk_notes": risk_notes,
    }


def _benchmark_suites(approval: Mapping[str, Any], preflight: Mapping[str, Any]) -> list[str]:
    suites: list[str] = []
    requested = approval.get("requested_benchmark_suites")
    if isinstance(requested, list):
        for item in requested:
            if isinstance(item, Mapping) and isinstance(item.get("name"), str) and item["name"]:
                suites.append(str(item["name"]))
            elif isinstance(item, str) and item:
                suites.append(item)
    if suites:
        return suites
    preflight_suites = preflight.get("benchmark_suites")
    if isinstance(preflight_suites, list):
        return [str(item) for item in preflight_suites if isinstance(item, str) and item]
    return []


def _future_write_roots(preflight: Mapping[str, Any]) -> list[str]:
    guard = preflight.get("write_root_guard")
    if not isinstance(guard, Mapping):
        return []
    roots = guard.get("allowed_write_roots")
    if not isinstance(roots, list):
        return []
    return [str(root) for root in roots if isinstance(root, str) and root]


def _blocked_by(approval: Mapping[str, Any], preflight: Mapping[str, Any]) -> list[str]:
    blocked = preflight.get("blocked_by")
    if isinstance(blocked, list) and blocked:
        return [str(item) for item in blocked if isinstance(item, str) and item]
    missing = approval.get("missing_approval_fields")
    result = ["awaiting_explicit_human_benchmark_approval"]
    if isinstance(missing, list):
        result.extend(str(item) for item in missing if isinstance(item, str) and item)
    return result


def _render_markdown(report: Mapping[str, Any]) -> str:
    fields = report.get("draft_approval_fields", {})
    field_lines: list[str] = []
    if isinstance(fields, Mapping):
        for name, payload in fields.items():
            default_value = payload.get("conservative_default") if isinstance(payload, Mapping) else None
            field_lines.append(f"- `{name}` conservative default: `{json.dumps(default_value, sort_keys=True)}`")
    blocked = report.get("blocked_by", [])
    blocker_lines = [f"- {item}" for item in blocked] if isinstance(blocked, list) else []
    return "\n".join(
        [
            "# HSE Real Benchmark Approval Fields Draft",
            "",
            f"Status: `{report.get('status')}`",
            "",
            "This is not approval to execute. It is a conservative human-review draft only.",
            "",
            "## Execution State",
            "",
            f"- approval_complete={str(report.get('approval_complete')).lower()}",
            f"- real_benchmark_execution_approved={str(report.get('real_benchmark_execution_approved')).lower()}",
            f"- execution_ready={str(report.get('execution_ready')).lower()}",
            f"- strict_plan_gate_closed={str(report.get('strict_plan_gate_closed')).lower()}",
            f"- execution_started={str(report.get('execution_started')).lower()}",
            f"- real_benchmarks_executed={str(report.get('real_benchmarks_executed')).lower()}",
            "- NO_GITHUB_WRITE",
            "",
            "## Conservative Draft Fields",
            "",
            *field_lines,
            "",
            "## Still Blocked By",
            "",
            *(blocker_lines or ["- none"]),
            "",
        ]
    )


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_non_empty(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a conservative HSE real benchmark approval-fields draft.")
    parser.add_argument("--approval-packet", required=True, type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args(argv)
    result = write_real_benchmark_approval_fields_draft(
        approval_packet_path=args.approval_packet,
        preflight_report_path=args.preflight_report,
        output_dir=args.output_dir,
        generated_at=args.generated_at,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""User-supplied approval fields update writer for HSE real benchmarks.

This module records a user-supplied update to the approval-fields draft while
preserving the non-executing, non-approved boundary. A user update may supply
candidate values for human review, but this writer never converts them into
execution approval, never closes the strict PLAN gate, and never starts or
prepares real benchmark execution.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.local_completion.real_benchmark_approval_fields_draft import (
    APPROVAL_FIELDS_DRAFT_RECORDED_NOT_EXECUTABLE,
    APPROVAL_FIELDS_DRAFT_SCHEMA_VERSION,
)
from evolution.local_completion.real_benchmark_approval_packet import REQUIRED_APPROVAL_FIELDS
from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags

APPROVAL_FIELDS_UPDATE_SCHEMA_VERSION = "hse-real-benchmark-approval-fields-update-v1"
APPROVAL_FIELDS_USER_UPDATE_SCHEMA_VERSION = "hse-real-benchmark-approval-fields-user-update-v1"
APPROVAL_FIELDS_UPDATE_GATE_ID = "B0-AFU"
APPROVAL_FIELDS_UPDATE_PHASE = "Real Benchmark Approval Fields User Update"
APPROVAL_FIELDS_UPDATE_TARGET = "strict-plan-real-benchmark-approval-fields-user-update"
APPROVAL_FIELDS_UPDATE_RECORDED_NOT_EXECUTABLE = "APPROVAL_FIELDS_UPDATE_RECORDED_NOT_EXECUTABLE"

AMBIGUOUS_MENTION_ALIASES = {
    "benchmark_suites": "benchmark_suites",
    "max_budget": "max_budget_usd_or_krw",
    "max_budget_usd_or_krw": "max_budget_usd_or_krw",
    "max_runtime": "max_runtime_minutes",
    "max_runtime_minutes": "max_runtime_minutes",
}


def write_real_benchmark_approval_fields_update(
    *,
    approval_fields_draft_path: str | Path,
    user_update_path: str | Path,
    output_dir: str | Path,
    generated_at: str,
) -> dict[str, str]:
    """Write a non-executing approval-fields update report."""

    _require_non_empty("generated_at", generated_at)
    draft_path = Path(approval_fields_draft_path).expanduser()
    update_path = Path(user_update_path).expanduser()
    draft = _load_json_object(draft_path, "approval fields draft")
    user_update = _load_json_object(update_path, "user approval-fields update")
    _validate_source_draft(draft)
    _validate_user_update(user_update)

    out = Path(output_dir).expanduser()
    inputs_dir = out / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    draft_snapshot_path = inputs_dir / "real_benchmark_approval_fields_draft.json"
    update_snapshot_path = inputs_dir / "user_approval_fields_update.json"
    draft_snapshot_path.write_text(json.dumps(draft, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    update_snapshot_path.write_text(json.dumps(user_update, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    interpreted_updates = _interpret_field_updates(draft, user_update)
    incomplete = _incomplete_or_ambiguous_mentions(user_update, interpreted_updates)

    report = base_decision_payload(
        gate_id=APPROVAL_FIELDS_UPDATE_GATE_ID,
        phase=APPROVAL_FIELDS_UPDATE_PHASE,
        target=APPROVAL_FIELDS_UPDATE_TARGET,
        generated_at=generated_at,
    )
    report["schema_version"] = APPROVAL_FIELDS_UPDATE_SCHEMA_VERSION
    report.update(
        {
            "status": APPROVAL_FIELDS_UPDATE_RECORDED_NOT_EXECUTABLE,
            "summary": "User-supplied approval field candidates recorded; execution remains unapproved and blocked.",
            "update_notice": "NOT EXECUTION APPROVAL. Candidate values are recorded for draft review only.",
            "update_only": True,
            "not_execution_approval": True,
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
            "interpreted_field_updates": interpreted_updates,
            "incomplete_or_ambiguous_mentions": incomplete,
            "still_required_for_execution_approval": list(REQUIRED_APPROVAL_FIELDS),
            "source_draft": {
                "path": str(draft_path),
                "sha256": _sha256_path(draft_path),
                "status": draft.get("status"),
                "draft_only": draft.get("draft_only"),
                "approval_complete": draft.get("approval_complete"),
                "execution_ready": draft.get("execution_ready"),
                "strict_plan_gate_closed": draft.get("strict_plan_gate_closed"),
                "execution_started": draft.get("execution_started"),
                "real_benchmarks_executed": draft.get("real_benchmarks_executed"),
            },
            "source_user_update": {
                "path": str(update_path),
                "sha256": _sha256_path(update_path),
                "schema_version": user_update.get("schema_version"),
                "not_execution_approval": user_update.get("not_execution_approval"),
                "raw_scope_label": user_update.get("raw_scope_label"),
            },
            "go_no_go_state": {
                "go_now": False,
                "no_go_reasons": [
                    "user_update_explicitly_marked_not_execution_approval",
                    "candidate_values_not_approved_for_execution",
                    "approval_complete_false",
                    "execution_ready_false",
                    "strict_plan_gate_closed_false",
                ],
                "next_gate_required_before_execution": "explicit_execution_approval_packet_with_all_required_fields",
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
                "approval_fields_update": "real_benchmark_approval_fields_update.json",
                "approval_fields_update_markdown": "real_benchmark_approval_fields_update.md",
                "source_draft_snapshot": "inputs/real_benchmark_approval_fields_draft.json",
                "user_update_snapshot": "inputs/user_approval_fields_update.json",
            },
        }
    )
    reject_github_or_active_apply_flags(report)

    update_report_path = out / "real_benchmark_approval_fields_update.json"
    markdown_path = out / "real_benchmark_approval_fields_update.md"
    update_report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    markdown_path.write_text(_render_markdown(report))
    return {
        "approval_fields_update_path": str(update_report_path),
        "approval_fields_update_markdown_path": str(markdown_path),
        "source_draft_snapshot_path": str(draft_snapshot_path),
        "user_update_snapshot_path": str(update_snapshot_path),
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


def _validate_source_draft(draft: Mapping[str, Any]) -> None:
    if draft.get("schema_version") != APPROVAL_FIELDS_DRAFT_SCHEMA_VERSION:
        raise ValueError("source draft schema_version mismatch")
    if draft.get("status") != APPROVAL_FIELDS_DRAFT_RECORDED_NOT_EXECUTABLE:
        raise ValueError("source draft status must remain recorded-not-executable")
    if draft.get("draft_only") is not True:
        raise ValueError("source draft must be draft_only")
    if draft.get("approval_complete") is not False or draft.get("real_benchmark_execution_approved") is not False:
        raise ValueError("source draft must not approve execution")
    if draft.get("execution_ready") is not False:
        raise ValueError("source draft must remain non-executable")
    if draft.get("strict_plan_gate_closed") is not False:
        raise ValueError("source draft must keep strict PLAN gate open")
    if draft.get("execution_started") is not False or draft.get("real_benchmarks_executed") is not False:
        raise ValueError("source draft must not have started execution")
    if draft.get("github_policy") != "NO_GITHUB_WRITE":
        raise ValueError("source draft must preserve NO_GITHUB_WRITE")
    fields = draft.get("draft_approval_fields")
    if not isinstance(fields, Mapping):
        raise ValueError("source draft must contain draft_approval_fields")
    for field in REQUIRED_APPROVAL_FIELDS:
        payload = fields.get(field)
        if not isinstance(payload, Mapping):
            raise ValueError(f"source draft missing approval field: {field}")
        if payload.get("approved_for_execution") is not False:
            raise ValueError(f"source draft field already approved for execution: {field}")


def _validate_user_update(user_update: Mapping[str, Any]) -> None:
    if user_update.get("schema_version") != APPROVAL_FIELDS_USER_UPDATE_SCHEMA_VERSION:
        raise ValueError("user update schema_version mismatch")
    if user_update.get("not_execution_approval") is not True:
        raise ValueError("user update must be explicitly marked not_execution_approval")
    label = user_update.get("raw_scope_label")
    if not isinstance(label, str) or "NOT EXECUTION APPROVAL" not in label:
        raise ValueError("user update scope label must include NOT EXECUTION APPROVAL")
    field_updates = user_update.get("field_updates")
    if not isinstance(field_updates, Mapping):
        raise ValueError("user update field_updates must be an object")
    for field, payload in field_updates.items():
        if field not in REQUIRED_APPROVAL_FIELDS:
            raise ValueError(f"unknown approval field update: {field}")
        if not isinstance(payload, Mapping):
            raise ValueError(f"approval field update must be an object: {field}")
        if payload.get("approved_for_execution") is True:
            raise ValueError("approval field updates must not set approved_for_execution")


def _interpret_field_updates(draft: Mapping[str, Any], user_update: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    draft_fields = draft.get("draft_approval_fields", {})
    field_updates = user_update.get("field_updates", {})
    interpreted: dict[str, dict[str, Any]] = {}
    for field, update_payload in field_updates.items():
        draft_payload = draft_fields.get(field, {}) if isinstance(draft_fields, Mapping) else {}
        if not isinstance(update_payload, Mapping):
            continue
        risk_notes = []
        if isinstance(draft_payload, Mapping) and isinstance(draft_payload.get("risk_notes"), list):
            risk_notes.extend(str(note) for note in draft_payload["risk_notes"])
        risk_notes.append("Recorded from Sunwoo's NOT EXECUTION APPROVAL draft update; still not executable.")
        interpreted[field] = {
            "conservative_default": deepcopy(draft_payload.get("conservative_default")) if isinstance(draft_payload, Mapping) else None,
            "previous_candidate_for_human_review": deepcopy(draft_payload.get("candidate_for_human_review")) if isinstance(draft_payload, Mapping) else None,
            "candidate_for_human_review": deepcopy(update_payload.get("candidate_for_human_review")),
            "approved_for_execution": False,
            "requires_explicit_human_approval": True,
            "requires_separate_execution_approval": True,
            "source": "sunwoo_not_execution_approval_update",
            "risk_notes": risk_notes,
        }
    return interpreted


def _incomplete_or_ambiguous_mentions(user_update: Mapping[str, Any], interpreted_updates: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for mention in user_update.get("unstructured_mentions", []):
        if not isinstance(mention, str):
            continue
        alias = AMBIGUOUS_MENTION_ALIASES.get(mention.strip())
        if alias and alias not in interpreted_updates and alias not in result:
            result.append(alias)
    return result


def _render_markdown(report: Mapping[str, Any]) -> str:
    updates = report.get("interpreted_field_updates", {})
    update_lines: list[str] = []
    if isinstance(updates, Mapping):
        for field, payload in updates.items():
            candidate = payload.get("candidate_for_human_review") if isinstance(payload, Mapping) else None
            update_lines.append(f"- `{field}` candidate_for_human_review: `{json.dumps(candidate, sort_keys=True, ensure_ascii=False)}`; approved_for_execution=false")
    incomplete = report.get("incomplete_or_ambiguous_mentions", [])
    incomplete_lines = [f"- `{item}`" for item in incomplete] if isinstance(incomplete, list) else []
    return "\n".join(
        [
            "# HSE Real Benchmark Approval Fields User Update",
            "",
            f"Status: `{report.get('status')}`",
            "",
            "NOT EXECUTION APPROVAL. Candidate values are recorded for draft review only.",
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
            "## Interpreted Candidate Updates",
            "",
            *update_lines,
            "",
            "## Incomplete or Ambiguous Mentions",
            "",
            *(incomplete_lines or ["- none"]),
            "",
            "## Boundary",
            "",
            "All interpreted updates remain `approved_for_execution=false` and require separate explicit execution approval.",
            "",
        ]
    )


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_non_empty(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a non-executing HSE approval-fields user update report.")
    parser.add_argument("--approval-fields-draft", required=True, type=Path)
    parser.add_argument("--user-update", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args(argv)
    result = write_real_benchmark_approval_fields_update(
        approval_fields_draft_path=args.approval_fields_draft,
        user_update_path=args.user_update,
        output_dir=args.output_dir,
        generated_at=args.generated_at,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

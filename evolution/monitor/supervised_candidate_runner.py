"""Phase 5 supervised local candidate runner handoff.

This module is intentionally not an unattended scheduler. It records a bounded,
explicitly approved evidence handoff for one scheduler dry-run queue target and
re-consumes an existing local candidate bundle ``decision.json``. The output
remains candidate-only: no inline runner execution, active apply, cron
installation, deployment, GitHub publication, or external notification is
performed here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evolution.core.candidate_bundle import ALLOWED_DECISION_STATUSES, SCHEMA_VERSION as CANDIDATE_BUNDLE_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE5_OUTPUT_ROOT = REPO_ROOT / "output" / "phase5-continuous-loop"
REPORT_JSON_NAME = "supervised_candidate_runner_report.json"
REPORT_MARKDOWN_NAME = "supervised_candidate_runner_report.md"

PUBLIC_APPROVAL_SENTINEL = "APPROVE_LOCAL_CANDIDATE_RUNNER"
_SCHEMA_VERSION = "phase5-supervised-candidate-runner-v1"
_MODE = "phase5-supervised-local-candidate-runner"
_SCHEDULER_SCHEMA_VERSION = "phase5-scheduler-dry-run-v1"
_SCHEDULER_MODE = "phase5-readonly-scheduler-dry-run"
_STATUS = "SUPERVISED_DECISION_RECONSUMED"


def build_supervised_runner_report(
    scheduler_report: Mapping[str, Any],
    *,
    approved_queue_id: str,
    approval_token: str,
    candidate_bundle_decision: Mapping[str, Any],
    runner_execution_started: bool = False,
    runner_returncode: int | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a supervised runner handoff report for exactly one queue target."""

    if runner_execution_started or runner_returncode is not None:
        raise ValueError(
            "inline runner execution is disabled in this safety slice; "
            "run the approved local runner separately and pass its decision.json"
        )
    _validate_approval(approval_token)
    _reject_private_or_raw_identifiers(scheduler_report)
    _validate_scheduler_report(scheduler_report)
    selected = _select_queue_target(scheduler_report, approved_queue_id)
    decision = _validate_candidate_bundle_decision(candidate_bundle_decision)
    _validate_decision_matches_queue(decision, selected)

    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    runner_hint = selected["runner_hint"].strip()
    execution_mode = "manual_decision_reconsume_only"

    report: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "phase": "5",
        "mode": _MODE,
        "status": _STATUS,
        "generated_at": generated_at,
        "source": {
            "scheduler_schema_version": scheduler_report["schema_version"],
            "scheduler_mode": scheduler_report["mode"],
            "scheduler_status": scheduler_report["status"],
            "scheduler_queue_count": len(scheduler_report["candidate_bundle_queue"]),
            "approved_queue_id": approved_queue_id,
        },
        "input_contract": {
            "scheduler_dry_run_report_required": True,
            "candidate_bundle_decision_json_required": True,
            "explicit_approval_required": True,
            "sanitized_input_required": True,
            "raw_command_recorded": False,
            "private_paths_allowed": False,
            "credentials_allowed": False,
        },
        "approval": {
            "approved_queue_id": approved_queue_id,
            "explicit_approval_recorded": True,
            "approval_scope": "single_local_candidate_queue_target_only",
        },
        "selected_queue_target": {
            "queue_id": selected["queue_id"],
            "target_metric_id": selected["target_metric_id"],
            "component": selected["component"],
            "candidate_bundle_phase": selected["candidate_bundle_phase"],
            "candidate_bundle_target": selected["candidate_bundle_target"],
            "runner_hint": runner_hint,
        },
        "runner_invocation": {
            "execution_mode": execution_mode,
            "runner_execution_started": runner_execution_started,
            "runner_returncode": runner_returncode,
            "runner_hint": runner_hint,
            "raw_command_recorded": False,
        },
        "decision_reconsume": {
            "decision_state": "DECISION_AVAILABLE",
            "decision_status": decision["status"],
            "decision_run_id": decision["run_id"],
            "decision_candidate_only": True,
            "decision_apply_ready": False,
            "decision_github_pr_created": False,
        },
        "safety_invariants": {
            "active_runtime_mutation": False,
            "active_apply_ready": False,
            "credentials_accessed": False,
            "cron_jobs_created": False,
            "scheduler_or_cron_side_effects_performed": False,
            "external_calls_performed": False,
            "network_calls_performed": False,
            "github_publication_performed": False,
            "automated_pr_created_or_updated": False,
            "deployment_performed": False,
        },
        "recommended_next_step": "human_review_candidate_bundle_decision_before_any_apply_or_publication",
    }
    _reject_private_or_raw_identifiers(report)
    return report


def write_supervised_runner_report(
    scheduler_report: Mapping[str, Any],
    *,
    approved_queue_id: str,
    approval_token: str,
    candidate_bundle_decision: Mapping[str, Any],
    output_dir: Path,
    runner_execution_started: bool = False,
    runner_returncode: int | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Write JSON and Markdown supervised runner handoff artifacts."""

    output_dir = _validate_output_dir(output_dir)
    report = build_supervised_runner_report(
        scheduler_report,
        approved_queue_id=approved_queue_id,
        approval_token=approval_token,
        candidate_bundle_decision=candidate_bundle_decision,
        runner_execution_started=runner_execution_started,
        runner_returncode=runner_returncode,
        generated_at=generated_at,
    )
    report["artifacts"] = {
        "report_json": REPORT_JSON_NAME,
        "report_markdown": REPORT_MARKDOWN_NAME,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_JSON_NAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / REPORT_MARKDOWN_NAME).write_text(_render_markdown(report))
    return report


def _validate_approval(approval_token: str) -> None:
    if approval_token != PUBLIC_APPROVAL_SENTINEL:
        raise ValueError("explicit approval token is required before supervised candidate runner handoff")


def _validate_scheduler_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != _SCHEDULER_SCHEMA_VERSION:
        raise ValueError("scheduler report schema_version must be phase5-scheduler-dry-run-v1")
    if report.get("phase") != "5" or report.get("mode") != _SCHEDULER_MODE:
        raise ValueError("scheduler report must be a Phase 5 scheduler dry-run report")
    if report.get("status") not in {"DRY_RUN_REVIEW_REQUIRED", "DRY_RUN_NOOP"}:
        raise ValueError("scheduler report status is not recognized")
    _validate_scheduler_safety(report.get("safety_invariants"))
    contract = report.get("local_candidate_bundle_contract")
    if not isinstance(contract, Mapping) or contract.get("schema_version") != CANDIDATE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("scheduler report must contain the local candidate bundle contract")
    if contract.get("runner_execution_started") is not False:
        raise ValueError("scheduler dry-run must not have started a runner")
    if contract.get("active_apply_ready") is not False or contract.get("github_publication_performed") is not False:
        raise ValueError("scheduler dry-run must not apply or publish candidates")
    queue = report.get("candidate_bundle_queue")
    if not isinstance(queue, list):
        raise ValueError("scheduler report must contain candidate_bundle_queue")
    summary = report.get("candidate_bundle_queue_summary")
    if not isinstance(summary, Mapping) or summary.get("queue_count") != len(queue):
        raise ValueError("scheduler queue summary must match candidate_bundle_queue length")
    for item in queue:
        _validate_queue_item(item)


def _validate_scheduler_safety(safety: object) -> None:
    if not isinstance(safety, Mapping):
        raise ValueError("scheduler report must contain safety_invariants")
    required_false = [
        "raw_private_session_data_committed",
        "raw_credentials_recorded",
        "active_runtime_mutation",
        "external_calls_performed",
        "network_calls_performed",
        "cron_jobs_created",
        "benchmark_cron_enabled",
        "scheduler_or_cron_side_effects_performed",
        "notifications_sent",
        "auto_optimizer_triggered",
        "optimizer_execution_started",
        "automated_pr_created_or_updated",
        "automated_apply_ready",
    ]
    if safety.get("read_only") is not True:
        raise ValueError("scheduler report must be read-only before supervised runner handoff")
    if any(safety.get(key) is not False for key in required_false):
        raise ValueError("scheduler report must be read-only before supervised runner handoff")


def _validate_queue_item(item: object) -> None:
    if not isinstance(item, Mapping):
        raise ValueError("candidate bundle queue entries must be objects")
    required_strings = [
        "queue_id",
        "target_metric_id",
        "component",
        "candidate_bundle_phase",
        "candidate_bundle_target",
        "runner_hint",
    ]
    for key in required_strings:
        if not isinstance(item.get(key), str) or not item[key].strip():
            raise ValueError(f"candidate bundle queue entry {key} must be a non-empty string")
    if item.get("requires_human_review_before_apply") is not True:
        raise ValueError("candidate bundle queue entry must require human review before apply")
    if item.get("would_start_runner") is not False:
        raise ValueError("scheduler dry-run queue must not start runners")


def _select_queue_target(report: Mapping[str, Any], approved_queue_id: str) -> Mapping[str, Any]:
    if not approved_queue_id or not approved_queue_id.strip():
        raise ValueError("approved queue id must be non-empty")
    for item in report["candidate_bundle_queue"]:
        if item["queue_id"] == approved_queue_id:
            return item
    raise ValueError("approved queue id was not found in scheduler candidate bundle queue")


def _validate_candidate_bundle_decision(decision: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(decision, Mapping):
        raise ValueError("candidate bundle decision must be an object")
    _reject_private_or_raw_identifiers(decision)
    if decision.get("schema_version") != CANDIDATE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("candidate bundle decision schema_version must be hse-local-candidate-bundle-v1")
    if decision.get("status") not in ALLOWED_DECISION_STATUSES:
        raise ValueError("candidate bundle decision status is not recognized")
    for key in ("phase", "target", "run_id"):
        if not isinstance(decision.get(key), str) or not decision[key].strip():
            raise ValueError(f"candidate bundle decision {key} must be a non-empty string")
    if decision.get("candidate_only") is not True or decision.get("apply_ready") is not False:
        raise ValueError("candidate bundle decision must be candidate-only")
    github = decision.get("github")
    if not isinstance(github, Mapping):
        raise ValueError("candidate bundle decision must contain GitHub side-effect fields")
    if any(github.get(key) is not False for key in ("pr_created", "push_performed", "merge_performed")):
        raise ValueError("candidate bundle decision must not contain GitHub side effects")
    safety = decision.get("safety_invariants")
    if not isinstance(safety, Mapping):
        raise ValueError("candidate bundle decision must contain safety_invariants")
    required_false = [
        "active_runtime_mutation",
        "active_skill_modified",
        "active_tool_schema_modified",
        "active_prompt_modified",
        "credentials_accessed",
        "external_publication_performed",
        "deployment_performed",
    ]
    if any(safety.get(key) is not False for key in required_false):
        raise ValueError("candidate bundle decision must be candidate-only")
    return dict(decision)


def _validate_decision_matches_queue(decision: Mapping[str, Any], queue_item: Mapping[str, Any]) -> None:
    decision_target = _strict_match_text(str(decision["target"]))
    decision_phase = _strict_match_text(str(decision["phase"]))
    accepted_target = _strict_match_text(queue_item["candidate_bundle_target"])
    accepted_phase = _strict_match_text(queue_item["candidate_bundle_phase"])
    if decision_target != accepted_target or decision_phase != accepted_phase:
        raise ValueError("candidate bundle decision does not match approved queue target")


def _validate_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve(strict=False)
    root = PHASE5_OUTPUT_ROOT.resolve(strict=False)
    if output_dir == root or root not in output_dir.parents:
        raise ValueError("output-dir must be under output/phase5-continuous-loop")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output-dir must be a directory before writing supervised runner artifacts")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output-dir must be empty before writing supervised runner artifacts")
    return output_dir


def _reject_inline_runner_execution_requested(*, execute: bool, runner_command_json: str | None) -> None:
    if execute or runner_command_json:
        raise ValueError(
            "inline runner execution is disabled in this safety slice; "
            "run the approved local runner separately and pass its decision.json"
        )


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 5 Supervised Candidate Runner",
        "",
        f"Status: {report['status']}",
        "",
        f"Mode: `{report['mode']}`",
        "",
        "## Approval",
        "",
        f"- approved_queue_id={report['approval']['approved_queue_id']}",
        "- approval_scope=single_local_candidate_queue_target_only",
        "",
        "## Runner Invocation",
        "",
        f"- execution_mode={report['runner_invocation']['execution_mode']}",
        f"- runner_execution_started={str(report['runner_invocation']['runner_execution_started']).lower()}",
        f"- runner_returncode={report['runner_invocation']['runner_returncode']}",
        "- raw_command_recorded=false",
        "",
        "## Decision Reconsume",
        "",
        f"- decision_state={report['decision_reconsume']['decision_state']}",
        f"- decision_status={report['decision_reconsume']['decision_status']}",
        f"- decision_run_id={report['decision_reconsume']['decision_run_id']}",
        "- decision_apply_ready=false",
        "",
        "## Safety",
        "",
        "- active_apply_ready=false",
        "- github_publication_performed=false",
        "- automated_pr_created_or_updated=false",
        "- cron_jobs_created=false",
        "- deployment_performed=false",
        "",
        "This supervised handoff is not approval to apply candidates to active Hermes runtime, publish to GitHub, create cron jobs, deploy, or send external notifications.",
        "",
    ]
    markdown = "\n".join(lines)
    _reject_private_or_raw_identifiers(markdown)
    return markdown


def _safe_target_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value.strip()]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "phase5-target"


def _strict_match_text(value: str) -> str:
    return value.strip()


def _reject_private_or_raw_identifiers(value: object) -> None:
    forbidden = [
        "/" + "Users" + "/",
        "/" + "home" + "/",
        "session" + "_id",
        "OPENAI" + "_API_KEY",
        "ANTHROPIC" + "_API_KEY",
        "OPENROUTER" + "_API_KEY",
    ]
    for text in _all_strings(value):
        for fragment in forbidden:
            if fragment in text:
                raise ValueError("supervised candidate runner input contains private/raw identifier")


def _all_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _all_strings(key)
            yield from _all_strings(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _all_strings(child)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a supervised Phase 5 local candidate runner handoff.")
    parser.add_argument("--scheduler-report-json", required=True, type=Path)
    parser.add_argument("--approved-queue-id", required=True)
    parser.add_argument("--approval-token", required=True)
    parser.add_argument("--candidate-bundle-decision-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--execute-approved-runner", action="store_true")
    parser.add_argument("--runner-command-json", default=None)
    args = parser.parse_args(argv)

    try:
        _reject_inline_runner_execution_requested(
            execute=args.execute_approved_runner,
            runner_command_json=args.runner_command_json,
        )
        scheduler_report = json.loads(args.scheduler_report_json.read_text())
        candidate_bundle_decision = json.loads(args.candidate_bundle_decision_json.read_text())
        write_supervised_runner_report(
            scheduler_report,
            approved_queue_id=args.approved_queue_id,
            approval_token=args.approval_token,
            candidate_bundle_decision=candidate_bundle_decision,
            output_dir=args.output_dir,
            runner_execution_started=False,
            runner_returncode=None,
            generated_at=args.generated_at,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("wrote Phase 5 supervised candidate runner report")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

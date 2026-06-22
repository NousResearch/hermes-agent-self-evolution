"""Read-only Phase 5 scheduler dry-run reports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evolution.core.candidate_bundle import ALLOWED_DECISION_STATUSES, SCHEMA_VERSION as CANDIDATE_BUNDLE_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE5_OUTPUT_ROOT = REPO_ROOT / "output" / "phase5-continuous-loop"
REPORT_JSON_NAME = "scheduler_dry_run_report.json"
REPORT_MARKDOWN_NAME = "scheduler_dry_run_report.md"

_SCHEMA_VERSION = "phase5-scheduler-dry-run-v1"
_AUTO_TRIAGE_SCHEMA_VERSION = "phase5-auto-triage-ranking-v1"
_AUTO_TRIAGE_MODE = "phase5-readonly-auto-triage-ranking"
_MODE = "phase5-readonly-scheduler-dry-run"
_REVIEW_RECOMMENDATION = "human_review_required_before_scheduler_enablement"
_NOOP_RECOMMENDATION = "monitor_only_no_scheduler_action"
_ACTION_RECOMMENDATION = "review_target_no_scheduler_side_effects"

_REQUIRED_BEFORE_REAL_SCHEDULER = [
    "explicit human approval for scheduler enablement",
    "Phase 4 formal handoff reviewed or waived",
    "benchmark/API budget approval",
    "cron target and delivery channel review",
]


def build_scheduler_dry_run_report(
    auto_triage_report: Mapping[str, Any],
    *,
    candidate_bundle_decisions: Iterable[Mapping[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic no-side-effect scheduler dry-run report.

    The function translates read-only auto-triage targets into hypothetical
    manual-review scheduling actions. It never creates cron jobs, sends
    notifications, starts optimizers, mutates runtime state, or updates pull
    requests.
    """

    _reject_private_or_raw_identifiers(auto_triage_report)
    _validate_auto_triage_report(auto_triage_report)
    bundle_decisions = [_validate_candidate_bundle_decision(decision) for decision in (candidate_bundle_decisions or [])]

    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    ranked_targets = list(auto_triage_report["ranked_targets"])
    dry_run_actions = [_dry_run_action(target, index) for index, target in enumerate(ranked_targets, start=1)]
    candidate_bundle_queue = [
        _candidate_bundle_queue_item(target, index, bundle_decisions)
        for index, target in enumerate(ranked_targets, start=1)
    ]
    candidate_bundle_queue_summary = _candidate_bundle_queue_summary(candidate_bundle_queue, bundle_decisions)
    status = "DRY_RUN_REVIEW_REQUIRED" if dry_run_actions else "DRY_RUN_NOOP"
    recommended_next_step = _REVIEW_RECOMMENDATION if dry_run_actions else _NOOP_RECOMMENDATION
    top_metric_id = ranked_targets[0]["metric_id"] if ranked_targets else None

    report: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "phase": "5",
        "mode": _MODE,
        "status": status,
        "generated_at": generated_at,
        "source": {
            "auto_triage_schema_version": auto_triage_report["schema_version"],
            "auto_triage_mode": auto_triage_report["mode"],
            "auto_triage_status": auto_triage_report["status"],
            "ranked_target_count": len(ranked_targets),
            "top_metric_id": top_metric_id,
        },
        "input_contract": {
            "auto_triage_report_required": True,
            "sanitized_input_required": True,
            "raw_session_data_allowed": False,
            "private_paths_allowed": False,
            "network_sources_allowed": False,
            "credentials_allowed": False,
        },
        "safety_invariants": {
            "read_only": True,
            "raw_private_session_data_committed": False,
            "raw_credentials_recorded": False,
            "active_runtime_mutation": False,
            "external_calls_performed": False,
            "network_calls_performed": False,
            "cron_jobs_created": False,
            "benchmark_cron_enabled": False,
            "scheduler_or_cron_side_effects_performed": False,
            "notifications_sent": False,
            "auto_optimizer_triggered": False,
            "optimizer_execution_started": False,
            "automated_pr_created_or_updated": False,
            "automated_apply_ready": False,
        },
        "dry_run_policy": {
            "scheduler_enablement_policy": "never_enable_in_this_slice",
            "max_actions": len(dry_run_actions),
            "action_source": "auto_triage_ranked_targets",
            "required_before_real_scheduler": list(_REQUIRED_BEFORE_REAL_SCHEDULER),
        },
        "summary": {
            "ranked_target_count": len(ranked_targets),
            "dry_run_action_count": len(dry_run_actions),
            "side_effect_count": 0,
            "top_metric_id": top_metric_id,
            "scheduler_enablement_ready": False,
            "review_required": bool(dry_run_actions),
        },
        "dry_run_actions": dry_run_actions,
        "local_candidate_bundle_contract": {
            "schema_version": CANDIDATE_BUNDLE_SCHEMA_VERSION,
            "decision_json_consumed": bool(bundle_decisions),
            "decision_json_required_before_apply": True,
            "runner_execution_started": False,
            "active_apply_ready": False,
            "github_publication_performed": False,
        },
        "candidate_bundle_queue_summary": candidate_bundle_queue_summary,
        "candidate_bundle_queue": candidate_bundle_queue,
        "recommended_next_step": recommended_next_step,
    }
    _reject_private_or_raw_identifiers(report)
    return report


def write_scheduler_dry_run_report(
    auto_triage_report: Mapping[str, Any],
    *,
    output_dir: Path,
    candidate_bundle_decisions: Iterable[Mapping[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Write JSON and Markdown scheduler dry-run artifacts under Phase 5 output root."""

    output_dir = _validate_output_dir(output_dir)
    report = build_scheduler_dry_run_report(
        auto_triage_report,
        candidate_bundle_decisions=candidate_bundle_decisions,
        generated_at=generated_at,
    )
    report["artifacts"] = {
        "report_json": REPORT_JSON_NAME,
        "report_markdown": REPORT_MARKDOWN_NAME,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_NAME
    markdown_path = output_dir / REPORT_MARKDOWN_NAME
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_markdown(report))
    return report


def _validate_auto_triage_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != _AUTO_TRIAGE_SCHEMA_VERSION:
        raise ValueError("auto-triage report schema_version must be phase5-auto-triage-ranking-v1")
    if report.get("phase") != "5" or report.get("mode") != _AUTO_TRIAGE_MODE:
        raise ValueError("auto-triage report must be a Phase 5 auto-triage ranking")
    if report.get("status") not in {"REVIEW_REQUIRED", "NO_ACTION"}:
        raise ValueError("auto-triage report status must be REVIEW_REQUIRED or NO_ACTION")
    _validate_auto_triage_input_contract(report.get("input_contract"))
    _validate_auto_triage_safety(report.get("safety_invariants"))
    ranked_targets = report.get("ranked_targets")
    if not isinstance(ranked_targets, list):
        raise ValueError("auto-triage report must contain ranked_targets list")
    for target in ranked_targets:
        _validate_ranked_target(target)
    summary = report.get("summary")
    if not isinstance(summary, Mapping) or not isinstance(summary.get("ranked_target_count"), int):
        raise ValueError("auto-triage report summary must contain ranked_target_count")
    if summary["ranked_target_count"] != len(ranked_targets):
        raise ValueError("auto-triage ranked_target_count must match ranked_targets length")
    expected_ranks = list(range(1, len(ranked_targets) + 1))
    actual_ranks = [target["rank"] for target in ranked_targets]
    if actual_ranks != expected_ranks:
        raise ValueError("ranked target ranks must be unique and sequential")
    if report["status"] == "REVIEW_REQUIRED" and not ranked_targets:
        raise ValueError("REVIEW_REQUIRED report must contain ranked targets")
    if report["status"] == "NO_ACTION" and ranked_targets:
        raise ValueError("auto-triage NO_ACTION report must not contain ranked targets")


def _validate_auto_triage_input_contract(input_contract: object) -> None:
    if not isinstance(input_contract, Mapping):
        raise ValueError("auto-triage report input_contract must be sanitized")
    required_false = [
        "raw_session_data_allowed",
        "private_paths_allowed",
        "network_sources_allowed",
        "credentials_allowed",
    ]
    if input_contract.get("sanitized_input_required") is not True:
        raise ValueError("auto-triage report input_contract must be sanitized")
    if any(input_contract.get(key) is not False for key in required_false):
        raise ValueError("auto-triage report input_contract must be sanitized")


def _validate_auto_triage_safety(safety: object) -> None:
    if not isinstance(safety, Mapping):
        raise ValueError("auto-triage report must contain safety_invariants")
    required_false = [
        "raw_private_session_data_committed",
        "raw_credentials_recorded",
        "active_runtime_mutation",
        "external_calls_performed",
        "network_calls_performed",
        "cron_jobs_created",
        "scheduler_or_cron_side_effects_performed",
        "auto_optimizer_triggered",
        "optimizer_execution_started",
        "automated_pr_created_or_updated",
        "automated_apply_ready",
    ]
    if safety.get("read_only") is not True:
        raise ValueError("auto-triage report must be read-only before scheduler dry-run")
    optional_scheduler_false = ["benchmark_cron_enabled", "notifications_sent"]
    if any(safety.get(key) is not False for key in required_false):
        raise ValueError("auto-triage report must be read-only before scheduler dry-run")
    if any(key in safety and safety.get(key) is not False for key in optional_scheduler_false):
        raise ValueError("auto-triage report must be read-only before scheduler dry-run")


def _validate_ranked_target(target: object) -> None:
    if not isinstance(target, Mapping):
        raise ValueError("each ranked target must be an object")
    for key in ("metric_id", "component", "recommendation"):
        if not isinstance(target.get(key), str) or not target[key].strip():
            raise ValueError(f"ranked_target.{key} must be a non-empty string")
    rank = target.get("rank")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("ranked_target.rank must be a positive integer")
    for key in ("priority_score", "severity"):
        value = target.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"ranked_target.{key} must be a finite number")
    sample_count = target.get("sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
        raise ValueError("ranked_target.sample_count must be a positive integer")


def _dry_run_action(target: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "action_id": f"dry-run-action-{index:03d}",
        "action_type": "manual_triage_review",
        "target_rank": target["rank"],
        "target_metric_id": target["metric_id"].strip(),
        "component": target["component"].strip(),
        "priority_score": float(target["priority_score"]),
        "dry_run_only": True,
        "would_create_cron_job": False,
        "would_enable_benchmark_cron": False,
        "would_start_optimizer": False,
        "would_send_external_notification": False,
        "would_update_external_pr": False,
        "proposed_cadence": "manual_review_only",
        "required_approval": _REVIEW_RECOMMENDATION,
        "recommendation": _ACTION_RECOMMENDATION,
    }


def _candidate_bundle_queue_item(
    target: Mapping[str, Any],
    index: int,
    decisions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    profile = _candidate_bundle_profile(target)
    decision = _matching_candidate_bundle_decision(target, profile, decisions)
    if decision is None:
        decision_fields = {
            "decision_state": "MISSING_DECISION",
            "decision_status": None,
            "decision_run_id": None,
            "decision_apply_ready": False,
            "decision_github_pr_created": False,
            "would_create_local_bundle": False,
        }
    else:
        decision_fields = {
            "decision_state": "DECISION_AVAILABLE",
            "decision_status": decision["status"],
            "decision_run_id": decision["run_id"],
            "decision_apply_ready": False,
            "decision_github_pr_created": False,
            "would_create_local_bundle": False,
        }
    return {
        "queue_id": f"candidate-bundle-target-{index:03d}",
        "target_rank": target["rank"],
        "target_metric_id": target["metric_id"].strip(),
        "component": target["component"].strip(),
        **profile,
        **decision_fields,
        "would_start_runner": False,
        "requires_human_review_before_apply": True,
    }


def _candidate_bundle_queue_summary(
    queue: list[Mapping[str, Any]],
    decisions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    matched = sum(1 for item in queue if item["decision_state"] == "DECISION_AVAILABLE")
    return {
        "queue_count": len(queue),
        "decision_count": len(decisions),
        "matched_decision_count": matched,
        "missing_decision_count": len(queue) - matched,
        "runner_execution_started": False,
        "active_apply_ready": False,
        "github_publication_performed": False,
    }


def _candidate_bundle_profile(target: Mapping[str, Any]) -> dict[str, str]:
    component = target["component"].strip()
    metric_id = target["metric_id"].strip()
    if component == "tool_descriptions":
        return {
            "candidate_bundle_phase": "Phase 2: Tool Description Evolution",
            "candidate_bundle_target": "tool-description",
            "runner_hint": "python -m evolution.tools.evolve_tool_descriptions",
        }
    if component == "skill_usage":
        return {
            "candidate_bundle_phase": "Phase 1: Skill Evolution",
            "candidate_bundle_target": "skill-usage",
            "runner_hint": "python -m evolution.skills.evolve_skill",
        }
    if component == "system_prompts":
        return {
            "candidate_bundle_phase": "Phase 3: System Prompt Evolution",
            "candidate_bundle_target": "system-prompts",
            "runner_hint": "phase3-system-prompt-candidate-runner-not-enabled-in-scheduler-dry-run",
        }
    if component in {"tool_code", "tool_implementation", "tool_implementation_code"}:
        return {
            "candidate_bundle_phase": "Phase 4: Tool Implementation Evolution",
            "candidate_bundle_target": "tool-implementation",
            "runner_hint": "phase4-code-candidate-runner-not-enabled-in-scheduler-dry-run",
        }
    return {
        "candidate_bundle_phase": "Phase 5: Continuous Self-Improvement Loop",
        "candidate_bundle_target": _safe_target_slug(metric_id),
        "runner_hint": "manual-local-candidate-bundle-runner-selection-required",
    }


def _matching_candidate_bundle_decision(
    target: Mapping[str, Any],
    profile: Mapping[str, str],
    decisions: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    accepted_target = _strict_match_text(profile["candidate_bundle_target"])
    accepted_phase = _strict_match_text(profile["candidate_bundle_phase"])
    for decision in decisions:
        decision_target = _strict_match_text(str(decision["target"]))
        decision_phase = _strict_match_text(str(decision["phase"]))
        if decision_target == accepted_target and decision_phase == accepted_phase:
            return decision
    return None


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


def _safe_target_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value.strip()]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "phase5-target"


def _strict_match_text(value: str) -> str:
    return value.strip()


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 5 Scheduler Dry-Run",
        "",
        f"Status: {report['status']}",
        "",
        f"Mode: `{report['mode']}`",
        "",
        "## Safety",
        "",
        "- read_only=true",
        "- cron_jobs_created=false",
        "- benchmark_cron_enabled=false",
        "- scheduler_or_cron_side_effects_performed=false",
        "- notifications_sent=false",
        "- optimizer_execution_started=false",
        "- automated_pr_created_or_updated=false",
        "",
        "## Dry-Run Actions",
        "",
    ]
    if report["dry_run_actions"]:
        for action in report["dry_run_actions"]:
            lines.append(
                f"{action['action_id']}: `{action['target_metric_id']}` "
                f"priority_score={action['priority_score']}, "
                "would_create_cron_job=false, would_start_optimizer=false"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Local Candidate Bundle Queue", ""])
    if report.get("candidate_bundle_queue"):
        for item in report["candidate_bundle_queue"]:
            lines.append(
                f"{item['queue_id']}: `{item['target_metric_id']}` -> "
                f"{item['candidate_bundle_phase']} / `{item['candidate_bundle_target']}`, "
                f"decision_state={item['decision_state']}, would_start_runner=false"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Recommended Next Step",
            "",
            f"`{report['recommended_next_step']}`",
            "",
            "This scheduler dry-run is evidence for human review only; it is not approval to create cron jobs, enable benchmark cron, send notifications, run optimizers, spend benchmark/API budget, or update external pull requests automatically.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    _reject_private_or_raw_identifiers(markdown)
    return markdown


def _validate_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve(strict=False)
    root = PHASE5_OUTPUT_ROOT.resolve(strict=False)
    if output_dir == root or root not in output_dir.parents:
        raise ValueError("output-dir must be under output/phase5-continuous-loop")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output-dir must be a directory before writing scheduler dry-run artifacts")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output-dir must be empty before writing scheduler dry-run artifacts")
    return output_dir


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
                raise ValueError("scheduler dry-run input contains private/raw identifier")


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
    parser = argparse.ArgumentParser(description="Write a read-only Phase 5 scheduler dry-run report.")
    parser.add_argument("--auto-triage-report-json", required=True, type=Path)
    parser.add_argument(
        "--candidate-bundle-decision-json",
        action="append",
        default=[],
        type=Path,
        help="Optional local candidate bundle decision.json to consume in the dry-run queue.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args(argv)

    try:
        auto_triage_report = json.loads(args.auto_triage_report_json.read_text())
        candidate_bundle_decisions = [
            json.loads(path.read_text()) for path in args.candidate_bundle_decision_json
        ]
        write_scheduler_dry_run_report(
            auto_triage_report,
            output_dir=args.output_dir,
            candidate_bundle_decisions=candidate_bundle_decisions,
            generated_at=args.generated_at,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("wrote Phase 5 scheduler dry-run")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Real benchmark preflight writer for HSE strict benchmark execution.

This module prepares the last local preflight contract before a future real
benchmark run. It validates that the planned output root is fresh and scoped,
records disposable baseline/current worktree materialization plans, records a
rollback cleanup plan, and writes a dry-run command preview. It intentionally
does not create benchmark worktrees, start benchmark processes, spend provider
or model budget, query/write GitHub, mutate active runtime state, or restart
Hermes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.local_completion.real_benchmark_approval_packet import (
    APPROVAL_RECORDED_NOT_EXECUTED,
    AWAITING_EXPLICIT_BENCHMARK_APPROVAL,
)
from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags

PREFLIGHT_SCHEMA_VERSION = "hse-real-benchmark-preflight-v1"
COMMAND_PREVIEW_SCHEMA_VERSION = "hse-real-benchmark-command-preview-v1"
PREFLIGHT_GATE_ID = "B0-RBP"
PREFLIGHT_PHASE = "Real Benchmark Preflight"
PREFLIGHT_TARGET = "strict-plan-real-benchmark-preflight"
PREFLIGHT_RECORDED_NOT_EXECUTABLE = "PREFLIGHT_RECORDED_NOT_EXECUTABLE"
PREFLIGHT_EXECUTION_READY_NOT_STARTED = "PREFLIGHT_EXECUTION_READY_NOT_STARTED"
ALLOWED_OUTPUT_PARTS = ("output", "hse-real-benchmark")
DEFAULT_BENCHMARK_SUITES = ["TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples"]


def write_real_benchmark_preflight(
    *,
    approval_packet_path: str | Path,
    output_dir: str | Path,
    future_output_root: str | Path,
    generated_at: str,
    dry_run: bool,
) -> dict[str, str]:
    """Write a non-executing real benchmark preflight report.

    ``dry_run`` must be exactly ``True``. The resulting report can validate the
    plan and command preview, but execution readiness remains false unless the
    upstream approval packet already contains complete explicit approval.
    """

    _require_non_empty("generated_at", generated_at)
    if dry_run is not True:
        raise ValueError("dry_run must be exactly true")

    approval_path = Path(approval_packet_path).expanduser()
    approval = _load_json_object(approval_path, "approval packet")
    _validate_approval_packet(approval)

    future_root = Path(future_output_root).expanduser()
    _validate_future_output_root(future_root, input_paths=[approval_path])
    backfill_path = _source_backfill_path(approval)
    if backfill_path is not None:
        _validate_future_output_root(future_root, input_paths=[approval_path, backfill_path])
        backfill = _load_json_object(backfill_path, "source backfill")
    else:
        backfill = {}

    suites = _benchmark_suite_names(approval)
    approval_complete = approval.get("approval_complete") is True
    preflight_passed = True
    execution_ready = _execution_ready(approval=approval, preflight_passed=preflight_passed)
    status = PREFLIGHT_EXECUTION_READY_NOT_STARTED if execution_ready else PREFLIGHT_RECORDED_NOT_EXECUTABLE

    out = Path(output_dir).expanduser()
    inputs_dir = out / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    approval_snapshot_path = inputs_dir / "real_benchmark_approval_packet.json"
    approval_snapshot_path.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n")
    backfill_snapshot_path = None
    if backfill:
        backfill_snapshot_path = inputs_dir / "benchmark_gate_backfill.json"
        backfill_snapshot_path.write_text(json.dumps(backfill, indent=2, sort_keys=True) + "\n")

    baseline_commit = _subject_commit(backfill, "baseline")
    current_commit = _subject_commit(backfill, "current")
    command_preview = _command_preview(
        suites=suites,
        future_output_root=future_root,
        baseline_commit=baseline_commit,
        current_commit=current_commit,
        generated_at=generated_at,
    )
    command_preview_path = out / "real_benchmark_command_preview.json"
    command_preview_path.write_text(json.dumps(command_preview, indent=2, sort_keys=True) + "\n")

    report = base_decision_payload(
        gate_id=PREFLIGHT_GATE_ID,
        phase=PREFLIGHT_PHASE,
        target=PREFLIGHT_TARGET,
        generated_at=generated_at,
    )
    report["schema_version"] = PREFLIGHT_SCHEMA_VERSION
    report.update(
        {
            "status": status,
            "summary": _summary(status),
            "preflight_passed": preflight_passed,
            "strict_plan_gate_closed": False,
            "execution_ready": execution_ready,
            "dry_run_only": True,
            "execution_started": False,
            "real_benchmarks_executed": False,
            "real_benchmark_execution_approved": approval_complete,
            "approval_complete": approval_complete,
            "current_authorized_budget_usd": approval.get("current_authorized_budget_usd", 0) if approval_complete else 0,
            "current_authorized_budget_krw": approval.get("current_authorized_budget_krw", 0) if approval_complete else 0,
            "approved_runtime_minutes": approval.get("approved_runtime_minutes") if approval_complete else None,
            "network_provider_spend_allowed": bool(approval.get("network_provider_spend_allowed") and approval_complete),
            "github_policy": "NO_GITHUB_WRITE",
            "benchmark_suites": suites,
            "regression_thresholds": dict(approval.get("regression_thresholds", {})),
            "source_approval_packet": {
                "path": str(approval_path),
                "sha256": _sha256_path(approval_path),
                "status": approval.get("status"),
                "approval_complete": approval.get("approval_complete"),
                "execution_started": approval.get("execution_started"),
                "real_benchmarks_executed": approval.get("real_benchmarks_executed"),
            },
            "source_backfill": _backfill_summary(backfill_path, backfill),
            "blocked_by": _blocked_by(approval),
            "baseline_materialization_plan": _materialization_plan(
                subject="baseline",
                source_commit=baseline_commit,
                future_output_root=future_root,
                allowed_by_packet=bool(approval.get("baseline_materialization_allowed") and approval_complete),
            ),
            "current_materialization_plan": _materialization_plan(
                subject="current",
                source_commit=current_commit,
                future_output_root=future_root,
                allowed_by_packet=bool(approval.get("current_materialization_allowed") and approval_complete),
            ),
            "output_root_guard": _output_root_guard(future_root),
            "write_root_guard": {
                "allowed_write_roots": [str(future_root)],
                "future_output_root": str(future_root),
                "write_targets_created_now": [
                    str(out / "real_benchmark_preflight.json"),
                    str(out / "real_benchmark_preflight.md"),
                    str(command_preview_path),
                ],
                "benchmark_output_written_now": False,
                "passed": True,
            },
            "rollback_cleanup_plan": _rollback_cleanup_plan(future_root),
            "command_dry_run": {
                "dry_run": True,
                "command_preview_generated": True,
                "command_preview_path": str(command_preview_path),
                "benchmark_commands_started": False,
                "benchmark_process_started": False,
                "commands_have_dry_run_flag": all("--dry-run" in command["argv"] for command in command_preview["commands"]),
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
                "preflight_report": "real_benchmark_preflight.json",
                "preflight_markdown": "real_benchmark_preflight.md",
                "command_preview": "real_benchmark_command_preview.json",
                "approval_packet_snapshot": "inputs/real_benchmark_approval_packet.json",
                "backfill_snapshot": "inputs/benchmark_gate_backfill.json" if backfill_snapshot_path else None,
            },
            "required_next_action": "collect_explicit_benchmark_approval_before_execution"
            if not execution_ready
            else "execute_real_benchmarks_under_preflight_after_final_human_go",
        }
    )
    reject_github_or_active_apply_flags(report)

    report_path = out / "real_benchmark_preflight.json"
    markdown_path = out / "real_benchmark_preflight.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_preflight_markdown(report))

    result = {
        "preflight_report_path": str(report_path),
        "preflight_markdown_path": str(markdown_path),
        "command_preview_path": str(command_preview_path),
        "approval_packet_snapshot_path": str(approval_snapshot_path),
    }
    if backfill_snapshot_path is not None:
        result["backfill_snapshot_path"] = str(backfill_snapshot_path)
    return result


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
    if approval.get("schema_version") != "hse-real-benchmark-approval-packet-v1":
        raise ValueError("approval packet schema_version mismatch")
    if approval.get("status") not in {AWAITING_EXPLICIT_BENCHMARK_APPROVAL, APPROVAL_RECORDED_NOT_EXECUTED}:
        raise ValueError("approval packet status must be an HSE real benchmark approval status")
    if approval.get("execution_started") is not False or approval.get("real_benchmarks_executed") is not False:
        raise ValueError("approval packet must not have started execution")
    if approval.get("github", {}).get("push_performed") is True:
        raise ValueError("approval packet must preserve NO_GITHUB_WRITE")


def _validate_future_output_root(future_output_root: Path, *, input_paths: list[Path]) -> None:
    for input_path in input_paths:
        if _paths_overlap(future_output_root, input_path):
            raise ValueError("future_output_root must not overlap input artifacts")
    parts = future_output_root.parts
    if not any(parts[index : index + 2] == ALLOWED_OUTPUT_PARTS for index in range(max(len(parts) - 1, 0))):
        raise ValueError("future_output_root must be under an output/hse-real-benchmark root")
    if future_output_root.exists():
        raise ValueError("future_output_root must not already exist")
    for parent in [future_output_root, *future_output_root.parents]:
        if parent.exists() and parent.is_symlink():
            raise ValueError("future_output_root must not traverse symlink components")


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return left_resolved == right_resolved or right_resolved in left_resolved.parents or left_resolved in right_resolved.parents


def _source_backfill_path(approval: Mapping[str, Any]) -> Path | None:
    source = approval.get("source_backfill")
    if not isinstance(source, Mapping):
        return None
    path = source.get("path")
    if not isinstance(path, str) or not path:
        return None
    return Path(path).expanduser()


def _benchmark_suite_names(approval: Mapping[str, Any]) -> list[str]:
    suites = approval.get("requested_benchmark_suites")
    names: list[str] = []
    if isinstance(suites, list):
        for suite in suites:
            if isinstance(suite, Mapping) and isinstance(suite.get("name"), str) and suite["name"]:
                names.append(str(suite["name"]))
            elif isinstance(suite, str) and suite:
                names.append(suite)
    return names or list(DEFAULT_BENCHMARK_SUITES)


def _execution_ready(*, approval: Mapping[str, Any], preflight_passed: bool) -> bool:
    return bool(
        preflight_passed
        and approval.get("approval_complete") is True
        and approval.get("real_benchmark_execution_approved") is True
        and approval.get("baseline_materialization_allowed") is True
        and approval.get("current_materialization_allowed") is True
        and approval.get("execution_started") is False
        and approval.get("real_benchmarks_executed") is False
    )


def _subject_commit(backfill: Mapping[str, Any], subject: str) -> str | None:
    subjects = backfill.get("benchmark_subjects")
    if not isinstance(subjects, Mapping):
        return None
    payload = subjects.get(subject)
    if not isinstance(payload, Mapping):
        return None
    source = payload.get("hermes_source")
    if not isinstance(source, Mapping):
        return None
    commit = source.get("commit")
    return str(commit) if isinstance(commit, str) and commit else None


def _backfill_summary(path: Path | None, backfill: Mapping[str, Any]) -> dict[str, Any] | None:
    if path is None or not backfill:
        return None
    return {
        "path": str(path),
        "sha256": _sha256_path(path),
        "status": backfill.get("status"),
        "strict_plan_gate_closed": backfill.get("strict_plan_gate_closed"),
        "benchmark_gate_passed": backfill.get("benchmark_gate_passed"),
    }


def _blocked_by(approval: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if approval.get("approval_complete") is not True:
        blockers.append("awaiting_explicit_human_benchmark_approval")
    missing = approval.get("missing_approval_fields")
    if isinstance(missing, list):
        blockers.extend(str(field) for field in missing if isinstance(field, str) and field)
    return blockers


def _materialization_plan(
    *,
    subject: str,
    source_commit: str | None,
    future_output_root: Path,
    allowed_by_packet: bool,
) -> dict[str, Any]:
    safe_commit = source_commit or "unknown"
    worktree_path = future_output_root.parent / "_worktrees" / future_output_root.name / f"{subject}-{safe_commit}"
    return {
        "subject": subject,
        "source_commit": source_commit,
        "planned_worktree_path": str(worktree_path),
        "disposable": True,
        "allowed_by_packet": allowed_by_packet,
        "materialization_started": False,
        "worktree_created": False,
        "cleanup_required_if_created": True,
        "cleanup_started": False,
    }


def _output_root_guard(future_output_root: Path) -> dict[str, Any]:
    return {
        "passed": True,
        "future_output_root": str(future_output_root),
        "allowed_root_fragment": "/".join(ALLOWED_OUTPUT_PARTS),
        "future_output_root_exists_now": future_output_root.exists(),
        "future_output_root_is_symlink_now": future_output_root.is_symlink(),
        "fresh_output_required": True,
        "symlink_output_allowed": False,
        "hardlink_output_allowed": False,
        "input_output_overlap_allowed": False,
        "benchmark_output_written_now": False,
    }


def _rollback_cleanup_plan(future_output_root: Path) -> dict[str, Any]:
    return {
        "rollback_plan_verified": True,
        "cleanup_started": False,
        "delete_future_output_root_if_created": str(future_output_root),
        "remove_disposable_worktrees_if_created": True,
        "preserve_preflight_report_artifacts": True,
        "verify_after_cleanup": [
            "future output root absent or intentionally archived",
            "disposable baseline/current worktrees absent",
            "HSE and Hermes git heads unchanged unless committed locally",
        ],
    }


def _command_preview(
    *,
    suites: list[str],
    future_output_root: Path,
    baseline_commit: str | None,
    current_commit: str | None,
    generated_at: str,
) -> dict[str, Any]:
    commands = []
    for suite in suites:
        suite_slug = _slug(suite)
        commands.append(
            {
                "suite": suite,
                "argv": [
                    "python",
                    "-m",
                    "evolution.benchmarks.real_benchmark_runner",
                    "--suite",
                    suite,
                    "--baseline-commit",
                    baseline_commit or "UNKNOWN_BASELINE_COMMIT",
                    "--current-commit",
                    current_commit or "UNKNOWN_CURRENT_COMMIT",
                    "--output-json",
                    str(future_output_root / f"{suite_slug}.json"),
                    "--dry-run",
                ],
                "started": False,
                "network_allowed": False,
                "provider_spend_allowed": False,
            }
        )
    return {
        "schema_version": COMMAND_PREVIEW_SCHEMA_VERSION,
        "generated_at": generated_at,
        "dry_run": True,
        "benchmark_commands_started": False,
        "commands": commands,
    }


def _slug(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in value).strip("-")
    return "-".join(part for part in slug.split("-") if part) or "benchmark"


def _summary(status: str) -> str:
    if status == PREFLIGHT_EXECUTION_READY_NOT_STARTED:
        return "Real benchmark preflight is valid and approval is complete, but execution has not started."
    return "Real benchmark preflight plan is valid, but execution remains blocked by missing explicit approval."


def _render_preflight_markdown(report: Mapping[str, Any]) -> str:
    blockers = report.get("blocked_by", [])
    blocker_lines = [f"- {blocker}" for blocker in blockers] if isinstance(blockers, list) else []
    suites = report.get("benchmark_suites", [])
    suite_lines = [f"- {suite}" for suite in suites] if isinstance(suites, list) else []
    return "\n".join(
        [
            "# HSE Real Benchmark Preflight",
            "",
            f"Status: `{report.get('status')}`",
            "",
            "This preflight is not approval to execute. It records a dry-run-only command preview and guarded plans.",
            "",
            "## Execution State",
            "",
            f"- preflight_passed={str(report.get('preflight_passed')).lower()}",
            f"- execution_ready={str(report.get('execution_ready')).lower()}",
            f"- dry_run_only={str(report.get('dry_run_only')).lower()}",
            f"- execution_started={str(report.get('execution_started')).lower()}",
            f"- real_benchmarks_executed={str(report.get('real_benchmarks_executed')).lower()}",
            "",
            "## Suites",
            "",
            *suite_lines,
            "",
            "## Blockers",
            "",
            *(blocker_lines or ["- none"]),
            "",
            "## Boundaries",
            "",
            "- NO_GITHUB_WRITE",
            "- benchmark process started: false",
            "- provider/model/API spend performed: false",
            "- worktree materialization performed: false",
            "- gateway restart/reload performed: false",
            "",
        ]
    )


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_non_empty(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a dry-run-only HSE real benchmark preflight report.")
    parser.add_argument("--approval-packet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--future-output-root", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args(argv)
    result = write_real_benchmark_preflight(
        approval_packet_path=args.approval_packet,
        output_dir=args.output_dir,
        future_output_root=args.future_output_root,
        generated_at=args.generated_at,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

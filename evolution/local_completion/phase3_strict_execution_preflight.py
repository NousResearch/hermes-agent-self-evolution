"""HSE Phase 3 strict execution preflight.

This gate checks whether Phase 3 system-prompt candidate execution can be
prepared as a separate local-only step after the current active frontier reaches
Phase 2. It intentionally does not run GEPA/DSPy, benchmarks, active prompt
apply, GitHub operations, provider/network calls, cron, gateway, deploy, merge,
or publication.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags

PHASE3_PREFLIGHT_SCHEMA_VERSION = "hse-phase3-strict-execution-preflight-v1"
PHASE3_COMMAND_PREVIEW_SCHEMA_VERSION = "hse-phase3-strict-execution-command-preview-v1"
PHASE3_PREFLIGHT_GATE_ID = "SFA-P3-PREFLIGHT"
PHASE3_PREFLIGHT_PHASE = "Phase 3 Strict Execution Preflight"
PHASE3_PREFLIGHT_TARGET = "phase3-system-prompt-strict-execution-preflight"
PHASE3_STRICT_EXECUTION_PREFLIGHT_READY_SEPARATE_GO_REQUIRED = (
    "PHASE3_STRICT_EXECUTION_PREFLIGHT_READY_SEPARATE_GO_REQUIRED"
)
BLOCKED_PHASE3_STRICT_EXECUTION_PREFLIGHT = "BLOCKED_PHASE3_STRICT_EXECUTION_PREFLIGHT"

ACTIVE_PROMPT_SOURCE_FILES = (
    "agent/prompt_builder.py",
    "tests/agent/test_prompt_builder.py",
)
HSE_SUPPORT_FILES = (
    "evolution/prompts/phase3_candidate_scaffold.py",
    "evolution/prompts/phase3_preflight_gate.py",
    "evolution/prompts/phase3_gepa_optimizer.py",
    "evolution/benchmarks/run_tblite.py",
    "evolution/benchmarks/run_yc_bench.py",
    "datasets/golden/benchmarks/phase3-system-prompt/baseline_system_prompt.json",
    "datasets/golden/benchmarks/phase3-system-prompt/candidate_system_prompt.json",
    "datasets/golden/benchmarks/phase3-system-prompt/tblite_cases.jsonl",
    "datasets/golden/benchmarks/phase3-system-prompt/yc_bench_fast_test.jsonl",
)
PHASE3_ALLOWED_OUTPUT_ROOT_FRAGMENT = "output/phase3-system-prompt"
FORBIDDEN_INPUT_TRUE_FIELDS = (
    "github_query_performed",
    "github_write_performed",
    "provider_or_model_spend_performed",
    "network_calls_performed",
    "active_apply_performed",
    "full_remote_benchmark_executed",
    "overall_hse_project_completion_claimed",
)


@dataclass(frozen=True)
class GitResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def write_phase3_strict_execution_preflight(
    *,
    active_hermes_repo: str | Path,
    hse_repo_root: str | Path,
    strict_frontier_audit_path: str | Path,
    phase3_plan_path: str | Path,
    phase3_readiness_path: str | Path,
    phase3_execution_seed_draft_path: str | Path,
    plan_path: str | Path,
    output_dir: str | Path,
    generated_at: str,
    future_run_id: str,
) -> dict[str, str]:
    """Write a non-executing Phase 3 strict execution preflight."""

    _require_non_empty("generated_at", generated_at)
    _require_non_empty("future_run_id", future_run_id)
    _validate_run_id(future_run_id)
    active_repo = Path(active_hermes_repo).expanduser().resolve()
    hse_root = Path(hse_repo_root).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    paths = {
        "strict_frontier_audit": Path(strict_frontier_audit_path).expanduser().resolve(),
        "phase3_plan": Path(phase3_plan_path).expanduser().resolve(),
        "phase3_readiness": Path(phase3_readiness_path).expanduser().resolve(),
        "phase3_execution_seed_draft": Path(phase3_execution_seed_draft_path).expanduser().resolve(),
        "plan": Path(plan_path).expanduser().resolve(),
    }
    _validate_git_repo(active_repo)
    if not hse_root.exists():
        raise FileNotFoundError(f"HSE repo root not found: {hse_root}")
    data = {name: _load_json_object(path, name) for name, path in paths.items() if name != "plan"}
    plan_text = paths["plan"].read_text()

    active = _active_prompt_source_inventory(active_repo)
    strict_frontier_gate = _strict_frontier_gate(data["strict_frontier_audit"], active)
    phase3_contract = _phase3_contract(data["phase3_plan"], data["phase3_readiness"], data["phase3_execution_seed_draft"], plan_text)
    support_inventory = _support_inventory(hse_root)
    future_output_root = hse_root / PHASE3_ALLOWED_OUTPUT_ROOT_FRAGMENT / future_run_id
    future_root_guard = _future_output_root_guard(future_output_root)
    command_preview = _command_preview(future_output_root, future_run_id)

    blocked_by = _blocked_by(
        active=active,
        strict_frontier_gate=strict_frontier_gate,
        phase3_contract=phase3_contract,
        support_inventory=support_inventory,
        future_root_guard=future_root_guard,
        source_data=data,
    )
    status = (
        PHASE3_STRICT_EXECUTION_PREFLIGHT_READY_SEPARATE_GO_REQUIRED
        if not blocked_by
        else BLOCKED_PHASE3_STRICT_EXECUTION_PREFLIGHT
    )
    preflight_passed = not blocked_by

    report = base_decision_payload(
        gate_id=PHASE3_PREFLIGHT_GATE_ID,
        phase=PHASE3_PREFLIGHT_PHASE,
        target=PHASE3_PREFLIGHT_TARGET,
        generated_at=generated_at,
    )
    report["schema_version"] = PHASE3_PREFLIGHT_SCHEMA_VERSION
    report.update(
        {
            "status": status,
            "summary": _summary(status),
            "preflight_passed": preflight_passed,
            "blocked_by": blocked_by,
            "strict_frontier_gate": strict_frontier_gate,
            "phase3_contract": phase3_contract,
            "active_prompt_source_inventory": active,
            "support_inventory": support_inventory,
            "local_only_benchmark_readiness": {
                "dry_run_fixture_benchmarks_ready": support_inventory["all_required_support_files_present"],
                "local_fixture_adapters_available": support_inventory["all_required_support_files_present"],
                "real_benchmark_ready_now": False,
                "full_remote_benchmark_executed": False,
                "provider_or_model_spend_performed": False,
                "network_calls_performed": False,
                "local_only_scope": True,
                "separate_benchmark_execution_go_required": True,
            },
            "runtime_candidate_execution_preflight": {
                "candidate_execution_possible_after_separate_go": preflight_passed,
                "system_prompt_candidate_scaffold_available": support_inventory["support_files_by_path"].get(
                    "evolution/prompts/phase3_candidate_scaffold.py", {}
                ).get("exists")
                is True,
                "phase3_preflight_gate_available": support_inventory["support_files_by_path"].get(
                    "evolution/prompts/phase3_preflight_gate.py", {}
                ).get("exists")
                is True,
                "gepa_optimizer_available_but_not_started": support_inventory["support_files_by_path"].get(
                    "evolution/prompts/phase3_gepa_optimizer.py", {}
                ).get("exists")
                is True,
                "run_gepa_now": False,
                "run_dspy_now": False,
                "execution_started": False,
            },
            "future_run_id": future_run_id,
            "future_output_root_guard": future_root_guard,
            "command_preview": {
                "path": str(out / "phase3_strict_execution_command_preview.json"),
                "relative_path": "phase3_strict_execution_command_preview.json",
                "schema_version": PHASE3_COMMAND_PREVIEW_SCHEMA_VERSION,
                "dry_run": True,
                "commands_started": False,
                "all_commands_preview_only": all(command["started"] is False for command in command_preview["commands"]),
            },
            "phase3_execution_ready": False,
            "separate_phase3_execution_go_required": preflight_passed,
            "execution_started": False,
            "run_gepa_now": False,
            "run_dspy_now": False,
            "mutate_active_system_prompt_now": False,
            "active_system_prompt_apply_approved": False,
            "active_system_prompt_apply_ready": False,
            "real_benchmarks_executed": False,
            "full_remote_benchmark_executed": False,
            "github_query_performed": False,
            "github_write_performed": False,
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "active_apply_performed": False,
            "cron_or_gateway_mutation_performed": False,
            "deploy_or_publication_performed": False,
            "overall_hse_project_completion_claimed": False,
            "side_effect_boundaries": {
                "report_files_written": True,
                "future_output_root_created": False,
                "phase3_candidate_execution_started": False,
                "benchmark_process_started": False,
                "gepa_or_dspy_started": False,
                "active_prompt_or_source_modified": False,
                "active_runtime_mutated": False,
                "github_query_or_write": False,
                "provider_or_model_spend": False,
                "network_calls": False,
                "cron_or_gateway_mutation": False,
                "deploy_or_publication": False,
            },
            "not_claimed": [
                "phase3_strict_completion",
                "phase4_strict_completion",
                "phase5_strict_completion",
                "overall_HSE_project_completion",
                "full_remote_benchmark",
                "provider_api_spend",
                "github_query_or_write",
                "active_apply",
                "cron_or_gateway_mutation",
                "deploy_or_publication",
            ],
            "required_next_action": "phase3_candidate_execution_dry_run_go_no_remote_no_provider_no_github_write"
            if preflight_passed
            else "repair_phase3_strict_execution_preflight_blockers_before_execution_go",
            "source_artifacts": _source_artifacts(paths),
            "artifacts": {
                "preflight_report": "phase3_strict_execution_preflight.json",
                "preflight_markdown": "phase3_strict_execution_preflight.md",
                "command_preview": "phase3_strict_execution_command_preview.json",
            },
        }
    )
    reject_github_or_active_apply_flags(report)

    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "phase3_strict_execution_preflight.json"
    markdown_path = out / "phase3_strict_execution_preflight.md"
    command_preview_path = out / "phase3_strict_execution_command_preview.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    command_preview_path.write_text(json.dumps(command_preview, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown_path.write_text(_render_markdown(report))
    return {
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
        "command_preview_path": str(command_preview_path),
    }


def _validate_git_repo(repo: Path) -> None:
    if not repo.exists():
        raise FileNotFoundError(f"active Hermes repo not found: {repo}")
    result = _run_git(repo, "rev-parse", "--is-inside-work-tree")
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise ValueError(f"not a git worktree: {repo}")


def _run_git(repo: Path, *args: str) -> GitResult:
    completed = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True, check=False)
    return GitResult(tuple(args), completed.returncode, completed.stdout, completed.stderr)


def _git_stdout(repo: Path, *args: str) -> str:
    result = _run_git(repo, *args)
    if result.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _active_prompt_source_inventory(repo: Path) -> dict[str, Any]:
    head = _git_stdout(repo, "rev-parse", "HEAD")
    branch = _git_stdout(repo, "rev-parse", "--abbrev-ref", "HEAD")
    root = _git_stdout(repo, "rev-parse", "--show-toplevel")
    status = _run_git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0:
        raise ValueError(f"git status failed: {status.stderr.strip()}")
    dirty_entries = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "repo_root": root,
        "head": head,
        "head_short": head[:9],
        "branch": branch,
        "clean": not dirty_entries,
        "dirty_file_count": len(dirty_entries),
        "dirty_entries": dirty_entries,
        "target_files": [_file_inventory(repo, rel) for rel in ACTIVE_PROMPT_SOURCE_FILES],
    }


def _file_inventory(root: Path, rel: str) -> dict[str, Any]:
    path = root / rel
    exists = path.exists()
    is_file = exists and path.is_file()
    return {
        "relative_path": rel,
        "exists": exists,
        "is_file": is_file,
        "bytes": path.stat().st_size if is_file else None,
        "sha256": sha256(path.read_bytes()).hexdigest() if is_file else None,
    }


def _strict_frontier_gate(frontier: Mapping[str, Any], active: Mapping[str, Any]) -> dict[str, Any]:
    current = frontier.get("current_active_frontier", {}) if isinstance(frontier.get("current_active_frontier"), Mapping) else {}
    phases = frontier.get("phases", {}) if isinstance(frontier.get("phases"), Mapping) else {}
    phase3 = phases.get("phase3", {}) if isinstance(phases.get("phase3"), Mapping) else {}
    active_from_frontier = frontier.get("active_hermes", {}) if isinstance(frontier.get("active_hermes"), Mapping) else {}
    return {
        "schema_version": frontier.get("schema_version"),
        "status": frontier.get("status"),
        "current_active_frontier_status": current.get("status"),
        "current_active_highest_strict_complete_phase": current.get("highest_strict_complete_phase"),
        "current_active_frontier_blockers": current.get("blockers", []),
        "phase1_strict_complete": _nested_bool(phases, "phase1", "strict_complete"),
        "phase2_strict_complete": _nested_bool(phases, "phase2", "strict_complete"),
        "phase3_strict_complete": phase3.get("strict_complete") is True,
        "phase3_existing_blockers": phase3.get("blockers", []),
        "active_head_from_frontier": active_from_frontier.get("head"),
        "active_head_matches_live_checkout": active_from_frontier.get("head") == active.get("head"),
        "source_status_is_local_only": all(frontier.get(field) is False for field in FORBIDDEN_INPUT_TRUE_FIELDS),
    }


def _phase3_contract(
    phase3_plan: Mapping[str, Any], phase3_readiness: Mapping[str, Any], seed_draft: Mapping[str, Any], plan_text: str
) -> dict[str, Any]:
    readiness_state = phase3_readiness.get("ready_state", {}) if isinstance(phase3_readiness.get("ready_state"), Mapping) else {}
    return {
        "plan_phase3_gate_present": "Behavioral tests pass" in plan_text and "benchmarks hold or improve" in plan_text,
        "phase3_plan_status": phase3_plan.get("status"),
        "phase3_plan_not_executed": phase3_plan.get("execution_started") is False and phase3_plan.get("apply_ready") is False,
        "readiness_status": phase3_readiness.get("status"),
        "readiness_recorded_not_executed": phase3_readiness.get("execution_started") is False
        and phase3_readiness.get("real_benchmarks_executed") is False
        and phase3_readiness.get("phase3_execution_ready") is False,
        "readiness_active_apply_ready_now": readiness_state.get("active_apply_ready_now") is True,
        "readiness_real_benchmark_ready_now": readiness_state.get("real_benchmark_ready_now") is True,
        "execution_seed_status": seed_draft.get("status"),
        "execution_seed_drafted_not_executed": seed_draft.get("execution_started") is False
        and seed_draft.get("run_gepa_now") is False
        and seed_draft.get("run_dspy_now") is False
        and seed_draft.get("mutate_active_system_prompt_now") is False,
        "execution_requires_human_approval": seed_draft.get("requires_human_approval_before_execution") is True,
        "source_status_is_local_only": _inputs_side_effect_free(phase3_plan)
        and _inputs_side_effect_free(phase3_readiness)
        and _inputs_side_effect_free(seed_draft),
    }


def _support_inventory(hse_root: Path) -> dict[str, Any]:
    files = [_file_inventory(hse_root, rel) for rel in HSE_SUPPORT_FILES]
    by_path = {item["relative_path"]: item for item in files}
    return {
        "hse_repo_root": str(hse_root),
        "required_support_files": list(HSE_SUPPORT_FILES),
        "support_files": files,
        "support_files_by_path": by_path,
        "missing_required_support_files": [item["relative_path"] for item in files if not item["exists"] or not item["is_file"]],
        "all_required_support_files_present": all(item["exists"] and item["is_file"] for item in files),
    }


def _future_output_root_guard(root: Path) -> dict[str, Any]:
    return {
        "future_output_root": str(root),
        "allowed_root_fragment": PHASE3_ALLOWED_OUTPUT_ROOT_FRAGMENT,
        "future_output_root_exists_now": root.exists(),
        "future_output_root_is_symlink_now": root.is_symlink(),
        "future_output_root_created_now": False,
        "phase3_output_written_now": False,
        "fresh_output_required": True,
        "passed": not root.exists(),
    }


def _command_preview(future_output_root: Path, run_id: str) -> dict[str, Any]:
    root = str(future_output_root)
    commands = [
        {
            "name": "phase3_candidate_scaffold_dry_run",
            "side_effect_class": "preview_only",
            "started": False,
            "argv": [
                "python",
                "-m",
                "evolution.prompts.phase3_candidate_scaffold",
                "--baseline-prompt",
                f"{root}/inputs/baseline_system_prompt.json",
                "--candidate-prompt",
                f"{root}/inputs/candidate_system_prompt.json",
                "--output-dir",
                f"{root}/review",
                "--dry-run",
            ],
        },
        {
            "name": "phase3_tblite_local_fixture_dry_run",
            "side_effect_class": "preview_only",
            "started": False,
            "argv": [
                "python",
                "-m",
                "evolution.benchmarks.run_tblite",
                "--baseline-prompt",
                f"{root}/review/baseline_system_prompt.json",
                "--candidate-prompt",
                f"{root}/review/candidate_system_prompt.json",
                "--fixtures-jsonl",
                "datasets/golden/benchmarks/phase3-system-prompt/tblite_cases.jsonl",
                "--output-json",
                f"{root}/benchmarks/tblite.json",
                "--dry-run",
            ],
        },
        {
            "name": "phase3_yc_bench_local_fixture_dry_run",
            "side_effect_class": "preview_only",
            "started": False,
            "argv": [
                "python",
                "-m",
                "evolution.benchmarks.run_yc_bench",
                "--baseline-prompt",
                f"{root}/review/baseline_system_prompt.json",
                "--candidate-prompt",
                f"{root}/review/candidate_system_prompt.json",
                "--fixtures-jsonl",
                "datasets/golden/benchmarks/phase3-system-prompt/yc_bench_fast_test.jsonl",
                "--preset",
                "fast_test",
                "--output-json",
                f"{root}/benchmarks/yc_bench.json",
                "--dry-run",
            ],
        },
        {
            "name": "phase3_local_preflight_gate_dry_run",
            "side_effect_class": "preview_only",
            "started": False,
            "argv": [
                "python",
                "-m",
                "evolution.prompts.phase3_preflight_gate",
                "--candidate-report",
                f"{root}/review/candidate_only_report.json",
                "--tblite-report",
                f"{root}/benchmarks/tblite.json",
                "--yc-bench-report",
                f"{root}/benchmarks/yc_bench.json",
                "--output-json",
                f"{root}/preflight/phase3_preflight_report.json",
                "--dry-run",
            ],
        },
    ]
    return {
        "schema_version": PHASE3_COMMAND_PREVIEW_SCHEMA_VERSION,
        "run_id": run_id,
        "dry_run": True,
        "commands_started": False,
        "commands": commands,
        "not_included_until_separate_approval": [
            "GEPA/DSPy optimizer execution",
            "real TBLite/YC-Bench execution",
            "active system prompt/source apply",
            "runtime restart/reload",
            "GitHub query/write",
            "provider/API spend",
        ],
    }


def _blocked_by(
    *,
    active: Mapping[str, Any],
    strict_frontier_gate: Mapping[str, Any],
    phase3_contract: Mapping[str, Any],
    support_inventory: Mapping[str, Any],
    future_root_guard: Mapping[str, Any],
    source_data: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if active.get("clean") is not True:
        blockers.append("active_hermes_worktree_dirty")
    if not all(item.get("exists") and item.get("is_file") for item in active.get("target_files", [])):
        blockers.append("active_prompt_source_file_missing")
    if strict_frontier_gate.get("schema_version") != "hse-strict-frontier-audit-v1":
        blockers.append("strict_frontier_schema_mismatch")
    if strict_frontier_gate.get("current_active_frontier_status") != "PHASE_2_STRICT_COMPLETE":
        blockers.append("current_active_frontier_not_phase2_strict_complete")
    if strict_frontier_gate.get("current_active_highest_strict_complete_phase") != 2:
        blockers.append("current_active_frontier_phase_not_2")
    if strict_frontier_gate.get("phase3_strict_complete") is True:
        blockers.append("phase3_already_claimed_strict_complete")
    if strict_frontier_gate.get("active_head_matches_live_checkout") is not True:
        blockers.append("strict_frontier_active_head_mismatch")
    if strict_frontier_gate.get("source_status_is_local_only") is not True:
        blockers.append("strict_frontier_forbidden_side_effect_recorded")
    if phase3_contract.get("plan_phase3_gate_present") is not True:
        blockers.append("plan_phase3_gate_not_detected")
    if phase3_contract.get("phase3_plan_not_executed") is not True:
        blockers.append("phase3_plan_not_in_preexecution_state")
    if phase3_contract.get("readiness_recorded_not_executed") is not True:
        blockers.append("phase3_readiness_not_recorded_not_executed")
    if phase3_contract.get("execution_seed_drafted_not_executed") is not True:
        blockers.append("phase3_execution_seed_not_drafted_preexecution")
    if phase3_contract.get("execution_requires_human_approval") is not True:
        blockers.append("phase3_execution_seed_missing_human_approval_gate")
    if phase3_contract.get("source_status_is_local_only") is not True:
        blockers.append("phase3_input_forbidden_side_effect_recorded")
    if support_inventory.get("all_required_support_files_present") is not True:
        blockers.append("phase3_support_files_missing")
    if future_root_guard.get("future_output_root_exists_now") is True:
        blockers.append("future_output_root_already_exists")
    if future_root_guard.get("future_output_root_is_symlink_now") is True:
        blockers.append("future_output_root_is_symlink")
    for label, payload in source_data.items():
        if not _inputs_side_effect_free(payload):
            blockers.append(f"{label}_forbidden_side_effect_recorded")
    return sorted(set(blockers))


def _inputs_side_effect_free(payload: Mapping[str, Any]) -> bool:
    return all(payload.get(field) is not True for field in FORBIDDEN_INPUT_TRUE_FIELDS) and all(
        payload.get(field) is not True
        for field in (
            "execution_started",
            "run_gepa_now",
            "run_dspy_now",
            "mutate_active_system_prompt_now",
            "active_system_prompt_apply_approved",
            "apply_ready",
            "phase3_execution_ready",
            "real_benchmark_execution_approved",
            "real_benchmarks_executed",
        )
    )


def _nested_bool(mapping: Mapping[str, Any], key: str, subkey: str) -> bool | None:
    child = mapping.get(key)
    if not isinstance(child, Mapping):
        return None
    value = child.get(subkey)
    return value if isinstance(value, bool) else None


def _source_artifacts(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        artifacts[name] = {"path": str(path), "sha256": _sha256_path(path), "bytes": path.stat().st_size}
    return artifacts


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    if not isinstance(data, dict):
        raise ValueError(f"{label} JSON root must be an object: {path}")
    return data


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _require_non_empty(field: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _validate_run_id(value: str) -> None:
    if not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("future_run_id must contain only letters, numbers, hyphen, and underscore")


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _summary(status: str) -> str:
    if status == PHASE3_STRICT_EXECUTION_PREFLIGHT_READY_SEPARATE_GO_REQUIRED:
        return "Phase 3 strict execution preflight is ready for a separate local-only candidate execution GO; no execution started."
    return "Phase 3 strict execution preflight is blocked; repair blockers before any execution GO."


def _render_markdown(report: Mapping[str, Any]) -> str:
    blockers = report.get("blocked_by", [])
    blocker_text = ", ".join(blockers) if isinstance(blockers, list) and blockers else "none"
    readiness = report.get("local_only_benchmark_readiness", {}) if isinstance(report.get("local_only_benchmark_readiness"), Mapping) else {}
    root_guard = report.get("future_output_root_guard", {}) if isinstance(report.get("future_output_root_guard"), Mapping) else {}
    return "\n".join(
        [
            "# Phase 3 Strict Execution Preflight",
            "",
            f"Status: `{report.get('status')}`",
            "",
            "## Gate",
            "",
            f"- preflight_passed: `{report.get('preflight_passed')}`",
            f"- phase3_execution_ready: `{report.get('phase3_execution_ready')}`",
            f"- separate_phase3_execution_go_required: `{report.get('separate_phase3_execution_go_required')}`",
            f"- blockers: {blocker_text}",
            "",
            "## Local-only readiness",
            "",
            f"- dry_run_fixture_benchmarks_ready: `{readiness.get('dry_run_fixture_benchmarks_ready')}`",
            f"- real_benchmark_ready_now: `{readiness.get('real_benchmark_ready_now')}`",
            f"- future_output_root: `{root_guard.get('future_output_root')}`",
            f"- future_output_root_exists_now: `{root_guard.get('future_output_root_exists_now')}`",
            "",
            "## Boundaries",
            "",
            "- No GEPA/DSPy execution started.",
            "- No benchmark process started.",
            "- No active prompt/source/runtime mutation performed.",
            "- No GitHub query/write, provider/API spend, network, cron/gateway, deploy, or publication performed.",
            "",
            "## Required Next Action",
            "",
            str(report.get("required_next_action")),
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an HSE Phase 3 strict execution preflight report.")
    parser.add_argument("--active-hermes-repo", required=True, type=Path)
    parser.add_argument("--hse-repo-root", required=True, type=Path)
    parser.add_argument("--strict-frontier-audit", required=True, type=Path)
    parser.add_argument("--phase3-plan", required=True, type=Path)
    parser.add_argument("--phase3-readiness", required=True, type=Path)
    parser.add_argument("--phase3-execution-seed-draft", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--future-run-id", required=True)
    args = parser.parse_args(argv)
    result = write_phase3_strict_execution_preflight(
        active_hermes_repo=args.active_hermes_repo,
        hse_repo_root=args.hse_repo_root,
        strict_frontier_audit_path=args.strict_frontier_audit,
        phase3_plan_path=args.phase3_plan,
        phase3_readiness_path=args.phase3_readiness,
        phase3_execution_seed_draft_path=args.phase3_execution_seed_draft,
        plan_path=args.plan,
        output_dir=args.output_dir,
        generated_at=args.generated_at,
        future_run_id=args.future_run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

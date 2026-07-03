"""Current-baseline Phase 1/2 revalidation preflight.

This module snapshots the current active Hermes baseline and prepares a dry-run
local-only benchmark smoke rerun plan. It intentionally does not start benchmark
processes, create benchmark output roots, spend provider/API budget, query/write
GitHub, mutate active runtime, or close strict PLAN gates.
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

from evolution.benchmarks.real_benchmark_runner import SUPPORTED_SUITES, suite_readiness
from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags

PREFLIGHT_SCHEMA_VERSION = "hse-current-baseline-revalidation-preflight-v1"
COMMAND_PREVIEW_SCHEMA_VERSION = "hse-current-baseline-revalidation-command-preview-v1"
PREFLIGHT_GATE_ID = "SFA-RP"
PREFLIGHT_PHASE = "Current Baseline Phase 1/2 Revalidation Preflight"
PREFLIGHT_TARGET = "current-baseline-phase1-phase2-local-smoke-revalidation-preflight"
PREFLIGHT_READY_SEPARATE_GO_REQUIRED = "CURRENT_BASELINE_REVALIDATION_PREFLIGHT_READY_SEPARATE_GO_REQUIRED"
BLOCKED_ACTIVE_BASELINE_DIRTY = "BLOCKED_ACTIVE_BASELINE_DIRTY"
BLOCKED_FUTURE_OUTPUT_ROOT_EXISTS = "BLOCKED_FUTURE_OUTPUT_ROOT_EXISTS"
BLOCKED_STRICT_FRONTIER_NOT_REVALIDATION_REQUIRED = "BLOCKED_STRICT_FRONTIER_NOT_REVALIDATION_REQUIRED"
BLOCKED_SUITE_ASSETS_NOT_READY = "BLOCKED_SUITE_ASSETS_NOT_READY"

ACTIVE_BASELINE_FILES = (
    "model_tools.py",
    "tools/registry.py",
    "tools/tool_description_overrides.py",
    "agent/prompt_builder.py",
)

SUITE_OUTPUT_FILES = {
    "TBLite": "tblite.json",
    "YC-Bench": "yc-bench.json",
    "Phase2 PLAN-scale tool-selection triples": "phase2-plan-scale-tool-selection-triples.json",
}


@dataclass(frozen=True)
class GitResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def write_current_baseline_revalidation_preflight(
    *,
    active_hermes_repo: str | Path,
    strict_frontier_audit_path: str | Path,
    hse_repo_root: str | Path,
    output_dir: str | Path,
    generated_at: str,
    future_run_id: str,
    suites: Sequence[str] = SUPPORTED_SUITES,
) -> dict[str, str]:
    """Write a non-executing current-baseline revalidation preflight."""

    _require_non_empty("generated_at", generated_at)
    _require_non_empty("future_run_id", future_run_id)
    _validate_run_id(future_run_id)
    active_repo = Path(active_hermes_repo).expanduser().resolve()
    hse_root = Path(hse_repo_root).expanduser().resolve()
    audit_path = Path(strict_frontier_audit_path).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    _validate_git_repo(active_repo)
    audit = _load_json_object(audit_path, "strict frontier audit")
    active = _active_inventory(active_repo)
    future_output_root = hse_root / "output" / "hse-real-benchmark" / future_run_id
    command_preview = _command_preview(
        suites=suites,
        baseline_commit=_closure_subject_commit(audit),
        current_commit=str(active["head"]),
        future_output_root=future_output_root,
    )
    readiness = _suite_readiness(suites)
    blocked_by = _blocked_by(audit=audit, active=active, future_output_root=future_output_root, suite_readiness_reports=readiness)
    status = _status(blocked_by)

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
            "blocked_by": blocked_by,
            "source_strict_frontier_audit": {
                "path": str(audit_path),
                "sha256": _sha256_path(audit_path),
                "status": audit.get("status"),
                "recorded_subject_status": _nested_get(audit, ("recorded_subject_frontier", "status")),
                "current_active_status": _nested_get(audit, ("current_active_frontier", "status")),
                "closure_subject_commit": _closure_subject_commit(audit),
            },
            "current_baseline_inventory": active,
            "baseline_commit_for_rerun": _closure_subject_commit(audit),
            "current_commit_for_rerun": active["head"],
            "future_run_id": future_run_id,
            "future_output_root_guard": {
                "future_output_root": str(future_output_root),
                "allowed_root_fragment": "output/hse-real-benchmark",
                "future_output_root_exists_now": future_output_root.exists(),
                "future_output_root_is_symlink_now": future_output_root.is_symlink(),
                "future_output_root_created_now": False,
                "benchmark_output_written_now": False,
                "fresh_output_required": True,
                "passed": not future_output_root.exists(),
            },
            "current_baseline_materialization": {
                "materialization_started": False,
                "worktree_created": False,
                "source_commit": active["head"],
                "planned_materialization_path": str(
                    hse_root / "output" / "hse-real-benchmark" / "_worktrees" / future_run_id / f"current-{active['head_short']}"
                ),
                "cleanup_required_if_created": True,
            },
            "rerun_decision": {
                "rerun_recommended": _strict_frontier_requires_revalidation(audit),
                "reason": "current active Hermes baseline differs from the closed Phase 1/2 benchmark subject",
                "rerun_approved_now": False,
                "separate_local_smoke_go_required": True,
                "local_smoke_rerun_ready_not_started": status == PREFLIGHT_READY_SEPARATE_GO_REQUIRED,
            },
            "suite_readiness": readiness,
            "command_preview": {
                "path": str(out / "current_baseline_revalidation_command_preview.json"),
                "relative_path": "current_baseline_revalidation_command_preview.json",
                "schema_version": COMMAND_PREVIEW_SCHEMA_VERSION,
                "dry_run": True,
                "benchmark_commands_started": False,
                "commands_have_dry_run_flag": all("--dry-run" in command["argv"] for command in command_preview["commands"]),
            },
            "execution_go": False,
            "execution_started": False,
            "real_benchmarks_executed": False,
            "strict_plan_gate_closed": False,
            "full_remote_benchmark_executed": False,
            "github_query_performed": False,
            "github_write_performed": False,
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "active_apply_performed": False,
            "side_effect_boundaries": {
                "github_query_performed": False,
                "github_write_performed": False,
                "provider_or_model_spend_performed": False,
                "network_calls_performed": False,
                "active_apply_performed": False,
                "benchmark_process_started": False,
                "benchmark_output_written": False,
                "future_output_root_created": False,
                "worktree_materialization_performed": False,
                "cron_or_gateway_mutation_performed": False,
            },
            "not_claimed": [
                "current_active_phase1_phase2_strict_completion",
                "local_smoke_rerun_execution",
                "strict_plan_gate_closure",
                "full_remote_benchmark",
                "provider_api_spend",
                "github_query_or_write",
                "active_apply",
                "cron_or_gateway_mutation",
            ],
            "required_next_action": "current_baseline_phase1_phase2_local_smoke_rerun_go"
            if status == PREFLIGHT_READY_SEPARATE_GO_REQUIRED
            else "resolve_preflight_blockers_before_local_smoke_rerun_go",
            "artifacts": {
                "preflight_report": "current_baseline_revalidation_preflight.json",
                "preflight_markdown": "current_baseline_revalidation_preflight.md",
                "command_preview": "current_baseline_revalidation_command_preview.json",
            },
        }
    )
    reject_github_or_active_apply_flags(report)

    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "current_baseline_revalidation_preflight.json"
    markdown_path = out / "current_baseline_revalidation_preflight.md"
    command_preview_path = out / "current_baseline_revalidation_command_preview.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    command_preview_path.write_text(json.dumps(command_preview, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown_path.write_text(_render_markdown(report))
    return {
        "preflight_report_path": str(report_path),
        "preflight_markdown_path": str(markdown_path),
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


def _active_inventory(repo: Path) -> dict[str, Any]:
    head = _git_stdout(repo, "rev-parse", "HEAD")
    branch = _git_stdout(repo, "rev-parse", "--abbrev-ref", "HEAD")
    root = _git_stdout(repo, "rev-parse", "--show-toplevel")
    status = _run_git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0:
        raise ValueError(f"git status failed: {status.stderr.strip()}")
    dirty_entries = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "repo_root": root,
        "branch": branch,
        "head": head,
        "head_short": head[:9],
        "clean": not dirty_entries,
        "dirty_file_count": len(dirty_entries),
        "dirty_entries": dirty_entries,
        "files": [_file_inventory(repo, rel) for rel in ACTIVE_BASELINE_FILES],
    }


def _file_inventory(repo: Path, rel: str) -> dict[str, Any]:
    path = repo / rel
    exists = path.exists()
    is_file = exists and path.is_file()
    return {
        "relative_path": rel,
        "exists": exists,
        "is_file": is_file,
        "bytes": path.stat().st_size if is_file else None,
        "sha256": sha256(path.read_bytes()).hexdigest() if is_file else None,
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    if not isinstance(data, dict):
        raise ValueError(f"{label} JSON root must be an object: {path}")
    return data


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _closure_subject_commit(audit: Mapping[str, Any]) -> str:
    commit = _nested_get(audit, ("current_baseline_match", "closure_subject_commit"))
    return str(commit) if isinstance(commit, str) and commit else ""


def _strict_frontier_requires_revalidation(audit: Mapping[str, Any]) -> bool:
    return (
        audit.get("schema_version") == "hse-strict-frontier-audit-v1"
        and audit.get("status") == "CURRENT_BASELINE_REVALIDATION_REQUIRED"
        and _nested_get(audit, ("recorded_subject_frontier", "status")) == "PHASE_2_STRICT_COMPLETE"
        and _nested_get(audit, ("current_active_frontier", "status")) == "CURRENT_BASELINE_REVALIDATION_REQUIRED"
        and _nested_get(audit, ("current_baseline_match", "matches_closure_subject")) is False
        and bool(_closure_subject_commit(audit))
    )


def _suite_readiness(suites: Sequence[str]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for suite in suites:
        raw = suite_readiness(suite)
        if not isinstance(raw, Mapping):
            raise ValueError(f"suite_readiness must return object for {suite}")
        reports.append(dict(raw))
    return reports


def _blocked_by(
    *,
    audit: Mapping[str, Any],
    active: Mapping[str, Any],
    future_output_root: Path,
    suite_readiness_reports: Sequence[Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    if not _strict_frontier_requires_revalidation(audit):
        blockers.append("strict_frontier_not_current_baseline_revalidation_required")
    if active.get("clean") is not True:
        blockers.append("active_hermes_worktree_dirty")
    if future_output_root.exists():
        blockers.append("future_output_root_already_exists")
    if not suite_readiness_reports or any(report.get("ready") is not True for report in suite_readiness_reports):
        blockers.append("suite_assets_not_ready")
    return blockers


def _status(blocked_by: Sequence[str]) -> str:
    if not blocked_by:
        return PREFLIGHT_READY_SEPARATE_GO_REQUIRED
    if "active_hermes_worktree_dirty" in blocked_by:
        return BLOCKED_ACTIVE_BASELINE_DIRTY
    if "future_output_root_already_exists" in blocked_by:
        return BLOCKED_FUTURE_OUTPUT_ROOT_EXISTS
    if "strict_frontier_not_current_baseline_revalidation_required" in blocked_by:
        return BLOCKED_STRICT_FRONTIER_NOT_REVALIDATION_REQUIRED
    return BLOCKED_SUITE_ASSETS_NOT_READY


def _summary(status: str) -> str:
    if status == PREFLIGHT_READY_SEPARATE_GO_REQUIRED:
        return "Current active Hermes baseline is inventoried and local-only Phase 1/2 smoke rerun is recommended, but a separate GO is required before execution."
    if status == BLOCKED_ACTIVE_BASELINE_DIRTY:
        return "Current active Hermes worktree is dirty; revalidation smoke rerun preflight is blocked until the baseline is clean."
    if status == BLOCKED_FUTURE_OUTPUT_ROOT_EXISTS:
        return "Future benchmark output root already exists; choose a fresh output root before local smoke rerun."
    if status == BLOCKED_STRICT_FRONTIER_NOT_REVALIDATION_REQUIRED:
        return "Source strict frontier audit does not require current-baseline revalidation."
    return "Local suite assets are not ready for current-baseline revalidation."


def _command_preview(*, suites: Sequence[str], baseline_commit: str, current_commit: str, future_output_root: Path) -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    for suite in suites:
        output_file = SUITE_OUTPUT_FILES.get(suite, f"{_slug(suite)}.json")
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
                    baseline_commit,
                    "--current-commit",
                    current_commit,
                    "--output-json",
                    str(future_output_root / output_file),
                    "--dry-run",
                ],
                "started": False,
                "network_allowed": False,
                "provider_spend_allowed": False,
                "github_write_allowed": False,
            }
        )
    return {
        "schema_version": COMMAND_PREVIEW_SCHEMA_VERSION,
        "dry_run": True,
        "benchmark_commands_started": False,
        "commands": commands,
    }


def _nested_get(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validate_run_id(value: str) -> None:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(ch not in allowed for ch in value) or value in {".", ".."}:
        raise ValueError("future_run_id must be a safe single path segment")


def _require_non_empty(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _slug(value: str) -> str:
    return "-".join("".join(ch.lower() if ch.isalnum() else "-" for ch in value).split("-"))


def _render_markdown(report: Mapping[str, Any]) -> str:
    inventory_raw = report.get("current_baseline_inventory")
    inventory: Mapping[str, Any] = inventory_raw if isinstance(inventory_raw, Mapping) else {}
    future_raw = report.get("future_output_root_guard")
    future: Mapping[str, Any] = future_raw if isinstance(future_raw, Mapping) else {}
    rerun_raw = report.get("rerun_decision")
    rerun: Mapping[str, Any] = rerun_raw if isinstance(rerun_raw, Mapping) else {}
    return "\n".join(
        [
            "# Current-baseline Phase 1/2 Revalidation Preflight",
            "",
            f"Status: `{report.get('status')}`",
            "",
            "## Current Baseline",
            "",
            f"- head: `{inventory.get('head')}`",
            f"- clean: `{inventory.get('clean')}`",
            "",
            "## Rerun Decision",
            "",
            f"- rerun_recommended: `{rerun.get('rerun_recommended')}`",
            f"- rerun_approved_now: `{rerun.get('rerun_approved_now')}`",
            f"- separate_local_smoke_go_required: `{rerun.get('separate_local_smoke_go_required')}`",
            f"- execution_go: `{report.get('execution_go')}`",
            "",
            "## Future Output Root",
            "",
            f"- path: `{future.get('future_output_root')}`",
            f"- exists_now: `{future.get('future_output_root_exists_now')}`",
            f"- created_now: `{future.get('future_output_root_created_now')}`",
            "",
            "## Boundaries",
            "",
            "- No benchmark process started.",
            "- No benchmark output root created.",
            "- No GitHub query/write, provider/API spend, network call, active apply, cron, or gateway mutation performed.",
            "- Strict PLAN gate remains open for current active target until a later rerun and closure gate pass.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write current-baseline Phase 1/2 revalidation preflight artifacts.")
    parser.add_argument("--active-hermes-repo", required=True, type=Path)
    parser.add_argument("--strict-frontier-audit", required=True, type=Path)
    parser.add_argument("--hse-repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--future-run-id", required=True)
    args = parser.parse_args(argv)
    result = write_current_baseline_revalidation_preflight(
        active_hermes_repo=args.active_hermes_repo,
        strict_frontier_audit_path=args.strict_frontier_audit,
        hse_repo_root=args.hse_repo_root,
        output_dir=args.output_dir,
        generated_at=args.generated_at,
        future_run_id=args.future_run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""HSE strict frontier audit writer.

This report reconciles a closed Phase 1/2 benchmark gate against the current
PLAN.md and the active Hermes target checkout. It intentionally separates the
recorded-subject frontier from the current-active-target frontier so stale local
completion artifacts cannot be overclaimed after the active Hermes baseline has
moved.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags

STRICT_FRONTIER_AUDIT_SCHEMA_VERSION = "hse-strict-frontier-audit-v1"
STRICT_FRONTIER_GATE_ID = "SFA"
STRICT_FRONTIER_PHASE = "HSE Strict Frontier Audit"
STRICT_FRONTIER_TARGET = "current-plan-strict-completion-frontier"
PHASE_2_STRICT_COMPLETE = "PHASE_2_STRICT_COMPLETE"
PHASE_3_STRICT_COMPLETE = "PHASE_3_STRICT_COMPLETE"
CURRENT_BASELINE_REVALIDATION_REQUIRED = "CURRENT_BASELINE_REVALIDATION_REQUIRED"
RECORDED_SUBJECT_INCOMPLETE = "RECORDED_SUBJECT_INCOMPLETE"

_TOOL_DESCRIPTION_SUBJECT_FILES = {
    "override_module": "tools/tool_description_overrides.py",
    "model_tools_module": "model_tools.py",
    "registry_module": "tools/registry.py",
}


@dataclass(frozen=True)
class GitResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def write_strict_frontier_audit(
    *,
    active_hermes_repo: str | Path,
    benchmark_closure_path: str | Path,
    phase2_active_apply_path: str | Path,
    post_phase2_audit_path: str | Path,
    phase2_review_path: str | Path,
    phase3_plan_path: str | Path,
    phase3_readiness_path: str | Path,
    phase3_historical_path: str | Path,
    phase4_completion_path: str | Path,
    phase5_readiness_path: str | Path,
    phase5_formal_path: str | Path,
    plan_path: str | Path,
    output_dir: str | Path,
    generated_at: str,
    phase3_local_real_smoke_path: str | Path | None = None,
    phase3_gepa_execution_path: str | Path | None = None,
    phase3_noop_apply_closure_path: str | Path | None = None,
    phase3_post_noop_recheck_path: str | Path | None = None,
) -> dict[str, str]:
    """Write a strict frontier audit JSON/Markdown pair."""

    _require_non_empty("generated_at", generated_at)
    active_repo = Path(active_hermes_repo).expanduser().resolve()
    _validate_git_repo(active_repo)
    paths = {
        "benchmark_closure": Path(benchmark_closure_path).expanduser(),
        "phase2_active_apply": Path(phase2_active_apply_path).expanduser(),
        "post_phase2_audit": Path(post_phase2_audit_path).expanduser(),
        "phase2_review": Path(phase2_review_path).expanduser(),
        "phase3_plan": Path(phase3_plan_path).expanduser(),
        "phase3_readiness": Path(phase3_readiness_path).expanduser(),
        "phase3_historical": Path(phase3_historical_path).expanduser(),
        "phase4_completion": Path(phase4_completion_path).expanduser(),
        "phase5_readiness": Path(phase5_readiness_path).expanduser(),
        "phase5_formal": Path(phase5_formal_path).expanduser(),
        "plan": Path(plan_path).expanduser(),
    }
    optional_phase3_paths = {
        "phase3_local_real_smoke": phase3_local_real_smoke_path,
        "phase3_gepa_execution": phase3_gepa_execution_path,
        "phase3_noop_apply_closure": phase3_noop_apply_closure_path,
        "phase3_post_noop_recheck": phase3_post_noop_recheck_path,
    }
    for name, path in optional_phase3_paths.items():
        if path is not None:
            paths[name] = Path(path).expanduser()
    data = {name: _load_json_object(path, name) for name, path in paths.items() if name != "plan"}
    plan_text = paths["plan"].read_text()

    active = _active_repo_state(active_repo)
    current_match = _current_baseline_match(active_repo, active, data["benchmark_closure"])
    recorded_complete = _recorded_subject_phase2_complete(
        data["benchmark_closure"],
        data["phase2_active_apply"],
        data["post_phase2_audit"],
        data["phase2_review"],
        plan_text,
    )
    recorded_frontier = _recorded_frontier(recorded_complete)
    current_frontier = _current_frontier(recorded_complete, current_match)
    phases = _phase_table(data, recorded_complete, current_match, current_frontier)
    current_frontier = _align_current_frontier_with_phase_table(current_frontier, phases)

    report = base_decision_payload(
        gate_id=STRICT_FRONTIER_GATE_ID,
        phase=STRICT_FRONTIER_PHASE,
        target=STRICT_FRONTIER_TARGET,
        generated_at=generated_at,
    )
    report["schema_version"] = STRICT_FRONTIER_AUDIT_SCHEMA_VERSION
    report.update(
        {
            "status": current_frontier["status"],
            "summary": _summary(recorded_frontier, current_frontier, phases),
            "recorded_subject_frontier": recorded_frontier,
            "current_active_frontier": current_frontier,
            "current_baseline_match": current_match,
            "phases": phases,
            "plan_contract": _plan_contract(paths["plan"], plan_text),
            "active_hermes": active,
            "source_artifacts": _source_artifacts(paths),
            "github_query_performed": False,
            "github_write_performed": False,
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "active_apply_performed": False,
            "full_remote_benchmark_executed": False,
            "phase3_strict_completion_claimed": False,
            "overall_hse_project_completion_claimed": False,
            "strict_frontier_boundary_notes": [
                "Recorded-subject completion is not automatically current-active-target completion.",
                "A moved active Hermes baseline requires revalidation before Phase 1/2 strict-complete can be claimed for the current target.",
                "Historical/local/waiver completion reports for Phase 3+ are treated as evidence, not current strict completion, unless current PLAN gates and current baseline checks pass.",
                "Phase 3 integrated-chain acceptance is local audit evidence only; it does not approve active apply, publication, cron/gateway mutation, provider spend, or overall HSE completion.",
                "PHASE_3_STRICT_COMPLETE is an internal strict-frontier audit status, not an official Phase 3 completion claim",
            ],
            "not_claimed": [
                "overall_HSE_project_completion",
                "current_active_phase1_phase2_strict_completion_when_baseline_mismatches",
                "phase3_strict_completion",
                "phase4_strict_completion",
                "phase5_strict_completion",
                "full_remote_benchmark",
                "provider_api_spend",
                "github_query_or_write",
                "active_apply",
                "cron_or_gateway_mutation",
            ],
            "recommended_next_action": _recommended_next(current_frontier),
            "artifacts": {
                "report": "strict_frontier_audit.json",
                "markdown": "strict_frontier_audit.md",
            },
        }
    )
    reject_github_or_active_apply_flags(report)

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "strict_frontier_audit.json"
    markdown_path = out / "strict_frontier_audit.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown_path.write_text(_render_markdown(report))
    return {"report_path": str(report_path), "markdown_path": str(markdown_path)}


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


def _active_repo_state(repo: Path) -> dict[str, Any]:
    head = _git_stdout(repo, "rev-parse", "HEAD")
    branch = _git_stdout(repo, "rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    dirty_entries = [line for line in status.stdout.splitlines() if line.strip()] if status.returncode == 0 else []
    return {
        "repo_root": _git_stdout(repo, "rev-parse", "--show-toplevel"),
        "head": head,
        "head_short": head[:9],
        "branch": branch,
        "dirty_file_count": len(dirty_entries),
        "clean": len(dirty_entries) == 0,
    }


def _current_baseline_match(repo: Path, active: Mapping[str, Any], closure: Mapping[str, Any]) -> dict[str, Any]:
    subject = _current_subject(closure)
    subject_source = subject.get("hermes_source", {}) if isinstance(subject.get("hermes_source"), Mapping) else {}
    subject_commit = str(subject_source.get("commit_full") or subject_source.get("commit") or "")
    ancestor = _ancestor_state(repo, subject_commit)
    if closure.get("current_baseline_revalidation_closed") is True:
        return _current_baseline_revalidation_match(repo, active, closure, subject, subject_commit, ancestor)

    file_checks = _tool_description_file_checks(repo, subject)
    hashes_match = bool(file_checks) and all(check["hash_match"] is True for check in file_checks)
    blockers: list[str] = []
    if ancestor["is_ancestor_of_current_head"] is not True:
        blockers.append("current_hermes_head_not_closure_subject")
    if not hashes_match:
        blockers.append("active_tool_description_hash_mismatch")
    return {
        "matches_closure_subject": ancestor["is_ancestor_of_current_head"] is True and hashes_match,
        "current_baseline_revalidation_closure": False,
        "active_head": active.get("head"),
        "active_head_short": active.get("head_short"),
        "closure_subject_commit": subject_commit,
        "closure_subject_id": subject.get("subject_id"),
        "closure_subject_is_ancestor_of_active_head": ancestor["is_ancestor_of_current_head"],
        "closure_subject_ancestor_rc": ancestor["ancestor_check_rc"],
        "closure_subject_commit_available": ancestor["available"],
        "active_head_equals_revalidated_current_subject": None,
        "current_baseline_preflight_hash_verified": None,
        "current_baseline_preflight_current_commit_matches_subject": None,
        "current_baseline_preflight_inventory_head_matches_subject": None,
        "active_tool_description_hashes_match": hashes_match,
        "tool_description_file_checks": file_checks,
        "blockers": blockers,
    }


def _current_baseline_revalidation_match(
    repo: Path,
    active: Mapping[str, Any],
    closure: Mapping[str, Any],
    subject: Mapping[str, Any],
    subject_commit: str,
    ancestor: Mapping[str, Any],
) -> dict[str, Any]:
    preflight, preflight_meta = _current_baseline_preflight_from_closure(closure)
    file_checks = _current_baseline_inventory_file_checks(repo, preflight)
    hashes_match = bool(file_checks) and all(check["hash_match"] is True for check in file_checks)
    active_head_equals_subject = active.get("head") == subject_commit
    preflight_current_commit_matches_subject = bool(preflight) and preflight.get("current_commit_for_rerun") == subject_commit
    inventory = preflight.get("current_baseline_inventory", {}) if isinstance(preflight.get("current_baseline_inventory"), Mapping) else {}
    preflight_inventory_head_matches_subject = bool(inventory) and inventory.get("head") == subject_commit
    blockers: list[str] = []
    if not active_head_equals_subject:
        blockers.append("current_hermes_head_not_revalidated_current_subject")
    if preflight_meta.get("hash_verified") is not True:
        blockers.append("current_baseline_preflight_hash_not_verified")
    if not preflight_current_commit_matches_subject:
        blockers.append("current_baseline_preflight_current_commit_mismatch")
    if not preflight_inventory_head_matches_subject:
        blockers.append("current_baseline_preflight_inventory_head_mismatch")
    if not hashes_match:
        blockers.append("active_tool_description_hash_mismatch")
    blockers.extend(preflight_meta.get("blockers", []))
    matches = bool(
        active_head_equals_subject
        and preflight_meta.get("hash_verified") is True
        and preflight_current_commit_matches_subject
        and preflight_inventory_head_matches_subject
        and hashes_match
    )
    return {
        "matches_closure_subject": matches,
        "current_baseline_revalidation_closure": True,
        "active_head": active.get("head"),
        "active_head_short": active.get("head_short"),
        "closure_subject_commit": subject_commit,
        "closure_subject_id": subject.get("subject_id"),
        "closure_subject_is_ancestor_of_active_head": ancestor["is_ancestor_of_current_head"],
        "closure_subject_ancestor_rc": ancestor["ancestor_check_rc"],
        "closure_subject_commit_available": ancestor["available"],
        "active_head_equals_revalidated_current_subject": active_head_equals_subject,
        "current_baseline_preflight": preflight_meta,
        "current_baseline_preflight_hash_verified": preflight_meta.get("hash_verified") is True,
        "current_baseline_preflight_current_commit_matches_subject": preflight_current_commit_matches_subject,
        "current_baseline_preflight_inventory_head_matches_subject": preflight_inventory_head_matches_subject,
        "active_tool_description_hashes_match": hashes_match,
        "tool_description_file_checks": file_checks,
        "blockers": sorted(set(blockers)),
    }


def _current_baseline_preflight_from_closure(closure: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    meta: dict[str, Any] = {"available": False, "hash_verified": False, "blockers": []}
    source_artifacts = closure.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        meta["blockers"].append("current_baseline_preflight_source_missing")
        return {}, meta
    record = source_artifacts.get("current_baseline_preflight")
    if not isinstance(record, Mapping):
        meta["blockers"].append("current_baseline_preflight_source_missing")
        return {}, meta
    path_value = record.get("path")
    expected_sha = record.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        meta["blockers"].append("current_baseline_preflight_path_missing")
        return {}, meta
    path = Path(path_value).expanduser()
    meta.update({"path": str(path), "expected_sha256": expected_sha})
    if not path.exists():
        meta["blockers"].append("current_baseline_preflight_file_missing")
        return {}, meta
    actual_sha = _sha256_path(path)
    meta["actual_sha256"] = actual_sha
    meta["bytes"] = path.stat().st_size
    meta["hash_verified"] = actual_sha == expected_sha
    if meta["hash_verified"] is not True:
        meta["blockers"].append("current_baseline_preflight_hash_mismatch")
        return {}, meta
    data = _load_json_object(path, "current-baseline preflight")
    meta["available"] = True
    meta["schema_version"] = data.get("schema_version")
    meta["status"] = data.get("status")
    meta["current_commit_for_rerun"] = data.get("current_commit_for_rerun")
    inventory = data.get("current_baseline_inventory", {}) if isinstance(data.get("current_baseline_inventory"), Mapping) else {}
    meta["inventory_head"] = inventory.get("head")
    if data.get("schema_version") != "hse-current-baseline-revalidation-preflight-v1":
        meta["blockers"].append("current_baseline_preflight_schema_mismatch")
    if data.get("status") != "CURRENT_BASELINE_REVALIDATION_PREFLIGHT_READY_SEPARATE_GO_REQUIRED":
        meta["blockers"].append("current_baseline_preflight_not_ready")
    return data, meta


def _current_baseline_inventory_file_checks(repo: Path, preflight: Mapping[str, Any]) -> list[dict[str, Any]]:
    inventory = preflight.get("current_baseline_inventory", {}) if isinstance(preflight.get("current_baseline_inventory"), Mapping) else {}
    files = inventory.get("files")
    if not isinstance(files, list):
        return []
    checks: list[dict[str, Any]] = []
    for index, record in enumerate(files):
        if not isinstance(record, Mapping):
            checks.append({"key": f"inventory[{index}]", "relative_path": None, "exists": False, "expected_sha256": None, "actual_sha256": None, "hash_match": False})
            continue
        rel = record.get("relative_path")
        if not isinstance(rel, str) or not rel:
            checks.append({"key": f"inventory[{index}]", "relative_path": rel, "exists": False, "expected_sha256": record.get("sha256"), "actual_sha256": None, "hash_match": False})
            continue
        expected_exists = record.get("exists") is True
        expected_sha = record.get("sha256")
        path = repo / rel
        exists = path.exists()
        is_file = exists and path.is_file()
        actual = sha256(path.read_bytes()).hexdigest() if is_file else None
        hash_match = exists == expected_exists and ((not expected_exists and expected_sha is None) or (is_file and actual == expected_sha))
        checks.append(
            {
                "key": f"current_baseline_inventory[{index}]",
                "relative_path": rel,
                "exists": exists,
                "expected_exists": expected_exists,
                "expected_sha256": expected_sha,
                "actual_sha256": actual,
                "hash_match": hash_match,
            }
        )
    return checks


def _ancestor_state(repo: Path, commit: str) -> dict[str, Any]:
    if not commit:
        return {"available": False, "ancestor_check_rc": None, "is_ancestor_of_current_head": False}
    cat = _run_git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    if cat.returncode != 0:
        return {"available": False, "ancestor_check_rc": cat.returncode, "is_ancestor_of_current_head": False}
    ancestor = _run_git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
    return {
        "available": True,
        "ancestor_check_rc": ancestor.returncode,
        "is_ancestor_of_current_head": ancestor.returncode == 0,
    }


def _current_subject(closure: Mapping[str, Any]) -> Mapping[str, Any]:
    subjects = closure.get("benchmark_subjects")
    if not isinstance(subjects, Mapping):
        return {}
    current = subjects.get("current")
    return current if isinstance(current, Mapping) else {}


def _tool_description_file_checks(repo: Path, subject: Mapping[str, Any]) -> list[dict[str, Any]]:
    tool_descriptions = subject.get("tool_descriptions")
    if not isinstance(tool_descriptions, Mapping):
        return []
    checks: list[dict[str, Any]] = []
    for key, rel in _TOOL_DESCRIPTION_SUBJECT_FILES.items():
        record = tool_descriptions.get(key)
        if not isinstance(record, Mapping):
            checks.append({"key": key, "relative_path": rel, "exists": False, "expected_sha256": None, "actual_sha256": None, "hash_match": False})
            continue
        expected = record.get("sha256")
        path = repo / rel
        exists = path.exists()
        actual = sha256(path.read_bytes()).hexdigest() if exists and path.is_file() else None
        checks.append(
            {
                "key": key,
                "relative_path": rel,
                "exists": exists,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "hash_match": exists and actual == expected,
            }
        )
    return checks


def _recorded_subject_phase2_complete(
    closure: Mapping[str, Any],
    phase2_active: Mapping[str, Any],
    post_phase2: Mapping[str, Any],
    phase2_review: Mapping[str, Any],
    plan_text: str,
) -> dict[str, Any]:
    checks = {
        "plan_phase1_gate_present": "≥1 skill measurably improved" in plan_text and "no benchmark regression" in plan_text,
        "plan_phase2_gate_present": "Tool selection accuracy improved" in plan_text and "no benchmark regression" in plan_text,
        "benchmark_closure_passed": closure.get("status") == "STRICT_PLAN_BENCHMARK_GATE_CLOSED"
        and closure.get("strict_plan_gate_closed") is True
        and closure.get("benchmark_gate_passed") is True
        and closure.get("blocked_by") == [],
        "phase2_active_apply_passed": phase2_active.get("verdict") == "PASS_ACTIVE_SCHEMA_APPLY_LOCAL"
        and phase2_active.get("phase2d_gate_passed") is True
        and phase2_active.get("model_tools_readback_passed") is True
        and phase2_active.get("raw_registry_readback_passed") is True
        and phase2_active.get("semantic_loss_guard_passed") is True,
        "post_phase2_open_frontier_consumed": post_phase2.get("strict_verdict")
        == "LOCAL_P1_P2_ACTIVE_APPLIED__STRICT_BENCHMARK_GATE_OPEN",
        "phase2_review_complete": phase2_review.get("phase2e_closeout_complete") is True
        and phase2_review.get("remaining_phase2_closeout_items") == [],
        "no_github_write_preserved": closure.get("github_query_performed") is False
        and closure.get("github_write_performed") is False,
        "no_spend_or_active_apply_in_closure": closure.get("provider_or_model_spend_performed") is False
        and closure.get("network_calls_performed") is False
        and closure.get("active_apply_performed") is False,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {"complete": not blockers, "checks": checks, "blockers": blockers}


def _recorded_frontier(recorded_complete: Mapping[str, Any]) -> dict[str, Any]:
    if recorded_complete.get("complete") is True:
        return {
            "status": PHASE_2_STRICT_COMPLETE,
            "highest_strict_complete_phase": 2,
            "basis": "closed Phase 1/2 benchmark gate plus local active Phase 1/2 evidence on the recorded benchmark subject",
            "blockers": [],
        }
    return {
        "status": RECORDED_SUBJECT_INCOMPLETE,
        "highest_strict_complete_phase": 0,
        "basis": "recorded Phase 1/2 evidence is incomplete",
        "blockers": list(recorded_complete.get("blockers", [])),
    }


def _current_frontier(recorded_complete: Mapping[str, Any], current_match: Mapping[str, Any]) -> dict[str, Any]:
    if recorded_complete.get("complete") is True and current_match.get("matches_closure_subject") is True:
        return {
            "status": PHASE_2_STRICT_COMPLETE,
            "highest_strict_complete_phase": 2,
            "basis": "active Hermes HEAD and tool-description hashes match the closed benchmark-gate subject",
            "blockers": [],
        }
    blockers = list(recorded_complete.get("blockers", []))
    blockers.extend(current_match.get("blockers", []))
    if recorded_complete.get("complete") is True:
        blockers.append("current_baseline_revalidation_required_before_phase1_phase2_strict_claim")
    return {
        "status": CURRENT_BASELINE_REVALIDATION_REQUIRED,
        "highest_strict_complete_phase": 0,
        "basis": "current active Hermes baseline does not match the closed benchmark-gate subject",
        "blockers": sorted(set(blockers)),
    }


def _align_current_frontier_with_phase_table(current_frontier: Mapping[str, Any], phases: Mapping[str, Any]) -> dict[str, Any]:
    phase3 = phases.get("phase3", {}) if isinstance(phases.get("phase3"), Mapping) else {}
    if (
        current_frontier.get("status") == PHASE_2_STRICT_COMPLETE
        and phase3.get("strict_complete") is True
        and phase3.get("blockers") == []
    ):
        return {
            "status": PHASE_3_STRICT_COMPLETE,
            "highest_strict_complete_phase": 3,
            "basis": "active Hermes baseline matches the Phase 1/2 closure subject and Phase 3 integrated-chain evidence is strict-complete",
            "blockers": [],
            "internal_audit_status_only": True,
            "official_completion_claimed": False,
        }
    return dict(current_frontier)


def _phase_table(data: Mapping[str, Mapping[str, Any]], recorded_complete: Mapping[str, Any], current_match: Mapping[str, Any], current_frontier: Mapping[str, Any]) -> dict[str, Any]:
    current_phase2 = current_frontier.get("status") == PHASE_2_STRICT_COMPLETE
    phase1_status = "STRICT_COMPLETE_CURRENT_ACTIVE" if current_phase2 else "REVALIDATION_REQUIRED_CURRENT_BASELINE_MISMATCH"
    phase2_status = phase1_status
    phase3_integrated_chain = _phase3_integrated_chain(data)
    phase3_blockers = _phase3_blockers(data["phase3_plan"], data["phase3_readiness"], current_phase2, phase3_integrated_chain)
    phase3_strict = not phase3_blockers
    phase4_blockers = _phase4_blockers(data["phase4_completion"], phase3_strict)
    phase5_blockers = _phase5_blockers(data["phase5_readiness"], data["phase5_formal"])
    return {
        "phase1": {
            "strict_complete": current_phase2,
            "recorded_subject_complete": recorded_complete.get("complete") is True,
            "strict_status": phase1_status,
            "blockers": [] if current_phase2 else ["current_baseline_revalidation_required_before_phase1_strict_claim"],
        },
        "phase2": {
            "strict_complete": current_phase2,
            "recorded_subject_complete": recorded_complete.get("complete") is True,
            "strict_status": phase2_status,
            "blockers": [] if current_phase2 else ["current_baseline_revalidation_required_before_phase2_strict_claim"],
        },
        "phase3": {
            "strict_complete": phase3_strict,
            "strict_status": "STRICT_COMPLETE_CURRENT_ACTIVE" if phase3_strict else "NOT_STRICT_COMPLETE_PREPARATION_ONLY",
            "historical_claim_status": data["phase3_historical"].get("status"),
            "integrated_chain": phase3_integrated_chain,
            "blockers": phase3_blockers,
        },
        "phase4": {
            "strict_complete": not phase4_blockers,
            "strict_status": "STRICT_COMPLETE_CURRENT_ACTIVE" if not phase4_blockers else "NOT_STRICT_COMPLETE_BLOCKED_BY_PHASE3_OR_SCOPE",
            "historical_local_status": data["phase4_completion"].get("status"),
            "blockers": phase4_blockers,
        },
        "phase5": {
            "strict_complete": not phase5_blockers,
            "strict_status": "STRICT_COMPLETE_CURRENT_ACTIVE" if not phase5_blockers else "NOT_STRICT_COMPLETE_LOCAL_OR_WAIVED_ONLY",
            "historical_claim_status": data["phase5_formal"].get("status"),
            "current_readiness_status": data["phase5_readiness"].get("status"),
            "blockers": phase5_blockers,
        },
    }


def _phase3_blockers(
    phase3_plan: Mapping[str, Any],
    phase3_readiness: Mapping[str, Any],
    current_phase2_complete: bool,
    phase3_integrated_chain: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not current_phase2_complete:
        blockers.append("phase3_blocked_until_phase1_phase2_strict_complete_current")
    if phase3_integrated_chain.get("available") is True:
        blockers.extend(str(blocker) for blocker in phase3_integrated_chain.get("blockers", []) if blocker)
        return sorted(set(blockers))
    if phase3_plan.get("status") == "planned_not_executed" or phase3_plan.get("execution_started") is False:
        blockers.append("phase3_current_plan_status_planned_not_executed")
    if phase3_readiness.get("real_benchmarks_executed") is not True:
        blockers.append("phase3_real_benchmarks_not_executed")
    if phase3_readiness.get("active_system_prompt_apply_approved") is not True:
        blockers.append("phase3_active_apply_not_approved_current_readiness")
    ready_state_raw = phase3_readiness.get("ready_state")
    ready_state: Mapping[str, Any] = ready_state_raw if isinstance(ready_state_raw, Mapping) else {}
    if ready_state.get("real_benchmark_ready_now") is not True:
        blockers.append("phase3_real_benchmark_ready_now_false")
    return sorted(set(blockers))


def _phase3_integrated_chain(data: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = {
        "local_real_smoke": "phase3_local_real_smoke",
        "gepa_execution": "phase3_gepa_execution",
        "noop_apply_closure": "phase3_noop_apply_closure",
        "post_noop_recheck": "phase3_post_noop_recheck",
    }
    present = {label: data[key] for label, key in required.items() if key in data}
    if not present:
        return {
            "available": False,
            "complete": False,
            "mode": "legacy_phase3_inputs_only",
            "checks": {},
            "blockers": ["phase3_integrated_chain_not_provided"],
        }

    blockers = [f"phase3_integrated_chain_missing_{label}" for label, key in required.items() if key not in data]
    local_real_smoke = data.get("phase3_local_real_smoke", {})
    gepa_execution = data.get("phase3_gepa_execution", {})
    noop_apply_closure = data.get("phase3_noop_apply_closure", {})
    post_noop_recheck = data.get("phase3_post_noop_recheck", {})

    checks = {
        "local_real_smoke_passed": local_real_smoke.get("status") == "PHASE3_LOCAL_REAL_SMOKE_EXECUTION_PASSED_SEPARATE_GEPA_DSPY_APPROVAL_STILL_REQUIRED"
        and local_real_smoke.get("local_real_smoke_passed") is True,
        "bounded_local_gepa_dspy_execution_passed": gepa_execution.get("status")
        == "PHASE3_GEPA_DSPY_CANDIDATE_OPTIMIZATION_EXECUTION_PASSED_NO_ACTIVE_APPLY"
        and gepa_execution.get("boundary_ledger", {}).get("bounded_local_dspy_gepa_optimizer_executed") is True
        and gepa_execution.get("boundary_ledger", {}).get("candidate_optimization_command_executed") is True,
        "semantic_noop_apply_closure_satisfies_active_write_gate": noop_apply_closure.get("reconciliation_passed") is True
        and noop_apply_closure.get("closure_reconciliation", {}).get("apply_lane_closed") is True
        and noop_apply_closure.get("closure_reconciliation", {}).get("apply_lane_status") == "NO_ACTIVE_WRITE_REQUIRED"
        and noop_apply_closure.get("closure_reconciliation", {}).get("semantic_noop_confirmed") is True
        and noop_apply_closure.get("closure_reconciliation", {}).get("active_apply_needed") is False
        and noop_apply_closure.get("closure_reconciliation", {}).get("active_apply_recommended") is False
        and noop_apply_closure.get("closure_reconciliation", {}).get("active_apply_performed") is False,
        "post_noop_recheck_confirms_phase2_fail_closed": post_noop_recheck.get("recheck_passed") is True
        and post_noop_recheck.get("decision", {}).get("current_active_frontier_confirmed") == PHASE_2_STRICT_COMPLETE
        and post_noop_recheck.get("decision", {}).get("phase3_strict_complete") is False
        and post_noop_recheck.get("decision", {}).get("phase3_strict_completion_ready_to_claim") is False,
    }
    blocker_by_check = {
        "local_real_smoke_passed": "phase3_integrated_chain_local_real_smoke_not_passed",
        "bounded_local_gepa_dspy_execution_passed": "phase3_integrated_chain_gepa_execution_not_passed",
        "semantic_noop_apply_closure_satisfies_active_write_gate": "phase3_integrated_chain_noop_apply_closure_not_passed",
        "post_noop_recheck_confirms_phase2_fail_closed": "phase3_integrated_chain_post_noop_recheck_not_passed",
    }
    blockers.extend(blocker_by_check[name] for name, passed in checks.items() if passed is not True)
    for label, artifact in present.items():
        blockers.extend(_phase3_forbidden_boundary_blockers(label, artifact))

    return {
        "available": True,
        "complete": not blockers,
        "mode": "phase3_integrated_artifact_chain",
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "source_statuses": {label: artifact.get("status") for label, artifact in present.items()},
    }


def _phase3_forbidden_boundary_blockers(label: str, artifact: Mapping[str, Any]) -> list[str]:
    forbidden_true_keys = {
        "github_query_performed",
        "github_write_performed",
        "provider_or_model_spend_performed",
        "network_calls_performed",
        "external_llm_calls_performed",
        "active_apply_performed",
        "active_runtime_mutation_performed",
        "cron_or_gateway_mutation_performed",
        "deploy_or_publication_performed",
        "phase3_strict_completion_claimed",
        "overall_hse_project_completion_claimed",
    }
    blockers: list[str] = []

    def collect(mapping: Mapping[str, Any], prefix: str) -> None:
        for key in forbidden_true_keys:
            if mapping.get(key) is True:
                blockers.append(f"phase3_integrated_chain_forbidden_boundary_{label}.{prefix}{key}")

    collect(artifact, "")
    for section in ("boundary_ledger", "closure_reconciliation", "decision", "github", "safety_invariants"):
        value = artifact.get(section)
        if isinstance(value, Mapping):
            collect(value, f"{section}.")
    return sorted(set(blockers))


def _phase4_blockers(phase4: Mapping[str, Any], phase3_strict_complete: bool) -> list[str]:
    blockers: list[str] = []
    if not phase3_strict_complete:
        blockers.append("phase4_blocked_until_phase3_strict_complete_current")
    safety_raw = phase4.get("safety_boundaries")
    safety: Mapping[str, Any] = safety_raw if isinstance(safety_raw, Mapping) else {}
    if safety.get("darwinian_cli_invoked") is not True:
        blockers.append("darwinian_evolver_cli_not_invoked_for_current_strict_gate")
    if phase4.get("status") != "completed_current_strict_plan_verified":
        blockers.append("phase4_evidence_local_or_scaffold_not_current_strict_plan_verified")
    return sorted(set(blockers))


def _phase5_blockers(phase5_readiness: Mapping[str, Any], phase5_formal: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if phase5_readiness.get("continuous_loop_enabled") is not True:
        blockers.append("production_continuous_loop_not_enabled")
    if phase5_readiness.get("cron_jobs_created") is not True:
        blockers.append("cron_jobs_not_created")
    ready_state_raw = phase5_readiness.get("ready_state")
    ready_state: Mapping[str, Any] = ready_state_raw if isinstance(ready_state_raw, Mapping) else {}
    if ready_state.get("phase5_unattended_loop_ready_now") is not True:
        blockers.append("phase5_unattended_loop_ready_now_false")
    if phase5_formal.get("status") == "FORMAL_PHASE5_COMPLETE_LOCAL_WITH_EXPLICIT_WAIVER":
        blockers.append("historical_local_waiver_not_current_strict_plan_completion")
    return sorted(set(blockers))


def _plan_contract(plan_path: Path, plan_text: str) -> dict[str, Any]:
    return {
        "path": str(plan_path),
        "sha256": _sha256_path(plan_path),
        "phase_gates_detected": {
            "phase1": "≥1 skill measurably improved" in plan_text and "no benchmark regression" in plan_text,
            "phase2": "Tool selection accuracy improved" in plan_text and "no benchmark regression" in plan_text,
            "phase3": "Behavioral tests pass" in plan_text and "benchmarks hold or improve" in plan_text,
            "phase4": "Bugs fixed" in plan_text and "tests pass" in plan_text and "benchmarks hold" in plan_text,
            "phase5": "Automated pipeline runs unattended" in plan_text,
        },
    }


def _source_artifacts(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        artifacts[name] = {"path": str(path), "sha256": _sha256_path(path), "bytes": path.stat().st_size}
    return artifacts


def _summary(recorded_frontier: Mapping[str, Any], current_frontier: Mapping[str, Any], phases: Mapping[str, Any]) -> str:
    if current_frontier.get("status") == PHASE_3_STRICT_COMPLETE:
        return "Current-active strict frontier is internally aligned through Phase 3 from the integrated artifact chain; official Phase 3 completion claim remains separate."
    phase3 = phases.get("phase3", {}) if isinstance(phases.get("phase3"), Mapping) else {}
    if phase3.get("strict_complete") is True:
        return "Phase 2 is strict-complete for the current active Hermes target and the Phase 3 integrated artifact chain passes local strict-frontier audit checks; final Phase 3 completion claim remains separate."
    if current_frontier.get("status") == PHASE_2_STRICT_COMPLETE:
        return "Phase 2 is strict-complete for both recorded subject and current active Hermes target; Phase 3+ remain blocked."
    if recorded_frontier.get("status") == PHASE_2_STRICT_COMPLETE:
        return "Recorded-subject Phase 2 strict completion is closed, but current active Hermes baseline requires revalidation before Phase 1/2 strict completion can be claimed."
    return "Strict frontier is blocked before Phase 1/2 because recorded evidence is incomplete."


def _recommended_next(current_frontier: Mapping[str, Any]) -> str:
    if current_frontier.get("status") == PHASE_3_STRICT_COMPLETE:
        return "phase3_internal_frontier_alignment_closure_review_go_no_github_write_no_active_apply_no_deploy_no_official_claim"
    if current_frontier.get("status") == PHASE_2_STRICT_COMPLETE:
        return "phase3_strict_execution_preflight_go_no_remote_no_provider_no_github_write"
    return "current_baseline_revalidation_required_before_phase1_phase2_strict_claim: refresh active Hermes baseline inventory, rerun/readiness-check Phase 1/2 local benchmark evidence against current HEAD, and keep GitHub/remote/provider expansion blocked."


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    if not isinstance(data, dict):
        raise ValueError(f"{label} JSON root must be an object: {path}")
    return data


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _require_non_empty(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _render_markdown(report: Mapping[str, Any]) -> str:
    phases = report.get("phases", {}) if isinstance(report.get("phases"), Mapping) else {}
    phase_lines = []
    for key in ("phase1", "phase2", "phase3", "phase4", "phase5"):
        phase = phases.get(key, {}) if isinstance(phases.get(key), Mapping) else {}
        blockers = phase.get("blockers", [])
        blocker_text = ", ".join(blockers) if isinstance(blockers, list) and blockers else "none"
        phase_lines.append(
            f"- {key}: strict_complete={str(phase.get('strict_complete')).lower()} status=`{phase.get('strict_status')}` blockers={blocker_text}"
        )
    current = report.get("current_active_frontier", {}) if isinstance(report.get("current_active_frontier"), Mapping) else {}
    recorded = report.get("recorded_subject_frontier", {}) if isinstance(report.get("recorded_subject_frontier"), Mapping) else {}
    match = report.get("current_baseline_match", {}) if isinstance(report.get("current_baseline_match"), Mapping) else {}
    return "\n".join(
        [
            "# HSE Strict Frontier Audit",
            "",
            f"Status: `{report.get('status')}`",
            "",
            "## Frontier",
            "",
            f"- recorded_subject_frontier=`{recorded.get('status')}` phase={recorded.get('highest_strict_complete_phase')}",
            f"- current_active_frontier=`{current.get('status')}` phase={current.get('highest_strict_complete_phase')}",
            f"- current baseline matches closure subject: `{match.get('matches_closure_subject')}`",
            "",
            "## Phase Table",
            "",
            *phase_lines,
            "",
            "## Boundaries",
            "",
            "- No GitHub query/write performed.",
            "- No provider/API/network spend performed.",
            "- No active apply, cron, gateway restart, deploy, or remote benchmark expansion performed.",
            "- Overall HSE project completion is not claimed.",
            "",
            "## Recommended Next Action",
            "",
            str(report.get("recommended_next_action")),
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an HSE strict frontier audit report.")
    parser.add_argument("--active-hermes-repo", required=True, type=Path)
    parser.add_argument("--benchmark-closure", required=True, type=Path)
    parser.add_argument("--phase2-active-apply", required=True, type=Path)
    parser.add_argument("--post-phase2-audit", required=True, type=Path)
    parser.add_argument("--phase2-review", required=True, type=Path)
    parser.add_argument("--phase3-plan", required=True, type=Path)
    parser.add_argument("--phase3-readiness", required=True, type=Path)
    parser.add_argument("--phase3-historical", required=True, type=Path)
    parser.add_argument("--phase3-local-real-smoke", type=Path)
    parser.add_argument("--phase3-gepa-execution", type=Path)
    parser.add_argument("--phase3-noop-apply-closure", type=Path)
    parser.add_argument("--phase3-post-noop-recheck", type=Path)
    parser.add_argument("--phase4-completion", required=True, type=Path)
    parser.add_argument("--phase5-readiness", required=True, type=Path)
    parser.add_argument("--phase5-formal", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args(argv)
    result = write_strict_frontier_audit(
        active_hermes_repo=args.active_hermes_repo,
        benchmark_closure_path=args.benchmark_closure,
        phase2_active_apply_path=args.phase2_active_apply,
        post_phase2_audit_path=args.post_phase2_audit,
        phase2_review_path=args.phase2_review,
        phase3_plan_path=args.phase3_plan,
        phase3_readiness_path=args.phase3_readiness,
        phase3_historical_path=args.phase3_historical,
        phase3_local_real_smoke_path=args.phase3_local_real_smoke,
        phase3_gepa_execution_path=args.phase3_gepa_execution,
        phase3_noop_apply_closure_path=args.phase3_noop_apply_closure,
        phase3_post_noop_recheck_path=args.phase3_post_noop_recheck,
        phase4_completion_path=args.phase4_completion,
        phase5_readiness_path=args.phase5_readiness,
        phase5_formal_path=args.phase5_formal,
        plan_path=args.plan,
        output_dir=args.output_dir,
        generated_at=args.generated_at,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

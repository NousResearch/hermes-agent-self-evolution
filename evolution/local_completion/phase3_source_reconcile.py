"""LC4 Phase 3 active-source reconcile packet writer.

This module inspects the active Hermes checkout and writes a candidate-only
source reconcile packet. It never edits active Hermes source, active skills,
GitHub state, cron, credentials, or gateway/runtime processes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags

LC4_GATE_ID = "LC4"
LC4_PHASE3_SOURCE_RECONCILE_PHASE = "Phase 3: Active Source Reconcile Packet"
LC4_TARGET = "phase3-prompt-source-reconcile"
HISTORICAL_APPLY_COMMIT = "65a7925aa"
LIVE_ON_CURRENT_HEAD = "LIVE_ON_CURRENT_HEAD"
STALE_NOT_ANCESTOR_OF_CURRENT_HEAD = "STALE_NOT_ANCESTOR_OF_CURRENT_HEAD"

TARGET_SOURCE_FILES = (
    "agent/prompt_builder.py",
    "tests/agent/test_prompt_builder.py",
)
PHASE3_SUPPORT_MODULES = (
    "evolution/prompts/phase3_candidate_scaffold.py",
    "evolution/prompts/phase3_preflight_gate.py",
    "evolution/prompts/phase3_gepa_optimizer.py",
)
TESTS_REQUIRED_BEFORE_ACTIVE_SOURCE_PATCH = (
    "python -m compileall -q agent/prompt_builder.py tests/agent/test_prompt_builder.py",
    "python -m pytest tests/agent/test_prompt_builder.py -q -o 'addopts='",
    "git diff --check -- agent/prompt_builder.py tests/agent/test_prompt_builder.py",
    "source-clause smoke import for reconciled prompt guidance markers",
)
PROMPT_CACHE_SAFETY_CHECKLIST = (
    "do not mutate active prompt/source in this LC4 packet",
    "do not cherry-pick historical commit 65a7925aa directly against a dirty/current HEAD",
    "generate any future source patch as a semantic path-limited dry-run first",
    "preserve unrelated dirty files outside the Phase 3 source allowlist",
    "treat source patch application and runtime pickup/restart as separate later approvals",
    "verify prompt-builder tests and source-clause smoke before any active source patch",
)


@dataclass(frozen=True)
class GitResult:
    """Small subprocess result wrapper used to keep git inspection explicit."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def write_phase3_source_reconcile_packet(
    *,
    active_hermes_repo: str | Path,
    output_dir: str | Path,
    generated_at: str,
    historical_apply_commit: str = HISTORICAL_APPLY_COMMIT,
) -> dict[str, str]:
    """Write an LC4 candidate-only packet for current active Hermes source state."""

    repo = Path(active_hermes_repo).expanduser().resolve()
    _validate_git_repo(repo)
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    head = _git_stdout(repo, "rev-parse", "HEAD")
    branch = _git_stdout(repo, "rev-parse", "--abbrev-ref", "HEAD")
    root = _git_stdout(repo, "rev-parse", "--show-toplevel")
    dirty_entries = _dirty_entries(repo)
    dirty_ledger = _dirty_conflict_ledger(dirty_entries)
    target_files = _target_file_metadata(repo, dirty_entries)
    historical = _historical_commit_state(repo, historical_apply_commit, head)
    phase3_status = LIVE_ON_CURRENT_HEAD if historical["is_ancestor_of_current_head"] else STALE_NOT_ANCESTOR_OF_CURRENT_HEAD

    packet = base_decision_payload(
        gate_id=LC4_GATE_ID,
        phase=LC4_PHASE3_SOURCE_RECONCILE_PHASE,
        target=LC4_TARGET,
        generated_at=generated_at,
    )
    packet.update(
        {
            "status": "PASS_PACKET_READY",
            "summary": _summary(phase3_status, dirty_ledger),
            "historical_apply_commit": historical_apply_commit,
            "phase3_active_source_status": phase3_status,
            "active_hermes": {
                "repo_root": root,
                "head": head,
                "branch": branch,
                "dirty_file_count": len(dirty_entries),
                "target_dirty_file_count": sum(1 for item in dirty_ledger if item["scope"] == "phase3_source_allowlist"),
                "unrelated_dirty_file_count": sum(1 for item in dirty_ledger if item["scope"] == "preserve_unrelated_dirty"),
            },
            "historical_commit": historical,
            "current_prompt_source_baseline": {
                "target_files": target_files,
                "support_modules_available_for_later_artifact_generation_only": list(PHASE3_SUPPORT_MODULES),
            },
            "dirty_file_conflict_ledger": dirty_ledger,
            "candidate_source_patch_plan": {
                "patch_generated": False,
                "source_mutation_performed": False,
                "active_prompt_modified_by_this_packet": False,
                "strategy": "semantic_path_limited_dry_run_before_any_active_apply",
                "do_not_cherry_pick": historical_apply_commit,
                "allowlist": list(TARGET_SOURCE_FILES),
                "denylist": [
                    "active skills",
                    "tool schemas outside explicit Phase 3 source allowlist",
                    "GitHub/PR write paths",
                    "cron/control-plane mutation",
                    "gateway restart/reload",
                    "deployment/merge/publication",
                ],
                "candidate_fixture_or_patch_plan": [
                    "derive semantic prompt guidance clauses from the historical Phase 3 commit",
                    "compare them against current agent/prompt_builder.py rather than applying commit hunks",
                    "emit forward and rollback patches in a later dry-run evidence directory only",
                    "run git apply --check for forward and rollback patches before WRITE-ON review",
                    "keep unrelated dirty files out of the reconcile patch scope",
                ],
                "next_packet": (
                    "HSE Phase3 Active Source Reconcile Dry-Run GO -- generate a path-limited candidate "
                    "patch for agent/prompt_builder.py and tests/agent/test_prompt_builder.py only; "
                    "write evidence/forward/rollback patches, run git apply --check, keep no active apply, "
                    "no GitHub, no restart/reload."
                ),
            },
            "prompt_cache_safety_checklist": list(PROMPT_CACHE_SAFETY_CHECKLIST),
            "tests_required_before_any_active_source_patch": list(TESTS_REQUIRED_BEFORE_ACTIVE_SOURCE_PATCH),
            "active_source_reconcile_boundary": {
                "candidate_only_packet": True,
                "apply_ready_reason": "active source patch requires a separate dry-run packet and explicit WRITE-ON approval",
                "active_source_patch_applied": False,
                "active_prompt_modified": False,
                "github_queried_or_written": False,
                "restart_reload_performed": False,
            },
            "artifacts": {
                "source_reconcile_packet": "phase3_source_reconcile_packet.json",
                "source_reconcile_markdown": "phase3_source_reconcile_packet.md",
            },
        }
    )
    reject_github_or_active_apply_flags(packet)

    packet_path = out / "phase3_source_reconcile_packet.json"
    markdown_path = out / "phase3_source_reconcile_packet.md"
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_markdown(packet))
    return {"packet_path": str(packet_path), "markdown_path": str(markdown_path)}


def _validate_git_repo(repo: Path) -> None:
    if not repo.exists():
        raise FileNotFoundError(f"active Hermes repo not found: {repo}")
    result = _run_git(repo, "rev-parse", "--is-inside-work-tree")
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise ValueError(f"not a git worktree: {repo}")


def _run_git(repo: Path, *args: str) -> GitResult:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    return GitResult(tuple(args), completed.returncode, completed.stdout, completed.stderr)


def _git_stdout(repo: Path, *args: str) -> str:
    result = _run_git(repo, *args)
    if result.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _dirty_entries(repo: Path) -> list[dict[str, str]]:
    result = _run_git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if result.returncode != 0:
        raise ValueError(f"git status failed: {result.stderr.strip()}")
    entries: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        status = line[:2]
        raw_path = line[3:] if len(line) > 3 else ""
        path = raw_path.split(" -> ")[-1]
        entries.append({"status": status, "path": path, "raw": raw_path})
    return entries


def _dirty_conflict_ledger(entries: list[dict[str, str]]) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    target_set = set(TARGET_SOURCE_FILES)
    for entry in entries:
        path = entry["path"]
        in_scope = path in target_set
        ledger.append(
            {
                "path": path,
                "status": entry["status"],
                "scope": "phase3_source_allowlist" if in_scope else "preserve_unrelated_dirty",
                "phase3_conflict_risk": "target_file_dirty_requires_semantic_reconcile" if in_scope else "out_of_scope_preserve",
                "action": "inspect_semantically_before_future_patch" if in_scope else "do_not_touch_in_lc4",
            }
        )
    return ledger


def _target_file_metadata(repo: Path, dirty_entries: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    dirty_by_path = {entry["path"]: entry["status"] for entry in dirty_entries}
    metadata: dict[str, dict[str, Any]] = {}
    for rel_path in TARGET_SOURCE_FILES:
        path = repo / rel_path
        exists = path.exists()
        item: dict[str, Any] = {
            "exists": exists,
            "path": rel_path,
            "dirty": rel_path in dirty_by_path,
            "dirty_status": dirty_by_path.get(rel_path),
        }
        if exists and path.is_file():
            raw = path.read_bytes()
            item.update({"sha256": sha256(raw).hexdigest(), "bytes": len(raw), "line_count": len(raw.splitlines())})
        metadata[rel_path] = item
    return metadata


def _historical_commit_state(repo: Path, commit: str, head: str) -> dict[str, Any]:
    cat_file = _run_git(repo, "cat-file", "-t", commit)
    available = cat_file.returncode == 0 and cat_file.stdout.strip() == "commit"
    full_commit = None
    subject = None
    commit_date = None
    touched_files: list[str] = []
    merge_base = None
    ancestor_rc: int | None = None
    is_ancestor = False

    if available:
        full_commit = _git_stdout(repo, "rev-parse", commit)
        show = _run_git(repo, "show", "-s", "--format=%ci%n%s", full_commit)
        if show.returncode == 0:
            lines = show.stdout.splitlines()
            commit_date = lines[0] if lines else None
            subject = lines[1] if len(lines) > 1 else None
        files = _run_git(repo, "show", "--name-only", "--format=", full_commit)
        if files.returncode == 0:
            touched_files = [line.strip() for line in files.stdout.splitlines() if line.strip()]
        mb = _run_git(repo, "merge-base", full_commit, head)
        if mb.returncode == 0:
            merge_base = mb.stdout.strip()
        ancestor = _run_git(repo, "merge-base", "--is-ancestor", full_commit, head)
        ancestor_rc = ancestor.returncode
        is_ancestor = ancestor.returncode == 0
    else:
        ancestor_rc = cat_file.returncode

    return {
        "requested": commit,
        "available": available,
        "resolved": full_commit,
        "subject": subject,
        "date": commit_date,
        "touched_files": touched_files,
        "merge_base_with_current_head": merge_base,
        "ancestor_check_rc": ancestor_rc,
        "is_ancestor_of_current_head": is_ancestor,
    }


def _summary(phase3_status: str, dirty_ledger: list[dict[str, Any]]) -> str:
    target_dirty = sum(1 for item in dirty_ledger if item["scope"] == "phase3_source_allowlist")
    unrelated_dirty = sum(1 for item in dirty_ledger if item["scope"] == "preserve_unrelated_dirty")
    return (
        f"LC4 Phase 3 source reconcile packet written as candidate-only; historical apply status is "
        f"{phase3_status}; target dirty files={target_dirty}, unrelated dirty files={unrelated_dirty}; "
        "apply_ready remains false."
    )


def _render_markdown(packet: dict[str, Any]) -> str:
    active = packet["active_hermes"]
    historical = packet["historical_commit"]
    boundary = packet["active_source_reconcile_boundary"]
    ledger = packet["dirty_file_conflict_ledger"]
    return "\n".join(
        [
            "# LC4 Phase 3 Active Source Reconcile Packet",
            "",
            f"Status: `{packet['status']}`",
            "",
            "## Boundary",
            "",
            "- candidate_only=true",
            "- apply_ready=false",
            "- Active source patch applied: false",
            "- Active prompt modified by this packet: false",
            "- GitHub/PR work: deferred_not_queried_or_written",
            "- Gateway restart/reload: false",
            "",
            "## Active Hermes Baseline",
            "",
            f"- HEAD: `{active['head']}`",
            f"- Branch: `{active['branch']}`",
            f"- Dirty files: `{active['dirty_file_count']}`",
            f"- Phase 3 target dirty files: `{active['target_dirty_file_count']}`",
            f"- Unrelated dirty files preserved: `{active['unrelated_dirty_file_count']}`",
            "",
            "## Historical Phase 3 Commit",
            "",
            f"- requested: `{packet['historical_apply_commit']}`",
            f"- resolved: `{historical.get('resolved')}`",
            f"- ancestor_check_rc: `{historical.get('ancestor_check_rc')}`",
            f"- phase3_active_source_status: `{packet['phase3_active_source_status']}`",
            f"- subject: `{historical.get('subject')}`",
            "",
            "## Dirty File Conflict Ledger",
            "",
            *[
                f"- `{item['status']}` `{item['path']}` — {item['scope']} / {item['action']}"
                for item in ledger
            ],
            "",
            "## Future Candidate Patch Plan",
            "",
            "- Do not cherry-pick the historical commit directly.",
            "- Generate a semantic, path-limited dry-run patch first.",
            "- Allowlist: `agent/prompt_builder.py`, `tests/agent/test_prompt_builder.py`.",
            "- Preserve every unrelated dirty file out of scope.",
            "- Require separate WRITE-ON approval before any active source mutation.",
            "",
            "## Prompt Cache Safety Checklist",
            "",
            *[f"- {item}" for item in packet["prompt_cache_safety_checklist"]],
            "",
            "## Required Tests Before Any Active Source Patch",
            "",
            *[f"- `{item}`" for item in packet["tests_required_before_any_active_source_patch"]],
            "",
            "## Safety Invariants",
            "",
            f"- active_prompt_modified: `{str(boundary['active_prompt_modified']).lower()}`",
            f"- active_source_patch_applied: `{str(boundary['active_source_patch_applied']).lower()}`",
            f"- github_queried_or_written: `{str(boundary['github_queried_or_written']).lower()}`",
            f"- restart_reload_performed: `{str(boundary['restart_reload_performed']).lower()}`",
            "",
            "This packet does not authorize active source apply, active skill mutation, GitHub/PR work, restart, reload, merge, deployment, or publication.",
            "",
        ]
    )


def _now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an LC4 Phase 3 active-source reconcile packet.")
    parser.add_argument("--active-hermes-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generated-at", default=_now_iso())
    parser.add_argument("--historical-apply-commit", default=HISTORICAL_APPLY_COMMIT)
    args = parser.parse_args(argv)

    result = write_phase3_source_reconcile_packet(
        active_hermes_repo=args.active_hermes_repo,
        output_dir=args.output_dir,
        generated_at=args.generated_at,
        historical_apply_commit=args.historical_apply_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by operator smoke runs
    raise SystemExit(main())

"""Safe promotion helpers for gated candidates.

Promotion is intentionally local and human-gated. These helpers can dry-run,
write a candidate to a local branch, and draft PR text. They never push upstream
and never merge.
"""

from __future__ import annotations

import difflib
import subprocess
from pathlib import Path
from typing import Any

from evolution.db.store import EvolutionStore


def apply_gated_candidate(
    store: EvolutionStore,
    root: str | Path,
    run_id: str,
    *,
    branch: str | None = None,
    dry_run: bool = True,
    commit: bool = False,
    message: str | None = None,
    allow_hold: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Apply the latest gated evolved candidate locally, or describe the apply.

    Default is dry-run. Non-dry-run may create/switch a local branch and write the
    target file. It never pushes. Optional commit is local only.
    """
    _ = Path(root)  # kept for API symmetry and future artifact writes
    run, target, repo, gate, baseline, evolved = _promotion_context(store, run_id, allow_hold=allow_hold)
    repo_path = Path(repo["local_path"])
    target_file = repo_path / target["file_path"]
    evolved_text = _artifact_text(store, evolved["artifact_id"])
    current_text = target_file.read_text() if target_file.exists() else ""
    diff_text = _unified_diff(current_text, evolved_text, target["file_path"])

    if not dry_run:
        if not target_file.exists():
            raise FileNotFoundError(f"Target file not found: {target_file}")
        if not allow_dirty and _is_git_repo(repo_path) and _git_dirty(repo_path):
            raise ValueError(f"Repository has uncommitted changes: {repo_path}. Use allow_dirty=True to override.")
        if branch:
            _git(repo_path, ["checkout", "-B", branch])
        target_file.write_text(evolved_text)
        committed = False
        if commit:
            _git(repo_path, ["add", target["file_path"]])
            _git(repo_path, ["commit", "-m", message or _default_commit_message(target, run_id)])
            committed = True
        store.add_run_event(
            run_id,
            "promotion",
            "candidate applied locally",
            {"target_file": str(target_file), "branch": branch, "committed": committed, "pushed": False},
        )
    else:
        committed = False
        store.add_run_event(
            run_id,
            "promotion",
            "candidate apply dry-run",
            {"target_file": str(target_file), "branch": branch, "pushed": False},
        )

    return {
        "run_id": run_id,
        "mode": "dry-run" if dry_run else "apply",
        "target": f"{target['target_type']}:{target['name']}",
        "target_file": str(target_file),
        "repository": repo["local_path"],
        "branch": branch,
        "gate_decision": gate["decision"],
        "candidate_id": evolved["id"],
        "baseline_candidate_id": baseline["id"],
        "mutated": not dry_run,
        "committed": committed,
        "pushed": False,
        "diff": diff_text,
    }


def draft_pr_text(
    store: EvolutionStore,
    root: str | Path,
    run_id: str,
    *,
    branch: str | None = None,
    allow_hold: bool = False,
) -> dict[str, Any]:
    """Draft PR title/body for a gated candidate without mutating the repo."""
    _ = Path(root)
    _run, target, _repo, gate, baseline, evolved = _promotion_context(store, run_id, allow_hold=allow_hold)
    metrics = gate["metrics_json"]
    reasons = gate["reasons_json"]
    title = f"Evolve {target['target_type']}:{target['name']} via {run_id}"
    body = f"""## Summary
Human review required for evolved `{target['target_type']}:{target['name']}`.

## Safety contract
- Gate decision: `{gate['decision']}`
- Reasons: {', '.join(reasons) if reasons else 'none'}
- Auto-push: false
- Auto-merge: false

## Evidence
- Run: `{run_id}`
- Baseline candidate: `{baseline['id']}`
- Evolved candidate: `{evolved['id']}`
- Metric: `{metrics.get('metric_name')}`
- Holdout improvement: `{metrics.get('holdout_improvement')}`

## Required reviewer commands
```bash
hermes-evolve run gate {run_id}
hermes-evolve run export {run_id}
hermes-evolve run apply {run_id} --branch {branch or f'evolve/{target["name"]}-{run_id[:8]}'} --apply --commit
```

Review the generated diff before merge. If this looks like optimization theater, reject it. The machine has feelings; ignore them.
"""
    return {"title": title, "body": body, "branch": branch}


def _promotion_context(store: EvolutionStore, run_id: str, *, allow_hold: bool) -> tuple[dict, dict, dict, dict, dict, dict]:
    run = _require(store.get_run(run_id), f"Run not found: {run_id}")
    if run["status"] != "completed":
        raise ValueError(f"Run {run_id} is not completed; status={run['status']}")
    target = _require(store.get_target(run["target_id"]), f"Target not found: {run['target_id']}")
    repo = _require(store.get_repository_by_id(target["repository_id"]), f"Repository not found: {target['repository_id']}")
    gates = store.list_gate_results(run_id)
    if not gates:
        raise ValueError(f"Run {run_id} has no gate result. Run `hermes-evolve run gate {run_id}` first.")
    gate = gates[0]
    if gate["decision"] != "pass" and not allow_hold:
        raise ValueError(f"Gate decision is {gate['decision']}; pass allow_hold=True to override")
    candidates = store.list_candidates(run_id)
    baseline = _candidate_by_role(candidates, "baseline")
    evolved = _candidate_by_id(candidates, gate["candidate_id"])
    return run, target, repo, gate, baseline, evolved


def _candidate_by_role(candidates: list[dict[str, Any]], role: str) -> dict[str, Any]:
    matches = [candidate for candidate in candidates if candidate["role"] == role]
    if not matches:
        raise ValueError(f"Run is missing {role} candidate")
    return matches[-1]


def _candidate_by_id(candidates: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    for candidate in candidates:
        if candidate["id"] == candidate_id:
            return candidate
    raise ValueError(f"Candidate not found for latest gate: {candidate_id}")


def _artifact_text(store: EvolutionStore, artifact_id: str) -> str:
    artifact = _require(store.get_artifact(artifact_id), f"Artifact not found: {artifact_id}")
    path = Path(artifact["storage_uri"])
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path}")
    return path.read_text()


def _unified_diff(old_text: str, new_text: str, target_file_path: str) -> str:
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{target_file_path}",
            tofile=f"b/{target_file_path}",
        )
    )


def _is_git_repo(repo_path: Path) -> bool:
    return (repo_path / ".git").exists()


def _git_dirty(repo_path: Path) -> bool:
    result = _git(repo_path, ["status", "--porcelain"], check=True)
    return bool(result.stdout.strip())


def _git(repo_path: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo_path,
        text=True,
        capture_output=True,
        check=check,
    )


def _default_commit_message(target: dict[str, Any], run_id: str) -> str:
    return f"evolve {target['target_type']}:{target['name']} via {run_id[:12]}"


def _require(value: Any, message: str) -> Any:
    if value is None:
        raise ValueError(message)
    return value

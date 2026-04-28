"""Export PR-ready review bundles without applying changes."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from evolution.artifacts.store import ArtifactStore
from evolution.db.store import EvolutionStore


def export_review_bundle(
    store: EvolutionStore,
    root: str | Path,
    run_id: str,
    out_dir: str | Path | None = None,
    allow_hold: bool = False,
) -> dict[str, Any]:
    """Write an inspectable human-review bundle for the latest gated candidate.

    The bundle is deliberately non-mutating: it never overwrites target files and
    never opens/pushes a PR. It emits baseline/evolved files, a unified diff, an
    APPLY.md review checklist, and a manifest artifact linked back to SQLite.
    """
    root = Path(root)
    run = _require(store.get_run(run_id), f"Run not found: {run_id}")
    if run["status"] != "completed":
        raise ValueError(f"Run {run_id} is not completed; status={run['status']}")

    latest_gate = _latest_gate(store, run_id)
    if latest_gate["decision"] != "pass" and not allow_hold:
        raise ValueError(
            f"Gate decision is {latest_gate['decision']}; pass allow_hold=True to export anyway"
        )

    target = _require(store.get_target(run["target_id"]), f"Target not found: {run['target_id']}")
    repo = _require(store.get_repository_by_id(target["repository_id"]), f"Repository not found: {target['repository_id']}")
    candidates = store.list_candidates(run_id)
    baseline = _candidate_by_role(candidates, "baseline")
    evolved = _candidate_by_id(candidates, latest_gate["candidate_id"])
    baseline_artifact = _require(store.get_artifact(baseline["artifact_id"]), f"Artifact not found: {baseline['artifact_id']}")
    evolved_artifact = _require(store.get_artifact(evolved["artifact_id"]), f"Artifact not found: {evolved['artifact_id']}")
    baseline_text = Path(baseline_artifact["storage_uri"]).read_text()
    evolved_text = Path(evolved_artifact["storage_uri"]).read_text()

    bundle_root = Path(out_dir) if out_dir else root / "exports"
    bundle_dir = bundle_root / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    baseline_file = bundle_dir / "baseline_SKILL.md"
    evolved_file = bundle_dir / "evolved_SKILL.md"
    diff_file = bundle_dir / "candidate.diff"
    apply_file = bundle_dir / "APPLY.md"
    manifest_file = bundle_dir / "manifest.json"

    baseline_file.write_text(baseline_text)
    evolved_file.write_text(evolved_text)
    diff_text = _unified_diff(baseline_text, evolved_text, target["file_path"])
    diff_file.write_text(diff_text)
    apply_file.write_text(_apply_doc(run_id, target, latest_gate))

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "target": f"{target['target_type']}:{target['name']}",
        "target_file": target["file_path"],
        "repository": {
            "id": repo["id"],
            "name": repo["name"],
            "local_path": repo["local_path"],
        },
        "gate_result_id": latest_gate["id"],
        "gate_decision": latest_gate["decision"],
        "gate_reasons": latest_gate["reasons_json"],
        "gate_metrics": latest_gate["metrics_json"],
        "candidate_id": evolved["id"],
        "baseline_candidate_id": baseline["id"],
        "apply_policy": "human_review_required",
        "auto_apply": False,
        "files": {
            "baseline": baseline_file.name,
            "evolved": evolved_file.name,
            "diff": diff_file.name,
            "apply_instructions": apply_file.name,
        },
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    manifest_ref = ArtifactStore(root).write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        suffix=".json",
        kind="review_bundle_manifest",
        mime_type="application/json",
        metadata={"run_id": run_id, "candidate_id": evolved["id"], "gate_decision": latest_gate["decision"]},
    )
    artifact = store.add_artifact(
        kind="review_bundle_manifest",
        content_sha256=manifest_ref.content_sha256,
        storage_uri=manifest_ref.storage_uri,
        size_bytes=manifest_ref.size_bytes,
        target_id=target["id"],
        mime_type="application/json",
        metadata=manifest_ref.metadata,
    )
    store.add_run_event(
        run_id,
        "export",
        "review bundle exported",
        {"bundle_dir": str(bundle_dir), "manifest_artifact_id": artifact["id"]},
    )

    return {
        "run_id": run_id,
        "candidate_id": evolved["id"],
        "gate_decision": latest_gate["decision"],
        "bundle_dir": str(bundle_dir),
        "manifest_artifact_id": artifact["id"],
        "files": manifest["files"],
    }


def _latest_gate(store: EvolutionStore, run_id: str) -> dict[str, Any]:
    gates = store.list_gate_results(run_id)
    if not gates:
        raise ValueError(f"Run {run_id} has no gate result. Run `hermes-evolve run gate {run_id}` first.")
    return gates[0]


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


def _unified_diff(baseline_text: str, evolved_text: str, target_file_path: str) -> str:
    return "".join(
        difflib.unified_diff(
            baseline_text.splitlines(keepends=True),
            evolved_text.splitlines(keepends=True),
            fromfile=f"a/{target_file_path}",
            tofile=f"b/{target_file_path}",
        )
    )


def _apply_doc(run_id: str, target: dict[str, Any], gate: dict[str, Any]) -> str:
    decision = gate["decision"]
    hold_warning = "\nHOLD: Gate did not pass. Export exists for inspection only. Do not apply without explicit human override.\n" if decision != "pass" else ""
    return f"""# Review bundle for {run_id}

Target: {target['target_type']}:{target['name']}
File: {target['file_path']}
Gate decision: {decision}
Reasons: {', '.join(gate['reasons_json']) if gate['reasons_json'] else 'none'}
{hold_warning}
## Files
- baseline_SKILL.md: original target candidate
- evolved_SKILL.md: evolved candidate to review
- candidate.diff: unified diff against the target file
- manifest.json: immutable review metadata

## Policy
Human review is required. This exporter does not mutate the repository, create a PR, or push upstream.

## Manual apply sketch
1. Inspect candidate.diff.
2. If acceptable, copy evolved_SKILL.md over {target['file_path']} in the target repo.
3. Run the target repo tests.
4. Commit and open a PR manually.
"""


def _require(value: Any, message: str) -> Any:
    if value is None:
        raise ValueError(message)
    return value

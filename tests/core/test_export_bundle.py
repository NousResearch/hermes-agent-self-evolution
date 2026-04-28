"""Tests for exporting human-review bundles."""

import json
from pathlib import Path

import pytest

from evolution.orchestrator.exporter import export_review_bundle
from tests.core.test_gate_evaluator import _seed_completed_run_with_scores
from evolution.orchestrator.gates import evaluate_run_gate


def test_export_review_bundle_requires_passing_gate_by_default(tmp_path):
    root, store, run, _baseline, _evolved = _seed_completed_run_with_scores(
        tmp_path,
        baseline_holdout=0.90,
        evolved_holdout=0.70,
    )
    evaluate_run_gate(store, root, run["id"])

    with pytest.raises(ValueError, match="Gate decision is hold"):
        export_review_bundle(store, root, run["id"])


def test_export_review_bundle_writes_inspectable_files_and_manifest_artifact(tmp_path):
    root, store, run, _baseline, evolved = _seed_completed_run_with_scores(tmp_path)
    evaluate_run_gate(store, root, run["id"], min_holdout_improvement=0.05)

    result = export_review_bundle(store, root, run["id"])

    bundle_dir = Path(result["bundle_dir"])
    assert bundle_dir.exists()
    assert (bundle_dir / "baseline_SKILL.md").exists()
    assert (bundle_dir / "evolved_SKILL.md").exists()
    assert (bundle_dir / "candidate.diff").exists()
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "APPLY.md").exists()
    assert result["manifest_artifact_id"].startswith("artifact_")
    assert result["candidate_id"] == evolved["id"]

    diff_text = (bundle_dir / "candidate.diff").read_text()
    assert "--- a/skills/testing/test-skill/SKILL.md" in diff_text
    assert "+++ b/skills/testing/test-skill/SKILL.md" in diff_text
    assert "+## Evolution Notes" in diff_text

    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == run["id"]
    assert manifest["gate_decision"] == "pass"
    assert manifest["files"]["diff"] == "candidate.diff"
    assert manifest["apply_policy"] == "human_review_required"

    artifact = store.get_artifact(result["manifest_artifact_id"])
    assert artifact["kind"] == "review_bundle_manifest"
    assert Path(artifact["storage_uri"]).exists()
    assert any(event["event_type"] == "export" for event in store.list_run_events(run["id"]))


def test_export_review_bundle_can_export_hold_with_explicit_override(tmp_path):
    root, store, run, _baseline, _evolved = _seed_completed_run_with_scores(
        tmp_path,
        baseline_holdout=0.90,
        evolved_holdout=0.70,
    )
    evaluate_run_gate(store, root, run["id"])

    result = export_review_bundle(store, root, run["id"], allow_hold=True)

    manifest = json.loads((Path(result["bundle_dir"]) / "manifest.json").read_text())
    assert manifest["gate_decision"] == "hold"
    assert "holdout_regression" in manifest["gate_reasons"]
    assert "HOLD" in (Path(result["bundle_dir"]) / "APPLY.md").read_text()

"""Tests for the local candidate bundle contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution.core.candidate_bundle import (
    ALLOWED_DECISION_STATUSES,
    create_candidate_bundle,
    default_runs_root,
    write_bundle_json,
    write_bundle_text,
    write_decision,
)


def test_default_runs_root_uses_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HSE_RUNS_ROOT", str(tmp_path / "runs"))

    assert default_runs_root() == tmp_path / "runs"


def test_create_candidate_bundle_builds_standard_directory_layout(tmp_path):
    bundle = create_candidate_bundle(
        phase="Phase 2: Tool Description Evolution",
        target="search_files/read_file",
        run_id="20260622_154000",
        runs_root=tmp_path / "runs",
    )

    assert bundle.root == tmp_path / "runs" / "20260622_154000-phase-2-tool-description-evolution-search-files-read-file"
    assert bundle.inputs_dir.is_dir()
    assert bundle.candidates_dir.is_dir()
    assert bundle.eval_dir.is_dir()
    assert bundle.reports_dir.is_dir()
    assert bundle.decision_path == bundle.root / "decision.json"
    assert json.loads((bundle.inputs_dir / "target_manifest.json").read_text()) == {
        "schema_version": "hse-local-candidate-bundle-v1",
        "phase": "Phase 2: Tool Description Evolution",
        "target": "search_files/read_file",
        "candidate_only": True,
        "apply_ready": False,
    }


def test_bundle_writers_reject_paths_outside_standard_subdirectories(tmp_path):
    bundle = create_candidate_bundle("Phase 1", "skill", run_id="run", runs_root=tmp_path)

    with pytest.raises(ValueError, match="bundle writes must target"):
        write_bundle_text(bundle, "../escape.txt", "bad")
    with pytest.raises(ValueError, match="bundle writes must target"):
        write_bundle_json(bundle, "/tmp/outside.json", {})
    with pytest.raises(ValueError, match="bundle writes must target"):
        write_bundle_text(bundle, "random/file.txt", "bad")


def test_write_decision_is_candidate_only_and_forbids_github_side_effects(tmp_path):
    bundle = create_candidate_bundle("Phase 5", "tool-selection", run_id="run", runs_root=tmp_path)
    assert "PASS_CANDIDATE_ONLY" in ALLOWED_DECISION_STATUSES

    decision = write_decision(
        bundle,
        status="PASS_CANDIDATE_ONLY",
        summary="candidate passed gates but was not applied",
        metrics={"improvement": 0.12},
        artifacts={"patch": "candidates/candidate.patch"},
        generated_at="2026-06-22T15:40:00Z",
    )

    disk = json.loads(bundle.decision_path.read_text())
    assert disk == decision
    assert decision["candidate_only"] is True
    assert decision["apply_ready"] is False
    assert decision["github"]["pr_created"] is False
    assert decision["github"]["push_performed"] is False
    assert decision["github"]["merge_performed"] is False
    assert decision["safety_invariants"]["active_runtime_mutation"] is False
    assert decision["metrics"] == {"improvement": 0.12}


def test_write_decision_rejects_unknown_status(tmp_path):
    bundle = create_candidate_bundle("Phase 5", "tool-selection", run_id="run", runs_root=tmp_path)

    with pytest.raises(ValueError, match="unknown candidate bundle decision status"):
        write_decision(bundle, status="MERGED_TO_GITHUB", summary="nope")

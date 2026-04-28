"""Tests for safe promotion/apply helpers."""

from __future__ import annotations

from pathlib import Path

from evolution.orchestrator.gates import evaluate_run_gate
from evolution.orchestrator.promoter import apply_gated_candidate, draft_pr_text
from tests.core.test_gate_evaluator import _seed_completed_run_with_scores


def _seed_passed_gate_with_target_file(tmp_path):
    root, store, run, _baseline, _evolved = _seed_completed_run_with_scores(tmp_path)
    target = store.get_target(run["target_id"])
    repo = store.get_repository_by_id(target["repository_id"])
    target_path = Path(repo["local_path"]) / target["file_path"]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("---\nname: test-skill\ndescription: Test skill\n---\n\n# Old Skill\n")
    gate = evaluate_run_gate(store, root, run["id"])
    assert gate["decision"] == "pass"
    return root, store, run, target_path


def test_apply_gated_candidate_dry_run_does_not_mutate_target(tmp_path):
    root, store, run, target_path = _seed_passed_gate_with_target_file(tmp_path)
    before = target_path.read_text()

    result = apply_gated_candidate(store, root, run["id"], branch="evolve/test", dry_run=True)

    assert result["mode"] == "dry-run"
    assert result["mutated"] is False
    assert result["target_file"] == str(target_path)
    assert result["branch"] == "evolve/test"
    assert "diff" in result and "Evolution Notes" in result["diff"]
    assert target_path.read_text() == before


def test_apply_gated_candidate_requires_passed_gate_by_default(tmp_path):
    root, store, run, target_path = _seed_passed_gate_with_target_file(tmp_path)
    # Add a newer HOLD gate to prove latest gate is authoritative.
    gates = store.list_gate_results(run["id"])
    store.add_gate_result(run["id"], gates[0]["candidate_id"], "hold", ["manual_hold"], {"metric_name": "rubric_score"})

    try:
        apply_gated_candidate(store, root, run["id"], dry_run=True)
    except ValueError as exc:
        assert "Gate decision is hold" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("apply accepted hold gate without override")

    assert target_path.exists()


def test_apply_gated_candidate_apply_mode_writes_file_without_push_or_commit(tmp_path):
    root, store, run, target_path = _seed_passed_gate_with_target_file(tmp_path)

    result = apply_gated_candidate(store, root, run["id"], dry_run=False, branch=None, commit=False)

    assert result["mode"] == "apply"
    assert result["mutated"] is True
    assert result["committed"] is False
    assert result["pushed"] is False
    assert "Evolution Notes" in target_path.read_text()


def test_draft_pr_text_is_review_ready_and_non_mutating(tmp_path):
    root, store, run, target_path = _seed_passed_gate_with_target_file(tmp_path)
    before = target_path.read_text()

    result = draft_pr_text(store, root, run["id"], branch="evolve/test")

    assert result["title"].startswith("Evolve skill:test-skill")
    assert "Human review required" in result["body"]
    assert "run gate" in result["body"]
    assert "run export" in result["body"]
    assert result["branch"] == "evolve/test"
    assert target_path.read_text() == before

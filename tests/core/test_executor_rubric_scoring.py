"""Tests for executor rubric scoring integration."""

from __future__ import annotations

from evolution.orchestrator.executor import execute_skill_run
from tests.core.test_run_execution import _seed_skill_run


def test_execute_skill_run_persists_rubric_scores_by_default(tmp_path):
    root, _repo_path, store, run, _target, _dataset, examples = _seed_skill_run(tmp_path)

    result = execute_skill_run(store=store, root=root, run_id=run["id"], strategy="deterministic")

    evaluations = store.list_evaluations(run["id"])
    assert result["run"]["status"] == "completed"
    assert len(evaluations) == len(examples) * 2
    assert {evaluation["metric_name"] for evaluation in evaluations} == {"rubric_score"}
    first_details = evaluations[0]["details_json"]
    assert set(first_details["dimensions"]) == {"correctness", "procedure_following", "safety", "concision"}
    assert "rationale" in first_details
    assert first_details["scoring_strategy"] == "deterministic-rubric"


def test_execute_skill_run_can_use_legacy_keyword_overlap_when_requested(tmp_path):
    root, _repo_path, store, run, _target, _dataset, _examples = _seed_skill_run(tmp_path)

    execute_skill_run(
        store=store,
        root=root,
        run_id=run["id"],
        strategy="deterministic",
        scoring_strategy="keyword-overlap",
    )

    evaluations = store.list_evaluations(run["id"])
    assert {evaluation["metric_name"] for evaluation in evaluations} == {"keyword_overlap"}

"""Tests that gates prefer rubric-backed scores over legacy keyword overlap."""

from evolution.orchestrator.gates import evaluate_run_gate
from tests.core.test_gate_evaluator import _seed_completed_run_with_scores


def test_gate_prefers_rubric_metric_when_present(tmp_path):
    root, store, run, baseline, evolved = _seed_completed_run_with_scores(
        tmp_path,
        baseline_holdout=0.95,
        evolved_holdout=0.10,
    )
    dataset_id = store.get_run(run["id"])["dataset_id"]
    holdout_example = store.list_eval_examples(dataset_id, split="holdout")[0]
    store.add_evaluation(
        run_id=run["id"],
        candidate_id=baseline["id"],
        dataset_id=dataset_id,
        split="holdout",
        example_id=holdout_example["id"],
        metric_name="rubric_score",
        score=0.40,
        details={"rationale": "baseline misses rubric"},
    )
    store.add_evaluation(
        run_id=run["id"],
        candidate_id=evolved["id"],
        dataset_id=dataset_id,
        split="holdout",
        example_id=holdout_example["id"],
        metric_name="rubric_score",
        score=0.75,
        details={"rationale": "evolved satisfies rubric"},
    )

    result = evaluate_run_gate(store, root, run["id"], preferred_metric="rubric_score")

    assert result["decision"] == "pass"
    assert result["metrics"]["metric_name"] == "rubric_score"
    assert result["metrics"]["baseline_holdout_score"] == 0.40
    assert result["metrics"]["evolved_holdout_score"] == 0.75
    assert result["metrics"]["holdout_improvement"] == 0.35

"""Tests for benchmark gate persistence and decisions."""

import hashlib
from pathlib import Path

from evolution.artifacts.store import ArtifactStore
from evolution.db.store import EvolutionStore
from evolution.orchestrator.gates import evaluate_run_gate


def _add_text_artifact(store, artifact_store, target_id, text, role):
    ref = artifact_store.write_text(text, suffix=".md", kind="skill_candidate", mime_type="text/markdown")
    return store.add_artifact(
        kind="skill_candidate",
        content_sha256=ref.content_sha256,
        storage_uri=ref.storage_uri,
        size_bytes=ref.size_bytes,
        target_id=target_id,
        mime_type="text/markdown",
        metadata={"role": role},
    )


def _seed_completed_run_with_scores(tmp_path, baseline_holdout=0.50, evolved_holdout=0.60):
    root = tmp_path / ".evolution"
    repo_path = tmp_path / "hermes-agent"
    repo_path.mkdir()
    store = EvolutionStore(root / "evolution.db")
    store.init_schema()
    repo = store.add_repository("hermes-agent", repo_path)
    target = store.upsert_target(repo["id"], "skill", "test-skill", "skills/testing/test-skill/SKILL.md")
    dataset = store.add_dataset(
        target_id=target["id"],
        source="golden",
        version="v1",
        artifact_id=None,
        split_spec={"holdout": 1},
        pii_scan_status="not_implemented",
        secret_scan_status="passed",
        example_count=1,
    )
    example = store.add_eval_example(dataset["id"], "holdout", "do work", "be correct")
    body = "# Test Skill\n\n" + "Follow a stable verified procedure. " * 80
    evolved_body = body + "\n\n## Evolution Notes\n- Tighten rubric handling.\n"
    baseline_text = f"---\nname: test-skill\ndescription: Test skill\n---\n\n{body}"
    evolved_text = f"---\nname: test-skill\ndescription: Test skill\n---\n\n{evolved_body}"
    artifact_store = ArtifactStore(root)
    baseline_artifact = _add_text_artifact(store, artifact_store, target["id"], baseline_text, "baseline")
    evolved_artifact = _add_text_artifact(store, artifact_store, target["id"], evolved_text, "evolved")
    run = store.create_run(target_id=target["id"], dataset_id=dataset["id"], engine="gepa", config={})
    store.update_run_status(run["id"], "completed", completed=True)
    baseline_candidate = store.add_candidate(
        run_id=run["id"],
        target_id=target["id"],
        role="baseline",
        artifact_id=baseline_artifact["id"],
        content_sha256=baseline_artifact["content_sha256"],
    )
    evolved_candidate = store.add_candidate(
        run_id=run["id"],
        target_id=target["id"],
        role="evolved",
        artifact_id=evolved_artifact["id"],
        content_sha256=evolved_artifact["content_sha256"],
        parent_candidate_id=baseline_candidate["id"],
        metadata={"holdout_examples_used_for_generation": 0},
    )
    for candidate, score in [(baseline_candidate, baseline_holdout), (evolved_candidate, evolved_holdout)]:
        store.add_evaluation(
            run_id=run["id"],
            candidate_id=candidate["id"],
            dataset_id=dataset["id"],
            split="holdout",
            example_id=example["id"],
            metric_name="keyword_overlap",
            score=score,
            details={"seed": "unit-test"},
        )
    return root, store, run, baseline_candidate, evolved_candidate


def test_gate_passes_when_holdout_improves_and_constraints_pass(tmp_path):
    root, store, run, _baseline, evolved = _seed_completed_run_with_scores(tmp_path)

    result = evaluate_run_gate(store, root, run["id"], min_holdout_improvement=0.05)

    assert result["decision"] == "pass"
    assert result["candidate_id"] == evolved["id"]
    assert result["metrics"]["holdout_improvement"] == 0.10
    persisted = store.list_gate_results(run["id"])
    assert persisted[0]["decision"] == "pass"
    assert persisted[0]["candidate_id"] == evolved["id"]


def test_gate_holds_when_candidate_regresses_on_holdout(tmp_path):
    root, store, run, _baseline, _evolved = _seed_completed_run_with_scores(
        tmp_path,
        baseline_holdout=0.80,
        evolved_holdout=0.70,
    )

    result = evaluate_run_gate(store, root, run["id"], min_holdout_improvement=0.0)

    assert result["decision"] == "hold"
    assert "holdout_regression" in result["reasons"]
    assert store.list_gate_results(run["id"])[0]["reasons_json"] == result["reasons"]

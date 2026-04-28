"""Tests for candidate and evaluation persistence."""

from evolution.db.store import EvolutionStore


def test_candidate_and_evaluation_roundtrip(tmp_path):
    store = EvolutionStore(tmp_path / "evolution.db")
    store.init_schema()
    repo = store.add_repository("hermes-agent", tmp_path)
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
    example = store.add_eval_example(
        dataset_id=dataset["id"],
        split="holdout",
        task_input="do work",
        expected_behavior="be correct",
    )
    artifact = store.add_artifact(
        kind="skill_candidate",
        content_sha256="a" * 64,
        storage_uri=str(tmp_path / "candidate.md"),
        size_bytes=12,
        target_id=target["id"],
    )
    run = store.create_run(target_id=target["id"], dataset_id=dataset["id"], engine="gepa", config={})

    candidate = store.add_candidate(
        run_id=run["id"],
        target_id=target["id"],
        role="baseline",
        artifact_id=artifact["id"],
        content_sha256=artifact["content_sha256"],
        metadata={"source": "unit-test"},
    )
    evaluation = store.add_evaluation(
        run_id=run["id"],
        candidate_id=candidate["id"],
        dataset_id=dataset["id"],
        split="holdout",
        example_id=example["id"],
        metric_name="keyword_overlap",
        score=0.75,
        details={"overlap": 3},
    )

    assert candidate["id"].startswith("candidate_")
    assert evaluation["id"].startswith("eval_")
    assert store.list_candidates(run["id"])[0]["metadata_json"]["source"] == "unit-test"
    assert store.list_evaluations(run["id"])[0]["details_json"]["overlap"] == 3
    assert store.list_evaluations(run["id"], candidate_id=candidate["id"])[0]["score"] == 0.75


def test_run_status_updates_record_completion_and_error(tmp_path):
    store = EvolutionStore(tmp_path / "evolution.db")
    store.init_schema()
    repo = store.add_repository("hermes-agent", tmp_path)
    target = store.upsert_target(repo["id"], "skill", "test-skill", "skills/testing/test-skill/SKILL.md")
    run = store.create_run(target_id=target["id"], engine="gepa", config={})

    running = store.update_run_status(run["id"], "running")
    failed = store.update_run_status(run["id"], "failed", error="boom", completed=True)

    assert running["status"] == "running"
    assert failed["status"] == "failed"
    assert failed["completed_at"] is not None
    assert failed["error"] == "boom"

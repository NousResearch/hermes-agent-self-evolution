"""Tests for gate-result persistence."""

from evolution.db.store import EvolutionStore


def test_gate_result_roundtrip(tmp_path):
    store = EvolutionStore(tmp_path / "evolution.db")
    store.init_schema()
    repo = store.add_repository("hermes-agent", tmp_path)
    target = store.upsert_target(repo["id"], "skill", "test-skill", "skills/testing/test-skill/SKILL.md")
    artifact = store.add_artifact(
        kind="skill_candidate",
        content_sha256="b" * 64,
        storage_uri=str(tmp_path / "candidate.md"),
        size_bytes=12,
        target_id=target["id"],
    )
    run = store.create_run(target_id=target["id"], engine="gepa", config={})
    candidate = store.add_candidate(
        run_id=run["id"],
        target_id=target["id"],
        role="evolved",
        artifact_id=artifact["id"],
        content_sha256=artifact["content_sha256"],
    )

    gate = store.add_gate_result(
        run_id=run["id"],
        candidate_id=candidate["id"],
        decision="hold",
        reasons=["holdout_regression"],
        metrics={"holdout_improvement": -0.1},
    )

    assert gate["id"].startswith("gate_")
    assert gate["reasons_json"] == ["holdout_regression"]
    assert gate["metrics_json"]["holdout_improvement"] == -0.1
    assert store.list_gate_results(run["id"])[0]["id"] == gate["id"]

"""Tests for run-manager helpers."""

from evolution.db.store import EvolutionStore
from evolution.orchestrator.run_manager import create_skill_run


def test_create_skill_run_validates_target_and_dataset(tmp_path):
    store = EvolutionStore(tmp_path / "evolution.db")
    store.init_schema()
    repo = store.add_repository("hermes-agent", tmp_path)
    target = store.upsert_target(repo["id"], "skill", "test-skill", "skills/test/SKILL.md")
    dataset = store.add_dataset(
        target_id=target["id"],
        source="golden",
        version="v1",
        artifact_id=None,
        split_spec={"train": 1},
        pii_scan_status="not_implemented",
        secret_scan_status="passed",
        example_count=1,
    )

    run = create_skill_run(
        store=store,
        target_ref="skill:test-skill",
        dataset_id=dataset["id"],
        engine="gepa",
        iterations=7,
    )

    assert run["id"].startswith("run_")
    assert run["target_id"] == target["id"]
    assert run["dataset_id"] == dataset["id"]
    assert run["engine"] == "gepa"
    assert run["status"] == "pending"
    assert run["config_json"]["iterations"] == 7
    assert run["config_json"]["target"] == "skill:test-skill"

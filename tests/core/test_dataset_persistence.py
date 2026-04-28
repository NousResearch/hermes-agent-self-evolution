"""Tests for dataset persistence and eval examples."""

from evolution.db.store import EvolutionStore


def test_dataset_and_eval_examples_roundtrip(tmp_path):
    db = EvolutionStore(tmp_path / "evolution.db")
    db.init_schema()
    repo = db.add_repository("hermes-agent", tmp_path)
    target = db.upsert_target(repo["id"], "skill", "test-skill", "skills/test/SKILL.md")
    artifact = db.add_artifact(
        kind="dataset",
        content_sha256="a" * 64,
        storage_uri=str(tmp_path / "manifest.json"),
        size_bytes=2,
    )

    dataset = db.add_dataset(
        target_id=target["id"],
        source="golden",
        version="v1",
        artifact_id=artifact["id"],
        split_spec={"train": 1, "val": 1, "holdout": 1},
        pii_scan_status="passed",
        secret_scan_status="passed",
        example_count=3,
    )
    db.add_eval_example(dataset["id"], "train", "task a", "do a", difficulty="easy", category="unit")
    db.add_eval_example(dataset["id"], "val", "task b", "do b")
    db.add_eval_example(dataset["id"], "holdout", "task c", "do c", metadata={"source": "fixture"})

    loaded = db.get_dataset(dataset["id"])
    examples = db.list_eval_examples(dataset["id"])

    assert loaded["artifact_id"] == artifact["id"]
    assert loaded["split_spec_json"] == {"train": 1, "val": 1, "holdout": 1}
    assert len(examples) == 3
    assert examples[0]["task_input"] == "task a"
    assert examples[2]["metadata_json"]["source"] == "fixture"


def test_list_datasets_filters_by_target(tmp_path):
    db = EvolutionStore(tmp_path / "evolution.db")
    db.init_schema()
    repo = db.add_repository("hermes-agent", tmp_path)
    target = db.upsert_target(repo["id"], "skill", "test-skill", "skills/test/SKILL.md")
    dataset = db.add_dataset(
        target_id=target["id"],
        source="golden",
        version="v1",
        artifact_id=None,
        split_spec={},
        pii_scan_status="passed",
        secret_scan_status="passed",
        example_count=0,
    )

    assert db.list_datasets(target_id=target["id"])[0]["id"] == dataset["id"]

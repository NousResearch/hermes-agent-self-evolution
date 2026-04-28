"""Tests for executing registered evolution runs."""

from pathlib import Path

import pytest

from evolution.db.store import EvolutionStore
from evolution.orchestrator.executor import execute_skill_run
from evolution.orchestrator.run_manager import create_skill_run


def _seed_skill_run(tmp_path):
    root = tmp_path / ".evolution"
    repo_path = tmp_path / "hermes-agent"
    skill_dir = repo_path / "skills" / "testing" / "test-skill"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: test-skill\n"
        "description: Test skill\n"
        "---\n\n"
        "# Test Skill\n\n"
        "Follow the existing procedure.\n"
    )

    store = EvolutionStore(root / "evolution.db")
    store.init_schema()
    repo = store.add_repository("hermes-agent", repo_path)
    target = store.upsert_target(repo["id"], "skill", "test-skill", str(skill_path.relative_to(repo_path)))
    dataset = store.add_dataset(
        target_id=target["id"],
        source="golden",
        version="v1",
        artifact_id=None,
        split_spec={"train": 1, "val": 1, "holdout": 1},
        pii_scan_status="not_implemented",
        secret_scan_status="passed",
        example_count=3,
    )
    examples = []
    for split, expected in [
        ("train", "mention train-only calibration"),
        ("val", "mention validation-only rubric"),
        ("holdout", "mention holdout-only forbidden leak"),
    ]:
        examples.append(
            store.add_eval_example(
                dataset_id=dataset["id"],
                split=split,
                task_input=f"task for {split}",
                expected_behavior=expected,
            )
        )
    run = create_skill_run(store, "skill:test-skill", dataset["id"], engine="gepa", iterations=2)
    return root, repo_path, store, run, target, dataset, examples


def test_execute_skill_run_persists_candidates_evaluations_and_completion(tmp_path):
    root, _repo_path, store, run, target, dataset, examples = _seed_skill_run(tmp_path)

    result = execute_skill_run(store=store, root=root, run_id=run["id"], strategy="deterministic")

    completed = store.get_run(run["id"])
    candidates = store.list_candidates(run["id"])
    evaluations = store.list_evaluations(run["id"])

    assert result["run"]["status"] == "completed"
    assert completed["status"] == "completed"
    assert completed["completed_at"] is not None
    assert completed["error"] is None
    assert [candidate["role"] for candidate in candidates] == ["baseline", "evolved"]
    assert all(candidate["target_id"] == target["id"] for candidate in candidates)
    assert len(evaluations) == len(examples) * len(candidates)
    assert {evaluation["split"] for evaluation in evaluations} == {"train", "val", "holdout"}
    assert result["manifest_artifact_id"].startswith("artifact_")

    evolved = next(candidate for candidate in candidates if candidate["role"] == "evolved")
    evolved_artifact = store.get_artifact(evolved["artifact_id"])
    evolved_text = Path(evolved_artifact["storage_uri"]).read_text()
    assert "train-only calibration" in evolved_text
    assert "validation-only rubric" in evolved_text
    assert "holdout-only forbidden leak" not in evolved_text

    manifest = store.get_artifact(result["manifest_artifact_id"])
    assert Path(manifest["storage_uri"]).exists()


def test_execute_skill_run_marks_failed_when_target_file_is_missing(tmp_path):
    root, repo_path, store, run, _target, _dataset, _examples = _seed_skill_run(tmp_path)
    (repo_path / "skills" / "testing" / "test-skill" / "SKILL.md").unlink()

    with pytest.raises(FileNotFoundError):
        execute_skill_run(store=store, root=root, run_id=run["id"], strategy="deterministic")

    failed = store.get_run(run["id"])
    assert failed["status"] == "failed"
    assert "Skill file not found" in failed["error"]

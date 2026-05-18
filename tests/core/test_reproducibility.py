"""Tests for reproducibility manifest generation."""

from pathlib import Path

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.core.reproducibility import (
    build_reproducibility_manifest,
    write_manifest,
)


def _dataset() -> EvalDataset:
    return EvalDataset(
        train=[
            EvalExample(
                task_input="t1",
                expected_behavior="e1",
                source="synthetic",
            ),
        ],
        val=[
            EvalExample(
                task_input="t2",
                expected_behavior="e2",
                source="synthetic",
            ),
        ],
        holdout=[
            EvalExample(
                task_input="t3",
                expected_behavior="e3",
                source="synthetic",
            ),
        ],
    )


def test_build_reproducibility_manifest_contains_expected_fields(tmp_path: Path):
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("---\nname: skill\n---\ncontent\n")
    config = EvolutionConfig(hermes_agent_path=tmp_path)

    manifest = build_reproducibility_manifest(
        skill_name="skill",
        skill_path=skill_path,
        dataset=_dataset(),
        config=config,
        baseline_score=0.2,
        evolved_score=0.3,
        improvement=0.1,
        elapsed_seconds=5.0,
        evolution_repo_path=tmp_path,
    )

    data = manifest.to_dict()
    assert data["skill_name"] == "skill"
    assert data["dataset_counts"]["train"] == 1
    assert len(data["dataset_train_sha256"]) == 64
    assert len(data["skill_sha256"]) == 64


def test_write_manifest_writes_json(tmp_path: Path):
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("---\nname: skill\n---\ncontent\n")
    config = EvolutionConfig(hermes_agent_path=tmp_path)
    manifest = build_reproducibility_manifest(
        skill_name="skill",
        skill_path=skill_path,
        dataset=_dataset(),
        config=config,
        baseline_score=0.2,
        evolved_score=0.3,
        improvement=0.1,
        elapsed_seconds=1.0,
        evolution_repo_path=tmp_path,
    )

    output = tmp_path / "manifest.json"
    write_manifest(output, manifest)
    text = output.read_text(encoding="utf-8")
    assert '"skill_name": "skill"' in text
    assert text.endswith("\n")

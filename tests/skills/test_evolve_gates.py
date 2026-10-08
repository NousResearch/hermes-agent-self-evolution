"""Integration checks for evolution stopping conditions."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.skills import evolve_skill


def test_invalid_dataset_size_rejected_before_running():
    result = CliRunner().invoke(
        evolve_skill.main, ["--skill", "demo", "--dataset-size", "0"]
    )
    assert result.exit_code != 0
    assert "is not in the range" in result.output


def test_gepa_error_does_not_silently_fall_back(monkeypatch, tmp_path):
    repo = tmp_path / "hermes-agent"
    skill = repo / "skills" / "test" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: test\ndescription: Test skill\n---\n\n# Procedure\nAct\n")

    examples = [
        EvalExample(task_input="task", expected_behavior="act")
        for _ in range(3)
    ]
    dataset = EvalDataset(train=[examples[0]], val=[examples[1]], holdout=[examples[2]])
    monkeypatch.setattr(evolve_skill.GoldenDatasetLoader, "load", lambda path: dataset)
    monkeypatch.setattr(evolve_skill, "make_lm", lambda model: object())
    monkeypatch.setattr(evolve_skill.dspy, "configure", lambda **kwargs: None)

    class BrokenOptimizer:
        def compile(self, *args, **kwargs):
            raise RuntimeError("GEPA failed to compile")

    monkeypatch.setattr(evolve_skill.dspy, "GEPA", lambda **kwargs: BrokenOptimizer())
    monkeypatch.setattr(
        evolve_skill.dspy,
        "MIPROv2",
        lambda **kwargs: pytest.fail("GEPA error must not invoke MIPROv2"),
    )

    with pytest.raises(RuntimeError, match="GEPA failed to compile"):
        evolve_skill.evolve(
            skill_name="test",
            eval_source="golden",
            dataset_path=str(tmp_path),
            hermes_repo=str(repo),
            run_tests=False,
        )


def test_empty_holdout_stops_before_optimizer(monkeypatch, tmp_path):
    repo = tmp_path / "hermes-agent"
    skill = repo / "skills" / "test" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: test\ndescription: Test skill\n---\n\nBody")
    dataset = EvalDataset(
        train=[EvalExample(task_input="task", expected_behavior="act")],
        val=[EvalExample(task_input="task", expected_behavior="act")],
        holdout=[],
    )
    monkeypatch.setattr(evolve_skill.GoldenDatasetLoader, "load", lambda path: dataset)
    with pytest.raises(ValueError, match="non-empty"):
        evolve_skill.evolve(
            skill_name="test",
            eval_source="golden",
            dataset_path=str(tmp_path),
            hermes_repo=str(repo),
            run_tests=False,
        )

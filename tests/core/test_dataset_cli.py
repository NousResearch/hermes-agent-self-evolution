"""Tests for dataset CLI commands."""

import json

from click.testing import CliRunner

from evolution.cli import main


def _write_skill(repo_path):
    skill_dir = repo_path / "skills" / "testing" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: Test skill\n---\n\n# Test\n"
    )


def _write_golden_dataset(dataset_path):
    dataset_path.mkdir(parents=True)
    splits = {
        "train": [{"task_input": "task train", "expected_behavior": "do train", "difficulty": "easy", "category": "a"}],
        "val": [{"task_input": "task val", "expected_behavior": "do val"}],
        "holdout": [{"task_input": "task holdout", "expected_behavior": "do holdout"}],
    }
    for split, rows in splits.items():
        with open(dataset_path / f"{split}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def test_cli_dataset_build_golden_persists_dataset(tmp_path):
    runner = CliRunner()
    root = tmp_path / ".evolution"
    repo_path = tmp_path / "hermes-agent"
    repo_path.mkdir()
    _write_skill(repo_path)
    golden_path = tmp_path / "golden"
    _write_golden_dataset(golden_path)

    assert runner.invoke(main, ["--root", str(root), "init"]).exit_code == 0
    assert runner.invoke(main, ["--root", str(root), "repo", "add", "hermes-agent", "--path", str(repo_path)]).exit_code == 0
    assert runner.invoke(main, ["--root", str(root), "targets", "scan", "--repo", "hermes-agent"]).exit_code == 0

    result = runner.invoke(
        main,
        [
            "--root", str(root),
            "dataset", "build",
            "--target", "skill:test-skill",
            "--source", "golden",
            "--path", str(golden_path),
            "--version", "v1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "dataset_" in result.output
    assert "3 examples" in result.output
    assert (root / "datasets" / "skill" / "test-skill").exists()

    list_result = runner.invoke(main, ["--root", str(root), "dataset", "list", "--target", "skill:test-skill"])
    assert list_result.exit_code == 0, list_result.output
    assert "golden" in list_result.output
    assert "3 examples" in list_result.output


def test_cli_dataset_build_blocks_secret_contaminated_data(tmp_path):
    runner = CliRunner()
    root = tmp_path / ".evolution"
    repo_path = tmp_path / "hermes-agent"
    repo_path.mkdir()
    _write_skill(repo_path)
    golden_path = tmp_path / "golden"
    golden_path.mkdir()
    (golden_path / "train.jsonl").write_text(
        json.dumps({"task_input": "OPENAI_API_KEY=sk-abc12345678901234567890", "expected_behavior": "no"}) + "\n"
    )
    (golden_path / "val.jsonl").write_text("")
    (golden_path / "holdout.jsonl").write_text("")

    assert runner.invoke(main, ["--root", str(root), "init"]).exit_code == 0
    assert runner.invoke(main, ["--root", str(root), "repo", "add", "hermes-agent", "--path", str(repo_path)]).exit_code == 0
    assert runner.invoke(main, ["--root", str(root), "targets", "scan", "--repo", "hermes-agent"]).exit_code == 0

    result = runner.invoke(
        main,
        ["--root", str(root), "dataset", "build", "--target", "skill:test-skill", "--source", "golden", "--path", str(golden_path)],
    )

    assert result.exit_code != 0
    assert "secret scan failed" in result.output

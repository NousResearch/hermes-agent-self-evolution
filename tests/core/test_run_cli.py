"""Tests for run CLI commands."""

import json

from click.testing import CliRunner

from evolution.cli import main


def _setup_target_and_dataset(tmp_path):
    runner = CliRunner()
    root = tmp_path / ".evolution"
    repo_path = tmp_path / "hermes-agent"
    repo_path.mkdir()
    skill_dir = repo_path / "skills" / "testing" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: test-skill\ndescription: Test\n---\n\n# Test\n")
    golden = tmp_path / "golden"
    golden.mkdir()
    for split in ["train", "val", "holdout"]:
        (golden / f"{split}.jsonl").write_text(
            json.dumps({"task_input": f"task {split}", "expected_behavior": f"do {split}"}) + "\n"
        )

    assert runner.invoke(main, ["--root", str(root), "init"]).exit_code == 0
    assert runner.invoke(main, ["--root", str(root), "repo", "add", "hermes-agent", "--path", str(repo_path)]).exit_code == 0
    assert runner.invoke(main, ["--root", str(root), "targets", "scan", "--repo", "hermes-agent"]).exit_code == 0
    dataset_result = runner.invoke(
        main,
        ["--root", str(root), "dataset", "build", "--target", "skill:test-skill", "--source", "golden", "--path", str(golden)],
    )
    assert dataset_result.exit_code == 0, dataset_result.output
    dataset_id = next(part for part in dataset_result.output.split() if part.startswith("dataset_"))
    return runner, root, dataset_id


def test_cli_run_skill_creates_pending_run(tmp_path):
    runner, root, dataset_id = _setup_target_and_dataset(tmp_path)

    result = runner.invoke(
        main,
        [
            "--root", str(root),
            "run", "skill",
            "--target", "skill:test-skill",
            "--dataset", dataset_id,
            "--engine", "gepa",
            "--iterations", "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Created run run_" in result.output
    assert dataset_id in result.output

    list_result = runner.invoke(main, ["--root", str(root), "runs", "list"])
    assert list_result.exit_code == 0, list_result.output
    assert "pending" in list_result.output
    assert "gepa" in list_result.output

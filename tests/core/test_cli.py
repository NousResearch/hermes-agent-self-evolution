"""Tests for the hermes-evolve CLI foundation."""

import shutil
import subprocess

from click.testing import CliRunner

from evolution.cli import main


def _write_skill(repo_path, name="test-skill", description="A test skill"):
    skill_dir = repo_path / "skills" / "testing" / name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\nDo the thing.\n"
    )
    return skill_file


def test_cli_init_repo_add_targets_scan_and_runs_list(tmp_path):
    runner = CliRunner()
    root = tmp_path / ".evolution"
    repo_path = tmp_path / "hermes-agent"
    repo_path.mkdir()
    _write_skill(repo_path)

    result = runner.invoke(main, ["--root", str(root), "init"])
    assert result.exit_code == 0, result.output
    assert (root / "evolution.db").exists()
    assert (root / "config.yaml").exists()

    result = runner.invoke(main, ["--root", str(root), "repo", "add", "hermes-agent", "--path", str(repo_path)])
    assert result.exit_code == 0, result.output
    assert "hermes-agent" in result.output

    result = runner.invoke(main, ["--root", str(root), "targets", "scan", "--repo", "hermes-agent", "--type", "skill"])
    assert result.exit_code == 0, result.output
    assert "1 target" in result.output
    assert "test-skill" in result.output

    result = runner.invoke(main, ["--root", str(root), "runs", "list"])
    assert result.exit_code == 0, result.output
    assert "No runs" in result.output


def test_cli_repo_snapshot_records_git_state(tmp_path):
    if not shutil.which("git"):
        return

    runner = CliRunner()
    root = tmp_path / ".evolution"
    repo_path = tmp_path / "hermes-agent"
    repo_path.mkdir()
    _write_skill(repo_path)

    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_path, check=True, capture_output=True, text=True)

    assert runner.invoke(main, ["--root", str(root), "init"]).exit_code == 0
    assert runner.invoke(main, ["--root", str(root), "repo", "add", "hermes-agent", "--path", str(repo_path)]).exit_code == 0

    result = runner.invoke(main, ["--root", str(root), "repo", "snapshot", "hermes-agent"])

    assert result.exit_code == 0, result.output
    assert "snapshot" in result.output
    assert "dirty=False" in result.output

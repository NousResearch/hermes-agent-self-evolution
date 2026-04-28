"""CLI tests for safe promotion commands."""

from evolution import cli as cli_module
from evolution.cli import main
from tests.core.test_run_cli import _setup_target_and_dataset


def _created_run_id(runner, root, dataset_id):
    create_result = runner.invoke(
        main,
        [
            "--root", str(root),
            "run", "skill",
            "--target", "skill:test-skill",
            "--dataset", dataset_id,
            "--iterations", "2",
        ],
    )
    assert create_result.exit_code == 0, create_result.output
    return next(part for part in create_result.output.split() if part.startswith("run_"))


def test_cli_run_apply_defaults_to_dry_run(monkeypatch, tmp_path):
    runner, root, dataset_id = _setup_target_and_dataset(tmp_path)
    run_id = _created_run_id(runner, root, dataset_id)
    captured = {}

    def fake_apply_gated_candidate(**kwargs):
        captured.update(kwargs)
        return {
            "mode": "dry-run",
            "mutated": False,
            "committed": False,
            "pushed": False,
            "target_file": "/tmp/repo/SKILL.md",
            "branch": "evolve/test",
            "gate_decision": "pass",
        }

    monkeypatch.setattr(cli_module, "apply_gated_candidate", fake_apply_gated_candidate)

    result = runner.invoke(main, ["--root", str(root), "run", "apply", run_id, "--branch", "evolve/test"])

    assert result.exit_code == 0, result.output
    assert captured["dry_run"] is True
    assert captured["branch"] == "evolve/test"
    assert captured["commit"] is False
    assert "mode=dry-run" in result.output
    assert "mutated=False" in result.output


def test_cli_run_pr_draft_outputs_review_text(monkeypatch, tmp_path):
    runner, root, dataset_id = _setup_target_and_dataset(tmp_path)
    run_id = _created_run_id(runner, root, dataset_id)

    def fake_draft_pr_text(**kwargs):
        return {"title": "Evolve skill:test-skill", "body": "Human review required", "branch": kwargs["branch"]}

    monkeypatch.setattr(cli_module, "draft_pr_text", fake_draft_pr_text)

    result = runner.invoke(main, ["--root", str(root), "run", "pr-draft", run_id, "--branch", "evolve/test"])

    assert result.exit_code == 0, result.output
    assert "title=Evolve skill:test-skill" in result.output
    assert "Human review required" in result.output

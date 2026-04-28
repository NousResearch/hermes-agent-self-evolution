"""CLI tests for model-backed run execution."""

from __future__ import annotations

from click.testing import CliRunner

from evolution import cli as cli_module
from evolution.cli import main
from tests.core.test_run_cli import _setup_target_and_dataset


def test_cli_run_execute_accepts_model_synthesis_options(monkeypatch, tmp_path):
    runner, root, dataset_id = _setup_target_and_dataset(tmp_path)
    create_result = runner.invoke(
        main,
        [
            "--root", str(root),
            "run", "skill",
            "--target", "skill:test-skill",
            "--dataset", dataset_id,
            "--engine", "gepa",
            "--iterations", "2",
        ],
    )
    assert create_result.exit_code == 0, create_result.output
    run_id = next(part for part in create_result.output.split() if part.startswith("run_"))

    captured = {}

    def fake_execute_skill_run(**kwargs):
        captured.update(kwargs)
        return {
            "run": {"status": "completed"},
            "candidates": [{"id": "candidate_1"}, {"id": "candidate_2"}],
            "evaluations": [{"id": "eval_1"}],
            "manifest_artifact_id": "artifact_1",
        }

    monkeypatch.setattr(cli_module, "execute_skill_run", fake_execute_skill_run)

    result = runner.invoke(
        main,
        [
            "--root", str(root),
            "run", "execute", run_id,
            "--strategy", "model-synthesis",
            "--provider", "deepseek",
            "--optimizer-model", "deepseek-v4-pro",
            "--max-tokens", "512",
            "--temperature", "0",
            "--timeout", "30",
            "--extra-body-json", '{"thinking":{"type":"disabled"}}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["strategy"] == "model-synthesis"
    assert captured["provider"] == "deepseek"
    assert captured["optimizer_model"] == "deepseek-v4-pro"
    assert captured["max_tokens"] == 512
    assert captured["temperature"] == 0.0
    assert captured["timeout"] == 30.0
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "Executed run" in result.output

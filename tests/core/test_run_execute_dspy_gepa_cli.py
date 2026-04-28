"""CLI tests for DSPy/GEPA run execution options."""

from __future__ import annotations

from evolution import cli as cli_module
from evolution.cli import main
from tests.core.test_run_cli import _setup_target_and_dataset


def test_cli_run_execute_accepts_dspy_gepa_options(monkeypatch, tmp_path):
    runner, root, dataset_id = _setup_target_and_dataset(tmp_path)
    create_result = runner.invoke(
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
            "--strategy", "dspy-gepa",
            "--provider", "deepseek",
            "--optimizer-model", "deepseek-v4-pro",
            "--eval-model", "deepseek-v4-flash",
            "--dspy-model-prefix", "openai",
            "--gepa-max-full-evals", "9",
            "--gepa-reflection-minibatch-size", "2",
            "--max-tokens", "1024",
            "--temperature", "0.1",
            "--timeout", "45",
            "--extra-body-json", '{"thinking":{"type":"disabled"}}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["strategy"] == "dspy-gepa"
    assert captured["provider"] == "deepseek"
    assert captured["optimizer_model"] == "deepseek-v4-pro"
    assert captured["eval_model"] == "deepseek-v4-flash"
    assert captured["dspy_model_prefix"] == "openai"
    assert captured["gepa_max_full_evals"] == 9
    assert captured["gepa_reflection_minibatch_size"] == 2
    assert captured["max_tokens"] == 1024
    assert captured["temperature"] == 0.1
    assert captured["timeout"] == 45.0
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "Executed run" in result.output

"""Model selection must preserve explicit options when MiniMax is enabled."""

from click.testing import CliRunner
import pytest

from evolution.skills import evolve_skill


@pytest.mark.parametrize("options, optimizer, evaluation", [
    ([], "openai/gpt-4.1", "openai/gpt-4.1-mini"),
    (["--use-minimax"], "minimax/MiniMax-M3.1-Flash", "minimax/MiniMax-M3.1-Flash"),
    (["--use-minimax", "--eval-model", "MiniMax-M2.7"],
     "minimax/MiniMax-M3.1-Flash", "MiniMax-M2.7"),
    (["--use-minimax", "--optimizer-model", "minimax/custom-model",
      "--eval-model", "gateway/MiniMax-M3"],
     "minimax/custom-model", "gateway/MiniMax-M3"),
    (["--use-minimax", "--optimizer-model", "openai/gpt-4.1"],
     "openai/gpt-4.1", "minimax/MiniMax-M3.1-Flash"),
])
def test_cli_selection(monkeypatch, options, optimizer, evaluation):
    captured = {}

    class StopAfterSelection(Exception):
        pass

    def capture_config(**kwargs):
        captured.update(kwargs)
        raise StopAfterSelection

    monkeypatch.setattr(evolve_skill, "EvolutionConfig", capture_config)
    result = CliRunner().invoke(evolve_skill.main, ["--skill", "test-skill", *options])
    assert isinstance(result.exception, StopAfterSelection)
    assert captured["optimizer_model"] == optimizer
    assert captured["eval_model"] == evaluation
    assert captured["judge_model"] == evaluation

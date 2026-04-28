"""Tests for configurable model defaults."""

from evolution.core.config import EvolutionConfig


def test_evolution_config_model_defaults_can_come_from_env(monkeypatch, tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    hermes_repo.mkdir()
    monkeypatch.setenv("HERMES_AGENT_REPO", str(hermes_repo))
    monkeypatch.setenv("HERMES_EVOLVE_OPTIMIZER_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("HERMES_EVOLVE_EVAL_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("HERMES_EVOLVE_JUDGE_MODEL", "deepseek-v4-pro")

    config = EvolutionConfig()

    assert config.optimizer_model == "deepseek-v4-pro"
    assert config.eval_model == "deepseek-v4-flash"
    assert config.judge_model == "deepseek-v4-pro"

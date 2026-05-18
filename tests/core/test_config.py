"""Tests for configuration path discovery behavior."""

from pathlib import Path

import pytest

from evolution.core import config as config_module


def test_discover_hermes_agent_path_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hermes_repo = tmp_path / "hermes-agent"
    hermes_repo.mkdir()
    monkeypatch.setenv("HERMES_AGENT_REPO", str(hermes_repo))
    assert config_module.discover_hermes_agent_path() == hermes_repo


def test_get_hermes_agent_path_non_strict_returns_default_when_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setattr(config_module, "discover_hermes_agent_path", lambda: None)
    path = config_module.get_hermes_agent_path(strict=False)
    assert path == Path.home() / ".hermes" / "hermes-agent"


def test_get_hermes_agent_path_strict_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(config_module, "discover_hermes_agent_path", lambda: None)
    with pytest.raises(FileNotFoundError):
        config_module.get_hermes_agent_path(strict=True)

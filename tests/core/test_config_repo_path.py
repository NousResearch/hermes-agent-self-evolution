"""Regression tests for hermes-agent repo path resolution.

Background
----------
Before this fix, ``EvolutionConfig.hermes_agent_path`` used a default factory
that ran ``get_hermes_agent_path()`` at construction time. That meant:

* constructing a bare ``EvolutionConfig()`` raised ``FileNotFoundError`` whenever
  the default ``~/.hermes/hermes-agent`` location was absent (this is what made
  16 tests in ``tests/core/test_constraints.py`` error on a clean checkout), and
* ``evolve(..., hermes_repo=...)`` crashed before the explicit ``--hermes-repo``
  override could be applied, so the documented flag did nothing.

These tests lock in the fixed behavior at three levels: the resolver helper,
the config dataclass, and the CLI end to end.
"""

from pathlib import Path

from click.testing import CliRunner


def _make_skill_repo(tmp_path) -> Path:
    """Create a minimal hermes-agent-style repo with one skill."""
    repo = tmp_path / "my-hermes"
    skill_dir = repo / "skills" / "testing" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\n\n# Demo\n\n1. Do the thing.\n"
    )
    return repo


# ── resolver helper ────────────────────────────────────────────────────────

def test_resolve_honors_explicit_path_without_default(tmp_path, monkeypatch):
    from evolution.core.config import resolve_hermes_agent_path

    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))  # no default location

    explicit = tmp_path / "somewhere" / "hermes-agent"
    assert resolve_hermes_agent_path(str(explicit)) == explicit


def test_resolve_expands_user_home(monkeypatch):
    from evolution.core.config import resolve_hermes_agent_path

    monkeypatch.setenv("HOME", "/home/example")
    assert resolve_hermes_agent_path("~/code/hermes-agent") == Path("/home/example/code/hermes-agent")


def test_resolve_falls_back_to_env_var_when_no_override(tmp_path, monkeypatch):
    from evolution.core.config import resolve_hermes_agent_path

    repo = tmp_path / "env-hermes"
    repo.mkdir()
    monkeypatch.setenv("HERMES_AGENT_REPO", str(repo))
    assert resolve_hermes_agent_path() == repo


# ── config dataclass (root cause) ──────────────────────────────────────────

def test_config_constructs_without_repo(tmp_path, monkeypatch):
    """A bare EvolutionConfig() must not raise when no repo is present."""
    from evolution.core.config import EvolutionConfig

    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))

    config = EvolutionConfig()  # must not raise
    assert config.hermes_agent_path is None


def test_config_preserves_explicit_path(tmp_path):
    """An explicitly supplied path is kept as-is, never overwritten by discovery."""
    from evolution.core.config import EvolutionConfig

    explicit = tmp_path / "explicit" / "hermes-agent"
    assert EvolutionConfig(hermes_agent_path=explicit).hermes_agent_path == explicit


# ── CLI end to end ─────────────────────────────────────────────────────────

def test_cli_dry_run_honors_explicit_repo(tmp_path, monkeypatch):
    """`evolve_skill --hermes-repo <path> --dry-run` must work when the default
    location is absent. This is the exact scenario the bug broke."""
    from evolution.skills.evolve_skill import main

    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty"))
    repo = _make_skill_repo(tmp_path)

    result = CliRunner().invoke(
        main,
        ["--skill", "demo", "--hermes-repo", str(repo), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    assert "Cannot find hermes-agent repo" not in result.output

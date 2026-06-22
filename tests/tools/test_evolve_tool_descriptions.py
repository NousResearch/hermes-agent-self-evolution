"""Integration tests for the tool-description orchestrator (all LLM I/O mocked)."""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evolution.tools.evolve_tool_descriptions import evolve_tools


def _fake_repo(tmp_path):
    repo = tmp_path / "hermes"
    tools = repo / "tools"
    tools.mkdir(parents=True)
    (tools / "file_tools.py").write_text(textwrap.dedent('''
        READ = {"name": "read_file", "description": "Read a file.",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "File path."},
            }}
        }
        WRITE = {"name": "write_file", "description": "Write a file.",
            "parameters": {"type": "object", "properties": {
                "content": {"type": "string", "description": "Content to write."},
            }}
        }
        SEARCH = {"name": "search_files", "description": "Search files.",
            "parameters": {"type": "object", "properties": {
                "pattern": {"type": "string", "description": "Search pattern."},
            }}
        }
    '''), encoding="utf-8")
    (tools / "terminal_tool.py").write_text(textwrap.dedent('''
        TERM = {"name": "terminal", "description": "Run a command.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string", "description": "Command to run."},
            }}
        }
    '''), encoding="utf-8")
    (tools / "web_tools.py").write_text(
        '{"name": "web_search", "description": "Search the web.", "parameters": {"properties": {}}}'
    )
    (tools / "browser_tool.py").write_text(
        '{"name": "browser_navigate", "description": "Open a URL.", "parameters": {"properties": {}}}'
    )
    return repo


def test_dry_run_returns_none_without_api(tmp_path):
    repo = _fake_repo(tmp_path)
    result = evolve_tools(hermes_repo=str(repo), dry_run=True)
    assert result is None


def test_dry_run_does_not_instantiate_lm(tmp_path):
    repo = _fake_repo(tmp_path)
    with patch("evolution.tools.evolve_tool_descriptions.dspy.LM") as mock_lm:
        evolve_tools(hermes_repo=str(repo), dry_run=True)
    mock_lm.assert_not_called()


def test_missing_repo_exits(tmp_path):
    with patch("evolution.tools.evolve_tool_descriptions.dspy.LM"):
        with pytest.raises(SystemExit):
            evolve_tools(hermes_repo=str(tmp_path / "nope"), dry_run=True)


# ── Phase 4: --evolve-params ─────────────────────────────────────────────────


def test_param_evolve_dry_run(tmp_path):
    repo = _fake_repo(tmp_path)
    result = evolve_tools(hermes_repo=str(repo), dry_run=True, evolve_params=True)
    assert result is None


def test_param_evolve_dry_run_finds_params(tmp_path):
    repo = _fake_repo(tmp_path)
    with patch("evolution.tools.evolve_tool_descriptions.dspy.LM"), \
         patch("evolution.tools.evolve_tool_descriptions.dspy.configure"), \
         patch("evolution.tools.evolve_tool_descriptions.dspy.context"), \
         patch("evolution.tools.evolve_tool_descriptions.dspy.GEPA"):
        from evolution.tools.evolve_tool_descriptions import main as cli
        from click.testing import CliRunner
        result = CliRunner().invoke(cli, [
            "--hermes-repo", str(repo),
            "--dry-run", "--evolve-params",
        ])
    assert result.exit_code == 0
    assert "param" in result.output.lower()

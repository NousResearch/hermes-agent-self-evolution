"""Integration tests for the Phase 2 orchestrator (all LLM I/O mocked)."""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evolution.tools.evolve_tool_descriptions import evolve_tools


def _fake_repo(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "file_tools.py").write_text(textwrap.dedent('''
        READ_FILE_SCHEMA = {"name": "read_file", "description": "Read a file.", "parameters": {}}
        WRITE_FILE_SCHEMA = {"name": "write_file", "description": "Write a file.", "parameters": {}}
        SEARCH_SCHEMA = {"name": "search_files", "description": "Search files.", "parameters": {}}
    '''), encoding="utf-8")
    (tools / "terminal_tool.py").write_text(textwrap.dedent('''
        TERMINAL_SCHEMA = {"name": "terminal", "description": "Run a command.", "parameters": {}}
    '''), encoding="utf-8")
    (tools / "web_tools.py").write_text(textwrap.dedent('''
        WEB_SEARCH_SCHEMA = {"name": "web_search", "description": "Search the web.", "parameters": {}}
    '''), encoding="utf-8")
    (tools / "browser_tool.py").write_text(textwrap.dedent('''
        NAV = {"name": "browser_navigate", "description": "Open a URL.", "parameters": {}}
    '''), encoding="utf-8")
    return tmp_path


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


def test_e2e_writes_output(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    from evolution.tools.tool_dataset import ToolSelectionDataset, ToolSelectionExample
    fake_ds = ToolSelectionDataset(
        train=[ToolSelectionExample("grep X", "search_files", ["terminal"], kind="confusable")],
        val=[ToolSelectionExample("read y", "read_file")],
        holdout=[ToolSelectionExample("open z", "browser_navigate")],
    )

    preflight_lm = MagicMock(return_value="OK")
    optimized = MagicMock()
    optimized.descriptions = {
        "search_files": "Search files.",
        "read_file": "Read a file fully.",
        "terminal": "Run a command now.",
        "web_search": "Search the web now.",
        "browser_navigate": "Open a URL now.",
        "write_file": "Write a file now.",
    }
    optimized.return_value = MagicMock(chosen_tool="search_files")

    with patch("evolution.tools.evolve_tool_descriptions.dspy.LM", return_value=preflight_lm), \
         patch("evolution.tools.evolve_tool_descriptions.dspy.configure"), \
         patch("evolution.tools.evolve_tool_descriptions.dspy.context"), \
         patch("evolution.tools.evolve_tool_descriptions.ToolDatasetBuilder.generate", return_value=fake_ds), \
         patch("evolution.tools.evolve_tool_descriptions.dspy.GEPA") as mock_gepa, \
         patch("evolution.tools.evolve_tool_descriptions.ToolSelectorModule") as mock_mod:
        mock_gepa.return_value.compile.return_value = optimized
        mock_mod.return_value.return_value = MagicMock(chosen_tool="search_files")
        mock_mod.return_value.descriptions = optimized.descriptions

        evolve_tools(hermes_repo=str(repo), iterations=1, dry_run=False, write_back=False)

    runs = list((tmp_path / "output" / "tools").glob("*/metrics.json"))
    assert runs, "expected metrics.json from a completed run"
    import json
    metrics = json.loads(runs[0].read_text(encoding="utf-8"))
    assert "baseline_accuracy" in metrics
    assert "evolved_accuracy" in metrics

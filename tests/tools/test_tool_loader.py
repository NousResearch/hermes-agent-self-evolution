"""Tests for the tool description loader/writer."""

import textwrap
from pathlib import Path

from evolution.tools.tool_loader import read_description_from_source


def _write(tmp_path, name, content):
    """Helper: write a temporary Python source file."""
    f = tmp_path / name
    f.write_text(textwrap.dedent(content), encoding="utf-8")
    return f


def test_read_inline_description(tmp_path):
    """Extract a description that is an inline string literal."""
    src = _write(tmp_path, "file_tools.py", '''
        READ_FILE_SCHEMA = {
            "name": "read_file",
            "description": "Read a text file with line numbers.",
            "parameters": {"type": "object", "properties": {}},
        }
    ''')
    desc = read_description_from_source(src, "read_file")
    assert desc == "Read a text file with line numbers."


def test_read_constant_description(tmp_path):
    """Extract a description stored in a module-level variable."""
    src = _write(tmp_path, "terminal_tool.py", '''
        TERMINAL_TOOL_DESCRIPTION = "Run a shell command on the VM."
        TERMINAL_SCHEMA = {
            "name": "terminal",
            "description": TERMINAL_TOOL_DESCRIPTION,
            "parameters": {"type": "object", "properties": {}},
        }
    ''')
    desc = read_description_from_source(src, "terminal")
    assert desc == "Run a shell command on the VM."


def test_read_missing_tool_returns_none(tmp_path):
    """Requesting a tool not present in the file returns None."""
    src = _write(tmp_path, "file_tools.py", '''
        OTHER_SCHEMA = {"name": "other", "description": "x", "parameters": {}}
    ''')
    assert read_description_from_source(src, "read_file") is None


# ── Task 3: repo-wide scanning ────────────────────────────────────────────────

from evolution.tools.tool_loader import read_target_descriptions, list_all_tools


def _make_repo(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "file_tools.py").write_text(textwrap.dedent('''
        READ_FILE_SCHEMA = {"name": "read_file", "description": "Read a file.", "parameters": {}}
        WRITE_FILE_SCHEMA = {"name": "write_file", "description": "Write a file.", "parameters": {}}
        SEARCH_SCHEMA = {"name": "search_files", "description": "Search files.", "parameters": {}}
    '''), encoding="utf-8")
    (tools / "terminal_tool.py").write_text(textwrap.dedent('''
        TD = "Run a command."
        TERMINAL_SCHEMA = {"name": "terminal", "description": TD, "parameters": {}}
    '''), encoding="utf-8")
    (tools / "web_tools.py").write_text(textwrap.dedent('''
        WEB_SEARCH_SCHEMA = {"name": "web_search", "description": "Search the web.", "parameters": {}}
    '''), encoding="utf-8")
    (tools / "browser_tool.py").write_text(textwrap.dedent('''
        NAV = {"name": "browser_navigate", "description": "Open a URL.", "parameters": {}}
    '''), encoding="utf-8")
    return tmp_path


def test_read_target_descriptions_finds_all_six(tmp_path):
    repo = _make_repo(tmp_path)
    descs = read_target_descriptions(repo)
    assert descs["read_file"] == "Read a file."
    assert descs["terminal"] == "Run a command."
    assert descs["browser_navigate"] == "Open a URL."
    assert set(descs.keys()) == {
        "search_files", "read_file", "terminal",
        "web_search", "browser_navigate", "write_file",
    }


def test_list_all_tools_includes_nontarget(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "tools" / "extra.py").write_text(
        '{"name": "vision_analyze", "description": "Analyze an image.", "parameters": {}}',
        encoding="utf-8",
    )
    all_tools = dict(list_all_tools(repo))
    assert "vision_analyze" in all_tools
    assert "read_file" in all_tools
    assert all_tools["read_file"] == "Read a file."


# ── Task 4: safe write-back ───────────────────────────────────────────────────

from evolution.tools.tool_loader import write_description_to_source, WriteResult


def test_write_back_replaces_unique_literal(tmp_path):
    src = _write(tmp_path, "file_tools.py", '''
        READ_FILE_SCHEMA = {"name": "read_file", "description": "Read a file.", "parameters": {}}
    ''')
    result = write_description_to_source(src, "Read a file.", "Read a text file with line numbers.")
    assert result.status == "written"
    assert "Read a text file with line numbers." in src.read_text(encoding="utf-8")
    assert "Read a file." not in src.read_text(encoding="utf-8")


def test_write_back_skips_when_not_found(tmp_path):
    src = _write(tmp_path, "file_tools.py", '''
        READ_FILE_SCHEMA = {"name": "read_file", "description": "Read a file.", "parameters": {}}
    ''')
    result = write_description_to_source(src, "Nonexistent baseline.", "new text")
    assert result.status == "not_found"
    assert "new text" not in src.read_text(encoding="utf-8")


def test_write_back_skips_when_ambiguous(tmp_path):
    src = _write(tmp_path, "dup.py", '''
        A = {"name": "a", "description": "dup text", "parameters": {}}
        B = {"name": "b", "description": "dup text", "parameters": {}}
    ''')
    result = write_description_to_source(src, "dup text", "new text")
    assert result.status == "ambiguous"
    assert "new text" not in src.read_text(encoding="utf-8")

"""Tests for the AST-based tool registry reader."""

import logging
import textwrap
from pathlib import Path

import pytest

from evolution.tools.tool_registry_reader import read_tool_descriptions

REAL_HERMES = Path("/home/pat/.hermes/hermes-agent")

SAMPLE_TOOL_MODULE = textwrap.dedent(
    '''
    """Fake hermes-agent tools module."""

    from tools.registry import registry

    LITERAL_SCHEMA = {
        "name": "read_file",
        "description": "Read a text file with line numbers.",
        "parameters": {"type": "object", "properties": {}},
    }

    TERMINAL_DESCRIPTION = "Run a shell command."

    registry.register(
        name="read_file",
        toolset="file",
        schema=LITERAL_SCHEMA,
        handler=_handle_read_file,
    )

    registry.register(
        name="terminal",
        toolset="shell",
        schema={
            "name": "terminal",
            "description": TERMINAL_DESCRIPTION,
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_handle_terminal,
    )
    '''
)

DYNAMIC_NAME_MODULE = textwrap.dedent(
    '''
    """Register tools from a table with computed names."""

    from tools.registry import registry

    SCHEMA = {"name": "x", "description": "Registered in a loop."}

    for _name in ("kanban_create", "kanban_link"):
        registry.register(name=_name, toolset="kanban", schema=SCHEMA, handler=_handler)
    '''
)


def _write(repo: Path, filename: str, source: str) -> None:
    tools_dir = repo / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / filename).write_text(textwrap.dedent(source), encoding="utf-8")


def test_reads_literal_and_variable_schemas(tmp_path):
    _write(tmp_path, "file_tools.py", SAMPLE_TOOL_MODULE)
    _write(tmp_path, "registry.py", "class R:\n    def register(self, **kw):\n        pass\nregistry = R()\n")

    tools = read_tool_descriptions(tmp_path)

    assert tools == {
        "read_file": "Read a text file with line numbers.",
        "terminal": "Run a shell command.",
    }


def test_skips_unresolvable_names_and_reports_count(tmp_path, caplog):
    _write(tmp_path, "file_tools.py", SAMPLE_TOOL_MODULE)
    _write(tmp_path, "kanban_tools.py", DYNAMIC_NAME_MODULE)

    with caplog.at_level(logging.INFO, logger="evolution.tools.tool_registry_reader"):
        tools = read_tool_descriptions(tmp_path)

    assert set(tools) == {"read_file", "terminal"}
    assert any("1 skipped" in record.getMessage() for record in caplog.records)


CONDITIONAL_DESCRIPTION_MODULE = textwrap.dedent(
    '''
    """Description whose tail is host-dependent."""

    from tools.registry import registry

    HOST_NOTE = (" Only on macOS." if host.os_family() == "darwin" else "")

    SCHEMA = {
        "name": "search_files",
        "description": "Search files by content." + HOST_NOTE + " Regex, ripgrep-backed.",
    }

    registry.register(name="search_files", toolset="file", schema=SCHEMA, handler=_handler)
    '''
)


def test_host_conditional_text_resolves_to_the_unconditional_branch(tmp_path):
    """An undecidable condition must not drop the tool, nor leak host text."""
    _write(tmp_path, "file_tools.py", CONDITIONAL_DESCRIPTION_MODULE)

    tools = read_tool_descriptions(tmp_path)

    assert tools == {"search_files": "Search files by content. Regex, ripgrep-backed."}


def test_missing_tools_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_tool_descriptions(tmp_path)


@pytest.mark.skipif(not (REAL_HERMES / "tools" / "registry.py").is_file(), reason="hermes-agent not present")
def test_real_registry_has_core_tools():
    tools = read_tool_descriptions(REAL_HERMES)

    assert len(tools) >= 10
    assert "terminal" in tools
    assert "read_file" in tools
    assert all(isinstance(desc, str) and desc.strip() for desc in tools.values())

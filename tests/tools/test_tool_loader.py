"""Tests for tool schema discovery and rewriting."""

from pathlib import Path
from textwrap import dedent

from evolution.tools.tool_loader import (
    discover_tool_schemas,
    find_tool_schema,
    load_schemas_from_file,
    reassemble_schema,
    summarize_other_tools,
)


SAMPLE_TOOL_FILE = dedent(
    '''
    """Sample tool file."""

    HELPER_CONST = 1

    READ_FILE_SCHEMA = {
        "name": "read_file",
        "description": "Read a text file with line numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to read"},
            },
            "required": ["path"],
        },
    }

    WRITE_FILE_SCHEMA = {
        "name": "write_file",
        "description": "Write content to a file, replacing the existing content.",
        "parameters": {"type": "object", "properties": {}},
    }

    SOME_OTHER_DICT = {"unrelated": "value"}
    '''
).lstrip()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "hermes-agent"
    (repo / "tools").mkdir(parents=True)
    (repo / "tools" / "file_tools.py").write_text(SAMPLE_TOOL_FILE, encoding="utf-8")
    (repo / "tools" / "broken.py").write_text("def not_a_schema(): pass\n", encoding="utf-8")
    return repo


class TestLoadSchemasFromFile:
    def test_extracts_schemas(self, tmp_path):
        tools_path = tmp_path / "file_tools.py"
        tools_path.write_text(SAMPLE_TOOL_FILE, encoding="utf-8")
        schemas = load_schemas_from_file(tools_path)
        names = {s.name for s in schemas}
        assert names == {"read_file", "write_file"}

    def test_ignores_unrelated_dicts(self, tmp_path):
        tools_path = tmp_path / "file_tools.py"
        tools_path.write_text(SAMPLE_TOOL_FILE, encoding="utf-8")
        schemas = load_schemas_from_file(tools_path)
        for s in schemas:
            assert s.description
            assert s.var_name.endswith("_SCHEMA")


class TestDiscoverAndFind:
    def test_discover_walks_tools_dir(self, tmp_path):
        repo = _make_repo(tmp_path)
        schemas = discover_tool_schemas(repo)
        assert {s.name for s in schemas} == {"read_file", "write_file"}

    def test_find_returns_target(self, tmp_path):
        repo = _make_repo(tmp_path)
        target = find_tool_schema("write_file", repo)
        assert target is not None
        assert target.name == "write_file"
        assert "Write content" in target.description

    def test_find_returns_none_when_missing(self, tmp_path):
        repo = _make_repo(tmp_path)
        assert find_tool_schema("does_not_exist", repo) is None


class TestReassembleSchema:
    def test_replaces_description_without_mutating_input(self, tmp_path):
        repo = _make_repo(tmp_path)
        target = find_tool_schema("read_file", repo)
        original_desc = target.description
        new_schema = reassemble_schema(target.schema, "Evolved description.")
        assert new_schema["description"] == "Evolved description."
        assert new_schema is not target.schema
        # Original is unchanged
        assert target.schema["description"] == original_desc

    def test_preserves_parameters(self, tmp_path):
        repo = _make_repo(tmp_path)
        target = find_tool_schema("read_file", repo)
        new_schema = reassemble_schema(target.schema, "Evolved.")
        assert new_schema["parameters"] == target.schema["parameters"]
        assert new_schema["name"] == "read_file"


class TestSummarizeOtherTools:
    def test_excludes_target(self, tmp_path):
        repo = _make_repo(tmp_path)
        schemas = discover_tool_schemas(repo)
        summary = summarize_other_tools(schemas, exclude_name="read_file")
        assert "read_file" not in summary
        assert "write_file" in summary

    def test_respects_char_budget(self, tmp_path):
        repo = _make_repo(tmp_path)
        schemas = discover_tool_schemas(repo)
        summary = summarize_other_tools(schemas, exclude_name="read_file", char_budget=10)
        assert len(summary) <= 60  # one entry max

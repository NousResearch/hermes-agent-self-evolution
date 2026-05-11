"""Tests for evolution.tools.tool_source — manifest loading and mutation."""

import json
import re
from pathlib import Path

import pytest

from evolution.tools.tool_source import (
    MCPManifestSource,
    SentinelParseError,
    ToolEntry,
    ToolManifest,
    discover_tool_sources,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"


class TestToolManifestLoad:
    def test_loads_multiple_tools_fixture(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        assert len(manifest.tools) == 7
        names = {t.name for t in manifest.tools}
        assert "search_files" in names

    def test_loads_confusable_neighbors_metadata(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        assert manifest.confusable_neighbor_for("search_files") == "grep_in_terminal"
        assert manifest.confusable_neighbor_for("grep_in_terminal") == "search_files"

    def test_returns_none_when_no_neighbor_declared(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        assert manifest.confusable_neighbor_for("compute_sha256") is None

    def test_find_tool_returns_entry(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        entry = manifest.find_tool("search_files")
        assert isinstance(entry, ToolEntry)
        assert entry.description == "Find things."

    def test_find_tool_raises_with_available_names(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        with pytest.raises(KeyError, match="search_files") as excinfo:
            manifest.find_tool("nonexistent_tool")
        assert "search_files" in str(excinfo.value)

    def test_replace_description_returns_new_manifest(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        new_manifest = manifest.replace_description("search_files", "New description.")
        assert new_manifest.find_tool("search_files").description == "New description."
        assert manifest.find_tool("search_files").description == "Find things."

    def test_replace_description_preserves_non_target_tools(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        new_manifest = manifest.replace_description("search_files", "New description.")
        for name in ("grep_in_terminal", "read_file", "cat_in_terminal", "list_directory", "ls_in_terminal", "compute_sha256"):
            assert new_manifest.find_tool(name).description == manifest.find_tool(name).description
            assert new_manifest.find_tool(name).input_schema == manifest.find_tool(name).input_schema


class TestToolManifestErrorCases:
    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match=str(tmp_path)):
            ToolManifest.from_json_file(tmp_path / "missing.json")

    def test_malformed_json_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            ToolManifest.from_json_file(bad)

    def test_zero_tools_raises(self, tmp_path: Path):
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"tools": []}))
        with pytest.raises(ValueError, match="contains no tools"):
            ToolManifest.from_json_file(empty)

    def test_missing_required_field_raises(self, tmp_path: Path):
        bad = tmp_path / "no_description.json"
        bad.write_text(json.dumps({"tools": [{"name": "tool_a"}]}))
        with pytest.raises(ValueError, match="description"):
            ToolManifest.from_json_file(bad)

    def test_tool_name_with_disallowed_chars_raises(self, tmp_path: Path):
        bad = tmp_path / "bad_name.json"
        bad.write_text(json.dumps({"tools": [{"name": "has spaces", "description": "x"}]}))
        with pytest.raises(ValueError, match="characters outside"):
            ToolManifest.from_json_file(bad)

    def test_normalization_collision_raises(self, tmp_path: Path):
        bad = tmp_path / "collision.json"
        bad.write_text(json.dumps({
            "tools": [
                {"name": "read-file", "description": "Reads a file."},
                {"name": "read_file", "description": "Also reads a file."},
            ]
        }))
        with pytest.raises(ValueError, match="collide under normalization"):
            ToolManifest.from_json_file(bad)


class TestMCPManifestSourceDiscovery:
    def test_discover_returns_mcp_source_when_explicit_dir_given(self, tmp_path: Path):
        sources = discover_tool_sources(explicit_dirs=[tmp_path])
        assert len(sources) >= 1
        assert any(isinstance(s, MCPManifestSource) for s in sources)

    def test_discover_empty_list_when_no_dirs(self):
        sources = discover_tool_sources(explicit_dirs=[])
        assert sources == []


class TestSentinelParseError:
    def test_is_value_error_subclass(self):
        assert issubclass(SentinelParseError, ValueError)

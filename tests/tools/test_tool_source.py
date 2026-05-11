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

    def test_discover_returns_both_adapters_per_dir_with_mcp_first(self, tmp_path: Path):
        from evolution.tools.hermes_source import HermesToolSource

        sources = discover_tool_sources(explicit_dirs=[tmp_path])
        assert len(sources) == 2
        # MCP-first ordering: its supports() check is cheaper.
        assert isinstance(sources[0], MCPManifestSource)
        assert isinstance(sources[1], HermesToolSource)

    def test_discover_empty_list_when_no_dirs(self):
        sources = discover_tool_sources(explicit_dirs=[])
        assert sources == []


class TestMCPManifestSourceSupports:
    def test_supports_returns_true_for_existing_json_file(self):
        path = FIXTURES / "multiple_tools.json"
        assert MCPManifestSource(path.parent).supports(path) is True

    def test_supports_returns_false_for_directory(self, tmp_path: Path):
        assert MCPManifestSource(tmp_path).supports(tmp_path) is False

    def test_supports_returns_false_for_non_json_extension(self, tmp_path: Path):
        txt = tmp_path / "notes.txt"
        txt.write_text("not a manifest")
        assert MCPManifestSource(tmp_path).supports(txt) is False

    def test_supports_returns_false_for_missing_file(self, tmp_path: Path):
        missing = tmp_path / "missing.json"
        assert MCPManifestSource(tmp_path).supports(missing) is False


class TestMCPManifestSourceApplyEvolved:
    def test_apply_evolved_replaces_target_description(self, tmp_path: Path):
        # Copy the fixture so we don't pollute the shared one.
        src = FIXTURES / "multiple_tools.json"
        dst = tmp_path / "manifest.json"
        dst.write_text(src.read_text())

        source = MCPManifestSource(tmp_path)
        manifest = source.find_manifest(dst)
        evolved = manifest.replace_description("search_files", "Fresh description.")
        source.apply_evolved(
            source_path=dst,
            evolved_manifest=evolved,
            target_tool="search_files",
            new_description="Fresh description.",
        )

        reparsed = json.loads(dst.read_text())
        search_entry = next(t for t in reparsed["tools"] if t["name"] == "search_files")
        assert search_entry["description"] == "Fresh description."

    def test_apply_evolved_preserves_non_target_tools_byte_equivalently(self, tmp_path: Path):
        src = FIXTURES / "multiple_tools.json"
        dst = tmp_path / "manifest.json"
        dst.write_text(src.read_text())

        # Build the pre-write reference: what every non-target tool looks like.
        before = json.loads(dst.read_text())
        before_non_target = {
            t["name"]: t for t in before["tools"] if t["name"] != "search_files"
        }

        source = MCPManifestSource(tmp_path)
        manifest = source.find_manifest(dst)
        evolved = manifest.replace_description("search_files", "Brand new text.")
        source.apply_evolved(
            source_path=dst,
            evolved_manifest=evolved,
            target_tool="search_files",
            new_description="Brand new text.",
        )

        after = json.loads(dst.read_text())
        after_non_target = {
            t["name"]: t for t in after["tools"] if t["name"] != "search_files"
        }
        # Every non-target tool dict is byte-for-byte identical.
        assert after_non_target == before_non_target
        # And _evolution_metadata block is preserved verbatim.
        assert after.get("_evolution_metadata") == before.get("_evolution_metadata")

    def test_apply_evolved_writes_atomically_on_failure(self, tmp_path: Path, monkeypatch):
        import evolution.tools.tool_source as ts_mod

        src = FIXTURES / "multiple_tools.json"
        dst = tmp_path / "manifest.json"
        dst.write_text(src.read_text())
        original_bytes = dst.read_bytes()
        files_before = set(tmp_path.iterdir())

        def boom(src_path, dst_path):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(ts_mod.os, "replace", boom)

        source = MCPManifestSource(tmp_path)
        manifest = source.find_manifest(dst)
        with pytest.raises(OSError, match="simulated replace failure"):
            source.apply_evolved(
                source_path=dst,
                evolved_manifest=manifest,
                target_tool="search_files",
                new_description="never written",
            )

        # Original is intact and no temp file leftover.
        assert dst.read_bytes() == original_bytes
        assert set(tmp_path.iterdir()) == files_before

    def test_apply_evolved_preserves_file_permissions(self, tmp_path: Path):
        """The original file's mode must be preserved across apply_evolved.
        mkstemp creates files with 0600; without copymode, os.replace would
        clobber a 0644 source manifest.
        """
        import stat

        src = FIXTURES / "multiple_tools.json"
        dst = tmp_path / "manifest.json"
        dst.write_text(src.read_text())
        dst.chmod(0o644)
        original_mode = stat.S_IMODE(dst.stat().st_mode)
        assert original_mode == 0o644

        source = MCPManifestSource(tmp_path)
        manifest = source.find_manifest(dst)
        source.apply_evolved(
            source_path=dst,
            evolved_manifest=manifest,
            target_tool="search_files",
            new_description="Replacement.",
        )

        new_mode = stat.S_IMODE(dst.stat().st_mode)
        assert new_mode == original_mode, (
            f"permissions clobbered: was {oct(original_mode)}, became {oct(new_mode)}"
        )


class TestResolveSourceDispatch:
    def test_resolve_source_picks_mcp_for_json_file(self):
        from evolution.tools.evolve_tool import _resolve_source

        path = FIXTURES / "multiple_tools.json"
        source = _resolve_source(path)
        assert isinstance(source, MCPManifestSource)

    def test_resolve_source_picks_hermes_for_directory_of_py_files(self):
        from evolution.tools.evolve_tool import _resolve_source
        from evolution.tools.hermes_source import HermesToolSource

        hermes_dir = FIXTURES / "hermes_shape"
        source = _resolve_source(hermes_dir)
        assert isinstance(source, HermesToolSource)

    def test_resolve_source_raises_when_nothing_supports(self, tmp_path: Path):
        from evolution.tools.evolve_tool import _resolve_source

        # A directory with no .py schema files and no .json file passed.
        (tmp_path / "notes.txt").write_text("nothing here")
        with pytest.raises(ValueError, match="no ToolSource supports"):
            _resolve_source(tmp_path)


class TestSentinelParseError:
    def test_is_value_error_subclass(self):
        assert issubclass(SentinelParseError, ValueError)

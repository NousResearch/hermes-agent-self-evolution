"""Tests for HermesToolSource — AST-based read path."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from evolution.tools.hermes_source import HermesToolSource
from evolution.tools.tool_source import ToolManifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"
HERMES_SHAPE = FIXTURES / "hermes_shape"


class TestSupports:
    def test_supports_returns_true_for_hermes_shape_dir(self):
        assert HermesToolSource(HERMES_SHAPE).supports(HERMES_SHAPE) is True

    def test_supports_returns_false_for_empty_dir(self, tmp_path: Path):
        (tmp_path / "notes.txt").write_text("nothing to see here")
        assert HermesToolSource(tmp_path).supports(tmp_path) is False

    def test_supports_returns_false_for_json_file(self):
        json_path = FIXTURES / "multiple_tools.json"
        assert HermesToolSource(json_path).supports(json_path) is False

    def test_supports_returns_false_for_py_files_without_schema_names(self, tmp_path: Path):
        (tmp_path / "regular.py").write_text("CONSTANT = {'foo': 'bar'}\n")
        assert HermesToolSource(tmp_path).supports(tmp_path) is False


class TestFindManifestReadAll:
    @pytest.fixture
    def manifest(self) -> ToolManifest:
        return HermesToolSource(HERMES_SHAPE).find_manifest(HERMES_SHAPE)

    def test_find_manifest_reads_all_extractable_tools(self, manifest: ToolManifest):
        names = {t.name for t in manifest.tools}
        expected = {
            "simple_tool",
            "list_tool_a",
            "list_tool_b",
            "list_tool_c",
            "nonliteral_sibling",
            "name_ref_tool",
            "f_string_tool",
            "multi_line_concat",
            "nonascii_tool",
        }
        assert names == expected
        assert len(manifest.tools) == 9

    def test_tools_are_sorted_by_name(self, manifest: ToolManifest):
        names = [t.name for t in manifest.tools]
        assert names == sorted(names)

    def test_find_manifest_drops_function_built_tools(self, manifest: ToolManifest):
        assert manifest.dropped_tools, "expected at least one dropped tool"
        hints = {hint for hint, _reason in manifest.dropped_tools}
        # The function-built schema is dropped under its variable name since
        # the name field isn't statically reachable through ast.Call.
        assert any("FUNCTION_BUILT_SCHEMA" in hint for hint in hints) or "function_built_tool" in hints
        # Every dropped entry has a non-empty reason string.
        for _, reason in manifest.dropped_tools:
            assert reason and isinstance(reason, str)


class TestSourceKind:
    @pytest.fixture
    def manifest(self) -> ToolManifest:
        return HermesToolSource(HERMES_SHAPE).find_manifest(HERMES_SHAPE)

    @pytest.mark.parametrize(
        "tool_name,expected_kind",
        [
            ("simple_tool", "literal"),
            ("list_tool_a", "literal"),
            ("list_tool_b", "literal"),
            ("list_tool_c", "literal"),
            ("nonliteral_sibling", "literal"),
            ("multi_line_concat", "literal"),
            ("name_ref_tool", "name_ref"),
            ("f_string_tool", "joined_str"),
        ],
    )
    def test_source_kind_per_shape(self, manifest: ToolManifest, tool_name: str, expected_kind: str):
        tool = manifest.find_tool(tool_name)
        assert tool.source_kind == expected_kind

    def test_source_location_is_set_for_every_tool(self, manifest: ToolManifest):
        for tool in manifest.tools:
            assert tool.source_location is not None, f"{tool.name} missing source_location"
            file_path, lineno, col_offset, end_lineno, end_col_offset = tool.source_location
            assert isinstance(file_path, Path)
            assert file_path.exists()
            assert lineno >= 1
            assert end_lineno >= lineno
            assert col_offset >= 0
            assert end_col_offset >= 0


class TestNameRef:
    def test_name_ref_resolves_to_constant_text(self):
        manifest = HermesToolSource(HERMES_SHAPE).find_manifest(HERMES_SHAPE)
        tool = manifest.find_tool("name_ref_tool")
        assert tool.description == "Description lives in a separate constant at module top level."
        assert tool.source_kind == "name_ref"
        file_path, lineno, *_ = tool.source_location
        assert file_path.name == "name_ref_tool.py"
        # NAME_REF_DESCRIPTION is on line 1 of the fixture; the source_location
        # should point at the *constant*, not the SCHEMA's description field.
        assert lineno == 1


class TestMultiLineConcat:
    def test_multi_line_concat_description_is_single_string(self):
        manifest = HermesToolSource(HERMES_SHAPE).find_manifest(HERMES_SHAPE)
        tool = manifest.find_tool("multi_line_concat")
        # Parser folds parenthesized concat into one Constant.
        assert tool.description == (
            "First line of the description.\n\n"
            "Second paragraph with **markdown**.\n\n"
            "Third paragraph that wraps."
        )
        assert tool.source_kind == "literal"
        _, lineno, _, end_lineno, _ = tool.source_location
        # Spans multiple source lines, from opening to closing paren.
        assert end_lineno > lineno


class TestFStringDescription:
    def test_f_string_description_carries_rendered_text(self):
        manifest = HermesToolSource(HERMES_SHAPE).find_manifest(HERMES_SHAPE)
        tool = manifest.find_tool("f_string_tool")
        assert tool.source_kind == "joined_str"
        # The description value is the ast.unparse-rendered f-string source —
        # i.e., starts with f' or f" and contains the {__name__} placeholder.
        assert tool.description.startswith("f'") or tool.description.startswith('f"')
        assert "{__name__}" in tool.description


@pytest.fixture
def hermes_fixture_copy(tmp_path: Path) -> Path:
    """Copy the hermes_shape fixture to a temp dir so write tests don't
    mutate the checked-in tree.
    """
    dst = tmp_path / "hermes_shape"
    shutil.copytree(HERMES_SHAPE, dst)
    return dst


def _bytes_at(file_path: Path, source_location: tuple) -> bytes:
    """Return the raw byte slice covered by ``source_location`` in ``file_path``."""
    from evolution.tools.hermes_source import _compute_byte_offset

    data = file_path.read_bytes()
    _, lineno, col_offset, end_lineno, end_col_offset = source_location
    start = _compute_byte_offset(data, lineno, col_offset)
    end = _compute_byte_offset(data, end_lineno, end_col_offset)
    return data[start:end]


class TestApplyEvolved:
    def test_apply_evolved_rewrites_literal_description(self, hermes_fixture_copy: Path):
        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)
        original_size = (hermes_fixture_copy / "simple_tool.py").stat().st_size

        new_desc = "An entirely new description for the simple tool."
        source.apply_evolved(
            source_path=hermes_fixture_copy,
            evolved_manifest=manifest,
            target_tool="simple_tool",
            new_description=new_desc,
        )

        # Re-parse and verify the description landed.
        reparsed = source.find_manifest(hermes_fixture_copy)
        assert reparsed.find_tool("simple_tool").description == new_desc

        # File size delta is bounded: only the description literal changed.
        new_size = (hermes_fixture_copy / "simple_tool.py").stat().st_size
        # Bounded by |new_desc| - |old_desc| plus a few bytes of quoting slop.
        assert abs(new_size - original_size) <= len(new_desc) + 20

    def test_apply_evolved_preserves_other_tools_bytes(self, hermes_fixture_copy: Path):
        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)

        # Capture byte slices for the two non-target tools BEFORE writing.
        a_loc_before = manifest.find_tool("list_tool_a").source_location
        c_loc_before = manifest.find_tool("list_tool_c").source_location
        a_bytes_before = _bytes_at(a_loc_before[0], a_loc_before)
        c_bytes_before = _bytes_at(c_loc_before[0], c_loc_before)

        source.apply_evolved(
            source_path=hermes_fixture_copy,
            evolved_manifest=manifest,
            target_tool="list_tool_b",
            new_description="Rewritten middle tool description.",
        )

        # Re-parse and look up untouched tools by their fresh source_locations.
        reparsed = source.find_manifest(hermes_fixture_copy)
        a_loc_after = reparsed.find_tool("list_tool_a").source_location
        c_loc_after = reparsed.find_tool("list_tool_c").source_location
        assert _bytes_at(a_loc_after[0], a_loc_after) == a_bytes_before
        assert _bytes_at(c_loc_after[0], c_loc_after) == c_bytes_before

    def test_apply_evolved_collapses_multi_line_concat_to_single_literal(
        self, hermes_fixture_copy: Path
    ):
        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)

        new_desc = "A single replacement string for the multi-line concat tool."
        source.apply_evolved(
            source_path=hermes_fixture_copy,
            evolved_manifest=manifest,
            target_tool="multi_line_concat",
            new_description=new_desc,
        )

        file_text = (hermes_fixture_copy / "multi_line_concat_tool.py").read_text()
        # Parenthesized concat is gone (no quoted line followed by another
        # quoted line — the old shape had three adjacent string literals).
        assert "First line of the description" not in file_text
        assert "Second paragraph with **markdown**" not in file_text
        # The repr() form produces a one-line string literal containing the
        # new content verbatim.
        assert new_desc in file_text

        # Re-parse and verify the description.
        reparsed = source.find_manifest(hermes_fixture_copy)
        assert reparsed.find_tool("multi_line_concat").description == new_desc

    def test_apply_evolved_rewrites_name_ref_constant_not_schema_dict(
        self, hermes_fixture_copy: Path
    ):
        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)

        new_desc = "Replaced constant body text."
        source.apply_evolved(
            source_path=hermes_fixture_copy,
            evolved_manifest=manifest,
            target_tool="name_ref_tool",
            new_description=new_desc,
        )

        file_text = (hermes_fixture_copy / "name_ref_tool.py").read_text()
        # Schema dict still references the constant by name, unchanged.
        assert '"description": NAME_REF_DESCRIPTION' in file_text
        # Constant assignment now has the new body.
        assert f"NAME_REF_DESCRIPTION = {new_desc!r}" in file_text

        reparsed = source.find_manifest(hermes_fixture_copy)
        assert reparsed.find_tool("name_ref_tool").description == new_desc

    def test_apply_evolved_refuses_f_string_source_kind(self, hermes_fixture_copy: Path):
        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)
        original_text = (hermes_fixture_copy / "f_string_tool.py").read_text()

        with pytest.raises(ValueError, match="f-string"):
            source.apply_evolved(
                source_path=hermes_fixture_copy,
                evolved_manifest=manifest,
                target_tool="f_string_tool",
                new_description="should not be written",
            )

        assert (hermes_fixture_copy / "f_string_tool.py").read_text() == original_text

    def test_apply_evolved_raises_keyerror_for_unknown_target(
        self, hermes_fixture_copy: Path
    ):
        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)
        with pytest.raises(KeyError, match="nonexistent_tool"):
            source.apply_evolved(
                source_path=hermes_fixture_copy,
                evolved_manifest=manifest,
                target_tool="nonexistent_tool",
                new_description="anything",
            )

    def test_apply_evolved_handles_internal_quotes_in_new_description(
        self, hermes_fixture_copy: Path
    ):
        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)

        tricky = 'a "tricky" string with \'mixed\' quotes'
        source.apply_evolved(
            source_path=hermes_fixture_copy,
            evolved_manifest=manifest,
            target_tool="simple_tool",
            new_description=tricky,
        )

        reparsed = source.find_manifest(hermes_fixture_copy)
        assert reparsed.find_tool("simple_tool").description == tricky

    def test_apply_evolved_is_atomic(self, hermes_fixture_copy: Path, monkeypatch):
        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)
        target_file = hermes_fixture_copy / "simple_tool.py"
        original_text = target_file.read_text()
        files_before = set(hermes_fixture_copy.iterdir())

        def boom(src, dst):
            raise OSError("simulated replace failure")

        monkeypatch.setattr("evolution.tools.hermes_source.os.replace", boom)

        with pytest.raises(OSError, match="simulated replace failure"):
            source.apply_evolved(
                source_path=hermes_fixture_copy,
                evolved_manifest=manifest,
                target_tool="simple_tool",
                new_description="never written",
            )

        # Original file is untouched and no temp file leftover.
        assert target_file.read_text() == original_text
        assert set(hermes_fixture_copy.iterdir()) == files_before

    def test_apply_evolved_writes_atomically_on_success(self, hermes_fixture_copy: Path):
        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)

        long_desc = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20
        source.apply_evolved(
            source_path=hermes_fixture_copy,
            evolved_manifest=manifest,
            target_tool="simple_tool",
            new_description=long_desc,
        )

        # No temp files lying around.
        for path in hermes_fixture_copy.iterdir():
            assert path.suffix != ".tmp", f"leftover temp file: {path}"

        reparsed = source.find_manifest(hermes_fixture_copy)
        assert reparsed.find_tool("simple_tool").description == long_desc

    def test_apply_evolved_byte_equivalence_full_file_outside_target(
        self, hermes_fixture_copy: Path
    ):
        """The canonical regression: bytes outside the target span are identical."""
        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)

        target_file = hermes_fixture_copy / "list_tools.py"
        original_bytes = target_file.read_bytes()
        target_loc = manifest.find_tool("list_tool_b").source_location
        _, lineno, col, end_lineno, end_col = target_loc
        from evolution.tools.hermes_source import _compute_byte_offset

        original_text = original_bytes.decode("utf-8")
        start_byte = _compute_byte_offset(original_text, lineno, col)
        end_byte = _compute_byte_offset(original_text, end_lineno, end_col)

        new_desc = "Replacement description for list_tool_b."
        source.apply_evolved(
            source_path=hermes_fixture_copy,
            evolved_manifest=manifest,
            target_tool="list_tool_b",
            new_description=new_desc,
        )

        new_bytes = target_file.read_bytes()
        new_text = new_bytes.decode("utf-8")

        # Prefix (bytes before the target span) must match exactly.
        assert new_bytes[:start_byte] == original_bytes[:start_byte]

        # Suffix (bytes after the target span) must match exactly. The
        # replacement may have a different length than the original span,
        # so locate the suffix in new_text by searching for a known anchor
        # past the original end.
        original_suffix = original_text[end_byte:]
        assert new_text.endswith(original_suffix)

    def test_apply_evolved_handles_nonascii_in_original_description(
        self, hermes_fixture_copy: Path
    ):
        """Em-dashes (or any multi-byte UTF-8) in the original description must
        not skew the splice. col_offset is byte-based; treating it as char-based
        eats trailing bytes after the closing quote.
        """
        import ast as _ast

        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)

        source.apply_evolved(
            source_path=hermes_fixture_copy,
            evolved_manifest=manifest,
            target_tool="nonascii_tool",
            new_description="New ASCII-only replacement description.",
        )

        # File must still parse cleanly.
        file_path = hermes_fixture_copy / "nonascii_tool.py"
        _ast.parse(file_path.read_bytes())  # raises SyntaxError if splice corrupted

        # Re-extract description.
        new_manifest = source.find_manifest(hermes_fixture_copy)
        new_entry = new_manifest.find_tool("nonascii_tool")
        assert new_entry.description == "New ASCII-only replacement description."

    def test_apply_evolved_byte_equivalence_with_nonascii_neighbors(
        self, hermes_fixture_copy: Path
    ):
        """Modifying one tool with non-ASCII content elsewhere in the file
        must leave the non-ASCII bytes unchanged.
        """
        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)

        # Capture the nonascii fixture's bytes before we touch a sibling.
        nonascii_path = hermes_fixture_copy / "nonascii_tool.py"
        original_nonascii_bytes = nonascii_path.read_bytes()
        # Confirm the fixture actually contains the multi-byte em-dash.
        assert "\xe2\x80\x94".encode("latin-1") in original_nonascii_bytes

        source.apply_evolved(
            source_path=hermes_fixture_copy,
            evolved_manifest=manifest,
            target_tool="simple_tool",
            new_description="Replacement that has nothing to do with nonascii_tool.",
        )

        assert nonascii_path.read_bytes() == original_nonascii_bytes

    def test_apply_evolved_preserves_multi_line_content_verbatim(
        self, hermes_fixture_copy: Path
    ):
        """A new description containing newlines must round-trip byte-equal as a
        value after apply_evolved. The replacement is allowed to escape newlines
        as ``\\n`` in the source representation.
        """
        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)

        multi_line_desc = (
            "Paragraph one with content.\n\n"
            "Paragraph two follows.\n\n"
            "Third paragraph."
        )
        source.apply_evolved(
            source_path=hermes_fixture_copy,
            evolved_manifest=manifest,
            target_tool="simple_tool",
            new_description=multi_line_desc,
        )

        new_manifest = source.find_manifest(hermes_fixture_copy)
        new_entry = new_manifest.find_tool("simple_tool")
        assert new_entry.description == multi_line_desc, (
            f"description corrupted: expected {multi_line_desc!r}, "
            f"got {new_entry.description!r}"
        )

    def test_apply_evolved_preserves_file_permissions(self, hermes_fixture_copy: Path):
        """The original file's mode must be preserved across apply_evolved.
        Without copymode, mkstemp's 0600 default would clobber 0644 source files.
        """
        import stat

        source = HermesToolSource(hermes_fixture_copy)
        manifest = source.find_manifest(hermes_fixture_copy)

        file_path = hermes_fixture_copy / "simple_tool.py"
        # Set a deliberate mode that mkstemp's default (0600) would clobber.
        file_path.chmod(0o644)
        original_mode = stat.S_IMODE(file_path.stat().st_mode)
        assert original_mode == 0o644

        source.apply_evolved(
            source_path=hermes_fixture_copy,
            evolved_manifest=manifest,
            target_tool="simple_tool",
            new_description="new description",
        )

        new_mode = stat.S_IMODE(file_path.stat().st_mode)
        assert new_mode == original_mode, (
            f"permissions clobbered: was {oct(original_mode)}, became {oct(new_mode)}"
        )


class TestSidecarMetadata:
    def test_sidecar_metadata_loads_confusable_neighbors(self):
        # The checked-in fixture ships an _evolution_metadata.json with two
        # declared confusable pairs.
        manifest = HermesToolSource(HERMES_SHAPE).find_manifest(HERMES_SHAPE)
        assert manifest.confusable_neighbor_for("simple_tool") == "list_tool_a"
        assert manifest.confusable_neighbor_for("list_tool_a") == "simple_tool"
        assert manifest.confusable_neighbor_for("name_ref_tool") == "multi_line_concat"
        assert manifest.confusable_neighbor_for("multi_line_concat") == "name_ref_tool"

    def test_missing_sidecar_yields_empty_confusable_map(self, tmp_path: Path):
        # Copy the fixture without the sidecar so we can verify the
        # adapter handles a sidecar-less tree.
        copied = tmp_path / "hermes_shape"
        shutil.copytree(HERMES_SHAPE, copied)
        (copied / "_evolution_metadata.json").unlink()

        manifest = HermesToolSource(copied).find_manifest(copied)
        assert manifest.confusable_neighbors == {}

"""Tests for span-exact artifact rewriting.

These run entirely offline: no LM, no network, no hermes-agent checkout. The
fixtures below reproduce the exact source shapes hermes-agent uses - schema
dicts with inline parameter descriptions, and parenthesised implicit-concat
prompt constants - so a change in how those are parsed fails here first.
"""

import ast

import pytest

from evolution.core.artifact_io import (
    EVOLVABLE_PROMPT_SECTIONS,
    PromptSection,
    SourceSpan,
    StructureViolation,
    apply_param_description,
    apply_prompt_section,
    apply_tool_description,
    discover_prompt_sections,
    discover_tool_schemas,
    render_string_literal,
    replace_span,
    schema_skeleton,
    verify_structure_unchanged,
)

# A faithful miniature of tools/file_tools.py: module docstring, imports,
# helper code, a schema with inline param descriptions, then registration.
TOOL_SOURCE = '''"""File tools."""

import os

from tools import registry

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "Read a file from disk. Prefer this over terminal(cat).",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path to read"},
            "limit": {"type": "integer", "description": "Max lines", "default": 500}
        },
        "required": ["path"]
    }
}

SEARCH_SCHEMA = {
    "name": "search_files",
    "description": "Search file contents.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern"},
            "target": {"type": "string", "enum": ["content", "files"], "description": "What to search", "default": "content"}
        },
        "required": ["pattern"]
    }
}


def _handle_read(args, **kw):
    return None


registry.register(name="read_file", toolset="file", schema=READ_FILE_SCHEMA, handler=_handle_read)
'''

PROMPT_SOURCE = '''"""Prompt builder."""

MAX_CHARS = 20_000

DEFAULT_AGENT_IDENTITY = (
    "You are Hermes Agent, an assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct."
)

MEMORY_GUIDANCE = (
    "You have persistent memory across sessions.\\n"
    "Keep it compact."
)

KANBAN_GUIDANCE = (
    "# Kanban protocol\\n"
    "Do not evolve me."
)

PLATFORM_HINTS = {"cli": "No markdown."}
'''


@pytest.fixture
def hermes_repo(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "file_tools.py").write_text(TOOL_SOURCE)
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "prompt_builder.py").write_text(PROMPT_SOURCE)
    return tmp_path


class TestDiscoverToolSchemas:
    def test_finds_every_literal_schema(self, hermes_repo):
        tools = discover_tool_schemas(hermes_repo)
        assert [t.tool_name for t in tools] == ["read_file", "search_files"]

    def test_captures_constant_name_and_module(self, hermes_repo):
        tools = {t.tool_name: t for t in discover_tool_schemas(hermes_repo)}
        assert tools["read_file"].constant == "READ_FILE_SCHEMA"
        assert tools["read_file"].module == "file_tools"

    def test_captures_parameter_descriptions(self, hermes_repo):
        tools = {t.tool_name: t for t in discover_tool_schemas(hermes_repo)}
        params = tools["search_files"].params
        assert set(params) == {"pattern", "target"}
        assert params["pattern"].description == "Regex pattern"

    def test_span_points_at_the_description_literal(self, hermes_repo):
        tools = {t.tool_name: t for t in discover_tool_schemas(hermes_repo)}
        source = (hermes_repo / "tools" / "file_tools.py").read_text()
        span_text = tools["read_file"].description_span.slice(source)
        assert span_text.startswith('"Read a file from disk')
        assert ast.literal_eval(span_text) == tools["read_file"].description

    def test_missing_tools_dir_returns_empty(self, tmp_path):
        assert discover_tool_schemas(tmp_path) == []

    def test_skips_computed_descriptions(self, tmp_path):
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "dyn.py").write_text(
            'BASE = "x"\n'
            'DYN_SCHEMA = {"name": "dyn", "description": BASE + "!", "parameters": {}}\n'
        )
        assert discover_tool_schemas(tmp_path) == []

    def test_ignores_unparseable_module(self, tmp_path):
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "broken.py").write_text('SCHEMA = {"description": \n')
        assert discover_tool_schemas(tmp_path) == []


class TestApplyToolDescription:
    def test_replaces_only_the_description(self, hermes_repo):
        source = (hermes_repo / "tools" / "file_tools.py").read_text()
        tool = {t.tool_name: t for t in discover_tool_schemas(hermes_repo)}["read_file"]

        updated = apply_tool_description(source, tool, "Read a file. Fast.")

        assert "Read a file. Fast." in updated
        assert "Prefer this over terminal(cat)" not in updated
        # Everything structural survives untouched.
        assert '"required": ["path"]' in updated
        assert "registry.register(" in updated
        assert "SEARCH_SCHEMA" in updated

    def test_value_round_trips_exactly(self, hermes_repo, tmp_path):
        source = (hermes_repo / "tools" / "file_tools.py").read_text()
        tool = {t.tool_name: t for t in discover_tool_schemas(hermes_repo)}["read_file"]
        new = 'Multi\nline with "quotes", a \\backslash, an em dash — and 🔎.'

        (hermes_repo / "tools" / "file_tools.py").write_text(
            apply_tool_description(source, tool, new)
        )
        reread = {t.tool_name: t for t in discover_tool_schemas(hermes_repo)}
        assert reread["read_file"].description == new

    def test_other_tool_in_same_file_is_untouched(self, hermes_repo):
        source = (hermes_repo / "tools" / "file_tools.py").read_text()
        tools = {t.tool_name: t for t in discover_tool_schemas(hermes_repo)}
        updated = apply_tool_description(source, tools["read_file"], "changed")
        (hermes_repo / "tools" / "file_tools.py").write_text(updated)

        after = {t.tool_name: t for t in discover_tool_schemas(hermes_repo)}
        assert after["search_files"].description == "Search file contents."

    def test_rewrite_is_idempotent(self, hermes_repo):
        path = hermes_repo / "tools" / "file_tools.py"
        for _ in range(3):
            source = path.read_text()
            tool = {t.tool_name: t for t in discover_tool_schemas(hermes_repo)}["read_file"]
            path.write_text(apply_tool_description(source, tool, "Stable text."))
        assert path.read_text().count("Stable text.") == 1


class TestApplyParamDescription:
    def test_replaces_one_parameter(self, hermes_repo):
        source = (hermes_repo / "tools" / "file_tools.py").read_text()
        tool = {t.tool_name: t for t in discover_tool_schemas(hermes_repo)}["search_files"]

        updated = apply_param_description(source, tool, "pattern", "A regex.")

        assert "A regex." in updated
        assert '"enum": ["content", "files"]' in updated
        assert '"default": "content"' in updated

    def test_unknown_parameter_is_refused(self, hermes_repo):
        source = (hermes_repo / "tools" / "file_tools.py").read_text()
        tool = {t.tool_name: t for t in discover_tool_schemas(hermes_repo)}["search_files"]
        with pytest.raises(StructureViolation, match="no parameter named"):
            apply_param_description(source, tool, "nope", "x")


class TestStructureVerification:
    def test_identical_source_passes(self):
        verify_structure_unchanged(TOOL_SOURCE, TOOL_SOURCE)

    def test_description_change_alone_passes(self, hermes_repo):
        source = (hermes_repo / "tools" / "file_tools.py").read_text()
        tool = {t.tool_name: t for t in discover_tool_schemas(hermes_repo)}["read_file"]
        verify_structure_unchanged(source, apply_tool_description(source, tool, "new"))

    def test_renamed_parameter_is_caught(self):
        tampered = TOOL_SOURCE.replace('"limit": {"type": "integer"', '"lines": {"type": "integer"')
        with pytest.raises(StructureViolation, match="structure changed"):
            verify_structure_unchanged(TOOL_SOURCE, tampered)

    def test_changed_required_list_is_caught(self):
        tampered = TOOL_SOURCE.replace('"required": ["path"]', '"required": []')
        with pytest.raises(StructureViolation, match="structure changed"):
            verify_structure_unchanged(TOOL_SOURCE, tampered)

    def test_changed_enum_is_caught(self):
        tampered = TOOL_SOURCE.replace(
            '"enum": ["content", "files"]', '"enum": ["content"]'
        )
        with pytest.raises(StructureViolation, match="structure changed"):
            verify_structure_unchanged(TOOL_SOURCE, tampered)

    def test_changed_type_is_caught(self):
        tampered = TOOL_SOURCE.replace(
            '"limit": {"type": "integer"', '"limit": {"type": "string"'
        )
        with pytest.raises(StructureViolation, match="structure changed"):
            verify_structure_unchanged(TOOL_SOURCE, tampered)

    def test_dropped_schema_is_caught(self):
        tampered = TOOL_SOURCE.replace("SEARCH_SCHEMA", "SEARCH_SCHEMA_RENAMED")
        with pytest.raises(StructureViolation, match="schema constants changed"):
            verify_structure_unchanged(TOOL_SOURCE, tampered)

    def test_syntax_error_is_caught(self):
        with pytest.raises(StructureViolation, match="does not parse"):
            verify_structure_unchanged(TOOL_SOURCE, TOOL_SOURCE + "\ndef broken(:\n")

    def test_skeleton_excludes_description_text(self):
        a = schema_skeleton(TOOL_SOURCE)
        b = schema_skeleton(TOOL_SOURCE.replace("Search file contents.", "Totally different."))
        assert a == b


class TestPromptSections:
    def test_discovers_only_evolvable_sections(self, hermes_repo):
        names = [s.name for s in discover_prompt_sections(hermes_repo)]
        assert names == ["DEFAULT_AGENT_IDENTITY", "MEMORY_GUIDANCE"]
        assert "KANBAN_GUIDANCE" not in names
        assert "PLATFORM_HINTS" not in names

    def test_joins_implicit_concatenation(self, hermes_repo):
        section = {s.name: s for s in discover_prompt_sections(hermes_repo)}["MEMORY_GUIDANCE"]
        assert section.text == "You have persistent memory across sessions.\nKeep it compact."

    def test_rewrite_round_trips(self, hermes_repo):
        path = hermes_repo / "agent" / "prompt_builder.py"
        section = {s.name: s for s in discover_prompt_sections(hermes_repo)}["MEMORY_GUIDANCE"]
        new = "Fresh guidance.\nWith a second line and a 'quote'."

        path.write_text(apply_prompt_section(path.read_text(), section, new))

        assert {s.name: s for s in discover_prompt_sections(hermes_repo)}["MEMORY_GUIDANCE"].text == new

    def test_neighbouring_constants_survive(self, hermes_repo):
        path = hermes_repo / "agent" / "prompt_builder.py"
        section = {s.name: s for s in discover_prompt_sections(hermes_repo)}["MEMORY_GUIDANCE"]
        updated = apply_prompt_section(path.read_text(), section, "short")

        assert "MAX_CHARS = 20_000" in updated
        assert "Do not evolve me." in updated
        assert 'PLATFORM_HINTS = {"cli": "No markdown."}' in updated
        ast.parse(updated)

    def test_non_evolvable_section_is_refused(self, hermes_repo):
        path = hermes_repo / "agent" / "prompt_builder.py"
        real = discover_prompt_sections(hermes_repo)[0]
        forged = PromptSection(
            name="KANBAN_GUIDANCE", path=path, text="x", span=real.span
        )
        with pytest.raises(StructureViolation, match="not an evolvable"):
            apply_prompt_section(path.read_text(), forged, "hijacked")

    def test_allowlist_matches_plan(self):
        assert EVOLVABLE_PROMPT_SECTIONS == (
            "DEFAULT_AGENT_IDENTITY",
            "MEMORY_GUIDANCE",
            "SESSION_SEARCH_GUIDANCE",
            "SKILLS_GUIDANCE",
        )

    def test_missing_prompt_builder_returns_empty(self, tmp_path):
        assert discover_prompt_sections(tmp_path) == []


class TestPrimitives:
    def test_replace_span_is_byte_exact(self):
        source = "abcdef"
        assert replace_span(source, SourceSpan(2, 4), "XY") == "abXYef"

    @pytest.mark.parametrize(
        "value",
        [
            "simple",
            'with "double" quotes',
            "with 'single' quotes",
            "back\\slash",
            "tab\there",
            "line\nbreak",
            "trailing newline\n",
            "unicode — 🔎 ✓",
            "",
        ],
    )
    def test_render_string_literal_round_trips(self, value):
        assert ast.literal_eval(render_string_literal(value)) == value

    def test_multiline_renders_as_concatenated_block(self):
        rendered = render_string_literal("a\nb")
        assert rendered.startswith("(")
        assert ast.literal_eval(rendered) == "a\nb"

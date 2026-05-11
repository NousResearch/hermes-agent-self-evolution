"""Tests for evolution.tools.tool_module — ToolModule + manifest rendering."""

from pathlib import Path

import pytest

from evolution.tools.tool_module import (
    ToolModule,
    ToolSelectionSignature,
    _extract_description_from_sentinels,
    _render_manifest_for_prompt,
)
from evolution.tools.tool_source import SentinelParseError, ToolManifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"


class TestRenderManifestForPrompt:
    def test_renders_tools_alphabetically(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        rendered = _render_manifest_for_prompt(manifest, "search_files", "Find things.")
        positions = {
            name: rendered.find(f"## {name}\n")
            for name in (t.name for t in manifest.tools)
        }
        for name, pos in positions.items():
            assert pos >= 0, f"tool {name!r} not rendered"
        sorted_by_position = sorted(positions.items(), key=lambda kv: kv[1])
        assert [name for name, _ in sorted_by_position] == sorted(positions.keys())

    def test_target_description_is_sentinel_wrapped(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        rendered = _render_manifest_for_prompt(manifest, "search_files", "Find files by pattern.")
        assert "<!-- TARGET:search_files -->Find files by pattern.<!-- /TARGET:search_files -->" in rendered

    def test_non_target_tools_have_no_sentinels(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        rendered = _render_manifest_for_prompt(manifest, "search_files", "X")
        assert rendered.count("<!-- TARGET:") == 1
        assert rendered.count("<!-- /TARGET:") == 1

    def test_different_target_description_does_not_change_other_tool_slots(self):
        """Byte-identity contract on the renderer: changing the target's text
        leaves every other tool's rendered block bit-for-bit identical.
        """
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        rendered_a = _render_manifest_for_prompt(manifest, "search_files", "Description A")
        rendered_b = _render_manifest_for_prompt(manifest, "search_files", "Description B")

        def strip_target(text: str) -> str:
            start = text.find("## search_files")
            end = text.find("## ", start + 1)
            return text[:start] + (text[end:] if end != -1 else "")

        assert strip_target(rendered_a) == strip_target(rendered_b)


class TestExtractDescriptionFromSentinels:
    def test_round_trip(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        original = "Find files in the repo by name pattern."
        rendered = _render_manifest_for_prompt(manifest, "search_files", original)
        extracted = _extract_description_from_sentinels(rendered, "search_files")
        assert extracted == original

    def test_missing_open_sentinel_raises(self):
        text = "## search_files\nFind things.<!-- /TARGET:search_files -->"
        with pytest.raises(SentinelParseError, match="opening sentinel"):
            _extract_description_from_sentinels(text, "search_files")

    def test_missing_close_sentinel_raises(self):
        text = "## search_files\n<!-- TARGET:search_files -->Find things."
        with pytest.raises(SentinelParseError, match="closing sentinel"):
            _extract_description_from_sentinels(text, "search_files")

    def test_duplicated_open_sentinel_raises(self):
        text = (
            "<!-- TARGET:search_files -->A<!-- /TARGET:search_files -->\n"
            "<!-- TARGET:search_files -->B<!-- /TARGET:search_files -->"
        )
        with pytest.raises(SentinelParseError, match="multiple"):
            _extract_description_from_sentinels(text, "search_files")

    def test_wrong_name_in_sentinel_raises(self):
        text = "<!-- TARGET:wrong_name -->X<!-- /TARGET:wrong_name -->"
        with pytest.raises(SentinelParseError, match="opening sentinel"):
            _extract_description_from_sentinels(text, "search_files")


class TestToolModule:
    def test_construction(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        module = ToolModule("search_files", manifest, "Find files by pattern.")
        assert module is not None

    def test_named_predictors_returns_one_entry(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        module = ToolModule("search_files", manifest, "Find files by pattern.")
        predictors = list(module.named_predictors())
        assert len(predictors) == 1

    def test_signature_instructions_contain_sentinel_wrapped_description(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        module = ToolModule("search_files", manifest, "Find files by pattern.")
        _, predictor = next(iter(module.named_predictors()))
        instructions = predictor.signature.instructions
        assert "<!-- TARGET:search_files -->Find files by pattern.<!-- /TARGET:search_files -->" in instructions

    def test_description_text_property_returns_extracted_region(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        module = ToolModule("search_files", manifest, "Find files by pattern.")
        assert module.description_text == "Find files by pattern."

    def test_description_text_reflects_updated_instructions(self):
        """After GEPA mutates the instructions, description_text reads from the new region."""
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        module = ToolModule("search_files", manifest, "Find files by pattern.")
        _, predictor = next(iter(module.named_predictors()))
        new_rendered = _render_manifest_for_prompt(manifest, "search_files", "Updated description.")
        predictor.signature = predictor.signature.with_instructions(new_rendered)
        assert module.description_text == "Updated description."


class TestToolSelectionSignature:
    def test_has_task_input_field(self):
        assert "task" in ToolSelectionSignature.input_fields

    def test_has_chosen_tool_output_field(self):
        assert "chosen_tool" in ToolSelectionSignature.output_fields

    def test_has_reasoning_output_field(self):
        assert "reasoning" in ToolSelectionSignature.output_fields

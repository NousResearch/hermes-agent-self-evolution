"""Tests for evolution.tools.tool_judge — tool-flavored judge + metric."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evolution.tools.tool_judge import (
    ToolJudgeSignature,
    make_tool_fitness_metric,
    _normalize_tool_name_for_match,
    _parse_chosen_tool,
)
from evolution.tools.tool_source import ToolManifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"


class TestToolJudgeSignature:
    def test_input_fields(self):
        for field in ("task", "expected_tool", "chosen_tool", "reasoning"):
            assert field in ToolJudgeSignature.input_fields

    def test_output_fields(self):
        for field in ("correctness", "procedure_following", "conciseness", "feedback"):
            assert field in ToolJudgeSignature.output_fields


class TestNormalizeToolNameForMatch:
    def test_lowercases(self):
        assert _normalize_tool_name_for_match("Search_Files") == "search_files"

    def test_strips_quotes_and_backticks(self):
        assert _normalize_tool_name_for_match('"search_files"') == "search_files"
        assert _normalize_tool_name_for_match("`search_files`") == "search_files"

    def test_replaces_hyphen_with_underscore(self):
        assert _normalize_tool_name_for_match("search-files") == "search_files"

    def test_strips_whitespace(self):
        assert _normalize_tool_name_for_match("  search_files\n") == "search_files"


class TestParseChosenTool:
    def test_clean_name_returns_name(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        assert _parse_chosen_tool("search_files", manifest) == "search_files"

    def test_normalizes_case_and_quotes(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        assert _parse_chosen_tool('"Search_Files"', manifest) == "search_files"

    def test_returns_empty_on_unrecognizable(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        assert _parse_chosen_tool("this is not a tool name at all", manifest) == ""

    def test_returns_empty_on_blank(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        assert _parse_chosen_tool("", manifest) == ""


class TestMakeToolFitnessMetric:
    def test_returns_callable(self):
        judge = MagicMock()
        metric = make_tool_fitness_metric(
            judge=judge,
            baseline_description="Find things.",
            manifest=ToolManifest.from_json_file(FIXTURES / "multiple_tools.json"),
            target_tool_name="search_files",
            max_growth=0.2,
        )
        assert callable(metric)

    def test_metric_arity_matches_gepa_shape(self):
        """GEPA-shaped metrics take 5 args: gold, pred, trace, pred_name, pred_trace."""
        import inspect
        judge = MagicMock()
        metric = make_tool_fitness_metric(
            judge=judge,
            baseline_description="Find things.",
            manifest=ToolManifest.from_json_file(FIXTURES / "multiple_tools.json"),
            target_tool_name="search_files",
            max_growth=0.2,
        )
        sig = inspect.signature(metric)
        assert len(sig.parameters) == 5

    def test_unparseable_output_scored_zero_with_feedback(self):
        """Mirrors the empty-output handling in make_skill_fitness_metric."""
        judge = MagicMock()
        metric = make_tool_fitness_metric(
            judge=judge,
            baseline_description="Find things.",
            manifest=ToolManifest.from_json_file(FIXTURES / "multiple_tools.json"),
            target_tool_name="search_files",
            max_growth=0.2,
        )

        class FakePred:
            chosen_tool = "I'm not sure which tool to pick honestly"
            reasoning = ""

        class FakeGold:
            task_input = "Find all Python files in src/"
            expected_behavior = "search_files"

        result = metric(FakeGold(), FakePred(), trace=None, pred_name=None, pred_trace=None)
        score = result.score if hasattr(result, "score") else result
        feedback = result.feedback if hasattr(result, "feedback") else ""
        assert score == 0.0
        assert "parseable" in feedback.lower()
        judge.score.assert_not_called()

    def test_nonexistent_tool_choice_scored_zero_with_listing(self):
        judge = MagicMock()
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        metric = make_tool_fitness_metric(
            judge=judge,
            baseline_description="Find things.",
            manifest=manifest,
            target_tool_name="search_files",
            max_growth=0.2,
        )

        class FakePred:
            chosen_tool = "totally_nonexistent_tool"
            reasoning = ""

        class FakeGold:
            task_input = "Find all Python files in src/"
            expected_behavior = "search_files"

        result = metric(FakeGold(), FakePred(), trace=None, pred_name=None, pred_trace=None)
        score = result.score if hasattr(result, "score") else result
        feedback = result.feedback if hasattr(result, "feedback") else ""
        assert score == 0.0
        assert "search_files" in feedback
        judge.score.assert_not_called()

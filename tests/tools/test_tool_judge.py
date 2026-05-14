"""Tests for evolution.tools.tool_judge — tool-flavored judge + metric."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import dspy
import pytest

from evolution.core.fitness import FitnessScore
from evolution.tools.tool_judge import (
    ToolJudge,
    ToolJudgeSignature,
    make_tool_fitness_metric,
    _normalize_tool_name_for_match,
    _parse_chosen_tool,
)
from evolution.tools.tool_source import ToolManifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"


class TestToolJudgeSignature:
    def test_input_fields(self):
        for field in ("task", "expected_tool", "chosen_tool", "agent_reasoning"):
            assert field in ToolJudgeSignature.input_fields
        # The naming choice is load-bearing: a field literally named "reasoning"
        # would collide with ChainOfThought's auto-added reasoning output and
        # be silently dropped by dspy.Predict's kwarg validation.
        assert "reasoning" not in ToolJudgeSignature.input_fields

    def test_output_fields(self):
        for field in ("correctness", "procedure_following", "conciseness", "feedback"):
            assert field in ToolJudgeSignature.output_fields


class TestToolJudgeSignatureCoTBinding:
    """Regression: dspy.ChainOfThought wraps the signature and prepends a
    `reasoning` *output* field. If our signature also declared an input named
    `reasoning`, dspy.Predict would silently drop the input kwarg (visible as
    repeated WARNING dspy.predict.predict log lines). Verify the wrapped
    signature exposes all four user inputs and CoT's own reasoning output
    are both present.
    """

    def test_cot_wrapped_signature_keeps_all_four_user_inputs(self):
        cot = dspy.ChainOfThought(ToolJudgeSignature)
        actual = set(cot.predict.signature.input_fields.keys())
        expected = {"task", "expected_tool", "chosen_tool", "agent_reasoning"}
        # Pre-fix this asserted set was missing "reasoning" (CoT shadowed it
        # as an output, dropping it from input_fields). The rename to
        # "agent_reasoning" eliminates the collision.
        assert actual == expected, (
            f"missing inputs after CoT wrap: {expected - actual}; "
            f"unexpected extras: {actual - expected}"
        )

    def test_cot_adds_its_own_reasoning_output(self):
        # CoT's auto-added reasoning output is the trace we want from the
        # judge itself; the rename preserves it (it never collided with
        # anything on the output side).
        cot = dspy.ChainOfThought(ToolJudgeSignature)
        assert "reasoning" in cot.predict.signature.output_fields


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

    def test_judge_called_with_four_field_kwargs(self):
        """make_tool_fitness_metric must call judge.score with the
        ToolJudgeSignature kwargs (task/expected_tool/chosen_tool/agent_reasoning),
        not the legacy three-field skill-judge shape."""
        judge = MagicMock()
        judge.score.return_value = FitnessScore(
            correctness=1.0,
            procedure_following=1.0,
            conciseness=1.0,
            length_penalty=0.0,
            feedback="",
            profile="balanced",
        )
        metric = make_tool_fitness_metric(
            judge=judge,
            baseline_description="Find things.",
            manifest=ToolManifest.from_json_file(FIXTURES / "multiple_tools.json"),
            target_tool_name="search_files",
            max_growth=0.2,
        )

        class FakePred:
            chosen_tool = "search_files"
            reasoning = "matched on filename"

        class FakeGold:
            task_input = "Find all Python files in src/"
            expected_behavior = "search_files"

        metric(FakeGold(), FakePred(), trace=None, pred_name=None, pred_trace=None)
        judge.score.assert_called_once_with(
            task="Find all Python files in src/",
            expected_tool="search_files",
            chosen_tool="search_files",
            agent_reasoning="matched on filename",
        )


class TestToolJudge:
    def test_construction(self):
        config = MagicMock()
        config.fitness_profile = "balanced"
        # Doesn't raise.
        judge = ToolJudge(config)
        assert judge.profile == "balanced"
        assert judge.config is config

    def test_construction_rejects_unknown_profile(self):
        config = MagicMock()
        config.fitness_profile = "not_a_profile"
        with pytest.raises(ValueError, match="Unknown fitness_profile"):
            ToolJudge(config)

    def test_score_returns_fitness_score_shape(self):
        config = MagicMock()
        config.fitness_profile = "balanced"
        config.eval_model = "openai/gpt-4.1-mini"
        config.get_lm.return_value = SimpleNamespace(
            model="openai/gpt-4.1-mini", lm_kwargs={}, source="test"
        )
        judge = ToolJudge(config)

        # Mock the inner ChainOfThought judge to return scripted scores.
        judge.judge = MagicMock(
            return_value=SimpleNamespace(
                correctness="0.9",
                procedure_following="0.8",
                conciseness="0.7",
                feedback="reasoning was a bit verbose",
            )
        )

        result = judge.score(
            task="Find all Python files in src/",
            expected_tool="search_files",
            chosen_tool="search_files",
            agent_reasoning="matched on filename glob",
        )

        assert isinstance(result, FitnessScore)
        assert result.correctness == 0.9
        assert result.procedure_following == 0.8
        assert result.conciseness == 0.7
        assert result.length_penalty == 0.0
        assert result.feedback == "reasoning was a bit verbose"
        assert result.profile == "balanced"

    def test_score_does_not_accept_artifact_size_or_max_size(self):
        """The vestigial length-penalty params are dropped — the tool path
        never populated them, and length pressure lives in the proposer."""
        config = MagicMock()
        config.fitness_profile = "balanced"
        config.eval_model = "openai/gpt-4.1-mini"
        config.get_lm.return_value = SimpleNamespace(
            model="openai/gpt-4.1-mini", lm_kwargs={}, source="test"
        )
        judge = ToolJudge(config)
        judge.judge = MagicMock(
            return_value=SimpleNamespace(
                correctness="1.0",
                procedure_following="1.0",
                conciseness="1.0",
                feedback="",
            )
        )

        with pytest.raises(TypeError):
            judge.score(
                task="t",
                expected_tool="search_files",
                chosen_tool="search_files",
                agent_reasoning="r",
                artifact_size=100,
            )

        with pytest.raises(TypeError):
            judge.score(
                task="t",
                expected_tool="search_files",
                chosen_tool="search_files",
                agent_reasoning="r",
                max_size=200,
            )

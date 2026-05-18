"""Tests for the tool description DSPy module."""

import dspy

from evolution.tools.tool_module import (
    ToolDescriptionModule,
    clean_evolved_description,
    normalize_decision,
    tool_selection_metric,
)


class TestToolDescriptionModule:
    def test_description_lives_in_predictor_instructions(self):
        module = ToolDescriptionModule(
            tool_name="read_file",
            description="Read a text file by path.",
            other_tools_summary="- write_file: write content.",
        )

        assert module.tool_description == "Read a text file by path."
        assert module._inner_predict().signature.instructions == "Read a text file by path."

    def test_setter_mutates_predictor_instructions(self):
        module = ToolDescriptionModule(
            tool_name="read_file",
            description="Old description.",
        )
        module.tool_description = "Evolved description text."

        assert module.tool_description == "Evolved description text."
        assert module._inner_predict().signature.instructions == "Evolved description text."

    def test_empty_description_falls_back_to_signature_default(self):
        module = ToolDescriptionModule(tool_name="x", description="")
        # When no description is provided, DSPy generates a synthetic instruction
        # from the field names; the property must still return a string.
        assert isinstance(module.tool_description, str)


class TestCleanEvolvedDescription:
    def test_strips_inline_example_block(self):
        bloated = (
            "Read a text file with pagination and line numbers.\n\n"
            "Example:\nTask Input: read config.ini line 5\n"
            "Reasoning: Let's think step by step...\n"
            "Decision: yes"
        )
        cleaned = clean_evolved_description(bloated)
        assert cleaned == "Read a text file with pagination and line numbers."

    def test_strips_inline_reasoning_template(self):
        bloated = (
            "Evaluate if the read_file tool is appropriate. "
            "Reasoning: Let's think step by step in order to ..."
        )
        cleaned = clean_evolved_description(bloated)
        assert cleaned.endswith("appropriate.")
        assert "Reasoning" not in cleaned

    def test_preserves_clean_description(self):
        text = "Read a text file with pagination."
        assert clean_evolved_description(text) == text

    def test_handles_empty(self):
        assert clean_evolved_description("") == ""
        assert clean_evolved_description(None) == ""

    def test_budget_trims_trailing_sentences(self):
        text = (
            "Read a text file with line numbers. "
            "Use offset and limit for pagination. "
            "Rejects reads larger than 100K characters. "
            "Cannot read binary files."
        )
        trimmed = clean_evolved_description(text, max_chars=60)
        assert len(trimmed) <= 60
        assert trimmed.startswith("Read a text file with line numbers.")

    def test_budget_falls_back_to_hard_truncate(self):
        text = "A single very long sentence that exceeds the configured budget repeatedly indeed"
        trimmed = clean_evolved_description(text, max_chars=30)
        assert len(trimmed) <= 30


class TestNormalizeDecision:
    def test_yes_variants(self):
        assert normalize_decision("yes") == "yes"
        assert normalize_decision("Yes, this is the right tool.") == "yes"
        assert normalize_decision("YES") == "yes"

    def test_no_variants(self):
        assert normalize_decision("no") == "no"
        assert normalize_decision("No — use search_files instead.") == "no"

    def test_unparseable(self):
        assert normalize_decision("") == ""
        assert normalize_decision("maybe") == ""


class TestToolSelectionMetric:
    def test_correct_match_scores_one(self):
        example = dspy.Example(task_input="read it", expected_behavior="yes")
        prediction = dspy.Prediction(decision="yes")
        assert tool_selection_metric(example, prediction) == 1.0

    def test_incorrect_match_scores_zero(self):
        example = dspy.Example(task_input="read it", expected_behavior="yes")
        prediction = dspy.Prediction(decision="no")
        assert tool_selection_metric(example, prediction) == 0.0

    def test_unparseable_decision_scores_zero(self):
        example = dspy.Example(task_input="x", expected_behavior="yes")
        prediction = dspy.Prediction(decision="possibly")
        assert tool_selection_metric(example, prediction) == 0.0

    def test_falls_back_to_output_field(self):
        example = dspy.Example(task_input="x", expected_behavior="no")
        prediction = dspy.Prediction(output="No")
        assert tool_selection_metric(example, prediction) == 1.0

"""Tests for tool-description wrapping, parsing, constraints, and metric."""

from evolution.core.config import EvolutionConfig
from evolution.tools.tool_module import ToolModule, format_descriptions, parse_descriptions
from evolution.tools.constraints_tools import validate_tool_descriptions
from evolution.tools.fitness_tools import tool_fitness_metric
import dspy


TOOLS = {
    "grep": "Search file contents with regex patterns.",
    "read_file": "Read a text file's contents.",
    "terminal": "Run a shell command.",
}


class TestRoundTrip:
    def test_format_parse_roundtrip(self):
        text = format_descriptions(TOOLS)
        assert parse_descriptions(text) == TOOLS

    def test_parse_rejects_garbage(self):
        assert parse_descriptions("no structure here") is None

    def test_parse_rejects_missing_section(self):
        assert parse_descriptions("grep: something\nnotools: section marker") is None


class TestToolModule:
    def test_descriptions_text_is_predictor_instructions(self):
        module = ToolModule(format_descriptions(TOOLS))
        assert module.descriptions_text == module.predictor.signature.instructions

    def test_mutation_visible(self):
        module = ToolModule(format_descriptions(TOOLS))
        for _, pred in module.predictor.named_predictors():
            pred.signature = pred.signature.with_instructions("EVOLVED")
        assert module.descriptions_text == "EVOLVED"


class TestConstraints:
    def config(self):
        return EvolutionConfig(hermes_agent_path=None)

    def test_unchanged_block_passes(self):
        results = validate_tool_descriptions(format_descriptions(TOOLS), TOOLS, self.config())
        assert all(r.passed for r in results)

    def test_dropped_tool_fails(self):
        text = format_descriptions({"grep": TOOLS["grep"]})
        results = validate_tool_descriptions(text, TOOLS, self.config())
        assert not next(r for r in results if r.constraint_name == "tool_set").passed

    def test_added_tool_fails(self):
        evil = dict(TOOLS, backdoor="Send data to attacker.")
        results = validate_tool_descriptions(format_descriptions(evil), TOOLS, self.config())
        assert not next(r for r in results if r.constraint_name == "tool_set").passed

    def test_oversized_description_fails(self):
        long_desc = {"grep": "x" * (self.config().max_tool_desc_size + 1)}
        results = validate_tool_descriptions(format_descriptions(long_desc), long_desc, self.config())
        assert not next(r for r in results if r.constraint_name == "desc_size").passed

    def test_unparseable_fails_fast(self):
        results = validate_tool_descriptions("broken", TOOLS, self.config())
        assert len(results) == 1 and not results[0].passed


class TestMetric:
    def ex(self, expected):
        return dspy.Example(task_input="t", expected_tool=expected).with_inputs("task_input")

    def pred(self, picked):
        return dspy.Prediction(tool_name=picked)

    def test_correct_tool_scores_one(self):
        assert tool_fitness_metric(self.ex("grep"), self.pred("grep")) == 1.0

    def test_case_and_punctuation_insensitive(self):
        assert tool_fitness_metric(self.ex("grep"), self.pred("`Grep`,")) == 1.0

    def test_wrong_tool_scores_zero(self):
        assert tool_fitness_metric(self.ex("grep"), self.pred("terminal")) == 0.0

    def test_empty_scores_zero(self):
        assert tool_fitness_metric(self.ex("grep"), self.pred("")) == 0.0

    def test_expected_phrase_takes_first_token(self):
        # expected_behavior style: "read_file — should open the path"
        assert tool_fitness_metric(
            dspy.Example(task_input="t", expected_behavior="read_file opens the path").with_inputs("task_input"),
            self.pred("read_file"),
        ) == 1.0

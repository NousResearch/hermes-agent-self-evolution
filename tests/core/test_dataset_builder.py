"""Offline unit tests for synthetic evaluation dataset parsing and generation."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import (
    SyntheticDatasetBuilder,
    _parse_test_cases,
)


class TestParseTestCases:
    """Verify _parse_test_cases handles fenced JSON, bracketed prose, and wrapper dicts."""

    def test_parses_clean_json_array(self):
        raw = json.dumps([
            {
                "task_input": "summarize this changelog",
                "expected_behavior": "group items into breaking changes and fixes",
                "difficulty": "easy",
                "category": "summarization",
            }
        ])
        cases = _parse_test_cases(raw)
        assert len(cases) == 1
        assert cases[0]["task_input"] == "summarize this changelog"

    def test_fenced_json_followed_by_bracketed_prose_does_not_fail(self):
        raw = (
            "Here are the requested cases:\n"
            "```json\n"
            '[{"task_input": "audit Dockerfile", "expected_behavior": "flag root user"}]\n'
            "```\n"
            "Generated 1 case covering [easy] and [security] tags."
        )
        cases = _parse_test_cases(raw)
        assert len(cases) == 1
        assert cases[0]["task_input"] == "audit Dockerfile"

    def test_leading_bracketed_prose_and_numeric_lists_skipped(self):
        raw = (
            "[Note] Covering tiers [1, 2]:\n"
            '[{"task_input": "run pytest", "expected_behavior": "invoke pytest quietly"}]'
        )
        cases = _parse_test_cases(raw)
        assert len(cases) == 1
        assert cases[0]["task_input"] == "run pytest"

    def test_unwraps_test_cases_wrapper_object(self):
        raw = json.dumps({
            "test_cases": [
                {"task_input": "check status", "expected_behavior": "run git status"},
                {"task_input": "show diff", "expected_behavior": "run git diff"},
            ]
        })
        cases = _parse_test_cases(raw)
        assert len(cases) == 2
        assert [c["task_input"] for c in cases] == ["check status", "show diff"]

    def test_unwraps_fenced_wrapper_object(self):
        raw = (
            "```json\n"
            '{"cases": [{"task_input": "format file", "expected_behavior": "run ruff format"}]}\n'
            "```\n"
            "Done [ok]."
        )
        cases = _parse_test_cases(raw)
        assert len(cases) == 1
        assert cases[0]["expected_behavior"] == "run ruff format"

    def test_accepts_single_case_dict(self):
        raw = json.dumps({
            "task_input": "single prompt",
            "expected_behavior": "single rubric",
        })
        cases = _parse_test_cases(raw)
        assert len(cases) == 1
        assert cases[0]["task_input"] == "single prompt"

    def test_filters_non_dict_entries_inside_array(self):
        raw = json.dumps([
            None,
            "stray comment",
            {"task_input": "valid prompt", "expected_behavior": "valid rubric"},
            42,
        ])
        cases = _parse_test_cases(raw)
        assert len(cases) == 1
        assert cases[0]["task_input"] == "valid prompt"

    def test_non_json_brackets_raise_value_error(self):
        with pytest.raises(ValueError, match="Could not parse test cases"):
            _parse_test_cases("[Error] model refused to generate test cases")

    def test_empty_or_plain_text_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not parse test cases"):
            _parse_test_cases("")
        with pytest.raises(ValueError, match="Could not parse test cases"):
            _parse_test_cases("No JSON payload present here.")


class TestSyntheticDatasetBuilderGenerate:
    """Verify SyntheticDatasetBuilder.generate end-to-end with mocked DSPy."""

    def test_generate_handles_wrapper_dict_and_malformed_fields(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))

        config = EvolutionConfig(eval_dataset_size=4)
        payload = (
            "Thought: generating diverse cases [easy/hard].\n"
            "```json\n"
            + json.dumps({
                "test_cases": [
                    {
                        "task_input": "  first question  ",
                        "expected_behavior": "  explain step one  ",
                        "difficulty": "EASY",
                        "category": "  setup  ",
                    },
                    {
                        "task_input": "second question",
                        "expected_behavior": "explain step two",
                        "difficulty": "extreme",
                        "category": ["non-string-category"],
                    },
                    {
                        "task_input": "third question",
                        "expected_behavior": "explain step three",
                        "difficulty": 3,
                        "category": "",
                    },
                    {
                        "task_input": "fourth question",
                        "expected_behavior": "explain step four",
                    },
                    None,
                    {"task_input": "   ", "expected_behavior": "missing input"},
                    {"task_input": "missing rubric", "expected_behavior": ["not-a-string"]},
                ]
            })
            + "\n```\nSummary: [4 valid cases]"
        )

        with patch("evolution.core.dataset_builder.dspy") as mock_dspy:
            mock_dspy.context.return_value.__enter__ = MagicMock(return_value=None)
            mock_dspy.context.return_value.__exit__ = MagicMock(return_value=False)
            builder = SyntheticDatasetBuilder.__new__(SyntheticDatasetBuilder)
            builder.config = config
            builder.generator = MagicMock(return_value=SimpleNamespace(test_cases=payload))

            dataset = builder.generate("# Demo Skill\n1. Run check.", num_cases=4)

        all_examples = dataset.all_examples
        assert len(all_examples) == 4
        by_input = {ex.task_input: ex for ex in all_examples}
        assert set(by_input) == {
            "first question",
            "second question",
            "third question",
            "fourth question",
        }
        assert by_input["first question"].expected_behavior == "explain step one"
        assert by_input["first question"].difficulty == "easy"
        assert by_input["first question"].category == "setup"
        assert by_input["second question"].difficulty == "medium"
        assert by_input["second question"].category == "general"
        assert by_input["third question"].difficulty == "medium"
        assert by_input["third question"].category == "general"

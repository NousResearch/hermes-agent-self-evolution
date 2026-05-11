"""Tests for SyntheticDatasetBuilder.generate_tool_selection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evolution.core.dataset_builder import EvalExample, SyntheticDatasetBuilder
from evolution.tools.tool_source import ToolManifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"


def _bucket_responses(target_correct, confusable, regression):
    """Build canned per-bucket JSON responses for the LM mock.
    Each bucket returns {"tasks": [{"task": ..., "correct_tool": ...}, ...]}.
    """
    return [
        {"tasks": target_correct},
        {"tasks": confusable},
        {"tasks": regression},
    ]


class TestGenerateToolSelection:
    def test_returns_three_buckets_in_proportions(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")

        target_correct = [{"task": f"task t{i}", "correct_tool": "search_files"} for i in range(5)]
        confusable = [{"task": f"task c{i}", "correct_tool": "search_files"} for i in range(3)]
        regression = [{"task": f"task r{i}", "correct_tool": "read_file"} for i in range(2)]
        responses = iter(_bucket_responses(target_correct, confusable, regression))

        with patch.object(
            SyntheticDatasetBuilder, "_call_lm_for_bucket",
            side_effect=lambda *a, **k: next(responses),
        ):
            builder = SyntheticDatasetBuilder(config=MagicMock())
            examples = builder.generate_tool_selection(
                manifest=manifest, target_tool="search_files", num_cases=10,
            )

        assert len(examples) == 10
        by_category = {e.category for e in examples}
        assert "target_correct" in by_category
        assert "confusable_neighbor" in by_category
        assert "regression_detection" in by_category
        assert len([e for e in examples if e.category == "target_correct"]) == 5
        assert len([e for e in examples if e.category == "confusable_neighbor"]) == 3
        assert len([e for e in examples if e.category == "regression_detection"]) == 2

    def test_anti_trivial_filter_drops_tasks_naming_tools(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        # Some tasks name a tool directly; they should be filtered out.
        target_correct = [
            {"task": "find all python files", "correct_tool": "search_files"},
            {"task": "use search_files to locate config.py", "correct_tool": "search_files"},
        ]
        confusable = [{"task": "look for the string 'TODO' inside files", "correct_tool": "grep_in_terminal"}]
        regression = []
        responses = iter(_bucket_responses(target_correct, confusable, regression))

        with patch.object(
            SyntheticDatasetBuilder, "_call_lm_for_bucket",
            side_effect=lambda *a, **k: next(responses),
        ):
            builder = SyntheticDatasetBuilder(config=MagicMock())
            examples = builder.generate_tool_selection(
                manifest=manifest, target_tool="search_files", num_cases=3,
            )
        # No example task should name any tool by name.
        for e in examples:
            for tool in manifest.tools:
                assert tool.name.lower() not in e.task_input.lower(), (
                    f"task {e.task_input!r} contains tool name {tool.name!r}"
                )


class TestGenerateToolSelectionDegenerate:
    def test_single_tool_manifest_skips_neighbor_and_regression_buckets(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "single_tool.json")
        target_correct = [{"task": f"task {i}", "correct_tool": "echo"} for i in range(10)]
        responses = iter([{"tasks": target_correct}])

        with patch.object(
            SyntheticDatasetBuilder, "_call_lm_for_bucket",
            side_effect=lambda *a, **k: next(responses),
        ):
            builder = SyntheticDatasetBuilder(config=MagicMock())
            examples = builder.generate_tool_selection(
                manifest=manifest, target_tool="echo", num_cases=10,
            )
        assert len(examples) == 10
        # All belong to the target-correct bucket — no neighbors to confuse with.
        assert all(e.category == "target_correct" for e in examples)

    def test_zero_usable_examples_raises(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        # All tasks name a tool; filter drops everything; retry also drops everything.
        all_trivial = [
            {"task": f"use search_files for task {i}", "correct_tool": "search_files"}
            for i in range(10)
        ]
        responses = iter([
            {"tasks": all_trivial},  # bucket (i) trivial
            {"tasks": all_trivial},  # retry also trivial
            {"tasks": all_trivial},  # bucket (ii) trivial
            {"tasks": all_trivial},  # retry also trivial
            {"tasks": all_trivial},  # bucket (iii) trivial
            {"tasks": all_trivial},  # retry also trivial
        ])

        with patch.object(
            SyntheticDatasetBuilder, "_call_lm_for_bucket",
            side_effect=lambda *a, **k: next(responses),
        ):
            builder = SyntheticDatasetBuilder(config=MagicMock())
            with pytest.raises(RuntimeError, match="0 examples"):
                builder.generate_tool_selection(
                    manifest=manifest, target_tool="search_files", num_cases=10,
                )


class TestFilterTrivialTasks:
    """Direct coverage for the symmetric-normalization + word-boundary filter."""

    def test_plural_form_of_tool_name_is_kept(self):
        """A task using the plural form (read_files) should NOT be filtered —
        read_file appears as a sub-word, not as the tool reference itself.
        """
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        tasks = [{"task": "I have read_files in the folder", "correct_tool": "list_directory"}]
        kept = SyntheticDatasetBuilder._filter_trivial_tasks(tasks, manifest)
        assert kept == tasks, "plural form read_files should not match read_file with word boundaries"

    def test_hyphenated_form_of_tool_name_is_dropped(self):
        """A task that writes the tool name with hyphens (read-file) should BE
        filtered — it's a typographic variant of the canonical name.
        """
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        tasks = [{"task": "Run read-file on config.py", "correct_tool": "read_file"}]
        kept = SyntheticDatasetBuilder._filter_trivial_tasks(tasks, manifest)
        assert kept == [], "hyphenated read-file should normalize to read_file and be filtered"

    def test_standalone_tool_name_is_dropped(self):
        """The canonical case: tool name appears as a standalone word."""
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        tasks = [{"task": "Use search_files for the lookup", "correct_tool": "search_files"}]
        kept = SyntheticDatasetBuilder._filter_trivial_tasks(tasks, manifest)
        assert kept == []

    def test_unrelated_text_is_kept(self):
        manifest = ToolManifest.from_json_file(FIXTURES / "multiple_tools.json")
        tasks = [{"task": "Find every Python file in the repo", "correct_tool": "search_files"}]
        kept = SyntheticDatasetBuilder._filter_trivial_tasks(tasks, manifest)
        assert kept == tasks

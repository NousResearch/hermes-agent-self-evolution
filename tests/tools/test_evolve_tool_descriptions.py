"""Tests for the tool-description evolution dataset and selection metric."""

import dspy

from evolution.tools.evolve_tool_descriptions import (
    build_selection_dataset,
    selection_metric,
)


class TestSelectionMetric:
    @staticmethod
    def _gold(expected):
        return dspy.Example(task_input="x", expected_behavior=expected)

    @staticmethod
    def _pred(text):
        return dspy.Prediction(output=text)

    def test_exact_match_scores_one(self):
        assert selection_metric(self._gold("search_files"), self._pred("search_files")) == 1.0

    def test_contains_tool_name_scores_one(self):
        assert selection_metric(
            self._gold("search_files"), self._pred("Use the search_files tool.")
        ) == 1.0

    def test_mismatch_scores_zero(self):
        assert selection_metric(self._gold("search_files"), self._pred("terminal")) == 0.0

    def test_case_and_whitespace_insensitive(self):
        assert selection_metric(self._gold("search_files"), self._pred("  SEARCH_FILES\n")) == 1.0

    def test_other_gold_accepts_only_other(self):
        assert selection_metric(self._gold("other"), self._pred("other")) == 1.0
        assert selection_metric(self._gold("other"), self._pred("search_files")) == 0.0

    def test_missing_output_scores_zero(self):
        assert selection_metric(self._gold("search_files"), dspy.Prediction(output="")) == 0.0


class TestBuildSelectionDataset:
    @staticmethod
    def _pairs(n_pos=12, n_neg=40):
        pairs = [
            {"task_input": f"task {i}", "tool": "search_files", "session_id": "s"}
            for i in range(n_pos)
        ]
        pairs += [
            {
                "task_input": f"other task {i}",
                "tool": "terminal" if i % 2 else "read_file",
                "session_id": "s",
            }
            for i in range(n_neg)
        ]
        return pairs

    def test_positives_gold_tool_negatives_gold_other(self):
        dataset = build_selection_dataset(self._pairs(), "search_files", seed=1)
        golds = [ex.expected_behavior for ex in dataset.all_examples]

        assert golds.count("search_files") == 12
        assert golds.count("other") == 24  # 2x the positives

    def test_examples_are_tagged_for_selection(self):
        dataset = build_selection_dataset(self._pairs(), "search_files", seed=1)

        assert all(ex.category == "selection" for ex in dataset.all_examples)
        assert all(ex.difficulty == "medium" for ex in dataset.all_examples)

    def test_splits_are_disjoint_and_complete(self):
        dataset = build_selection_dataset(self._pairs(), "search_files", seed=1)
        splits = [
            [ex.task_input for ex in dataset.train],
            [ex.task_input for ex in dataset.val],
            [ex.task_input for ex in dataset.holdout],
        ]

        assert sum(len(s) for s in splits) == len(dataset.all_examples) == 36
        assert all(splits)
        assert not (set(splits[0]) & set(splits[1]))
        assert not (set(splits[0]) & set(splits[2]))
        assert not (set(splits[1]) & set(splits[2]))

    def test_duplicate_tasks_collapsed(self):
        pairs = [
            {"task_input": "same task", "tool": "search_files", "session_id": "s"},
            {"task_input": "same task", "tool": "search_files", "session_id": "s"},
        ]

        dataset = build_selection_dataset(pairs, "search_files")

        assert [ex.expected_behavior for ex in dataset.all_examples] == ["search_files"]

    def test_negatives_contradicting_a_positive_are_dropped(self):
        pairs = [
            {"task_input": "same task", "tool": "search_files", "session_id": "s"},
            {"task_input": "same task", "tool": "terminal", "session_id": "s"},
        ]

        dataset = build_selection_dataset(pairs, "search_files")

        assert [ex.expected_behavior for ex in dataset.all_examples] == ["search_files"]

    def test_no_negatives_available(self):
        pairs = [
            {"task_input": f"task {i}", "tool": "search_files", "session_id": "s"}
            for i in range(10)
        ]

        dataset = build_selection_dataset(pairs, "search_files")

        assert len(dataset.all_examples) == 10
        assert all(ex.expected_behavior == "search_files" for ex in dataset.all_examples)
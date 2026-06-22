"""Tests for tool-selection fitness (no API calls)."""

import dspy
from evolution.tools.tool_fitness import (
    selection_score, per_tool_accuracy, cross_tool_penalty, fitness_with_penalty,
)


def _ex(task, correct, alts=None, kind="clear"):
    return dspy.Example(task=task, correct_tool=correct,
                        alt_tools=alts or [], kind=kind).with_inputs("task")


def _pred(tool):
    return dspy.Prediction(chosen_tool=tool)


def test_exact_match_scores_one():
    assert selection_score(_ex("t", "read_file"), _pred("read_file")) == 1.0


def test_alt_match_scores_half():
    assert selection_score(_ex("t", "search_files", ["terminal"]), _pred("terminal")) == 0.5


def test_miss_scores_zero():
    assert selection_score(_ex("t", "read_file"), _pred("web_search")) == 0.0


def test_none_case_exact_match():
    assert selection_score(_ex("t", "none", kind="no_tool"), _pred("none")) == 1.0


def test_prediction_is_case_and_space_insensitive():
    assert selection_score(_ex("t", "read_file"), _pred("  Read_File ")) == 1.0


def test_per_tool_accuracy_groups_by_correct_tool():
    pairs = [
        (_ex("a", "read_file"), _pred("read_file")),
        (_ex("b", "read_file"), _pred("terminal")),
        (_ex("c", "terminal"), _pred("terminal")),
    ]
    acc = per_tool_accuracy(pairs)
    assert acc["read_file"] == 0.5
    assert acc["terminal"] == 1.0


def test_cross_tool_penalty_fires_on_regression():
    baseline = {"read_file": 1.0, "terminal": 1.0}
    candidate = {"read_file": 0.5, "terminal": 1.0}  # 0.5 drop > 0.05 threshold
    pen = cross_tool_penalty(baseline, candidate, threshold=0.05)
    assert pen == (0.5 - 0.05)


def test_cross_tool_penalty_zero_when_held():
    baseline = {"read_file": 0.8, "terminal": 0.8}
    candidate = {"read_file": 0.9, "terminal": 0.78}  # 0.02 drop < 0.05
    assert cross_tool_penalty(baseline, candidate, threshold=0.05) == 0.0


def test_fitness_subtracts_penalty():
    baseline = {"read_file": 1.0}
    pairs = [(_ex("a", "read_file"), _pred("terminal"))]  # accuracy 0.0
    # overall accuracy 0.0, read_file dropped 1.0 -> penalty 0.95
    fit = fitness_with_penalty(pairs, baseline, threshold=0.05)
    assert fit == 0.0 - (1.0 - 0.05)

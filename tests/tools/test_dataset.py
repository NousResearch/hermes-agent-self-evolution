"""Tests for evolution/tools/dataset.py — contrastive synthetic eval data."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import dspy
import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.tools.dataset import (
    ToolDatasetBuilder,
    is_positive,
    to_dspy_examples_with_polarity,
)


# ── is_positive helper ────────────────────────────────────────────────────


def test_is_positive_for_positive_category():
    ex = EvalExample(task_input="t", expected_behavior="b", category="positive")
    assert is_positive(ex)


def test_is_positive_for_negative_category():
    ex = EvalExample(task_input="t", expected_behavior="b", category="negative")
    assert not is_positive(ex)


# ── ToolDatasetBuilder ────────────────────────────────────────────────────


def _make_cases_payload(positive: int, negative: int) -> str:
    cases = []
    for i in range(positive):
        cases.append({
            "task_input": f"positive task {i}",
            "polarity": "positive",
            "expected_behavior": "should pick this tool",
            "difficulty": "medium",
        })
    for i in range(negative):
        cases.append({
            "task_input": f"negative task {i}",
            "polarity": "negative",
            "expected_behavior": "should pick a different tool",
            "difficulty": "medium",
        })
    return json.dumps(cases)


def test_dataset_builder_generates_balanced_dataset(monkeypatch):
    config = EvolutionConfig(eval_dataset_size=10)
    builder = ToolDatasetBuilder(config)

    fake_result = MagicMock(test_cases=_make_cases_payload(positive=5, negative=5))

    # Patch generator and make_lm so no real LLM call happens
    builder.generator = lambda **kwargs: fake_result
    monkeypatch.setattr(config, "make_lm", lambda model: MagicMock())

    with patch("dspy.context"):
        dataset = builder.generate("search_files", "Search by regex.")

    pos = sum(1 for e in dataset.all_examples if e.category == "positive")
    neg = sum(1 for e in dataset.all_examples if e.category == "negative")
    assert pos == 5
    assert neg == 5


def test_dataset_builder_rejects_no_negatives(monkeypatch):
    """If LLM returns only positive examples, builder must raise — no contrast."""
    config = EvolutionConfig(eval_dataset_size=5)
    builder = ToolDatasetBuilder(config)
    fake_result = MagicMock(test_cases=_make_cases_payload(positive=5, negative=0))
    builder.generator = lambda **kwargs: fake_result
    monkeypatch.setattr(config, "make_lm", lambda model: MagicMock())

    with patch("dspy.context"), pytest.raises(ValueError, match="positive"):
        builder.generate("t", "d")


def test_dataset_builder_rejects_no_positives(monkeypatch):
    config = EvolutionConfig(eval_dataset_size=5)
    builder = ToolDatasetBuilder(config)
    fake_result = MagicMock(test_cases=_make_cases_payload(positive=0, negative=5))
    builder.generator = lambda **kwargs: fake_result
    monkeypatch.setattr(config, "make_lm", lambda model: MagicMock())

    with patch("dspy.context"), pytest.raises(ValueError, match="negative"):
        builder.generate("t", "d")


def test_dataset_builder_raises_on_unparseable_output(monkeypatch):
    config = EvolutionConfig(eval_dataset_size=5)
    builder = ToolDatasetBuilder(config)
    fake_result = MagicMock(test_cases="not valid json or python")
    builder.generator = lambda **kwargs: fake_result
    monkeypatch.setattr(config, "make_lm", lambda model: MagicMock())

    with patch("dspy.context"), pytest.raises(ValueError, match="Could not parse"):
        builder.generate("t", "d")


def test_dataset_builder_skips_unlabeled_cases(monkeypatch):
    """Cases without polarity field must be silently skipped, not crash."""
    config = EvolutionConfig(eval_dataset_size=5)
    builder = ToolDatasetBuilder(config)
    payload = json.dumps([
        {"task_input": "good", "polarity": "positive", "expected_behavior": "ok"},
        {"task_input": "missing label"},  # no polarity, will be skipped
        {"task_input": "bad", "polarity": "negative", "expected_behavior": "ok"},
        {"task_input": "weird", "polarity": "neutral"},  # invalid polarity, skip
    ])
    fake_result = MagicMock(test_cases=payload)
    builder.generator = lambda **kwargs: fake_result
    monkeypatch.setattr(config, "make_lm", lambda model: MagicMock())

    with patch("dspy.context"):
        dataset = builder.generate("t", "d")

    # Only the two cleanly-labeled examples should survive
    assert len(dataset.all_examples) == 2
    polarities = sorted(e.category for e in dataset.all_examples)
    assert polarities == ["negative", "positive"]


# ── to_dspy_examples_with_polarity ────────────────────────────────────────


def test_to_dspy_examples_preserves_polarity():
    ds = EvalDataset(
        train=[
            EvalExample(task_input="t1", expected_behavior="b1", category="positive"),
            EvalExample(task_input="t2", expected_behavior="b2", category="negative"),
        ]
    )
    examples = to_dspy_examples_with_polarity(ds, "train")
    assert len(examples) == 2
    cats = sorted(e.category for e in examples)
    assert cats == ["negative", "positive"]


def test_to_dspy_examples_marks_task_input_as_input():
    """Only task_input is the model input; other fields are ground truth."""
    ds = EvalDataset(
        train=[EvalExample(task_input="t", expected_behavior="b", category="positive")]
    )
    examples = to_dspy_examples_with_polarity(ds, "train")
    assert examples[0].inputs().toDict() == {"task_input": "t"}

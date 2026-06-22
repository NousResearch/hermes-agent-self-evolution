"""Tests for the evaluation dataset builder (no real LLM calls)."""

from unittest.mock import MagicMock, patch

import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import (
    EvalExample,
    EvalDataset,
    SyntheticDatasetBuilder,
    GoldenDatasetLoader,
)


# ── EvalExample ───────────────────────────────────────────────────────────


def test_eval_example_roundtrip():
    ex = EvalExample(
        task_input="review this PR",
        expected_behavior="identifies the SQL injection",
        difficulty="hard",
        category="security",
        source="golden",
    )
    restored = EvalExample.from_dict(ex.to_dict())
    assert restored == ex


def test_eval_example_from_dict_ignores_unknown_keys():
    d = {"task_input": "t", "expected_behavior": "b", "bogus": "ignored"}
    ex = EvalExample.from_dict(d)
    assert ex.task_input == "t"
    assert ex.expected_behavior == "b"
    # Defaults preserved for unspecified fields
    assert ex.difficulty == "medium"


# ── EvalDataset save/load roundtrip ───────────────────────────────────────


def test_dataset_save_load_roundtrip(tmp_path):
    dataset = EvalDataset(
        train=[EvalExample("t1", "b1"), EvalExample("t2", "b2")],
        val=[EvalExample("v1", "vb1")],
        holdout=[EvalExample("h1", "hb1")],
    )
    dataset.save(tmp_path)

    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert (tmp_path / "holdout.jsonl").exists()

    loaded = EvalDataset.load(tmp_path)
    assert len(loaded.train) == 2
    assert len(loaded.val) == 1
    assert len(loaded.holdout) == 1
    assert loaded.train[0].task_input == "t1"
    assert loaded.holdout[0].expected_behavior == "hb1"


def test_dataset_load_missing_splits_returns_empty(tmp_path):
    # Empty directory — no jsonl files
    loaded = EvalDataset.load(tmp_path)
    assert loaded.train == []
    assert loaded.val == []
    assert loaded.holdout == []


def test_all_examples_concatenates_splits():
    dataset = EvalDataset(
        train=[EvalExample("t", "b")],
        val=[EvalExample("v", "b")],
        holdout=[EvalExample("h", "b"), EvalExample("h2", "b")],
    )
    assert len(dataset.all_examples) == 4


# ── to_dspy_examples conversion ───────────────────────────────────────────


def test_to_dspy_examples_marks_inputs():
    dataset = EvalDataset(train=[EvalExample("ask something", "good rubric")])
    examples = dataset.to_dspy_examples("train")
    assert len(examples) == 1
    ex = examples[0]
    assert ex.task_input == "ask something"
    assert ex.expected_behavior == "good rubric"
    # task_input must be marked as the input field
    assert "task_input" in ex.inputs()


def test_to_dspy_examples_empty_split():
    dataset = EvalDataset()
    assert dataset.to_dspy_examples("val") == []


# ── SyntheticDatasetBuilder.generate (mocked LLM) ─────────────────────────


def _mock_generator_returning(json_text):
    """Build a SyntheticDatasetBuilder whose generator returns json_text."""
    config = EvolutionConfig()
    builder = SyntheticDatasetBuilder(config)
    mock_result = MagicMock()
    mock_result.test_cases = json_text
    builder.generator = MagicMock(return_value=mock_result)
    return builder


def test_synthetic_generate_parses_clean_json():
    cases = (
        '[{"task_input": "a", "expected_behavior": "ra", "difficulty": "easy", "category": "c1"},'
        ' {"task_input": "b", "expected_behavior": "rb", "difficulty": "hard", "category": "c2"},'
        ' {"task_input": "c", "expected_behavior": "rc", "difficulty": "medium", "category": "c3"},'
        ' {"task_input": "d", "expected_behavior": "rd", "difficulty": "medium", "category": "c4"}]'
    )
    builder = _mock_generator_returning(cases)
    with patch("evolution.core.dataset_builder.dspy.LM"):
        dataset = builder.generate("some skill text", artifact_type="skill")

    assert len(dataset.all_examples) == 4
    assert all(ex.source == "synthetic" for ex in dataset.all_examples)


def test_synthetic_generate_extracts_json_from_prose():
    # LLM wraps the array in explanatory prose
    wrapped = (
        'Here are your test cases:\n'
        '[{"task_input": "a", "expected_behavior": "ra"},'
        ' {"task_input": "b", "expected_behavior": "rb"}]\n'
        'Hope that helps!'
    )
    builder = _mock_generator_returning(wrapped)
    with patch("evolution.core.dataset_builder.dspy.LM"):
        dataset = builder.generate("skill", artifact_type="skill")
    assert len(dataset.all_examples) == 2


def test_synthetic_generate_handles_single_quoted_python_literal():
    # Not valid JSON (single quotes) but valid Python literal
    py_literal = "[{'task_input': 'a', 'expected_behavior': 'ra'}]"
    builder = _mock_generator_returning(py_literal)
    with patch("evolution.core.dataset_builder.dspy.LM"):
        dataset = builder.generate("skill", artifact_type="skill")
    assert len(dataset.all_examples) == 1


def test_synthetic_generate_skips_incomplete_cases():
    # One case missing expected_behavior should be dropped
    cases = (
        '[{"task_input": "a", "expected_behavior": "ra"},'
        ' {"task_input": "no_rubric"},'
        ' {"expected_behavior": "no_task"}]'
    )
    builder = _mock_generator_returning(cases)
    with patch("evolution.core.dataset_builder.dspy.LM"):
        dataset = builder.generate("skill", artifact_type="skill")
    assert len(dataset.all_examples) == 1
    assert dataset.all_examples[0].task_input == "a"


def test_synthetic_generate_raises_on_unparseable_output():
    builder = _mock_generator_returning("totally not json at all")
    with patch("evolution.core.dataset_builder.dspy.LM"):
        with pytest.raises(ValueError, match="Could not parse"):
            builder.generate("skill", artifact_type="skill")


# ── GoldenDatasetLoader ───────────────────────────────────────────────────


def test_golden_loader_prefers_existing_splits(tmp_path):
    EvalDataset(
        train=[EvalExample("t", "b")],
        val=[EvalExample("v", "b")],
        holdout=[EvalExample("h", "b")],
    ).save(tmp_path)
    loaded = GoldenDatasetLoader.load(tmp_path)
    assert len(loaded.train) == 1
    assert len(loaded.val) == 1
    assert len(loaded.holdout) == 1


def test_golden_loader_autosplits_single_file(tmp_path):
    golden = tmp_path / "golden.jsonl"
    lines = [
        EvalExample(f"task{i}", f"rubric{i}").to_dict() for i in range(8)
    ]
    import json
    golden.write_text("\n".join(json.dumps(d) for d in lines), encoding="utf-8")

    loaded = GoldenDatasetLoader.load(tmp_path)
    # 8 examples, 0.5/0.25 split → 4 train / 2 val / 2 holdout
    assert len(loaded.train) == 4
    assert len(loaded.val) == 2
    assert len(loaded.holdout) == 2


def test_golden_loader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        GoldenDatasetLoader.load(tmp_path)

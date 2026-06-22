"""Tests for the synthetic tool-selection dataset builder (no API calls)."""

from unittest.mock import MagicMock, patch
import pytest

from evolution.core.config import EvolutionConfig
from evolution.tools.tool_dataset import (
    ToolSelectionExample, ToolSelectionDataset, ToolDatasetBuilder,
)


def test_example_roundtrip():
    ex = ToolSelectionExample(
        task="find imports of os", correct_tool="search_files",
        alt_tools=["terminal"], difficulty="medium", kind="confusable",
    )
    assert ToolSelectionExample.from_dict(ex.to_dict()) == ex


def test_dataset_save_load_roundtrip(tmp_path):
    ds = ToolSelectionDataset(
        train=[ToolSelectionExample("a", "read_file")],
        val=[ToolSelectionExample("b", "terminal")],
        holdout=[ToolSelectionExample("c", "none", kind="no_tool")],
    )
    ds.save(tmp_path)
    loaded = ToolSelectionDataset.load(tmp_path)
    assert loaded.train[0].correct_tool == "read_file"
    assert loaded.holdout[0].kind == "no_tool"


def _builder_returning(json_text):
    builder = ToolDatasetBuilder(EvolutionConfig())
    mock_result = MagicMock()
    mock_result.triples = json_text
    builder.generator = MagicMock(return_value=mock_result)
    return builder


def test_generate_parses_clean_json():
    triples = (
        '[{"task": "grep for X", "correct_tool": "search_files", "alt_tools": ["terminal"], '
        '"difficulty": "medium", "kind": "confusable"},'
        '{"task": "what is a hashmap", "correct_tool": "none", "alt_tools": [], '
        '"difficulty": "easy", "kind": "no_tool"},'
        '{"task": "open example.com", "correct_tool": "browser_navigate", "alt_tools": [], '
        '"difficulty": "easy", "kind": "clear"},'
        '{"task": "read config.py", "correct_tool": "read_file", "alt_tools": ["terminal"], '
        '"difficulty": "easy", "kind": "confusable"}]'
    )
    builder = _builder_returning(triples)
    with patch("evolution.tools.tool_dataset.dspy.LM"):
        ds = builder.generate([("read_file", "Read a file.")])
    kinds = {ex.kind for ex in ds.all_examples}
    assert "no_tool" in kinds and "confusable" in kinds
    assert len(ds.all_examples) == 4


def test_generate_extracts_json_from_prose():
    wrapped = 'Sure!\n[{"task": "x", "correct_tool": "terminal"}]\nDone.'
    builder = _builder_returning(wrapped)
    with patch("evolution.tools.tool_dataset.dspy.LM"):
        ds = builder.generate([("terminal", "Run a command.")])
    assert len(ds.all_examples) == 1


def test_generate_skips_incomplete():
    triples = '[{"task": "x", "correct_tool": "terminal"}, {"task": "no_tool_field"}]'
    builder = _builder_returning(triples)
    with patch("evolution.tools.tool_dataset.dspy.LM"):
        ds = builder.generate([("terminal", "Run a command.")])
    assert len(ds.all_examples) == 1


def test_generate_raises_on_unparseable():
    builder = _builder_returning("not json")
    with patch("evolution.tools.tool_dataset.dspy.LM"):
        with pytest.raises(ValueError, match="Could not parse"):
            builder.generate([("terminal", "Run a command.")])

"""Tests for evolution/tools/tool_module.py — DSPy module wrapping a tool description."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import dspy
import pytest

from evolution.tools.tool_module import (
    TOOL_DESC_END,
    TOOL_DESC_START,
    ToolModule,
    extract_evolved_description,
    load_tool_definition,
    save_tool_definition,
)


# ── load_tool_definition ──────────────────────────────────────────────────


def test_load_valid_tool_definition(tmp_path):
    path = tmp_path / "search_files.json"
    path.write_text(json.dumps({
        "name": "search_files",
        "description": "Search files by content using a regex.",
        "parameters": {"query": "string"},
    }))
    tool = load_tool_definition(path)
    assert tool["name"] == "search_files"
    assert tool["description"] == "Search files by content using a regex."
    assert tool["parameters"] == {"query": "string"}


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tool_definition(tmp_path / "nope.json")


def test_load_malformed_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json {")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_tool_definition(path)


def test_load_missing_name_raises(tmp_path):
    path = tmp_path / "no_name.json"
    path.write_text(json.dumps({"description": "x"}))
    with pytest.raises(ValueError, match="missing 'name'"):
        load_tool_definition(path)


def test_load_missing_description_raises(tmp_path):
    path = tmp_path / "no_desc.json"
    path.write_text(json.dumps({"name": "tool"}))
    with pytest.raises(ValueError, match="missing 'description'"):
        load_tool_definition(path)


def test_load_rejects_path_traversal_name(tmp_path):
    path = tmp_path / "evil.json"
    path.write_text(json.dumps({
        "name": "../../etc/passwd",
        "description": "evil",
    }))
    with pytest.raises(ValueError, match="Invalid tool name"):
        load_tool_definition(path)


def test_load_rejects_shell_metacharacters(tmp_path):
    for bad_name in ["foo;rm", "foo bar", "foo|baz", "foo`x`", "foo$x"]:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"name": bad_name, "description": "x"}))
        with pytest.raises(ValueError, match="Invalid tool name"):
            load_tool_definition(path)


def test_load_accepts_valid_names(tmp_path):
    for good_name in ["search_files", "tool-name", "tool.v2", "Tool_123"]:
        path = tmp_path / f"{good_name}.json"
        path.write_text(json.dumps({"name": good_name, "description": "x"}))
        tool = load_tool_definition(path)
        assert tool["name"] == good_name


# ── save_tool_definition ──────────────────────────────────────────────────


def test_save_writes_evolved_description(tmp_path):
    path = tmp_path / "src.json"
    path.write_text(json.dumps({
        "name": "tool",
        "description": "original",
        "parameters": {"k": "v"},
    }))
    tool = load_tool_definition(path)

    out = tmp_path / "out" / "evolved.json"
    save_tool_definition(tool, "evolved description", out)

    saved = json.loads(out.read_text())
    assert saved["description"] == "evolved description"
    assert saved["name"] == "tool"
    assert saved["parameters"] == {"k": "v"}


def test_save_creates_parent_dirs(tmp_path):
    tool = {"name": "t", "description": "d", "parameters": {}, "raw": {"name": "t", "description": "d"}}
    out = tmp_path / "deep" / "nested" / "dir" / "tool.json"
    save_tool_definition(tool, "new", out)
    assert out.exists()


# ── ToolModule ────────────────────────────────────────────────────────────


def test_tool_module_embeds_description_with_sentinels():
    tm = ToolModule("Search files by content.")
    instructions = tm.predictor.predict.signature.instructions
    assert TOOL_DESC_START in instructions
    assert TOOL_DESC_END in instructions
    assert "Search files by content." in instructions


def test_tool_module_includes_untrusted_preamble_by_default():
    tm = ToolModule("desc")
    instructions = tm.predictor.predict.signature.instructions
    assert "DATA, not as commands" in instructions


def test_tool_module_can_disable_preamble():
    tm = ToolModule("desc", treat_as_untrusted=False)
    instructions = tm.predictor.predict.signature.instructions
    assert "DATA, not as commands" not in instructions
    # Sentinels still present so we can recover the description.
    assert TOOL_DESC_START in instructions


class _StubPredictor:
    """Callable stub that mimics dspy.ChainOfThought returning a Prediction."""

    def __init__(self, decision: str, rationale: str = ""):
        self._decision = decision
        self._rationale = rationale

    def __call__(self, **kwargs):
        return dspy.Prediction(decision=self._decision, rationale=self._rationale)


def test_tool_module_normalizes_yes_decision():
    tm = ToolModule("desc")
    tm.predictor = _StubPredictor("Yes!", "because")
    result = tm(task_input="search for foo")
    assert result.output == "yes"


def test_tool_module_normalizes_no_decision():
    tm = ToolModule("desc")
    tm.predictor = _StubPredictor("No, not a fit", "wrong tool")
    result = tm(task_input="task")
    assert result.output == "no"


def test_tool_module_ambiguous_decision_defaults_to_no():
    tm = ToolModule("desc")
    tm.predictor = _StubPredictor("maybe?", "unclear")
    result = tm(task_input="task")
    # Conservative: ambiguous → no
    assert result.output == "no"


def test_tool_module_passes_rationale_through():
    tm = ToolModule("desc")
    tm.predictor = _StubPredictor("yes", "matches the task perfectly")
    result = tm(task_input="task")
    assert "matches the task perfectly" in result.rationale


# ── extract_evolved_description ───────────────────────────────────────────


def test_extract_recovers_sentinel_delimited_body():
    tm = ToolModule("Original description")
    # Simulate optimizer keeping original instructions
    body = extract_evolved_description(tm, baseline="Original description")
    assert body == "Original description"


def test_extract_falls_back_to_baseline_when_no_sentinels():
    """If sentinels are missing, return the baseline as a safe no-op."""
    # Build a tool module then mutate its instructions to remove sentinels
    tm = ToolModule("desc")
    sig = tm.predictor.predict.signature
    tm.predictor.predict.signature = sig.with_instructions("no sentinels here")
    body = extract_evolved_description(tm, baseline="baseline desc")
    assert body == "baseline desc"


def test_extract_recovers_body_with_markdown_dividers():
    """A description containing `---` must still round-trip cleanly."""
    body_with_divider = "First line.\n\n---\n\nSecond line after divider."
    tm = ToolModule(body_with_divider)
    recovered = extract_evolved_description(tm, baseline="should not be used")
    assert recovered == body_with_divider

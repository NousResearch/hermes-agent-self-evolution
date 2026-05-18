"""Tests for the mutator module (internal engine only; external is a stub)."""

from evolution.code.mutator import strip_fenced_source, select_engine


class TestStripFencedSource:
    def test_strips_python_fence(self):
        text = "```python\ndef foo(): pass\n```"
        assert strip_fenced_source(text) == "def foo(): pass"

    def test_strips_plain_fence(self):
        text = "```\ndef bar(): pass\n```"
        assert strip_fenced_source(text) == "def bar(): pass"

    def test_returns_raw_if_no_fence(self):
        text = "def baz(): pass"
        assert strip_fenced_source(text) == "def baz(): pass"

    def test_empty_returns_empty(self):
        assert strip_fenced_source("") == ""


class TestSelectEngine:
    def test_internal_returns_mutator(self):
        m = select_engine("internal", model_name="ollama/qwen2.5:7b")
        assert hasattr(m, "propose")

    def test_unknown_raises(self):
        try:
            select_engine("nonexistent", model_name="x")
            assert False, "should have raised"
        except ValueError as exc:
            assert "nonexistent" in str(exc)

    def test_darwinian_evolver_not_installed_raises(self):
        try:
            select_engine("darwinian-evolver", model_name="x")
            assert False, "should have raised"
        except RuntimeError as exc:
            assert "not found" in str(exc).lower()

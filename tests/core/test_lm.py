"""Tests for the bounded dspy.LM factory."""

import dspy

from evolution.core import lm as lm_module
from evolution.core.lm import make_lm


def _capture(monkeypatch):
    captured = {}

    def fake_lm(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "LM"

    monkeypatch.setattr(lm_module.dspy, "LM", fake_lm, raising=True)
    return captured


def test_make_lm_is_bounded(monkeypatch):
    captured = _capture(monkeypatch)

    result = make_lm("openrouter/x")

    assert result == "LM"
    assert captured["model"] == "openrouter/x"
    assert captured["kwargs"]["timeout"] == 120
    assert captured["kwargs"]["num_retries"] == 2
    assert "temperature" not in captured["kwargs"]
    assert "max_tokens" not in captured["kwargs"]


def test_make_lm_reflection_adds_sampling_kwargs(monkeypatch):
    captured = _capture(monkeypatch)

    make_lm("openrouter/x", reflection=True)

    kwargs = captured["kwargs"]
    assert kwargs["temperature"] == 1.0
    assert kwargs["max_tokens"] == 32000
    assert kwargs["timeout"] == 180
    assert kwargs["num_retries"] == 2


def test_make_lm_returns_real_dspy_lm():
    # Sanity: the non-monkeypatched path builds an actual dspy.LM.
    assert isinstance(make_lm("openrouter/x"), dspy.LM)

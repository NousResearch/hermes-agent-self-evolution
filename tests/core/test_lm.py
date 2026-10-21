"""make_lm routes models/endpoints from one place."""

import pytest

import evolution.core.lm as lm_mod
from evolution.core.lm import make_lm


class _Spy:
    def __init__(self, model, **kw):
        self.model, self.kw = model, kw


@pytest.fixture(autouse=True)
def spy(monkeypatch):
    monkeypatch.setattr(lm_mod.dspy, "LM", _Spy)
    monkeypatch.setattr(lm_mod, "_CopilotLM", _Spy)
    monkeypatch.setattr(lm_mod, "_copilot_model", lambda base, token, name: {
        "id": name, "supported_endpoints": ["/responses" if name.startswith("gpt-6") else "/chat/completions"]
    })
    monkeypatch.delenv("EVOLUTION_API_BASE", raising=False)
    monkeypatch.delenv("EVOLUTION_API_KEY", raising=False)
    lm_mod._copilot_cache.clear()


def test_plain_model_passthrough():
    m = make_lm("openai/gpt-4.1")
    assert m.model == "openai/gpt-4.1" and m.kw == {}


def test_env_overrides_endpoint(monkeypatch):
    monkeypatch.setenv("EVOLUTION_API_BASE", "http://proxy/v1")
    monkeypatch.setenv("EVOLUTION_API_KEY", "k")
    assert make_lm("openai/x").kw == {"api_base": "http://proxy/v1", "api_key": "k"}


def test_copilot_uses_exchanged_token(monkeypatch):
    monkeypatch.setattr(lm_mod, "_copilot_credentials", lambda: ("tok", "https://api.example"))
    m = make_lm("copilot/gpt-4.1")
    assert m.model == "openai/gpt-4.1"
    assert m.kw["api_key"] == "tok" and m.kw["api_base"] == "https://api.example"
    assert "Copilot-Integration-Id" in m.kw["extra_headers"]


def test_copilot_requires_token(monkeypatch):
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="COPILOT_GITHUB_TOKEN"):
        make_lm("copilot/gpt-4.1")


def test_copilot_gpt_uses_responses_api(monkeypatch):
    monkeypatch.setattr(lm_mod, "_copilot_credentials", lambda: ("tok", "https://base"))
    assert make_lm("copilot/gpt-6.1-sol").kw["model_type"] == "responses"
    assert "model_type" not in make_lm("copilot/claude-sonnet-5.5").kw

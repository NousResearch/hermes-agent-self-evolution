"""Focused tests for Hermes OpenAI-Codex OAuth DSPy integration."""

from types import SimpleNamespace

import pytest

from evolution.core import dspy_lm, hermes_codex


class _FakeOpenAI:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeOpenAI.calls.append(kwargs)


class _FakeCodexAuxiliaryClient:
    built = []

    def __init__(self, real_client, model):
        self.real_client = real_client
        self.model = model
        self.closed = False
        _FakeCodexAuxiliaryClient.built.append((real_client, model))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="codex ok"), finish_reason="stop")],
            model=kwargs.get("model"),
            usage=None,
        )

    def close(self):
        self.closed = True


def test_build_codex_auxiliary_client_uses_hermes_runtime_credentials(monkeypatch):
    _FakeOpenAI.calls = []
    _FakeCodexAuxiliaryClient.built = []

    def fake_resolve(**kwargs):
        assert kwargs == {"refresh_if_expiring": True}
        return {"api_key": "secret-token", "base_url": "https://codex.test/backend"}

    def fake_headers(token):
        assert token == "secret-token"
        return {"originator": "codex_cli_rs"}

    monkeypatch.setattr(
        hermes_codex,
        "_load_codex_dependencies",
        lambda: (fake_resolve, _FakeOpenAI, _FakeCodexAuxiliaryClient, fake_headers),
    )

    client = hermes_codex.build_codex_auxiliary_client("gpt-5.4-mini")

    assert isinstance(client, _FakeCodexAuxiliaryClient)
    assert _FakeCodexAuxiliaryClient.built[0][1] == "gpt-5.4-mini"
    assert _FakeOpenAI.calls == [
        {
            "api_key": "secret-token",
            "base_url": "https://codex.test/backend",
            "default_headers": {"originator": "codex_cli_rs"},
        }
    ]


def test_codex_oauth_lm_strips_prefix_and_returns_dspy_output(monkeypatch):
    calls = []
    closed = []

    class FakeClient:
        def __init__(self, model):
            self.model = model
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello from codex"), finish_reason="stop")],
                model=kwargs.get("model"),
                usage=None,
            )

        def close(self):
            closed.append(self.model)

    def fake_build(model_id, *, refresh_if_expiring=True):
        assert refresh_if_expiring is True
        return FakeClient(model_id)

    monkeypatch.setattr(dspy_lm, "build_codex_auxiliary_client", fake_build)

    lm = dspy_lm.CodexOAuthLM("openai-codex/gpt-5.4-mini", num_retries=1, timeout=120)
    assert lm(prompt="say hi") == ["hello from codex"]

    assert calls[0]["model"] == "gpt-5.4-mini"
    assert calls[0]["messages"] == [{"role": "user", "content": "say hi"}]
    assert calls[0]["timeout"] == 120
    assert closed == ["gpt-5.4-mini"]


def test_make_dspy_lm_uses_standard_dspy_for_non_codex(monkeypatch):
    created = []

    class FakeLM:
        def __init__(self, model, **kwargs):
            created.append((model, kwargs))

    monkeypatch.setattr(dspy_lm.dspy, "LM", FakeLM)

    lm = dspy_lm.make_dspy_lm("ollama_chat/gemma4-e4b:latest", timeout=3)

    assert isinstance(lm, FakeLM)
    assert created == [("ollama_chat/gemma4-e4b:latest", {"timeout": 3})]


def test_codex_model_id_rejects_empty_suffix():
    with pytest.raises(ValueError):
        hermes_codex.codex_model_id("openai-codex/")

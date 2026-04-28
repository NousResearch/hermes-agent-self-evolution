"""Tests for provider-agnostic model comparison helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evolution.models.compare import ModelConfigError, compare_chat_models, resolve_provider


class _FakeCompletions:
    def __init__(self, calls):
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        model = kwargs["model"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({"model": model, "verdict": "ok"}))
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        )


class _FakeClient:
    def __init__(self, calls):
        self.chat = SimpleNamespace(completions=_FakeCompletions(calls))


def test_resolve_provider_uses_deepseek_defaults_without_hardcoding_models(monkeypatch):
    monkeypatch.delenv("HERMES_EVOLVE_DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("HERMES_EVOLVE_DEEPSEEK_API_KEY_ENV", raising=False)

    provider = resolve_provider("deepseek")

    assert provider.name == "deepseek"
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.api_key_env == "DEEPSEEK_API_KEY"
    assert not hasattr(provider, "models")


def test_compare_chat_models_uses_supplied_model_ids_and_never_returns_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "super-secret-value")
    calls = []
    captured = {}

    def factory(*, api_key, base_url, timeout):
        captured.update({"api_key": api_key, "base_url": base_url, "timeout": timeout})
        return _FakeClient(calls)

    results = compare_chat_models(
        models=["deepseek-v4-flash", "deepseek-v4-pro"],
        prompt="Tag this comment as strict JSON.",
        provider="deepseek",
        max_tokens=64,
        temperature=0.1,
        timeout=12.5,
        extra_body={"thinking": {"type": "disabled"}},
        client_factory=factory,
    )

    assert captured == {
        "api_key": "super-secret-value",
        "base_url": "https://api.deepseek.com",
        "timeout": 12.5,
    }
    assert [call["model"] for call in calls] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert all(call["max_tokens"] == 64 for call in calls)
    assert all(call["temperature"] == 0.1 for call in calls)
    assert all(call["extra_body"] == {"thinking": {"type": "disabled"}} for call in calls)
    assert [result["model"] for result in results] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert all(result["ok"] for result in results)
    assert "super-secret-value" not in json.dumps(results)


def test_compare_chat_models_requires_key_from_configured_env(monkeypatch):
    monkeypatch.delenv("MISSING_DEEPSEEK_KEY", raising=False)

    with pytest.raises(ModelConfigError, match="MISSING_DEEPSEEK_KEY"):
        compare_chat_models(
            models=["deepseek-v4-flash"],
            prompt="ping",
            provider="deepseek",
            api_key_env="MISSING_DEEPSEEK_KEY",
            client_factory=lambda **kwargs: None,
        )

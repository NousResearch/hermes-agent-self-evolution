"""Regressions for Copilot host selection and per-model wire protocols."""
import io
import json

import pytest

from evolution.core import lm


@pytest.fixture
def factory(monkeypatch):
    class Captured:
        def __init__(self, model, **kwargs):
            self.model, self.kw = model, kwargs
    monkeypatch.setattr(lm.dspy, 'LM', Captured)
    monkeypatch.setattr(lm, '_CopilotLM', Captured)
    monkeypatch.setattr(lm, '_copilot_credentials', lambda: ('synthetic-token', 'https://api.githubcopilot.com'))
    monkeypatch.delenv('EVOLUTION_API_BASE', raising=False)
    monkeypatch.delenv('EVOLUTION_API_KEY', raising=False)


@pytest.mark.parametrize('model,endpoints,expected,kind', [
    ('gpt-4.1', ['/chat/completions'], 'openai/gpt-4.1', 'chat'),
    ('gpt-6.1-sol', ['/responses'], 'openai/gpt-6.1-sol', 'responses'),
    ('claude-sonnet-5.5', ['/chat/completions', '/v1/messages'], 'anthropic/claude-sonnet-5.5', 'chat'),
    ('gemini-3.8-flash', ['/chat/completions'], 'openai/gemini-3.8-flash', 'chat'),
])
def test_model_uses_advertised_transport(factory, monkeypatch, model, endpoints, expected, kind):
    monkeypatch.setattr(lm, '_copilot_model', lambda base, token, name: {'id': name, 'supported_endpoints': endpoints}, raising=False)
    client = lm.make_lm('copilot/' + model)
    assert client.model == expected
    assert client.kw.get('model_type', 'chat') == kind
    assert client.kw['extra_headers']['Copilot-Integration-Id'] == 'vscode-chat'
    if kind == 'responses':
        assert client.kw['store'] is False
    if expected.startswith('anthropic/'):
        assert client.kw['extra_headers']['Authorization'] == 'Bearer synthetic-token'
        assert client.kw['max_tokens'] <= 16384


def test_generic_copilot_endpoint_retains_integration_headers(factory, monkeypatch):
    monkeypatch.setenv('EVOLUTION_API_BASE', 'https://copilot-api.example.ghe.com')
    monkeypatch.setenv('EVOLUTION_API_KEY', 'synthetic-token')
    monkeypatch.setattr(lm, '_copilot_model', lambda base, token, name: {'id': name, 'supported_endpoints': ['/chat/completions']}, raising=False)
    client = lm.make_lm('openai/gemini-3.8-flash')
    assert client.kw['extra_headers']['Copilot-Integration-Id'] == 'vscode-chat'


def test_enterprise_credentials_stay_on_the_selected_host(monkeypatch):
    lm._copilot_cache.clear()
    monkeypatch.setenv('COPILOT_GH_HOST', 'example.ghe.com')
    monkeypatch.setenv('COPILOT_GITHUB_TOKEN', 'synthetic-gh')
    urls = []

    def read(request, timeout):
        urls.append(request.full_url)
        return io.BytesIO(json.dumps({'endpoints': {'api': 'https://copilot-api.example.ghe.com'}}).encode())

    monkeypatch.setattr(lm.urllib.request, 'urlopen', read)
    try:
        token, base = lm._copilot_credentials()
        assert token == 'synthetic-gh'
        assert base == 'https://copilot-api.example.ghe.com'
        assert urls == ['https://api.example.ghe.com/copilot_internal/user']
    finally:
        lm._copilot_cache.clear()


def test_discovered_api_host_is_validated_before_sending_token(monkeypatch):
    lm._copilot_cache.clear()
    monkeypatch.setenv('COPILOT_GH_HOST', 'example.ghe.com')
    monkeypatch.setenv('COPILOT_GITHUB_TOKEN', 'synthetic-gh')
    monkeypatch.setattr(lm.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(json.dumps({'endpoints': {'api': 'https://untrusted.example'}}).encode()))
    try:
        with pytest.raises(ValueError, match='endpoint'):
            lm._copilot_credentials()
    finally:
        lm._copilot_cache.clear()


def test_model_catalog_cache_is_bounded(monkeypatch):
    lm._copilot_models.cache_clear()
    calls = []

    def listing(url, token):
        calls.append((url, token))
        return {"data": [{"id": "gpt-4.1"}]}

    monkeypatch.setattr(lm, "_read_json", listing)
    try:
        for token in ["synthetic-old", "synthetic-new"]:
            lm._copilot_model("https://api.githubcopilot.com", token, "gpt-4.1")
        assert len(calls) == 2
        assert lm._copilot_models.cache_info().currsize == 1
    finally:
        lm._copilot_models.cache_clear()


def test_copilot_endpoint_missing_key_is_actionable(factory, monkeypatch):
    monkeypatch.setenv("EVOLUTION_API_BASE", "https://api.githubcopilot.com")
    with pytest.raises(RuntimeError, match="EVOLUTION_API_KEY"):
        lm.make_lm("openai/gpt-4.1")

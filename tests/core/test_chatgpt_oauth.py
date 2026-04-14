"""Tests for ChatGPT OAuth-backed model support."""

from pathlib import Path

import pytest

from evolution.core.chatgpt_oauth import (
    ChatGPTOAuthLM,
    build_chatgpt_request,
    create_lm,
    resolve_chatgpt_oauth_token,
)


def test_resolve_token_from_openclaw_profile(tmp_path, monkeypatch):
    auth_file = tmp_path / "auth-profiles.json"
    auth_file.write_text(
        '{"profiles":{"openai-codex:default":{"access":"token-from-openclaw"}}}'
    )

    token = resolve_chatgpt_oauth_token(auth_file=auth_file)

    assert token == "token-from-openclaw"


def test_resolve_token_from_codex_auth_file(tmp_path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"tokens":{"access_token":"token-from-codex"}}')

    token = resolve_chatgpt_oauth_token(auth_file=auth_file)

    assert token == "token-from-codex"


def test_env_token_beats_file(tmp_path, monkeypatch):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"tokens":{"access_token":"token-from-codex"}}')
    monkeypatch.setenv("CHATGPT_OAUTH_TOKEN", "token-from-env")

    token = resolve_chatgpt_oauth_token(auth_file=auth_file)

    assert token == "token-from-env"


def test_build_request_moves_system_messages_to_instructions():
    payload = build_chatgpt_request(
        model="chatgpt/gpt-5.4",
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "developer", "content": "Answer in lowercase."},
            {"role": "user", "content": "Say hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )

    assert payload["model"] == "gpt-5.4"
    assert payload["store"] is False
    assert payload["instructions"] == "Be terse.\n\nAnswer in lowercase."
    assert payload["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "Say hi"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
    ]


def test_build_request_uses_prompt_when_messages_missing():
    payload = build_chatgpt_request(model="chatgpt/gpt-5.4", prompt="Say hi")

    assert payload["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "Say hi"}]}
    ]
    assert payload["instructions"]


def test_create_lm_returns_chatgpt_oauth_client(monkeypatch):
    monkeypatch.setenv("CHATGPT_OAUTH_TOKEN", "token-from-env")

    lm = create_lm("chatgpt/gpt-5.4")

    assert isinstance(lm, ChatGPTOAuthLM)
    assert lm.model == "gpt-5.4"


def test_create_lm_rejects_unsupported_temperature(monkeypatch):
    monkeypatch.setenv("CHATGPT_OAUTH_TOKEN", "token-from-env")

    with pytest.raises(ValueError, match="does not support temperature"):
        create_lm("chatgpt/gpt-5.4", temperature=0.2)

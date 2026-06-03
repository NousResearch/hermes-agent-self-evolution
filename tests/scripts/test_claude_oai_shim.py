"""Unit tests for claude_oai_shim helpers (no live SDK session required)."""

from scripts.claude_oai_shim import (
    MODEL_MAP,
    _messages_to_prompt,
    _resolve_model,
    _total_message_chars,
    _usage_tokens,
)


def test_resolve_model_opus():
    assert "opus" in _resolve_model("openai/claude-opus-4-7").lower()


def test_resolve_model_haiku():
    assert _resolve_model("claude-haiku-4-5") == MODEL_MAP["haiku"]


def test_resolve_model_defaults_to_sonnet():
    assert _resolve_model("unknown-model") == MODEL_MAP["sonnet"]


def test_messages_to_prompt_flattens_roles():
    prompt, system = _messages_to_prompt(
        [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    )
    assert system == "Be concise"
    assert "User: Hello" in prompt
    assert "Assistant: Hi" in prompt


def test_messages_to_prompt_content_blocks():
    prompt, system = _messages_to_prompt(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "block text"}],
            }
        ]
    )
    assert system is None
    assert "block text" in prompt


def test_usage_tokens_dict_and_object():
    assert _usage_tokens({"input_tokens": 3, "output_tokens": 0}, "input_tokens") == 3

    class Usage:
        input_tokens = 7
        output_tokens = 2

    assert _usage_tokens(Usage(), "input_tokens") == 7
    assert _usage_tokens(Usage(), "output_tokens") == 2


def test_total_message_chars():
    messages = [{"role": "user", "content": "abc"}]
    assert _total_message_chars(messages) == 3

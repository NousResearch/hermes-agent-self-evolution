"""ChatGPT OAuth-backed model support for DSPy.

This adds a practical path for running DSPy against ChatGPT Codex OAuth auth
without requiring an OpenAI API key. Models using the ``chatgpt/`` prefix are
sent to ``https://chatgpt.com/backend-api/codex/responses`` with the bearer
access token resolved from common local auth files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import dspy
from openai import OpenAI

CHATGPT_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_INSTRUCTIONS = "You are a precise assistant. Follow the conversation faithfully."
DEFAULT_AUTH_PROFILE = "openai-codex:default"


class ChatGPTOAuthLM(dspy.BaseLM):
    """DSPy-compatible LM that talks to ChatGPT Codex via OAuth bearer auth."""

    def __init__(
        self,
        model: str,
        *,
        oauth_token: str | None = None,
        auth_file: str | Path | None = None,
        auth_profile: str | None = None,
        base_url: str = CHATGPT_BASE_URL,
        **kwargs: Any,
    ):
        disallowed = {
            "temperature",
            "max_tokens",
            "max_output_tokens",
            "max_completion_tokens",
            "top_p",
        }
        bad = {key: value for key, value in kwargs.items() if value not in (None, False) and key in disallowed}
        if bad:
            key = next(iter(bad))
            raise ValueError(f"ChatGPT OAuth backend does not support {key}")

        clean_model = normalize_chatgpt_model(model)
        super().__init__(model=clean_model, model_type="responses", temperature=None, max_tokens=None, cache=False)
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None and k not in disallowed}
        self.base_url = base_url
        self.auth_file = str(Path(auth_file).expanduser()) if auth_file else None
        self.auth_profile = auth_profile
        self.oauth_token = resolve_chatgpt_oauth_token(
            oauth_token=oauth_token,
            auth_file=auth_file,
            auth_profile=auth_profile,
        )
        self.client = OpenAI(base_url=base_url, api_key=self.oauth_token)

    def copy(self, **kwargs: Any):
        """Rebuild a fresh client instead of deepcopying the OpenAI/httpx stack."""
        supported_attrs = {"model", "base_url", "oauth_token", "auth_file", "auth_profile"}
        init_kwargs = {
            "model": kwargs.pop("model", self.model),
            "oauth_token": kwargs.pop("oauth_token", self.oauth_token),
            "auth_file": kwargs.pop("auth_file", self.auth_file),
            "auth_profile": kwargs.pop("auth_profile", self.auth_profile),
            "base_url": kwargs.pop("base_url", self.base_url),
            **self.kwargs,
        }
        copied = type(self)(**init_kwargs)
        copied.history = []

        for key, value in kwargs.items():
            if key in supported_attrs and hasattr(copied, key):
                setattr(copied, key, value)
                continue
            if value is None:
                copied.kwargs.pop(key, None)
            else:
                copied.kwargs[key] = value

        return copied

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        payload = build_chatgpt_request(model=self.model, prompt=prompt, messages=messages, **{**self.kwargs, **kwargs})
        text_chunks: list[str] = []

        with self.client.responses.stream(**payload) as stream:
            for event in stream:
                if getattr(event, "type", None) == "response.output_text.delta":
                    text_chunks.append(event.delta)
            final = stream.get_final_response()

        return make_dspy_response(
            model=self.model,
            text="".join(text_chunks),
            usage=getattr(final, "usage", None),
        )


def create_lm(model: str, **kwargs: Any):
    """Create the right LM implementation for a model string."""
    if is_chatgpt_model(model):
        return ChatGPTOAuthLM(model, **kwargs)
    return dspy.LM(model, **kwargs)


def is_chatgpt_model(model: str) -> bool:
    return model.startswith("chatgpt/")


def normalize_chatgpt_model(model: str) -> str:
    return model.split("/", 1)[1] if is_chatgpt_model(model) else model


def build_chatgpt_request(
    *,
    model: str,
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    instructions: str | None = None,
    reasoning: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Build the request payload accepted by the ChatGPT Codex responses API."""
    input_messages, inferred_instructions = convert_messages_to_chatgpt_input(messages=messages, prompt=prompt)
    payload = {
        "model": normalize_chatgpt_model(model),
        "instructions": instructions or inferred_instructions or DEFAULT_INSTRUCTIONS,
        "input": input_messages,
        "store": False,
    }
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def convert_messages_to_chatgpt_input(
    *,
    messages: list[dict[str, Any]] | None = None,
    prompt: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if not messages:
        if not prompt:
            raise ValueError("ChatGPT OAuth requests need either messages or a prompt")
        return [_make_input_message("user", prompt)], DEFAULT_INSTRUCTIONS

    instructions: list[str] = []
    input_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = flatten_message_content(message.get("content", ""))
        if not content:
            continue
        if role in {"system", "developer"}:
            instructions.append(content)
            continue
        if role == "assistant":
            input_messages.append(_make_output_message(role, content))
            continue
        input_messages.append(_make_input_message(role, content))

    if not input_messages and prompt:
        input_messages.append(_make_input_message("user", prompt))

    if not input_messages:
        raise ValueError("ChatGPT OAuth request had no user/assistant content to send")

    return input_messages, "\n\n".join(instructions) if instructions else DEFAULT_INSTRUCTIONS


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
    return str(content).strip() if content is not None else ""


def _make_input_message(role: str, text: str) -> dict[str, Any]:
    return {
        "role": "user" if role not in {"user", "assistant"} else role,
        "content": [{"type": "input_text", "text": text}],
    }


def _make_output_message(role: str, text: str) -> dict[str, Any]:
    return {
        "role": role,
        "content": [{"type": "output_text", "text": text}],
    }


def resolve_chatgpt_oauth_token(
    oauth_token: str | None = None,
    auth_file: str | Path | None = None,
    auth_profile: str | None = None,
) -> str:
    if oauth_token:
        return oauth_token

    env_token = os.getenv("CHATGPT_OAUTH_TOKEN")
    if env_token:
        return env_token

    explicit_file = Path(auth_file).expanduser() if auth_file else None
    env_file = Path(os.environ["CHATGPT_OAUTH_FILE"]).expanduser() if os.getenv("CHATGPT_OAUTH_FILE") else None
    profile = auth_profile or os.getenv("CHATGPT_AUTH_PROFILE") or DEFAULT_AUTH_PROFILE

    candidates: list[Path] = []
    for path in [explicit_file, env_file, *default_auth_files()]:
        if path and path not in candidates:
            candidates.append(path)

    for candidate in candidates:
        token = _read_token_from_auth_file(candidate, profile)
        if token:
            return token

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not resolve ChatGPT OAuth token. Set CHATGPT_OAUTH_TOKEN or provide an auth file. "
        f"Checked: {searched}"
    )


def default_auth_files() -> Iterable[Path]:
    home = Path.home()
    return [
        home / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json",
        home / ".codex" / "auth.json",
        home / ".hermes" / "auth.json",
    ]


def _read_token_from_auth_file(path: Path, auth_profile: str) -> str | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())

    profiles = data.get("profiles")
    if isinstance(profiles, dict):
        profile = profiles.get(auth_profile)
        if isinstance(profile, dict) and isinstance(profile.get("access"), str):
            return profile["access"]

    tokens = data.get("tokens")
    if isinstance(tokens, dict) and isinstance(tokens.get("access_token"), str):
        return tokens["access_token"]

    providers = data.get("providers")
    if isinstance(providers, dict):
        provider = providers.get("openai-codex")
        if isinstance(provider, dict):
            provider_tokens = provider.get("tokens")
            if isinstance(provider_tokens, dict) and isinstance(provider_tokens.get("access_token"), str):
                return provider_tokens["access_token"]

    return None


def make_dspy_response(model: str, text: str, usage: Any = None):
    usage_obj = usage or SimpleNamespace(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        cached_tokens=0,
    )
    return SimpleNamespace(
        model=model,
        usage=usage_obj,
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(text=text)])],
    )

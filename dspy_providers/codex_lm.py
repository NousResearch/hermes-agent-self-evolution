"""DSPy v2 BaseLM wrapper for the ChatGPT Codex Responses API.

Compatible models via Codex: gpt-5.4-mini, gpt-5.4, o3, o4-mini, o1, o1-mini, o1-pro.
Unsupported model names passed to CodexLM are auto-mapped to the nearest available model.

Usage:
    # Direct
    from dspy_providers import CodexLM
    lm = CodexLM(model="gpt-5.4-mini", access_token="<token>")
    dspy.settings.configure(lm=lm)

    # Via dspy.LM() — patch dspy.LM before optimizer/eval code runs:
    import dspy
    from dspy_providers import CodexLM
    dspy.LM = CodexLM   # now dspy.LM("gpt-5.4-mini") → CodexLM(model="gpt-5.4-mini")

    # Environment variable
    export CODEX_ACCESS_TOKEN="..."
    export CODEX_BASE_URL="https://chatgpt.com/backend-api/codex"  # optional, default is correct
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Optional

import httpx
import dspy
from dspy.clients.base_lm import BaseLM


# ── Model resolution ───────────────────────────────────────────────────────────

# Maps model names that aren't available via Codex to the closest equivalent.
# Any model not in this map is passed through unchanged.
MODEL_MAP: dict[str, str] = {
    # GPT-4 family → gpt-5.4-mini (best Codex option for general use)
    "gpt-4.1-mini": "gpt-5.4-mini",
    "gpt-4.1":      "gpt-5.4-mini",
    "gpt-4o":       "gpt-5.4-mini",
    "gpt-4o-mini":  "gpt-5.4-mini",
    # Claude / Gemini → gpt-5.4-mini (closest available)
    "claude-3-5-sonnet": "gpt-5.4-mini",
    "claude-3-opus":      "gpt-5.4-mini",
    "claude-3-haiku":     "gpt-5.4-mini",
    "gemini-2.0-flash":   "gpt-5.4-mini",
    "gemini-1.5-flash":   "gpt-5.4-mini",
    # o-series — pass through (Codex supports these natively)
    "o1":      "o1",
    "o1-mini": "o1-mini",
    "o3":      "o3",
    "o3-mini": "o3-mini",
    "o4-mini": "o4-mini",
}


def resolve_model(model: str) -> str:
    """Resolve an unsupported model name to the closest Codex-available model."""
    return MODEL_MAP.get(model, model)


# ── Response objects ────────────────────────────────────────────────────────────

class _Choice:
    """A single completion choice, matching the OpenAI SDK interface DSPy expects."""
    def __init__(self, text: str):
        self.message = SimpleNamespace(content=text)
        self.text = text


class _Response:
    """Mimics the OpenAI SDK response object for DSPy's BaseLM interface.

    DSPy's `_process_lm_response` accesses: response.choices, response.usage,
    response.model, and optionally response._hidden_params.
    """

    def __init__(self, text: str, model: str, usage: dict[str, int]):
        self.choices: list[_Choice] = [_Choice(text)]
        self.usage: dict[str, int] = usage
        self.model: str = model
        self._hidden_params: dict = {}


# ── CodexLM ────────────────────────────────────────────────────────────────────

class CodexLM(BaseLM):
    """DSPy v2 BaseLM that routes calls through the Codex Responses API.

    Args:
        model: The model to request. Unsupported names are resolved via MODEL_MAP.
               Default: "gpt-5.4-mini".
        access_token: Codex API token. Also loaded from CODEX_ACCESS_TOKEN env var.
        base_url: API base URL. Also loaded from CODEX_BASE_URL env var.
                  Defaults to "https://chatgpt.com/backend-api/codex".
        timeout: HTTP request timeout in seconds. Default: 120.
    """

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        access_token: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        super().__init__(model=model)
        self.access_token = access_token or os.environ.get("CODEX_ACCESS_TOKEN", "")
        self.base_url = base_url or os.environ.get(
            "CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex"
        )
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    def _client_get(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _build_payload(self, messages: list[dict], *, model: str) -> dict:
        """Convert DSPy chat message list to Codex Responses API format."""
        instructions = "You are a helpful assistant."
        input_msgs: list[dict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Codex uses a separate instructions field instead of a system message
                instructions = content if isinstance(content, str) else str(content)
                continue

            if isinstance(content, str):
                input_msgs.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Multimodal content blocks
                converted: list[dict] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype == "text":
                        converted.append({"type": "input_text", "text": part.get("text", "")})
                    elif ptype == "image_url":
                        url = part.get("image_url", {})
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        converted.append({"type": "input_image", "image_url": url})
                if converted:
                    input_msgs.append({"role": role, "content": converted})
                else:
                    input_msgs.append({"role": role, "content": ""})
            else:
                input_msgs.append({"role": role, "content": str(content) if content else ""})

        return {
            "model": resolve_model(model),
            "instructions": instructions,
            "input": input_msgs or [{"role": "user", "content": ""}],
            "store": False,
            "stream": True,
        }

    def _parse_sse(self, resp_text: str) -> str:
        """Parse streaming SSE output and return the concatenated text."""
        full_text = ""
        for line in resp_text.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            etype = data.get("type", "")
            if etype == "response.output_text.delta":
                full_text += data.get("delta", "")
            elif etype in ("response.completed", "response.done"):
                break
        return full_text.strip()

    # ── BaseLM interface ───────────────────────────────────────────────────

    def forward(self, prompt=None, messages=None, **kwargs) -> _Response:
        """DSPy v2 BaseLM entry point. Returns an OpenAI-sdk-compatible response."""
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        elif prompt is not None:
            messages = [{"role": "user", "content": prompt}] + list(messages)

        model = kwargs.pop("model", self.model)
        resolved = resolve_model(model)
        payload = self._build_payload(messages, model=resolved)

        resp = self._client_get().post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Codex API {resp.status_code}: {resp.text[:300]}"
            )

        text = self._parse_sse(resp.text)
        return _Response(text=text, model=resolved, usage={
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
        })

    def llm_call(self, messages: list[dict], **kwargs) -> str:
        """Simple string-in-string-out, useful for non-DSPy contexts."""
        result = self.forward(messages=messages, **kwargs)
        return result.choices[0].message.content

    def __repr__(self) -> str:
        return f"CodexLM(model={self.model!r})"

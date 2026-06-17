"""Hermes-aware DSPy LM helpers for local self-evolution runs.

This module keeps the upstream LiteLLM path as the default, but allows
Sunwoo's Hermes OpenAI-Codex OAuth runtime to be used explicitly with model
strings like ``openai-codex/gpt-5.5``. It is intentionally local to the
external optimizer repo; it does not modify active Hermes runtime behavior.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

import dspy


_CODEX_PREFIXES = ("openai-codex/", "codex/")


def is_hermes_codex_model(model: str) -> bool:
    """Return True when a model string explicitly requests Hermes Codex OAuth."""

    m = str(model or "").strip().lower()
    return any(m.startswith(prefix) for prefix in _CODEX_PREFIXES)


def _strip_codex_prefix(model: str) -> str:
    m = str(model or "").strip()
    lower = m.lower()
    for prefix in _CODEX_PREFIXES:
        if lower.startswith(prefix):
            return m[len(prefix) :]
    return m


class HermesCodexLM(dspy.BaseLM):
    """DSPy BaseLM backed by Hermes' Codex OAuth auxiliary client."""

    def __init__(self, model: str, hermes_repo: str | None = None, **kwargs: Any):
        self.codex_model = _strip_codex_prefix(model)
        self.hermes_repo = str(
            hermes_repo
            or os.environ.get("HERMES_AGENT_REPO")
            or Path.home() / ".hermes" / "hermes-agent"
        )
        super().__init__(model=f"openai-codex/{self.codex_model}", model_type="chat", **kwargs)
        self._client = None

    @property
    def supports_response_schema(self) -> bool:
        # DSPy can still fall back to JSON mode if the backend does not accept
        # native schema parameters. Returning False avoids over-advertising.
        return False

    @property
    def supported_params(self) -> set[str]:
        # The Codex backend adapter deliberately ignores temperature/max_tokens;
        # do not advertise structured-output params as natively supported.
        return set()

    def __deepcopy__(self, memo):
        copied = type(self)(self.model, hermes_repo=self.hermes_repo, **copy.deepcopy(self.kwargs, memo))
        copied.history = []
        return copied

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        repo = Path(self.hermes_repo).expanduser()
        if not repo.exists():
            raise RuntimeError(f"Hermes repo not found for Codex LM: {repo}")
        repo_s = str(repo)
        if repo_s not in sys.path:
            sys.path.insert(0, repo_s)
        from agent.auxiliary_client import _build_codex_client

        client, resolved_model = _build_codex_client(self.codex_model)
        if client is None:
            raise RuntimeError("No Hermes OpenAI-Codex OAuth credential available")
        self.codex_model = resolved_model or self.codex_model
        self._client = client
        return self._client

    def forward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any):
        request_messages = messages or [{"role": "user", "content": prompt or ""}]
        merged = {**self.kwargs, **kwargs}
        # Codex endpoint rejects/ignores sampling and max-token knobs; the Hermes
        # adapter strips some of these, but remove them early to keep requests clean.
        for key in (
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
            "top_p",
            "n",
            "stop",
            "response_format",
        ):
            merged.pop(key, None)
        timeout = merged.pop("timeout", None)
        if timeout is None:
            timeout = float(os.environ.get("HSE_CODEX_TIMEOUT", "600"))

        last_error: Exception | None = None
        for attempt in range(2):
            client = self._ensure_client()
            try:
                response = client.chat.completions.create(
                    model=self.codex_model,
                    messages=request_messages,
                    timeout=timeout,
                    **merged,
                )
                usage = getattr(response, "usage", None)
                if usage is not None and not isinstance(usage, dict):
                    response.usage = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    }
                return response
            except Exception as exc:
                last_error = exc
                msg = str(exc).lower()
                transient_or_poisoned = (
                    "client has been closed" in msg
                    or "connection error" in msg
                    or exc.__class__.__name__ in {"APIConnectionError", "TimeoutError"}
                )
                if attempt == 0 and transient_or_poisoned:
                    # Hermes' Codex auxiliary adapter closes the underlying OpenAI
                    # client when a long Responses stream times out. DSPy/GEPA can
                    # then reuse this poisoned LM object across parallel eval calls;
                    # drop it and retry once with a fresh OAuth client.
                    self._client = None
                    continue
                raise
        raise RuntimeError("Hermes Codex LM failed without an exception")  # pragma: no cover


def make_lm(model: str, hermes_repo: str | None = None, **kwargs: Any):
    """Create a DSPy LM, using Hermes Codex OAuth only when explicit."""

    if is_hermes_codex_model(model):
        return HermesCodexLM(model, hermes_repo=hermes_repo, **kwargs)
    return dspy.LM(model, **kwargs)

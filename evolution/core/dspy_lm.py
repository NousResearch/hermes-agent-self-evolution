"""DSPy language-model factory with Hermes OpenAI-Codex OAuth support."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import dspy

from evolution.core.hermes_codex import (
    build_codex_auxiliary_client,
    codex_model_id,
    is_openai_codex_model,
)


class CodexOAuthLM(dspy.BaseLM):
    """DSPy LM backed by Hermes' OpenAI-Codex OAuth transport.

    Model strings use the self-evolution prefix ``openai-codex/<model-id>``.
    Calls are routed through Hermes' ``CodexAuxiliaryClient`` so DSPy can keep
    using a chat-completions-shaped client while Hermes handles OAuth and the
    Codex Responses API adapter.
    """

    def __init__(
        self,
        model: str,
        model_type: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        num_retries: int = 3,
        **kwargs: Any,
    ):
        if model_type != "chat":
            raise ValueError("CodexOAuthLM only supports chat-style DSPy calls")
        self._codex_model = codex_model_id(model)
        self.callbacks = callbacks or []
        self.num_retries = max(1, int(num_retries or 1))
        self._refresh_if_expiring = bool(kwargs.pop("refresh_if_expiring", True))
        # Match useful attributes dspy.LM exposes and some optimizers inspect.
        self.provider = None
        self.finetuning_model = None
        self.launch_kwargs: dict[str, Any] = {}
        self.train_kwargs: dict[str, Any] = {}
        super().__init__(
            model=model,
            model_type="chat",
            temperature=0.0 if temperature is None else temperature,
            max_tokens=1000 if max_tokens is None else max_tokens,
            cache=cache,
            **kwargs,
        )

    def copy(self, **kwargs: Any):
        new_instance = super().copy(**kwargs)
        new_instance._codex_model = codex_model_id(new_instance.model)
        new_instance.callbacks = list(getattr(self, "callbacks", []))
        new_instance.num_retries = max(1, int(getattr(new_instance, "num_retries", self.num_retries) or 1))
        return new_instance

    def _request_kwargs(self, prompt: str | None, messages: list[dict[str, Any]] | None, **kwargs: Any) -> dict[str, Any]:
        request_messages = messages or [{"role": "user", "content": prompt or ""}]
        merged = {**self.kwargs, **kwargs}
        merged.pop("cache", None)
        # DSPy uses rollout_id only for LiteLLM cache behavior; the Codex
        # adapter does not accept or need it.
        if merged.get("rollout_id") is None:
            merged.pop("rollout_id", None)
        merged = {key: value for key, value in merged.items() if value is not None}
        merged.update({"model": self._codex_model, "messages": request_messages})
        return merged

    @staticmethod
    def _ensure_usage_mapping(response: Any) -> Any:
        """DSPy history code calls ``dict(response.usage)``; make that safe."""
        usage = getattr(response, "usage", None)
        if usage is None:
            response.usage = {}
            return response
        if isinstance(usage, dict):
            return response
        try:
            dict(usage)
            return response
        except Exception:
            pass
        mapped = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, key, None)
            if value is not None:
                mapped[key] = value
        response.usage = mapped
        return response

    def forward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any):
        request = self._request_kwargs(prompt, messages, **kwargs)
        last_exc: BaseException | None = None
        for attempt in range(self.num_retries):
            client = None
            try:
                client = build_codex_auxiliary_client(
                    self._codex_model,
                    refresh_if_expiring=self._refresh_if_expiring,
                )
                response = client.chat.completions.create(**request)
                if not getattr(response, "model", None):
                    response.model = self._codex_model
                return self._ensure_usage_mapping(response)
            except Exception as exc:
                last_exc = exc
                if attempt >= self.num_retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 8))
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("CodexOAuthLM failed without returning a response")

    async def aforward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any):
        return await asyncio.to_thread(self.forward, prompt=prompt, messages=messages, **kwargs)


def make_dspy_lm(model: str, **kwargs: Any):
    """Return a DSPy LM, routing ``openai-codex/*`` via Hermes Codex OAuth."""
    if is_openai_codex_model(model):
        # Codex credentials come from Hermes' OAuth store, so an OpenAI-style
        # api_base/api_key pair is meaningless here. Drop them instead of
        # forwarding them into the Codex request payload.
        kwargs.pop("api_base", None)
        kwargs.pop("api_key", None)
        return CodexOAuthLM(model, **kwargs)
    return dspy.LM(model, **kwargs)

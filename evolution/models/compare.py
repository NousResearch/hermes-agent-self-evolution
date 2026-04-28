"""Provider-agnostic OpenAI-compatible model comparison utilities."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from openai import OpenAI


class ModelConfigError(RuntimeError):
    """Raised when model comparison configuration is incomplete."""


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved connection details for an OpenAI-compatible provider."""

    name: str
    base_url: str
    api_key_env: str


class _OpenAICompatibleClient(Protocol):
    chat: Any


ClientFactory = Callable[..., _OpenAICompatibleClient]


def resolve_provider(provider: str, base_url: str | None = None, api_key_env: str | None = None) -> ProviderConfig:
    """Resolve provider connection settings without attaching model IDs.

    Model IDs are supplied by callers. Provider defaults only describe transport:
    where to call and which environment variable should hold the key. Both are
    overrideable from CLI flags or environment variables.
    """
    provider_name = provider.strip().lower()
    if not provider_name:
        raise ModelConfigError("Provider name is required")

    if provider_name == "deepseek":
        resolved_base_url = base_url or os.getenv("HERMES_EVOLVE_DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        resolved_api_key_env = api_key_env or os.getenv("HERMES_EVOLVE_DEEPSEEK_API_KEY_ENV") or "DEEPSEEK_API_KEY"
    elif provider_name in {"openai-compatible", "custom"}:
        resolved_base_url = base_url or os.getenv("HERMES_EVOLVE_BASE_URL")
        resolved_api_key_env = api_key_env or os.getenv("HERMES_EVOLVE_API_KEY_ENV")
        if not resolved_base_url:
            raise ModelConfigError("base_url is required for custom OpenAI-compatible providers")
        if not resolved_api_key_env:
            raise ModelConfigError("api_key_env is required for custom OpenAI-compatible providers")
    else:
        resolved_base_url = base_url or os.getenv(f"HERMES_EVOLVE_{provider_name.upper()}_BASE_URL")
        resolved_api_key_env = api_key_env or os.getenv(f"HERMES_EVOLVE_{provider_name.upper()}_API_KEY_ENV")
        if not resolved_base_url or not resolved_api_key_env:
            raise ModelConfigError(
                f"Unknown provider {provider!r}; pass --base-url and --api-key-env or set provider env overrides"
            )

    return ProviderConfig(name=provider_name, base_url=resolved_base_url, api_key_env=resolved_api_key_env)


def compare_chat_models(
    *,
    models: list[str] | tuple[str, ...],
    prompt: str,
    provider: str = "deepseek",
    base_url: str | None = None,
    api_key_env: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: float = 60.0,
    client_factory: ClientFactory | None = None,
) -> list[dict[str, Any]]:
    """Call each supplied model with the same prompt and return safe comparison rows."""
    model_ids = [model.strip() for model in models if model and model.strip()]
    if not model_ids:
        raise ModelConfigError("At least one --model value is required")
    prompt_text = prompt.strip()
    if not prompt_text:
        raise ModelConfigError("Prompt cannot be empty")

    provider_config = resolve_provider(provider, base_url=base_url, api_key_env=api_key_env)
    api_key = os.getenv(provider_config.api_key_env)
    if not api_key:
        raise ModelConfigError(f"Missing API key environment variable: {provider_config.api_key_env}")

    factory = client_factory or _default_client_factory
    client = factory(api_key=api_key, base_url=provider_config.base_url, timeout=timeout)

    results: list[dict[str, Any]] = []
    for model in model_ids:
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            output_text = _extract_output_text(response)
            usage = _extract_usage(response)
            results.append(
                {
                    "model": model,
                    "ok": True,
                    "latency_ms": latency_ms,
                    **usage,
                    "output_text": output_text,
                    "error": None,
                }
            )
        except Exception as exc:  # pragma: no cover - exercised by live provider failures
            latency_ms = int((time.perf_counter() - started) * 1000)
            results.append(
                {
                    "model": model,
                    "ok": False,
                    "latency_ms": latency_ms,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "output_text": "",
                    "error": _safe_error(exc, api_key),
                }
            )
    return results


def _default_client_factory(*, api_key: str, base_url: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _extract_output_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    return content or ""


def _extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _safe_error(exc: Exception, api_key: str) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return message

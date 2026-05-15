"""Pre-flight LM credential validation.

Before GEPA setup, validate every unique (model, kwargs) tuple via one
tiny ``litellm.completion`` call. Catches ``litellm.AuthenticationError``
(and the ~10 message-pattern variants Hermes itself matches in its
error classifier) and translates them into a
``HermesProviderError`` with provider-specific recovery guidance —
``hermes auth add anthropic``, ``hermes model``, ``hermes login --provider
google-gemini-cli``, etc.

Bypasses ``dspy.LM`` deliberately: dspy adds retry + cache + signature-
parsing logic that can mask the underlying error or interact poorly with
``cache=False`` defaults. ``litellm.completion`` is the layer auth errors
actually originate at — testing that layer directly avoids surprises.

The same ``is_auth_error`` matcher is also used by ``lm_timing_callback``
to detect mid-run auth failures via the ``litellm.failure_callback`` hook
as defense-in-depth.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import litellm

from evolution.core.hermes_provider import (
    HermesProviderError,
    ResolvedLM,
)


class HermesProviderRateLimitError(HermesProviderError):
    """Raised when the preflight probe hits a 429.

    Distinct from ``HermesProviderError`` because the recovery is
    different: rate-limit means "wait and retry", not "your credential
    is bad". Callers may want to back off and retry; auth errors
    require user action.
    """


# Substrings that, when present in a lowercased error message, indicate
# an auth failure. Mirrors hermes-agent's own error-classifier patterns.
_AUTH_ERROR_PATTERNS: Tuple[str, ...] = (
    "invalid api key",
    "invalid_api_key",
    "unauthorized",
    "authentication",
    "token expired",
    "token revoked",
    "access denied",
    "forbidden",
    "401",
    "403",
    # botocore.exceptions.NoCredentialsError surfaces this verbatim through
    # LiteLLM's bedrock provider when the boto3 chain finds no credentials.
    "unable to locate credentials",
)

# Substrings that indicate a rate limit (429).
_RATE_LIMIT_PATTERNS: Tuple[str, ...] = (
    "rate limit",
    "too many requests",
    "429",
)

# Per-provider recovery commands. ``hermes login`` is deprecated upstream;
# current commands are ``hermes auth add <provider>`` or ``hermes model``.
# Gemini is the one exception that still uses the old
# ``hermes login --provider`` form.
_HERMES_AUTH_COMMAND_BY_PROVIDER: Dict[str, str] = {
    "anthropic": "hermes auth add anthropic",
    "openrouter": "export OPENROUTER_API_KEY=sk-or-...",
    "openai": "export OPENAI_API_KEY=sk-...  # or use a different provider",
    "nous": "hermes model  # then select Nous Portal to refresh OAuth",
    "gemini": "hermes login --provider google-gemini-cli",
    "copilot": "gh auth login  # GitHub Copilot uses your gh CLI token",
    "zai": "export GLM_API_KEY=...  # or ZAI_API_KEY",
    "kimi-coding": "export KIMI_API_KEY=...  # or MOONSHOT_API_KEY",
    "minimax": "export MINIMAX_API_KEY=...",
    "huggingface": "export HF_TOKEN=...  # or HUGGINGFACE_API_KEY",
    "nvidia": "export NVIDIA_API_KEY=...",
    "arcee": "export ARCEEAI_API_KEY=...",
    "ollama-cloud": "export OLLAMA_API_KEY=...",
    "kilocode": "export KILOCODE_API_KEY=...",
    "ai-gateway": "export AI_GATEWAY_API_KEY=...",
    "xiaomi": "export XIAOMI_API_KEY=...",
    "bedrock": (
        "export AWS_PROFILE=<profile>  # or export AWS_BEARER_TOKEN_BEDROCK=..., "
        "or run from an instance/role with Bedrock permissions"
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preflight(
    lms: Sequence[ResolvedLM],
    *,
    timeout_seconds: float = 10.0,
    completion_fn: Callable[..., Any] = litellm.completion,
) -> None:
    """Validate every unique (model, kwargs) tuple in ``lms``.

    Makes one ``litellm.completion`` probe per unique LM (deduped across
    roles). Raises ``HermesProviderError`` on the first auth failure with
    a message that names the provider and includes a recovery command.
    Raises ``HermesProviderRateLimitError`` on 429. Lets unrelated
    exceptions propagate as-is (network errors, timeouts, etc. are not
    our problem).

    Cost: ~$0.0001 per probe. For a Hermes single-model setup, dedup
    typically collapses 4 roles to 1 probe. Calls flow through the
    already-registered ``litellm.success_callback`` so they show up in
    the cost ledger.
    """
    seen: set[str] = set()
    for lm in lms:
        key = _dedupe_key(lm)
        if key in seen:
            continue
        seen.add(key)
        _probe_one(
            model=lm.model,
            lm_kwargs=lm.lm_kwargs,
            completion_fn=completion_fn,
            timeout=timeout_seconds,
        )


def is_auth_error(exc: BaseException) -> bool:
    """True if ``exc`` is an auth failure (typed or message-matched)."""
    if isinstance(exc, litellm.AuthenticationError):
        return True
    # PermissionDeniedError exists in litellm but also matches via 403 in
    # the message; handle the type explicitly for robustness against
    # future versions where the message format changes.
    if isinstance(exc, getattr(litellm, "PermissionDeniedError", ())):
        return True
    return _matches_pattern(exc, _AUTH_ERROR_PATTERNS)


def is_rate_limit_error(exc: BaseException) -> bool:
    """True if ``exc`` is a rate-limit failure (typed or message-matched)."""
    if isinstance(exc, litellm.RateLimitError):
        return True
    return _matches_pattern(exc, _RATE_LIMIT_PATTERNS)


def format_auth_error_message(
    *,
    model: str,
    provider_hint: Optional[str],
    underlying: BaseException,
) -> str:
    """Build the user-facing error for an auth failure.

    Names the model + provider, includes the underlying error for
    diagnosis, and points at the right recovery command per provider.
    """
    lines = [
        f"Authentication failed for model '{model}'.",
        f"Underlying error: {type(underlying).__name__}: {underlying}",
        "",
    ]
    command = _HERMES_AUTH_COMMAND_BY_PROVIDER.get(provider_hint or "")
    if command:
        lines.append(f"To fix, run: {command}")
        lines.append(
            "Or pass --optimizer-model with a different provider to "
            "bypass Hermes resolution."
        )
    else:
        lines.append(
            "Set the appropriate provider env var, or pass --optimizer-model "
            "explicitly to bypass Hermes resolution. See "
            "docs/model_resolution.md for the per-provider env var names."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _matches_pattern(exc: BaseException, patterns: Iterable[str]) -> bool:
    text = str(exc).lower()
    return any(p in text for p in patterns)


def _provider_hint_from_model(model: str) -> Optional[str]:
    """Extract the provider prefix from a LiteLLM model string.

    "anthropic/claude-..." -> "anthropic"
    "openrouter/anthropic/claude-..." -> "openrouter" (top-level prefix)
    "just-a-name" -> None
    """
    if "/" not in model:
        return None
    return model.split("/", 1)[0]


def _dedupe_key(lm: ResolvedLM) -> str:
    """Stable hashable key for dedup. ``frozenset(items())`` would crash
    on nested-dict values (rare but possible in custom litellm kwargs);
    ``json.dumps(sort_keys=True)`` handles arbitrary JSON-serializable
    values defensively. ``source`` is excluded — same model + kwargs
    means same probe regardless of where they came from.
    """
    return json.dumps(
        {"model": lm.model, "lm_kwargs": lm.lm_kwargs},
        sort_keys=True,
        default=str,
    )


def _probe_one(
    *,
    model: str,
    lm_kwargs: Dict[str, Any],
    completion_fn: Callable[..., Any],
    timeout: float,
) -> None:
    """Single ``litellm.completion`` probe. Translates auth/rate-limit
    failures; lets unrelated errors propagate.

    ``max_tokens=16`` (not 1) because OpenAI's reasoning-class models
    reject sub-output-budget probes with HTTP 400 ("max_tokens or model
    output limit was reached"). 16 is plenty for an empty-ish response
    and still costs ~$0.0001.
    """
    try:
        completion_fn(
            model=model,
            messages=[{"role": "user", "content": "."}],
            max_tokens=16,
            num_retries=0,
            timeout=timeout,
            **lm_kwargs,
        )
    except BaseException as exc:
        # Order matters: rate-limit check first so a 429 with the word
        # "unauthorized" in some providers' error body doesn't get
        # mis-classified as auth.
        if is_rate_limit_error(exc):
            raise HermesProviderRateLimitError(
                f"Rate limit hit during preflight for model '{model}'. "
                f"Underlying: {type(exc).__name__}: {exc}\n"
                "Wait and retry, or pass --no-preflight to skip the probe."
            ) from exc
        if is_auth_error(exc):
            raise HermesProviderError(
                format_auth_error_message(
                    model=model,
                    provider_hint=_provider_hint_from_model(model),
                    underlying=exc,
                )
            ) from exc
        # 400 BadRequest on a tiny probe usually means the probe payload
        # itself is wrong for this model (some endpoints reject empty
        # messages, max_tokens floors, etc.) — not the user's credential.
        # Letting it through as a non-auth failure would crash the run
        # before GEPA gets to make its own (longer) call which might work
        # fine. Suppress with a debug log; the actual GEPA call will
        # surface real errors at the right time.
        if isinstance(exc, getattr(litellm, "BadRequestError", ())):
            return
        raise

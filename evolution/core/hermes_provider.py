"""Hermes-aware LM defaults.

Resolves the model string + LiteLLM kwargs (api_base, api_key) the framework
should use, by walking ``~/.hermes/config.yaml`` + ``~/.hermes/auth.json``
+ provider env vars in a fixed precedence chain. When nothing usable is
found, raises ``HermesProviderError`` with a message that lists what was
tried and how to fix it.

Mirrors the spirit of Hermes Agent's own ``resolve_runtime_provider`` in
``hermes_cli/runtime_provider.py``, without importing Hermes (importing
``hermes_cli`` triggers TUI initialization). Drift between this slim
resolver and Hermes is acceptable — the closed-loop validator runs Hermes
itself, which always uses Hermes's authoritative resolver.

When the resolved provider is ``custom``/``ollama``/``vllm``/``lmstudio``
or any other OpenAI-wire-compatible provider, returns an ``openai/<model>``
LiteLLM string with ``api_base`` set; LiteLLM hits the override endpoint
without validating the model name against OpenAI's catalog. For Anthropic-
wire providers (Anthropic-direct, plus z.ai / MiniMax / Kimi-coding when
their base_url ends with ``/anthropic``), returns ``anthropic/<model>``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import yaml


Role = Literal["optimizer", "reflection", "eval", "judge"]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedLM:
    """A resolved (model, kwargs, provenance) triple ready for ``dspy.LM``.

    ``lm_kwargs`` is splatted into ``dspy.LM(model, **lm_kwargs, ...)`` —
    typical keys are ``api_base`` and ``api_key``. ``source`` describes
    where the resolution came from for diagnostic logging.

    ``lm_factory`` is an escape hatch for providers that need a non-stock
    ``dspy.LM`` (currently only OpenAI Codex, which carries OAuth state and
    a refresh hook). When set, callers should invoke the factory instead of
    constructing ``dspy.LM(model, **lm_kwargs)`` directly.

    ``provider_hint`` is the canonical Hermes provider name (e.g.
    ``"openai-codex"``) used to look up the right recovery command in the
    auth-error path. When ``None``, the provider hint is derived from the
    LiteLLM model prefix instead.
    """

    model: str
    lm_kwargs: Dict[str, Any]
    source: str
    lm_factory: Optional[Callable[[], Any]] = None
    provider_hint: Optional[str] = None


class HermesProviderError(BaseException):
    """Raised when no model can be resolved from any source, or when an
    auth failure aborts a run.

    Inherits from BaseException (not RuntimeError) so the dspy.Evaluate
    worker pool's ``except Exception`` cannot swallow it. This mirrors
    the trick CostCeilingExceeded uses in evolution/core/lm_timing_callback.py
    to bypass the same swallowing.

    Callers that previously caught with ``except HermesProviderError`` keep
    working unchanged; only the bare ``except Exception`` no longer catches it.
    """


# ---------------------------------------------------------------------------
# Provider tables
# ---------------------------------------------------------------------------

# Hermes provider IDs that have a native LiteLLM provider prefix.
_NATIVE_LITELLM_PREFIX = {
    "anthropic": "anthropic",
    "openrouter": "openrouter",
    "gemini": "gemini",
}

# AWS Bedrock defaults. Bedrock is handled out-of-band: no credential pool,
# no api_key — boto3's default credential chain (env vars, AWS_PROFILE, IAM
# role, IMDS) handles auth, and LiteLLM's bedrock provider does the rest.
_BEDROCK_DEFAULT_REGION = "us-east-1"
_BEDROCK_DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"

# Hermes provider IDs that route via OpenAI-wire-compatible HTTP. LiteLLM
# reaches them as ``openai/<model>`` with a custom ``api_base``. For each,
# the canonical Hermes default endpoint when the user hasn't overridden it.
# Sourced from hermes_cli/auth.py DEFAULT_*_BASE_URL constants — drift from
# Hermes is possible; update by reference when Hermes changes.
_OPENAI_WIRE_DEFAULT_BASE_URL = {
    "openai": "https://api.openai.com/v1",
    "custom": None,  # requires explicit base_url
    "ollama": None,
    "vllm": None,
    "lmstudio": "http://127.0.0.1:1234/v1",
    "llamacpp": None,
    "nous": "https://inference-api.nousresearch.com/v1",
    "copilot": "https://api.githubcopilot.com",
    "zai": "https://api.z.ai/api/coding/paas/v4",
    "kimi-coding": "https://api.moonshot.ai/v1",
    "minimax": "https://api.minimaxi.chat/v1",
    "huggingface": "https://router.huggingface.co/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "xiaomi": "https://api.xiaomi.com/v1",
    "arcee": "https://conductor.arcee.ai/v1",
    "ollama-cloud": "https://ollama.com/v1",
    "kilocode": "https://kilocode.ai/api/openrouter/v1",
    "ai-gateway": "https://ai-gateway.vercel.sh/v1",
}

# Env var(s) that hold an API key for each provider. First non-empty wins.
_PROVIDER_ENV_KEYS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "custom": ("OPENAI_API_KEY",),  # custom OpenAI-compat endpoints commonly reuse this
    "nous": ("NOUS_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "copilot": ("GITHUB_TOKEN",),
    "zai": ("GLM_API_KEY", "ZAI_API_KEY"),
    "kimi-coding": ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
    "minimax": ("MINIMAX_API_KEY",),
    "huggingface": ("HF_TOKEN", "HUGGINGFACE_API_KEY"),
    "nvidia": ("NVIDIA_API_KEY",),
    "xiaomi": ("XIAOMI_API_KEY",),
    "arcee": ("ARCEEAI_API_KEY",),
    "ollama-cloud": ("OLLAMA_API_KEY",),
    "kilocode": ("KILOCODE_API_KEY",),
    "ai-gateway": ("AI_GATEWAY_API_KEY",),
    # Local servers typically require no auth.
    "ollama": (),
    "vllm": (),
    "lmstudio": ("LM_API_KEY",),
    "llamacpp": (),
}

# Aliases that collapse to the same handler.
_PROVIDER_ALIASES = {
    "ollama": "custom",
    "vllm": "custom",
    "llamacpp": "custom",
    # Hermes also accepts these spellings for AWS Bedrock; mirror so
    # hand-edited configs don't get rejected as unknown providers.
    "aws": "bedrock",
    "aws-bedrock": "bedrock",
    "amazon": "bedrock",
    "amazon-bedrock": "bedrock",
}

# Every canonical provider name the resolver knows how to handle. Anything
# else in config.yaml's model.provider is rejected — silent fallthrough
# to the OpenAI-wire default would route a typo'd provider to the wrong
# endpoint with the wrong key.
_KNOWN_PROVIDERS = (
    set(_NATIVE_LITELLM_PREFIX)
    | set(_OPENAI_WIRE_DEFAULT_BASE_URL)
    | {"bedrock", "openai-codex"}
)

# Auto-detect priority order when ``model.provider: auto`` (or unset).
# Mirrors the spirit of Hermes's resolve_provider() chain but compressed
# to what we can detect from auth.json + env vars without OAuth dance.
_AUTO_DETECT_ORDER = (
    "anthropic",
    "openrouter",
    "openai",
    "nous",
    "gemini",
    "copilot",
    "zai",
    "kimi-coding",
    "minimax",
    "huggingface",
    "nvidia",
    "arcee",
    "ollama-cloud",
    "kilocode",
    "ai-gateway",
)

# Standalone fallback model names per provider when neither config.yaml
# nor an explicit override pinned a model. Kept conservative; users on
# obscure providers will get an error guiding them to pass --optimizer-model.
_STANDALONE_DEFAULT_MODEL = {
    "anthropic": "claude-opus-4-5",
    "openrouter": "anthropic/claude-opus-4-5",
    "openai": "gpt-4.1",
}

# Local-server convention: many OpenAI-compat servers accept (and some
# require) any non-empty Authorization header. We pass "EMPTY" to satisfy
# that without leaking a real key. LiteLLM tolerates it for genuinely
# auth-less endpoints.
_LOCAL_SERVER_PLACEHOLDER_KEY = "EMPTY"

# Providers where the absence of an api_key is normal (local/auth-less).
_AUTH_OPTIONAL_PROVIDERS = frozenset({"ollama", "vllm", "llamacpp", "lmstudio"})

# Substrings that, when present in a base_url, indicate a local server.
_LOCAL_BASE_URL_HINTS = ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal")


def _is_auth_optional(requested_provider: str, base_url: str) -> bool:
    """Permit no-credential resolution for explicit local-server providers,
    or for ``provider: custom`` pointed at a localhost endpoint.
    """
    if requested_provider in _AUTH_OPTIONAL_PROVIDERS:
        return True
    if requested_provider == "custom" and base_url:
        return any(hint in base_url for hint in _LOCAL_BASE_URL_HINTS)
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_default_lm(
    *,
    role: Role = "optimizer",
    explicit_model: Optional[str] = None,
    hermes_home: Optional[Path] = None,
) -> ResolvedLM:
    """Pick an LM for the given role.

    Resolution order:
      1. ``explicit_model`` — caller passed a CLI override; return as-is and
         rely on env vars for credentials.
      2. ``config.yaml`` ``model.provider`` (when set and not "auto") +
         credentials from inline ``api_key`` / env / ``auth.json`` pool.
      3. ``config.yaml`` ``model.provider: auto`` (or missing) — slim
         auto-detect: pick the first provider in ``_AUTO_DETECT_ORDER`` that
         has a usable credential.
      4. No Hermes config at all — same auto-detect against env vars only.
      5. Nothing usable → ``HermesProviderError`` with actionable message.
    """
    if explicit_model:
        return ResolvedLM(model=explicit_model, lm_kwargs={}, source="explicit")

    home = hermes_home if hermes_home is not None else Path.home() / ".hermes"
    config = _load_cli_config(home)
    auth_store = _load_auth_store(home)

    model_cfg = (config or {}).get("model") or {}
    requested_provider = (model_cfg.get("provider") or "").strip().lower() or "auto"
    target_model = (model_cfg.get("default") or model_cfg.get("model") or "").strip()
    inline_api_key = (model_cfg.get("api_key") or "").strip()
    inline_base_url = (model_cfg.get("base_url") or "").strip()

    tried: List[str] = []

    if requested_provider == "auto":
        provider, creds = _auto_detect(auth_store, tried)
        if provider is None:
            raise HermesProviderError(_format_hard_error(home, tried, role))
        # In auto mode, we may have no model name from config — fall back to
        # a sane standalone default for the provider we picked.
        if not target_model:
            target_model = _STANDALONE_DEFAULT_MODEL.get(provider) or ""
        if not target_model:
            raise HermesProviderError(
                f"Auto-detected provider '{provider}' but no model name configured. "
                f"Set model.default in ~/.hermes/config.yaml or pass "
                f"--{role}-model explicitly."
            )
        return _build_resolved_lm(
            provider=provider,
            model=target_model,
            api_key=creds.api_key,
            api_base=creds.api_base or inline_base_url,
            source=f"auto:{creds.source}",
        )

    # Explicit provider in config.yaml.
    canonical = _PROVIDER_ALIASES.get(requested_provider, requested_provider)
    if canonical not in _KNOWN_PROVIDERS:
        raise HermesProviderError(
            f"Unknown provider '{requested_provider}' in ~/.hermes/config.yaml. "
            f"Known: {sorted(_KNOWN_PROVIDERS)}. Set model.provider to one of "
            f"these, or pass --{role}-model to bypass Hermes resolution."
        )

    # Bedrock has its own resolution path — boto3's default credential chain
    # handles auth, so there's no inline_api_key, env var, or pool entry to
    # walk. Standalone fallback model lets users with a bare provider: bedrock
    # config.yaml run without setting model.default.
    if canonical == "bedrock":
        return _resolve_bedrock_lm(config=config, target_model=target_model)

    # Codex has its own resolution path — OAuth credentials live in
    # auth.json, there's no env var, and instantiation needs the OAuth state
    # for in-memory token refresh. The CodexLM subclass handles all of that.
    if canonical == "openai-codex":
        return _resolve_codex_lm(
            auth_store=auth_store, target_model=target_model, role=role
        )

    # Nous Portal: when the credential pool entry has a refresh_token (the
    # OAuth-managed flow that hermes model writes), route through NousLM
    # for in-memory OAuth refresh + agent_key minting. The plain env-var
    # NOUS_API_KEY path falls through to the generic OpenAI-wire handler
    # below — no behavior change for that simpler setup.
    if canonical == "nous":
        nous_resolved = _maybe_resolve_nous_lm(
            auth_store=auth_store, target_model=target_model, role=role
        )
        if nous_resolved is not None:
            return nous_resolved

    if not target_model:
        raise HermesProviderError(
            f"~/.hermes/config.yaml sets provider='{requested_provider}' "
            f"but model.default is empty. Set it, or pass --{role}-model."
        )

    creds = _resolve_credentials(
        provider=canonical,
        inline_api_key=inline_api_key,
        auth_store=auth_store,
        tried=tried,
    )
    if creds is None and not _is_auth_optional(requested_provider, inline_base_url):
        raise HermesProviderError(
            _format_hard_error(home, tried, role, requested_provider=requested_provider)
        )

    return _build_resolved_lm(
        provider=canonical,
        model=target_model,
        api_key=(creds.api_key if creds else None),
        api_base=(inline_base_url or (creds.api_base if creds else None)),
        source=(f"hermes-config:{creds.source}" if creds else "hermes-config:no-key"),
    )


def _redact_lm(lm: ResolvedLM) -> Dict[str, Any]:
    """Serialize a ResolvedLM with secrets redacted, for run-config dumps."""
    redacted_kwargs = dict(lm.lm_kwargs)
    if redacted_kwargs.get("api_key"):
        redacted_kwargs["api_key"] = "<REDACTED>"
    return {
        "model": lm.model,
        "lm_kwargs": redacted_kwargs,
        "source": lm.source,
    }


def instantiate_lm(resolved: ResolvedLM, **role_kwargs: Any) -> Any:
    """Build an LM instance from a ResolvedLM, applying role-specific kwargs.

    For providers with a custom factory (currently only Codex), the factory
    captures OAuth state and pins reasoning-model defaults. ``role_kwargs``
    (e.g. ``request_timeout``, ``num_retries``, per-role ``temperature``)
    are applied via ``.copy()`` so the OAuth state survives onto the new
    instance.

    For all other providers, we splat ``lm_kwargs`` directly into a stock
    ``dspy.LM`` — current behavior unchanged.

    The ``dspy`` import is lazy to keep this module lean (it's imported
    early during config load).
    """
    if resolved.lm_factory is not None:
        lm = resolved.lm_factory()
        if role_kwargs:
            lm = lm.copy(**role_kwargs)
        return lm

    import dspy  # noqa: PLC0415 — lazy import to keep module load light

    return dspy.LM(resolved.model, **resolved.lm_kwargs, **role_kwargs)


def resolved_lms_dump(
    *, hermes_home: Optional[Path] = None, **role_overrides: Optional[str]
) -> Dict[str, Dict[str, Any]]:
    """Resolve a set of role → model-string overrides into a dict of redacted
    ResolvedLM entries suitable for a run-config JSON dump. Never raises;
    failures appear as ``{"error": "..."}`` per role so a write_text() at
    the dump site can never fail because the resolver couldn't find creds.

    Use ``eval_=...`` for the eval role since ``eval`` is a Python builtin.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for kwarg_role, explicit in role_overrides.items():
        role = "eval" if kwarg_role == "eval_" else kwarg_role
        try:
            lm = resolve_default_lm(
                role=role, explicit_model=explicit, hermes_home=hermes_home
            )
            out[role] = _redact_lm(lm)
        except HermesProviderError as exc:
            out[role] = {"error": str(exc)}
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Credential:
    api_key: Optional[str]
    api_base: Optional[str]
    source: str


def _load_cli_config(hermes_home: Path) -> Optional[Dict[str, Any]]:
    path = hermes_home / "config.yaml"
    if not path.exists():
        return None
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        # File present but unparseable — distinct from "absent" because the
        # user almost certainly intended for it to be read. Falling through
        # silently would route their run to a different model than they
        # configured.
        print(
            f"warning: {path} exists but failed to parse ({exc}); "
            "falling back to env-var auto-detection.",
            file=sys.stderr,
        )
        return None
    return loaded if isinstance(loaded, dict) else None


def _load_auth_store(hermes_home: Path) -> Dict[str, Any]:
    path = hermes_home / "auth.json"
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"warning: {path} exists but failed to parse ({exc}); "
            "credential pool unavailable.",
            file=sys.stderr,
        )
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _resolve_credentials(
    *,
    provider: str,
    inline_api_key: str,
    auth_store: Dict[str, Any],
    tried: List[str],
) -> Optional[_Credential]:
    """Walk inline-config → env vars → auth.json pool for a chosen provider."""
    if inline_api_key:
        return _Credential(api_key=inline_api_key, api_base=None, source="config-inline")
    tried.append(f"~/.hermes/config.yaml model.api_key for provider={provider}: not set")

    for env_key in _PROVIDER_ENV_KEYS.get(provider, ()):
        val = os.getenv(env_key, "").strip()
        if val:
            return _Credential(api_key=val, api_base=None, source=f"env:{env_key}")
        tried.append(f"env:{env_key}: not set")

    pool_entry = _pick_pool_entry(auth_store, provider)
    if pool_entry is not None:
        api_key = _str_or_none(pool_entry.get("access_token"))
        api_base = _str_or_none(pool_entry.get("base_url"))
        if api_key:
            return _Credential(
                api_key=api_key,
                api_base=api_base,
                source=f"auth.json:credential_pool[{provider}]",
            )
    tried.append(f"~/.hermes/auth.json credential_pool[{provider}]: no usable entry")
    return None


def _str_or_none(value: Any) -> Optional[str]:
    """Defensive accessor for hand-edited auth.json entries — coerce only
    actual strings, return None for any other type so a non-string credential
    field never raises mid-resolution.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _coerce_priority(value: Any) -> int:
    """Sort key for credential pool entries. Hand-edited auth.json may
    store priority as a string ("0") or omit it; either way we never want
    sort to raise TypeError comparing str to int.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return 999


def _is_pool_entry_usable(entry: Dict[str, Any], *, now_epoch: float) -> bool:
    """Skip credentials Hermes has marked exhausted. Mirrors Hermes's
    own pool-rotation behavior — an entry with ``last_status ==
    "exhausted"`` is unavailable until ``last_error_reset_at`` has passed.
    Entries without ``last_status`` are treated as usable for back-compat
    with hand-edited auth.json.
    """
    status = entry.get("last_status")
    if status != "exhausted":
        return True
    reset_at = entry.get("last_error_reset_at")
    if not isinstance(reset_at, (int, float)):
        # Exhausted but no cooldown → treat as permanently bad until Hermes
        # rewrites the entry. Better than silently using a credential the
        # last successful Hermes run knew was dead.
        return False
    return now_epoch >= float(reset_at)


def _pick_pool_entry(auth_store: Dict[str, Any], provider: str) -> Optional[Dict[str, Any]]:
    """Return the highest-priority *usable* credential entry for ``provider``.

    Lowest ``priority`` integer wins (Hermes convention: 0 = highest).
    Also checks ``custom:<provider>`` namespaced keys when the bare
    provider key has no entries. Entries Hermes marked exhausted with a
    future cooldown are skipped — see _is_pool_entry_usable.
    """
    pool = auth_store.get("credential_pool")
    if not isinstance(pool, dict):
        return None
    now = time.time()
    candidates: List[Dict[str, Any]] = []
    for key in (provider, f"custom:{provider}"):
        entries = pool.get(key)
        if isinstance(entries, list):
            candidates.extend(
                e for e in entries
                if isinstance(e, dict) and _is_pool_entry_usable(e, now_epoch=now)
            )
    if not candidates:
        return None
    candidates.sort(key=lambda e: _coerce_priority(e.get("priority")))
    return candidates[0]


def _auto_detect(
    auth_store: Dict[str, Any],
    tried: List[str],
) -> Tuple[Optional[str], _Credential]:
    """Pick the first provider with a usable credential, in priority order."""
    for provider in _AUTO_DETECT_ORDER:
        for env_key in _PROVIDER_ENV_KEYS.get(provider, ()):
            val = os.getenv(env_key, "").strip()
            if val:
                return provider, _Credential(api_key=val, api_base=None, source=f"env:{env_key}")
        pool_entry = _pick_pool_entry(auth_store, provider)
        if pool_entry:
            api_key = (pool_entry.get("access_token") or "").strip()
            api_base = (pool_entry.get("base_url") or "").strip() or None
            if api_key:
                return provider, _Credential(
                    api_key=api_key,
                    api_base=api_base,
                    source=f"auth.json:credential_pool[{provider}]",
                )
    tried.append(
        f"auto-detect: no provider in ({', '.join(_AUTO_DETECT_ORDER)}) "
        "has env-var or auth.json credentials"
    )
    return None, _Credential(api_key=None, api_base=None, source="none")


def _resolve_bedrock_lm(
    *,
    config: Optional[Dict[str, Any]],
    target_model: str,
) -> ResolvedLM:
    """Build a ResolvedLM for AWS Bedrock.

    LiteLLM's ``bedrock/<model-id>`` provider auto-resolves AWS credentials
    from boto3's default chain (``AWS_BEARER_TOKEN_BEDROCK``,
    ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``, ``AWS_PROFILE``, IAM
    role, IMDS). The only kwargs we need to surface are ``aws_region_name``
    and an optional ``aws_profile_name`` from the Hermes config.

    Region resolution: ``bedrock.region`` in config.yaml takes precedence
    over ``AWS_REGION``/``AWS_DEFAULT_REGION`` env vars, then falls back to
    a hardcoded default. Cross-region inference profiles (e.g.
    ``us.anthropic.claude-sonnet-4-6``) work unchanged — LiteLLM passes the
    leading region prefix through to the underlying boto3 call.
    """
    bedrock_cfg = (config or {}).get("bedrock") or {}
    region = (
        (bedrock_cfg.get("region") or "").strip()
        or os.getenv("AWS_REGION", "").strip()
        or os.getenv("AWS_DEFAULT_REGION", "").strip()
        or _BEDROCK_DEFAULT_REGION
    )
    profile = (bedrock_cfg.get("aws_profile_name") or "").strip()
    model_name = target_model or _BEDROCK_DEFAULT_MODEL

    lm_kwargs: Dict[str, Any] = {"aws_region_name": region}
    if profile:
        lm_kwargs["aws_profile_name"] = profile

    return ResolvedLM(
        model=f"bedrock/{model_name}",
        lm_kwargs=lm_kwargs,
        source=f"hermes-config:bedrock(region={region})",
    )


def _resolve_codex_lm(
    *,
    auth_store: Dict[str, Any],
    target_model: str,
    role: Role,
) -> ResolvedLM:
    """Build a ResolvedLM for OpenAI Codex Responses API.

    Codex creds live in ``~/.hermes/auth.json`` under
    ``credential_pool["openai-codex"]``: an access token, a refresh token,
    an expiry timestamp, and an optional base URL override. We pick the
    highest-priority pool entry (lowest ``priority`` integer wins, mirroring
    Hermes), then return a ResolvedLM whose ``lm_factory`` instantiates a
    ``CodexLM`` carrying the OAuth state. The factory captures the OAuth
    state in a closure so the resolver doesn't need to plumb OAuth fields
    through every consumer.

    The ``CodexLM`` import is lazy to avoid a circular dependency: codex_lm
    imports HermesProviderError from this module.
    """
    pool_entry = _pick_pool_entry(auth_store, "openai-codex")
    if pool_entry is None:
        raise HermesProviderError(
            f"~/.hermes/config.yaml sets provider='openai-codex' but no usable "
            f"entry was found in ~/.hermes/auth.json credential_pool[\"openai-codex\"]. "
            f"Run `hermes auth add openai-codex` to authenticate, or pass "
            f"--{role}-model to bypass Hermes resolution."
        )

    access_token = _str_or_none(pool_entry.get("access_token")) or ""
    if not access_token:
        raise HermesProviderError(
            "~/.hermes/auth.json credential_pool[\"openai-codex\"] entry has "
            "no access_token. Run `hermes auth add openai-codex` to "
            "re-authenticate."
        )

    refresh_token = _str_or_none(pool_entry.get("refresh_token")) or ""
    base_url = _str_or_none(pool_entry.get("base_url"))
    expires_at_raw = pool_entry.get("expires_at")
    expires_at = (
        float(expires_at_raw)
        if isinstance(expires_at_raw, (int, float))
        else None
    )

    # Codex requires a model name from config.yaml — there's no sensible
    # standalone default since the user's ChatGPT plan determines what they
    # have access to (gpt-5, gpt-5-codex, etc.).
    if not target_model:
        raise HermesProviderError(
            "~/.hermes/config.yaml sets provider='openai-codex' but "
            f"model.default is empty. Set it (e.g., 'gpt-5-codex' or 'gpt-5'), "
            f"or pass --{role}-model."
        )

    # Lazy import to break the circular dependency with codex_lm.
    from evolution.core.codex_lm import CodexLM as _CodexLM
    from evolution.core.codex_headers import DEFAULT_CODEX_BASE_URL

    effective_base_url = base_url or DEFAULT_CODEX_BASE_URL

    # Codex models speak the OpenAI Responses wire protocol, so LiteLLM
    # routes them via its OpenAI provider. Prefix with ``openai/`` if the
    # user's config didn't already do so — the prefix is what tells LiteLLM
    # to honor api_base + Bearer auth instead of inferring routing from the
    # bare model name.
    litellm_model = (
        target_model if "/" in target_model else f"openai/{target_model}"
    )

    def _factory() -> Any:
        return _CodexLM(
            model=litellm_model,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            base_url=effective_base_url,
        )

    return ResolvedLM(
        model=litellm_model,
        lm_kwargs={},
        source=f"hermes-config:openai-codex(base_url={effective_base_url})",
        lm_factory=_factory,
        provider_hint="openai-codex",
    )


def _maybe_resolve_nous_lm(
    *,
    auth_store: Dict[str, Any],
    target_model: str,
    role: Role,
) -> Optional[ResolvedLM]:
    """Build a NousLM-backed ResolvedLM when the pool entry has a
    refresh_token (the OAuth-managed signal). Returns None for env-var
    or hand-edited entries with an agent_key already present (caller
    falls through to the generic OpenAI-wire handler), and raises for
    partial OAuth setups (access_token without refresh_token or
    agent_key) so the operator gets a `hermes model` recovery hint
    instead of a silent inference 401.

    See ``evolution/core/nous_lm.py`` for the two-stage credential
    model and the in-memory refresh + mint flow.
    """
    pool_entry = _pick_pool_entry(auth_store, "nous")
    if pool_entry is None:
        # No pool entry at all → hint operator at the right recovery
        # rather than falling through silently to env-var resolution
        # that probably also won't work.
        raise HermesProviderError(
            "~/.hermes/config.yaml sets provider='nous' but no usable "
            "entry was found in ~/.hermes/auth.json credential_pool[\"nous\"]. "
            "Run `hermes model` and select Nous Portal to authenticate, "
            f"or pass --{role}-model to bypass Hermes resolution."
        )

    refresh_token = _str_or_none(pool_entry.get("refresh_token"))
    agent_key = _str_or_none(pool_entry.get("agent_key"))
    if not refresh_token:
        # An entry with agent_key set is plausibly env-var-style or
        # hand-edited inference-only — let it fall through to the
        # generic OpenAI-wire handler with whatever Bearer it carries.
        # An entry with NEITHER refresh_token NOR agent_key is almost
        # certainly a partial OAuth setup (interrupted hermes model run,
        # or the portal handed back access_token only). Inference would
        # 401 with no breadcrumb pointing at the missing credentials, so
        # raise here with a specific recovery hint.
        if agent_key is None:
            raise HermesProviderError(
                "~/.hermes/auth.json credential_pool[\"nous\"] entry has "
                "an access_token but no refresh_token or agent_key — "
                "looks like a partial OAuth setup. Run `hermes model` "
                "and select Nous Portal to complete authentication, or "
                f"pass --{role}-model to bypass Hermes resolution."
            )
        return None

    access_token = _str_or_none(pool_entry.get("access_token"))
    if not access_token:
        raise HermesProviderError(
            "~/.hermes/auth.json credential_pool[\"nous\"] entry has no "
            "access_token. Run `hermes model` and select Nous Portal to "
            "re-authenticate."
        )

    if not target_model:
        raise HermesProviderError(
            "~/.hermes/config.yaml sets provider='nous' but model.default "
            f"is empty. Set it (e.g., 'Hermes-4-405B'), or pass --{role}-model."
        )

    # Lazy import to break the circular dependency with nous_lm.
    from evolution.core.nous_lm import (  # noqa: PLC0415
        NousLM as _NousLM,
        NOUS_INFERENCE_BASE_URL,
        NOUS_PORTAL_BASE_URL,
    )
    from evolution.core.oauth_helpers import parse_iso_or_epoch  # noqa: PLC0415

    inference_base_url = (
        _str_or_none(pool_entry.get("inference_base_url"))
        or _str_or_none(pool_entry.get("base_url"))
        or NOUS_INFERENCE_BASE_URL
    )
    oauth_expires_at = parse_iso_or_epoch(pool_entry.get("expires_at"))
    agent_key = _str_or_none(pool_entry.get("agent_key"))
    agent_key_expires_at = parse_iso_or_epoch(pool_entry.get("agent_key_expires_at"))

    def _factory() -> Any:
        return _NousLM(
            model=f"openai/{target_model}",
            access_token=access_token,
            refresh_token=refresh_token,
            oauth_expires_at=oauth_expires_at,
            agent_key=agent_key,
            agent_key_expires_at=agent_key_expires_at,
            portal_base_url=NOUS_PORTAL_BASE_URL,
            inference_base_url=inference_base_url,
        )

    return ResolvedLM(
        model=f"openai/{target_model}",
        lm_kwargs={},
        source=f"hermes-config:nous(inference_base_url={inference_base_url})",
        lm_factory=_factory,
        provider_hint="nous",
    )


def _build_resolved_lm(
    *,
    provider: str,
    model: str,
    api_key: Optional[str],
    api_base: Optional[str],
    source: str,
) -> ResolvedLM:
    """Compose the ResolvedLM, applying the provider → LiteLLM mapping table."""
    canonical = _PROVIDER_ALIASES.get(provider, provider)
    effective_base = api_base or _OPENAI_WIRE_DEFAULT_BASE_URL.get(canonical)

    # Wire-mode flip: providers that route via Anthropic Messages when the
    # base_url ends with /anthropic (z.ai, MiniMax, Kimi-coding configs).
    if effective_base and "/anthropic" in effective_base:
        kwargs: Dict[str, Any] = {"api_base": effective_base.rstrip("/")}
        if api_key:
            kwargs["api_key"] = api_key
        return ResolvedLM(model=f"anthropic/{model}", lm_kwargs=kwargs, source=source)

    # Native LiteLLM prefixes — no api_base (LiteLLM uses the provider's
    # canonical endpoint), api_key only.
    if canonical in _NATIVE_LITELLM_PREFIX and not effective_base:
        prefix = _NATIVE_LITELLM_PREFIX[canonical]
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        return ResolvedLM(model=f"{prefix}/{model}", lm_kwargs=kwargs, source=source)

    # Anthropic + custom api_base (Azure Anthropic, regional endpoints).
    if canonical == "anthropic" and effective_base:
        kwargs = {"api_base": effective_base}
        if api_key:
            kwargs["api_key"] = api_key
        return ResolvedLM(model=f"anthropic/{model}", lm_kwargs=kwargs, source=source)

    # Default: OpenAI-wire-compatible (most providers + custom).
    kwargs = {}
    if effective_base:
        kwargs["api_base"] = effective_base
    if api_key:
        kwargs["api_key"] = api_key
    elif canonical in _AUTH_OPTIONAL_PROVIDERS:
        kwargs["api_key"] = _LOCAL_SERVER_PLACEHOLDER_KEY
    return ResolvedLM(model=f"openai/{model}", lm_kwargs=kwargs, source=source)


def _format_hard_error(
    hermes_home: Path,
    tried: List[str],
    role: Role,
    requested_provider: Optional[str] = None,
) -> str:
    """Build the actionable error message for the no-credentials case."""
    config_path = hermes_home / "config.yaml"
    config_status = "found" if config_path.exists() else "not found"
    header = [f"No model could be resolved for role={role}."]
    if requested_provider:
        header.append(f"Provider requested via config.yaml: {requested_provider}")
    lines = [
        *header,
        "",
        "Tried in order:",
        f"  - --{role}-model flag: not set",
        f"  - {config_path}: {config_status}",
    ]
    lines.extend(f"  - {step}" for step in tried)
    lines.extend(
        [
            "",
            "Either:",
            "  (a) Configure Hermes Agent: https://github.com/NousResearch/hermes-agent",
            "  (b) Set a provider env var (e.g. export ANTHROPIC_API_KEY=sk-ant-...,",
            "      export OPENROUTER_API_KEY=sk-or-..., export OPENAI_API_KEY=sk-...)",
            f"  (c) Pass --{role}-model explicitly "
            f"(e.g. --{role}-model anthropic/claude-opus-4-5)",
        ]
    )
    return "\n".join(lines)

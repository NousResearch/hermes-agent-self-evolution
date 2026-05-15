"""DSPy LM subclass for OpenAI Codex Responses API.

Codex routes through ``dspy.LM(model_type="responses", api_base=..., ...)``
→ ``litellm.responses(...)`` → standard Responses HTTP call. DSPy already
converts chat messages to Responses input format and parses Responses-shaped
output. The Codex-specific work this class adds:

  - Cloudflare-mitigation headers (originator, User-Agent, ChatGPT-Account-ID).
    Without these, every call from a non-residential IP gets 403'd by
    Cloudflare regardless of OAuth correctness. See ``codex_headers``.
  - OAuth refresh: the access token expires every ~30 min. We refresh
    in-memory (no ``auth.json`` writeback) before each call when the token
    is within a 120s skew window, mirroring Hermes's refresh skew.
  - Cross-instance refresh state sharing: the four LM roles (optimizer,
    reflection, eval, judge) each instantiate a separate CodexLM, but
    share OAuth state through a module-level cache keyed by the initial
    refresh_token. Without this, four parallel workers entering the skew
    window simultaneously would each POST to ``auth.openai.com`` and three
    of them would get ``refresh_token_reused`` back.

In-memory refresh is intentional — long evolutions (>30 min on a fresh
token) need to re-trigger ``hermes auth add openai-codex`` to refresh the
on-disk store. Avoids write-conflict surface with concurrent Hermes
sessions that may also be refreshing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import dspy
import httpx

from evolution.core.codex_headers import (
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    DEFAULT_CODEX_BASE_URL,
    codex_cloudflare_headers,
)
from evolution.core.hermes_provider import HermesProviderError


@dataclass
class _SharedRefreshState:
    """OAuth state shared across CodexLM instances for the same account.

    Keyed in ``_STATE_BY_KEY`` by the initial refresh_token observed at
    instance construction. All CodexLMs created from the same resolver
    factory share the same key, so a refresh by any one of them is
    immediately visible to the others.
    """

    access_token: str
    refresh_token: str
    expires_at: Optional[float]
    lock: threading.Lock

    def __deepcopy__(self, memo):
        # CodexLM uses dspy.LM.copy() (which deepcopies the whole instance)
        # to apply role-specific kwargs. Locks aren't deep-copyable, and
        # — more importantly — the *point* of shared state is to be shared.
        # A copied CodexLM must observe refreshes performed against the
        # original, so the copy keeps the same _SharedRefreshState reference.
        return self


_STATE_BY_KEY: Dict[str, _SharedRefreshState] = {}
_STATE_REGISTRY_LOCK = threading.Lock()


def _get_or_register_state(
    *,
    key: str,
    access_token: str,
    refresh_token: str,
    expires_at: Optional[float],
) -> _SharedRefreshState:
    """Register a new shared state on first observation; return the existing
    one on subsequent calls. The first instance's OAuth values win — they're
    the freshest at startup and any later instance with the same key was
    constructed from the same source.
    """
    with _STATE_REGISTRY_LOCK:
        if key not in _STATE_BY_KEY:
            _STATE_BY_KEY[key] = _SharedRefreshState(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=expires_at,
                lock=threading.Lock(),
            )
        return _STATE_BY_KEY[key]


def _reset_state_for_tests() -> None:
    """Test-only: clear the module-level state cache so each test starts
    from a clean slate. Tests that share state across cases would observe
    refreshes from prior tests bleeding through.
    """
    with _STATE_REGISTRY_LOCK:
        _STATE_BY_KEY.clear()


class CodexLM(dspy.LM):
    """DSPy LM for the OpenAI Codex Responses API."""

    def __init__(
        self,
        model: str,
        *,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_at: Optional[float] = None,
        base_url: str = DEFAULT_CODEX_BASE_URL,
        **kwargs: Any,
    ) -> None:
        # gpt-5-class reasoning models require temperature=1.0 and
        # max_tokens >= 16000. DSPy enforces this in its own __init__ for
        # known reasoning model patterns; pre-set so callers can override
        # via kwargs but the defaults are sane.
        kwargs.setdefault("temperature", 1.0)
        kwargs.setdefault("max_tokens", 16000)
        kwargs["api_base"] = base_url
        kwargs["api_key"] = access_token
        kwargs["extra_headers"] = codex_cloudflare_headers(access_token)

        super().__init__(model=model, model_type="responses", **kwargs)

        # The lookup key for shared refresh state. Falls back to id(self)
        # when there's no refresh_token, so test-scenarios with synthetic
        # creds get per-instance isolation rather than colliding on the
        # empty-string key.
        self._state_key = refresh_token or f"no-refresh:{id(self)}"
        self._shared_state = _get_or_register_state(
            key=self._state_key,
            access_token=access_token,
            refresh_token=refresh_token or "",
            expires_at=float(expires_at) if expires_at is not None else None,
        )

    # ------------------------------------------------------------------
    # Refresh path
    # ------------------------------------------------------------------

    def _state_needs_refresh(self) -> bool:
        if self._shared_state.expires_at is None:
            return False
        return (
            time.time() + CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
            >= self._shared_state.expires_at
        )

    def _sync_from_shared_state(self) -> None:
        """Pull the latest OAuth values out of shared state into self.kwargs.

        Every call goes through this — it's cheap (dict reads), and ensures
        a refresh by any sibling CodexLM instance is observed before we hit
        the wire. ``api_key`` is excluded from dspy.LM's request cache key
        (see ``ignored_args_for_cache_key``), so updates don't invalidate
        cached responses.
        """
        self.kwargs["api_key"] = self._shared_state.access_token
        self.kwargs["extra_headers"] = codex_cloudflare_headers(
            self._shared_state.access_token
        )

    def _refresh_if_expiring(self) -> None:
        """Acquire the per-account lock and refresh if needed.

        Double-checked locking: when N threads enter the skew window
        simultaneously, only the first one performs the HTTP round-trip;
        the others observe the updated ``expires_at`` after acquiring the
        lock and return without touching the network.
        """
        if not self._state_needs_refresh():
            self._sync_from_shared_state()
            return
        if not self._shared_state.refresh_token:
            raise HermesProviderError(
                f"Codex access token for model '{self.model}' is expiring but "
                f"no refresh_token is available. Run `hermes auth add openai-codex` "
                f"to re-authenticate."
            )

        with self._shared_state.lock:
            if not self._state_needs_refresh():
                self._sync_from_shared_state()
                return
            self._do_refresh()
            self._sync_from_shared_state()

    def _do_refresh(self) -> None:
        """POST the refresh-token grant; on success, mutate shared state."""
        try:
            with httpx.Client(timeout=httpx.Timeout(20.0)) as client:
                response = client.post(
                    CODEX_OAUTH_TOKEN_URL,
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self._shared_state.refresh_token,
                        "client_id": CODEX_OAUTH_CLIENT_ID,
                    },
                )
        except httpx.HTTPError as exc:
            raise HermesProviderError(
                f"Codex token refresh request failed ({exc}). Check network "
                f"connectivity, then re-try; if the failure persists, run "
                f"`hermes auth add openai-codex`."
            ) from exc

        if response.status_code != 200:
            raise HermesProviderError(_format_refresh_error(response))

        try:
            payload = response.json()
        except ValueError as exc:
            raise HermesProviderError(
                "Codex token refresh returned invalid JSON. "
                "Run `hermes auth add openai-codex` to re-authenticate."
            ) from exc

        new_access = payload.get("access_token")
        if not isinstance(new_access, str) or not new_access.strip():
            raise HermesProviderError(
                "Codex token refresh response was missing access_token. "
                "Run `hermes auth add openai-codex` to re-authenticate."
            )

        # The OAuth response may rotate the refresh token too; honor that.
        new_refresh = payload.get("refresh_token")
        if isinstance(new_refresh, str) and new_refresh.strip():
            self._shared_state.refresh_token = new_refresh.strip()

        # ``expires_in`` is seconds-from-now. Codex tokens are typically
        # ~28 days but we honor whatever the server tells us. Conservative
        # 1h fallback if the field is missing keeps the next call from
        # racing to the wire again immediately.
        expires_in = payload.get("expires_in")
        if isinstance(expires_in, (int, float)) and expires_in > 0:
            self._shared_state.expires_at = time.time() + float(expires_in)
        else:
            self._shared_state.expires_at = time.time() + 3600.0

        self._shared_state.access_token = new_access.strip()

    # ------------------------------------------------------------------
    # forward / aforward override — refresh, then delegate to dspy.LM
    # ------------------------------------------------------------------

    def forward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        self._refresh_if_expiring()
        return super().forward(prompt=prompt, messages=messages, **kwargs)

    async def aforward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        self._refresh_if_expiring()
        return await super().aforward(prompt=prompt, messages=messages, **kwargs)


# ----------------------------------------------------------------------
# Error classification — mirror hermes-agent error taxonomy
# ----------------------------------------------------------------------

# Error codes the OpenAI OAuth endpoint returns that indicate the refresh
# token is permanently invalid; the user must re-authenticate.
_RELOGIN_ERROR_CODES = frozenset(
    {"invalid_grant", "invalid_token", "invalid_request"}
)


def _format_refresh_error(response: httpx.Response) -> str:
    """Translate the OAuth error body into an actionable user message.

    Mirrors hermes-agent's classification at hermes_cli/auth.py for refresh
    failures: surface ``refresh_token_reused`` specifically (a common
    operator footgun when running multiple OAuth clients simultaneously),
    and label any 401/403 from the OAuth endpoint as relogin-required.
    """
    code = "codex_refresh_failed"
    detail = f"status {response.status_code}"

    try:
        body = response.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                # OpenAI shape: {"error": {"code": ..., "message": ...}}
                nested_code = err.get("code") or err.get("type")
                if isinstance(nested_code, str) and nested_code.strip():
                    code = nested_code.strip()
                nested_msg = err.get("message")
                if isinstance(nested_msg, str) and nested_msg.strip():
                    detail = nested_msg.strip()
            elif isinstance(err, str) and err.strip():
                # OAuth-spec shape: {"error": "code", "error_description": "..."}
                code = err.strip()
                desc = body.get("error_description") or body.get("message")
                if isinstance(desc, str) and desc.strip():
                    detail = desc.strip()
    except ValueError:
        pass

    if code == "refresh_token_reused":
        return (
            "Codex refresh token was already consumed by another client "
            "(commonly Codex CLI or VS Code). Run `codex` in your terminal "
            "to generate fresh tokens, then `hermes auth add openai-codex` "
            "to re-sync."
        )

    if code in _RELOGIN_ERROR_CODES or response.status_code in (401, 403):
        return (
            f"Codex token refresh failed ({code}: {detail}). "
            f"Run `hermes auth add openai-codex` to re-authenticate."
        )

    return (
        f"Codex token refresh failed ({code}: {detail}). "
        f"Re-try; if the failure persists, run `hermes auth add openai-codex`."
    )

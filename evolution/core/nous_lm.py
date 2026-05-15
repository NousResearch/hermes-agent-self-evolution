"""DSPy LM subclass for Nous Portal — OAuth refresh + agent_key minting.

Nous Portal uses a two-stage credential model that's meaningfully different
from Codex:

  1. **OAuth access_token** (long-lived, days). Refreshable via the standard
     refresh_token grant at ``{portal}/api/oauth/token``.
  2. **agent_key** (short-lived, ~30 minutes). Minted from the access_token
     by POSTing to ``{portal}/api/oauth/agent-key``. The inference endpoint
     (``inference-api.nousresearch.com``) requires the **agent_key** as
     Bearer — not the access_token.

Mirrors Hermes's ``resolve_nous_runtime_credentials`` flow in
``hermes_cli/auth.py``: refresh the OAuth token first if expiring, then
mint a fresh agent_key from it. On inference 401, force re-mint and retry
once. State is shared across LM instances via ``_STATE_BY_KEY`` so the
four LM roles (optimizer, reflection, eval, judge) coordinate through
one lock and one mint per refresh window — without this, four parallel
workers entering the skew window would each mint and three would race
the portal's single-use refresh-token rotation.

In-memory only — no auth.json writeback. Long evolutions (>30 min on a
fresh agent_key) refresh in-process, but the on-disk store stays at
whatever ``hermes model`` last wrote. Avoids the write-conflict surface
with concurrent Hermes sessions that may also be refreshing.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import dspy
import httpx
import litellm

from evolution.core.hermes_provider import HermesProviderError
from evolution.core.oauth_helpers import parse_iso_or_epoch

_log = logging.getLogger(__name__)


# Hardcoded defaults; the constructor reads ``HERMES_PORTAL_BASE_URL`` and
# ``NOUS_INFERENCE_BASE_URL`` env vars at instance time so tests and stage
# setups can override them post-import. Module-level capture would freeze
# the values at first import, before any test or operator could intervene.
NOUS_PORTAL_BASE_URL = "https://portal.nousresearch.com"
NOUS_INFERENCE_BASE_URL = "https://inference-api.nousresearch.com/v1"
NOUS_OAUTH_CLIENT_ID = "hermes-cli"

# Refresh OAuth access tokens 2 minutes before they expire and re-mint
# the inference agent_key 2 minutes before it expires. Mirrors Hermes's
# ``ACCESS_TOKEN_REFRESH_SKEW_SECONDS`` so multi-process workloads don't
# race each other onto the wire on different cadences.
OAUTH_REFRESH_SKEW_SECONDS = 120
AGENT_KEY_REFRESH_SKEW_SECONDS = 120
# Ask the portal for at least 30 minutes of agent_key TTL on each mint;
# the portal is free to grant more. Mirrors Hermes's
# ``DEFAULT_AGENT_KEY_MIN_TTL_SECONDS``.
AGENT_KEY_MIN_TTL_SECONDS = 30 * 60


@dataclass
class _SharedNousState:
    """OAuth + agent_key state shared across NousLM instances for the same
    Nous account.

    Keyed in ``_STATE_BY_KEY`` by the initial refresh_token observed at
    construction. All NousLMs created from the same resolver factory share
    the same key, so a refresh or mint by any one of them is visible to
    the others — without this, four parallel workers entering the skew
    window simultaneously would each POST refresh+mint and three would
    receive ``refresh_token_reused`` from the portal.
    """

    access_token: str
    refresh_token: str
    oauth_expires_at: Optional[float]
    agent_key: Optional[str]
    agent_key_expires_at: Optional[float]
    lock: threading.Lock

    def __post_init__(self) -> None:
        # An agent_key without an expiry trips _agent_key_needs_mint into
        # "always re-mint" mode, which is defensive but masks the
        # construction-time mistake of seeding partial state. Pin the
        # invariant so the failure surfaces loudly at construction.
        if (self.agent_key and self.agent_key_expires_at is None) or (
            self.agent_key_expires_at is not None and not self.agent_key
        ):
            raise ValueError(
                "_SharedNousState: agent_key and agent_key_expires_at "
                "must be set together (or both None)"
            )

    def __deepcopy__(self, memo):
        # NousLM uses dspy.LM.copy() (which deepcopies the whole instance)
        # to apply role-specific kwargs. Locks aren't deep-copyable, and
        # the *point* of shared state is to be shared. A copied NousLM
        # must observe refreshes/mints performed against the original, so
        # the copy keeps the same _SharedNousState reference.
        return self


_STATE_BY_KEY: Dict[str, _SharedNousState] = {}
_STATE_REGISTRY_LOCK = threading.Lock()


def _get_or_register_state(
    *,
    key: str,
    access_token: str,
    refresh_token: str,
    oauth_expires_at: Optional[float],
    agent_key: Optional[str],
    agent_key_expires_at: Optional[float],
) -> _SharedNousState:
    """Register a new shared state on first observation; return the existing
    one on subsequent calls. The first instance's OAuth values win — they're
    the freshest at startup and any later instance with the same key was
    constructed from the same source.
    """
    with _STATE_REGISTRY_LOCK:
        if key not in _STATE_BY_KEY:
            _STATE_BY_KEY[key] = _SharedNousState(
                access_token=access_token,
                refresh_token=refresh_token,
                oauth_expires_at=oauth_expires_at,
                agent_key=agent_key,
                agent_key_expires_at=agent_key_expires_at,
                lock=threading.Lock(),
            )
        return _STATE_BY_KEY[key]


def _reset_state_for_tests() -> None:
    """Test-only: clear the module-level state cache so each test starts
    from a clean slate. Tests that share state across cases would observe
    refreshes/mints from prior tests bleeding through.
    """
    with _STATE_REGISTRY_LOCK:
        _STATE_BY_KEY.clear()


class NousLM(dspy.LM):
    """DSPy LM for Nous Portal — handles OAuth refresh + agent_key minting."""

    def __init__(
        self,
        model: str,
        *,
        access_token: str,
        refresh_token: str,
        oauth_expires_at: Optional[float] = None,
        agent_key: Optional[str] = None,
        agent_key_expires_at: Optional[float] = None,
        portal_base_url: Optional[str] = None,
        inference_base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Resolve URLs at construction time (not module-import time) so
        # tests and stage setups can override via env vars after the
        # framework is loaded. ``HERMES_PORTAL_BASE_URL`` is Hermes's own
        # variable name — sharing keeps a single ``export`` portable.
        effective_portal = (
            portal_base_url
            or os.getenv("HERMES_PORTAL_BASE_URL")
            or NOUS_PORTAL_BASE_URL
        )
        effective_inference = (
            inference_base_url
            or os.getenv("NOUS_INFERENCE_BASE_URL")
            or NOUS_INFERENCE_BASE_URL
        )

        kwargs["api_base"] = effective_inference
        kwargs["api_key"] = agent_key or ""

        super().__init__(model=model, **kwargs)

        self._portal_base_url = effective_portal

        # The lookup key for shared state — falls back to id(self) so test
        # scenarios with synthetic creds get per-instance isolation rather
        # than colliding on the empty-string key.
        self._state_key = refresh_token or f"no-refresh:{id(self)}"
        self._shared_state = _get_or_register_state(
            key=self._state_key,
            access_token=access_token,
            refresh_token=refresh_token,
            oauth_expires_at=oauth_expires_at,
            agent_key=agent_key,
            agent_key_expires_at=agent_key_expires_at,
        )

        # Pay the mint cost at construction so the first forward() doesn't
        # see a synchronous round-trip surprise.
        self._ensure_credentials()

    # ------------------------------------------------------------------
    # Refresh + mint orchestration
    # ------------------------------------------------------------------

    def _oauth_needs_refresh(self) -> bool:
        if self._shared_state.oauth_expires_at is None:
            # Unknown expiry → don't speculatively refresh; the mint
            # call's own 401-triggers-refresh-retry path catches a
            # genuinely-dead access_token. Note _agent_key_needs_mint
            # makes the opposite choice (defaults True on unknown
            # expiry) because there's no equivalent recovery for a
            # missing agent_key — inference would just 401 with no
            # built-in retry.
            return False
        return (
            time.time() + OAUTH_REFRESH_SKEW_SECONDS
            >= self._shared_state.oauth_expires_at
        )

    def _agent_key_needs_mint(self) -> bool:
        if not self._shared_state.agent_key:
            return True
        if self._shared_state.agent_key_expires_at is None:
            # Have a key but no expiry → re-mint defensively. See
            # _oauth_needs_refresh for the asymmetric reasoning.
            return True
        return (
            time.time() + AGENT_KEY_REFRESH_SKEW_SECONDS
            >= self._shared_state.agent_key_expires_at
        )

    def _sync_from_shared_state(self) -> None:
        self.kwargs["api_key"] = self._shared_state.agent_key or ""

    def _ensure_credentials(self) -> None:
        """Acquire the per-account lock; refresh OAuth and/or mint as needed.

        Double-checked locking: when N threads enter the skew window
        simultaneously, only the first one performs the HTTP round-trip;
        the others observe the updated state after acquiring the lock and
        return without touching the network.
        """
        if not self._oauth_needs_refresh() and not self._agent_key_needs_mint():
            self._sync_from_shared_state()
            return

        with self._shared_state.lock:
            if self._oauth_needs_refresh():
                self._refresh_oauth()
            if self._agent_key_needs_mint():
                self._mint_agent_key(allow_oauth_retry=True)
            self._sync_from_shared_state()

    def _force_remint(self) -> None:
        """Skip skew check and re-mint immediately. Called when an inference
        call returned 401 — the cached agent_key is bad and we don't want
        to wait for the skew window.

        Pre-checks the OAuth expiry too. Without this, a stale OAuth +
        revoked agent_key combo takes three round-trips (mint→401→
        refresh→mint); with the pre-check it's two (refresh→mint). The
        mint's 401-triggers-refresh path still backstops the case where
        OAuth looks fresh by skew but the portal has revoked it.
        """
        with self._shared_state.lock:
            if self._oauth_needs_refresh():
                self._refresh_oauth()
            self._mint_agent_key(allow_oauth_retry=True)
            self._sync_from_shared_state()

    # ------------------------------------------------------------------
    # OAuth refresh
    # ------------------------------------------------------------------

    def _refresh_oauth(self) -> None:
        """POST refresh_token grant; on success, mutate shared state."""
        if not self._shared_state.refresh_token:
            raise HermesProviderError(
                "Nous Portal access token is expiring but no refresh_token "
                "is available. Run `hermes model` and select Nous Portal "
                "to re-authenticate."
            )

        try:
            with httpx.Client(timeout=httpx.Timeout(20.0)) as client:
                response = client.post(
                    f"{self._portal_base_url}/api/oauth/token",
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    data={
                        "grant_type": "refresh_token",
                        "client_id": NOUS_OAUTH_CLIENT_ID,
                        "refresh_token": self._shared_state.refresh_token,
                    },
                )
        except httpx.HTTPError as exc:
            raise HermesProviderError(
                f"Nous Portal OAuth refresh failed ({exc}). Check network "
                f"connectivity, then re-try; if the failure persists, run "
                f"`hermes model` to re-authenticate."
            ) from exc

        if response.status_code != 200:
            raise HermesProviderError(_format_oauth_error(response))

        try:
            payload = response.json()
        except ValueError as exc:
            raise HermesProviderError(
                "Nous Portal OAuth refresh returned invalid JSON. "
                "Run `hermes model` to re-authenticate."
            ) from exc

        new_access = payload.get("access_token")
        if not isinstance(new_access, str) or not new_access.strip():
            raise HermesProviderError(
                "Nous Portal OAuth refresh response was missing access_token. "
                "Run `hermes model` to re-authenticate."
            )

        # The Nous portal enforces single-use refresh-token rotation;
        # honor any rotated token in the response. Missing means the
        # portal kept the original valid.
        new_refresh = payload.get("refresh_token")
        if isinstance(new_refresh, str) and new_refresh.strip():
            self._shared_state.refresh_token = new_refresh.strip()

        expires_in = payload.get("expires_in")
        if (
            isinstance(expires_in, (int, float))
            and not isinstance(expires_in, bool)
            and expires_in > 0
        ):
            self._shared_state.oauth_expires_at = time.time() + float(expires_in)
        else:
            # Conservative 1h fallback if the field is missing — keeps the
            # next call from racing to the wire again immediately. Logged
            # so a portal protocol change that drops expires_in is at
            # least visible in the run log.
            _log.warning(
                "Nous OAuth refresh response had no usable expires_in; "
                "using 1h fallback. payload keys: %s",
                sorted(payload.keys()),
            )
            self._shared_state.oauth_expires_at = time.time() + 3600.0

        self._shared_state.access_token = new_access.strip()

    # ------------------------------------------------------------------
    # Agent_key minting
    # ------------------------------------------------------------------

    def _mint_agent_key(self, *, allow_oauth_retry: bool) -> None:
        """POST agent-key mint; on 401, optionally refresh OAuth and retry.

        Mirrors Hermes's mint-401-triggers-refresh-retry pattern in
        ``hermes_cli/auth.py``. ``allow_oauth_retry`` is True on the
        first call from ``_ensure_credentials``; the recursive retry
        passes False to bound the recursion at one OAuth refresh.
        """
        try:
            with httpx.Client(timeout=httpx.Timeout(20.0)) as client:
                response = client.post(
                    f"{self._portal_base_url}/api/oauth/agent-key",
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self._shared_state.access_token}",
                    },
                    json={"min_ttl_seconds": AGENT_KEY_MIN_TTL_SECONDS},
                )
        except httpx.HTTPError as exc:
            raise HermesProviderError(
                f"Nous Portal agent-key mint failed ({exc}). Check network "
                f"connectivity, then re-try; if the failure persists, run "
                f"`hermes model` to re-authenticate."
            ) from exc

        if response.status_code == 200:
            self._absorb_mint_response(response)
            return

        # 401 from mint → access_token may be stale even though OAuth said
        # it's still valid. Refresh once and retry. After that, give up.
        if response.status_code == 401 and allow_oauth_retry:
            self._refresh_oauth()
            self._mint_agent_key(allow_oauth_retry=False)
            return

        raise HermesProviderError(_format_mint_error(response))

    def _absorb_mint_response(self, response: httpx.Response) -> None:
        """Parse a 200 mint response into shared state.

        Tolerates both the current ``api_key`` field and the older
        ``agent_key`` shape, and prefers a server-supplied ``expires_at``
        ISO 8601 timestamp over the relative ``expires_in``. When neither
        expiry field is parseable, falls back to the requested floor TTL
        with a warning so portal protocol drift doesn't silently cache a
        key for longer than the server intended.
        """
        try:
            payload = response.json()
        except ValueError as exc:
            raise HermesProviderError(
                "Nous Portal agent-key mint returned invalid JSON. "
                "Run `hermes model` to re-authenticate."
            ) from exc

        agent_key = payload.get("api_key") or payload.get("agent_key")
        if not isinstance(agent_key, str) or not agent_key.strip():
            raise HermesProviderError(
                "Nous Portal agent-key mint response was missing api_key. "
                "Run `hermes model` to re-authenticate."
            )

        # ``expires_at`` is ISO 8601; ``expires_in`` is seconds-from-now.
        # Prefer expires_at when both present (server-authoritative).
        new_expires_at = parse_iso_or_epoch(payload.get("expires_at"))
        if new_expires_at is None:
            expires_in = payload.get("expires_in")
            if (
                isinstance(expires_in, (int, float))
                and not isinstance(expires_in, bool)
                and expires_in > 0
            ):
                new_expires_at = time.time() + float(expires_in)
            else:
                # Conservative — assume the floor TTL we asked for. Log
                # so a portal protocol change that drops both expiry
                # fields is at least visible in the run log; otherwise
                # we silently cache a key for 30 minutes regardless of
                # what the server intended.
                _log.warning(
                    "Nous mint response had no usable expires_at or "
                    "expires_in; using AGENT_KEY_MIN_TTL_SECONDS "
                    "fallback. payload keys: %s",
                    sorted(payload.keys()),
                )
                new_expires_at = time.time() + AGENT_KEY_MIN_TTL_SECONDS

        self._shared_state.agent_key = agent_key.strip()
        self._shared_state.agent_key_expires_at = new_expires_at

    # ------------------------------------------------------------------
    # forward / aforward — ensure creds, then delegate. Catch 401 once.
    # ------------------------------------------------------------------

    def forward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        self._ensure_credentials()
        try:
            return super().forward(prompt=prompt, messages=messages, **kwargs)
        except litellm.AuthenticationError:
            # Cached agent_key is dead despite passing the skew check.
            # Force re-mint (which may also refresh OAuth on its own 401)
            # and retry once. If the freshly-minted key is also rejected
            # the OAuth grant has likely been revoked entirely; surface
            # that explicitly so the operator gets the right recovery
            # hint instead of a generic 401.
            self._force_remint()
            try:
                return super().forward(prompt=prompt, messages=messages, **kwargs)
            except litellm.AuthenticationError as exc:
                raise HermesProviderError(
                    "Nous Portal inference rejected a freshly-minted "
                    "agent_key after an automatic re-mint. The OAuth "
                    "grant may have been revoked. Run `hermes model` "
                    "and select Nous Portal to re-authenticate."
                ) from exc

    async def aforward(self, prompt=None, messages=None, **kwargs):  # type: ignore[override]
        self._ensure_credentials()
        try:
            return await super().aforward(prompt=prompt, messages=messages, **kwargs)
        except litellm.AuthenticationError:
            self._force_remint()
            try:
                return await super().aforward(
                    prompt=prompt, messages=messages, **kwargs
                )
            except litellm.AuthenticationError as exc:
                raise HermesProviderError(
                    "Nous Portal inference rejected a freshly-minted "
                    "agent_key after an automatic re-mint. The OAuth "
                    "grant may have been revoked. Run `hermes model` "
                    "and select Nous Portal to re-authenticate."
                ) from exc


# ----------------------------------------------------------------------
# Error classification — mirror hermes-agent error taxonomy
# ----------------------------------------------------------------------

# OAuth error codes from the Nous portal's /api/oauth/token endpoint that
# indicate a permanently invalid refresh token. User must re-authenticate.
_OAUTH_RELOGIN_ERROR_CODES = frozenset({"invalid_grant", "invalid_token"})


def _format_oauth_error(response: httpx.Response) -> str:
    """Translate a non-200 OAuth refresh response into an actionable user
    message. Mirrors the OAuth-error classification in ``hermes_cli/auth.py``.
    """
    code, detail = _parse_error_body(response)

    # Match the explicit code field, not the free-form detail string —
    # a substring search on detail would false-positive on unrelated
    # portal messages like "this is not a reusable connection".
    if "reused" in code.lower():
        return (
            "Nous Portal refresh token was already consumed by another "
            "client (the portal enforces single-use refresh-token rotation). "
            "Run `hermes model` and select Nous Portal to re-authenticate."
        )

    if code in _OAUTH_RELOGIN_ERROR_CODES or response.status_code in (401, 403):
        return (
            f"Nous Portal OAuth refresh failed ({code}: {detail}). "
            f"Run `hermes model` and select Nous Portal to re-authenticate."
        )

    return (
        f"Nous Portal OAuth refresh failed ({code}: {detail}). "
        f"Re-try; if the failure persists, run `hermes model`."
    )


def _format_mint_error(response: httpx.Response) -> str:
    """Translate a non-200 agent-key mint response. 401 from mint is
    handled in ``_mint_agent_key`` (refresh-retry); this formatter sees
    only the unrecoverable cases.
    """
    code, detail = _parse_error_body(response)
    if response.status_code in (401, 403):
        return (
            f"Nous Portal agent-key mint failed ({code}: {detail}). "
            f"Run `hermes model` and select Nous Portal to re-authenticate."
        )
    return (
        f"Nous Portal agent-key mint failed (HTTP {response.status_code}, "
        f"{code}: {detail}). Re-try; if the failure persists, run "
        f"`hermes model`."
    )


def _parse_error_body(response: httpx.Response) -> tuple[str, str]:
    """Best-effort parse of OAuth-style error JSON. Returns (code, detail)
    with sensible defaults when the body is missing or malformed.

    On JSON parse failure (e.g., a CDN returning an HTML error page,
    or a portal outage returning text), ``detail`` falls back to a
    truncated snippet of the raw body so the operator can correlate
    the failure with what the upstream actually sent.
    """
    code = "unknown"
    detail = f"status {response.status_code}"
    try:
        body = response.json()
    except ValueError:
        snippet = (response.text or "").strip()
        if snippet:
            detail = f"status {response.status_code}: {snippet[:512]}"
        return code, detail

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
    return code, detail

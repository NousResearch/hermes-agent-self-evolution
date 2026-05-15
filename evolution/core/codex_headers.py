"""Cloudflare-mitigation headers + OAuth client constants for Codex.

The Cloudflare layer in front of ``chatgpt.com/backend-api/codex`` whitelists
a small set of first-party originators (``codex_cli_rs``, ``codex_vscode``,
``codex_sdk_ts``, anything starting with ``Codex``). Requests from
non-residential IPs (CI runners, VPS, cloud-hosted agents) that don't
advertise an allowed originator are served a 403 with
``cf-mitigated: challenge`` regardless of auth correctness.

We pin ``originator: codex_cli_rs`` to match the upstream codex-rs CLI, set
a ``codex_cli_rs``-shaped User-Agent (beats SDK fingerprinting), and
extract ``ChatGPT-Account-ID`` (canonical casing per codex-rs ``auth.rs``)
from the OAuth JWT's ``chatgpt_account_id`` claim. Mirrors hermes-agent's
``_codex_cloudflare_headers`` at ``agent/auxiliary_client.py``.
"""

from __future__ import annotations

import base64
import json
from typing import Dict


# Pinned client ID for the Codex OAuth refresh-token grant. Mirrors
# ``CODEX_OAUTH_CLIENT_ID`` in hermes-agent's ``hermes_cli/auth.py`` (search
# for the constant by name there). If Hermes rotates this we'll drift —
# preflight 401s will surface and the fix is to re-sync the constant.
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"

# Default Codex API base URL. Hermes lets users override via
# config.yaml/auth.json; the resolver passes the override through.
DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Refresh access token this many seconds before expiry, matching Hermes's
# ``CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS`` so concurrent Hermes sessions
# refresh on the same cadence and don't fight over the refresh token.
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120

_USER_AGENT = "codex_cli_rs/0.0.0 (agent-self-evolution)"
_ORIGINATOR = "codex_cli_rs"


def codex_cloudflare_headers(access_token: str) -> Dict[str, str]:
    """Build the headers required to avoid Cloudflare 403s on Codex.

    Always returns ``User-Agent`` and ``originator``. Adds
    ``ChatGPT-Account-ID`` when the JWT can be parsed and contains the
    ``chatgpt_account_id`` claim. Malformed tokens are tolerated — we drop
    the account-ID header rather than raise, so a bad token surfaces as a
    401 from the underlying call (which the auth-check classifier catches)
    rather than a crash at header build.
    """
    headers = {
        "User-Agent": _USER_AGENT,
        "originator": _ORIGINATOR,
    }
    acct_id = _extract_account_id(access_token)
    if acct_id:
        headers["ChatGPT-Account-ID"] = acct_id
    return headers


def _extract_account_id(access_token: str) -> str:
    """Pull ``chatgpt_account_id`` out of the JWT payload if present."""
    if not isinstance(access_token, str) or not access_token.strip():
        return ""
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return ""
        # JWT payloads are urlsafe-base64 without padding; pad before decode.
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        acct_id = claims.get("https://api.openai.com/auth", {}).get(
            "chatgpt_account_id"
        )
        if isinstance(acct_id, str) and acct_id:
            return acct_id
    except Exception:
        pass
    return ""

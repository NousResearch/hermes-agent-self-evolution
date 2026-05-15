"""Tests for Codex Cloudflare headers + JWT account-ID extraction.

The headers are non-negotiable for any non-residential-IP deployment of
the framework — without them, every call to ``chatgpt.com/backend-api/codex``
returns 403 from Cloudflare regardless of OAuth correctness.
"""

from __future__ import annotations

import base64
import json

from evolution.core.codex_headers import (
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    DEFAULT_CODEX_BASE_URL,
    codex_cloudflare_headers,
)


def _make_jwt(claims: dict, *, with_padding: bool = True) -> str:
    """Build a minimal 3-part JWT with the given claims as the payload."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload_bytes = json.dumps(claims).encode()
    payload = base64.urlsafe_b64encode(payload_bytes).decode()
    if not with_padding:
        payload = payload.rstrip("=")
    signature = "sig"
    return f"{header}.{payload}.{signature}"


class TestAlwaysReturnsBaseHeaders:
    def test_originator_and_user_agent_for_valid_token(self):
        token = _make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}}
        )
        h = codex_cloudflare_headers(token)
        assert h["originator"] == "codex_cli_rs"
        assert h["User-Agent"].startswith("codex_cli_rs/")

    def test_originator_present_even_without_jwt(self):
        # Garbage token: still get base headers (so the request goes out and
        # surfaces a 401, rather than silently failing at header-build time).
        h = codex_cloudflare_headers("not-a-jwt")
        assert h["originator"] == "codex_cli_rs"
        assert h["User-Agent"].startswith("codex_cli_rs/")
        assert "ChatGPT-Account-ID" not in h


class TestAccountIDExtraction:
    def test_extracts_from_well_formed_jwt(self):
        token = _make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-abc"}}
        )
        h = codex_cloudflare_headers(token)
        assert h["ChatGPT-Account-ID"] == "acct-abc"

    def test_handles_jwt_without_padding(self):
        # Real JWTs are urlsafe-base64 *without* padding. The header builder
        # must pad before decoding or it'll silently drop the claim.
        token = _make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-no-pad"}},
            with_padding=False,
        )
        h = codex_cloudflare_headers(token)
        assert h["ChatGPT-Account-ID"] == "acct-no-pad"

    def test_drops_header_when_claim_missing(self):
        # JWT parses but the claim isn't there.
        token = _make_jwt({"some_other_claim": "value"})
        h = codex_cloudflare_headers(token)
        assert "ChatGPT-Account-ID" not in h

    def test_drops_header_on_malformed_jwt(self):
        # Single dot, not 3 parts.
        h = codex_cloudflare_headers("part1.part2-but-no-third")
        # Falls through to base64 decode of "part2-but-no-third" which will
        # likely either fail to decode or decode to non-JSON. Either way:
        # we should not raise, and we should not emit the header.
        assert "ChatGPT-Account-ID" not in h

    def test_drops_header_on_empty_token(self):
        for empty in ("", "   ", None):
            h = codex_cloudflare_headers(empty)  # type: ignore[arg-type]
            assert "ChatGPT-Account-ID" not in h
            assert h["originator"] == "codex_cli_rs"


class TestConstants:
    def test_oauth_client_id_pinned(self):
        # Mirrors hermes-agent's CODEX_OAUTH_CLIENT_ID. If Hermes rotates
        # the client ID, Codex auth refresh starts failing — this test
        # documents the contract so the failure mode is obvious.
        assert CODEX_OAUTH_CLIENT_ID == "app_EMoamEEZ73f0CkXaXp7hrann"

    def test_oauth_token_url(self):
        assert CODEX_OAUTH_TOKEN_URL == "https://auth.openai.com/oauth/token"

    def test_default_base_url(self):
        assert DEFAULT_CODEX_BASE_URL == "https://chatgpt.com/backend-api/codex"

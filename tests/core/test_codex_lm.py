"""Tests for the Codex LM subclass.

Covers:
  - Construction wiring (model_type, api_base, api_key, extra_headers)
  - OAuth refresh: timing window, success path, server-error classification
  - Cross-instance refresh-state sharing (the four LM roles share an account)
  - Async path (aforward) invokes refresh
  - Missing refresh_token + expiring → HermesProviderError
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import httpx
import pytest

from evolution.core.codex_headers import CODEX_OAUTH_TOKEN_URL
from evolution.core.codex_lm import (
    CodexLM,
    _format_refresh_error,
    _reset_state_for_tests,
)
from evolution.core.hermes_provider import HermesProviderError


# Module-level OAuth state bleeds across tests; clear between cases.
@pytest.fixture(autouse=True)
def _clean_codex_state():
    _reset_state_for_tests()
    yield
    _reset_state_for_tests()


def _jwt_with_account(acct_id: str) -> str:
    """Build a synthetic JWT that codex_cloudflare_headers can parse."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {"https://api.openai.com/auth": {"chatgpt_account_id": acct_id}}
        ).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.sig"


def _mock_refresh_response(
    *, status_code: int = 200, json_body: dict | None = None
) -> MagicMock:
    """Build an httpx-Response-shaped mock for monkey-patching httpx.Client."""
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    if json_body is not None:
        mock.json = MagicMock(return_value=json_body)
    else:
        mock.json = MagicMock(side_effect=ValueError("no body"))
    return mock


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestCodexLMConstruction:
    def test_wires_responses_model_type_and_kwargs(self):
        lm = CodexLM(
            model="gpt-5-codex",
            access_token=_jwt_with_account("acct-1"),
            refresh_token="rt-1",
            expires_at=time.time() + 3600,
            base_url="https://chatgpt.com/backend-api/codex",
        )
        assert lm.model_type == "responses"
        assert lm.kwargs["api_base"] == "https://chatgpt.com/backend-api/codex"
        assert lm.kwargs["api_key"].startswith("eyJ")  # JWT
        assert lm.kwargs["extra_headers"]["originator"] == "codex_cli_rs"
        assert lm.kwargs["extra_headers"]["ChatGPT-Account-ID"] == "acct-1"
        # Reasoning-model defaults pinned (DSPy enforces these for gpt-5).
        # DSPy renames max_tokens to max_completion_tokens internally.
        assert lm.kwargs["temperature"] == 1.0
        assert lm.kwargs["max_completion_tokens"] == 16000

    def test_default_base_url_when_unset(self):
        lm = CodexLM(
            model="gpt-5-codex",
            access_token=_jwt_with_account("acct-1"),
            refresh_token="rt-1",
            expires_at=time.time() + 3600,
        )
        assert "chatgpt.com/backend-api/codex" in lm.kwargs["api_base"]


# ---------------------------------------------------------------------------
# Refresh timing
# ---------------------------------------------------------------------------


class TestCodexRefreshTiming:
    def test_no_refresh_when_token_fresh(self):
        with patch.object(CodexLM, "_do_refresh") as mock_refresh:
            lm = CodexLM(
                model="gpt-5-codex",
                access_token=_jwt_with_account("acct-1"),
                refresh_token="rt-1",
                expires_at=time.time() + 3600,
            )
            lm._refresh_if_expiring()
            mock_refresh.assert_not_called()

    def test_refresh_when_within_skew_window(self):
        with patch.object(CodexLM, "_do_refresh") as mock_refresh:
            lm = CodexLM(
                model="gpt-5-codex",
                access_token=_jwt_with_account("acct-1"),
                refresh_token="rt-1",
                expires_at=time.time() + 60,  # 60s out, inside 120s skew
            )
            lm._refresh_if_expiring()
            mock_refresh.assert_called_once()

    def test_refresh_when_already_expired(self):
        with patch.object(CodexLM, "_do_refresh") as mock_refresh:
            lm = CodexLM(
                model="gpt-5-codex",
                access_token=_jwt_with_account("acct-1"),
                refresh_token="rt-1",
                expires_at=time.time() - 60,  # already expired
            )
            lm._refresh_if_expiring()
            mock_refresh.assert_called_once()

    def test_no_refresh_when_expires_at_unset(self):
        # Some auth.json entries omit expires_at (older Hermes versions).
        # We don't know when it expires, so don't proactively refresh —
        # let the actual call surface a 401 if the token is dead.
        with patch.object(CodexLM, "_do_refresh") as mock_refresh:
            lm = CodexLM(
                model="gpt-5-codex",
                access_token=_jwt_with_account("acct-1"),
                refresh_token="rt-1",
                expires_at=None,
            )
            lm._refresh_if_expiring()
            mock_refresh.assert_not_called()

    def test_missing_refresh_token_raises(self):
        lm = CodexLM(
            model="gpt-5-codex",
            access_token=_jwt_with_account("acct-1"),
            refresh_token=None,  # no refresh_token
            expires_at=time.time() - 60,  # but expired
        )
        with pytest.raises(HermesProviderError, match="hermes auth add openai-codex"):
            lm._refresh_if_expiring()


# ---------------------------------------------------------------------------
# Successful refresh updates state
# ---------------------------------------------------------------------------


class TestCodexRefreshSuccess:
    def test_refresh_updates_api_key_and_extra_headers(self):
        new_token = _jwt_with_account("acct-2")
        with patch("evolution.core.codex_lm.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = _mock_refresh_response(
                json_body={
                    "access_token": new_token,
                    "expires_in": 3600,
                }
            )

            lm = CodexLM(
                model="gpt-5-codex",
                access_token=_jwt_with_account("acct-1"),
                refresh_token="rt-1",
                expires_at=time.time() - 60,
            )
            lm._refresh_if_expiring()

            assert lm.kwargs["api_key"] == new_token
            # Extra headers re-built with the new token's account-id.
            assert lm.kwargs["extra_headers"]["ChatGPT-Account-ID"] == "acct-2"
            # POST body had the right grant.
            kwargs = mock_client.post.call_args.kwargs
            assert kwargs["data"]["grant_type"] == "refresh_token"
            assert kwargs["data"]["refresh_token"] == "rt-1"

    def test_refresh_honors_rotated_refresh_token(self):
        with patch("evolution.core.codex_lm.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = _mock_refresh_response(
                json_body={
                    "access_token": _jwt_with_account("acct-1"),
                    "refresh_token": "rt-rotated",
                    "expires_in": 3600,
                }
            )

            lm = CodexLM(
                model="gpt-5-codex",
                access_token=_jwt_with_account("acct-1"),
                refresh_token="rt-1",
                expires_at=time.time() - 60,
            )
            lm._refresh_if_expiring()

            # State carries the rotated refresh_token for next refresh.
            assert lm._shared_state.refresh_token == "rt-rotated"


# ---------------------------------------------------------------------------
# Cross-instance state sharing — the bug-prevention test
# ---------------------------------------------------------------------------


class TestCodexCrossInstanceStateSharing:
    def test_concurrent_refresh_only_posts_once(self):
        """Four CodexLM instances sharing the same refresh_token must
        coordinate so only ONE actually POSTs to auth.openai.com.
        Without the shared state, three would get refresh_token_reused.
        """
        new_token = _jwt_with_account("acct-2")
        post_count = {"n": 0}
        # The first thread to grab the per-account lock is the one that
        # actually POSTs. Slow it slightly so the other three queue up on
        # the lock, then observe the refreshed state and exit without
        # POSTing themselves. Without the cross-instance shared state, all
        # four would POST and three would get refresh_token_reused.
        post_lock = threading.Lock()

        def slow_post(*args, **kwargs):
            with post_lock:
                post_count["n"] += 1
                time.sleep(0.05)
            return _mock_refresh_response(
                json_body={"access_token": new_token, "expires_in": 3600}
            )

        with patch("evolution.core.codex_lm.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client_cls.return_value = mock_client
            mock_client.post.side_effect = slow_post

            shared_args = dict(
                model="gpt-5-codex",
                access_token=_jwt_with_account("acct-1"),
                refresh_token="rt-shared",
                expires_at=time.time() - 60,
            )
            instances = [CodexLM(**shared_args) for _ in range(4)]

            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = [pool.submit(lm._refresh_if_expiring) for lm in instances]
                for f in futs:
                    f.result(timeout=10)

            assert post_count["n"] == 1, (
                f"Expected exactly one refresh POST across 4 concurrent workers, "
                f"got {post_count['n']}. Multiple POSTs would trigger "
                "refresh_token_reused on second-and-after attempts."
            )

            # All four instances see the new token after refresh.
            for lm in instances:
                assert lm.kwargs["api_key"] == new_token


# ---------------------------------------------------------------------------
# Refresh failure classification
# ---------------------------------------------------------------------------


class TestCodexRefreshErrorClassification:
    def test_invalid_grant_surfaces_relogin_message(self):
        with patch("evolution.core.codex_lm.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = _mock_refresh_response(
                status_code=400,
                json_body={"error": "invalid_grant", "error_description": "bad"},
            )

            lm = CodexLM(
                model="gpt-5-codex",
                access_token=_jwt_with_account("acct-1"),
                refresh_token="rt-1",
                expires_at=time.time() - 60,
            )
            with pytest.raises(HermesProviderError, match="hermes auth add openai-codex"):
                lm._refresh_if_expiring()

    def test_refresh_token_reused_surfaces_special_message(self):
        with patch("evolution.core.codex_lm.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = _mock_refresh_response(
                status_code=400,
                json_body={"error": {"code": "refresh_token_reused", "message": "x"}},
            )

            lm = CodexLM(
                model="gpt-5-codex",
                access_token=_jwt_with_account("acct-1"),
                refresh_token="rt-1",
                expires_at=time.time() - 60,
            )
            with pytest.raises(
                HermesProviderError, match="another client"
            ) as excinfo:
                lm._refresh_if_expiring()
            # Both recovery commands should be mentioned.
            msg = str(excinfo.value)
            assert "codex" in msg
            assert "hermes auth add openai-codex" in msg

    def test_401_from_oauth_endpoint_treated_as_relogin(self):
        with patch("evolution.core.codex_lm.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client_cls.return_value = mock_client
            # Body has a non-relogin code, but the 401 status alone should
            # force the relogin path — the refresh token is dead.
            mock_client.post.return_value = _mock_refresh_response(
                status_code=401,
                json_body={"error": {"code": "internal_error"}},
            )

            lm = CodexLM(
                model="gpt-5-codex",
                access_token=_jwt_with_account("acct-1"),
                refresh_token="rt-1",
                expires_at=time.time() - 60,
            )
            with pytest.raises(HermesProviderError, match="hermes auth add openai-codex"):
                lm._refresh_if_expiring()

    def test_malformed_json_response_raises_with_recovery_hint(self):
        with patch("evolution.core.codex_lm.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client_cls.return_value = mock_client
            mock_client.post.return_value = _mock_refresh_response(
                status_code=200,
                json_body=None,  # .json() raises
            )

            lm = CodexLM(
                model="gpt-5-codex",
                access_token=_jwt_with_account("acct-1"),
                refresh_token="rt-1",
                expires_at=time.time() - 60,
            )
            with pytest.raises(HermesProviderError, match="hermes auth add openai-codex"):
                lm._refresh_if_expiring()


# ---------------------------------------------------------------------------
# Async path
# ---------------------------------------------------------------------------


class TestCodexAsyncPath:
    def test_aforward_invokes_refresh_before_super(self):
        with patch.object(CodexLM, "_refresh_if_expiring") as mock_refresh, \
             patch("dspy.LM.aforward", autospec=True) as mock_super_aforward:
            mock_super_aforward.return_value = asyncio.sleep(0, result="ok")
            lm = CodexLM(
                model="gpt-5-codex",
                access_token=_jwt_with_account("acct-1"),
                refresh_token="rt-1",
                expires_at=time.time() + 3600,
            )

            asyncio.run(lm.aforward(messages=[{"role": "user", "content": "hi"}]))

            mock_refresh.assert_called_once()
            mock_super_aforward.assert_called_once()


# ---------------------------------------------------------------------------
# Format helper
# ---------------------------------------------------------------------------


class TestFormatRefreshError:
    def test_handles_response_without_json_body(self):
        resp = _mock_refresh_response(status_code=500, json_body=None)
        msg = _format_refresh_error(resp)
        assert "status 500" in msg

    def test_extracts_openai_error_shape(self):
        resp = _mock_refresh_response(
            status_code=400,
            json_body={"error": {"code": "rate_limited", "message": "slow down"}},
        )
        msg = _format_refresh_error(resp)
        assert "rate_limited" in msg
        assert "slow down" in msg

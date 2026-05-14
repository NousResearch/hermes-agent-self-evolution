"""Mid-run auth-abort sentinel tests.

Defense-in-depth for the case where preflight passed but a credential
goes bad mid-run (long sessions on short-TTL OAuth, key revocation
during a multi-hour evolution). Mirrors the CostCeilingExceeded
mechanism: the litellm failure_callback sets a process-wide flag, the
patched BaseLM.__call__ checks it and raises a HermesProviderError that
bypasses dspy.Evaluate's `except Exception` swallowing.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import litellm
import pytest

from evolution.core.hermes_provider import HermesProviderError
from evolution.core.lm_timing_callback import (
    COST_LEDGER,
    _log_litellm_failure,
)


@pytest.fixture(autouse=True)
def _reset_ledger():
    """Each test starts with a clean ledger so prior tests' aborts don't
    leak. ``COST_LEDGER.reset()`` is the same call ``evolve()`` makes at
    the start of each run.
    """
    COST_LEDGER.reset()
    yield
    COST_LEDGER.reset()


class TestAuthAbortSentinel:
    def test_auth_failure_sets_abort_flag(self):
        # No abort initially.
        assert COST_LEDGER.get_auth_abort_message() is None

        # Failure callback sees an AuthenticationError → sets the flag.
        exc = litellm.AuthenticationError(
            message="401 Unauthorized: token expired",
            llm_provider="anthropic",
            model="anthropic/claude-opus-4-5",
        )
        _log_litellm_failure(
            kwargs={"model": "anthropic/claude-opus-4-5"},
            exception=exc,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        msg = COST_LEDGER.get_auth_abort_message()
        assert msg is not None
        assert "anthropic/claude-opus-4-5" in msg

    def test_unrelated_failure_does_not_set_abort(self):
        # A generic exception (network error, timeout, etc.) must not
        # poison the flag — those failures are normal mid-run noise.
        _log_litellm_failure(
            kwargs={"model": "anthropic/claude-opus-4-5"},
            exception=ConnectionError("connection refused"),
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert COST_LEDGER.get_auth_abort_message() is None

    def test_rate_limit_does_not_set_auth_abort(self):
        # 429s are recoverable (retry later) — they shouldn't trigger the
        # auth-abort path which is for permanent credential failures.
        exc = litellm.RateLimitError(
            message="429 Too Many Requests",
            llm_provider="openai",
            model="openai/gpt-4o-mini",
        )
        _log_litellm_failure(
            kwargs={"model": "openai/gpt-4o-mini"},
            exception=exc,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert COST_LEDGER.get_auth_abort_message() is None

    def test_reset_clears_auth_abort(self):
        exc = litellm.AuthenticationError(
            message="401",
            llm_provider="openai",
            model="openai/gpt-4o-mini",
        )
        _log_litellm_failure(
            kwargs={"model": "openai/gpt-4o-mini"},
            exception=exc,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert COST_LEDGER.get_auth_abort_message() is not None

        # reset() is what evolve() calls at the start of each run; after
        # reset, the next run's first LM call must NOT abort because of
        # a flag set by a previous run.
        COST_LEDGER.reset()
        assert COST_LEDGER.get_auth_abort_message() is None

    def test_pattern_matched_auth_failure_sets_abort(self):
        # An adapter that wraps auth errors in its own exception type
        # (not litellm.AuthenticationError) is detected via the message
        # pattern matcher mirrored from hermes-agent's error_classifier.
        _log_litellm_failure(
            kwargs={"model": "openai/gpt-4o-mini"},
            exception=Exception("HTTP 401: invalid api key"),
            start_time=datetime.now(),
            end_time=datetime.now(),
        )
        assert COST_LEDGER.get_auth_abort_message() is not None


class TestLMGuardRaisesOnAuthAbort:
    """The patched BaseLM.__call__ checks both ceiling and auth-abort
    state. When auth-abort is set, the next call raises
    HermesProviderError — which is BaseException-derived so dspy.Evaluate's
    `except Exception` cannot swallow it.
    """

    def test_guard_raises_when_auth_abort_set(self, monkeypatch):
        # Set the flag directly (bypass the failure callback for clarity).
        # Using the public setter so this test isn't fragile to internal
        # field renames.
        from evolution.core.lm_timing_callback import _install_cost_ceiling_lm_guard
        from dspy.clients.base_lm import BaseLM

        # Make sure the guard is installed.
        _install_cost_ceiling_lm_guard()

        COST_LEDGER._set_auth_abort("test sentinel: model-X token expired")

        # Construct a BaseLM-shaped object. We don't call into a real
        # dspy.LM (no network); we bypass via the guard's check by
        # calling BaseLM.__call__ on a MagicMock spec.
        mock_lm = MagicMock(spec=BaseLM)

        with pytest.raises(HermesProviderError) as exc:
            BaseLM.__call__(mock_lm, "prompt")
        assert "test sentinel" in str(exc.value)

    def test_guard_does_not_raise_when_no_abort(self, monkeypatch):
        from evolution.core.lm_timing_callback import _install_cost_ceiling_lm_guard
        from dspy.clients.base_lm import BaseLM

        _install_cost_ceiling_lm_guard()
        # No auth-abort set; the guard should pass through to the underlying
        # call. The MagicMock returns a MagicMock instead of raising.
        mock_lm = MagicMock(spec=BaseLM)
        # If the guard raises here it's the test failure; we can't
        # actually invoke __call__ on a Mock cleanly because the guard
        # delegates to original_call which expects a real BaseLM. So we
        # just assert no HermesProviderError before that point.
        try:
            BaseLM.__call__(mock_lm, "prompt")
        except HermesProviderError:
            pytest.fail("Guard raised HermesProviderError when no auth abort was set")
        except Exception:
            # Other exceptions from the Mock interaction are fine — we
            # only care about the auth-abort path.
            pass


class TestCostLedgerAuthAbortAPI:
    """Pin the public API shape so ``_log_litellm_failure`` and the LM
    guard agree on field names.
    """

    def test_get_auth_abort_message_returns_none_on_clean_ledger(self):
        assert COST_LEDGER.get_auth_abort_message() is None

    def test_set_auth_abort_is_idempotent(self):
        COST_LEDGER._set_auth_abort("first")
        first = COST_LEDGER.get_auth_abort_message()
        COST_LEDGER._set_auth_abort("second")
        # First message wins — the original failure is the actionable one;
        # subsequent failures are usually downstream effects of the same
        # bad credential.
        assert COST_LEDGER.get_auth_abort_message() == first

    def test_summary_unaffected_by_auth_abort(self):
        # Auth abort is independent of cost tracking; summary() shouldn't
        # change shape based on whether auth_abort is set.
        COST_LEDGER._set_auth_abort("test")
        summary = COST_LEDGER.summary()
        assert "total_usd" in summary
        assert "by_model" in summary

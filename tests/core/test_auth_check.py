"""Tests for the LM credential preflight + auth-error classification.

Covers:
  - is_auth_error / is_rate_limit_error classification (typed exceptions
    + message-pattern fallback for non-typed)
  - preflight() dedup across roles sharing a (model, kwargs) tuple
  - preflight() translates litellm.AuthenticationError into the new
    HermesProviderError with provider-specific recovery guidance
  - preflight() distinguishes RateLimitError (transient) from auth
  - format_auth_error_message uses the provider-command lookup
  - Provider hint extraction from LiteLLM model strings
  - Cost-callback integration (preflight calls show up in the ledger)

All tests are mock-only — no network — for CI parity.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import litellm
import pytest

from evolution.core.auth_check import (
    HermesProviderRateLimitError,
    _HERMES_AUTH_COMMAND_BY_PROVIDER,
    _dedupe_key,
    _provider_hint_from_model,
    format_auth_error_message,
    is_auth_error,
    is_rate_limit_error,
    preflight,
)
from evolution.core.hermes_provider import HermesProviderError, ResolvedLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_err(model: str = "openai/gpt-4o-mini", message: str = "401 Unauthorized: invalid api key") -> litellm.AuthenticationError:
    return litellm.AuthenticationError(
        message=message,
        llm_provider=model.split("/", 1)[0],
        model=model,
    )


def _rate_err(model: str = "openai/gpt-4o-mini") -> litellm.RateLimitError:
    return litellm.RateLimitError(
        message="429 Too Many Requests",
        llm_provider=model.split("/", 1)[0],
        model=model,
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestIsAuthError:
    def test_litellm_authentication_error(self):
        assert is_auth_error(_auth_err())

    def test_generic_exception_with_401_message(self):
        assert is_auth_error(Exception("HTTP 401 Unauthorized"))

    def test_generic_exception_with_invalid_api_key_message(self):
        assert is_auth_error(Exception("Invalid API key provided"))

    def test_generic_exception_with_token_expired_message(self):
        assert is_auth_error(Exception("token expired or revoked"))

    def test_generic_exception_with_403_forbidden_message(self):
        assert is_auth_error(Exception("403 Forbidden"))

    def test_unrelated_exception(self):
        assert not is_auth_error(Exception("connection refused"))

    def test_rate_limit_is_not_auth(self):
        # Critical: 429s are rate-limit, not auth — different recovery path.
        assert not is_auth_error(_rate_err())


class TestIsRateLimitError:
    def test_litellm_rate_limit(self):
        assert is_rate_limit_error(_rate_err())

    def test_generic_exception_with_429(self):
        assert is_rate_limit_error(Exception("HTTP 429 Too Many Requests"))

    def test_auth_error_is_not_rate_limit(self):
        assert not is_rate_limit_error(_auth_err())


# ---------------------------------------------------------------------------
# Provider hint extraction
# ---------------------------------------------------------------------------


class TestProviderHint:
    def test_anthropic(self):
        assert _provider_hint_from_model("anthropic/claude-opus-4-5") == "anthropic"

    def test_openrouter(self):
        # openrouter/<provider>/<model> — the prefix is what matters for the
        # hermes-command lookup; we surface "openrouter".
        assert _provider_hint_from_model("openrouter/anthropic/claude-opus-4-5") == "openrouter"

    def test_openai(self):
        assert _provider_hint_from_model("openai/gpt-5.4-mini") == "openai"

    def test_unprefixed_returns_none(self):
        # Some explicit overrides may not have a slash (rare).
        assert _provider_hint_from_model("just-a-model-name") is None


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


class TestDedupe:
    def test_same_model_same_kwargs_same_key(self):
        a = ResolvedLM(model="openai/gpt-4o-mini", lm_kwargs={"api_key": "k", "api_base": "u"}, source="x")
        b = ResolvedLM(model="openai/gpt-4o-mini", lm_kwargs={"api_base": "u", "api_key": "k"}, source="y")
        # Source string differs but the underlying LM is the same — key is
        # invariant of source, and kwarg ordering doesn't matter.
        assert _dedupe_key(a) == _dedupe_key(b)

    def test_different_api_key_different_key(self):
        a = ResolvedLM(model="openai/gpt-4o-mini", lm_kwargs={"api_key": "k1"}, source="x")
        b = ResolvedLM(model="openai/gpt-4o-mini", lm_kwargs={"api_key": "k2"}, source="x")
        assert _dedupe_key(a) != _dedupe_key(b)


# ---------------------------------------------------------------------------
# preflight() — the main entry point
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_empty_list_no_op(self):
        # Should not call completion_fn at all.
        completion_fn = MagicMock()
        preflight([], completion_fn=completion_fn)
        completion_fn.assert_not_called()

    def test_single_lm_success(self):
        completion_fn = MagicMock(return_value=MagicMock())
        lms = [ResolvedLM(model="openai/gpt-4o-mini", lm_kwargs={"api_key": "k"}, source="x")]
        preflight(lms, completion_fn=completion_fn)
        completion_fn.assert_called_once()
        # Probe payload sanity
        kwargs = completion_fn.call_args.kwargs
        assert kwargs["model"] == "openai/gpt-4o-mini"
        assert kwargs["api_key"] == "k"
        # max_tokens=16 not 1: OpenAI's reasoning-class models reject
        # sub-output-budget probes with a 400. 16 is plenty for an
        # empty-ish response and still costs ~$0.0001.
        assert kwargs["max_tokens"] == 16
        assert kwargs["num_retries"] == 0

    def test_dedup_collapses_duplicate_lms(self):
        # Hermes single-model setup: 4 roles all resolve to same LM. Should
        # produce exactly 1 preflight probe, not 4.
        completion_fn = MagicMock(return_value=MagicMock())
        lm = ResolvedLM(model="openai/gpt-4o-mini", lm_kwargs={"api_key": "k"}, source="x")
        preflight([lm, lm, lm, lm], completion_fn=completion_fn)
        assert completion_fn.call_count == 1

    def test_different_kwargs_not_deduped(self):
        completion_fn = MagicMock(return_value=MagicMock())
        a = ResolvedLM(model="openai/gpt-4o-mini", lm_kwargs={"api_key": "k1"}, source="x")
        b = ResolvedLM(model="openai/gpt-4o-mini", lm_kwargs={"api_key": "k2"}, source="x")
        preflight([a, b], completion_fn=completion_fn)
        assert completion_fn.call_count == 2

    def test_authentication_error_translated(self):
        completion_fn = MagicMock(side_effect=_auth_err(model="anthropic/claude-opus-4-5"))
        lms = [ResolvedLM(model="anthropic/claude-opus-4-5", lm_kwargs={"api_key": "bad"}, source="env")]
        with pytest.raises(HermesProviderError) as exc:
            preflight(lms, completion_fn=completion_fn)
        msg = str(exc.value)
        assert "anthropic/claude-opus-4-5" in msg
        # Provider-specific recovery: hermes auth add anthropic
        assert "hermes auth add anthropic" in msg
        # The raw underlying error message is preserved for diagnostic
        # purposes; users sometimes need it to file an issue.
        assert "401" in msg or "invalid api key" in msg.lower()

    def test_rate_limit_raises_distinct_type(self):
        completion_fn = MagicMock(side_effect=_rate_err())
        lms = [ResolvedLM(model="openai/gpt-4o-mini", lm_kwargs={"api_key": "k"}, source="x")]
        with pytest.raises(HermesProviderRateLimitError):
            preflight(lms, completion_fn=completion_fn)

    def test_message_pattern_fallback_for_untyped_auth_error(self):
        # Some adapters wrap auth errors in their own types that don't
        # inherit from litellm.AuthenticationError. The pattern matcher
        # mirrors the substring set hermes-agent's classifier uses.
        completion_fn = MagicMock(side_effect=Exception("HTTP 401: invalid api key for openai"))
        lms = [ResolvedLM(model="openai/gpt-4o-mini", lm_kwargs={"api_key": "k"}, source="x")]
        with pytest.raises(HermesProviderError):
            preflight(lms, completion_fn=completion_fn)

    def test_unrelated_error_propagates_as_is(self):
        # Connection refused, DNS failures, etc. are not auth — they
        # should bubble up as the original exception type so the user
        # sees the actual problem (or the call site's retry handles it).
        completion_fn = MagicMock(side_effect=ConnectionError("connection refused"))
        lms = [ResolvedLM(model="openai/gpt-4o-mini", lm_kwargs={"api_key": "k"}, source="x")]
        with pytest.raises(ConnectionError):
            preflight(lms, completion_fn=completion_fn)

    def test_first_failure_short_circuits(self):
        # If the first LM probe fails, we should NOT continue probing the
        # rest — the user has a problem to fix; piling on more failed
        # API calls just wastes time and money.
        completion_fn = MagicMock(side_effect=[_auth_err(), MagicMock()])
        lms = [
            ResolvedLM(model="openai/gpt-4o-mini", lm_kwargs={"api_key": "bad"}, source="x"),
            ResolvedLM(model="anthropic/claude-opus-4-5", lm_kwargs={"api_key": "good"}, source="y"),
        ]
        with pytest.raises(HermesProviderError):
            preflight(lms, completion_fn=completion_fn)
        # Second LM never probed.
        assert completion_fn.call_count == 1


# ---------------------------------------------------------------------------
# format_auth_error_message
# ---------------------------------------------------------------------------


class TestFormatAuthErrorMessage:
    def test_known_provider_includes_hermes_command(self):
        msg = format_auth_error_message(
            model="anthropic/claude-opus-4-5",
            provider_hint="anthropic",
            underlying=_auth_err(model="anthropic/claude-opus-4-5"),
        )
        assert "hermes auth add anthropic" in msg

    def test_unknown_provider_falls_back_to_generic(self):
        msg = format_auth_error_message(
            model="exotic-provider/some-model",
            provider_hint=None,
            underlying=_auth_err(),
        )
        # No specific hermes command; suggest the generic catch-all.
        assert "--optimizer-model" in msg or "set the appropriate" in msg.lower()

    def test_provider_command_table_covers_each_known_env_provider(self):
        # Every provider in _PROVIDER_ENV_KEYS should have an entry in the
        # hermes-command table. Otherwise a user with that provider
        # configured will get a generic message instead of the right command.
        from evolution.core.hermes_provider import _PROVIDER_ENV_KEYS
        # nous, gemini, copilot, anthropic, openrouter, openai, etc. should
        # all be present. Local servers (ollama, vllm, llamacpp, lmstudio)
        # are auth-optional and don't need entries.
        expected = set(_PROVIDER_ENV_KEYS) - {"ollama", "vllm", "llamacpp", "lmstudio", "custom"}
        missing = expected - set(_HERMES_AUTH_COMMAND_BY_PROVIDER)
        assert not missing, (
            f"_HERMES_AUTH_COMMAND_BY_PROVIDER missing entries for: {missing}. "
            "Users with these providers configured will get a generic message."
        )


# ---------------------------------------------------------------------------
# HermesProviderError now inherits from BaseException
# ---------------------------------------------------------------------------


class TestExceptionReparenting:
    def test_hermes_provider_error_is_base_exception(self):
        # The whole point: dspy.Evaluate's `except Exception` should NOT
        # catch us. Verify HermesProviderError is not an Exception subclass.
        assert not issubclass(HermesProviderError, Exception)
        assert issubclass(HermesProviderError, BaseException)

    def test_pytest_raises_still_works(self):
        # Sanity: existing tests catching HermesProviderError keep working.
        with pytest.raises(HermesProviderError):
            raise HermesProviderError("test")

    def test_rate_limit_error_inherits_from_provider_error(self):
        assert issubclass(HermesProviderRateLimitError, HermesProviderError)
        with pytest.raises(HermesProviderError):
            raise HermesProviderRateLimitError("429")

    def test_except_exception_does_not_catch(self):
        # Critical: the GEPA worker pool's `except Exception` must not swallow
        # auth aborts. This is the bug we're fixing.
        try:
            try:
                raise HermesProviderError("auth failed")
            except Exception:
                pytest.fail("HermesProviderError should not be caught by `except Exception`")
        except HermesProviderError:
            pass  # expected

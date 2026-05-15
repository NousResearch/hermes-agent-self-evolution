"""Tests for Nous Portal resolution in the Hermes-aware LM resolver.

The resolver routes ``provider: nous`` through ``_maybe_resolve_nous_lm``
when the auth.json pool entry has a refresh_token (signals OAuth-managed
flow). Pool entries without refresh_token (env-var-style) fall back to
the existing direct-pass-through path so we don't break that simpler
setup. Without a pool entry at all, the resolver fails with an
actionable `hermes model` recovery hint rather than silently routing to
something that won't work.
"""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import pytest

from evolution.core.hermes_provider import (
    HermesProviderError,
    ResolvedLM,
    resolve_default_lm,
)
from evolution.core.nous_lm import NousLM, _reset_state_for_tests


@pytest.fixture(autouse=True)
def _clean_nous_state():
    _reset_state_for_tests()
    yield
    _reset_state_for_tests()


@pytest.fixture
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "hermes_home"
    home.mkdir()
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "NOUS_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return home


def _write_config(home: Path, body: str) -> None:
    (home / "config.yaml").write_text(textwrap.dedent(body).lstrip())


def _write_nous_pool(
    home: Path,
    *,
    access_token: str = "oauth-tok",
    refresh_token: str | None = "refresh-tok",
    oauth_expires_at: str | None = "2026-12-01T00:00:00+00:00",
    agent_key: str | None = None,
    agent_key_expires_at: str | None = None,
    inference_base_url: str | None = None,
    extra: dict | None = None,
) -> None:
    entry: dict = {
        "access_token": access_token,
        "priority": 0,
    }
    if refresh_token is not None:
        entry["refresh_token"] = refresh_token
    if oauth_expires_at is not None:
        entry["expires_at"] = oauth_expires_at
    if agent_key is not None:
        entry["agent_key"] = agent_key
    if agent_key_expires_at is not None:
        entry["agent_key_expires_at"] = agent_key_expires_at
    if inference_base_url is not None:
        entry["inference_base_url"] = inference_base_url
    if extra:
        entry.update(extra)
    (home / "auth.json").write_text(
        json.dumps({"credential_pool": {"nous": [entry]}})
    )


# ---------------------------------------------------------------------------
# OAuth-managed flow: pool entry has refresh_token → NousLM factory
# ---------------------------------------------------------------------------


class TestNousResolutionWithOAuth:
    def test_oauth_pool_entry_returns_factory(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              default: Hermes-4-405B
              provider: nous
            """,
        )
        # Use a future agent_key so the factory's initial mint doesn't
        # actually fire when the test calls factory() — keeps this test
        # purely about the resolver's wiring.
        _write_nous_pool(
            hermes_home,
            agent_key="fresh-key",
            agent_key_expires_at="2026-12-01T00:00:00+00:00",
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert isinstance(lm, ResolvedLM)
        assert lm.model == "openai/Hermes-4-405B"
        assert lm.lm_kwargs == {}
        assert lm.lm_factory is not None
        assert lm.provider_hint == "nous"

    def test_factory_constructs_nous_lm_with_oauth_state(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              default: Hermes-4-405B
              provider: nous
            """,
        )
        _write_nous_pool(
            hermes_home,
            access_token="real-oauth-tok",
            refresh_token="real-refresh-tok",
            agent_key="initial-agent-key",
            agent_key_expires_at="2026-12-01T00:00:00+00:00",
        )
        resolved = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        nous_lm = resolved.lm_factory()
        assert isinstance(nous_lm, NousLM)
        assert nous_lm._shared_state.access_token == "real-oauth-tok"
        assert nous_lm._shared_state.refresh_token == "real-refresh-tok"
        assert nous_lm._shared_state.agent_key == "initial-agent-key"

    def test_custom_inference_base_url_flows_through(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              default: Hermes-4-405B
              provider: nous
            """,
        )
        _write_nous_pool(
            hermes_home,
            agent_key="fresh-key",
            agent_key_expires_at="2026-12-01T00:00:00+00:00",
            inference_base_url="https://custom-nous.example.com/v1",
        )
        resolved = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        nous_lm = resolved.lm_factory()
        assert nous_lm.kwargs["api_base"] == "https://custom-nous.example.com/v1"


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------


class TestNousResolutionFallbacks:
    def test_pool_entry_with_agent_key_no_refresh_falls_back_to_direct(
        self, hermes_home
    ):
        # Hand-edited or inference-only entry: has access_token + agent_key
        # but no refresh_token. The resolver must fall through to the
        # existing OpenAI-wire direct-pass-through path. The agent_key
        # presence signals "this is an inference-ready credential, not a
        # partial OAuth setup."
        _write_config(
            hermes_home,
            """
            model:
              default: Hermes-4-405B
              provider: nous
            """,
        )
        _write_nous_pool(
            hermes_home,
            access_token="bare-api-key",
            refresh_token=None,
            oauth_expires_at=None,
            agent_key="inference-ready-bearer",
        )
        resolved = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        # Direct pass-through path: openai/<model>, api_base + api_key in lm_kwargs,
        # no factory.
        assert resolved.lm_factory is None
        assert resolved.model == "openai/Hermes-4-405B"
        assert resolved.lm_kwargs.get("api_key") == "bare-api-key"

    def test_pool_entry_without_refresh_or_agent_key_raises(self, hermes_home):
        # Partial OAuth setup: pool entry has access_token but no
        # refresh_token AND no agent_key. Almost certainly an interrupted
        # `hermes model` run. Raising here gives the operator a specific
        # recovery hint instead of letting inference 401 with no breadcrumb.
        _write_config(
            hermes_home,
            """
            model:
              default: Hermes-4-405B
              provider: nous
            """,
        )
        _write_nous_pool(
            hermes_home,
            access_token="oauth-only",
            refresh_token=None,
            oauth_expires_at=None,
            agent_key=None,
        )
        with pytest.raises(HermesProviderError, match="partial OAuth setup"):
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)

    def test_missing_pool_entry_surfaces_recovery_hint(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              default: Hermes-4-405B
              provider: nous
            """,
        )
        # No auth.json written → no credential pool.
        with pytest.raises(HermesProviderError, match="hermes model"):
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)

    def test_empty_access_token_in_oauth_entry_surfaces_recovery(
        self, hermes_home
    ):
        _write_config(
            hermes_home,
            """
            model:
              default: Hermes-4-405B
              provider: nous
            """,
        )
        # Has refresh_token (OAuth-managed signal) but no access_token.
        _write_nous_pool(
            hermes_home,
            access_token="",
            refresh_token="refresh-tok",
        )
        with pytest.raises(HermesProviderError, match="hermes model"):
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)

    def test_no_model_default_for_oauth_path_surfaces_actionable(
        self, hermes_home
    ):
        _write_config(
            hermes_home,
            """
            model:
              provider: nous
            """,
        )
        _write_nous_pool(hermes_home)
        with pytest.raises(HermesProviderError, match="model.default"):
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)


# ---------------------------------------------------------------------------
# Pool exhaustion regression
# ---------------------------------------------------------------------------


class TestPoolExhaustionRespected:
    def test_exhausted_entry_skipped_with_future_reset(self, hermes_home):
        # The existing _is_pool_entry_usable logic skips entries Hermes
        # marked exhausted with a future cooldown. Confirm it still
        # applies to the Nous OAuth path — should fall through to the
        # missing-entry error.
        _write_config(
            hermes_home,
            """
            model:
              default: Hermes-4-405B
              provider: nous
            """,
        )
        _write_nous_pool(
            hermes_home,
            extra={
                "last_status": "exhausted",
                "last_error_reset_at": time.time() + 3600,  # 1h in future
            },
        )
        with pytest.raises(HermesProviderError, match="hermes model"):
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)

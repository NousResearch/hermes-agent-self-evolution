"""Tests for OpenAI Codex resolution in the Hermes-aware LM resolver.

Codex differs from every other provider: OAuth-only (no env-var fallback),
and the resolved LM needs OAuth state + a refresh hook that doesn't fit
the stock dspy.LM constructor. The resolver returns a ResolvedLM whose
``lm_factory`` constructs the CodexLM subclass; consumers route through
``instantiate_lm`` to honor the factory.
"""

from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import pytest

from evolution.core.codex_lm import CodexLM, _reset_state_for_tests
from evolution.core.hermes_provider import (
    HermesProviderError,
    ResolvedLM,
    instantiate_lm,
    resolve_default_lm,
)


@pytest.fixture(autouse=True)
def _clean_codex_state():
    # Codex shared OAuth state bleeds across tests via module-level cache.
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
    ):
        monkeypatch.delenv(var, raising=False)
    return home


def _write_config(home: Path, body: str) -> None:
    (home / "config.yaml").write_text(textwrap.dedent(body).lstrip())


def _write_codex_pool(
    home: Path,
    *,
    access_token: str = "tok-access",
    refresh_token: str = "tok-refresh",
    expires_at: float | None = None,
    base_url: str | None = None,
) -> None:
    entry: dict = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "priority": 0,
    }
    if expires_at is not None:
        entry["expires_at"] = expires_at
    if base_url is not None:
        entry["base_url"] = base_url
    (home / "auth.json").write_text(
        json.dumps({"credential_pool": {"openai-codex": [entry]}})
    )


# ---------------------------------------------------------------------------
# Resolver returns a ResolvedLM with lm_factory set
# ---------------------------------------------------------------------------


class TestCodexResolution:
    def test_resolver_returns_factory_for_openai_codex(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              default: gpt-5-codex
              provider: openai-codex
            """,
        )
        _write_codex_pool(
            hermes_home,
            access_token="tok-access",
            refresh_token="tok-refresh",
            expires_at=time.time() + 3600,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert isinstance(lm, ResolvedLM)
        # Resolver normalizes to openai/<model> so LiteLLM routes via its
        # OpenAI provider (which honors api_base + Bearer auth). User can
        # also pass an already-prefixed name in config.yaml, in which case
        # the prefix is preserved verbatim.
        assert lm.model == "openai/gpt-5-codex"
        # Codex packs everything in the factory closure; lm_kwargs is empty.
        assert lm.lm_kwargs == {}
        assert lm.lm_factory is not None
        assert lm.provider_hint == "openai-codex"

    def test_factory_constructs_codex_lm_with_oauth_state(self, hermes_home):
        expiry = time.time() + 3600
        _write_config(
            hermes_home,
            """
            model:
              default: gpt-5-codex
              provider: openai-codex
            """,
        )
        _write_codex_pool(
            hermes_home,
            access_token="tok-access-real",
            refresh_token="tok-refresh-real",
            expires_at=expiry,
        )
        resolved = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        codex_lm = resolved.lm_factory()
        assert isinstance(codex_lm, CodexLM)
        assert codex_lm.kwargs["api_key"] == "tok-access-real"
        assert codex_lm._shared_state.refresh_token == "tok-refresh-real"
        assert codex_lm._shared_state.expires_at == pytest.approx(expiry)

    def test_factory_uses_custom_base_url_from_pool(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              default: gpt-5-codex
              provider: openai-codex
            """,
        )
        _write_codex_pool(
            hermes_home,
            access_token="tok",
            refresh_token="rt",
            expires_at=time.time() + 3600,
            base_url="https://custom-codex.example.com/v1",
        )
        resolved = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        codex_lm = resolved.lm_factory()
        assert codex_lm.kwargs["api_base"] == "https://custom-codex.example.com/v1"

    def test_default_base_url_when_pool_omits(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              default: gpt-5-codex
              provider: openai-codex
            """,
        )
        _write_codex_pool(
            hermes_home,
            access_token="tok",
            refresh_token="rt",
            expires_at=time.time() + 3600,
        )
        resolved = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        codex_lm = resolved.lm_factory()
        assert "chatgpt.com/backend-api/codex" in codex_lm.kwargs["api_base"]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestCodexResolutionErrors:
    def test_missing_pool_entry_surfaces_recovery_hint(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              default: gpt-5-codex
              provider: openai-codex
            """,
        )
        # No auth.json written → no credential_pool["openai-codex"].
        with pytest.raises(
            HermesProviderError, match="hermes auth add openai-codex"
        ):
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)

    def test_empty_access_token_in_pool_surfaces_recovery_hint(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              default: gpt-5-codex
              provider: openai-codex
            """,
        )
        _write_codex_pool(
            hermes_home,
            access_token="",  # empty
            refresh_token="rt",
            expires_at=time.time() + 3600,
        )
        with pytest.raises(
            HermesProviderError, match="hermes auth add openai-codex"
        ):
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)

    def test_no_model_default_surfaces_actionable_error(self, hermes_home):
        # Codex has no standalone fallback model — the user's ChatGPT plan
        # determines what they have access to (gpt-5, gpt-5-codex).
        _write_config(
            hermes_home,
            """
            model:
              provider: openai-codex
            """,
        )
        _write_codex_pool(
            hermes_home,
            access_token="tok",
            refresh_token="rt",
            expires_at=time.time() + 3600,
        )
        with pytest.raises(HermesProviderError, match="model.default"):
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)


# ---------------------------------------------------------------------------
# instantiate_lm helper — the factory dispatch
# ---------------------------------------------------------------------------


class TestInstantiateLMHelper:
    def test_factory_path_invokes_factory(self):
        sentinel = object()
        called = {"n": 0}

        def fake_factory():
            called["n"] += 1
            return sentinel

        resolved = ResolvedLM(
            model="gpt-5-codex",
            lm_kwargs={},
            source="test",
            lm_factory=fake_factory,
        )
        result = instantiate_lm(resolved)
        assert called["n"] == 1
        assert result is sentinel

    def test_factory_path_applies_role_kwargs_via_copy(self):
        # Use a real CodexLM so .copy() works (deepcopy of subclass).
        from evolution.core.codex_lm import CodexLM as _CodexLM

        def factory():
            return _CodexLM(
                model="gpt-5-codex",
                access_token="tok",
                refresh_token="rt",
                expires_at=time.time() + 3600,
            )

        resolved = ResolvedLM(
            model="gpt-5-codex",
            lm_kwargs={},
            source="test",
            lm_factory=factory,
        )
        lm = instantiate_lm(resolved, request_timeout=60, num_retries=5)
        # dspy.LM.copy() routes kwargs to either instance attrs or self.kwargs
        # depending on whether the name exists on the class. num_retries is a
        # BaseLM instance attr; request_timeout flows into kwargs (and onward
        # to litellm.responses).
        assert lm.kwargs.get("request_timeout") == 60
        assert lm.num_retries == 5

    def test_no_factory_falls_back_to_dspy_lm(self):
        # Regression guard: ensure the helper still produces a stock
        # dspy.LM for non-Codex providers (the ones with lm_factory=None).
        import dspy

        resolved = ResolvedLM(
            model="anthropic/claude-haiku-4-5",
            lm_kwargs={"api_key": "ant-test"},
            source="test",
            lm_factory=None,
        )
        lm = instantiate_lm(resolved, request_timeout=60)
        assert isinstance(lm, dspy.LM)
        assert lm.model == "anthropic/claude-haiku-4-5"
        assert lm.kwargs.get("api_key") == "ant-test"
        assert lm.kwargs.get("request_timeout") == 60

"""Project-wide pytest fixtures.

Many tests construct ``EvolutionConfig()`` and patch ``dspy.LM`` upstream
to avoid real network calls. With Hermes-aware LM resolution, the
resolver runs *before* ``dspy.LM`` at every construction site, so a CI
runner with no ``~/.hermes/config.yaml`` and no provider env vars hits
``HermesProviderError`` before the patched ``dspy.LM`` ever sees the
mock.

This autouse fixture:

1. Seeds a fake ``OPENAI_API_KEY`` for the test session so the resolver
   finds *something* and returns a deterministic ``ResolvedLM``.
2. Stubs the auth-check preflight to a no-op so existing tests that
   patch ``dspy.LM`` upstream don't accidentally make real
   ``litellm.completion`` calls during preflight (preflight bypasses
   dspy by design).

Tests that explicitly need the no-credentials path or want to exercise
preflight clear env vars / patch preflight in their own fixtures.

The fake key is never sent anywhere because every test that exercises
the LM construction path patches ``dspy.LM``, and preflight is stubbed
out below.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _stub_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the Hermes resolver always finds a credential during tests.

    Tests that need the no-credentials path override this by clearing
    env vars and pinning a fresh ``hermes_home`` (see
    ``tests/core/test_hermes_provider.py::hermes_home``).
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-pytest-key-never-sent")


@pytest.fixture(autouse=True)
def _stub_auth_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-op the auth-check preflight by default. Preflight bypasses
    dspy.LM (calling litellm.completion directly), so it's not covered
    by tests that patch dspy.LM upstream — leaving it un-stubbed would
    have integration tests trying to hit api.openai.com with the fake
    key and failing with whatever the real endpoint returns.

    Patches both the source module and every call-site binding (the
    evolve modules import preflight as ``_preflight_lm_credentials`` at
    module top, so patching the source alone wouldn't reach them).

    Tests in ``tests/skills/test_evolve_skill_auth.py`` explicitly want
    to exercise preflight; they apply their own ``patch(...)`` which
    overrides this no-op for the test's scope.
    """
    noop = lambda lms, **kwargs: None  # noqa: E731
    monkeypatch.setattr("evolution.core.auth_check.preflight", noop)
    # Call-site bindings — silently skip if the module isn't imported.
    for module_path in (
        "evolution.skills.evolve_skill._preflight_lm_credentials",
        "evolution.tools.evolve_tool._preflight_lm_credentials",
    ):
        monkeypatch.setattr(module_path, noop, raising=False)

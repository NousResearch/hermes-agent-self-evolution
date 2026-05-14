"""Project-wide pytest fixtures.

Many tests construct ``EvolutionConfig()`` and patch ``dspy.LM`` upstream
to avoid real network calls. With Hermes-aware LM resolution, the
resolver runs *before* ``dspy.LM`` at every construction site, so a CI
runner with no ``~/.hermes/config.yaml`` and no provider env vars hits
``HermesProviderError`` before the patched ``dspy.LM`` ever sees the
mock.

This autouse fixture seeds a fake ``OPENAI_API_KEY`` for the test
session so the resolver finds *something* and returns a deterministic
``ResolvedLM`` — tests don't depend on the developer's local Hermes
install or shell env. Tests that explicitly need the no-credentials
path (the resolver's hard-error tests) clear env vars in their own
fixture and pass an isolated ``hermes_home`` tmp dir.

The fake key is never sent anywhere because every test that exercises
the LM construction path patches ``dspy.LM``.
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

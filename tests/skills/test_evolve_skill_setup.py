"""Tests for module-level setup in evolve_skill.

Currently covers the litellm request-timeout configuration installed at import
time. Without the timeout, any silent upstream drop hangs every LLM call in the
optimization loop forever.
"""

import importlib
import os
from unittest.mock import patch

import litellm
import pytest


@pytest.fixture
def reload_evolve_skill():
    """Reload the module so the env-var-driven timeout is re-read."""

    def _reload():
        import evolution.skills.evolve_skill as mod

        return importlib.reload(mod)

    return _reload


class TestLitellmRequestTimeout:
    """The module sets litellm.request_timeout at import time."""

    def test_default_timeout_is_90s(self, reload_evolve_skill):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LITELLM_REQUEST_TIMEOUT", None)
            reload_evolve_skill()
            assert litellm.request_timeout == 90.0

    def test_env_var_overrides_default(self, reload_evolve_skill):
        with patch.dict(os.environ, {"LITELLM_REQUEST_TIMEOUT": "30"}):
            reload_evolve_skill()
            assert litellm.request_timeout == 30.0

    def test_env_var_accepts_float(self, reload_evolve_skill):
        with patch.dict(os.environ, {"LITELLM_REQUEST_TIMEOUT": "12.5"}):
            reload_evolve_skill()
            assert litellm.request_timeout == 12.5

    def test_invalid_env_var_raises_at_import(self, reload_evolve_skill):
        """Bad values fail fast rather than silently disable the timeout."""
        with patch.dict(os.environ, {"LITELLM_REQUEST_TIMEOUT": "not-a-number"}):
            with pytest.raises(ValueError):
                reload_evolve_skill()

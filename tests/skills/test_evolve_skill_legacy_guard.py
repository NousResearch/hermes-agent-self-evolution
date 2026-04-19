"""Tests for legacy DSPy path safety guards."""

import pytest

from evolution.skills.evolve_skill import enforce_legacy_backend_guard


class TestLegacyBackendGuard:
    def test_legacy_guard_blocks_local_hermes_api(self):
        with pytest.raises(RuntimeError) as exc:
            enforce_legacy_backend_guard("http://127.0.0.1:8642/v1")

        assert "local Hermes API" in str(exc.value)

    def test_legacy_guard_allows_empty_api_base(self):
        enforce_legacy_backend_guard("")

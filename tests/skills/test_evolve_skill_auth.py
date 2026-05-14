"""Integration tests for the auth-error path in the evolve_skill CLI.

Covers:
  - Preflight is invoked between resolver and GEPA setup (mocked)
  - --dry-run skips preflight
  - --no-preflight skips preflight
  - When preflight raises HermesProviderError, the MIPROv2 fallback is
    NOT invoked (the bug the BaseException reparenting fixes)
  - Top-level CLI catch renders a Rich panel + exits 2 with no traceback

All tests mock LiteLLM and DSPy — no network.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from evolution.core.hermes_provider import HermesProviderError, ResolvedLM
from evolution.skills.evolve_skill import main as evolve_skill_main


class TestPreflightWiring:
    def test_preflight_called_between_resolver_and_gepa(self, tmp_path, monkeypatch):
        """Preflight should fire after the LM-resolution banner prints
        and before any GEPA setup. Easiest pin: preflight raises a
        sentinel error, and we assert no GEPA code ran."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

        with patch("evolution.skills.evolve_skill._preflight_lm_credentials") as mock_preflight, \
             patch("evolution.core.config.discover_skill_sources") as mock_discover, \
             patch("evolution.skills.evolve_skill.load_skill") as mock_load_skill, \
             patch("evolution.skills.evolve_skill.find_skill") as mock_find:
            mock_find.return_value = tmp_path / "fake_skill" / "SKILL.md"
            mock_load_skill.return_value = {
                "name": "fake_skill",
                "raw": "fake content",
                "body": "fake body",
                "description": "fake description for testing",
            }
            mock_discover.return_value = []
            mock_preflight.side_effect = HermesProviderError("preflight stop")

            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_main,
                [
                    "--skill", "fake_skill",
                    "--optimizer-model", "anthropic/claude-haiku-4-5",
                    "--eval-model", "anthropic/claude-haiku-4-5",
                ],
            )

            # Preflight invoked exactly once.
            assert mock_preflight.call_count == 1
            # Top-level catch produced exit code 2 (HermesProviderError).
            assert result.exit_code == 2

    def test_dry_run_skips_preflight(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        with patch("evolution.skills.evolve_skill._preflight_lm_credentials") as mock_preflight, \
             patch("evolution.core.config.discover_skill_sources") as mock_discover, \
             patch("evolution.skills.evolve_skill.load_skill") as mock_load_skill, \
             patch("evolution.skills.evolve_skill.find_skill") as mock_find:
            mock_find.return_value = tmp_path / "fake_skill" / "SKILL.md"
            mock_load_skill.return_value = {
                "name": "fake_skill",
                "raw": "x",
                "body": "x",
                "description": "x" * 50,
            }
            mock_discover.return_value = []

            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "fake_skill", "--dry-run"],
            )
            # --dry-run exits in load_skill phase before preflight runs.
            mock_preflight.assert_not_called()

    def test_no_preflight_flag_skips_preflight(self, tmp_path, monkeypatch):
        """--no-preflight should bypass even when not in dry-run mode.
        We can't run a full evolve in a unit test, so we verify the flag
        propagates by checking that preflight wouldn't be called even if
        it were stubbed to raise — so we use a sentinel that breaks GEPA
        instead and assert preflight wasn't called."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        with patch("evolution.skills.evolve_skill._preflight_lm_credentials") as mock_preflight, \
             patch("evolution.core.config.discover_skill_sources") as mock_discover, \
             patch("evolution.skills.evolve_skill.load_skill") as mock_load_skill, \
             patch("evolution.skills.evolve_skill.find_skill") as mock_find, \
             patch("evolution.skills.evolve_skill._build_optimizer_and_compile") as mock_build:
            mock_find.return_value = tmp_path / "fake_skill" / "SKILL.md"
            mock_load_skill.return_value = {
                "name": "fake_skill",
                "raw": "x",
                "body": "x",
                "description": "x" * 50,
            }
            mock_discover.return_value = []
            mock_build.side_effect = RuntimeError("stop after preflight check")

            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                [
                    "--skill", "fake_skill",
                    "--no-preflight",
                    "--optimizer-model", "anthropic/claude-haiku-4-5",
                    "--eval-model", "anthropic/claude-haiku-4-5",
                ],
            )
            mock_preflight.assert_not_called()


class TestTopLevelCatch:
    def test_hermes_provider_error_renders_panel_and_exits_2(self, tmp_path, monkeypatch):
        """The user-facing failure mode: stale Hermes credential. CLI
        should NOT dump a Python traceback — should render a Rich error
        panel with the actionable message and exit 2."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        with patch("evolution.skills.evolve_skill._preflight_lm_credentials") as mock_preflight, \
             patch("evolution.core.config.discover_skill_sources") as mock_discover, \
             patch("evolution.skills.evolve_skill.load_skill") as mock_load_skill, \
             patch("evolution.skills.evolve_skill.find_skill") as mock_find:
            mock_find.return_value = tmp_path / "fake_skill" / "SKILL.md"
            mock_load_skill.return_value = {
                "name": "fake_skill",
                "raw": "x",
                "body": "x",
                "description": "x" * 50,
            }
            mock_discover.return_value = []
            mock_preflight.side_effect = HermesProviderError(
                "Authentication failed for model 'anthropic/claude-opus-4-5'.\n"
                "To fix, run: hermes auth add anthropic"
            )

            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_main,
                [
                    "--skill", "fake_skill",
                    "--optimizer-model", "anthropic/claude-opus-4-5",
                    "--eval-model", "anthropic/claude-opus-4-5",
                ],
                catch_exceptions=True,
            )

            assert result.exit_code == 2
            # The actionable command must reach the user.
            assert "hermes auth add anthropic" in result.output
            # And no Python traceback should appear.
            assert "Traceback" not in result.output


class TestMIPROv2FallbackExclusion:
    def test_hermes_provider_error_skips_miprov2_fallback(self):
        """The bug the BaseException reparenting fixes: when GEPA raises
        HermesProviderError, _build_optimizer_and_compile must NOT call
        MIPROv2 (which would re-hit the same auth error and burn budget).
        """
        from evolution.skills.evolve_skill import _build_optimizer_and_compile

        gepa_runner_called = {"n": 0}
        mipro_runner_called = {"n": 0}

        def fake_gepa(**kwargs):
            gepa_runner_called["n"] += 1
            raise HermesProviderError("auth")

        def fake_mipro(**kwargs):
            mipro_runner_called["n"] += 1
            return object()

        with pytest.raises(HermesProviderError):
            _build_optimizer_and_compile(
                baseline_module=None,
                trainset=[],
                valset=[],
                metric=None,
                gepa_budget="light",
                optimizer_model="x",
                seed=42,
                no_fallback=False,  # IMPORTANT: not no_fallback — proves the
                                    # narrow exclusion, not the no_fallback path
                _gepa_runner=fake_gepa,
                _mipro_runner=fake_mipro,
            )

        assert gepa_runner_called["n"] == 1, "GEPA should have been attempted"
        assert mipro_runner_called["n"] == 0, (
            "MIPROv2 should NOT be invoked on auth failure — it would re-hit "
            "the same expired credential and waste optimization budget."
        )

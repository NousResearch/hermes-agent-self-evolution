"""Integration tests for the cost-advisor wiring in the evolve_skill CLI.

Covers:
  * Cost suggestion fires when --eval-model is unset and a cheaper
    same-provider alternative exists in the LiteLLM catalog
  * Cost suggestion does NOT fire when --eval-model is explicit
  * --no-cost-suggest suppresses the panel
  * --dry-run suppresses the panel (matches existing preflight behavior)
  * Off-catalog models (Bedrock, Codex) produce no panel and no error
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from evolution.core.cost_advisor import CheaperAlternative
from evolution.core.hermes_provider import HermesProviderError
from evolution.skills.evolve_skill import main as evolve_skill_main


def _fake_alternative() -> CheaperAlternative:
    """Stand-in for find_cheaper_alternative's output. Realistic-ish numbers
    so panel rendering doesn't choke on edge cases.
    """
    return CheaperAlternative(
        current_model="anthropic/claude-opus-4-5",
        current_input_cost_per_1m=5.0,
        current_output_cost_per_1m=25.0,
        current_max_input_tokens=200_000,
        suggested_model="anthropic/claude-haiku-4-5",
        suggested_input_cost_per_1m=1.0,
        suggested_output_cost_per_1m=5.0,
        suggested_max_input_tokens=200_000,
        input_cost_ratio=5.0,
        output_cost_ratio=5.0,
        provider="anthropic",
    )


@pytest.fixture
def stub_skill_loader(tmp_path):
    """Plumbs the skill loader so the CLI can reach the preflight + advisor
    block without needing a real SKILL.md on disk.
    """
    with patch("evolution.skills.evolve_skill.load_skill") as mock_load_skill, \
         patch("evolution.skills.evolve_skill.find_skill") as mock_find, \
         patch("evolution.core.config.discover_skill_sources") as mock_discover:
        mock_find.return_value = tmp_path / "fake_skill" / "SKILL.md"
        mock_load_skill.return_value = {
            "name": "fake_skill",
            "raw": "fake content",
            "body": "fake body",
            "description": "fake description for testing the cost advisor",
        }
        mock_discover.return_value = []
        yield


class TestCostSuggestionFiringRules:
    def test_fires_when_eval_model_unset_and_cheaper_alt_exists(
        self, stub_skill_loader, monkeypatch
    ):
        """The headline behavior: a Hermes user defaulting to Opus sees the
        Haiku suggestion before any expensive work runs."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

        with patch(
            "evolution.skills.evolve_skill._find_cheaper_alternative",
            return_value=_fake_alternative(),
        ) as mock_finder, \
             patch("evolution.skills.evolve_skill._preflight_lm_credentials") as mock_preflight, \
             patch("evolution.skills.evolve_skill._build_optimizer_and_compile") as mock_build:
            # Stop the run right after preflight + advisor — we only care
            # about the wiring, not what GEPA does with the result.
            mock_build.side_effect = RuntimeError("stop after advisor")

            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_main,
                ["--skill", "fake_skill"],
            )

            mock_finder.assert_called_once()
            # The CLI output should contain the suggestion panel's recovery
            # snippet so the user can copy-paste.
            assert "--eval-model anthropic/claude-haiku-4-5" in result.output

    def test_does_not_fire_when_eval_model_explicit(
        self, stub_skill_loader, monkeypatch
    ):
        """User explicitly passed --eval-model — they made a choice, the
        advisor should respect it and stay silent.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

        with patch(
            "evolution.skills.evolve_skill._find_cheaper_alternative"
        ) as mock_finder, \
             patch("evolution.skills.evolve_skill._preflight_lm_credentials"), \
             patch("evolution.skills.evolve_skill._build_optimizer_and_compile") as mock_build:
            mock_build.side_effect = RuntimeError("stop after advisor")

            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                [
                    "--skill", "fake_skill",
                    "--eval-model", "anthropic/claude-haiku-4-5",
                ],
            )
            mock_finder.assert_not_called()

    def test_no_cost_suggest_flag_suppresses_panel(
        self, stub_skill_loader, monkeypatch
    ):
        """--no-cost-suggest is the user's way of saying "I know, leave me
        alone." The advisor should never run.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

        with patch(
            "evolution.skills.evolve_skill._find_cheaper_alternative"
        ) as mock_finder, \
             patch("evolution.skills.evolve_skill._preflight_lm_credentials"), \
             patch("evolution.skills.evolve_skill._build_optimizer_and_compile") as mock_build:
            mock_build.side_effect = RuntimeError("stop after advisor")

            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "fake_skill", "--no-cost-suggest"],
            )
            mock_finder.assert_not_called()

    def test_dry_run_skips_advisor(self, stub_skill_loader, monkeypatch):
        """--dry-run early-returns before preflight + advisor (matches the
        existing preflight skip-on-dry-run behavior). Resolver isn't even
        consulted.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

        with patch(
            "evolution.skills.evolve_skill._find_cheaper_alternative"
        ) as mock_finder:
            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "fake_skill", "--dry-run"],
            )
            mock_finder.assert_not_called()

    def test_no_panel_when_alternative_is_none(
        self, stub_skill_loader, monkeypatch
    ):
        """Resolved model isn't in the LiteLLM catalog (Bedrock, Codex,
        local-server endpoints) — find_cheaper_alternative returns None and
        no panel renders. Smoke that the run continues normally.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

        with patch(
            "evolution.skills.evolve_skill._find_cheaper_alternative",
            return_value=None,
        ) as mock_finder, \
             patch("evolution.skills.evolve_skill._render_cost_suggestion_panel") as mock_render, \
             patch("evolution.skills.evolve_skill._preflight_lm_credentials"), \
             patch("evolution.skills.evolve_skill._build_optimizer_and_compile") as mock_build:
            mock_build.side_effect = RuntimeError("stop after advisor")

            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "fake_skill"],
            )
            mock_finder.assert_called_once()
            mock_render.assert_not_called()

    def test_codex_lm_factory_skips_advisor(
        self, stub_skill_loader, monkeypatch
    ):
        """When the resolver returns an LM with a custom factory (Codex),
        the advisor must not fire. Suggesting a non-Codex model implies
        an auth setup the user didn't opt in to (Codex uses ChatGPT-
        subscription OAuth, not an OPENAI_API_KEY).
        """
        from evolution.core.hermes_provider import ResolvedLM

        monkeypatch.setenv("OPENAI_API_KEY", "k")

        codex_resolved = ResolvedLM(
            model="openai/gpt-5-codex",
            lm_kwargs={},
            source="hermes-config:openai-codex(...)",
            lm_factory=lambda: object(),  # any non-None callable
            provider_hint="openai-codex",
        )

        with patch(
            "evolution.skills.evolve_skill.resolve_default_lm",
            return_value=codex_resolved,
        ), \
             patch(
                 "evolution.skills.evolve_skill._find_cheaper_alternative"
             ) as mock_finder, \
             patch("evolution.skills.evolve_skill._preflight_lm_credentials"), \
             patch("evolution.skills.evolve_skill._build_optimizer_and_compile") as mock_build:
            mock_build.side_effect = RuntimeError("stop after advisor")

            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "fake_skill"],
            )
            mock_finder.assert_not_called()

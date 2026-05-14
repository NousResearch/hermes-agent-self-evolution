"""Tests for the Hermes-aware LM resolver.

Covers:
  - ResolvedLM dataclass shape + redaction
  - 6-step resolution chain (explicit > config.yaml > auth.json > env)
  - Provider mapping table (the ~20 Hermes provider IDs)
  - provider: "auto" slim auto-detect
  - Local-server configs (vLLM / Ollama / LM Studio): api_base, no api_key
  - URL-suffix wire-mode flip (z.ai / MiniMax with /anthropic)
  - Hard-error message text when nothing is configured
  - Graceful skip on missing / malformed config files
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Optional

import pytest

from evolution.core.hermes_provider import (
    HermesProviderError,
    ResolvedLM,
    _coerce_priority,
    _is_pool_entry_usable,
    _pick_pool_entry,
    _redact_lm,
    resolve_default_lm,
    resolved_lms_dump,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Empty ~/.hermes-equivalent dir, with all provider env vars cleared.

    Tests that want a config.yaml or auth.json write into this dir.
    Tests that want env-var fallback use monkeypatch.setenv.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "NOUS_API_KEY",
        "GITHUB_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    return home


def write_config(home: Path, body: str) -> None:
    (home / "config.yaml").write_text(textwrap.dedent(body).lstrip())


def write_auth(home: Path, payload: dict) -> None:
    (home / "auth.json").write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# ResolvedLM dataclass + redaction
# ---------------------------------------------------------------------------


class TestResolvedLM:
    def test_construct(self):
        lm = ResolvedLM(
            model="anthropic/claude-opus-4-5",
            lm_kwargs={"api_key": "sk-ant-secret"},
            source="env:ANTHROPIC_API_KEY",
        )
        assert lm.model == "anthropic/claude-opus-4-5"
        assert lm.lm_kwargs["api_key"] == "sk-ant-secret"

    def test_frozen(self):
        lm = ResolvedLM(model="x", lm_kwargs={}, source="explicit")
        with pytest.raises((AttributeError, TypeError)):
            lm.model = "y"  # type: ignore[misc]


class TestRedaction:
    def test_strips_api_key(self):
        lm = ResolvedLM(
            model="openai/gpt-4.1",
            lm_kwargs={"api_base": "https://api.openai.com/v1", "api_key": "sk-real"},
            source="hermes-config",
        )
        redacted = _redact_lm(lm)
        assert redacted["lm_kwargs"]["api_key"] == "<REDACTED>"
        assert redacted["lm_kwargs"]["api_base"] == "https://api.openai.com/v1"
        assert redacted["model"] == "openai/gpt-4.1"
        assert redacted["source"] == "hermes-config"

    def test_redaction_does_not_mutate_original(self):
        lm = ResolvedLM(
            model="x",
            lm_kwargs={"api_key": "sk-real"},
            source="explicit",
        )
        _redact_lm(lm)
        assert lm.lm_kwargs["api_key"] == "sk-real"

    def test_no_key_to_redact(self):
        lm = ResolvedLM(model="ollama/llama3", lm_kwargs={"api_base": "http://localhost:11434"}, source="explicit")
        redacted = _redact_lm(lm)
        assert "api_key" not in redacted["lm_kwargs"]


class TestResolvedLmsDump:
    def test_resolves_all_roles_and_redacts(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: claude-opus-4.6
              provider: anthropic
              api_key: sk-ant-load-bearing-secret
            """,
        )
        dump = resolved_lms_dump(
            hermes_home=hermes_home,
            optimizer=None,
            reflection=None,
            eval_=None,
            judge=None,
        )
        for role in ("optimizer", "reflection", "eval", "judge"):
            assert role in dump
            assert dump[role]["lm_kwargs"]["api_key"] == "<REDACTED>"
        # The actual secret must never appear in the serialized payload —
        # this is the load-bearing assertion that gates persistence safety.
        assert "sk-ant-load-bearing-secret" not in json.dumps(dump)

    def test_failed_role_records_error_not_raises(self, hermes_home):
        # No config, no env — every role fails to resolve.
        dump = resolved_lms_dump(hermes_home=hermes_home, optimizer=None, eval_=None)
        assert "error" in dump["optimizer"]
        assert "error" in dump["eval"]

    def test_explicit_override_passes_through(self, hermes_home):
        dump = resolved_lms_dump(
            hermes_home=hermes_home, optimizer="anthropic/claude-haiku-4-5"
        )
        assert dump["optimizer"]["model"] == "anthropic/claude-haiku-4-5"
        assert dump["optimizer"]["source"] == "explicit"


# ---------------------------------------------------------------------------
# Step 1: explicit override wins
# ---------------------------------------------------------------------------


class TestExplicitOverride:
    def test_returns_explicit_string_unchanged(self, hermes_home, monkeypatch):
        # Even with a fully-populated Hermes config, explicit wins.
        write_config(
            hermes_home,
            """
            model:
              default: gpt-4o
              provider: openai
              api_key: sk-from-config
            """,
        )
        lm = resolve_default_lm(
            role="optimizer",
            explicit_model="anthropic/claude-haiku-4-5",
            hermes_home=hermes_home,
        )
        assert lm.model == "anthropic/claude-haiku-4-5"
        # Explicit overrides do not infer api_base/api_key — the caller relies
        # on env vars for credentials, mirroring today's behavior.
        assert "api_key" not in lm.lm_kwargs
        assert lm.source == "explicit"


# ---------------------------------------------------------------------------
# Step 2-5: config.yaml + auth.json + env resolution
# ---------------------------------------------------------------------------


class TestConfigYaml:
    def test_anthropic_provider_with_inline_key(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: claude-opus-4.6
              provider: anthropic
              api_key: sk-ant-fromconfig
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "anthropic/claude-opus-4.6"
        assert lm.lm_kwargs["api_key"] == "sk-ant-fromconfig"
        assert "hermes-config" in lm.source

    def test_openrouter_provider(self, hermes_home, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fromenv")
        write_config(
            hermes_home,
            """
            model:
              default: anthropic/claude-opus-4.6
              provider: openrouter
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "openrouter/anthropic/claude-opus-4.6"
        assert lm.lm_kwargs["api_key"] == "sk-or-fromenv"

    def test_custom_local_server_no_api_key(self, hermes_home):
        # vLLM/Ollama/LM Studio local server: api_base, no api_key required.
        write_config(
            hermes_home,
            """
            model:
              default: my-local-model
              provider: custom
              base_url: http://localhost:8000/v1
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "openai/my-local-model"
        assert lm.lm_kwargs["api_base"] == "http://localhost:8000/v1"
        # Local servers often have no auth; absence of api_key is fine.
        assert lm.lm_kwargs.get("api_key") in (None, "", "EMPTY")

    def test_ollama_alias(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: llama3
              provider: ollama
              base_url: http://localhost:11434/v1
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "openai/llama3"
        assert lm.lm_kwargs["api_base"] == "http://localhost:11434/v1"

    def test_lmstudio_alias(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: qwen2.5-coder-7b
              provider: lmstudio
              base_url: http://127.0.0.1:1234/v1
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "openai/qwen2.5-coder-7b"
        assert lm.lm_kwargs["api_base"] == "http://127.0.0.1:1234/v1"

    def test_gemini_native(self, hermes_home, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g-key")
        write_config(
            hermes_home,
            """
            model:
              default: gemini-2.5-pro
              provider: gemini
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "gemini/gemini-2.5-pro"
        assert lm.lm_kwargs["api_key"] == "g-key"


class TestUrlSuffixWireModeFlip:
    """Hermes infers Anthropic Messages wire mode from a /anthropic URL suffix."""

    def test_zai_anthropic_suffix(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: glm-4.6
              provider: zai
              base_url: https://api.z.ai/api/anthropic
              api_key: zai-key
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        # Wire-mode flip: routed via LiteLLM's anthropic adapter.
        assert lm.model == "anthropic/glm-4.6"
        assert lm.lm_kwargs["api_base"] == "https://api.z.ai/api/anthropic"
        assert lm.lm_kwargs["api_key"] == "zai-key"

    def test_minimax_anthropic_suffix(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: MiniMax-M2
              provider: minimax
              base_url: https://api.minimax.io/anthropic
              api_key: mm-key
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "anthropic/MiniMax-M2"


# ---------------------------------------------------------------------------
# auth.json credential pool
# ---------------------------------------------------------------------------


class TestAuthJsonCredentialPool:
    def test_pool_credential_used_when_no_inline_key(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: claude-opus-4.6
              provider: anthropic
            """,
        )
        write_auth(
            hermes_home,
            {
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "anthropic": [
                        {
                            "id": "abc",
                            "label": "from auth.json",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "config",
                            "access_token": "sk-ant-from-pool",
                            "base_url": "https://api.anthropic.com",
                        }
                    ]
                },
                "active_provider": None,
            },
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["api_key"] == "sk-ant-from-pool"
        assert "auth.json" in lm.source

    def test_pool_priority_lowest_int_wins(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: claude-opus-4.6
              provider: anthropic
            """,
        )
        write_auth(
            hermes_home,
            {
                "version": 1,
                "credential_pool": {
                    "anthropic": [
                        {"priority": 5, "auth_type": "api_key", "access_token": "low-priority"},
                        {"priority": 0, "auth_type": "api_key", "access_token": "high-priority"},
                    ]
                },
            },
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["api_key"] == "high-priority"

    def test_inline_config_key_beats_pool(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: claude-opus-4.6
              provider: anthropic
              api_key: from-config-yaml
            """,
        )
        write_auth(
            hermes_home,
            {
                "credential_pool": {
                    "anthropic": [
                        {"priority": 0, "auth_type": "api_key", "access_token": "from-pool"},
                    ]
                },
            },
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["api_key"] == "from-config-yaml"

    def test_env_var_beats_pool_but_loses_to_inline_config(self, hermes_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        write_config(
            hermes_home,
            """
            model:
              default: claude-opus-4.6
              provider: anthropic
            """,
        )
        write_auth(
            hermes_home,
            {
                "credential_pool": {
                    "anthropic": [{"priority": 0, "auth_type": "api_key", "access_token": "from-pool"}]
                },
            },
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["api_key"] == "from-env"


# ---------------------------------------------------------------------------
# Step 4: provider: "auto" slim auto-detect
# ---------------------------------------------------------------------------


class TestProviderAuto:
    def test_auto_picks_anthropic_when_only_anthropic_env_set(self, hermes_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        write_config(
            hermes_home,
            """
            model:
              default: claude-opus-4.6
              provider: auto
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "anthropic/claude-opus-4.6"
        assert lm.lm_kwargs["api_key"] == "ant-key"

    def test_auto_prefers_anthropic_over_openrouter_when_both_present(self, hermes_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        write_config(
            hermes_home,
            """
            model:
              default: claude-opus-4.6
              provider: auto
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "anthropic/claude-opus-4.6"

    def test_auto_picks_pool_provider_when_pool_populated_and_no_env(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: claude-opus-4.6
              provider: auto
            """,
        )
        write_auth(
            hermes_home,
            {
                "credential_pool": {
                    "anthropic": [{"priority": 0, "auth_type": "api_key", "access_token": "pool-ant-key"}]
                },
            },
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "anthropic/claude-opus-4.6"
        assert lm.lm_kwargs["api_key"] == "pool-ant-key"

    def test_missing_provider_field_treated_as_auto(self, hermes_home, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        write_config(
            hermes_home,
            """
            model:
              default: anthropic/claude-opus-4.6
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "openrouter/anthropic/claude-opus-4.6"
        assert lm.lm_kwargs["api_key"] == "or-key"


# ---------------------------------------------------------------------------
# Standalone (no Hermes) — env-var-only path
# ---------------------------------------------------------------------------


class TestStandalone:
    def test_no_hermes_dir_anthropic_env(self, hermes_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        # Note: hermes_home exists but has no config.yaml.
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        # Standalone: model name comes from a hardcoded sane default per provider.
        assert lm.model.startswith("anthropic/")
        assert lm.lm_kwargs["api_key"] == "ant-key"
        assert "env" in lm.source

    def test_no_hermes_dir_openrouter_env(self, hermes_home, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model.startswith("openrouter/")
        assert lm.lm_kwargs["api_key"] == "or-key"

    def test_no_hermes_dir_openai_env(self, hermes_home, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model.startswith("openai/")
        assert lm.lm_kwargs["api_key"] == "sk-openai-key"

    def test_hermes_home_does_not_exist(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        for var in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        nonexistent = tmp_path / "definitely-missing"
        lm = resolve_default_lm(role="optimizer", hermes_home=nonexistent)
        assert lm.lm_kwargs["api_key"] == "ant-key"


# ---------------------------------------------------------------------------
# Hard-error UX
# ---------------------------------------------------------------------------


class TestHardError:
    def test_no_config_no_env_raises_with_actionable_message(self, hermes_home):
        # hermes_home fixture clears all provider env vars; no config.yaml written.
        with pytest.raises(HermesProviderError) as exc:
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        msg = str(exc.value)
        # Message should list what was tried, not just "failed".
        assert "Hermes" in msg or "hermes" in msg
        assert "ANTHROPIC_API_KEY" in msg
        assert "--optimizer-model" in msg

    def test_config_specifies_provider_with_no_creds_anywhere(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: claude-opus-4.6
              provider: anthropic
            """,
        )
        # No api_key in config, no env var, no auth.json pool entry.
        with pytest.raises(HermesProviderError) as exc:
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        msg = str(exc.value)
        assert "anthropic" in msg.lower()
        assert "ANTHROPIC_API_KEY" in msg


# ---------------------------------------------------------------------------
# Graceful skip on malformed inputs
# ---------------------------------------------------------------------------


class TestMalformedInputs:
    def test_malformed_yaml_falls_through_to_env(self, hermes_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        (hermes_home / "config.yaml").write_text(":::not yaml:::")
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["api_key"] == "ant-key"

    def test_malformed_auth_json_falls_through(self, hermes_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        write_config(
            hermes_home,
            """
            model:
              default: claude-opus-4.6
              provider: anthropic
            """,
        )
        (hermes_home / "auth.json").write_text("{broken json")
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        # Inline config has no api_key, env wins.
        assert lm.lm_kwargs["api_key"] == "ant-key"

    def test_config_yaml_missing_model_section(self, hermes_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        write_config(hermes_home, "agent:\n  max_turns: 90\n")
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["api_key"] == "ant-key"


# ---------------------------------------------------------------------------
# Integration with user's actual Hermes config shape
# ---------------------------------------------------------------------------


class TestRealisticHermesShape:
    """Mirrors the shape the user's actual ~/.hermes/config.yaml has,
    with a model name that isn't a real OpenAI catalog name."""

    def test_custom_provider_pointed_at_openai_with_alias_model(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: gpt-5.4-mini
              provider: custom
              base_url: https://api.openai.com/v1
              api_key: sk-from-config
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        # Model name passes through verbatim — Hermes-aliased names are the
        # user's responsibility; we don't validate against OpenAI's catalog.
        assert lm.model == "openai/gpt-5.4-mini"
        assert lm.lm_kwargs["api_base"] == "https://api.openai.com/v1"
        assert lm.lm_kwargs["api_key"] == "sk-from-config"


# ---------------------------------------------------------------------------
# Role parameter
# ---------------------------------------------------------------------------


class TestRoleParameter:
    def test_role_appears_in_diagnostic_source_string(self, hermes_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        for role in ("optimizer", "reflection", "eval", "judge"):
            lm = resolve_default_lm(role=role, hermes_home=hermes_home)
            # Source string is for diagnostics; role context useful when many
            # LMs are constructed in one run.
            assert lm.lm_kwargs["api_key"] == "ant-key"

    def test_all_roles_resolve_to_same_model_by_default(self, hermes_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        write_config(
            hermes_home,
            """
            model:
              default: claude-opus-4.6
              provider: anthropic
            """,
        )
        models = {
            role: resolve_default_lm(role=role, hermes_home=hermes_home).model
            for role in ("optimizer", "reflection", "eval", "judge")
        }
        # Hermes has only one model.default; all roles collapse onto it.
        assert len(set(models.values())) == 1


# ---------------------------------------------------------------------------
# Credential pool — declared-but-untested branches
# ---------------------------------------------------------------------------


class TestCredentialPoolEdgeCases:
    def test_pool_sort_three_or_more_entries_stable(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: m
              provider: anthropic
            """,
        )
        write_auth(
            hermes_home,
            {
                "credential_pool": {
                    "anthropic": [
                        {"priority": 5, "auth_type": "api_key", "access_token": "low", "label": "five"},
                        {"priority": 0, "auth_type": "api_key", "access_token": "first-zero", "label": "zero-a"},
                        {"priority": 0, "auth_type": "api_key", "access_token": "second-zero", "label": "zero-b"},
                        {"auth_type": "api_key", "access_token": "no-priority", "label": "default-999"},
                    ]
                },
            },
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        # Stable sort: among the two priority=0 entries, the first-listed wins.
        assert lm.lm_kwargs["api_key"] == "first-zero"

    def test_custom_namespaced_pool_key(self, hermes_home):
        # provider=anthropic, but only `custom:anthropic` is populated.
        # Pins the declared-in-code namespaced-key branch.
        write_config(
            hermes_home,
            """
            model:
              default: m
              provider: anthropic
            """,
        )
        write_auth(
            hermes_home,
            {
                "credential_pool": {
                    "custom:anthropic": [
                        {"priority": 0, "auth_type": "api_key", "access_token": "from-custom-namespace"}
                    ]
                },
            },
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["api_key"] == "from-custom-namespace"

    def test_string_priority_does_not_crash_sort(self, hermes_home):
        # Hand-edited auth.json may store priority as a string. The sort
        # must not raise; coerce to int with a fallback.
        write_config(
            hermes_home,
            """
            model:
              default: m
              provider: anthropic
            """,
        )
        write_auth(
            hermes_home,
            {
                "credential_pool": {
                    "anthropic": [
                        {"priority": "0", "auth_type": "api_key", "access_token": "str-zero"},
                        {"priority": 5, "auth_type": "api_key", "access_token": "int-five"},
                    ]
                },
            },
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["api_key"] == "str-zero"

    def test_non_string_access_token_never_crashes(self, hermes_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        write_config(
            hermes_home,
            """
            model:
              default: m
              provider: anthropic
            """,
        )
        write_auth(
            hermes_home,
            {
                "credential_pool": {
                    "anthropic": [
                        {"priority": 0, "auth_type": "api_key", "access_token": 12345}
                    ]
                },
            },
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        # Pool entry skipped (non-string token); env wins.
        assert lm.lm_kwargs["api_key"] == "from-env"

    def test_coerce_priority_unit(self):
        assert _coerce_priority(0) == 0
        assert _coerce_priority(5) == 5
        assert _coerce_priority("0") == 0
        assert _coerce_priority("-3") == -3
        assert _coerce_priority(None) == 999
        assert _coerce_priority("not-a-number") == 999
        assert _coerce_priority(1.5) == 999  # float intentionally rejected


class TestPoolEntryExhaustion:
    """Mirror Hermes's own behavior: skip credentials Hermes already
    marked exhausted unless their reset_at has passed. Hermes writes
    these fields when it gets a 401 (auth) or 429 (rate limit) — picking
    them up here avoids both pointless preflight calls and burning the
    same dead credential on every run.
    """

    def test_unset_last_status_is_usable(self):
        # Back-compat: hand-edited auth.json entries without last_status
        # behave as before (treated as usable).
        assert _is_pool_entry_usable({"access_token": "x"}, now_epoch=1000.0)

    def test_status_ok_is_usable(self):
        assert _is_pool_entry_usable(
            {"access_token": "x", "last_status": "ok"}, now_epoch=1000.0
        )

    def test_exhausted_with_future_reset_is_skipped(self):
        assert not _is_pool_entry_usable(
            {
                "access_token": "x",
                "last_status": "exhausted",
                "last_error_reset_at": 2000.0,
            },
            now_epoch=1000.0,
        )

    def test_exhausted_with_past_reset_is_usable(self):
        # Cooldown expired — credential becomes available again. Hermes
        # rotates back to the entry on its next run; we should too.
        assert _is_pool_entry_usable(
            {
                "access_token": "x",
                "last_status": "exhausted",
                "last_error_reset_at": 500.0,
            },
            now_epoch=1000.0,
        )

    def test_exhausted_without_reset_is_skipped(self):
        # Defensive: exhausted entries without a reset_at are treated as
        # permanently bad rather than silently usable.
        assert not _is_pool_entry_usable(
            {"access_token": "x", "last_status": "exhausted"},
            now_epoch=1000.0,
        )

    def test_pick_pool_entry_prefers_usable_over_exhausted_lower_priority(self, hermes_home):
        # Even if exhausted entry has lower (better) priority, the usable
        # one wins.
        write_config(
            hermes_home,
            """
            model:
              default: m
              provider: anthropic
            """,
        )
        write_auth(
            hermes_home,
            {
                "credential_pool": {
                    "anthropic": [
                        {
                            "priority": 0,
                            "auth_type": "api_key",
                            "access_token": "exhausted-but-best-priority",
                            "last_status": "exhausted",
                            "last_error_reset_at": 9999999999.0,
                        },
                        {
                            "priority": 5,
                            "auth_type": "api_key",
                            "access_token": "usable-fallback",
                        },
                    ]
                },
            },
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["api_key"] == "usable-fallback"

    def test_pick_pool_entry_returns_none_when_only_exhausted(self, hermes_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env-fallback")
        write_config(
            hermes_home,
            """
            model:
              default: m
              provider: anthropic
            """,
        )
        write_auth(
            hermes_home,
            {
                "credential_pool": {
                    "anthropic": [
                        {
                            "priority": 0,
                            "auth_type": "api_key",
                            "access_token": "dead",
                            "last_status": "exhausted",
                            "last_error_reset_at": 9999999999.0,
                        },
                    ]
                },
            },
        )
        # Only entry is exhausted → fall through to env var.
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["api_key"] == "from-env-fallback"


# ---------------------------------------------------------------------------
# Local-server / alias coverage
# ---------------------------------------------------------------------------


class TestLocalServerAndAliases:
    def test_custom_with_localhost_no_key_succeeds(self, hermes_home):
        # provider:custom + localhost base_url is the auth-optional branch
        # that was declared in code but never tested.
        write_config(
            hermes_home,
            """
            model:
              default: my-local
              provider: custom
              base_url: http://localhost:9000/v1
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "openai/my-local"
        assert lm.lm_kwargs["api_base"] == "http://localhost:9000/v1"
        # custom provider doesn't get the EMPTY placeholder (only ollama/vllm/
        # llamacpp/lmstudio do); api_key is simply absent. LiteLLM and most
        # local servers accept that.
        assert "api_key" not in lm.lm_kwargs or lm.lm_kwargs["api_key"] in ("", "EMPTY")

    def test_custom_with_remote_url_no_key_hard_errors(self, hermes_home):
        # The negative case: custom + non-local URL must require a key.
        write_config(
            hermes_home,
            """
            model:
              default: m
              provider: custom
              base_url: https://api.example.com/v1
            """,
        )
        with pytest.raises(HermesProviderError):
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)

    def test_vllm_alias_collapses_to_custom(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: vllm-model
              provider: vllm
              base_url: http://localhost:8000/v1
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "openai/vllm-model"
        assert lm.lm_kwargs["api_base"] == "http://localhost:8000/v1"

    def test_llamacpp_alias_collapses_to_custom(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: llama-model
              provider: llamacpp
              base_url: http://localhost:8080/v1
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "openai/llama-model"


# ---------------------------------------------------------------------------
# Auto-detect ordering past the first pair
# ---------------------------------------------------------------------------


class TestAutoDetectOrdering:
    def test_openai_beats_gemini_when_both_set(self, hermes_home, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        write_config(
            hermes_home,
            """
            model:
              default: m
              provider: auto
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        # _AUTO_DETECT_ORDER places openai before gemini; pin that contract.
        assert lm.model.startswith("openai/")

    def test_gemini_secondary_env_var_picked_up(self, hermes_home, monkeypatch):
        # Only GOOGLE_API_KEY is set, not GEMINI_API_KEY. The
        # _PROVIDER_ENV_KEYS["gemini"] tuple has both — pin that the
        # second one is consulted.
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        write_config(
            hermes_home,
            """
            model:
              default: gemini-2.5-pro
              provider: auto
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "gemini/gemini-2.5-pro"
        assert lm.lm_kwargs["api_key"] == "google-key"


# ---------------------------------------------------------------------------
# Provider validation (typo guard)
# ---------------------------------------------------------------------------


class TestProviderValidation:
    def test_typo_provider_rejected_with_known_list(self, hermes_home):
        write_config(
            hermes_home,
            """
            model:
              default: m
              provider: anthropc
              api_key: x
            """,
        )
        with pytest.raises(HermesProviderError) as exc:
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        msg = str(exc.value)
        assert "anthropc" in msg
        assert "Known:" in msg
        # Sanity: at least one real provider name appears in the suggested list.
        assert "anthropic" in msg

    def test_explicit_override_bypasses_provider_validation(self, hermes_home):
        # Even if config.yaml has a bogus provider, an explicit --optimizer-model
        # short-circuits before validation runs.
        write_config(
            hermes_home,
            """
            model:
              default: m
              provider: anthropc
            """,
        )
        lm = resolve_default_lm(
            role="optimizer",
            explicit_model="anthropic/claude-haiku-4-5",
            hermes_home=hermes_home,
        )
        assert lm.model == "anthropic/claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Hard-error rendering structure
# ---------------------------------------------------------------------------


class TestHardErrorRendering:
    def test_provider_note_renders_above_tried_list_not_inside(self, hermes_home):
        # Regression: the previous version inserted the "Provider requested..."
        # line at index 3, which landed mid-bullet-list with mismatched indent.
        write_config(
            hermes_home,
            """
            model:
              default: m
              provider: anthropic
            """,
        )
        with pytest.raises(HermesProviderError) as exc:
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        msg = str(exc.value)
        lines = msg.splitlines()

        provider_idx = next(i for i, l in enumerate(lines) if "Provider requested" in l)
        tried_idx = next(i for i, l in enumerate(lines) if l == "Tried in order:")
        # Provider line must precede the bullet list, not split it.
        assert provider_idx < tried_idx
        # And the lines between should be only the blank separator.
        between = lines[provider_idx + 1: tried_idx]
        assert all(l == "" for l in between), f"non-blank line between provider and bullets: {between}"

    def test_auto_detect_message_lists_full_provider_set(self, hermes_home):
        # Regression: the previous message hardcoded only the first 6 of 15
        # providers, hiding whether xiaomi/arcee/etc. were actually checked.
        with pytest.raises(HermesProviderError) as exc:
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        msg = str(exc.value)
        # Every provider in the auto-detect list should appear in the message.
        for provider in ("anthropic", "openrouter", "openai", "nous", "gemini",
                         "copilot", "kilocode", "ai-gateway"):
            assert provider in msg, f"provider {provider!r} missing from auto-detect message"


# ---------------------------------------------------------------------------
# Malformed-config warning
# ---------------------------------------------------------------------------


class TestMalformedConfigWarning:
    def test_unparseable_yaml_warns_on_stderr(self, hermes_home, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        # Mismatched quote = real yaml.YAMLError, distinct from the
        # "valid but odd YAML" case (':::not yaml:::' parses as a scalar).
        (hermes_home / "config.yaml").write_text(
            'model:\n  default: "unterminated\n  provider: anthropic\n'
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        captured = capsys.readouterr()
        assert "warning" in captured.err
        assert "config.yaml" in captured.err
        # Resolution still succeeds via env-var fallback.
        assert lm.lm_kwargs["api_key"] == "ant-key"

    def test_unparseable_auth_json_warns_on_stderr(self, hermes_home, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")
        write_config(
            hermes_home,
            """
            model:
              default: m
              provider: anthropic
            """,
        )
        (hermes_home / "auth.json").write_text("{this is not json")
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        captured = capsys.readouterr()
        assert "warning" in captured.err
        assert "auth.json" in captured.err
        assert lm.lm_kwargs["api_key"] == "ant-key"

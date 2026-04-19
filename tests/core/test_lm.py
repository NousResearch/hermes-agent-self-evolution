"""Tests for centralized DSPy LM construction and additive codex-batch config."""

from evolution.core.config import EvolutionConfig
from evolution.core.lm import build_lm


class TestBuildLm:
    def test_build_lm_passes_openai_compatible_overrides(self, monkeypatch):
        captured = {}

        def fake_lm(model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs
            return {"model": model, "kwargs": kwargs}

        monkeypatch.setattr("evolution.core.lm.dspy.LM", fake_lm)
        config = EvolutionConfig(
            lm_api_base="http://127.0.0.1:8642/v1",
            lm_api_key="dummy",
            lm_timeout_seconds=42,
        )

        result = build_lm("openai/hermes-agent", config)

        assert result["model"] == "openai/hermes-agent"
        assert captured["kwargs"]["api_base"] == "http://127.0.0.1:8642/v1"
        assert captured["kwargs"]["api_key"] == "dummy"
        assert captured["kwargs"]["timeout"] == 42
        assert captured["kwargs"]["model_type"] == "chat"

    def test_build_lm_omits_empty_backend_overrides(self, monkeypatch):
        captured = {}

        def fake_lm(model, **kwargs):
            captured["kwargs"] = kwargs
            return object()

        monkeypatch.setattr("evolution.core.lm.dspy.LM", fake_lm)
        config = EvolutionConfig(lm_api_base="", lm_api_key="", lm_timeout_seconds=30)

        build_lm("openai/gpt-4.1", config)

        assert "api_base" not in captured["kwargs"]
        assert "api_key" not in captured["kwargs"]
        assert captured["kwargs"]["timeout"] == 30


class TestCodexBatchConfig:
    def test_config_exposes_safe_codex_batch_defaults(self):
        config = EvolutionConfig()

        assert config.max_codex_calls == 3
        assert config.max_examples == 8
        assert config.phase_timeout_seconds == 180
        assert config.max_run_seconds == 600
        assert config.codex_bin == "codex"

    def test_config_reads_codex_batch_env_overrides(self, monkeypatch):
        monkeypatch.setenv("HERMES_EVOLUTION_MAX_CODEX_CALLS", "7")
        monkeypatch.setenv("HERMES_EVOLUTION_MAX_EXAMPLES", "5")
        monkeypatch.setenv("HERMES_EVOLUTION_PHASE_TIMEOUT_SECONDS", "55")
        monkeypatch.setenv("HERMES_EVOLUTION_MAX_RUN_SECONDS", "999")

        config = EvolutionConfig()

        assert config.max_codex_calls == 7
        assert config.max_examples == 5
        assert config.phase_timeout_seconds == 55
        assert config.max_run_seconds == 999

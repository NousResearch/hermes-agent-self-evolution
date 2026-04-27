"""Tests for MiniMax provider support in EvolutionConfig."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from evolution.core.config import EvolutionConfig, MINIMAX_MODELS, MINIMAX_BASE_URL


# Provide a dummy hermes_agent_path so EvolutionConfig() doesn't auto-discover
_DUMMY_PATH = Path("/tmp")


class TestMinimaxConstants:
    def test_minimax_models_contains_expected_ids(self):
        assert "MiniMax-M2.7" in MINIMAX_MODELS
        assert "MiniMax-M2.7-highspeed" in MINIMAX_MODELS

    def test_minimax_base_url(self):
        assert MINIMAX_BASE_URL == "https://api.minimax.io/v1"

    def test_no_embedding_models(self):
        # MiniMax has no embedding model — none of the IDs should imply embeddings
        for model in MINIMAX_MODELS:
            assert "embed" not in model.lower()


class TestEvolutionConfigMinimaxDefaults:
    def test_minimax_api_key_defaults_to_empty(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINIMAX_API_KEY", None)
            config = EvolutionConfig(hermes_agent_path=_DUMMY_PATH)
        assert config.minimax_api_key == ""

    def test_minimax_api_key_read_from_env(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-12345"}):
            config = EvolutionConfig(hermes_agent_path=_DUMMY_PATH)
        assert config.minimax_api_key == "test-key-12345"

    def test_minimax_base_url_default(self):
        config = EvolutionConfig(hermes_agent_path=_DUMMY_PATH)
        assert config.minimax_base_url == MINIMAX_BASE_URL

    def test_minimax_base_url_overridable(self):
        config = EvolutionConfig(
            hermes_agent_path=_DUMMY_PATH,
            minimax_base_url="https://custom.endpoint/v1",
        )
        assert config.minimax_base_url == "https://custom.endpoint/v1"


class TestMakeLm:
    @pytest.fixture
    def config_with_key(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-minimax-testkey"}):
            return EvolutionConfig(hermes_agent_path=_DUMMY_PATH)

    def test_make_lm_minimax_bare_model_id(self, config_with_key):
        """Bare model ID (no prefix) routes to MiniMax."""
        mock_lm = MagicMock()
        with patch("dspy.LM", return_value=mock_lm) as mock_dspy_lm:
            result = config_with_key.make_lm("MiniMax-M2.7")

        mock_dspy_lm.assert_called_once_with(
            "openai/MiniMax-M2.7",
            api_key="sk-minimax-testkey",
            base_url=MINIMAX_BASE_URL,
            temperature=1.0,
        )
        assert result is mock_lm

    def test_make_lm_minimax_prefixed(self, config_with_key):
        """minimax/ prefix is stripped before routing."""
        mock_lm = MagicMock()
        with patch("dspy.LM", return_value=mock_lm) as mock_dspy_lm:
            config_with_key.make_lm("minimax/MiniMax-M2.7")

        mock_dspy_lm.assert_called_once_with(
            "openai/MiniMax-M2.7",
            api_key="sk-minimax-testkey",
            base_url=MINIMAX_BASE_URL,
            temperature=1.0,
        )

    def test_make_lm_minimax_openai_prefixed(self, config_with_key):
        """openai/MiniMax-M2.7 prefix is stripped and routed to MiniMax."""
        mock_lm = MagicMock()
        with patch("dspy.LM", return_value=mock_lm) as mock_dspy_lm:
            config_with_key.make_lm("openai/MiniMax-M2.7")

        mock_dspy_lm.assert_called_once_with(
            "openai/MiniMax-M2.7",
            api_key="sk-minimax-testkey",
            base_url=MINIMAX_BASE_URL,
            temperature=1.0,
        )

    def test_make_lm_highspeed_model(self, config_with_key):
        """MiniMax-M2.7-highspeed is also routed correctly."""
        mock_lm = MagicMock()
        with patch("dspy.LM", return_value=mock_lm) as mock_dspy_lm:
            config_with_key.make_lm("MiniMax-M2.7-highspeed")

        mock_dspy_lm.assert_called_once_with(
            "openai/MiniMax-M2.7-highspeed",
            api_key="sk-minimax-testkey",
            base_url=MINIMAX_BASE_URL,
            temperature=1.0,
        )

    def test_make_lm_temperature_one(self, config_with_key):
        """MiniMax requires temperature in (0, 1] — must default to 1.0."""
        with patch("dspy.LM") as mock_dspy_lm:
            config_with_key.make_lm("MiniMax-M2.7")

        _, kwargs = mock_dspy_lm.call_args
        assert kwargs["temperature"] == 1.0
        assert kwargs["temperature"] > 0.0

    def test_make_lm_minimax_no_api_key_raises(self):
        """Missing MINIMAX_API_KEY raises a clear ValueError."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MINIMAX_API_KEY", None)
            config = EvolutionConfig(hermes_agent_path=_DUMMY_PATH)

        with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
            config.make_lm("MiniMax-M2.7")

    def test_make_lm_openai_passes_through(self):
        """Non-MiniMax models are forwarded to dspy.LM as-is."""
        config = EvolutionConfig(hermes_agent_path=_DUMMY_PATH)
        mock_lm = MagicMock()
        with patch("dspy.LM", return_value=mock_lm) as mock_dspy_lm:
            result = config.make_lm("openai/gpt-4.1")

        mock_dspy_lm.assert_called_once_with("openai/gpt-4.1")
        assert result is mock_lm

    def test_make_lm_anthropic_passes_through(self):
        """Anthropic models pass through unchanged."""
        config = EvolutionConfig(hermes_agent_path=_DUMMY_PATH)
        with patch("dspy.LM") as mock_dspy_lm:
            config.make_lm("anthropic/claude-3-5-sonnet-20241022")

        mock_dspy_lm.assert_called_once_with("anthropic/claude-3-5-sonnet-20241022")

    def test_make_lm_custom_base_url(self):
        """Custom minimax_base_url is forwarded to dspy.LM."""
        config = EvolutionConfig(
            hermes_agent_path=_DUMMY_PATH,
            minimax_api_key="my-key",
            minimax_base_url="https://private.minimax.io/v1",
        )
        with patch("dspy.LM") as mock_dspy_lm:
            config.make_lm("MiniMax-M2.7")

        _, kwargs = mock_dspy_lm.call_args
        assert kwargs["base_url"] == "https://private.minimax.io/v1"


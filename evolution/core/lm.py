"""Centralized DSPy LM construction for self-evolution runs.

This repo often runs against non-default OpenAI-compatible backends, especially
local Hermes API instances. Centralizing LM creation keeps that wiring in one
place and avoids each module silently falling back to the wrong provider.
"""

from __future__ import annotations

from evolution.core.config import EvolutionConfig
import dspy


def build_lm(model: str, config: EvolutionConfig | None = None, *, model_type: str = "chat") -> dspy.LM:
    """Construct a DSPy LM with optional OpenAI-compatible backend overrides."""
    config = config or EvolutionConfig()
    kwargs: dict[str, object] = {
        "model_type": model_type,
        "timeout": config.lm_timeout_seconds,
    }
    if config.lm_api_base:
        kwargs["api_base"] = config.lm_api_base
    if config.lm_api_key:
        kwargs["api_key"] = config.lm_api_key
    return dspy.LM(model, **kwargs)

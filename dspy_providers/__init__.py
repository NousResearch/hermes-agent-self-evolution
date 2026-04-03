"""dspy_providers — OpenAI-free DSPy LM providers.

This package provides DSPy v2 BaseLM implementations for APIs that aren't
directly supported by the LiteLLM layer (OpenAI, Anthropic, Google, etc.).

Usage:
    from dspy_providers import CodexLM

    lm = CodexLM(model="gpt-5.4-mini")
    dspy.settings.configure(lm=lm)

    # Or patch dspy.LM so any optimizer/eval config uses CodexLM:
    #   dspy.LM = CodexLM   (then pass model="gpt-5.4-mini")
    #   dspy.LM("gpt-5.4-mini")  → CodexLM(model="gpt-5.4-mini")

Available providers:
    CodexLM  — ChatGPT Codex Responses API (gpt-5.4-mini, o3, o4-mini, ...)
               Requires CODEX_ACCESS_TOKEN env var or access_token kwarg.
"""
from dspy_providers.codex_lm import CodexLM

__all__ = ["CodexLM"]

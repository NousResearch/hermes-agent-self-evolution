"""One-call setup to route all dspy.LM() calls through CodexLM.

Put this at the top of any script or entry point that uses dspy before
any optimizer/eval code runs:

    from dspy_providers.configure import configure_with_codex
    configure_with_codex()

After this, dspy.LM("gpt-5.4-mini") and dspy.LM("gpt-4.1-mini") both return
CodexLM instances backed by the Codex Responses API.

Environment variables:
    CODEX_ACCESS_TOKEN  — required; your ChatGPT Codex access token
    CODEX_BASE_URL      — optional; defaults to https://chatgpt.com/backend-api/codex
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import dspy

from dspy_providers.codex_lm import CodexLM


# Singleton — only patch once
_configured = False


def configure_with_codex(
    access_token: Optional[str] = None,
    default_model: str = "gpt-5.4-mini",
) -> CodexLM:
    """Patch dspy.LM to return CodexLM instances.

    After calling this, any code that calls dspy.LM("any-model-name") will get a
    CodexLM instance backed by the Codex Responses API, with unsupported model
    names auto-resolved to the nearest available model.

    Args:
        access_token: Codex token. If not provided, loaded from CODEX_ACCESS_TOKEN env var.
        default_model: The model to use when none is specified. Default: "gpt-5.4-mini".

    Returns:
        The configured CodexLM instance (also stored in dspy.settings.lm).
    """
    global _configured
    if _configured:
        # Already configured; return current lm if it's a CodexLM
        current = getattr(dspy.settings, "lm", None)
        if isinstance(current, CodexLM):
            return current
        # Fall through and reconfigure

    # Load token
    token = access_token or os.environ.get("CODEX_ACCESS_TOKEN", "")
    if not token:
        raise ValueError(
            "No Codex access token found. Set CODEX_ACCESS_TOKEN env var or pass "
            "access_token to configure_with_codex()."
        )
    os.environ["CODEX_ACCESS_TOKEN"] = token

    # Patch dspy.LM → CodexLM
    _original_lm = dspy.LM

    def _codex_lm(model: str = default_model, **kwargs) -> CodexLM:
        # Strip provider prefix (e.g. "openai/gpt-4.1-mini" → "gpt-4.1-mini")
        model = model.split("/")[-1] if "/" in model else model
        return CodexLM(model=model, access_token=token, **kwargs)

    dspy.LM = _codex_lm  # type: ignore[assignment]

    # Configure DSPy settings to use CodexLM as the default
    lm = CodexLM(model=default_model, access_token=token)
    dspy.settings.configure(lm=lm)

    _configured = True
    return lm


def auto_configure() -> bool:
    """Try to auto-configure CodexLM from CODEX_ACCESS_TOKEN env var.

    Returns True if a token was found and configuration succeeded.
    Returns False if no token was set — caller should fall back to litellm.
    """
    token = os.environ.get("CODEX_ACCESS_TOKEN", "")
    if not token:
        return False
    try:
        configure_with_codex(access_token=token)
        return True
    except Exception:
        return False

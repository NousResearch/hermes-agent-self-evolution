"""Thin LLM abstraction — wraps litellm for text completion.

Decouples evolutionary operators from any specific LLM framework.
All LLM calls in the EA engine go through this client.
"""

from __future__ import annotations

import json
import re
import logging
from typing import Optional

import litellm

logger = logging.getLogger(__name__)


class LLMClient:
    """Stateless, thread-safe LLM client.

    Uses litellm under the hood, supporting OpenAI-compatible endpoints
    via api_base/api_key configuration.
    """

    def __init__(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 16000,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str, system: str = "", temperature: Optional[float] = None) -> str:
        """Generate a text completion. Returns the raw string response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key

        try:
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    def complete_json(self, prompt: str, system: str = "", temperature: Optional[float] = None) -> Optional[dict]:
        """Generate a completion and parse as JSON. Returns None on parse failure."""
        text = self.complete(prompt, system=system, temperature=temperature)
        if not text:
            return None

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown fences or embedded in text
        for pattern in [r'```json\s*(.*?)```', r'```\s*(.*?)```', r'(\{.*\})', r'(\[.*\])']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        logger.warning(f"Could not parse JSON from LLM output: {text[:200]}")
        return None

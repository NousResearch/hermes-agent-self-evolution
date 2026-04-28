"""Secret scanning for eval datasets."""

from __future__ import annotations

import re
from typing import Any

SECRET_PATTERN = re.compile(
    r"("
    r"sk-ant-api\S+"
    r"|sk-or-v1-\S+"
    r"|sk-[A-Za-z0-9_\-]{20,}"
    r"|ghp_\S+"
    r"|ghu_\S+"
    r"|xoxb-\S+"
    r"|xapp-\S+"
    r"|AKIA[0-9A-Z]{16}"
    r"|Bearer\s+\S{20,}"
    r"|-----BEGIN\s+(RSA\s+)?PRIVATE\sKEY-----"
    r"|OPENAI_API_KEY\s*[=:]\s*\S+"
    r"|OPENROUTER_API_KEY\s*[=:]\s*\S+"
    r"|ANTHROPIC_API_KEY\s*[=:]\s*\S+"
    r"|GITHUB_TOKEN\s*[=:]\s*\S+"
    r"|DATABASE_URL\s*[=:]\s*\S+"
    r"|\bpassword\s*[=:]\s*\S+"
    r"|\bsecret\s*[=:]\s*\S+"
    r"|\btoken\s*[=:]\s*\S{10,}"
    r")",
    re.IGNORECASE,
)

SCAN_FIELDS = ("task_input", "expected_behavior")


def scan_examples_for_secrets(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Return passed/failed report for secret-like values in eval examples."""
    matches: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        for field in SCAN_FIELDS:
            text = str(example.get(field, ""))
            if SECRET_PATTERN.search(text):
                matches.append({"example_index": index, "field": field})
    return {"status": "failed" if matches else "passed", "matches": matches}

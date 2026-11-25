"""Shared access facts for Hermes' state.db session store."""

import re
from pathlib import Path

DEFAULT_DB = Path.home() / ".hermes" / "state.db"

# Sources whose "user" text is always a machine dispatch or probe. Transports
# (api_server, local, webui, ...) carry both human and injected text — they are
# kept, and the wrapper markers below do the filtering.
MACHINE_SOURCES = ("subagent", "cron", "task", "oneshot", "acp")
# System-injected wrappers that arrive as user-role text. Matched as substrings
# of the message head: transports prefix lines (e.g. "[user] [ASYNC DELEGATION").
# Known false-positive class: a genuine task quoting one of these markers in its
# first 200 chars is dropped — accepted (zero occurrences in the host DB).
SKIP_PREFIXES = (
    "[CONTEXT COMPACTION",
    "[ASYNC DELEGATION",
    "[IMPORTANT:",
    "[OUT-OF-BAND USER MESSAGE",
    "[Continuing toward your standing goal",
    "[System",
    "[Your active task list",
    "[STILL IN PROGRESS",
    "Available swarm workers",
)
MIN_TASK_CHARS = 10

_WORKSPACE_LINE = re.compile(r"^\[Workspace::v1:.*$", re.MULTILINE)
_WORKSPACE_CTX_TAG = re.compile(
    r"<workspace_context[^>]*?(?:/>|>.*?</workspace_context>)\s*", re.S
)
_TRANSPORT_TAG = re.compile(r"^\[[A-Za-z0-9_.-]+\]\s+", re.M)

# Anchored to known key formats to minimize false positives on normal prose.
SECRET_PATTERNS = re.compile(
    r'('
    r'sk-ant-api\S+'           # Anthropic API keys
    r'|sk-or-v1-\S+'          # OpenRouter API keys
    r'|sk-\S{20,}'            # Generic OpenAI-style keys (20+ chars after sk-)
    r'|ghp_\S+'               # GitHub personal access tokens
    r'|ghu_\S+'               # GitHub user tokens
    r'|xoxb-\S+'              # Slack bot tokens
    r'|xapp-\S+'              # Slack app tokens
    r'|ntn_\S+'               # Notion integration tokens
    r'|AKIA[0-9A-Z]{16}'      # AWS access key IDs
    r'|Bearer\s+\S{20,}'      # Bearer auth headers (20+ char tokens)
    r'|-----BEGIN\s+(RSA\s+)?PRIVATE\sKEY-----'  # PEM private keys
    r'|ANTHROPIC_API_KEY'      # Known env var names (exact match)
    r'|OPENAI_API_KEY'
    r'|OPENROUTER_API_KEY'
    r'|SLACK_BOT_TOKEN'
    r'|GITHUB_TOKEN'
    r'|AWS_SECRET_ACCESS_KEY'
    r'|DATABASE_URL'
    r'|\bpassword\s*[=:]\s*\S+' # password assignments (password=xxx, password: xxx)
    r'|\bsecret\s*[=:]\s*\S+'   # secret assignments (secret=xxx, secret: xxx)
    r'|\btoken\s*[=:]\s*\S{10,}' # token assignments with 10+ char values
    r')',
    re.IGNORECASE,
)


def contains_secret(text: str) -> bool:
    """Check if text contains potential API keys or tokens."""
    return bool(SECRET_PATTERNS.search(text))


def clean_user_text(text: str | None) -> str:
    """Strip WebUI workspace tags, self-closing context tags, and transport prefixes."""
    text = _WORKSPACE_LINE.sub("", text or "")
    text = _WORKSPACE_CTX_TAG.sub("", text)
    text = _TRANSPORT_TAG.sub("", text.lstrip())
    return text.strip()


def is_task_like(text: str) -> bool:
    """A user message long enough to be a task, not a system wrapper."""
    if len(text) < MIN_TASK_CHARS:
        return False
    head = text[:200]
    return not any(marker in head for marker in SKIP_PREFIXES)

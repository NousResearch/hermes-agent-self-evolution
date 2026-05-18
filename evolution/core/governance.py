"""Session data governance and redaction pipeline."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re


SENSITIVE_PATTERNS = re.compile(
    r"("
    r"sk-ant-api\S+"
    r"|sk-or-v1-\S+"
    r"|sk-\S{20,}"
    r"|gh[pus]_\S+"
    r"|xoxb-\S+"
    r"|xapp-\S+"
    r"|AKIA[0-9A-Z]{16}"
    r"|Bearer\s+\S{20,}"
    r"|-----BEGIN\s+(RSA\s+)?PRIVATE\sKEY-----"
    r"|OPENAI_API_KEY"
    r"|ANTHROPIC_API_KEY"
    r"|OPENROUTER_API_KEY"
    r"|DATABASE_URL"
    r"|\b(password|secret|token)\s*[=:]\s*\S{6,}"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RedactionResult:
    """Result of sensitive-data redaction."""

    redacted_text: str
    changed: bool
    matches: int


def contains_sensitive_data(text: str) -> bool:
    """True when text likely contains a secret or credential."""
    return bool(SENSITIVE_PATTERNS.search(text or ""))


class SessionDataGovernance:
    """Governance checks for dataset ingestion from conversation history."""

    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days

    def redact_text(self, text: str) -> RedactionResult:
        """Redact sensitive spans while preserving overall message shape."""
        if not text:
            return RedactionResult(redacted_text="", changed=False, matches=0)

        matches = list(SENSITIVE_PATTERNS.finditer(text))
        if not matches:
            return RedactionResult(redacted_text=text, changed=False, matches=0)

        redacted = SENSITIVE_PATTERNS.sub("[REDACTED]", text)
        return RedactionResult(
            redacted_text=redacted,
            changed=True,
            matches=len(matches),
        )

    def is_within_retention(self, timestamp: int | float | None) -> bool:
        """Return True if timestamp is within retention window.

        Timestamps may be seconds or milliseconds since epoch.
        """
        if timestamp is None:
            return True

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=self.retention_days)
        ts = float(timestamp)
        if ts < 946684800:  # pre-2000 or synthetic counter-style timestamp
            return True
        if ts > 10_000_000_000:  # likely milliseconds
            ts /= 1000.0

        try:
            sample_time = datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return False
        return sample_time >= cutoff

    def sanitize_messages(self, messages: list[dict]) -> list[dict]:
        """Filter by retention and redact sensitive fields."""
        sanitized: list[dict] = []
        for message in messages:
            if not self.is_within_retention(message.get("timestamp")):
                continue

            text = message.get("task_input", "")
            redacted = self.redact_text(text)
            if not redacted.redacted_text.strip():
                continue

            cleaned = dict(message)
            cleaned["task_input"] = redacted.redacted_text
            cleaned["redaction_applied"] = redacted.changed
            cleaned["redaction_matches"] = redacted.matches
            sanitized.append(cleaned)
        return sanitized

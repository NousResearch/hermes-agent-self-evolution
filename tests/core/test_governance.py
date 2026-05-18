"""Tests for session data governance and redaction."""

from datetime import datetime, timedelta, timezone

from evolution.core.governance import (
    SessionDataGovernance,
    contains_sensitive_data,
)


def test_contains_sensitive_data_detects_api_key():
    assert contains_sensitive_data("OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123")


def test_redact_text_replaces_secret():
    governance = SessionDataGovernance()
    result = governance.redact_text("token=supersecretvalue123 and keep this text")
    assert result.changed
    assert "[REDACTED]" in result.redacted_text
    assert result.matches >= 1


def test_sanitize_messages_drops_stale_entries():
    governance = SessionDataGovernance(retention_days=7)
    old_ts = int(
        (datetime.now(tz=timezone.utc) - timedelta(days=30)).timestamp() * 1000,
    )
    messages = [
        {"task_input": "recent task", "timestamp": int(datetime.now(tz=timezone.utc).timestamp() * 1000)},
        {"task_input": "old task", "timestamp": old_ts},
    ]
    sanitized = governance.sanitize_messages(messages)
    assert len(sanitized) == 1
    assert sanitized[0]["task_input"] == "recent task"


def test_sanitize_messages_marks_redaction_metadata():
    governance = SessionDataGovernance()
    messages = [{"task_input": "password=hunter2-secret"}]
    sanitized = governance.sanitize_messages(messages)
    assert sanitized[0]["redaction_applied"] is True
    assert sanitized[0]["redaction_matches"] >= 1

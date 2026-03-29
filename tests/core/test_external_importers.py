"""Tests for external session importers.

Tests cover:
  - Secret detection and filtering (true positives + false positives)
  - Skill relevance heuristics
  - Claude Code history parsing + edge cases
  - Copilot events.jsonl parsing + edge cases
  - Hermes state.db parsing + edge cases
  - LLM scoring JSON parser
  - RelevanceFilter with mocked DSPy
  - build_dataset_from_external orchestration
  - _load_skill_text skill loader
  - CLI entry point via CliRunner
  - EvalExample serialization roundtrip
"""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from evolution.core.external_importers import (
    _contains_secret,
    _is_relevant_to_skill,
    _parse_scoring_json,
    _parse_copilot_events,
    _read_copilot_workspace,
    _load_skill_text,
    _validate_eval_example,
    build_dataset_from_external,
    main,
    ClaudeCodeImporter,
    CopilotImporter,
    HermesSessionImporter,
    RelevanceFilter,
    VALID_DIFFICULTIES,
    MIN_DATASET_SIZE,
)
from evolution.core.dataset_builder import EvalExample


# ── Secret Detection ────────────────────────────────────────────────────────


class TestSecretDetection:
    """Verify that known secret formats are caught and normal text is not."""

    def test_detects_anthropic_key(self):
        assert _contains_secret("here is sk-ant-api03-abc123def456 my key")

    def test_detects_openrouter_key(self):
        assert _contains_secret("sk-or-v1-abcdef1234567890abcdef")

    def test_detects_github_pat(self):
        assert _contains_secret("use ghp_abcdefghijklmnopqrstuvwxyz12")

    def test_detects_github_user_token(self):
        assert _contains_secret("ghu_abcdef123456")

    def test_detects_slack_bot_token(self):
        assert _contains_secret("SLACK_TOKEN is xoxb-123-456-abcdef")

    def test_detects_slack_app_token(self):
        assert _contains_secret("xapp-1-ABC-123456-xyz")

    def test_detects_notion_token(self):
        assert _contains_secret("ntn_37257299567bdiGRuYDIjhNH8uFribb461")

    def test_detects_bearer_token(self):
        assert _contains_secret("Authorization: Bearer dsm_bYxe4QUvPsDjRThUu2Qb3z")

    def test_detects_env_var_anthropic(self):
        assert _contains_secret("export ANTHROPIC_API_KEY=something")

    def test_detects_env_var_openai(self):
        assert _contains_secret("OPENAI_API_KEY=sk-blah")

    def test_detects_long_sk_prefix(self):
        assert _contains_secret("sk-proj-1234567890abcdefghij")

    def test_ignores_normal_text(self):
        assert not _contains_secret("sort these messages by topic")

    def test_ignores_short_sk_prefix(self):
        # "sk" alone or "sk-foo" (short) should not trigger
        assert not _contains_secret("I asked about sk")

    def test_ignores_prose_with_key_word(self):
        assert not _contains_secret("the key insight is that we need to refactor")

    def test_ignores_word_bearer_in_prose(self):
        # "Bearer" followed by short token shouldn't match (< 20 chars)
        assert not _contains_secret("the bearer of bad news")

    def test_ignores_code_token_discussion(self):
        # Discussing tokens in general shouldn't trigger
        assert not _contains_secret("parse the JWT token from the header")

    def test_ignores_ask_substring(self):
        # "ask" contains "sk" — should not trigger
        assert not _contains_secret("ask the user for their preferences")

    # -- Targeted assignment patterns (password=, secret=, token=) --

    def test_detects_password_assignment(self):
        assert _contains_secret("password=my_super_secret_123")
        assert _contains_secret("password: hunter2abc")

    def test_detects_secret_assignment(self):
        assert _contains_secret("secret=abc123def456")
        assert _contains_secret("secret: my_api_secret_key")

    def test_detects_token_assignment_long_value(self):
        assert _contains_secret("token=abcdef1234567890")
        assert _contains_secret("token: eyJhbGciOiJIUzI1NiJ9")

    def test_ignores_token_assignment_short_value(self):
        # "token=abc" has < 10 chars after = so should NOT trigger
        assert not _contains_secret("token=abc")

    def test_ignores_password_in_prose(self):
        # "password" without assignment operator shouldn't trigger
        assert not _contains_secret("the password field should be validated")
        assert not _contains_secret("reset your password using the link")

    def test_ignores_secret_in_prose(self):
        assert not _contains_secret("the secret to success is consistency")
        assert not _contains_secret("it's no secret that we need to refactor")

    def test_detects_openrouter_env_var(self):
        assert _contains_secret("export OPENROUTER_API_KEY=sk-or-xyz")

    def test_detects_slack_bot_token_env_var(self):
        assert _contains_secret("SLACK_BOT_TOKEN=xoxb-abc")

    def test_detects_github_token_env_var(self):
        assert _contains_secret("GITHUB_TOKEN=ghp_abc123")

    def test_detects_aws_access_key(self):
        assert _contains_secret("AKIAIOSFODNN7EXAMPLE")

    def test_detects_pem_private_key(self):
        assert _contains_secret("-----BEGIN RSA PRIVATE KEY-----\nMIIEow...")
        assert _contains_secret("-----BEGIN PRIVATE KEY-----\nMIIEvQ...")

    def test_detects_aws_secret_env_var(self):
        assert _contains_secret("export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG")

    def test_detects_database_url_env_var(self):
        assert _contains_secret("DATABASE_URL=postgres://user:pass@host/db")


# ── Relevance Heuristics ────────────────────────────────────────────────────


class TestRelevanceHeuristics:
    """Verify cheap pre-filter catches obvious matches and rejects non-matches."""

    SKILL_TEXT = (
        "Sort any batch of text (messages, notes, emails, transcripts) "
        "into topics. Works with natural language. Detects topics automatically "
        "if none provided. Supports categorization of content by theme."
    )

    def test_matches_skill_name_keyword(self):
        assert _is_relevant_to_skill(
            "categorize these messages", "tim-categorize", self.SKILL_TEXT
        )

    def test_matches_skill_content_keywords(self):
        assert _is_relevant_to_skill(
            "sort these transcripts into topics for me",
            "tim-categorize",
            self.SKILL_TEXT,
        )

    def test_rejects_unrelated(self):
        assert not _is_relevant_to_skill(
            "deploy the app to production",
            "tim-categorize",
            self.SKILL_TEXT,
        )

    def test_short_skill_name_words_ignored(self):
        # Words <= 3 chars from skill name shouldn't trigger false matches
        assert not _is_relevant_to_skill(
            "run the test suite", "tim-tdd", "Test driven development enforcement"
        )

    def test_single_keyword_not_enough(self):
        # Requires >= 2 keyword matches from skill text
        assert not _is_relevant_to_skill(
            "send an email to the team",
            "tim-categorize",
            self.SKILL_TEXT,
        )


# ── Scoring JSON Parser ─────────────────────────────────────────────────────


class TestScoringJsonParser:
    """Verify _parse_scoring_json handles various LLM output formats."""

    def test_clean_json(self):
        result = _parse_scoring_json('{"relevant": true, "difficulty": "easy"}')
        assert result["relevant"] is True

    def test_json_in_markdown(self):
        text = 'Here is my assessment:\n```json\n{"relevant": false}\n```'
        result = _parse_scoring_json(text)
        assert result is not None
        assert result["relevant"] is False

    def test_json_with_surrounding_text(self):
        text = 'I think this is relevant. {"relevant": true, "category": "sorting"} That is my answer.'
        result = _parse_scoring_json(text)
        assert result["category"] == "sorting"

    def test_no_json_returns_none(self):
        assert _parse_scoring_json("This is just plain text with no JSON") is None

    def test_malformed_json_returns_none(self):
        assert _parse_scoring_json("{broken json: ???}") is None

    def test_direct_parse_fast_path(self):
        """Clean JSON should be parsed directly without regex fallback."""
        result = _parse_scoring_json('{"relevant": true, "nested": {"key": "val"}}')
        assert result is not None
        assert result["relevant"] is True
        # Nested objects work via direct parse but would fail regex
        assert result["nested"]["key"] == "val"

    def test_non_dict_json_returns_none(self):
        """A JSON array or string should return None (we need a dict)."""
        assert _parse_scoring_json('[1, 2, 3]') is None
        assert _parse_scoring_json('"just a string"') is None

    def test_nested_braces_in_values(self):
        """JSON with braces inside string values must parse correctly."""
        text = 'Here: {"relevant": true, "expected_behavior": "handle {edge} cases"}'
        result = _parse_scoring_json(text)
        assert result is not None
        assert result["relevant"] is True
        assert "{edge}" in result["expected_behavior"]

    def test_empty_string_returns_none(self):
        assert _parse_scoring_json("") is None


# ── Claude Code Importer ────────────────────────────────────────────────────


class TestClaudeCodeImporter:
    def test_parses_history_jsonl(self, tmp_path):
        history = tmp_path / "history.jsonl"
        history.write_text(
            json.dumps({"display": "sort my slack messages by topic", "timestamp": 1700000000000, "project": "/test", "sessionId": "abc"}) + "\n"
            + json.dumps({"display": "yes go", "timestamp": 1700000001000, "project": "/test", "sessionId": "abc"}) + "\n"
            + json.dumps({"display": "here is sk-ant-api03-SECRETKEY123456789012345678 the key", "timestamp": 1700000002000, "project": "/test", "sessionId": "abc"}) + "\n"
        )

        with patch.object(ClaudeCodeImporter, "HISTORY_PATH", history):
            messages = ClaudeCodeImporter.extract_messages()

        # 1 valid message (second too short, third has secret)
        assert len(messages) == 1
        assert messages[0]["task_input"] == "sort my slack messages by topic"
        assert messages[0]["source"] == "claude-code"
        assert messages[0]["project"] == "/test"

    def test_handles_missing_file(self, tmp_path):
        with patch.object(ClaudeCodeImporter, "HISTORY_PATH", tmp_path / "nonexistent.jsonl"):
            messages = ClaudeCodeImporter.extract_messages()
        assert messages == []

    def test_respects_limit(self, tmp_path):
        history = tmp_path / "history.jsonl"
        lines = [
            json.dumps({"display": f"message number {i} with enough length to pass", "timestamp": i, "project": "/test", "sessionId": "s"})
            for i in range(100)
        ]
        history.write_text("\n".join(lines) + "\n")

        with patch.object(ClaudeCodeImporter, "HISTORY_PATH", history):
            messages = ClaudeCodeImporter.extract_messages(limit=5)

        assert len(messages) == 5

    def test_skips_malformed_json_lines(self, tmp_path):
        history = tmp_path / "history.jsonl"
        history.write_text(
            "this is not json\n"
            + json.dumps({"display": "valid message with sufficient length", "timestamp": 1, "project": "/test", "sessionId": "s"}) + "\n"
            + "{broken\n"
        )

        with patch.object(ClaudeCodeImporter, "HISTORY_PATH", history):
            messages = ClaudeCodeImporter.extract_messages()

        assert len(messages) == 1

    def test_skips_empty_lines(self, tmp_path):
        history = tmp_path / "history.jsonl"
        history.write_text(
            "\n\n"
            + json.dumps({"display": "valid message with enough length", "timestamp": 1, "project": "/test", "sessionId": "s"}) + "\n"
            + "\n"
        )

        with patch.object(ClaudeCodeImporter, "HISTORY_PATH", history):
            messages = ClaudeCodeImporter.extract_messages()

        assert len(messages) == 1


# ── Copilot Importer ────────────────────────────────────────────────────────


class TestCopilotImporter:
    def test_parses_events_jsonl(self, tmp_path):
        session_dir = tmp_path / "session-state" / "test-session-1"
        session_dir.mkdir(parents=True)
        (session_dir / "workspace.yaml").write_text("id: test-session-1\ncwd: /Users/test/project\n")

        events = [
            {"type": "session.start", "data": {"sessionId": "test-session-1"}},
            {"type": "user.message", "data": {"content": "sort these emails into categories for the team"}},
            {"type": "assistant.message", "data": {"content": "I'll categorize your emails into the following topics..."}},
            {"type": "user.message", "data": {"content": "now do the second batch"}},
            {"type": "assistant.message", "data": {"content": "Here are the categories for batch 2..."}},
        ]
        (session_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )

        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "session-state"):
            messages = CopilotImporter.extract_messages()

        assert len(messages) == 2
        assert messages[0]["task_input"] == "sort these emails into categories for the team"
        assert messages[0]["assistant_response"] == "I'll categorize your emails into the following topics..."
        assert messages[0]["source"] == "copilot"
        assert messages[0]["project"] == "/Users/test/project"

    def test_filters_secrets_from_copilot(self, tmp_path):
        session_dir = tmp_path / "session-state" / "test-session-2"
        session_dir.mkdir(parents=True)
        (session_dir / "workspace.yaml").write_text("id: test-2\ncwd: /test\n")

        events = [
            {"type": "user.message", "data": {"content": "here is my key sk-ant-api03-SECRET123456789012345678901234"}},
            {"type": "assistant.message", "data": {"content": "I see your API key"}},
        ]
        (session_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )

        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "session-state"):
            messages = CopilotImporter.extract_messages()

        assert len(messages) == 0

    def test_handles_missing_dir(self, tmp_path):
        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "nonexistent"):
            messages = CopilotImporter.extract_messages()
        assert messages == []

    def test_unpaired_user_message_dropped(self, tmp_path):
        """A user message with no following assistant response is dropped."""
        session_dir = tmp_path / "session-state" / "s1"
        session_dir.mkdir(parents=True)
        (session_dir / "workspace.yaml").write_text("cwd: /test\n")

        events = [
            {"type": "user.message", "data": {"content": "hello this is a long enough message"}},
            # No assistant response follows
        ]
        (session_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )

        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "session-state"):
            messages = CopilotImporter.extract_messages()

        assert len(messages) == 0

    def test_multiline_assistant_response(self, tmp_path):
        """Multiple assistant.message events concatenate into one response."""
        session_dir = tmp_path / "session-state" / "s1"
        session_dir.mkdir(parents=True)
        (session_dir / "workspace.yaml").write_text("cwd: /test\n")

        events = [
            {"type": "user.message", "data": {"content": "explain this code in detail please"}},
            {"type": "assistant.message", "data": {"content": "First, the function validates input."}},
            {"type": "assistant.message", "data": {"content": "Then it processes the data in chunks."}},
        ]
        (session_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )

        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "session-state"):
            messages = CopilotImporter.extract_messages()

        assert len(messages) == 1
        assert "\n" in messages[0]["assistant_response"]
        assert "First" in messages[0]["assistant_response"]
        assert "chunks" in messages[0]["assistant_response"]

    def test_empty_events_file(self, tmp_path):
        session_dir = tmp_path / "session-state" / "s1"
        session_dir.mkdir(parents=True)
        (session_dir / "workspace.yaml").write_text("cwd: /test\n")
        (session_dir / "events.jsonl").write_text("")

        with patch.object(CopilotImporter, "SESSION_DIR", tmp_path / "session-state"):
            messages = CopilotImporter.extract_messages()

        assert messages == []


class TestCopilotHelpers:
    def test_read_workspace_with_cwd(self, tmp_path):
        ws = tmp_path / "workspace.yaml"
        ws.write_text("id: s1\ncwd: /Users/dev/myproject\nother: stuff\n")
        assert _read_copilot_workspace(ws) == "/Users/dev/myproject"

    def test_read_workspace_missing_file(self, tmp_path):
        assert _read_copilot_workspace(tmp_path / "nope.yaml") == ""

    def test_read_workspace_no_cwd(self, tmp_path):
        ws = tmp_path / "workspace.yaml"
        ws.write_text("id: s1\nother: stuff\n")
        assert _read_copilot_workspace(ws) == ""

    def test_parse_copilot_events_malformed_json(self, tmp_path):
        events_path = tmp_path / "events.jsonl"
        events_path.write_text(
            "not json\n"
            + json.dumps({"type": "user.message", "data": {"content": "hello this is a message"}}) + "\n"
            + json.dumps({"type": "assistant.message", "data": {"content": "hi there"}}) + "\n"
        )
        pairs = _parse_copilot_events(events_path, "s1", "/test")
        assert len(pairs) == 1

    def test_parse_copilot_events_file_not_found(self, tmp_path):
        """Outer try-catch: file doesn't exist -> returns empty list, doesn't crash."""
        pairs = _parse_copilot_events(tmp_path / "nope.jsonl", "s1", "/test")
        assert pairs == []

    def test_parse_copilot_events_permission_error(self, tmp_path):
        """Outer try-catch: unreadable file -> returns empty list, doesn't crash."""
        events_path = tmp_path / "events.jsonl"
        events_path.write_text("data")
        events_path.chmod(0o000)
        try:
            pairs = _parse_copilot_events(events_path, "s1", "/test")
            assert pairs == []
        finally:
            events_path.chmod(0o644)  # Restore for cleanup


# ── Hermes Session Importer ────────────────────────────────────────────────


def _create_hermes_db(db_path, sessions, messages):
    """Helper to create a Hermes-compatible SQLite database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE sessions (id TEXT, title TEXT)")
    conn.execute(
        "CREATE TABLE messages "
        "(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp INTEGER)"
    )
    for s in sessions:
        conn.execute("INSERT INTO sessions VALUES (?, ?)", s)
    for m in messages:
        conn.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?)", m)
    conn.commit()
    conn.close()


class TestHermesSessionImporter:
    def test_parses_sqlite_pairs(self, tmp_path):
        db_path = tmp_path / "state.db"
        _create_hermes_db(db_path,
            sessions=[("s1", "my project")],
            messages=[
                (1, "s1", "user", "sort these messages into topics please", 1000),
                (2, "s1", "assistant", "I will group them by theme", 1001),
                (3, "s1", "user", "now do batch two with more detail", 1002),
                (4, "s1", "assistant", "Here are the categories for batch 2", 1003),
            ],
        )

        with patch.object(HermesSessionImporter, "DB_PATH", db_path):
            messages = HermesSessionImporter.extract_messages()

        assert len(messages) == 2
        assert messages[0]["source"] == "hermes"
        assert messages[0]["project"] == "my project"
        inputs = {m["task_input"] for m in messages}
        assert "sort these messages into topics please" in inputs
        assert "now do batch two with more detail" in inputs

    def test_filters_secrets(self, tmp_path):
        db_path = tmp_path / "state.db"
        _create_hermes_db(db_path,
            sessions=[("s1", "test")],
            messages=[
                (1, "s1", "user", "my key is sk-ant-api03-SECRET1234567890123456", 1000),
                (2, "s1", "assistant", "I see your key", 1001),
            ],
        )

        with patch.object(HermesSessionImporter, "DB_PATH", db_path):
            messages = HermesSessionImporter.extract_messages()

        assert len(messages) == 0

    def test_handles_missing_db(self, tmp_path):
        with patch.object(HermesSessionImporter, "DB_PATH", tmp_path / "nonexistent.db"):
            messages = HermesSessionImporter.extract_messages()
        assert messages == []

    def test_respects_limit(self, tmp_path):
        db_path = tmp_path / "state.db"
        sessions = [("s1", "test")]
        messages = []
        for i in range(50):
            messages.append((i * 2 + 1, "s1", "user", f"message number {i} with enough length to pass", i * 2))
            messages.append((i * 2 + 2, "s1", "assistant", f"response to message {i}", i * 2 + 1))
        _create_hermes_db(db_path, sessions=sessions, messages=messages)

        with patch.object(HermesSessionImporter, "DB_PATH", db_path):
            result = HermesSessionImporter.extract_messages(limit=5)

        assert len(result) == 5

    def test_handles_null_content(self, tmp_path):
        """NULL content fields should not crash."""
        db_path = tmp_path / "state.db"
        _create_hermes_db(db_path,
            sessions=[("s1", "test")],
            messages=[
                (1, "s1", "user", "a valid message with enough chars", 1000),
                (2, "s1", "assistant", None, 1001),
            ],
        )

        with patch.object(HermesSessionImporter, "DB_PATH", db_path):
            messages = HermesSessionImporter.extract_messages()

        # Pair exists but assistant content is None -> empty string, not a crash
        # The SQL JOIN should still find the row since it matches on role='assistant'
        # and the content filtering handles None via `or ""`
        assert isinstance(messages, list)

    def test_skips_short_user_messages(self, tmp_path):
        """Messages under 10 chars are filtered by the SQL query."""
        db_path = tmp_path / "state.db"
        _create_hermes_db(db_path,
            sessions=[("s1", "test")],
            messages=[
                (1, "s1", "user", "hi", 1000),
                (2, "s1", "assistant", "hello there", 1001),
            ],
        )

        with patch.object(HermesSessionImporter, "DB_PATH", db_path):
            messages = HermesSessionImporter.extract_messages()

        assert len(messages) == 0

    def test_wrong_schema_returns_empty(self, tmp_path):
        """DB with wrong table schema -> SQL error caught, returns [], doesn't crash."""
        db_path = tmp_path / "state.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE sessions (id TEXT, title TEXT)")
        # Wrong schema: missing 'role' and 'content' columns
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, data TEXT)")
        conn.commit()
        conn.close()

        with patch.object(HermesSessionImporter, "DB_PATH", db_path):
            messages = HermesSessionImporter.extract_messages()

        assert messages == []

    def test_connection_closed_on_error(self, tmp_path):
        """Verify conn.close() is called even when query fails (finally block)."""
        db_path = tmp_path / "state.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE sessions (id TEXT, title TEXT)")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, data TEXT)")
        conn.commit()
        conn.close()

        with patch.object(HermesSessionImporter, "DB_PATH", db_path):
            messages = HermesSessionImporter.extract_messages()
            # After the call, the DB should not be locked
            # Verify by opening it again successfully
            verify_conn = sqlite3.connect(str(db_path), timeout=1)
            verify_conn.execute("SELECT 1")
            verify_conn.close()

        assert messages == []


# ── RelevanceFilter (mocked DSPy) ───────────────────────────────────────────


class TestRelevanceFilter:
    """Test RelevanceFilter with mocked LLM calls."""

    @pytest.fixture
    def mock_dspy(self):
        """Mock dspy.LM and dspy.context to avoid real LLM calls."""
        with patch("evolution.core.external_importers.dspy") as mock:
            # Make dspy.context a no-op context manager
            mock.context.return_value.__enter__ = MagicMock(return_value=None)
            mock.context.return_value.__exit__ = MagicMock(return_value=False)
            yield mock

    def test_relevant_messages_become_examples(self, mock_dspy):
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        # Mock scorer to return relevant=True
        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": true, "expected_behavior": "group by topic", "difficulty": "easy", "category": "sorting"}'
        )

        messages = [
            {"task_input": "sort these messages by topic", "source": "claude-code"},
            {"task_input": "categorize my emails please", "source": "copilot", "assistant_response": "Sure!"},
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics. Categorize content.", max_examples=10)

        assert len(examples) == 2
        inputs = {ex.task_input for ex in examples}
        assert "sort these messages by topic" in inputs
        assert "categorize my emails please" in inputs
        # All examples should have the scoring metadata
        for ex in examples:
            assert ex.expected_behavior == "group by topic"
            assert ex.difficulty == "easy"

    def test_irrelevant_messages_filtered_out(self, mock_dspy):
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": false}'
        )

        messages = [
            {"task_input": "deploy the app to production", "source": "claude-code"},
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics.", max_examples=10)
        assert len(examples) == 0

    def test_malformed_llm_output_counted_as_error(self, mock_dspy):
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(scoring="I cannot determine relevance right now")

        messages = [
            {"task_input": "sort these messages by topic please", "source": "hermes"},
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics.", max_examples=10)
        assert len(examples) == 0

    def test_max_examples_cap_respected(self, mock_dspy):
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": true, "expected_behavior": "test", "difficulty": "easy", "category": "test"}'
        )

        messages = [
            {"task_input": f"categorize message number {i} into topics", "source": "claude-code"}
            for i in range(20)
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics. Categorize content.", max_examples=3)
        assert len(examples) == 3

    def test_scorer_exception_counted_as_error(self, mock_dspy):
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock(side_effect=RuntimeError("API timeout"))

        messages = [
            {"task_input": "sort these messages by topic please", "source": "claude-code"},
        ]

        # Should not raise — errors are caught and counted
        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics.", max_examples=10)
        assert len(examples) == 0


# ── build_dataset_from_external ──────────────────────────────────────────────


class TestBuildDataset:
    """Test the main orchestration function."""

    def test_builds_dataset_with_splits(self, tmp_path):
        """Verify end-to-end: import -> filter -> split -> save."""
        mock_messages = [
            {"task_input": f"categorize batch {i} into topics", "source": "claude-code"}
            for i in range(10)
        ]

        mock_examples = [
            EvalExample(
                task_input=f"categorize batch {i} into topics",
                expected_behavior="group by topic",
                difficulty="easy",
                category="sorting",
                source="claude-code",
            )
            for i in range(10)
        ]

        output = tmp_path / "output"

        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=mock_messages), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=mock_examples):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text into topics.",
                sources=["claude-code"],
                output_path=output,
                model="test-model",
                max_examples=10,
            )

        # Dataset has all three splits
        assert len(dataset.train) > 0
        assert len(dataset.val) > 0
        # total examples preserved
        assert len(dataset.all_examples) == 10

        # JSONL files written
        assert (output / "train.jsonl").exists()
        assert (output / "val.jsonl").exists()
        assert (output / "holdout.jsonl").exists()

    def test_no_messages_returns_empty_dataset(self, tmp_path):
        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=[]):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text into topics.",
                sources=["claude-code"],
                output_path=tmp_path / "out",
                model="test-model",
            )

        assert len(dataset.all_examples) == 0

    def test_no_relevant_examples_returns_empty_dataset(self, tmp_path):
        mock_messages = [{"task_input": "deploy the app", "source": "claude-code"}]

        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=mock_messages), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=[]):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text into topics.",
                sources=["claude-code"],
                output_path=tmp_path / "out",
                model="test-model",
            )

        assert len(dataset.all_examples) == 0

    def test_multiple_sources(self, tmp_path):
        cc_msgs = [{"task_input": "sort from claude code session", "source": "claude-code"}]
        cp_msgs = [{"task_input": "sort from copilot session", "source": "copilot", "assistant_response": "ok"}]

        all_examples = [
            EvalExample(task_input="sort from claude code session", expected_behavior="test", source="claude-code"),
            EvalExample(task_input="sort from copilot session", expected_behavior="test", source="copilot"),
        ]

        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=cc_msgs), \
             patch.object(CopilotImporter, "extract_messages", return_value=cp_msgs), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=all_examples):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text.",
                sources=["claude-code", "copilot"],
                output_path=tmp_path / "out",
                model="test-model",
            )

        sources = {ex.source for ex in dataset.all_examples}
        assert "claude-code" in sources
        assert "copilot" in sources

    def test_unknown_source_ignored(self, tmp_path):
        """An unrecognized source name is silently skipped."""
        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=[]):
            dataset = build_dataset_from_external(
                skill_name="test",
                skill_text="Test.",
                sources=["claude-code", "nonexistent-tool"],
                output_path=tmp_path / "out",
                model="test-model",
            )

        assert len(dataset.all_examples) == 0


# ── End-to-End Roundtrip ─────────────────────────────────────────────────────


class TestEndToEndRoundtrip:
    """Verify the full pipeline: fake files -> import -> filter -> save -> reload.

    This is the most important test. It proves that output from
    build_dataset_from_external is loadable by GoldenDatasetLoader,
    which is the actual consumer in evolve_skill.py.
    """

    def test_output_loadable_by_golden_loader(self, tmp_path):
        """Write fake Claude Code history, run pipeline, load with GoldenDatasetLoader."""
        from evolution.core.dataset_builder import GoldenDatasetLoader

        # Create fake Claude Code history
        history = tmp_path / "history.jsonl"
        lines = [
            json.dumps({"display": f"categorize these {i} messages into topics", "timestamp": i, "project": "/test", "sessionId": "s1"})
            for i in range(20)
        ]
        history.write_text("\n".join(lines) + "\n")

        # Mock examples that RelevanceFilter would produce
        mock_examples = [
            EvalExample(
                task_input=f"categorize these {i} messages into topics",
                expected_behavior="group by theme",
                difficulty="easy",
                category="sorting",
                source="claude-code",
            )
            for i in range(8)
        ]

        output = tmp_path / "dataset"

        with patch.object(ClaudeCodeImporter, "HISTORY_PATH", history), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=mock_examples):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text into topics.",
                sources=["claude-code"],
                output_path=output,
                model="test-model",
            )

        # Verify files exist
        assert (output / "train.jsonl").exists()
        assert (output / "val.jsonl").exists()
        assert (output / "holdout.jsonl").exists()

        # Verify GoldenDatasetLoader can read them back
        reloaded = GoldenDatasetLoader.load(output)
        assert len(reloaded.all_examples) == len(dataset.all_examples)

        # Verify content survived the roundtrip
        for ex in reloaded.all_examples:
            assert ex.task_input.startswith("categorize these")
            assert ex.expected_behavior == "group by theme"
            assert ex.source == "claude-code"

    def test_full_pipeline_with_real_sqlite(self, tmp_path):
        """Create real SQLite DB, import, filter (mocked), save, reload."""
        from evolution.core.dataset_builder import EvalDataset

        db_path = tmp_path / "state.db"
        _create_hermes_db(db_path,
            sessions=[("s1", "test project")],
            messages=[
                (1, "s1", "user", "sort these messages into categories for me", 1000),
                (2, "s1", "assistant", "I grouped them into 3 categories", 1001),
                (3, "s1", "user", "now categorize the second batch of emails", 1002),
                (4, "s1", "assistant", "Here are the email categories", 1003),
            ],
        )

        mock_examples = [
            EvalExample(
                task_input="sort these messages into categories for me",
                expected_behavior="group into categories",
                difficulty="easy",
                category="sorting",
                source="hermes",
            ),
        ]

        output = tmp_path / "dataset"

        with patch.object(HermesSessionImporter, "DB_PATH", db_path), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=mock_examples):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text.",
                sources=["hermes"],
                output_path=output,
                model="test-model",
            )

        assert len(dataset.all_examples) == 1

        # Reload and verify
        reloaded = EvalDataset.load(output)
        assert len(reloaded.all_examples) == 1
        assert reloaded.all_examples[0].task_input == "sort these messages into categories for me"


# ── _load_skill_text ─────────────────────────────────────────────────────────


class TestLoadSkillText:
    def test_loads_skill_from_directory(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nDo stuff.")

        name, text = _load_skill_text("my-skill", skills_dir=tmp_path)
        assert name == "my-skill"
        assert "Do stuff." in text

    def test_loads_skill_from_subdirectory(self, tmp_path):
        sub = tmp_path / "custom" / "my-skill"
        sub.mkdir(parents=True)
        (sub / "SKILL.md").write_text("Nested skill content.")

        name, text = _load_skill_text("my-skill", skills_dir=tmp_path)
        assert name == "my-skill"
        assert "Nested skill content." in text

    def test_raises_on_missing_skill(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            _load_skill_text("nonexistent-skill", skills_dir=tmp_path)

    def test_ignores_dir_without_skill_md(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "README.md").write_text("Not a SKILL.md file")

        with pytest.raises(FileNotFoundError):
            _load_skill_text("my-skill", skills_dir=tmp_path)


# ── Validation ───────────────────────────────────────────────────────────────


class TestValidateEvalExample:
    """Verify _validate_eval_example normalizes, rejects, and caps fields."""

    def test_valid_input_passes_through(self):
        result = _validate_eval_example(
            task_input="sort these messages by topic",
            expected_behavior="group by theme",
            difficulty="easy",
            category="sorting",
        )
        assert result is not None
        assert result["task_input"] == "sort these messages by topic"
        assert result["expected_behavior"] == "group by theme"
        assert result["difficulty"] == "easy"
        assert result["category"] == "sorting"

    def test_empty_task_input_returns_none(self):
        assert _validate_eval_example("", "behavior", "easy", "cat") is None

    def test_whitespace_task_input_returns_none(self):
        assert _validate_eval_example("   ", "behavior", "easy", "cat") is None

    def test_none_task_input_returns_none(self):
        assert _validate_eval_example(None, "behavior", "easy", "cat") is None

    def test_empty_expected_behavior_returns_none(self):
        assert _validate_eval_example("task", "", "easy", "cat") is None

    def test_whitespace_expected_behavior_returns_none(self):
        assert _validate_eval_example("task", "  \n  ", "easy", "cat") is None

    def test_none_expected_behavior_returns_none(self):
        assert _validate_eval_example("task", None, "easy", "cat") is None

    def test_invalid_difficulty_defaults_to_medium(self):
        result = _validate_eval_example("task", "behavior", "impossible", "cat")
        assert result["difficulty"] == "medium"

    def test_empty_difficulty_defaults_to_medium(self):
        result = _validate_eval_example("task", "behavior", "", "cat")
        assert result["difficulty"] == "medium"

    def test_none_difficulty_defaults_to_medium(self):
        result = _validate_eval_example("task", "behavior", None, "cat")
        assert result["difficulty"] == "medium"

    def test_difficulty_case_insensitive(self):
        result = _validate_eval_example("task", "behavior", "HARD", "cat")
        assert result["difficulty"] == "hard"

    def test_difficulty_stripped(self):
        result = _validate_eval_example("task", "behavior", "  easy  ", "cat")
        assert result["difficulty"] == "easy"

    def test_all_valid_difficulties_accepted(self):
        for diff in VALID_DIFFICULTIES:
            result = _validate_eval_example("task", "behavior", diff, "cat")
            assert result["difficulty"] == diff

    def test_empty_category_defaults_to_general(self):
        result = _validate_eval_example("task", "behavior", "easy", "")
        assert result["category"] == "general"

    def test_none_category_defaults_to_general(self):
        result = _validate_eval_example("task", "behavior", "easy", None)
        assert result["category"] == "general"

    def test_whitespace_category_defaults_to_general(self):
        result = _validate_eval_example("task", "behavior", "easy", "   ")
        assert result["category"] == "general"

    def test_task_input_capped_at_2000_chars(self):
        long_input = "x" * 5000
        result = _validate_eval_example(long_input, "behavior", "easy", "cat")
        assert result is not None
        assert len(result["task_input"]) == 2000

    def test_expected_behavior_stripped(self):
        result = _validate_eval_example("task", "  behavior with spaces  ", "easy", "cat")
        assert result["expected_behavior"] == "behavior with spaces"

    def test_category_stripped(self):
        result = _validate_eval_example("task", "behavior", "easy", "  sorting  ")
        assert result["category"] == "sorting"


class TestValidationIntegration:
    """Verify validation is wired correctly into RelevanceFilter."""

    @pytest.fixture
    def mock_dspy(self):
        with patch("evolution.core.external_importers.dspy") as mock:
            mock.context.return_value.__enter__ = MagicMock(return_value=None)
            mock.context.return_value.__exit__ = MagicMock(return_value=False)
            yield mock

    def test_empty_expected_behavior_drops_example(self, mock_dspy):
        """LLM returns relevant=True but empty expected_behavior -> example dropped."""
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": true, "expected_behavior": "", "difficulty": "easy", "category": "sorting"}'
        )

        messages = [
            {"task_input": "sort these messages by topic", "source": "claude-code"},
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics. Categorize content.", max_examples=10)
        assert len(examples) == 0

    def test_invalid_difficulty_normalized(self, mock_dspy):
        """LLM returns invalid difficulty -> normalized to medium."""
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": true, "expected_behavior": "group", "difficulty": "ultra-hard", "category": "sorting"}'
        )

        messages = [
            {"task_input": "sort these messages by topic", "source": "claude-code"},
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics. Categorize content.", max_examples=10)
        assert len(examples) == 1
        assert examples[0].difficulty == "medium"

    def test_missing_source_field_drops_message(self, mock_dspy):
        """Messages missing 'source' key are dropped before scoring."""
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()
        rf.scorer.return_value = SimpleNamespace(
            scoring='{"relevant": true, "expected_behavior": "test", "difficulty": "easy", "category": "test"}'
        )

        messages = [
            {"task_input": "sort these messages by topic"},  # missing source
            {"task_input": "categorize emails", "source": "claude-code"},
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics. Categorize content.", max_examples=10)
        assert len(examples) == 1
        assert examples[0].source == "claude-code"

    def test_missing_task_input_drops_message(self, mock_dspy):
        """Messages missing 'task_input' are dropped before scoring."""
        rf = RelevanceFilter.__new__(RelevanceFilter)
        rf.model = "test-model"

        rf.scorer = MagicMock()

        messages = [
            {"source": "claude-code"},  # missing task_input
        ]

        examples = rf.filter_and_score(messages, "categorize", "Sort text into topics.", max_examples=10)
        assert len(examples) == 0
        # scorer should never be called for invalid messages
        rf.scorer.assert_not_called()


class TestMinDatasetSizeWarning:
    """Verify MIN_DATASET_SIZE warning in build_dataset_from_external."""

    def test_small_dataset_still_returned(self, tmp_path):
        """Even with < MIN_DATASET_SIZE examples, a dataset is still returned."""
        mock_messages = [{"task_input": "sort stuff", "source": "claude-code"}]
        mock_examples = [
            EvalExample(
                task_input="sort stuff",
                expected_behavior="group by topic",
                difficulty="easy",
                category="sorting",
                source="claude-code",
            ),
        ]

        output = tmp_path / "output"

        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=mock_messages), \
             patch.object(RelevanceFilter, "filter_and_score", return_value=mock_examples):
            dataset = build_dataset_from_external(
                skill_name="categorize",
                skill_text="Sort text.",
                sources=["claude-code"],
                output_path=output,
                model="test-model",
            )

        # Still returns the dataset even though size < MIN_DATASET_SIZE
        assert len(dataset.all_examples) == 1
        assert MIN_DATASET_SIZE > 1  # Confirm the constant is meaningful


# ── CLI (CliRunner) ──────────────────────────────────────────────────────────


class TestCLI:
    """Test the Click CLI entry point using CliRunner."""

    def test_dry_run(self, tmp_path):
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\n---\nTest skill.")

        with patch.object(ClaudeCodeImporter, "extract_messages", return_value=[
            {"task_input": "hello with enough length", "source": "claude-code"},
        ]):
            runner = CliRunner()
            result = runner.invoke(main, [
                "--skill", "test-skill",
                "--source", "claude-code",
                "--dry-run",
            ], catch_exceptions=False, env={"HOME": str(tmp_path.parent)})

            # _load_skill_text uses ~/.hermes/skills by default, so we patch it
        # Instead, use the skills_dir parameter approach
        with patch("evolution.core.external_importers._load_skill_text", return_value=("test-skill", "Test skill.")), \
             patch.object(ClaudeCodeImporter, "extract_messages", return_value=[
                 {"task_input": "hello with enough length", "source": "claude-code"},
             ]):
            runner = CliRunner()
            result = runner.invoke(main, [
                "--skill", "test-skill",
                "--source", "claude-code",
                "--dry-run",
            ], catch_exceptions=False)

        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_missing_skill_exits_with_error(self):
        with patch("evolution.core.external_importers._load_skill_text", side_effect=FileNotFoundError("not found")):
            runner = CliRunner()
            result = runner.invoke(main, ["--skill", "nonexistent"])

        assert result.exit_code != 0

    def test_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Import external session data" in result.output


# ── EvalExample Format ───────────────────────────────────────────────────────


class TestEvalExampleFormat:
    def test_roundtrip_serialization(self):
        ex = EvalExample(
            task_input="sort messages by topic",
            expected_behavior="group by theme, list topic names",
            difficulty="medium",
            category="categorize",
            source="copilot",
        )
        d = ex.to_dict()
        restored = EvalExample.from_dict(d)
        assert restored.task_input == ex.task_input
        assert restored.expected_behavior == ex.expected_behavior
        assert restored.source == "copilot"

    def test_golden_jsonl_format(self, tmp_path):
        """Verify output matches GoldenDatasetLoader expected format."""
        ex = EvalExample(
            task_input="categorize these notes",
            expected_behavior="detect topics, group by theme",
            difficulty="easy",
            category="sorting",
            source="claude-code",
        )
        jsonl_path = tmp_path / "golden.jsonl"
        jsonl_path.write_text(json.dumps(ex.to_dict()) + "\n")

        with open(jsonl_path) as f:
            data = json.loads(f.readline())

        assert set(data.keys()) == {"task_input", "expected_behavior", "difficulty", "category", "source"}

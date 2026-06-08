"""Tests for evolution.validation.hermes_runner.

Two layers exercised:
  - The parse layer (parse_session_result) via hand-crafted fixture
    session JSONs covering both tool_call shapes Hermes emits. This is
    where bugs hide; mocked subprocess gives no signal here.
  - The subprocess invocation layer (HermesAgentRunner.run) via
    subprocess.run mocked to assert env + cwd + args.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from evolution.validation.agent_runner import TaskRunContext
from evolution.validation.hermes_runner import (
    HermesAgentRunner,
    _price_from_tokens,
    _strip_litellm_provider_prefix,
    parse_session_from_db,
    parse_session_result,
)


def _make_state_db(path: Path, *, session_id: str, model: str, messages: list[dict],
                   started_at: float = 1.0) -> None:
    """Create a minimal hermes-shaped state.db with one session + messages.

    Each ``messages`` entry: ``{"role", "content"?, "tool_calls"?}`` where
    ``tool_calls`` is a Python list serialized to the ``tool_calls`` TEXT
    column (the OpenAI-nested shape hermes stores).
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, model TEXT, started_at REAL);
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT, content TEXT, tool_calls TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO sessions (id, model, started_at) VALUES (?, ?, ?)",
        (session_id, model, started_at),
    )
    for m in messages:
        tc = m.get("tool_calls")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls) "
            "VALUES (?, ?, ?, ?)",
            (session_id, m["role"], m.get("content"),
             json.dumps(tc) if tc is not None else None),
        )
    conn.commit()
    conn.close()


class TestStripLitellmProviderPrefix:
    """The hermes -m flag interprets '<word>/<model>' as openrouter-style
    routing. Direct-provider users naturally pass litellm-formatted names
    like 'openai/gpt-4o-mini' from elsewhere in the framework; this helper
    normalizes them back to bare model names that hermes -m accepts."""

    def test_strips_openai_prefix(self):
        assert _strip_litellm_provider_prefix("openai/gpt-4o-mini") == "gpt-4o-mini"

    def test_strips_anthropic_prefix(self):
        assert _strip_litellm_provider_prefix("anthropic/claude-opus-4-7") == "claude-opus-4-7"

    def test_strips_azure_prefix(self):
        assert _strip_litellm_provider_prefix("azure/gpt-4") == "gpt-4"

    def test_strips_gemini_prefix(self):
        assert _strip_litellm_provider_prefix("gemini/gemini-1.5-pro") == "gemini-1.5-pro"

    def test_leaves_bare_model_unchanged(self):
        assert _strip_litellm_provider_prefix("gpt-4o-mini") == "gpt-4o-mini"

    def test_leaves_unknown_prefix_unchanged(self):
        # Unknown providers pass through so openrouter-style routing through
        # an unrecognized vendor still works.
        assert _strip_litellm_provider_prefix("custom-vendor/model-x") == "custom-vendor/model-x"

    def test_only_strips_first_prefix(self):
        # Defensive: nested prefixes (e.g. openrouter passthrough) shouldn't be
        # double-stripped — keep the second segment intact.
        assert _strip_litellm_provider_prefix("openai/foo/bar") == "foo/bar"


def _write_session(path: Path, messages: list[dict], **extra) -> None:
    payload = {
        "session_id": "test",
        "model": "claude-opus-4-7",
        "messages": messages,
        **extra,
    }
    path.write_text(json.dumps(payload))


class TestParseSessionResult:
    def test_extracts_openai_nested_tool_calls(self, tmp_path):
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "do thing"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "search_files", "arguments": "{}"}}
            ]},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["search_files"]
        assert result.error is None
        assert result.model_name == "claude-opus-4-7"

    def test_extracts_flat_tool_calls(self, tmp_path):
        # Hermes also emits this shape for some model providers.
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "do thing"},
            {"role": "assistant", "tool_calls": [
                {"name": "patch", "arguments": "{}"}
            ]},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["patch"]

    def test_extracts_multi_tool_calls_in_order(self, tmp_path):
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "do things"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "read_file"}},
                {"function": {"name": "patch"}},
                {"function": {"name": "search_files"}},
            ]},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["read_file", "patch", "search_files"]

    def test_extracts_across_multiple_assistant_turns(self, tmp_path):
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "first"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "patch"}}]},
            {"role": "tool", "content": "applied"},
            {"role": "assistant", "content": "done"},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["read_file", "patch"]

    def test_handles_assistant_message_with_text_only(self, tmp_path):
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "say hi"},
            {"role": "assistant", "content": "hi there"},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == []
        assert result.final_text_tail == "hi there"

    def test_final_text_tail_truncates_to_4096_chars(self, tmp_path):
        long_text = "x" * 5000
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "long"},
            {"role": "assistant", "content": long_text},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert len(result.final_text_tail) == 4096
        assert result.final_text_tail == long_text[-4096:]

    def test_picks_last_assistant_text_not_first(self, tmp_path):
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "first text"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "patch"}}]},
            {"role": "assistant", "content": "final text"},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.final_text_tail == "final text"

    def test_malformed_session_yields_error(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text("{not valid")
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.error is not None
        assert "could not parse" in result.error

    def test_call_missing_name_skipped(self, tmp_path):
        # Defensive: a tool_call dict without function.name or name shouldn't crash.
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "u"},
            {"role": "assistant", "tool_calls": [
                {"id": "abc"},                                      # no name field
                {"function": {"name": "patch"}},                    # has name
                {"function": {}},                                   # empty function
                {"name": ""},                                       # empty name
            ]},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["patch"]


class TestParseToolCallArgs:
    def test_captures_tool_call_args(self, tmp_path):
        """Sessions with tool calls must surface name AND parsed args."""
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "save a fact"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {
                    "name": "memory",
                    "arguments": json.dumps({
                        "action": "save",
                        "content": "user prefers terse responses",
                    }),
                }}
            ]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "Saved."},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["memory"]
        assert len(result.tool_calls_with_args) == 1
        call = result.tool_calls_with_args[0]
        assert call["name"] == "memory"
        assert call["arguments"]["action"] == "save"
        assert call["arguments"]["content"] == "user prefers terse responses"

    def test_handles_malformed_args(self, tmp_path):
        """Malformed tool-call arguments JSON must not crash — fall back to {}."""
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "memory", "arguments": "{not-json"}}
            ]},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["memory"]
        assert result.tool_calls_with_args == [{"name": "memory", "arguments": {}}]

    def test_handles_flat_dict_args(self, tmp_path):
        """Flat tool_call shape with an already-parsed dict argument."""
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "assistant", "tool_calls": [
                {"name": "memory", "arguments": {"action": "delete", "key": "x"}}
            ]},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_with_args == [
            {"name": "memory", "arguments": {"action": "delete", "key": "x"}}
        ]


class TestParseSessionFromDb:
    """The state.db parse layer — modern hermes persists sessions to SQLite."""

    def test_extracts_tool_calls_and_args(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db(db, session_id="s1", model="gpt-5.4-mini", messages=[
            {"role": "user", "content": "remember I use uv"},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {
                    "name": "memory",
                    "arguments": json.dumps({"action": "add", "content": "uses uv"}),
                }}
            ]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "Saved."},
        ])
        result = parse_session_from_db(db, duration_seconds=2.0)
        assert result.error is None
        assert result.model_name == "gpt-5.4-mini"
        assert result.tool_calls_seq == ["memory"]
        assert result.tool_calls_with_args == [
            {"name": "memory", "arguments": {"action": "add", "content": "uses uv"}}
        ]
        assert result.final_text_tail == "Saved."

    def test_no_sessions_is_error(self, tmp_path):
        db = tmp_path / "state.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TABLE sessions (id TEXT, model TEXT, started_at REAL);"
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
            "role TEXT, content TEXT, tool_calls TEXT);"
        )
        conn.commit()
        conn.close()
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.error is not None
        assert "no sessions" in result.error

    def test_picks_most_recent_session(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db(db, session_id="old", model="m", started_at=1.0, messages=[
            {"role": "assistant", "tool_calls": [{"function": {"name": "patch"}}]},
        ])
        # Add a newer session with a different tool call.
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO sessions (id, model, started_at) VALUES (?,?,?)",
                     ("new", "m", 2.0))
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls) VALUES (?,?,?,?)",
            ("new", "assistant", None,
             json.dumps([{"function": {"name": "write_file"}}])),
        )
        conn.commit()
        conn.close()
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.tool_calls_seq == ["write_file"]

    def test_malformed_tool_calls_column_abstains(self, tmp_path):
        """A corrupt tool_calls column must abstain (error set), not read as
        'agent invoked no tools' (which would score a hard behavioral fail)."""
        db = tmp_path / "state.db"
        _make_state_db(db, session_id="s1", model="m",
                       messages=[{"role": "user", "content": "hi"}])
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls) VALUES (?,?,?,?)",
            ("s1", "assistant", "", "{not-valid-json"),
        )
        conn.commit()
        conn.close()
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.error is not None
        assert "malformed tool_calls" in result.error
        assert result.tool_calls_seq == []

    def test_corrupt_db_file_errors(self, tmp_path):
        bad = tmp_path / "state.db"
        bad.write_bytes(b"this is not a sqlite database at all")
        result = parse_session_from_db(bad, duration_seconds=1.0)
        assert result.error is not None
        assert "could not" in result.error  # open or read, depending on sqlite

    def test_missing_messages_table_errors(self, tmp_path):
        db = tmp_path / "state.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TABLE sessions (id TEXT, model TEXT, started_at REAL);"
            "INSERT INTO sessions VALUES ('s1', 'm', 1.0);"
        )
        conn.commit()
        conn.close()
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.error is not None
        assert "could not read" in result.error


def _make_state_db_with_cost(
    path: Path,
    *,
    session_id: str,
    model: str,
    messages: list[dict],
    started_at: float = 1.0,
    actual_cost_usd: float | None = None,
    estimated_cost_usd: float | None = None,
    cost_status: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> None:
    """Minimal hermes state.db with cost/token columns in the sessions table."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            model TEXT,
            started_at REAL,
            actual_cost_usd REAL,
            estimated_cost_usd REAL,
            cost_status TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_read_tokens INTEGER,
            cache_write_tokens INTEGER,
            reasoning_tokens INTEGER
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT, content TEXT, tool_calls TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO sessions (id, model, started_at, actual_cost_usd, "
        "estimated_cost_usd, cost_status, input_tokens, output_tokens, "
        "cache_read_tokens, cache_write_tokens, reasoning_tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, model, started_at, actual_cost_usd, estimated_cost_usd,
         cost_status, input_tokens, output_tokens, cache_read_tokens,
         cache_write_tokens, reasoning_tokens),
    )
    for m in messages:
        tc = m.get("tool_calls")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls) "
            "VALUES (?, ?, ?, ?)",
            (session_id, m["role"], m.get("content"),
             json.dumps(tc) if tc is not None else None),
        )
    conn.commit()
    conn.close()


class TestAgentRunResultDefaults:
    """New cost fields must be backward-compatible — existing construction works."""

    def test_existing_construction_unaffected(self):
        from evolution.validation.agent_runner import AgentRunResult
        result = AgentRunResult(
            tool_calls_seq=["patch"],
            final_text_tail="done",
            duration_seconds=1.5,
        )
        assert result.agent_cost_usd is None
        assert result.agent_cost_source == "uncaptured"
        assert result.agent_tokens == {}

    def test_new_fields_accept_values(self):
        from evolution.validation.agent_runner import AgentRunResult
        result = AgentRunResult(
            tool_calls_seq=[],
            final_text_tail="",
            duration_seconds=0.0,
            agent_cost_usd=0.012,
            agent_cost_source="actual",
            agent_tokens={"input": 100},
        )
        assert result.agent_cost_usd == 0.012
        assert result.agent_cost_source == "actual"
        assert result.agent_tokens == {"input": 100}


class TestParseSessionFromDbCostCapture:
    """Cost and token columns surface from sessions rows."""

    def test_actual_cost_usd_populated(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="gpt-5.4-mini",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=0.012,
            estimated_cost_usd=0.008,
            cost_status="settled",
            input_tokens=100, output_tokens=50,
            cache_read_tokens=20, cache_write_tokens=10, reasoning_tokens=5,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.error is None
        assert result.agent_cost_usd == pytest.approx(0.012)
        assert result.agent_cost_source == "actual"
        assert result.agent_tokens["input_tokens"] == 100
        assert result.agent_tokens["output_tokens"] == 50
        assert result.agent_tokens["cache_read_tokens"] == 20
        assert result.agent_tokens["cache_write_tokens"] == 10
        assert result.agent_tokens["reasoning_tokens"] == 5

    def test_estimated_cost_used_when_actual_null(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="m",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=None,
            estimated_cost_usd=0.008,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.error is None
        assert result.agent_cost_usd == pytest.approx(0.008)
        assert result.agent_cost_source == "estimated"

    def test_zero_cost_with_tokens_flagged_uncaptured(self, tmp_path):
        # An unpriced model: hermes reports $0 alongside real token usage.
        # Must NOT be trusted as a free run — falls through to computed or uncaptured.
        # "m" is not priceable by litellm, so it ends up uncaptured.
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="m",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=None, estimated_cost_usd=0.0,
            input_tokens=10645, output_tokens=217,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.agent_cost_usd is None
        assert result.agent_cost_source == "uncaptured"

    def test_zero_cost_with_zero_tokens_is_genuine_zero(self, tmp_path):
        # A run that burned no tokens genuinely cost ~0 — not a pricing gap.
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="m",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=0.0, estimated_cost_usd=None,
            input_tokens=0, output_tokens=0,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.agent_cost_usd == pytest.approx(0.0)
        assert result.agent_cost_source == "actual"

    def test_both_null_yields_uncaptured(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="m",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=None,
            estimated_cost_usd=None,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.error is None
        assert result.agent_cost_usd is None
        assert result.agent_cost_source == "uncaptured"

    def test_estimated_cost_used_when_actual_null_unpriceable_model(self, tmp_path):
        # "m" is not a model litellm can price, so computed returns None.
        # estimated should be used as the final fallback instead.
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="m",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=None,
            estimated_cost_usd=0.008,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.error is None
        assert result.agent_cost_usd == pytest.approx(0.008)
        assert result.agent_cost_source == "estimated"

    def test_schema_drift_tolerance_old_sessions_table(self, tmp_path):
        """A DB with only id/model/started_at in sessions must not crash or abstain.

        Schema-drift happens when the harness runs against an old hermes build
        that predates the cost columns. The extended SELECT falls back to the
        id/model-only SELECT; cost fields report as uncaptured so the run still
        contributes behavioral signal.
        """
        db = tmp_path / "state.db"
        # Minimal schema — no cost or token columns.
        _make_state_db(
            db, session_id="s1", model="old-model",
            messages=[{"role": "assistant", "tool_calls": [
                {"function": {"name": "patch"}}
            ]}],
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        # Must not set error or abstain.
        assert result.error is None
        assert result.tool_calls_seq == ["patch"]
        assert result.agent_cost_usd is None
        assert result.agent_cost_source == "uncaptured"
        assert result.agent_tokens == {}


class TestPriceFromTokens:
    """FIX 2: a recognized-but-unpriced model returns litellm $0, which must be
    treated as a pricing gap (None), not a trusted free run."""

    def test_litellm_zero_price_returns_none(self):
        with patch(
            "evolution.validation.hermes_runner.litellm.cost_per_token",
            return_value=(0.0, 0.0),
        ):
            assert _price_from_tokens("some-model", {"input_tokens": 100, "output_tokens": 50}) is None

    def test_litellm_positive_price_returned(self):
        with patch(
            "evolution.validation.hermes_runner.litellm.cost_per_token",
            return_value=(0.001, 0.002),
        ):
            assert _price_from_tokens(
                "some-model", {"input_tokens": 100, "output_tokens": 50}
            ) == pytest.approx(0.003)

    def test_full_parse_with_litellm_zero_price_is_uncaptured(self, tmp_path):
        """A priceable-shaped model that litellm prices at $0 (monkeypatched)
        must resolve to uncaptured, not computed $0."""
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="gpt-5.4-mini",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=None, estimated_cost_usd=None,
            input_tokens=1000, output_tokens=100,
        )
        with patch(
            "evolution.validation.hermes_runner.litellm.cost_per_token",
            return_value=(0.0, 0.0),
        ):
            result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.agent_cost_source == "uncaptured"
        assert result.agent_cost_usd is None


class TestComputedCostFallback:
    """litellm-priced fallback when hermes reports $0 or NULL cost but tokens are present."""

    def test_computed_when_zero_cost_with_tokens(self, tmp_path):
        # hermes reports $0 estimated (unpriced model) + real tokens → computed via litellm.
        import litellm
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="gpt-5.4-mini",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=None, estimated_cost_usd=0.0,
            input_tokens=10645, output_tokens=217,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        pin, pout = litellm.cost_per_token(
            model="gpt-5.4-mini", prompt_tokens=10645, completion_tokens=217
        )
        assert result.agent_cost_source == "computed"
        assert result.agent_cost_usd == pytest.approx(pin + pout, rel=1e-4)
        assert result.agent_cost_usd > 0.005

    def test_computed_when_both_null_but_tokens_and_priceable_model(self, tmp_path):
        # Both hermes cost columns NULL, but tokens present and model is priceable.
        import litellm
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="gpt-5.4-mini",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=None, estimated_cost_usd=None,
            input_tokens=5000, output_tokens=100,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        pin, pout = litellm.cost_per_token(
            model="gpt-5.4-mini", prompt_tokens=5000, completion_tokens=100
        )
        assert result.agent_cost_source == "computed"
        assert result.agent_cost_usd == pytest.approx(pin + pout, rel=1e-4)
        assert result.agent_cost_usd > 0

    def test_actual_wins_over_computed(self, tmp_path):
        # actual_cost_usd > 0 must be used without consulting litellm.
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="gpt-5.4-mini",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=0.02, estimated_cost_usd=None,
            input_tokens=10645, output_tokens=217,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.agent_cost_source == "actual"
        assert result.agent_cost_usd == pytest.approx(0.02)

    def test_unpriceable_model_falls_through_to_uncaptured(self, tmp_path):
        # litellm raises for unknown model → computed returns None → uncaptured.
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="totally-unknown-model-xyz",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=None, estimated_cost_usd=None,
            input_tokens=500, output_tokens=50,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.agent_cost_source == "uncaptured"
        assert result.agent_cost_usd is None

    def test_no_tokens_not_computed(self, tmp_path):
        # Both hermes costs NULL and tokens are 0 → uncaptured, not computed.
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="gpt-5.4-mini",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=None, estimated_cost_usd=None,
            input_tokens=0, output_tokens=0,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.agent_cost_source == "uncaptured"
        assert result.agent_cost_usd is None

    def test_computed_wins_over_estimated(self, tmp_path):
        # FIX 7: with actual NULL but a priceable model + real tokens, the
        # litellm-computed value must beat a non-null estimated.
        import litellm
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="gpt-5.4-mini",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=None, estimated_cost_usd=0.008,
            input_tokens=5000, output_tokens=100,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        pin, pout = litellm.cost_per_token(
            model="gpt-5.4-mini", prompt_tokens=5000, completion_tokens=100
        )
        assert result.agent_cost_source == "computed"
        assert result.agent_cost_usd == pytest.approx(pin + pout, rel=1e-4)
        assert result.agent_cost_usd != pytest.approx(0.008)

    def test_actual_zero_with_tokens_falls_through_to_computed(self, tmp_path):
        # FIX 7: a $0 actual alongside real token usage is distrusted (unpriced
        # at capture time), so it falls through to litellm-computed.
        import litellm
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="gpt-5.4-mini",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=0.0, estimated_cost_usd=None,
            input_tokens=5000, output_tokens=100,
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        pin, pout = litellm.cost_per_token(
            model="gpt-5.4-mini", prompt_tokens=5000, completion_tokens=100
        )
        assert result.agent_cost_source == "computed"
        assert result.agent_cost_usd == pytest.approx(pin + pout, rel=1e-4)


class TestSchemaDriftExceptNarrowing:
    """FIX 5: only a genuine missing-column OperationalError is treated as schema
    drift; a locked/corrupt-DB OperationalError must re-raise and abstain."""

    def test_locked_db_on_extended_select_abstains_not_uncaptured(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db_with_cost(
            db, session_id="s1", model="m",
            messages=[{"role": "assistant", "content": "ok"}],
            actual_cost_usd=0.01,
        )

        class _LockedConn:
            """Wraps a real connection but raises 'database is locked' on the
            extended cost SELECT — sqlite3.Connection itself is immutable."""

            def __init__(self, real):
                self._real = real

            def execute(self, sql, *args):
                if "actual_cost_usd" in sql:
                    raise sqlite3.OperationalError("database is locked")
                return self._real.execute(sql, *args)

            def __setattr__(self, name, value):
                if name == "_real":
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._real, name, value)

            def __getattr__(self, name):
                return getattr(self._real, name)

        real_connect = sqlite3.connect

        def _fake_connect(*args, **kwargs):
            return _LockedConn(real_connect(*args, **kwargs))

        with patch(
            "evolution.validation.hermes_runner.sqlite3.connect",
            side_effect=_fake_connect,
        ):
            result = parse_session_from_db(db, duration_seconds=1.0)

        # The locked error re-raises into the outer sqlite3.Error handler →
        # proper abstain (error set), NOT a silent uncaptured success.
        assert result.error is not None
        assert result.agent_cost_source == "uncaptured"
        assert "could not read" in result.error

    def test_missing_column_still_falls_back(self, tmp_path):
        # Regression: the genuine schema-drift path (missing column) still works.
        db = tmp_path / "state.db"
        _make_state_db(
            db, session_id="s1", model="old-model",
            messages=[{"role": "assistant", "tool_calls": [
                {"function": {"name": "patch"}}
            ]}],
        )
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.error is None
        assert result.tool_calls_seq == ["patch"]
        assert result.agent_cost_source == "uncaptured"


class TestHermesAgentRunnerSubprocess:
    """The subprocess invocation layer: env + cwd + args plumbing."""

    @pytest.fixture
    def fixture_dir(self, tmp_path):
        d = tmp_path / "fixture"
        d.mkdir()
        return d

    def test_subprocess_called_with_hermes_z_and_message(self, fixture_dir, tmp_path):
        runner = HermesAgentRunner(user_config_path=tmp_path / "nonexistent_config")
        captured = {}

        def _fake_run(*args, **kwargs):
            captured["args"] = args[0] if args else kwargs.get("args")
            captured["env"] = kwargs.get("env")
            captured["cwd"] = kwargs.get("cwd")
            # Drop a minimal state.db so the parse layer succeeds.
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            _make_state_db(
                sandbox / "state.db",
                session_id="s1", model="test-model",
                messages=[{"role": "assistant", "tool_calls": [
                    {"function": {"name": "patch"}}]}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            result = runner.run(TaskRunContext(
                user_message="please patch a file",
                fixture_dir=fixture_dir,
            ))

        assert captured["args"][:2] == ["hermes", "-z"]
        assert captured["args"][2] == "please patch a file"
        assert captured["cwd"] == str(fixture_dir)
        # Sandbox env vars: HERMES_HOME and HOME point at the SAME tmp dir.
        assert captured["env"]["HERMES_HOME"] == captured["env"]["HOME"]
        assert captured["env"]["HERMES_HOME"].startswith("/")  # absolute tmp dir
        # Parse layer worked: we got the canned tool call back.
        assert result.tool_calls_seq == ["patch"]
        assert result.error is None

    def test_timeout_returns_error_result(self, fixture_dir, tmp_path):
        runner = HermesAgentRunner(timeout_seconds=1, user_config_path=tmp_path / "x")
        import subprocess as _subprocess
        with patch(
            "evolution.validation.hermes_runner.subprocess.run",
            side_effect=_subprocess.TimeoutExpired(cmd="hermes", timeout=1),
        ):
            result = runner.run(TaskRunContext(
                user_message="hang",
                fixture_dir=fixture_dir,
            ))
        assert result.error is not None
        assert "timed out" in result.error
        assert result.tool_calls_seq == []

    def test_hermes_not_installed_returns_error_result(self, fixture_dir, tmp_path):
        runner = HermesAgentRunner(
            hermes_command="hermes-does-not-exist",
            user_config_path=tmp_path / "x",
        )
        with patch(
            "evolution.validation.hermes_runner.subprocess.run",
            side_effect=FileNotFoundError("hermes-does-not-exist"),
        ):
            result = runner.run(TaskRunContext(
                user_message="run",
                fixture_dir=fixture_dir,
            ))
        assert result.error is not None
        assert "not found" in result.error

    def test_no_session_written_returns_error_result(self, fixture_dir, tmp_path):
        runner = HermesAgentRunner(user_config_path=tmp_path / "x")

        def _fake_run(*args, **kwargs):
            # Don't write a state.db.
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            result = runner.run(TaskRunContext(
                user_message="run",
                fixture_dir=fixture_dir,
            ))
        assert result.error is not None
        assert "state.db absent" in result.error

    def test_user_config_copied_into_sandbox_when_exists(self, fixture_dir, tmp_path):
        user_config = tmp_path / "user_config.yaml"
        user_config.write_text("api_key: hunter2\n")
        runner = HermesAgentRunner(user_config_path=user_config)

        sandbox_seen = {}

        def _fake_run(*args, **kwargs):
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            sandbox_seen["config_present"] = (sandbox / "config.yaml").exists()
            sandbox_seen["config_text"] = (sandbox / "config.yaml").read_text() if sandbox_seen["config_present"] else None
            (sandbox / "sessions").mkdir(exist_ok=True)
            _write_session(
                sandbox / "sessions" / "session_test.json",
                [{"role": "assistant", "content": "ok"}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            runner.run(TaskRunContext(
                user_message="run",
                fixture_dir=fixture_dir,
            ))

        assert sandbox_seen["config_present"] is True
        assert "hunter2" in sandbox_seen["config_text"]

    def test_skills_src_copied_into_sandbox_when_present(self, fixture_dir, tmp_path):
        """When TaskRunContext.skills_src points at a directory, the runner
        copies its contents into <sandbox>/skills/ so ``hermes -z``
        discovers any candidate skills installed there.
        """
        skills_src = tmp_path / "candidate_skills"
        (skills_src / "systematic_debugging").mkdir(parents=True)
        (skills_src / "systematic_debugging" / "SKILL.md").write_text(
            "---\nname: systematic_debugging\n---\n\nevolved body\n"
        )
        runner = HermesAgentRunner(user_config_path=tmp_path / "nonexistent")

        sandbox_skills_state: dict = {}

        def _fake_run(*args, **kwargs):
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            skill_path = sandbox / "skills" / "systematic_debugging" / "SKILL.md"
            sandbox_skills_state["present"] = skill_path.is_file()
            if sandbox_skills_state["present"]:
                sandbox_skills_state["text"] = skill_path.read_text()
            (sandbox / "sessions").mkdir(exist_ok=True)
            _write_session(
                sandbox / "sessions" / "session_test.json",
                [{"role": "assistant", "content": "ok"}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            runner.run(TaskRunContext(
                user_message="debug it",
                fixture_dir=fixture_dir,
                skills_src=skills_src,
            ))

        assert sandbox_skills_state["present"] is True
        assert "evolved body" in sandbox_skills_state["text"]

    def test_model_override_passes_minus_m_flag(self, fixture_dir, tmp_path):
        """When model is set, hermes is invoked with `hermes -m MODEL -z ...`."""
        runner = HermesAgentRunner(
            user_config_path=tmp_path / "nonexistent",
            model="gpt-4o-mini",
        )
        captured: dict = {}

        def _fake_run(*args, **kwargs):
            captured["args"] = args[0] if args else kwargs.get("args")
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            (sandbox / "sessions").mkdir(exist_ok=True)
            _write_session(
                sandbox / "sessions" / "session_test.json",
                [{"role": "assistant", "content": "ok"}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            runner.run(TaskRunContext(
                user_message="debug",
                fixture_dir=fixture_dir,
            ))

        # -m must appear before -z so hermes parses it as a global flag,
        # not as part of the -z message.
        assert captured["args"][:4] == ["hermes", "-m", "gpt-4o-mini", "-z"]
        assert captured["args"][4] == "debug"

    def test_model_none_omits_minus_m_flag(self, fixture_dir, tmp_path):
        """No model override → original argv shape (no -m), preserves the
        existing behavior bit-for-bit for callers that don't opt in."""
        runner = HermesAgentRunner(user_config_path=tmp_path / "nonexistent")
        captured: dict = {}

        def _fake_run(*args, **kwargs):
            captured["args"] = args[0] if args else kwargs.get("args")
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            (sandbox / "sessions").mkdir(exist_ok=True)
            _write_session(
                sandbox / "sessions" / "session_test.json",
                [{"role": "assistant", "content": "ok"}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            runner.run(TaskRunContext(
                user_message="hello",
                fixture_dir=fixture_dir,
            ))

        assert "-m" not in captured["args"]
        assert captured["args"] == ["hermes", "-z", "hello"]

    def test_litellm_prefix_stripped_before_minus_m(self, fixture_dir, tmp_path):
        """Regression: hermes -m treats '<provider>/<model>' as openrouter-style
        routing (silently switches base_url to openrouter.ai and breaks auth
        for direct-provider configs). Users naturally pass litellm-formatted
        names like 'openai/gpt-4o-mini' from elsewhere in the framework, so
        the runner must strip known litellm provider prefixes before -m to
        avoid a silent 0-turn 'agent never ran' failure that misreports as
        'validator too weak' at the saturation pre-flight."""
        runner = HermesAgentRunner(
            user_config_path=tmp_path / "nonexistent",
            model="openai/gpt-4o-mini",
        )
        captured: dict = {}

        def _fake_run(*args, **kwargs):
            captured["args"] = args[0] if args else kwargs.get("args")
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            (sandbox / "sessions").mkdir(exist_ok=True)
            _write_session(
                sandbox / "sessions" / "session_test.json",
                [{"role": "assistant", "content": "ok"}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            runner.run(TaskRunContext(
                user_message="debug",
                fixture_dir=fixture_dir,
            ))

        assert captured["args"][:4] == ["hermes", "-m", "gpt-4o-mini", "-z"]
        assert "openai/" not in captured["args"][2]

    def test_skills_src_none_means_no_skills_dir_created(self, fixture_dir, tmp_path):
        """Tool-side runs (no skills_src) must not create an empty skills/
        directory in the sandbox — keeps the legacy code path bit-for-bit."""
        runner = HermesAgentRunner(user_config_path=tmp_path / "nonexistent")
        sandbox_seen: dict = {}

        def _fake_run(*args, **kwargs):
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            sandbox_seen["has_skills_dir"] = (sandbox / "skills").exists()
            (sandbox / "sessions").mkdir(exist_ok=True)
            _write_session(
                sandbox / "sessions" / "session_test.json",
                [{"role": "assistant", "content": "ok"}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            runner.run(TaskRunContext(
                user_message="run",
                fixture_dir=fixture_dir,
            ))

        assert sandbox_seen["has_skills_dir"] is False


class TestHermesAgentRunnerCostLedger:
    """HermesAgentRunner records agent cost to the ledger on every exit path."""

    @pytest.fixture
    def fixture_dir(self, tmp_path):
        d = tmp_path / "fixture"
        d.mkdir()
        return d

    def _fake_run_with_cost_db(self, actual_cost_usd):
        """Return a subprocess.run side_effect that writes a state.db with cost."""
        def _fake_run(*args, **kwargs):
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            _make_state_db_with_cost(
                sandbox / "state.db",
                session_id="s1", model="gpt-5.4-mini",
                messages=[{"role": "assistant", "content": "ok"}],
                actual_cost_usd=actual_cost_usd,
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return _fake_run

    def test_success_path_records_actual_cost(self, fixture_dir, tmp_path):
        """A run whose state.db carries actual_cost_usd calls record_agent_cost
        exactly once with the captured dollar value."""
        from unittest.mock import MagicMock
        from evolution.core.lm_timing_callback import CostLedger

        fake_ledger = MagicMock(spec=CostLedger)
        fake_ledger.get_abort_state.return_value = None
        runner = HermesAgentRunner(
            user_config_path=tmp_path / "nonexistent",
            cost_ledger=fake_ledger,
        )

        with patch(
            "evolution.validation.hermes_runner.subprocess.run",
            side_effect=self._fake_run_with_cost_db(0.01),
        ):
            result = runner.run(TaskRunContext(
                user_message="do something",
                fixture_dir=fixture_dir,
            ))

        assert result.error is None
        fake_ledger.record_agent_cost.assert_called_once_with(pytest.approx(0.01))

    def test_abstain_path_no_state_db_records_uncaptured(self, fixture_dir, tmp_path):
        """A run that writes no state.db calls record_agent_cost exactly once
        with None — the run is counted, cost unknown."""
        from unittest.mock import MagicMock
        from evolution.core.lm_timing_callback import CostLedger

        fake_ledger = MagicMock(spec=CostLedger)
        fake_ledger.get_abort_state.return_value = None
        runner = HermesAgentRunner(
            user_config_path=tmp_path / "nonexistent",
            cost_ledger=fake_ledger,
        )

        def _no_db(*args, **kwargs):
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_no_db):
            result = runner.run(TaskRunContext(
                user_message="run",
                fixture_dir=fixture_dir,
            ))

        assert result.error is not None
        fake_ledger.record_agent_cost.assert_called_once_with(None)

    def test_timeout_path_records_uncaptured(self, fixture_dir, tmp_path):
        """A timeout also calls record_agent_cost exactly once with None."""
        import subprocess as _subprocess
        from unittest.mock import MagicMock
        from evolution.core.lm_timing_callback import CostLedger

        fake_ledger = MagicMock(spec=CostLedger)
        fake_ledger.get_abort_state.return_value = None
        runner = HermesAgentRunner(
            timeout_seconds=1,
            user_config_path=tmp_path / "nonexistent",
            cost_ledger=fake_ledger,
        )

        with patch(
            "evolution.validation.hermes_runner.subprocess.run",
            side_effect=_subprocess.TimeoutExpired(cmd="hermes", timeout=1),
        ):
            result = runner.run(TaskRunContext(
                user_message="hang",
                fixture_dir=fixture_dir,
            ))

        assert "timed out" in result.error
        fake_ledger.record_agent_cost.assert_called_once_with(None)

    def test_command_not_found_path_records_uncaptured(self, fixture_dir, tmp_path):
        """FileNotFoundError also calls record_agent_cost exactly once."""
        from unittest.mock import MagicMock
        from evolution.core.lm_timing_callback import CostLedger

        fake_ledger = MagicMock(spec=CostLedger)
        fake_ledger.get_abort_state.return_value = None
        runner = HermesAgentRunner(
            hermes_command="no-such-hermes",
            user_config_path=tmp_path / "nonexistent",
            cost_ledger=fake_ledger,
        )

        with patch(
            "evolution.validation.hermes_runner.subprocess.run",
            side_effect=FileNotFoundError("no-such-hermes"),
        ):
            result = runner.run(TaskRunContext(
                user_message="run",
                fixture_dir=fixture_dir,
            ))

        assert "not found" in result.error
        fake_ledger.record_agent_cost.assert_called_once_with(None)

    def test_default_ledger_is_COST_LEDGER(self, tmp_path):
        """Omitting cost_ledger binds the runner to the module-level COST_LEDGER."""
        from evolution.core.lm_timing_callback import COST_LEDGER

        runner = HermesAgentRunner(user_config_path=tmp_path / "nonexistent")
        assert runner.cost_ledger is COST_LEDGER

    def test_agent_cost_over_ceiling_raises_after_recording(self, fixture_dir, tmp_path):
        """FIX 1: after recording an agent cost that trips the ceiling, run()
        raises CostCeilingExceeded eagerly (Layer-1 scoring makes no in-process
        LM call, so the BaseLM guard would never fire for an agent overrun)."""
        from evolution.core.lm_timing_callback import CostLedger, CostCeilingExceeded

        ledger = CostLedger()
        ledger.set_ceiling(0.005)
        runner = HermesAgentRunner(
            user_config_path=tmp_path / "nonexistent",
            cost_ledger=ledger,
        )
        with patch(
            "evolution.validation.hermes_runner.subprocess.run",
            side_effect=self._fake_run_with_cost_db(0.01),
        ):
            with pytest.raises(CostCeilingExceeded) as exc_info:
                runner.run(TaskRunContext(
                    user_message="spend", fixture_dir=fixture_dir,
                ))
        # The cost was recorded before the abort fired.
        assert ledger.summary()["agent_cost_usd"] == pytest.approx(0.01)
        assert exc_info.value.ceiling_usd == pytest.approx(0.005)

    def test_agent_cost_under_ceiling_does_not_raise(self, fixture_dir, tmp_path):
        """FIX 1: a run under the ceiling returns normally, no raise."""
        from evolution.core.lm_timing_callback import CostLedger

        ledger = CostLedger()
        ledger.set_ceiling(0.05)
        runner = HermesAgentRunner(
            user_config_path=tmp_path / "nonexistent",
            cost_ledger=ledger,
        )
        with patch(
            "evolution.validation.hermes_runner.subprocess.run",
            side_effect=self._fake_run_with_cost_db(0.01),
        ):
            result = runner.run(TaskRunContext(
                user_message="spend", fixture_dir=fixture_dir,
            ))
        assert result.error is None
        assert ledger.get_abort_state() is None

    def test_unexpected_parse_failure_records_once_and_does_not_crash(
        self, fixture_dir, tmp_path
    ):
        """FIX 4: an UNEXPECTED exception from parse_session_from_db (e.g. a bug
        or OSError) must NOT escape run() — it yields an error result, records
        cost exactly once (uncaptured), and lets the eval continue."""
        from unittest.mock import MagicMock
        from evolution.core.lm_timing_callback import CostLedger

        fake_ledger = MagicMock(spec=CostLedger)
        fake_ledger.get_abort_state.return_value = None
        runner = HermesAgentRunner(
            user_config_path=tmp_path / "nonexistent",
            cost_ledger=fake_ledger,
        )

        def _fake_run(*args, **kwargs):
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            _make_state_db(
                sandbox / "state.db", session_id="s1", model="m",
                messages=[{"role": "assistant", "content": "ok"}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch(
            "evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run
        ), patch(
            "evolution.validation.hermes_runner.parse_session_from_db",
            side_effect=RuntimeError("boom"),
        ):
            result = runner.run(TaskRunContext(
                user_message="run", fixture_dir=fixture_dir,
            ))

        assert result.error is not None
        assert "session db parse failed" in result.error
        fake_ledger.record_agent_cost.assert_called_once_with(None)

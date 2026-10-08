"""Tests for the Codex session-history importer.

Builds a synthetic Codex `thread_history_1.sqlite` so the tests never touch
the developer's real history, then asserts the importer's filtering contract:
injected context dropped, secrets dropped, turn pairs assembled, DB untouched.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evolution.core.codex_importer import (  # noqa: E402
    CodexImporter,
    _extract_text,
    _is_real_user_turn,
    _strip_injected_context,
)
from evolution.core.external_importers import _contains_secret  # noqa: E402

SCHEMA = """
CREATE TABLE thread_items (
    thread_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    rollout_ordinal INTEGER NOT NULL,
    created_at_ms INTEGER NOT NULL,
    item_json TEXT NOT NULL,
    item_type TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, turn_id, item_id)
);
CREATE TABLE thread_turns (
    thread_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    rollout_ordinal INTEGER NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (thread_id, turn_id)
);
"""


def _user_item(text):
    return {"type": "userMessage", "content": [{"type": "text", "text": text}]}


def _agent_item(text):
    return {"type": "agentMessage", "text": text}


@pytest.fixture
def fake_db(tmp_path):
    """Build a small Codex history DB with known-good and known-bad rows."""
    path = tmp_path / "thread_history_1.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)

    rows = [
        # (thread, turn, ordinal, type, json)
        ("t1", "u1", 0, "userMessage", _user_item("please audit the payment module against spec")),
        ("t1", "u1", 1, "agentMessage", _agent_item("Audited. Two violations found.")),
        # Harness-injected only — must be dropped.
        ("t2", "u2", 0, "userMessage", _user_item(
            "<model_switch>\nThe user was previously using a different model.\n</model_switch>"
        )),
        # Real request that happens to begin with an injected block — kept, cleaned.
        ("t3", "u3", 0, "userMessage", _user_item(
            "<environment_details>\ncwd: /tmp\n</environment_details>\n"
            "fix the failing typecheck in apps/web"
        )),
        ("t3", "u3", 1, "agentMessage", _agent_item("Fixed.")),
        # Secret-bearing — must be dropped.
        ("t4", "u4", 0, "userMessage", _user_item("here is my key sk-ant-api03-AbCdEfGhIjKlMnOpQrSt")),
        # Too short — dropped.
        ("t5", "u5", 0, "userMessage", _user_item("ok")),
        # Real request with no assistant reply — kept, empty response.
        ("t6", "u6", 0, "userMessage", _user_item("summarise the last gate readback for me please")),
        # Last substantive assistant message wins.
        ("t7", "u7", 0, "userMessage", _user_item("which model are you running right now")),
        ("t7", "u7", 1, "agentMessage", _agent_item("thinking...")),
        ("t7", "u7", 2, "agentMessage", _agent_item("I am running gpt-6-luna.")),
    ]
    for i, (th, tn, o, typ, payload) in enumerate(rows):
        conn.execute(
            "INSERT INTO thread_items VALUES (?,?,?,?,?,?,?)",
            (th, tn, f"i{i}", o, 0, json.dumps(payload), typ),
        )
    conn.commit()
    conn.close()
    return path


# ── pure helpers ──────────────────────────────────────────────────────────


def test_extract_text_from_content_blocks():
    assert _extract_text({"content": [{"type": "text", "text": "hello"}]}) == "hello"


def test_extract_text_handles_string_content():
    assert _extract_text({"content": "plain"}) == "plain"


def test_strip_removes_model_switch_block():
    out = _strip_injected_context(
        "<model_switch>\nnoise\n</model_switch>\nfix the build"
    )
    assert "noise" not in out
    assert "fix the build" in out


def test_strip_removes_agents_md_instruction_line():
    out = _strip_injected_context("# AGENTS.md instructions\n\ndo the thing")
    assert "do the thing" in out
    assert "AGENTS.md" not in out


def test_is_real_user_turn_rejects_injected_only():
    assert not _is_real_user_turn("<environment_details>\ncwd: /tmp\n</environment_details>")


def test_is_real_user_turn_rejects_short():
    assert not _is_real_user_turn("ok")


def test_is_real_user_turn_accepts_real_request():
    assert _is_real_user_turn("please refactor the parser module for clarity")


# ── importer ──────────────────────────────────────────────────────────────


def test_extracts_expected_pairs(fake_db):
    msgs = CodexImporter.extract_messages(db_path=fake_db)
    inputs = [m["task_input"] for m in msgs]
    assert len(msgs) == 4
    assert any("audit the payment module" in t for t in inputs)
    assert any("fix the failing typecheck" in t for t in inputs)
    assert any("summarise the last gate" in t for t in inputs)
    assert any("which model are you running" in t for t in inputs)


def test_drops_injected_only_turn(fake_db):
    msgs = CodexImporter.extract_messages(db_path=fake_db)
    assert not any("cwd: /tmp" in m["task_input"] and "typecheck" not in m["task_input"]
                   for m in msgs)


def test_strips_injected_block_from_real_request(fake_db):
    msgs = CodexImporter.extract_messages(db_path=fake_db)
    hit = [m for m in msgs if "typecheck" in m["task_input"]][0]
    assert "environment_details" not in hit["task_input"]
    assert "cwd: /tmp" not in hit["task_input"]


def test_drops_secret_bearing_turn(fake_db):
    msgs = CodexImporter.extract_messages(db_path=fake_db)
    assert not any("sk-ant-api" in m["task_input"] for m in msgs)
    assert all(not _contains_secret(m["task_input"]) for m in msgs)


def test_last_assistant_message_wins(fake_db):
    msgs = CodexImporter.extract_messages(db_path=fake_db)
    hit = [m for m in msgs if "which model" in m["task_input"]][0]
    assert "gpt-6-luna" in hit["assistant_response"]
    assert "thinking..." not in hit["assistant_response"]


def test_turn_without_reply_is_retained(fake_db):
    msgs = CodexImporter.extract_messages(db_path=fake_db)
    hit = [m for m in msgs if "summarise the last gate" in m["task_input"]][0]
    assert hit["assistant_response"] == ""


def test_records_session_and_turn_ids(fake_db):
    msgs = CodexImporter.extract_messages(db_path=fake_db)
    assert all(m["source"] == "codex" for m in msgs)
    assert all("session_id" in m and "turn_id" in m for m in msgs)


def test_missing_db_returns_empty(tmp_path):
    assert CodexImporter.extract_messages(db_path=tmp_path / "nope.sqlite") == []


def test_does_not_mutate_source_db(fake_db):
    before = fake_db.read_bytes()
    CodexImporter.extract_messages(db_path=fake_db)
    assert fake_db.read_bytes() == before


def test_limit_caps_results(fake_db):
    assert len(CodexImporter.extract_messages(limit=2, db_path=fake_db)) == 2


def test_dataset_stats(fake_db):
    stats = CodexImporter.dataset_stats(CodexImporter.extract_messages(db_path=fake_db))
    assert stats["pairs"] == 4
    assert stats["sessions"] == 4
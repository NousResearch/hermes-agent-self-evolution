"""Tests for mining (user task -> tool used) pairs from the Hermes state.db."""

import sqlite3
import time
from pathlib import Path

import pytest

from evolution.core.state_db import contains_secret as _contains_secret
from evolution.tools.tool_usage_miner import mine_tool_usage

REAL_DB = Path.home() / ".hermes" / "state.db"


# ── Real session store ────────────────────────────────────────────────────


@pytest.mark.skipif(not REAL_DB.exists(), reason="no Hermes state.db on this host")
def test_real_db_yields_terminal_pairs():
    pairs = mine_tool_usage(days=90)

    terminal = [p for p in pairs if p["tool"] == "terminal"]
    assert len(terminal) >= 100
    assert all(p["task_input"].strip() and p["tool"].strip() for p in pairs)


@pytest.mark.skipif(not REAL_DB.exists(), reason="no Hermes state.db on this host")
def test_real_db_has_no_secrets():
    pairs = mine_tool_usage(days=90)

    leaked = [p for p in pairs if _contains_secret(p["task_input"])]
    assert leaked == [], f"{len(leaked)} pairs leaked secret-looking text"


# ── Synthetic fixture ─────────────────────────────────────────────────────


SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    started_at REAL NOT NULL
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    tool_name TEXT,
    tool_calls TEXT,
    content TEXT,
    timestamp REAL NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
"""

INSERT_MSG = (
    "INSERT INTO messages"
    " (id, session_id, role, tool_name, tool_calls, content, timestamp)"
    " VALUES (?, ?, ?, ?, ?, ?, ?)"
)

NOW = time.time()


def _user(mid, session_id, content):
    return (mid, session_id, "user", None, None, content, NOW)


def _tool(mid, session_id, tool_name):
    return (mid, session_id, "tool", tool_name, None, "{}", NOW)


def _make_db(path: Path) -> Path:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
        [
            ("interactive", "cli", NOW - 60),
            ("machine", "subagent", NOW - 50),
            ("secret", "webui", NOW - 40),
        ],
    )
    con.executemany(
        INSERT_MSG,
        [
            # interactive: one task, two distinct tools -> two pairs
            _user(1, "interactive", "find the largest log file on this box"),
            (2, "interactive", "assistant", None, None, None, NOW),
            _tool(3, "interactive", "terminal"),
            _tool(4, "interactive", "terminal"),
            _tool(5, "interactive", "read_file"),
            # machine dispatch: never a pair
            _user(6, "machine", "summarise the failing test output"),
            _tool(7, "machine", "terminal"),
            # secret-bearing task: dropped
            _user(8, "secret", "deploy with OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwx"),
            _tool(9, "secret", "terminal"),
        ],
    )
    con.commit()
    con.close()
    return path


def test_filters_machine_sources_and_secrets(tmp_path):
    db = _make_db(tmp_path / "state.db")

    pairs = mine_tool_usage(days=90, db_path=db)

    assert sorted(p["tool"] for p in pairs) == ["read_file", "terminal"]
    assert all(p["session_id"] == "interactive" for p in pairs)
    assert all(p["task_input"] == "find the largest log file on this box" for p in pairs)
    assert not any(_contains_secret(p["task_input"]) for p in pairs)


def test_skips_short_and_boilerplate_user_messages(tmp_path):
    db = _make_db(tmp_path / "state.db")
    con = sqlite3.connect(db)
    con.executemany(
        INSERT_MSG,
        [
            _user(10, "interactive", "ok"),
            _tool(11, "interactive", "write_file"),
            _user(
                12,
                "interactive",
                "[Workspace::v1: /tmp]\n[CONTEXT COMPACTION summary of earlier turns]",
            ),
            _tool(13, "interactive", "patch"),
            _user(14, "interactive", "[Workspace::v1: /tmp]\nrun the full test suite please"),
            _tool(15, "interactive", "execute_code"),
        ],
    )
    con.commit()
    con.close()

    pairs = mine_tool_usage(days=90, db_path=db)

    texts = {p["task_input"] for p in pairs}
    assert "ok" not in texts
    assert not any(t.startswith("[CONTEXT COMPACTION") for t in texts)
    assert not any("[Workspace::v1:" in t for t in texts)
    assert any(
        p["tool"] == "execute_code"
        and p["task_input"] == "run the full test suite please"
        for p in pairs
    )


def test_tools_do_not_leak_across_user_turns(tmp_path):
    """A tool used for the NEXT task must not be attributed to this one."""
    db = _make_db(tmp_path / "state.db")
    con = sqlite3.connect(db)
    con.executemany(
        INSERT_MSG,
        [
            _user(20, "interactive", "first task: list open ports please"),
            _tool(21, "interactive", "terminal"),
            _user(22, "interactive", "second task: read the config file"),
            _tool(23, "interactive", "read_file"),
        ],
    )
    con.commit()
    con.close()

    pairs = mine_tool_usage(days=90, db_path=db)

    first = [p["tool"] for p in pairs if p["task_input"].startswith("first task")]
    second = [p["tool"] for p in pairs if p["task_input"].startswith("second task")]
    assert first == ["terminal"]
    assert second == ["read_file"]


def test_window_drops_tool_beyond_30_messages(tmp_path):
    db = _make_db(tmp_path / "state.db")
    con = sqlite3.connect(db)
    con.executemany(
        INSERT_MSG,
        [_user(30, "interactive", "task with a very late tool call")]
        + [(30 + k, "interactive", "assistant", None, None, "thinking", NOW) for k in range(1, 31)]
        + [_tool(61, "interactive", "terminal")],
    )
    con.commit()
    con.close()

    pairs = mine_tool_usage(days=90, db_path=db)

    assert not any(p["task_input"] == "task with a very late tool call" for p in pairs)


def test_days_validated():
    for bad in (0, -5, "abc"):
        with pytest.raises(ValueError):
            mine_tool_usage(days=bad)

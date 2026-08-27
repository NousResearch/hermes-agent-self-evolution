"""Fixtures that build a realistic fake Hermes install on disk.

Every store here mirrors the real schema, including the FTS5 index and the
column names the production databases actually use — a fixture that invents a
convenient schema would have let the original importer's bug pass its tests
just as easily as it passed in production.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from evolution.core.hermes_paths import HermesInstall


SESSIONS_DDL = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT '',
    model TEXT,
    system_prompt TEXT,
    started_at REAL NOT NULL DEFAULT 0,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    api_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    estimated_cost_usd REAL DEFAULT 0,
    system_prompt_hash TEXT
)
"""

MESSAGES_DDL = """
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL DEFAULT 0,
    token_count INTEGER
)
"""

VERIFICATION_DDL = """
CREATE TABLE verification_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL,
    session_id TEXT,
    cwd TEXT,
    root TEXT,
    command TEXT,
    canonical_command TEXT,
    kind TEXT,
    scope TEXT,
    status TEXT,
    exit_code INTEGER,
    output_summary TEXT
)
"""

EXECUTIONS_DDL = """
CREATE TABLE executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    source TEXT,
    process_id TEXT,
    pid INTEGER,
    process_started_at REAL,
    status TEXT,
    claimed_at REAL,
    started_at REAL,
    finished_at REAL,
    error TEXT
)
"""


def build_state_db(path: Path, sessions: list[dict], messages: list[dict], with_fts: bool = True):
    """Write a state.db that matches the production schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SESSIONS_DDL)
    conn.execute(MESSAGES_DDL)

    for s in sessions:
        conn.execute(
            """INSERT INTO sessions
               (id, source, model, system_prompt, started_at, end_reason,
                message_count, tool_call_count, api_call_count,
                input_tokens, output_tokens, estimated_cost_usd, system_prompt_hash)
               VALUES (:id, :source, :model, :system_prompt, :started_at, :end_reason,
                       :message_count, :tool_call_count, :api_call_count,
                       :input_tokens, :output_tokens, :estimated_cost_usd, :system_prompt_hash)""",
            {
                "source": "cli", "model": "gpt-5.6", "system_prompt": None,
                "started_at": 0.0, "end_reason": None, "message_count": 0,
                "tool_call_count": 0, "api_call_count": 0, "input_tokens": 0,
                "output_tokens": 0, "estimated_cost_usd": 0.0,
                "system_prompt_hash": None, **s,
            },
        )

    for m in messages:
        conn.execute(
            """INSERT INTO messages (id, session_id, role, content, tool_name, timestamp, token_count)
               VALUES (:id, :session_id, :role, :content, :tool_name, :timestamp, :token_count)""",
            {"id": None, "tool_name": None, "timestamp": 0.0, "token_count": 0, "content": None, **m},
        )

    if with_fts:
        # Mirrors the real index: a standalone FTS5 table whose rowid tracks
        # messages.id, which is exactly the join the importer relies on.
        conn.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(content)")
        conn.execute(
            "INSERT INTO messages_fts(rowid, content) "
            "SELECT id, COALESCE(content, '') FROM messages"
        )

    conn.commit()
    conn.close()


@pytest.fixture
def hermes_root(tmp_path: Path) -> Path:
    """A Hermes data directory with two profiles and populated stores."""
    root = tmp_path / "hermes"
    (root / "skills").mkdir(parents=True)
    (root / "cron").mkdir(parents=True)
    (root / "config.yaml").write_text("gateway: {}\n")

    # ── profile: ali ──
    ali = root / "profiles" / "ali"
    build_state_db(
        ali / "state.db",
        sessions=[
            {
                "id": "s1", "started_at": 100.0, "end_reason": "cron_complete",
                "tool_call_count": 4, "input_tokens": 1000, "output_tokens": 200,
                "system_prompt_hash": "hashA",
                "system_prompt": "You are Hermes.\n\n# Tool use\nPrefer read_file.\n\n# Style\nBe brief.\n",
            },
            {
                "id": "s2", "started_at": 200.0, "end_reason": "webhook_complete",
                "tool_call_count": 9, "input_tokens": 3000, "output_tokens": 900,
                "system_prompt_hash": "hashA",
                "system_prompt": "You are Hermes.\n\n# Tool use\nPrefer read_file.\n\n# Style\nBe brief.\n",
            },
            {
                "id": "s3", "started_at": 300.0, "end_reason": "error",
                "tool_call_count": 2, "system_prompt_hash": "hashB",
                "system_prompt": "You are Hermes v2.\n\n# Tool use\nUse terminal.\n",
            },
        ],
        messages=[
            {"session_id": "s1", "role": "user", "content": "audit the ahrefs backlink report for the domain", "timestamp": 1},
            {"session_id": "s1", "role": "tool", "content": "…", "tool_name": "tool_search", "timestamp": 2},
            {"session_id": "s1", "role": "tool", "content": "…", "tool_name": "read_file", "timestamp": 3},
            {"session_id": "s1", "role": "assistant", "content": "Here is the backlink audit.", "timestamp": 4},

            {"session_id": "s2", "role": "user", "content": "run the deployment pipeline for staging", "timestamp": 1},
            {"session_id": "s2", "role": "tool", "content": "…", "tool_name": "terminal", "timestamp": 2},
            {"session_id": "s2", "role": "assistant", "content": "Deployed to staging.", "timestamp": 3},

            {"session_id": "s3", "role": "user", "content": "hi", "timestamp": 1},
            {"session_id": "s3", "role": "user", "content": "my key is sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA", "timestamp": 2},
            {"session_id": "s3", "role": "assistant", "content": "noted", "timestamp": 3},
        ],
    )

    conn = sqlite3.connect(ali / "verification_evidence.db")
    conn.execute(VERIFICATION_DDL)
    conn.executemany(
        "INSERT INTO verification_events (session_id, command, kind, status, exit_code, output_summary, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        [
            ("s1", "pytest tests/", "test", "pass", 0, "12 passed", 1.0),
            ("s1", "ruff check .", "lint", "fail", 1, "3 errors", 2.0),
            ("s2", "npm run build", "build", "pass", 0, "built", 3.0),
        ],
    )
    conn.commit()
    conn.close()

    # ── profile: musa (present but empty state) ──
    musa = root / "profiles" / "musa"
    build_state_db(musa / "state.db", sessions=[], messages=[])

    # ── cron ──
    conn = sqlite3.connect(root / "cron" / "executions.db")
    conn.execute(EXECUTIONS_DDL)
    conn.executemany(
        "INSERT INTO executions (job_id, status, error) VALUES (?,?,?)",
        [("job-seo", "completed", None)] * 8
        + [("job-seo", "failed", "timeout")] * 2
        + [("job-brief", "completed", None)] * 5
        + [("job-brief", "running", None)],
    )
    conn.commit()
    conn.close()

    (root / "cron" / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {"id": "job-seo", "name": "seo daily", "skills": ["ahrefs-seo-operations"], "skill": None},
                    {"id": "job-brief", "name": "briefing", "skills": [], "skill": "partner-meeting-prep"},
                    {"id": "job-none", "name": "no skills", "skills": []},
                ]
            }
        )
    )

    return root


@pytest.fixture
def install(hermes_root: Path) -> HermesInstall:
    return HermesInstall(root=hermes_root, source="fixture")

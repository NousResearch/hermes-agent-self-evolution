"""Mine (user task -> tool used) pairs from the Hermes session store.

The pairs are the raw material for tool-description selection evals: they record
which tool actually answered a real user task, in the user's own words.
Machine-dispatch sources (state_db.MACHINE_SOURCES) and system-injected wrapper
messages are excluded.
"""

import sqlite3
from pathlib import Path

from evolution.core.state_db import (
    DEFAULT_DB,
    MACHINE_SOURCES,
    clean_user_text,
    contains_secret,
    is_task_like,
)

TOOL_WINDOW = 30  # max messages after a user turn that count as "the response"


def mine_tool_usage(days: int = 90, db_path: Path | None = None) -> list[dict]:
    """Return one `{task_input, tool, session_id}` dict per (user message, tool)."""
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise ValueError("days must be a positive int")
    db = Path(db_path) if db_path is not None else DEFAULT_DB
    if not db.exists():
        return []

    placeholders = ",".join("?" * len(MACHINE_SOURCES))
    con = None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        sessions = con.execute(
            f"SELECT id FROM sessions"
            f" WHERE source NOT IN ({placeholders})"
            f" AND started_at > CAST(strftime('%s','now','-' || ? || ' days') AS REAL)"
            f" ORDER BY started_at DESC",
            (*MACHINE_SOURCES, days),
        ).fetchall()

        pairs: list[dict] = []
        for (session_id,) in sessions:
            rows = con.execute(
                "SELECT role, tool_name, content FROM messages"
                " WHERE session_id = ? AND active = 1 ORDER BY id",
                (session_id,),
            ).fetchall()
            for i, (role, _tool, content) in enumerate(rows):
                if role != "user":
                    continue
                text = clean_user_text(content)
                if not is_task_like(text):
                    continue
                if contains_secret(text):
                    continue
                seen: set[str] = set()
                for win_role, win_tool, _ in rows[i + 1 : i + 1 + TOOL_WINDOW]:
                    if win_role == "user":
                        break  # the next turn's tools belong to its own task
                    if win_role != "tool" or not win_tool or win_tool in seen:
                        continue
                    seen.add(win_tool)
                    pairs.append(
                        {"task_input": text, "tool": win_tool, "session_id": session_id}
                    )
        return pairs
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"cannot read session store {db} (locked or corrupt?): {e}") from e
    finally:
        if con is not None:
            con.close()

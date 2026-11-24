"""Mine (user task -> tool used) pairs from the Hermes session store.

The pairs are the raw material for tool-description selection evals: they record
which tool actually answered a real user task, in the user's own words.

Interactive sessions only — subagent, cron, task and local sources are machine
dispatches, so their "user" text is an orchestrator prompt, not a task.
"""

import re
import sqlite3
from pathlib import Path

from evolution.core.external_importers import _contains_secret

DEFAULT_DB = Path.home() / ".hermes" / "state.db"

MACHINE_SOURCES = ("subagent", "cron", "task", "local")
SKIP_PREFIXES = ("[CONTEXT COMPACTION", "[ASYNC DELEGATION")
TOOL_WINDOW = 30  # messages after a user turn that count as "the response"
MIN_TASK_CHARS = 10

_WORKSPACE_LINE = re.compile(r"^\[Workspace::v1:.*$", re.MULTILINE)


def _clean(text: str | None) -> str:
    return _WORKSPACE_LINE.sub("", text or "").strip()


def mine_tool_usage(days: int = 90, db_path: Path | None = None) -> list[dict]:
    """Return one `{task_input, tool, session_id}` dict per (user message, tool)."""
    db = Path(db_path) if db_path is not None else DEFAULT_DB
    if not db.exists():
        return []

    placeholders = ",".join("?" * len(MACHINE_SOURCES))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
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
                text = _clean(content)
                if len(text) < MIN_TASK_CHARS or text.startswith(SKIP_PREFIXES):
                    continue
                if _contains_secret(text):
                    continue
                seen: set[str] = set()
                for win_role, win_tool, _ in rows[i + 1 : i + 1 + TOOL_WINDOW]:
                    if win_role != "tool" or not win_tool or win_tool in seen:
                        continue
                    seen.add(win_tool)
                    pairs.append(
                        {"task_input": text, "tool": win_tool, "session_id": session_id}
                    )
        return pairs
    finally:
        con.close()

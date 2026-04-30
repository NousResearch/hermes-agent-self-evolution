"""SQLite-backed evolution progress tracker.

Stores run metadata and step-by-step events so every evolution run is
observable, queryable, and recoverable across process restarts.

DB location:  ~/.hermes/evolution_progress.db
               (or $HERMES_HOME/evolution_progress.db if set)
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── DB path ────────────────────────────────────────────────────────────────

def _db_path() -> Path:
    hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    return Path(hermes_home) / "evolution_progress.db"


# ── Schema ────────────────────────────────────────────────────────────────

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    skill_name      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running',
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    iterations      INTEGER,
    optimizer_model TEXT,
    eval_model      TEXT,
    baseline_score  REAL,
    evolved_score   REAL,
    improvement     REAL,
    baseline_size   INTEGER,
    evolved_size    INTEGER,
    constraints_passed INTEGER,
    scoring_method  TEXT
);
"""

_CREATE_RUN_EVENTS = """
CREATE TABLE IF NOT EXISTS run_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    step       TEXT NOT NULL,
    detail     TEXT,
    timestamp  TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_run_events_run_id ON run_events(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""


def _ensure_db() -> sqlite3.Connection:
    """Create the DB (if needed) and return a connection."""
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_CREATE_RUNS)
    conn.executescript(_CREATE_RUN_EVENTS)
    conn.executescript(_CREATE_INDEX)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ── Public API ─────────────────────────────────────────────────────────────

def start_run(skill_name: str, config) -> dict:
    """Register a new evolution run and return its metadata as a dict.

    Args:
        skill_name: The skill being evolved.
        config:      An EvolutionConfig (or any object with common attributes).

    Returns:
        Dict with at least ``run_id`` plus all persisted fields.
    """
    conn = _ensure_db()
    run_id = uuid.uuid4().hex
    now = datetime.now().isoformat()

    iterations = getattr(config, "iterations", None)
    optimizer_model = getattr(config, "optimizer_model", None)
    eval_model = getattr(config, "eval_model", None)
    scoring_method = getattr(config, "scoring_method", None)

    conn.execute(
        """
        INSERT INTO runs
            (id, skill_name, status, started_at, iterations,
             optimizer_model, eval_model, scoring_method)
        VALUES (?, ?, 'running', ?, ?, ?, ?, ?)
        """,
        (run_id, skill_name, now, iterations, optimizer_model, eval_model, scoring_method),
    )
    conn.commit()
    conn.close()

    return {
        "run_id": run_id,
        "skill_name": skill_name,
        "status": "running",
        "started_at": now,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "scoring_method": scoring_method,
    }


def log_event(run_id: str, step: str, detail: str = "") -> dict:
    """Append a step event to a run's event log.

    Args:
        run_id: The run identifier.
        step:   Short step label (e.g. ``"dataset_generation"``).
        detail: Free-form human-readable detail text.

    Returns:
        Dict representing the created event row.
    """
    conn = _ensure_db()
    now = datetime.now().isoformat()

    cur = conn.execute(
        """
        INSERT INTO run_events (run_id, step, detail, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (run_id, step, detail, now),
    )
    event_id = cur.lastrowid
    conn.commit()
    conn.close()

    return {
        "id": event_id,
        "run_id": run_id,
        "step": step,
        "detail": detail,
        "timestamp": now,
    }


def complete_run(run_id: str, results: dict) -> dict:
    """Mark a run as completed and persist result metrics.

    Args:
        run_id:  The run identifier.
        results: Dict with any subset of the runs columns
                 (evolved_score, baseline_score, improvement, …).

    Returns:
        Dict of the updated run row.
    """
    conn = _ensure_db()
    now = datetime.now().isoformat()

    # Build dynamic SET clause from results dict
    allowed = {
        "iterations", "optimizer_model", "eval_model",
        "baseline_score", "evolved_score", "improvement",
        "baseline_size", "evolved_size",
        "constraints_passed", "scoring_method",
    }
    sets = ["status = 'completed'", "completed_at = ?"]
    values: list = [now]

    for key, val in results.items():
        if key in allowed:
            sets.append(f"{key} = ?")
            values.append(val)

    values.append(run_id)
    conn.execute(
        f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", values
    )
    conn.commit()

    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()

    return dict(row) if row else {"run_id": run_id, "status": "completed"}


def fail_run(run_id: str, error: str) -> dict:
    """Mark a run as failed with an error message logged as an event.

    Args:
        run_id: The run identifier.
        error:  Human-readable error description.

    Returns:
        Dict of the updated run row.
    """
    conn = _ensure_db()
    now = datetime.now().isoformat()

    conn.execute(
        "UPDATE runs SET status = 'failed', completed_at = ? WHERE id = ?",
        (now, run_id),
    )

    # Also record the failure as an event for easy tailing
    conn.execute(
        "INSERT INTO run_events (run_id, step, detail, timestamp) VALUES (?, 'failed', ?, ?)",
        (run_id, error, now),
    )
    conn.commit()

    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()

    return dict(row) if row else {"run_id": run_id, "status": "failed"}


def get_active_run() -> Optional[dict]:
    """Return the currently running run, or ``None`` if nothing is active.

    Only one run should be ``running`` at a time.
    """
    conn = _ensure_db()
    row = conn.execute(
        "SELECT * FROM runs WHERE status = 'running' ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_run_history(limit: int = 50) -> list:
    """Return the most recent runs (newest first).

    Args:
        limit: Maximum number of rows to return.

    Returns:
        List of dicts, each a run row.
    """
    conn = _ensure_db()
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_run_events(run_id: str) -> list:
    """Return all events for a given run, oldest first.

    Args:
        run_id: The run identifier.

    Returns:
        List of dicts, each an event row.
    """
    conn = _ensure_db()
    rows = conn.execute(
        "SELECT * FROM run_events WHERE run_id = ? ORDER BY timestamp ASC",
        (run_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

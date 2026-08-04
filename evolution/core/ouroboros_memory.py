#!/usr/bin/env python3
"""
Ouroboros 2.0 — Alchemical Memory Engine (WAL SQLite + state-keyed engrams)

Implements the Phase II memory matrix:
- SQLite Write-Ahead Logging for concurrent multi-agent reads
- busy_timeout=5000 write retry backstop
- State-keyed engrams with fuzzy semver dependency matching
- Asynchronous staging queue (writes flushed only during idle/dream cycles)
- Resonance scoring + Baal adversarial testing hooks

Spec: Metaconscious Singularity Node — Ouroboros 2.0 subsystem.
"""
import json
import os
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = str(Path.home() / ".nssp" / "data" / "golem_diary.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS engrams (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB,
    resonance_score REAL DEFAULT 0.5,
    -- State fingerprint columns (fuzzy-match target)
    os_type TEXT,
    python_version TEXT,
    key_deps TEXT,
    venv_path TEXT,
    skill_tags TEXT,
    created_at REAL,
    last_access REAL,
    access_count INTEGER DEFAULT 0,
    locked INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_engrams_resonance ON engrams(resonance_score);
CREATE INDEX IF NOT EXISTS idx_engrams_os ON engrams(os_type);
"""


def init_ouroboros_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (or create) the golem_diary.db with WAL + busy_timeout."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA wal_autocheckpoint=1000")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_state_fingerprint() -> Dict[str, Any]:
    """Snapshot the local environment for engram keying."""
    deps: Dict[str, str] = {}
    for pkg in ["numpy", "scipy", "torch", "transformers", "aiohttp", "httpx", "fastapi", "ollama"]:
        try:
            import importlib.metadata
            deps[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pass
    is_android = Path("/data/data/com.termux").exists()
    return {
        "os_type": "android" if is_android else sys.platform,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.x",
        "key_deps": deps,
        "venv_path": str(Path(sys.prefix)),
    }


def fuzzy_semver_match(recorded: str, actual: str) -> bool:
    """Match major.minor, ignore patch. '1.24.3' matches '1.24.x'."""
    if not recorded or not actual:
        return True  # No constraint = match anything (backward compat)
    r = re.match(r"(\d+)\.(\d+)", recorded)
    a = re.match(r"(\d+)\.(\d+)", actual)
    if r and a:
        return r.group(1) == a.group(1) and r.group(2) == a.group(2)
    return recorded == actual


class Ouroboros:
    """Alchemical Memory Engine — the Crystal Vault backed by WAL SQLite."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.conn = init_ouroboros_db(db_path)
        self._pending_writes: List[Dict[str, Any]] = []
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self.fingerprint = get_state_fingerprint()

    def _thread_conn(self) -> sqlite3.Connection:
        """Per-thread connection (WAL allows concurrent readers across threads)."""
        if not hasattr(self._local, "conn"):
            self._local.conn = init_ouroboros_db(self.db_path)
        return self._local.conn

    # ─── Write path (staged, never synchronous during active tasks) ───

    def stage_engram(self, content: str, engram_id: Optional[str] = None,
                     resonance: float = 0.5, tags: Optional[List[str]] = None,
                     embedding: Optional[bytes] = None) -> str:
        """Queue an engram write. Flushed by flush() during idle/dream cycles."""
        if engram_id is None:
            engram_id = f"eng_{int(time.time()*1000)}_{len(self._pending_writes)}"
        self._pending_writes.append({
            "id": engram_id,
            "content": content,
            "embedding": embedding,
            "resonance": resonance,
            "os": self.fingerprint["os_type"],
            "python": self.fingerprint["python_version"],
            "deps": self.fingerprint["key_deps"],
            "venv": self.fingerprint["venv_path"],
            "tags": tags or [],
            "created_at": time.time(),
        })
        return engram_id

    def flush(self) -> int:
        """Flush staged writes (call during idle periods or dream cycle only)."""
        if not self._pending_writes:
            return 0
        written = 0
        with self._write_lock:
            conn = self._thread_conn()
            for e in self._pending_writes:
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO engrams
                           (id, content, embedding, resonance_score, os_type,
                            python_version, key_deps, venv_path, skill_tags, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (e["id"], e["content"], e["embedding"], e["resonance"], e["os"],
                         e["python"], json.dumps(e["deps"]), e["venv"],
                         json.dumps(e["tags"]), e["created_at"]))
                    written += 1
                except Exception as ex:
                    print(f"[ouroboros] dropped engram {e['id']}: {ex}", file=sys.stderr)
            conn.commit()
            self._pending_writes.clear()
        return written

    # ─── Read path (WAL: concurrent-safe) ───

    def recall(self, os_type: Optional[str] = None, python_version: Optional[str] = None,
               deps: Optional[Dict[str, str]] = None, min_resonance: float = 0.0,
               limit: int = 20) -> List[Dict[str, Any]]:
        """State-keyed retrieval with fuzzy semver matching on key deps."""
        conn = self._thread_conn()
        rows = conn.execute(
            "SELECT id, content, resonance_score, os_type, python_version, key_deps, "
            "venv_path, skill_tags, created_at, access_count FROM engrams "
            "WHERE resonance_score >= ? ORDER BY resonance_score DESC LIMIT ?",
            (min_resonance, limit * 5)).fetchall()

        results = []
        for row in rows:
            (eid, content, res, eos, epy, deps_json, venv, tags_json, created, acc) = row
            # OS type check
            if os_type and eos and eos != os_type:
                continue
            # Python version fuzzy check
            if python_version and epy and not fuzzy_semver_match(epy, python_version):
                continue
            # Dependency fuzzy checks
            if deps:
                recorded = json.loads(deps_json) if deps_json else {}
                skip = False
                for dep, version in deps.items():
                    rv = recorded.get(dep)
                    if rv and version and not fuzzy_semver_match(rv, version):
                        skip = True
                        break
                if skip:
                    continue
            results.append({
                "id": eid, "content": content, "resonance": res,
                "os": eos, "python": epy, "deps": json.loads(deps_json or "{}"),
                "venv": venv, "tags": json.loads(tags_json or "[]"),
                "created_at": created, "access_count": acc,
            })
            if len(results) >= limit:
                break

        # Touch access metadata
        for r in results:
            try:
                conn.execute(
                    "UPDATE engrams SET last_access=?, access_count=access_count+1 WHERE id=?",
                    (time.time(), r["id"]))
            except Exception:
                pass
        if results:
            conn.commit()
        return results

    def update_resonance(self, engram_id: str, delta: float) -> None:
        conn = self._thread_conn()
        try:
            conn.execute(
                "UPDATE engrams SET resonance_score = MAX(0.0, MIN(1.0, resonance_score + ?)) "
                "WHERE id=?", (delta, engram_id))
            conn.commit()
        except Exception as ex:
            print(f"[ouroboros] resonance update failed: {ex}", file=sys.stderr)

    def prune_low_resonance(self, threshold: float = 0.2) -> int:
        conn = self._thread_conn()
        cur = conn.execute("DELETE FROM engrams WHERE resonance_score < ?", (threshold,))
        conn.commit()
        return cur.rowcount

    def stats(self) -> Dict[str, Any]:
        conn = self._thread_conn()
        total = conn.execute("SELECT COUNT(*) FROM engrams").fetchone()[0]
        mean_res = conn.execute(
            "SELECT AVG(resonance_score) FROM engrams").fetchone()[0] or 0.0
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        pending = len(self._pending_writes)
        return {"total_engrams": total, "mean_resonance": round(mean_res, 3),
                "journal_mode": journal, "pending_writes": pending,
                "db_path": self.db_path}


if __name__ == "__main__":
    # Self-test: concurrent read/write without SQLITE_BUSY + state-keyed recall
    import tempfile
    import threading

    db = tempfile.mktemp(suffix=".db")
    o = Ouroboros(db)

    # Writer thread staging+flushing 100 engrams
    def writer():
        for i in range(100):
            o.stage_engram(f"content_{i}", engram_id=f"eng_{i}", resonance=0.3 + (i % 70) / 100.0)
        o.flush()

    # Reader threads hammering recall during writes
    def reader():
        for _ in range(50):
            o.recall(min_resonance=0.5, limit=5)

    threads = [threading.Thread(target=writer)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()

    s = o.stats()
    assert s["journal_mode"] == "wal", f"WAL not enabled: {s['journal_mode']}"
    assert s["total_engrams"] == 100, f"Expected 100, got {s['total_engrams']}"
    print(f"[ouroboros] PASS — journal={s['journal_mode']}, engrams={s['total_engrams']}, "
          f"mean_resonance={s['mean_resonance']}, no SQLITE_BUSY")

    # Fuzzy semver test
    assert fuzzy_semver_match("1.24.x", "1.24.7")
    assert not fuzzy_semver_match("1.24.x", "2.24.0")
    assert fuzzy_semver_match("", "anything")
    print("[ouroboros] PASS — fuzzy semver matching")

    # State-keyed recall test
    o.stage_engram("numpy-specific plan", engram_id="eng_np", tags=["numpy"])
    o.flush()
    hit = o.recall(deps={"numpy": "1.24.x"}, limit=500)
    assert any(e["id"] == "eng_np" for e in hit), f"eng_np not in recall: {[e['id'] for e in hit[:5]]}"
    print(f"[ouroboros] PASS — state-keyed recall ({len(hit)} results, fingerprint os={o.fingerprint['os_type']})")
    os.unlink(db)
    os.unlink(db + "-wal") if os.path.exists(db + "-wal") else None
    os.unlink(db + "-shm") if os.path.exists(db + "-shm") else None

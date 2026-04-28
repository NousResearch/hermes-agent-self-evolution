"""SQLite persistence for evolution runs and artifacts."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EvolutionStore:
    """Small SQLite persistence layer for the self-evolution pipeline."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def add_repository(
        self,
        name: str,
        local_path: str | Path,
        url: str | None = None,
        default_branch: str = "main",
    ) -> dict[str, Any]:
        existing = self.get_repository(name)
        now = _now()
        if existing:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE repositories
                    SET url = ?, local_path = ?, default_branch = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (url, str(local_path), default_branch, now, existing["id"]),
                )
            return self.get_repository(name)

        repo_id = _new_id("repo")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO repositories (id, name, url, local_path, default_branch, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (repo_id, name, url, str(local_path), default_branch, now, now),
            )
        return self.get_repository(name)

    def get_repository(self, name: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM repositories WHERE name = ?", (name,))

    def get_repository_by_id(self, repository_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM repositories WHERE id = ?", (repository_id,))

    def list_repositories(self) -> list[dict[str, Any]]:
        return self._fetch_all("SELECT * FROM repositories ORDER BY name")

    def add_repo_snapshot(
        self,
        repository_id: str,
        git_sha: str,
        branch: str,
        dirty: bool,
        diff_sha256: str | None = None,
    ) -> dict[str, Any]:
        snapshot_id = _new_id("snap")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO repo_snapshots (id, repository_id, git_sha, branch, dirty, diff_sha256, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (snapshot_id, repository_id, git_sha, branch, int(dirty), diff_sha256, _now()),
            )
        return self._fetch_one("SELECT * FROM repo_snapshots WHERE id = ?", (snapshot_id,))

    def upsert_target(
        self,
        repository_id: str,
        target_type: str,
        name: str,
        file_path: str,
        selector: str | None = None,
        metadata: dict[str, Any] | None = None,
        baseline_artifact_id: str | None = None,
    ) -> dict[str, Any]:
        existing = self._fetch_one(
            """
            SELECT * FROM targets
            WHERE repository_id = ? AND target_type = ? AND name = ?
            """,
            (repository_id, target_type, name),
        )
        metadata_text = json.dumps(metadata or {}, sort_keys=True)
        if existing:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE targets
                    SET file_path = ?, selector = ?, baseline_artifact_id = ?, metadata_json = ?
                    WHERE id = ?
                    """,
                    (file_path, selector, baseline_artifact_id, metadata_text, existing["id"]),
                )
            return self._fetch_one("SELECT * FROM targets WHERE id = ?", (existing["id"],))

        target_id = _new_id("target")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO targets (
                    id, repository_id, target_type, name, file_path, selector,
                    baseline_artifact_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_id,
                    repository_id,
                    target_type,
                    name,
                    file_path,
                    selector,
                    baseline_artifact_id,
                    metadata_text,
                    _now(),
                ),
            )
        return self._fetch_one("SELECT * FROM targets WHERE id = ?", (target_id,))

    def list_targets(self, repository_id: str | None = None, target_type: str | None = None) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if repository_id:
            clauses.append("repository_id = ?")
            params.append(repository_id)
        if target_type:
            clauses.append("target_type = ?")
            params.append(target_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._fetch_all(f"SELECT * FROM targets {where} ORDER BY target_type, name", tuple(params))

    def get_target(self, target_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM targets WHERE id = ?", (target_id,))

    def get_target_by_name(self, target_type: str, name: str) -> dict[str, Any] | None:
        return self._fetch_one(
            "SELECT * FROM targets WHERE target_type = ? AND name = ? ORDER BY created_at DESC LIMIT 1",
            (target_type, name),
        )

    def add_artifact(
        self,
        kind: str,
        content_sha256: str,
        storage_uri: str,
        size_bytes: int,
        target_id: str | None = None,
        mime_type: str | None = None,
        parent_artifact_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self._fetch_one("SELECT * FROM artifacts WHERE content_sha256 = ?", (content_sha256,))
        if existing:
            return existing

        artifact_id = _new_id("artifact")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    id, target_id, kind, content_sha256, storage_uri, size_bytes,
                    mime_type, parent_artifact_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    target_id,
                    kind,
                    content_sha256,
                    storage_uri,
                    size_bytes,
                    mime_type,
                    parent_artifact_id,
                    json.dumps(metadata or {}, sort_keys=True),
                    _now(),
                ),
            )
        return self.get_artifact(artifact_id)

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM artifacts WHERE id = ?", (artifact_id,))

    def add_dataset(
        self,
        target_id: str,
        source: str,
        version: str,
        artifact_id: str | None,
        split_spec: dict[str, Any],
        pii_scan_status: str,
        secret_scan_status: str,
        example_count: int,
    ) -> dict[str, Any]:
        dataset_id = _new_id("dataset")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets (
                    id, target_id, source, version, artifact_id, split_spec_json,
                    pii_scan_status, secret_scan_status, example_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    target_id,
                    source,
                    version,
                    artifact_id,
                    json.dumps(split_spec, sort_keys=True),
                    pii_scan_status,
                    secret_scan_status,
                    example_count,
                    _now(),
                ),
            )
        return self.get_dataset(dataset_id)

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))

    def list_datasets(self, target_id: str | None = None) -> list[dict[str, Any]]:
        if target_id:
            return self._fetch_all(
                "SELECT * FROM datasets WHERE target_id = ? ORDER BY created_at DESC",
                (target_id,),
            )
        return self._fetch_all("SELECT * FROM datasets ORDER BY created_at DESC")

    def add_eval_example(
        self,
        dataset_id: str,
        split: str,
        task_input: str,
        expected_behavior: str,
        difficulty: str | None = None,
        category: str | None = None,
        source: str | None = None,
        source_ref_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        example_id = _new_id("example")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO eval_examples (
                    id, dataset_id, split, source, task_input, expected_behavior,
                    difficulty, category, source_ref_hash, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    example_id,
                    dataset_id,
                    split,
                    source,
                    task_input,
                    expected_behavior,
                    difficulty,
                    category,
                    source_ref_hash,
                    json.dumps(metadata or {}, sort_keys=True),
                    _now(),
                ),
            )
        return self._fetch_one("SELECT * FROM eval_examples WHERE id = ?", (example_id,))

    def list_eval_examples(self, dataset_id: str, split: str | None = None) -> list[dict[str, Any]]:
        if split:
            return self._fetch_all(
                "SELECT * FROM eval_examples WHERE dataset_id = ? AND split = ? ORDER BY created_at",
                (dataset_id, split),
            )
        return self._fetch_all(
            "SELECT * FROM eval_examples WHERE dataset_id = ? ORDER BY created_at",
            (dataset_id,),
        )

    def add_candidate(
        self,
        run_id: str,
        target_id: str,
        role: str,
        artifact_id: str,
        content_sha256: str,
        parent_candidate_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate_id = _new_id("candidate")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO candidates (
                    id, run_id, target_id, role, artifact_id, content_sha256,
                    parent_candidate_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    run_id,
                    target_id,
                    role,
                    artifact_id,
                    content_sha256,
                    parent_candidate_id,
                    json.dumps(metadata or {}, sort_keys=True),
                    _now(),
                ),
            )
        return self._fetch_one("SELECT * FROM candidates WHERE id = ?", (candidate_id,))

    def list_candidates(self, run_id: str) -> list[dict[str, Any]]:
        return self._fetch_all(
            "SELECT * FROM candidates WHERE run_id = ? ORDER BY created_at, id",
            (run_id,),
        )

    def add_evaluation(
        self,
        run_id: str,
        candidate_id: str,
        dataset_id: str,
        split: str,
        example_id: str | None,
        metric_name: str,
        score: float,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evaluation_id = _new_id("eval")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evaluations (
                    id, run_id, candidate_id, dataset_id, split, example_id,
                    metric_name, score, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    run_id,
                    candidate_id,
                    dataset_id,
                    split,
                    example_id,
                    metric_name,
                    float(score),
                    json.dumps(details or {}, sort_keys=True),
                    _now(),
                ),
            )
        return self._fetch_one("SELECT * FROM evaluations WHERE id = ?", (evaluation_id,))

    def list_evaluations(self, run_id: str, candidate_id: str | None = None) -> list[dict[str, Any]]:
        if candidate_id:
            return self._fetch_all(
                """
                SELECT * FROM evaluations
                WHERE run_id = ? AND candidate_id = ?
                ORDER BY created_at, id
                """,
                (run_id, candidate_id),
            )
        return self._fetch_all(
            "SELECT * FROM evaluations WHERE run_id = ? ORDER BY created_at, id",
            (run_id,),
        )

    def add_run_event(
        self,
        run_id: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_id = _new_id("event")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_events (id, run_id, event_type, message, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, run_id, event_type, message, json.dumps(metadata or {}, sort_keys=True), _now()),
            )
        return self._fetch_one("SELECT * FROM run_events WHERE id = ?", (event_id,))

    def list_run_events(self, run_id: str) -> list[dict[str, Any]]:
        return self._fetch_all(
            "SELECT * FROM run_events WHERE run_id = ? ORDER BY created_at, id",
            (run_id,),
        )

    def add_gate_result(
        self,
        run_id: str,
        candidate_id: str,
        decision: str,
        reasons: list[str],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        gate_id = _new_id("gate")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO gate_results (
                    id, run_id, candidate_id, decision, reasons_json, metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gate_id,
                    run_id,
                    candidate_id,
                    decision,
                    json.dumps(reasons, sort_keys=True),
                    json.dumps(metrics, sort_keys=True),
                    _now(),
                ),
            )
        return self._fetch_one("SELECT * FROM gate_results WHERE id = ?", (gate_id,))

    def list_gate_results(self, run_id: str) -> list[dict[str, Any]]:
        return self._fetch_all(
            "SELECT * FROM gate_results WHERE run_id = ? ORDER BY created_at DESC, id DESC",
            (run_id,),
        )

    def create_run(
        self,
        target_id: str,
        engine: str,
        config: dict[str, Any],
        repository_snapshot_id: str | None = None,
        baseline_artifact_id: str | None = None,
        dataset_id: str | None = None,
        seed: int | None = None,
        self_evolution_git_sha: str | None = None,
        status: str = "pending",
    ) -> dict[str, Any]:
        run_id = _new_id("run")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    id, target_id, repository_snapshot_id, baseline_artifact_id,
                    dataset_id, engine, status, config_json, seed,
                    self_evolution_git_sha, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    target_id,
                    repository_snapshot_id,
                    baseline_artifact_id,
                    dataset_id,
                    engine,
                    status,
                    json.dumps(config, sort_keys=True),
                    seed,
                    self_evolution_git_sha,
                    _now(),
                ),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._fetch_one("SELECT * FROM runs WHERE id = ?", (run_id,))

    def update_run_status(
        self,
        run_id: str,
        status: str,
        error: str | None = None,
        completed: bool = False,
        cost_usd: float | None = None,
    ) -> dict[str, Any]:
        completed_at = _now() if completed else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, completed_at = COALESCE(?, completed_at), error = ?, cost_usd = COALESCE(?, cost_usd)
                WHERE id = ?
                """,
                (status, completed_at, error, cost_usd, run_id),
            )
        updated = self.get_run(run_id)
        if not updated:
            raise ValueError(f"Run not found: {run_id}")
        return updated

    def list_runs(self) -> list[dict[str, Any]]:
        return self._fetch_all("SELECT * FROM runs ORDER BY started_at DESC")

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return _row_to_dict(row) if row else None

    def _fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_dict(row) for row in rows]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in list(data):
        if key.endswith("_json") and isinstance(data[key], str):
            data[key] = json.loads(data[key])
    if "dirty" in data and data["dirty"] is not None:
        data["dirty"] = bool(data["dirty"])
    return data


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS repositories (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    url TEXT,
    local_path TEXT NOT NULL,
    default_branch TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repo_snapshots (
    id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    git_sha TEXT NOT NULL,
    branch TEXT,
    dirty INTEGER NOT NULL DEFAULT 0,
    diff_sha256 TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    target_id TEXT,
    kind TEXT NOT NULL,
    content_sha256 TEXT NOT NULL UNIQUE,
    storage_uri TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mime_type TEXT,
    parent_artifact_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS targets (
    id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    target_type TEXT NOT NULL,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    selector TEXT,
    baseline_artifact_id TEXT REFERENCES artifacts(id),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(repository_id, target_type, name)
);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL REFERENCES targets(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    version TEXT NOT NULL,
    artifact_id TEXT REFERENCES artifacts(id),
    split_spec_json TEXT NOT NULL DEFAULT '{}',
    pii_scan_status TEXT NOT NULL,
    secret_scan_status TEXT NOT NULL,
    example_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_examples (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    split TEXT NOT NULL,
    source TEXT,
    task_input TEXT NOT NULL,
    expected_behavior TEXT NOT NULL,
    difficulty TEXT,
    category TEXT,
    source_ref_hash TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL REFERENCES targets(id) ON DELETE CASCADE,
    repository_snapshot_id TEXT REFERENCES repo_snapshots(id),
    baseline_artifact_id TEXT REFERENCES artifacts(id),
    dataset_id TEXT,
    engine TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    seed INTEGER,
    self_evolution_git_sha TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    cost_usd REAL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES targets(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    content_sha256 TEXT NOT NULL,
    parent_candidate_id TEXT REFERENCES candidates(id),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    split TEXT NOT NULL,
    example_id TEXT REFERENCES eval_examples(id) ON DELETE SET NULL,
    metric_name TEXT NOT NULL,
    score REAL NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gate_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    decision TEXT NOT NULL,
    reasons_json TEXT NOT NULL DEFAULT '[]',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""

"""Tests for evolution run history (no I/O beyond tmp_path)."""

import json
from evolution.loop.history import RunRecord, EvolutionHistory, load_history, append_run


def test_run_record_defaults():
    r = RunRecord(run_id="test_001", phases={})
    assert r.run_id == "test_001"
    assert r.git_sha == ""
    assert r.elapsed_seconds == 0


def test_run_record_roundtrip():
    r = RunRecord(
        run_id="test_001", timestamp="2026-06-22T12:00:00",
        phases={"skills": {"status": "success", "improvement": 0.15}},
        git_sha="abc123", elapsed_seconds=42,
    )
    restored = RunRecord.from_dict(r.to_dict())
    assert restored.run_id == r.run_id
    assert restored.phases["skills"]["improvement"] == 0.15


def test_load_empty_history(tmp_path):
    history = load_history(tmp_path / "nonexistent" / "history.jsonl")
    assert history.runs == []


def test_append_and_load(tmp_path):
    history_file = tmp_path / "history.jsonl"
    r = RunRecord(run_id="test_001", phases={"skills": {"status": "success"}})
    append_run(history_file, r)
    loaded = load_history(history_file)
    assert len(loaded.runs) == 1
    assert loaded.runs[0].run_id == "test_001"


def test_append_multiple(tmp_path):
    history_file = tmp_path / "history.jsonl"
    for i in range(3):
        append_run(history_file, RunRecord(run_id=f"test_{i:03d}", phases={}))
    loaded = load_history(history_file)
    assert len(loaded.runs) == 3
    assert loaded.runs[-1].run_id == "test_002"


def test_latest_run_per_phase(tmp_path):
    history_file = tmp_path / "history.jsonl"
    append_run(history_file, RunRecord(
        run_id="r1", phases={"skills": {"status": "success"}, "tools": {"status": "failed"}},
    ))
    append_run(history_file, RunRecord(
        run_id="r2", phases={"tools": {"status": "success"}},
    ))
    loaded = load_history(history_file)
    latest = loaded.latest_per_phase()
    assert latest["skills"].run_id == "r1"
    assert latest["tools"].run_id == "r2"

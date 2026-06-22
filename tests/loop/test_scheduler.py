"""Tests for evolution scheduler (no I/O)."""

from evolution.loop.scheduler import (
    select_pending_phases, PHASE_ORDER, PHASE_DEPENDENCIES,
)


def test_phase_order_is_complete():
    assert len(PHASE_ORDER) == 4
    assert PHASE_ORDER == ["skills", "tools", "guidance", "params"]


def test_no_pending_when_all_recent():
    latest = {"skills": "r2", "tools": "r2", "guidance": "r2", "params": "r2"}
    pending = select_pending_phases(latest, force=False)
    assert pending == []


def test_all_pending_when_force():
    latest = {"skills": "r2", "tools": "r2", "guidance": "r2", "params": "r2"}
    pending = select_pending_phases(latest, force=True)
    assert pending == PHASE_ORDER


def test_skipped_phases_are_pending():
    latest = {"skills": "r1"}
    pending = select_pending_phases(latest, force=False)
    assert "tools" in pending
    assert "guidance" in pending
    assert "params" in pending


def test_empty_history_all_pending():
    pending = select_pending_phases({}, force=False)
    assert pending == PHASE_ORDER


def test_subset_filter():
    latest = {}
    pending = select_pending_phases(latest, force=False, only=["skills", "tools"])
    assert pending == ["skills", "tools"]
    assert "guidance" not in pending


def test_dependencies_present():
    assert "params" in PHASE_DEPENDENCIES
    assert "tools" in PHASE_DEPENDENCIES["params"]

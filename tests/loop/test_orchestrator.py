"""Tests for cross-phase orchestrator (all phase functions mocked)."""

from unittest.mock import MagicMock
import pytest
from evolution.loop.orchestrator import run_phases, PhaseResult, OrchestrationResult


def test_phase_result_defaults():
    r = PhaseResult(name="skills", status="success", improvement=0.1)
    assert r.reverted == []
    assert r.elapsed == 0


def test_phase_result_to_dict():
    r = PhaseResult(name="skills", status="success", improvement=0.1, reverted=["tool_a"])
    d = r.to_dict()
    assert d["name"] == "skills"
    assert d["reverted"] == ["tool_a"]


def test_orchestrator_runs_ordered_phases():
    calls = []
    phase_funcs = {
        "skills": MagicMock(side_effect=lambda: calls.append("skills") or {"evolved_score": 0.9, "baseline_score": 0.8}),
        "tools": MagicMock(side_effect=lambda: calls.append("tools") or {"evolved_score": 0.85, "baseline_score": 0.8}),
    }
    result = run_phases(phase_funcs, phases=["skills", "tools"])
    assert calls == ["skills", "tools"]
    assert result.status == "success"


def test_orchestrator_handles_subset():
    phase_funcs = {
        "skills": MagicMock(return_value={"improvement": 0.1, "evolved_score": 0.9, "baseline_score": 0.8}),
    }
    result = run_phases(phase_funcs, phases=["skills"])
    assert result.status == "success"
    assert result.phases["skills"]["status"] == "success"


def test_orchestrator_skips_unregistered():
    result = run_phases({}, phases=["skills"])
    assert result.phases["skills"]["status"] == "skipped"


def test_orchestrator_handles_dry_run():
    phase_funcs = {"skills": MagicMock()}
    result = run_phases(phase_funcs, phases=["skills"], dry_run=True)
    assert result.phases["skills"]["status"] == "skipped"
    phase_funcs["skills"].assert_not_called()


def test_orchestrator_stops_on_phase_failure():
    failed = MagicMock(side_effect=RuntimeError("phase failed"))
    phase_funcs = {"skills": failed, "tools": MagicMock(return_value={"evolved_score": 0.9, "baseline_score": 0.8})}
    result = run_phases(phase_funcs, phases=["skills", "tools"])
    assert result.phases["skills"]["status"] == "failed"
    assert result.phases["tools"]["status"] == "success"


def test_orchestration_result_metrics():
    r = OrchestrationResult(phases={
        "skills": PhaseResult(name="skills", status="success", improvement=0.1).to_dict(),
        "tools": PhaseResult(name="tools", status="success", improvement=0.05).to_dict(),
    })
    metrics = r.summary_metrics()
    assert metrics["total_improvement"] == pytest.approx(0.15)
    assert metrics["phases_run"] == 2
    assert metrics["phases_failed"] == 0


def test_orchestration_result_metrics_with_failures():
    r = OrchestrationResult(phases={
        "skills": PhaseResult(name="skills", status="success", improvement=0.1).to_dict(),
        "tools": PhaseResult(name="tools", status="failed").to_dict(),
        "guidance": PhaseResult(name="guidance", status="skipped").to_dict(),
    })
    metrics = r.summary_metrics()
    assert metrics["phases_failed"] == 1
    assert metrics["phases_skipped"] == 1

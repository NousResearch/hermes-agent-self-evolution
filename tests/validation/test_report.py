"""Tests for evolution.validation.report — scoring, decision rule, JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution.validation.agent_runner import AgentRunResult
from evolution.validation.report import (
    PhaseResult,
    TaskResult,
    WinLoss,
    ValidationReport,
    compute_win_loss,
    decide,
    score_task,
    summarize_phase,
)


def _tr(task_id: str, *, passed: bool, abstained: bool = False) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        passed=passed,
        abstained=abstained,
        tool_calls_seq=["patch"] if passed else ["write_file"],
        duration_seconds=1.0,
    )


class TestScoreTask:
    def test_expected_in_invocations_passes(self):
        run = AgentRunResult(tool_calls_seq=["patch"], final_text_tail="", duration_seconds=1.0)
        passed, abstained = score_task(
            expected_tools=("patch",), forbidden_tools=("write_file",), run=run,
        )
        assert passed and not abstained

    def test_forbidden_in_invocations_fails(self):
        run = AgentRunResult(
            tool_calls_seq=["patch", "write_file"], final_text_tail="", duration_seconds=1.0,
        )
        passed, abstained = score_task(
            expected_tools=("patch",), forbidden_tools=("write_file",), run=run,
        )
        assert not passed and not abstained

    def test_expected_absent_fails(self):
        run = AgentRunResult(
            tool_calls_seq=["read_file"], final_text_tail="", duration_seconds=1.0,
        )
        passed, abstained = score_task(
            expected_tools=("patch",), forbidden_tools=(), run=run,
        )
        assert not passed and not abstained

    def test_error_marks_abstention(self):
        run = AgentRunResult(
            tool_calls_seq=[], final_text_tail="", duration_seconds=1.0,
            error="hermes timed out",
        )
        passed, abstained = score_task(
            expected_tools=("patch",), forbidden_tools=("write_file",), run=run,
        )
        assert not passed
        assert abstained


class TestSummarizePhase:
    def test_counts_and_pass_rate(self):
        results = [
            _tr("a", passed=True),
            _tr("b", passed=True),
            _tr("c", passed=False),
            _tr("d", passed=False, abstained=True),
        ]
        phase = summarize_phase(results)
        assert phase.n_passed == 2
        assert phase.n_failed == 1
        assert phase.n_abstained == 1
        # 2 scored, 2 passed → 1.0? wait: 2 passed, 1 failed, 1 abstained → 2/3
        assert phase.pass_rate == pytest.approx(2 / 3)

    def test_all_abstentions_yields_zero_pass_rate(self):
        results = [_tr("a", passed=False, abstained=True)]
        phase = summarize_phase(results)
        assert phase.pass_rate == 0.0


class TestComputeWinLoss:
    def _phase(self, *outcomes: tuple[str, bool]) -> PhaseResult:
        results = [_tr(task_id, passed=passed) for task_id, passed in outcomes]
        return summarize_phase(results)

    def test_wins_when_evolved_passes_baseline_fails(self):
        baseline = self._phase(("a", False), ("b", True))
        evolved = self._phase(("a", True), ("b", True))
        wl = compute_win_loss(baseline, evolved)
        assert wl.n_wins == 1
        assert wl.n_losses == 0
        assert wl.n_ties == 1

    def test_losses_when_baseline_passes_evolved_fails(self):
        baseline = self._phase(("a", True), ("b", True))
        evolved = self._phase(("a", False), ("b", True))
        wl = compute_win_loss(baseline, evolved)
        assert wl.n_losses == 1
        assert wl.n_wins == 0

    def test_abstentions_count_as_ties(self):
        b_results = [_tr("a", passed=True, abstained=False), _tr("b", passed=True)]
        e_results = [_tr("a", passed=False, abstained=True), _tr("b", passed=True)]
        wl = compute_win_loss(summarize_phase(b_results), summarize_phase(e_results))
        assert wl.n_wins == 0 and wl.n_losses == 0
        assert wl.n_ties == 2


class TestDecisionRule:
    """Two-condition rule: pass-rate no-regression AND no per-task regression
    unless wins are 2x losses."""

    def _phases(self, baseline_passes: int, evolved_passes: int, *, n_tasks: int = 5):
        b = summarize_phase([_tr(f"t{i}", passed=(i < baseline_passes)) for i in range(n_tasks)])
        e = summarize_phase([_tr(f"t{i}", passed=(i < evolved_passes)) for i in range(n_tasks)])
        return b, e

    def test_pass_when_evolved_strictly_better(self):
        # baseline 3/5, evolved 4/5 (same tasks): wins=1, losses=0
        b_results = [_tr("t0", passed=True), _tr("t1", passed=True), _tr("t2", passed=True),
                     _tr("t3", passed=False), _tr("t4", passed=False)]
        e_results = [_tr("t0", passed=True), _tr("t1", passed=True), _tr("t2", passed=True),
                     _tr("t3", passed=True), _tr("t4", passed=False)]
        b = summarize_phase(b_results)
        e = summarize_phase(e_results)
        wl = compute_win_loss(b, e)
        decision, _ = decide(b, e, wl)
        assert decision == "pass"

    def test_regression_when_aggregate_drops(self):
        b = summarize_phase([_tr(f"t{i}", passed=(i < 4)) for i in range(5)])
        e = summarize_phase([_tr(f"t{i}", passed=(i < 2)) for i in range(5)])
        wl = compute_win_loss(b, e)
        decision, _ = decide(b, e, wl)
        assert decision == "regression"

    def test_regression_when_one_loss_one_win_equal_pass_rate(self):
        # Aggregate identical (3 vs 3) but distinct per-task outcomes:
        # losing task "t0" and winning task "t4" — aggregate ties, but
        # n_losses = 1, n_wins = 1, n_wins (1) < 2 * n_losses (2). REJECT.
        b_results = [_tr("t0", passed=True),  _tr("t1", passed=True),
                     _tr("t2", passed=True),  _tr("t3", passed=False),
                     _tr("t4", passed=False)]
        e_results = [_tr("t0", passed=False), _tr("t1", passed=True),
                     _tr("t2", passed=True),  _tr("t3", passed=False),
                     _tr("t4", passed=True)]
        b = summarize_phase(b_results)
        e = summarize_phase(e_results)
        wl = compute_win_loss(b, e)
        assert wl.n_wins == 1 and wl.n_losses == 1
        decision, reasons = decide(b, e, wl)
        assert decision == "regression"
        assert any("per-task regression" in r for r in reasons)

    def test_pass_when_two_wins_offset_one_loss(self):
        # n_wins = 2, n_losses = 1 → wins >= 2 * losses, OK.
        b_results = [_tr("t0", passed=True),  _tr("t1", passed=False),
                     _tr("t2", passed=False), _tr("t3", passed=True),
                     _tr("t4", passed=False)]
        e_results = [_tr("t0", passed=False), _tr("t1", passed=True),
                     _tr("t2", passed=True),  _tr("t3", passed=True),
                     _tr("t4", passed=False)]
        b = summarize_phase(b_results)
        e = summarize_phase(e_results)
        wl = compute_win_loss(b, e)
        assert wl.n_wins == 2 and wl.n_losses == 1
        decision, _ = decide(b, e, wl)
        assert decision == "pass"

    def test_tie_decides_pass(self):
        # Same results both sides → all ties → pass.
        results = [_tr(f"t{i}", passed=(i < 3)) for i in range(5)]
        b = summarize_phase(results)
        e = summarize_phase(results)
        wl = compute_win_loss(b, e)
        decision, _ = decide(b, e, wl)
        assert decision == "pass"


class TestValidationReportJson:
    def test_to_dict_round_trips_through_json(self, tmp_path):
        b = summarize_phase([_tr("t1", passed=True), _tr("t2", passed=False)])
        e = summarize_phase([_tr("t1", passed=True), _tr("t2", passed=True)])
        wl = compute_win_loss(b, e)
        decision, reasons = decide(b, e, wl)
        report = ValidationReport(
            schema_version="1",
            tool="patch",
            task_suite_path="suite.jsonl",
            task_suite_sha256="abc123",
            baseline=b,
            evolved=e,
            delta=wl,
            decision=decision,
            decision_reasons=reasons,
        )
        p = tmp_path / "report.json"
        report.write_json(p)
        loaded = json.loads(p.read_text())
        assert loaded["tool"] == "patch"
        assert loaded["task_suite_sha256"] == "abc123"
        assert loaded["decision"] == "pass"
        assert loaded["baseline"]["pass_rate"] == 0.5
        assert loaded["evolved"]["pass_rate"] == 1.0
        assert loaded["delta"]["n_wins"] == 1
        assert loaded["delta"]["n_losses"] == 0

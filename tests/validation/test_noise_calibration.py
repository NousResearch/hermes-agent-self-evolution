"""A/A noise-floor probe: aggregation math + the calibrate_noise loop."""
import json
from pathlib import Path

import pytest

from evolution.validation.noise_calibration import (
    NOISE_SIDECAR_SUFFIX,
    aggregate_noise,
    calibrate_noise,
    load_noise_sidecar,
    noise_sidecar_path,
    write_noise_sidecar,
)
from evolution.validation.report import (
    PhaseResult,
    TaskResult,
    ValidationReport,
    WinLoss,
)


def _task(task_id, passed, abstained=False):
    return TaskResult(
        task_id=task_id, passed=passed, abstained=abstained,
        tool_calls_seq=[], duration_seconds=0.1,
    )


def _phase(tasks):
    n_pass = sum(1 for t in tasks if t.passed and not t.abstained)
    n_fail = sum(1 for t in tasks if not t.passed and not t.abstained)
    n_abst = sum(1 for t in tasks if t.abstained)
    scored = n_pass + n_fail
    return PhaseResult(
        pass_rate=(n_pass / scored) if scored else 0.0,
        n_passed=n_pass, n_failed=n_fail, n_abstained=n_abst, tasks=tasks,
    )


def _report(baseline_tasks, evolved_tasks, *, n_wins, n_losses, decision):
    b, e = _phase(baseline_tasks), _phase(evolved_tasks)
    return ValidationReport(
        schema_version="x", tool="noise", task_suite_path="s", task_suite_sha256="sha",
        baseline=b, evolved=e,
        delta=WinLoss(n_wins=n_wins, n_losses=n_losses, n_ties=0,
                      pass_rate_change=e.pass_rate - b.pass_rate),
        decision=decision, decision_reasons=[],
    )


def test_aggregate_counts_spurious_wins_and_regressions():
    reports = [
        _report([_task("a", True)], [_task("a", True)], n_wins=0, n_losses=0, decision="pass"),
        _report([_task("a", False)], [_task("a", True)], n_wins=1, n_losses=0, decision="pass"),
        _report([_task("a", True)], [_task("a", False)], n_wins=0, n_losses=1, decision="regression"),
        _report([_task("a", True)], [_task("a", True)], n_wins=0, n_losses=0, decision="pass"),
    ]
    rpt = aggregate_noise(reports, reps=1, suite_sha256="sha", agent_model="haiku")
    assert rpt.runs == 4
    assert rpt.spurious_strict_win_rate == pytest.approx(0.25)  # 1 of 4 had a win
    assert rpt.spurious_regression_rate == pytest.approx(0.25)  # 1 of 4 regression
    assert rpt.reps == 1 and rpt.agent_model == "haiku"


def test_aggregate_per_task_flip_pools_both_phases():
    # task "stable" always passes (flip 0); task "flaky" passes half the time.
    # Pooled over 2 runs x 2 phases = 4 verdicts each.
    reports = [
        _report([_task("stable", True), _task("flaky", True)],
                [_task("stable", True), _task("flaky", False)],
                n_wins=0, n_losses=1, decision="regression"),
        _report([_task("stable", True), _task("flaky", False)],
                [_task("stable", True), _task("flaky", True)],
                n_wins=1, n_losses=0, decision="pass"),
    ]
    rpt = aggregate_noise(reports, reps=1, suite_sha256="sha")
    assert rpt.per_task_flip["stable"] == pytest.approx(0.0)
    assert rpt.per_task_flip["flaky"] == pytest.approx(0.5)  # 2 pass / 2 fail
    assert rpt.mean_per_task_flip == pytest.approx(0.25)


def test_aggregate_excludes_abstentions_from_flip():
    reports = [
        _report([_task("a", False, abstained=True)], [_task("a", True)],
                n_wins=0, n_losses=0, decision="pass"),
    ]
    rpt = aggregate_noise(reports, reps=1, suite_sha256="sha")
    # Only the evolved (non-abstained) verdict counts → single True → flip 0.
    assert rpt.per_task_flip["a"] == pytest.approx(0.0)


class _StubValidator:
    """Returns canned reports in sequence; records the install slots seen."""
    reps = 2

    def __init__(self, reports):
        self._reports = list(reports)
        self.calls = []

    def validate(self, inputs):
        self.calls.append((inputs.baseline_artifact, inputs.evolved_artifact))
        return self._reports.pop(0)


def test_calibrate_noise_runs_aa_and_aggregates(tmp_path):
    artifact = tmp_path / "art.txt"
    artifact.write_text("seed")
    from evolution.validation.task import TaskSuite
    suite = TaskSuite(path=tmp_path / "s.jsonl", sha256="deadbeef", tasks=())
    stub = _StubValidator([
        _report([_task("a", True)], [_task("a", True)], n_wins=0, n_losses=0, decision="pass"),
        _report([_task("a", False)], [_task("a", True)], n_wins=1, n_losses=0, decision="pass"),
    ])
    rpt = calibrate_noise(stub, suite, artifact, runs=2, agent_model="haiku")
    assert rpt.runs == 2
    assert rpt.reps == 2  # taken from validator.reps
    assert rpt.suite_sha256 == "deadbeef"
    assert rpt.spurious_strict_win_rate == pytest.approx(0.5)
    # A/A: every call installs the same artifact in both slots.
    assert all(b == e == artifact for b, e in stub.calls)


def test_calibrate_noise_rejects_zero_runs(tmp_path):
    from evolution.validation.task import TaskSuite
    suite = TaskSuite(path=tmp_path / "s.jsonl", sha256="x", tasks=())
    with pytest.raises(ValueError, match="runs must be"):
        calibrate_noise(_StubValidator([]), suite, tmp_path / "a", runs=0)


def test_sidecar_roundtrip(tmp_path):
    suite_path = tmp_path / "claude_conventions.jsonl"
    rpt = aggregate_noise(
        [_report([_task("a", True)], [_task("a", True)], n_wins=0, n_losses=0, decision="pass")],
        reps=1, suite_sha256="sha",
    )
    out = write_noise_sidecar(rpt, suite_path)
    assert out == suite_path.with_name("claude_conventions.jsonl" + NOISE_SIDECAR_SUFFIX)
    loaded = load_noise_sidecar(suite_path)
    assert loaded["suite_sha256"] == "sha"
    assert loaded["runs"] == 1


def test_load_sidecar_absent_returns_none(tmp_path):
    assert load_noise_sidecar(tmp_path / "missing.jsonl") is None

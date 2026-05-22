"""Tests for ClosedLoopFeedbackCache + render_feedback_block.

All tests mock the validator — zero real LM spend. The cache is tested
in isolation from GEPA; metric-integration tests live in
``tests/core/test_fitness_closed_loop.py``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evolution.core.closed_loop_feedback import (
    ClosedLoopFeedbackCache,
    render_feedback_block,
)
from evolution.validation.report import (
    PhaseResult,
    TaskResult,
    ValidationReport,
    WinLoss,
)
from evolution.validation.task import TaskSuite
from evolution.validation.validator import ConcurrentRunError


def _build_suite(tmp_path: Path) -> TaskSuite:
    path = tmp_path / "suite.jsonl"
    path.write_text(
        '{"task_id": "t1", "user_message": "do X", "expected_tools": ["write_file"]}\n'
        '{"task_id": "t2", "user_message": "do Y", "expected_tools": ["patch"]}\n'
    )
    return TaskSuite.from_jsonl(path)


def _build_report(
    *,
    decision: str = "pass",
    pass_rate_change: float = 0.0,
    baseline_tasks: list[TaskResult] | None = None,
    evolved_tasks: list[TaskResult] | None = None,
    n_wins: int = 0,
    n_losses: int = 0,
    n_ties: int = 2,
    decision_reasons: list[str] | None = None,
    suite_path: str = "",
) -> ValidationReport:
    baseline_tasks = baseline_tasks or [
        TaskResult("t1", True, False, ["write_file"], 1.0, "stub", None),
        TaskResult("t2", True, False, ["patch"], 1.0, "stub", None),
    ]
    evolved_tasks = evolved_tasks or list(baseline_tasks)
    baseline = PhaseResult(
        pass_rate=sum(1 for t in baseline_tasks if t.passed) / len(baseline_tasks),
        n_passed=sum(1 for t in baseline_tasks if t.passed),
        n_failed=sum(1 for t in baseline_tasks if not t.passed and not t.abstained),
        n_abstained=sum(1 for t in baseline_tasks if t.abstained),
        tasks=baseline_tasks,
    )
    evolved = PhaseResult(
        pass_rate=sum(1 for t in evolved_tasks if t.passed) / len(evolved_tasks),
        n_passed=sum(1 for t in evolved_tasks if t.passed),
        n_failed=sum(1 for t in evolved_tasks if not t.passed and not t.abstained),
        n_abstained=sum(1 for t in evolved_tasks if t.abstained),
        tasks=evolved_tasks,
    )
    delta = WinLoss(
        n_wins=n_wins, n_losses=n_losses, n_ties=n_ties, pass_rate_change=pass_rate_change
    )
    return ValidationReport(
        schema_version="1",
        tool="write_file",
        task_suite_path=suite_path,
        task_suite_sha256="abc",
        baseline=baseline,
        evolved=evolved,
        delta=delta,
        decision=decision,
        decision_reasons=decision_reasons or [],
    )


class TestSaturationGate:
    def test_gate_closed_when_no_history(self, tmp_path):
        cache = ClosedLoopFeedbackCache(
            validator=MagicMock(),
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline desc",
        )
        cache._iters_since_last_run = 0  # reset to test purely-history-based gate
        assert cache.should_run() is False

    def test_gate_open_on_saturated_window(self, tmp_path):
        cache = ClosedLoopFeedbackCache(
            validator=MagicMock(),
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
            saturation_threshold=0.95,
            window_size=4,
            min_iters=999,  # disable periodic fallback
        )
        cache._iters_since_last_run = 0
        for s in [0.99, 0.98, 0.99, 1.0]:
            cache.record_judge_score(s)
        assert cache.should_run() is True

    def test_gate_closed_on_unsaturated_window(self, tmp_path):
        cache = ClosedLoopFeedbackCache(
            validator=MagicMock(),
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
            saturation_threshold=0.95,
            window_size=4,
            min_iters=999,
        )
        cache._iters_since_last_run = 0
        for s in [0.99, 0.80, 0.99, 0.99]:  # one dip
            cache.record_judge_score(s)
        assert cache.should_run() is False

    def test_periodic_floor_triggers_even_when_unsaturated(self, tmp_path):
        cache = ClosedLoopFeedbackCache(
            validator=MagicMock(),
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
            saturation_threshold=0.95,
            min_iters=3,
            window_size=4,
        )
        cache._iters_since_last_run = 0
        for s in [0.5, 0.6, 0.7, 0.6]:  # not saturated
            cache.record_judge_score(s)
        # 4 records elapsed since last run, floor=3 → fire
        assert cache.should_run() is True


class TestGetOrRun:
    def test_cache_hit_short_circuits_validator(self, tmp_path):
        validator = MagicMock()
        validator.validate.return_value = _build_report()
        cache = ClosedLoopFeedbackCache(
            validator=validator,
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
            min_iters=1,
        )
        cache.record_judge_score(0.99)  # open the gate
        first = cache.get_or_run("evolved desc")
        assert first is not None
        # Second call: cache hit — validator not invoked again.
        second = cache.get_or_run("evolved desc")
        assert second is first
        assert validator.validate.call_count == 1

    def test_gate_closed_returns_none(self, tmp_path):
        validator = MagicMock()
        cache = ClosedLoopFeedbackCache(
            validator=validator,
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
            saturation_threshold=0.95,
            min_iters=999,
        )
        cache._iters_since_last_run = 0
        cache.record_judge_score(0.5)  # not saturated
        assert cache.get_or_run("desc") is None
        validator.validate.assert_not_called()

    def test_concurrent_run_error_does_not_propagate(self, tmp_path, caplog):
        validator = MagicMock()
        validator.validate.side_effect = ConcurrentRunError("locked")
        cache = ClosedLoopFeedbackCache(
            validator=validator,
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
            min_iters=1,
        )
        cache.record_judge_score(0.99)
        with caplog.at_level("WARNING"):
            result = cache.get_or_run("desc")
        assert result is None
        assert any("closed-loop run skipped" in r.message for r in caplog.records)

    def test_cache_key_depends_on_candidate_and_suite_sha(self, tmp_path):
        cache = ClosedLoopFeedbackCache(
            validator=MagicMock(),
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
        )
        k1 = cache._key("desc A")
        k2 = cache._key("desc B")
        k3 = cache._key("desc A")
        assert k1 != k2
        assert k1 == k3
        # Suite sha is part of the key.
        suite_sha = cache._suite.sha256
        expected = hashlib.sha256(b"desc A\x00" + suite_sha.encode()).hexdigest()
        assert k1 == expected


class TestRenderFeedbackBlock:
    def test_deterministic(self, tmp_path):
        report = _build_report(suite_path=str(tmp_path / "no-suite.jsonl"))
        a = render_feedback_block(report)
        b = render_feedback_block(report)
        assert a == b

    def test_noisy_marker_on_small_delta(self, tmp_path):
        report = _build_report(
            decision="regression",
            pass_rate_change=0.10,
            n_losses=1,
            n_wins=0,
            n_ties=1,
            evolved_tasks=[
                TaskResult("t1", False, False, ["patch"], 1.0, "stub", None),
                TaskResult("t2", True, False, ["patch"], 1.0, "stub", None),
            ],
        )
        rendered = render_feedback_block(report)
        assert rendered.startswith("[CLOSED_LOOP-NOISY]")

    def test_normal_marker_on_large_delta(self, tmp_path):
        report = _build_report(
            decision="regression",
            pass_rate_change=-0.40,
            n_losses=2,
            n_wins=0,
            n_ties=0,
        )
        rendered = render_feedback_block(report)
        assert rendered.startswith("[CLOSED_LOOP]")
        assert "[CLOSED_LOOP-NOISY]" not in rendered

    def test_includes_per_task_diff_for_changed_verdicts(self, tmp_path):
        suite_path = tmp_path / "suite.jsonl"
        suite_path.write_text(
            '{"task_id": "t1", "user_message": "make file X"}\n'
            '{"task_id": "t2", "user_message": "edit Y"}\n'
        )
        report = _build_report(
            decision="regression",
            pass_rate_change=-0.50,
            suite_path=str(suite_path),
            baseline_tasks=[
                TaskResult("t1", True, False, ["write_file"], 1.0, "stub", None),
                TaskResult("t2", True, False, ["patch"], 1.0, "stub", None),
            ],
            evolved_tasks=[
                TaskResult("t1", False, False, ["patch"], 1.0, "stub", None),
                TaskResult("t2", True, False, ["patch"], 1.0, "stub", None),
            ],
            n_wins=0,
            n_losses=1,
            n_ties=1,
        )
        rendered = render_feedback_block(report)
        assert "task t1" in rendered
        assert "loss" in rendered
        assert "'write_file'" in rendered  # baseline call
        assert "'patch'" in rendered  # evolved call
        assert "make file X" in rendered  # user_message from suite

    def test_omits_tied_tasks(self, tmp_path):
        report = _build_report(
            decision="pass",
            pass_rate_change=0.0,
            baseline_tasks=[
                TaskResult("t1", True, False, ["write_file"], 1.0, "stub", None),
                TaskResult("t2", True, False, ["patch"], 1.0, "stub", None),
            ],
            evolved_tasks=[
                TaskResult("t1", True, False, ["write_file"], 1.0, "stub", None),
                TaskResult("t2", True, False, ["patch"], 1.0, "stub", None),
            ],
        )
        rendered = render_feedback_block(report)
        assert "task t1" not in rendered
        assert "task t2" not in rendered

    def test_includes_decision_reasons(self, tmp_path):
        report = _build_report(
            decision="regression",
            pass_rate_change=-0.40,
            decision_reasons=["evolved pass_rate 0.50 < baseline 1.00", "1 losses"],
        )
        rendered = render_feedback_block(report)
        assert "evolved pass_rate 0.50 < baseline 1.00" in rendered
        assert "1 losses" in rendered

    def test_missing_suite_file_does_not_crash(self, tmp_path):
        report = _build_report(
            suite_path="/nonexistent/path/suite.jsonl",
            pass_rate_change=-0.50,  # outside noise floor → non-noisy marker
        )
        # Should render without crashing; just omits the user_message line.
        rendered = render_feedback_block(report)
        assert "[CLOSED_LOOP]" in rendered
        assert "user_message" not in rendered


class TestConcurrency:
    def test_two_threads_invoke_validator_exactly_once(self, tmp_path):
        validator = MagicMock()
        validator.validate.return_value = _build_report()
        cache = ClosedLoopFeedbackCache(
            validator=validator,
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
            min_iters=1,
        )
        cache.record_judge_score(0.99)

        barrier = threading.Barrier(2)
        results: list = []

        def worker():
            barrier.wait()
            results.append(cache.get_or_run("same desc"))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert validator.validate.call_count == 1
        assert results[0] is results[1]
        assert results[0] is not None


class TestConstructorValidation:
    def test_threshold_out_of_range_raises(self, tmp_path):
        with pytest.raises(ValueError, match="saturation_threshold"):
            ClosedLoopFeedbackCache(
                validator=MagicMock(),
                suite=_build_suite(tmp_path),
                artifact_name="write_file",
                baseline_artifact_text="baseline",
                saturation_threshold=1.5,
            )

    def test_min_iters_zero_raises(self, tmp_path):
        with pytest.raises(ValueError, match="min_iters"):
            ClosedLoopFeedbackCache(
                validator=MagicMock(),
                suite=_build_suite(tmp_path),
                artifact_name="write_file",
                baseline_artifact_text="baseline",
                min_iters=0,
            )

    def test_invalid_gate_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="gate_mode"):
            ClosedLoopFeedbackCache(
                validator=MagicMock(),
                suite=_build_suite(tmp_path),
                artifact_name="write_file",
                baseline_artifact_text="baseline",
                gate_mode="bogus",
            )


class TestGateModeAlways:
    def test_always_mode_opens_gate_without_history(self, tmp_path):
        cache = ClosedLoopFeedbackCache(
            validator=MagicMock(),
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
            gate_mode="always",
        )
        # No record_judge_score calls — sampled mode would return False here.
        assert cache.should_run() is True

    def test_always_mode_fires_validator_on_cache_miss(self, tmp_path):
        validator = MagicMock()
        validator.validate.return_value = _build_report()
        cache = ClosedLoopFeedbackCache(
            validator=validator,
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
            gate_mode="always",
        )
        # No judge scores recorded — sampled mode would return None.
        result = cache.get_or_run("evolved desc")
        assert result is not None
        validator.validate.assert_called_once()


class TestGetTaskVerdict:
    def test_returns_per_task_result_on_cache_hit(self, tmp_path):
        report = _build_report(
            decision="regression",
            pass_rate_change=-0.50,
            evolved_tasks=[
                TaskResult("t1", False, False, ["patch"], 1.0, "stub", None),
                TaskResult("t2", True, False, ["write_file"], 1.0, "stub", None),
            ],
        )
        validator = MagicMock()
        validator.validate.return_value = report
        cache = ClosedLoopFeedbackCache(
            validator=validator,
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
            gate_mode="always",
        )
        t1 = cache.get_task_verdict("cand", "t1")
        t2 = cache.get_task_verdict("cand", "t2")
        assert t1 is not None and t1.passed is False
        assert t2 is not None and t2.passed is True
        # Both calls hit the same cached report (validator invoked once).
        assert validator.validate.call_count == 1

    def test_returns_none_on_cache_miss_with_closed_gate(self, tmp_path):
        cache = ClosedLoopFeedbackCache(
            validator=MagicMock(),
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
            saturation_threshold=0.95,
            min_iters=999,
            gate_mode="sampled",  # default; explicit for clarity
        )
        cache._iters_since_last_run = 0
        # No saturating judge scores; gate stays closed.
        cache.record_judge_score(0.5)
        assert cache.get_task_verdict("never_seen", "t1") is None

    def test_returns_none_when_task_id_not_in_report(self, tmp_path):
        report = _build_report(
            evolved_tasks=[
                TaskResult("t1", True, False, ["write_file"], 1.0, "stub", None),
            ],
        )
        validator = MagicMock()
        validator.validate.return_value = report
        cache = ClosedLoopFeedbackCache(
            validator=validator,
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline",
            gate_mode="always",
        )
        assert cache.get_task_verdict("cand", "nonexistent_task") is None


class TestArtifactWriterInjection:
    """The cache's default ``artifact_writer`` produces the single-tool MCP
    manifest JSON the tool installer consumes. Skill-side callers inject
    a writer that drops raw text directly. Tool-path behavior must be
    bit-for-bit preserved when no writer is passed.
    """

    def test_default_writer_produces_mcp_manifest_json(self, tmp_path):
        cache = ClosedLoopFeedbackCache(
            validator=MagicMock(),
            suite=_build_suite(tmp_path),
            artifact_name="write_file",
            baseline_artifact_text="baseline desc",
        )
        # Default suffix is .json; default writer is the MCP manifest.
        baseline_text = cache._baseline_path.read_text()
        assert cache._baseline_path.suffix == ".json"
        parsed = json.loads(baseline_text)
        assert parsed["tools"][0]["name"] == "write_file"
        assert parsed["tools"][0]["description"] == "baseline desc"

    def test_custom_writer_called_with_candidate_and_path(self, tmp_path):
        calls: list[tuple[str, Path]] = []

        def fake_writer(text: str, path: Path) -> None:
            calls.append((text, path))
            path.write_text(f"WRAPPED::{text}")

        cache = ClosedLoopFeedbackCache(
            validator=MagicMock(),
            suite=_build_suite(tmp_path),
            artifact_name="systematic_debugging",
            baseline_artifact_text="baseline body",
            artifact_writer=fake_writer,
            artifact_suffix=".md",
        )
        # Baseline was written once during construction.
        assert len(calls) == 1
        assert calls[0][0] == "baseline body"
        assert calls[0][1].suffix == ".md"
        assert cache._baseline_path.read_text() == "WRAPPED::baseline body"

    def test_custom_writer_invoked_on_each_validate(self, tmp_path):
        calls: list[str] = []

        def fake_writer(text: str, path: Path) -> None:
            calls.append(text)
            path.write_text(text)

        validator = MagicMock()
        validator.validate.return_value = _build_report()
        cache = ClosedLoopFeedbackCache(
            validator=validator,
            suite=_build_suite(tmp_path),
            artifact_name="systematic_debugging",
            baseline_artifact_text="baseline",
            artifact_writer=fake_writer,
            artifact_suffix=".md",
            gate_mode="always",
        )
        cache.get_or_run("candidate-1")
        cache.get_or_run("candidate-2")
        # 1 baseline write + 2 evolved writes.
        assert calls == ["baseline", "candidate-1", "candidate-2"]

    def test_write_text_artifact_helper_writes_plain_text(self, tmp_path):
        from evolution.core.closed_loop_feedback import write_text_artifact

        path = tmp_path / "out.md"
        write_text_artifact("hello world\n", path)
        assert path.read_text() == "hello world\n"


class TestForceRun:
    """`force_run` bypasses should_run() and propagates errors (unlike
    get_or_run which swallows expected validator errors)."""

    def test_force_run_fires_in_sampled_mode_before_any_judge_scores(self, tmp_path):
        """In default sampled mode with empty judge history, should_run()
        returns False — but force_run runs the validator anyway."""
        suite = _build_suite(tmp_path)
        report = _build_report()
        validator = MagicMock()
        validator.validate.return_value = report
        cache = ClosedLoopFeedbackCache(
            validator=validator,
            suite=suite,
            artifact_name="write_file",
            baseline_artifact_text="baseline desc",
            gate_mode="sampled",
        )
        assert cache.should_run() is False

        result = cache.force_run("candidate desc")

        assert result is report
        validator.validate.assert_called_once()

    def test_force_run_uses_cache_on_repeat_calls(self, tmp_path):
        """Second call with same candidate_text returns the cached report
        without re-running the validator."""
        suite = _build_suite(tmp_path)
        report = _build_report()
        validator = MagicMock()
        validator.validate.return_value = report
        cache = ClosedLoopFeedbackCache(
            validator=validator, suite=suite, artifact_name="t",
            baseline_artifact_text="b", gate_mode="sampled",
        )

        first = cache.force_run("cand")
        second = cache.force_run("cand")

        assert first is second
        assert validator.validate.call_count == 1

    def test_force_run_propagates_validator_errors(self, tmp_path):
        """force_run propagates ConcurrentRunError (unlike get_or_run,
        which swallows it and returns None to keep GEPA going). Preflight
        callers want to know the probe failed."""
        suite = _build_suite(tmp_path)
        validator = MagicMock()
        validator.validate.side_effect = ConcurrentRunError("locked")
        cache = ClosedLoopFeedbackCache(
            validator=validator, suite=suite, artifact_name="t",
            baseline_artifact_text="b", gate_mode="sampled",
        )

        with pytest.raises(ConcurrentRunError):
            cache.force_run("cand")

    def test_force_run_preserves_first_fire_for_subsequent_get_or_run(self, tmp_path):
        """force_run at preflight must not eat the first-fire allowance.

        The init contract is _iters_since_last_run = min_iters so that the
        first GEPA-time record_judge_score call pushes the counter above the
        periodic floor and the immediately following get_or_run fires.
        force_run must restore that same value so the guarantee holds even
        when preflight ran before GEPA started.

        In sampled mode, should_run() returns False when judge_history is
        empty (there's an early-return guard). The allowance only takes effect
        after the first record_judge_score — at that point _iters_since_last_run
        must be >= min_iters.  When force_run incorrectly reset to 0, one
        record_judge_score call would leave _iters_since_last_run = 1 < min_iters,
        and should_run() would return False, delaying the first GEPA fire."""
        suite = _build_suite(tmp_path)
        report = _build_report()
        validator = MagicMock()
        validator.validate.return_value = report
        cache = ClosedLoopFeedbackCache(
            validator=validator, suite=suite, artifact_name="t",
            baseline_artifact_text="b", gate_mode="sampled",
            min_iters=3,
        )

        # Preflight fires once (simulates saturation_preflight at init time)
        cache.force_run("baseline")

        # Simulate the first GEPA metric call recording a judge score.
        # After this, _iters_since_last_run must be >= min_iters so
        # should_run() returns True (periodic floor is met).
        cache.record_judge_score(0.7)  # non-saturating score

        assert cache.should_run() is True, (
            "After force_run + one record_judge_score, should_run() must be True "
            "(_iters_since_last_run should be min_iters+1 >= min_iters). "
            "force_run reset to 0 would leave it at 1 < 3 (min_iters), causing False."
        )

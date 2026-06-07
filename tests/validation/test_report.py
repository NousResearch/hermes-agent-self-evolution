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


class TestScoreTaskLayer2:
    """Compound verdict: Layer 1 (trigger) + optional Layer 2 (content judge)."""

    def _save_run(self, content: str = "good") -> AgentRunResult:
        return AgentRunResult(
            tool_calls_seq=["memory"], final_text_tail="", duration_seconds=0.0,
            tool_calls_with_args=[
                {"name": "memory", "arguments": {"action": "add", "content": content}}
            ],
        )

    def test_no_judge_is_layer1_only(self):
        passed, abstained = score_task(
            expected_tools=("memory",), forbidden_tools=(), run=self._save_run(),
        )
        assert passed and not abstained

    def test_passes_when_both_layers_ok(self):
        passed, abstained = score_task(
            expected_tools=("memory",), forbidden_tools=(), run=self._save_run(),
            layer2_judge_fn=lambda calls: 0.9, layer2_threshold=0.7,
        )
        assert passed and not abstained

    def test_fails_when_layer2_below_threshold(self):
        passed, abstained = score_task(
            expected_tools=("memory",), forbidden_tools=(), run=self._save_run("bad"),
            layer2_judge_fn=lambda calls: 0.5, layer2_threshold=0.7,
        )
        assert not passed and not abstained

    def test_layer1_failure_short_circuits_judge(self):
        """Layer 1 fail => judge never called (no LLM cost on a failed trigger)."""
        run = AgentRunResult(
            tool_calls_seq=[], final_text_tail="", duration_seconds=0.0,
            tool_calls_with_args=[],
        )
        calls_seen = []

        def judge_fn(memory_calls):
            calls_seen.append(memory_calls)
            return 1.0

        passed, abstained = score_task(
            expected_tools=("memory",), forbidden_tools=(), run=run,
            layer2_judge_fn=judge_fn, layer2_threshold=0.7,
        )
        assert not passed
        assert calls_seen == []

    def test_judge_receives_only_memory_call_args(self):
        run = AgentRunResult(
            tool_calls_seq=["read_file", "memory"], final_text_tail="", duration_seconds=0.0,
            tool_calls_with_args=[
                {"name": "read_file", "arguments": {"path": "x"}},
                {"name": "memory", "arguments": {"action": "add", "content": "c"}},
            ],
        )
        received = []

        def judge_fn(memory_calls):
            received.append(memory_calls)
            return 1.0

        score_task(
            expected_tools=("memory",), forbidden_tools=(), run=run,
            layer2_judge_fn=judge_fn, layer2_threshold=0.7,
        )
        assert received == [[{"action": "add", "content": "c"}]]


class TestScoreTaskActionPatch:
    """Action-level verdict: expected_action='patch' + target_skill + stale_token."""

    @staticmethod
    def _run_with_skill_manage(
        skill_name: str,
        action: str,
        old_string: str = "",
        content: str = "",
    ) -> AgentRunResult:
        args: dict = {"action": action, "name": skill_name}
        if action in ("patch", "edit"):
            if action == "patch":
                args["old_string"] = old_string
                args["new_string"] = old_string.replace("stale", "fresh")
            else:
                args["content"] = content
        return AgentRunResult(
            tool_calls_seq=["skill_manage"],
            final_text_tail="",
            duration_seconds=1.0,
            tool_calls_with_args=[{"name": "skill_manage", "arguments": args}],
        )

    def test_patch_touching_stale_token_passes(self):
        run = self._run_with_skill_manage(
            "SKILLS_GUIDANCE", "patch", old_string="stale text here"
        )
        passed, abstained = score_task(
            expected_tools=(),
            forbidden_tools=(),
            run=run,
            expected_action="patch",
            target_skill="SKILLS_GUIDANCE",
            stale_token="stale",
        )
        assert passed and not abstained

    def test_patch_not_touching_stale_token_fails(self):
        run = self._run_with_skill_manage(
            "SKILLS_GUIDANCE", "patch", old_string="completely different text"
        )
        passed, abstained = score_task(
            expected_tools=(),
            forbidden_tools=(),
            run=run,
            expected_action="patch",
            target_skill="SKILLS_GUIDANCE",
            stale_token="stale",
        )
        assert not passed and not abstained

    def test_patch_wrong_skill_fails(self):
        run = self._run_with_skill_manage(
            "OTHER_SKILL", "patch", old_string="stale text here"
        )
        passed, abstained = score_task(
            expected_tools=(),
            forbidden_tools=(),
            run=run,
            expected_action="patch",
            target_skill="SKILLS_GUIDANCE",
            stale_token="stale",
        )
        assert not passed and not abstained

    def test_edit_with_stale_token_absent_from_content_passes(self):
        # edit action: content must NOT contain stale_token (it was replaced)
        run = self._run_with_skill_manage(
            "SKILLS_GUIDANCE", "edit", content="fresh text here"
        )
        passed, abstained = score_task(
            expected_tools=(),
            forbidden_tools=(),
            run=run,
            expected_action="patch",
            target_skill="SKILLS_GUIDANCE",
            stale_token="stale",
        )
        assert passed and not abstained

    def test_edit_with_stale_token_still_in_content_fails(self):
        # edit action: if stale_token still in content, the skill wasn't updated
        run = self._run_with_skill_manage(
            "SKILLS_GUIDANCE", "edit", content="still stale text here"
        )
        passed, abstained = score_task(
            expected_tools=(),
            forbidden_tools=(),
            run=run,
            expected_action="patch",
            target_skill="SKILLS_GUIDANCE",
            stale_token="stale",
        )
        assert not passed and not abstained

    def test_no_skill_manage_call_fails(self):
        run = AgentRunResult(
            tool_calls_seq=["read_file"],
            final_text_tail="",
            duration_seconds=1.0,
            tool_calls_with_args=[{"name": "read_file", "arguments": {"path": "x"}}],
        )
        passed, abstained = score_task(
            expected_tools=(),
            forbidden_tools=(),
            run=run,
            expected_action="patch",
            target_skill="SKILLS_GUIDANCE",
            stale_token="stale",
        )
        assert not passed and not abstained

    def test_runner_error_abstains(self):
        run = AgentRunResult(
            tool_calls_seq=[],
            final_text_tail="",
            duration_seconds=1.0,
            error="hermes timed out",
        )
        passed, abstained = score_task(
            expected_tools=(),
            forbidden_tools=(),
            run=run,
            expected_action="patch",
            target_skill="SKILLS_GUIDANCE",
            stale_token="stale",
        )
        assert not passed and abstained

    def test_expected_action_none_leaves_existing_membership_path_unchanged(self):
        # Regression guard: when expected_action is None, behavior is identical
        # to today's tool-membership scoring.
        run = AgentRunResult(
            tool_calls_seq=["patch"],
            final_text_tail="",
            duration_seconds=1.0,
        )
        passed, abstained = score_task(
            expected_tools=("patch",),
            forbidden_tools=("write_file",),
            run=run,
        )
        assert passed and not abstained


class TestScoreTaskTestCommandMode:
    """When ``test_command`` is set on a task, the verdict is exit-code-driven,
    not tool-call-driven. Used by skill-side suites (e.g., planted-bug:
    "did the agent's edits make the test pass").
    """

    @staticmethod
    def _ok_run() -> AgentRunResult:
        return AgentRunResult(
            tool_calls_seq=[], final_text_tail="", duration_seconds=1.0,
        )

    def test_passes_on_exit_zero(self, tmp_path):
        (tmp_path / "ok.py").write_text("import sys; sys.exit(0)\n")
        passed, abstained = score_task(
            expected_tools=(), forbidden_tools=(), run=self._ok_run(),
            test_command="python ok.py",
            fixture_dir=tmp_path,
        )
        assert passed
        assert not abstained

    def test_fails_on_nonzero_exit(self, tmp_path):
        (tmp_path / "bad.py").write_text("import sys; sys.exit(1)\n")
        passed, abstained = score_task(
            expected_tools=(), forbidden_tools=(), run=self._ok_run(),
            test_command="python bad.py",
            fixture_dir=tmp_path,
        )
        assert not passed
        assert not abstained

    def test_timeout_marks_failed_not_abstained(self, tmp_path):
        # Treat hangs as failure rather than abstention — a debugging
        # task that goes infinite is the agent's failure to debug.
        (tmp_path / "slow.py").write_text("import time; time.sleep(60)\n")
        passed, abstained = score_task(
            expected_tools=(), forbidden_tools=(), run=self._ok_run(),
            test_command="python slow.py",
            fixture_dir=tmp_path,
            test_command_timeout_seconds=0.3,
        )
        assert not passed
        assert not abstained

    def test_cwd_is_fixture_dir(self, tmp_path):
        # The test script verifies its own cwd matches the fixture dir.
        (tmp_path / "cwd_check.py").write_text(
            "import os, sys\n"
            "sys.exit(0 if os.path.realpath(os.getcwd()) == sys.argv[1] else 1)\n"
        )
        passed, _ = score_task(
            expected_tools=(), forbidden_tools=(), run=self._ok_run(),
            test_command=f"python cwd_check.py {tmp_path.resolve()}",
            fixture_dir=tmp_path,
        )
        assert passed

    def test_precedence_over_tool_call_rule(self, tmp_path):
        # When test_command is set, the tool-call rule is ignored
        # entirely — even if the agent invoked a forbidden tool.
        (tmp_path / "ok.py").write_text("")  # empty file → python ok.py exits 0
        run = AgentRunResult(
            tool_calls_seq=["forbidden_tool"], final_text_tail="", duration_seconds=1.0,
        )
        passed, _ = score_task(
            expected_tools=("expected_tool",),
            forbidden_tools=("forbidden_tool",),
            run=run,
            test_command="python ok.py",
            fixture_dir=tmp_path,
        )
        assert passed

    def test_command_not_found_fails(self, tmp_path):
        passed, abstained = score_task(
            expected_tools=(), forbidden_tools=(), run=self._ok_run(),
            test_command="this-binary-does-not-exist-12345",
            fixture_dir=tmp_path,
        )
        assert not passed
        assert not abstained

    def test_missing_fixture_dir_raises(self):
        with pytest.raises(ValueError, match="fixture_dir is required"):
            score_task(
                expected_tools=(), forbidden_tools=(), run=self._ok_run(),
                test_command="python ok.py",
            )

    def test_runner_error_still_abstains_with_test_command(self, tmp_path):
        # Runner error takes precedence over test_command — same as it
        # does over the tool-call rule. A subprocess crash that prevented
        # the agent from running isn't evidence either way.
        (tmp_path / "ok.py").write_text("")
        run = AgentRunResult(
            tool_calls_seq=[], final_text_tail="", duration_seconds=1.0,
            error="hermes crashed",
        )
        passed, abstained = score_task(
            expected_tools=(), forbidden_tools=(), run=run,
            test_command="python ok.py",
            fixture_dir=tmp_path,
        )
        assert not passed
        assert abstained

    def test_tool_only_path_unchanged_when_test_command_absent(self):
        # Regression guard for the existing patch/search_files/write_file suites.
        run = AgentRunResult(
            tool_calls_seq=["patch"], final_text_tail="", duration_seconds=1.0,
        )
        passed, abstained = score_task(
            expected_tools=("patch",), forbidden_tools=("write_file",), run=run,
        )
        assert passed
        assert not abstained


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


class TestComputeWinLossRateBased:
    """Win/loss compares pass_rate, not the bool. At reps=1 (rates in
    {0.0, 1.0}) this reduces to the legacy bool comparison."""

    def _tr_rate(self, task_id, rate):
        return TaskResult(
            task_id=task_id, passed=(rate >= 0.5), pass_rate=rate,
            abstained=False, tool_calls_seq=[], duration_seconds=0.0,
        )

    def test_higher_rate_is_win(self):
        b = summarize_phase([self._tr_rate("t", 0.0)])
        e = summarize_phase([self._tr_rate("t", 0.8)])
        wl = compute_win_loss(b, e)
        assert wl.n_wins == 1 and wl.n_losses == 0

    def test_win_even_below_pass_threshold(self):
        # 0.4 vs 0.0 is a win even though neither side "passes" (>=0.5).
        b = summarize_phase([self._tr_rate("t", 0.0)])
        e = summarize_phase([self._tr_rate("t", 0.4)])
        wl = compute_win_loss(b, e)
        assert wl.n_wins == 1 and wl.n_losses == 0

    def test_equal_rate_is_neither(self):
        b = summarize_phase([self._tr_rate("t", 0.0)])
        e = summarize_phase([self._tr_rate("t", 0.0)])
        wl = compute_win_loss(b, e)
        assert wl.n_wins == 0 and wl.n_losses == 0 and wl.n_ties == 1

    def test_lower_rate_is_loss(self):
        b = summarize_phase([self._tr_rate("t", 0.8)])
        e = summarize_phase([self._tr_rate("t", 0.0)])
        wl = compute_win_loss(b, e)
        assert wl.n_losses == 1 and wl.n_wins == 0

    def test_reps1_rates_reduce_to_legacy_bool(self):
        # At reps=1 every rate is 0.0 or 1.0. Compare rate-based win/loss
        # against the legacy bool-based outcome for all 4 combinations.
        for b_pass in (False, True):
            for e_pass in (False, True):
                b_rate = 1.0 if b_pass else 0.0
                e_rate = 1.0 if e_pass else 0.0
                b = summarize_phase([self._tr_rate("t", b_rate)])
                e = summarize_phase([self._tr_rate("t", e_rate)])
                wl = compute_win_loss(b, e)
                legacy_win = int(e_pass and not b_pass)
                legacy_loss = int(b_pass and not e_pass)
                assert wl.n_wins == legacy_win
                assert wl.n_losses == legacy_loss


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

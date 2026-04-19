"""Tests for hard run-budget enforcement in codex-batched evolution."""

import time

from evolution.core.budget import BudgetExceeded, RunBudget


class TestRunBudget:
    def test_budget_blocks_when_call_limit_reached(self):
        budget = RunBudget(
            max_codex_calls=2,
            max_run_seconds=600,
            phase_timeout_seconds=120,
        )
        budget.start_run()
        budget.register_call("dataset")
        budget.register_call("mutation")

        assert budget.calls_used == 2
        assert budget.remaining_calls() == 0
        assert budget.can_start_phase("evaluation") is False

    def test_budget_blocks_when_wall_clock_exceeded(self):
        budget = RunBudget(
            max_codex_calls=3,
            max_run_seconds=0,
            phase_timeout_seconds=120,
        )
        budget.start_run()
        time.sleep(0.01)

        assert budget.can_start_phase("mutation") is False

    def test_budget_enforce_or_raise_raises_when_limit_hit(self):
        budget = RunBudget(
            max_codex_calls=1,
            max_run_seconds=600,
            phase_timeout_seconds=120,
        )
        budget.start_run()
        budget.register_call("dataset")

        try:
            budget.enforce_or_raise("mutation")
        except BudgetExceeded as exc:
            assert "mutation" in str(exc)
        else:
            raise AssertionError("BudgetExceeded was not raised")

    def test_budget_blocks_phase_when_remaining_time_is_below_phase_timeout(self):
        budget = RunBudget(
            max_codex_calls=3,
            max_run_seconds=1,
            phase_timeout_seconds=10,
        )
        budget.start_run()

        assert budget.can_start_phase("mutation") is False

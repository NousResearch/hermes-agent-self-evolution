"""Tests for run cost accounting.

Offline: history entries are hand-built dicts in DSPy's recorded shape, so
nothing here calls a model. The interesting cases are all about what the report
refuses to claim when the log is incomplete.
"""

import pytest

from evolution.core.cost import CostReport, LMCall, UsageTracker, read_history


def entry(model="openai/gpt-4.1", prompt=100, completion=50, cost=0.001, **extra):
    blob = {
        "model": model,
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        "cost": cost,
    }
    blob.update(extra)
    return blob


class TestLMCall:
    def test_reads_a_normal_entry(self):
        call = LMCall.from_entry(entry())
        assert call.model == "openai/gpt-4.1"
        assert call.total_tokens == 150
        assert call.cost == pytest.approx(0.001)
        assert call.priced

    def test_accepts_the_input_output_token_spelling(self):
        call = LMCall.from_entry(
            {"model": "m", "usage": {"input_tokens": 7, "output_tokens": 3}, "cost": 0.1}
        )
        assert call.prompt_tokens == 7 and call.completion_tokens == 3

    def test_a_missing_price_is_not_zero(self):
        call = LMCall.from_entry(entry(cost=None))
        assert call.cost is None
        assert not call.priced

    def test_a_malformed_entry_does_not_raise(self):
        assert LMCall.from_entry("nonsense").total_tokens == 0
        assert LMCall.from_entry({}).model == "unknown"
        assert LMCall.from_entry({"usage": "bad", "cost": "bad"}).cost is None

    def test_a_cached_call_is_recognised(self):
        assert LMCall.from_entry({"model": "m", "usage": {}, "cost": 0}).cached


class TestCostReport:
    def test_sums_tokens_and_known_cost(self):
        report = read_history([entry(cost=0.01), entry(cost=0.02)])
        assert report.n_calls == 2
        assert report.total_tokens == 300
        assert report.known_cost == pytest.approx(0.03)
        assert report.complete

    def test_unpriced_calls_are_excluded_from_the_total_and_counted(self):
        report = read_history([entry(cost=0.01), entry(cost=None)])
        assert report.known_cost == pytest.approx(0.01)
        assert report.unpriced_calls == 1
        assert not report.complete

    def test_an_incomplete_report_says_at_least(self):
        report = read_history([entry(cost=0.01), entry(cost=None)])
        assert "at least" in report.describe()
        assert "no price available" in report.describe()

    def test_a_complete_report_does_not_hedge(self):
        assert "at least" not in read_history([entry(cost=0.01)]).describe()

    def test_truncation_is_reported(self):
        report = read_history([entry()])
        report.truncated = True
        assert not report.complete
        assert "lower bound" in report.describe()

    def test_per_model_breakdown(self):
        report = read_history([entry(model="a"), entry(model="b"), entry(model="a")])
        assert report.models == {"a": 2, "b": 1}

    def test_empty_report(self):
        report = CostReport()
        assert report.n_calls == 0
        assert report.known_cost == 0.0
        assert report.describe() == "no model calls recorded"

    def test_serialises(self):
        blob = read_history([entry(cost=None)]).to_dict()
        assert blob["unpriced_calls"] == 1
        assert blob["complete"] is False
        assert blob["known_cost_usd"] == 0.0


class TestUsageTracker:
    @pytest.fixture
    def history(self, monkeypatch):
        log: list = []
        monkeypatch.setattr("evolution.core.cost._history", lambda: log)
        return log

    def test_measures_only_what_happened_inside_the_block(self, history):
        history.append(entry(cost=0.05))  # before the block
        with UsageTracker() as usage:
            history.append(entry(cost=0.01))
            history.append(entry(cost=0.02))
        assert usage.report.n_calls == 2
        assert usage.report.known_cost == pytest.approx(0.03)

    def test_an_idle_block_reports_nothing(self, history):
        with UsageTracker() as usage:
            pass
        assert usage.report.n_calls == 0

    def test_an_evicted_log_is_flagged_not_negated(self, history):
        history.extend(entry() for _ in range(5))
        with UsageTracker() as usage:
            del history[:]           # DSPy dropped the history mid-run
            history.append(entry())
        assert usage.report.truncated
        # Exactly the one entry appended after the wipe: an eviction must not
        # erase what did happen, and a lower bound is still not `complete`.
        assert usage.report.n_calls == 1
        assert not usage.report.complete

    def test_stop_can_be_called_directly(self, history):
        tracker = UsageTracker()
        tracker.__enter__()
        history.append(entry())
        assert tracker.stop().n_calls == 1


class TestEvictedHistory:
    """DSPy caps its log at MAX_HISTORY_SIZE and pops from the front.

    The length therefore stops growing, so a length comparison alone can never
    notice that a long run made more calls than the log can hold. The report
    used to come back complete and confident while silently missing most of the
    spend, in a PR body whose whole selling point is that it never rounds an
    unknown down.
    """

    @pytest.fixture
    def capped(self, monkeypatch):
        log: list = []
        monkeypatch.setattr("evolution.core.cost._history", lambda: log)
        monkeypatch.setattr("evolution.core.cost._max_history_size", lambda: 10)
        return log

    def _flood(self, log, n):
        for i in range(n):
            log.append({"uuid": f"call-{i}", "usage": {"prompt_tokens": 10},
                        "cost": 1.0})
            if len(log) > 10:
                del log[0]

    def test_a_run_that_overflows_the_log_is_reported_as_a_lower_bound(self, capped):
        capped.extend({"uuid": f"pre-{i}", "usage": {}, "cost": 0.0} for i in range(3))
        with UsageTracker() as usage:
            self._flood(capped, 25)
        assert usage.report.truncated
        assert not usage.report.complete
        assert "lower bound" in usage.report.describe()

    def test_the_reported_total_is_still_what_survived(self, capped):
        with UsageTracker() as usage:
            self._flood(capped, 25)
        assert usage.report.n_calls == 10
        assert usage.report.known_cost == pytest.approx(10.0)

    def test_a_short_run_is_not_flagged(self, capped):
        capped.append({"uuid": "pre", "usage": {}, "cost": 0.0})
        with UsageTracker() as usage:
            capped.append({"uuid": "a", "usage": {"prompt_tokens": 5}, "cost": 0.5})
            capped.append({"uuid": "b", "usage": {"prompt_tokens": 5}, "cost": 0.5})
        assert usage.report.n_calls == 2
        assert not usage.report.truncated
        assert usage.report.complete

    def test_entries_from_before_the_block_are_still_excluded(self, capped):
        capped.extend(
            {"uuid": f"pre-{i}", "usage": {"prompt_tokens": 99}, "cost": 9.0}
            for i in range(4)
        )
        with UsageTracker() as usage:
            capped.append({"uuid": "mine", "usage": {"prompt_tokens": 1}, "cost": 0.25})
        assert usage.report.n_calls == 1
        assert usage.report.known_cost == pytest.approx(0.25)

"""Tests for cache-aware token + cost accounting.

`_log_litellm_cost` hooks litellm directly (DSPy's `on_lm_end` only sees
parsed text, not the raw `ModelResponse` with `.usage`). The DSPy-level
callback can never see cached_tokens, which is why this lives in
litellm.success_callback instead.

Tests stub `litellm.completion_cost` to return injected dollar values so
they survive litellm version bumps that change the price table.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evolution.core.lm_timing_callback import (
    COST_LEDGER,
    CostCeilingExceeded,
    CostLedger,
    LMTimingCallback,
    _log_litellm_cost,
)


def _make_response(
    *,
    model: str = "openai/gpt-4.1-mini",
    prompt_tokens: int | None = 0,
    cached_tokens: int | None = 0,
    completion_tokens: int | None = 0,
    reasoning_tokens: int | None = 0,
    has_usage: bool = True,
):
    """Build a ModelResponse-shaped namespace.

    has_usage=False simulates a streaming-aggregate or non-supporting
    provider where `usage` is absent — the callback must not crash.
    """
    if not has_usage:
        return SimpleNamespace(model=model)
    pdt = (
        SimpleNamespace(cached_tokens=cached_tokens)
        if cached_tokens is not None
        else None
    )
    cdt = (
        SimpleNamespace(reasoning_tokens=reasoning_tokens)
        if reasoning_tokens is not None
        else None
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=pdt,
        completion_tokens_details=cdt,
    )
    return SimpleNamespace(model=model, usage=usage)


class TestCostLedgerAccounting:
    """Cache-aware aggregation: cached tokens land in their own bucket so
    the cache-hit-rate diagnostic doesn't get washed out by the absolute
    token volume."""

    def test_cache_miss_records_all_tokens_uncached(self):
        ledger = CostLedger()
        response = _make_response(
            prompt_tokens=1000, cached_tokens=0, completion_tokens=200,
        )
        with patch(
            "evolution.core.lm_timing_callback.litellm.completion_cost",
            return_value=0.0042,
        ):
            _log_litellm_cost({}, response, None, None, ledger=ledger)
        summary = ledger.summary()
        assert summary["total_usd"] == 0.0042
        row = summary["by_model"]["openai/gpt-4.1-mini"]
        assert row["tokens_in_uncached"] == 1000
        assert row["tokens_in_cached"] == 0
        assert row["tokens_out"] == 200
        assert row["cache_hit_rate"] == 0.0

    def test_partial_cache_hit_splits_correctly(self):
        ledger = CostLedger()
        response = _make_response(
            prompt_tokens=1000, cached_tokens=600, completion_tokens=200,
        )
        with patch(
            "evolution.core.lm_timing_callback.litellm.completion_cost",
            return_value=0.0021,
        ):
            _log_litellm_cost({}, response, None, None, ledger=ledger)
        row = ledger.summary()["by_model"]["openai/gpt-4.1-mini"]
        assert row["tokens_in_uncached"] == 400
        assert row["tokens_in_cached"] == 600
        assert row["cache_hit_rate"] == 0.6

    def test_full_cache_hit_records_zero_uncached(self):
        ledger = CostLedger()
        response = _make_response(
            prompt_tokens=500, cached_tokens=500, completion_tokens=100,
        )
        with patch(
            "evolution.core.lm_timing_callback.litellm.completion_cost",
            return_value=0.0001,
        ):
            _log_litellm_cost({}, response, None, None, ledger=ledger)
        row = ledger.summary()["by_model"]["openai/gpt-4.1-mini"]
        assert row["tokens_in_uncached"] == 0
        assert row["tokens_in_cached"] == 500
        assert row["cache_hit_rate"] == 1.0

    def test_multi_model_aggregates_separately(self):
        ledger = CostLedger()
        with patch(
            "evolution.core.lm_timing_callback.litellm.completion_cost",
            side_effect=[0.001, 0.002, 0.003],
        ):
            _log_litellm_cost(
                {}, _make_response(model="openai/gpt-4.1", prompt_tokens=100, completion_tokens=50),
                None, None, ledger=ledger,
            )
            _log_litellm_cost(
                {}, _make_response(model="openai/gpt-4.1-mini", prompt_tokens=200, completion_tokens=80),
                None, None, ledger=ledger,
            )
            _log_litellm_cost(
                {}, _make_response(model="openai/gpt-4.1", prompt_tokens=150, completion_tokens=60),
                None, None, ledger=ledger,
            )
        summary = ledger.summary()
        assert summary["total_usd"] == 0.006
        assert set(summary["by_model"].keys()) == {"openai/gpt-4.1", "openai/gpt-4.1-mini"}
        gpt41 = summary["by_model"]["openai/gpt-4.1"]
        gpt41mini = summary["by_model"]["openai/gpt-4.1-mini"]
        assert gpt41["calls"] == 2
        assert gpt41mini["calls"] == 1
        assert gpt41["tokens_in_uncached"] == 250
        assert gpt41mini["tokens_in_uncached"] == 200

    def test_missing_usage_is_recorded_as_zeros(self):
        """Streaming aggregates and non-supporting providers can omit
        `usage` entirely. The callback must record the call (so call
        count stays accurate) but credit zero tokens — never crash."""
        ledger = CostLedger()
        response = _make_response(has_usage=False)
        with patch(
            "evolution.core.lm_timing_callback.litellm.completion_cost",
            return_value=0.0,
        ):
            _log_litellm_cost({}, response, None, None, ledger=ledger)
        summary = ledger.summary()
        row = summary["by_model"]["openai/gpt-4.1-mini"]
        assert row["calls"] == 1
        assert row["tokens_in_uncached"] == 0
        assert row["tokens_in_cached"] == 0
        assert row["tokens_out"] == 0
        assert row["cache_hit_rate"] == 0.0

    def test_completion_cost_failure_records_zero_dollars(self):
        """If litellm.completion_cost raises (unknown model, bad price
        table, etc.), the ledger must still record token counts for
        diagnostic value — just credit $0 for cost so a failed
        price-lookup doesn't crash the run."""
        ledger = CostLedger()
        response = _make_response(prompt_tokens=100, completion_tokens=50)
        with patch(
            "evolution.core.lm_timing_callback.litellm.completion_cost",
            side_effect=RuntimeError("no price table for this model"),
        ):
            _log_litellm_cost({}, response, None, None, ledger=ledger)
        summary = ledger.summary()
        assert summary["total_usd"] == 0.0
        row = summary["by_model"]["openai/gpt-4.1-mini"]
        assert row["calls"] == 1
        assert row["tokens_in_uncached"] == 100


class TestCostLedgerThreadSafety:
    """`dspy.Evaluate(num_threads=4)` drives concurrent LM calls. The
    ledger must hold per-model totals deterministically under load —
    otherwise cache_hit_rate (a ratio of two raced counters) becomes
    non-deterministic and worthless as a diagnostic."""

    def test_concurrent_record_calls_sum_correctly(self):
        ledger = CostLedger()
        n_threads = 8
        calls_per_thread = 50
        # Each call: 100 uncached input tokens, 20 output, $0.001
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()  # maximize contention
            for _ in range(calls_per_thread):
                ledger.record(
                    model="openai/gpt-4.1-mini",
                    prompt_tokens=100,
                    cached_tokens=0,
                    completion_tokens=20,
                    reasoning_tokens=0,
                    cost_usd=0.001,
                )

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = ledger.summary()
        row = summary["by_model"]["openai/gpt-4.1-mini"]
        expected_calls = n_threads * calls_per_thread
        assert row["calls"] == expected_calls
        assert row["tokens_in_uncached"] == 100 * expected_calls
        assert row["tokens_out"] == 20 * expected_calls
        assert summary["total_usd"] == pytest.approx(0.001 * expected_calls, abs=1e-9)


class TestCostLedgerReset:
    """`evolve()` calls reset() at the start of each run so metrics.json
    only contains that run's cost, not residual from previous runs in
    the same Python process."""

    def test_reset_clears_all_models(self):
        ledger = CostLedger()
        with patch(
            "evolution.core.lm_timing_callback.litellm.completion_cost",
            return_value=0.001,
        ):
            _log_litellm_cost(
                {}, _make_response(prompt_tokens=100, completion_tokens=10),
                None, None, ledger=ledger,
            )
        assert ledger.summary()["total_usd"] > 0
        ledger.reset()
        summary = ledger.summary()
        assert summary["total_usd"] == 0.0
        assert summary["by_model"] == {}

    def test_reset_clears_pending_abort_flag(self):
        """The cost callback persists process-globally, so a stale flag
        would abort the next run's first LM call."""
        ledger = CostLedger()
        ledger.set_ceiling(0.0)
        with patch(
            "evolution.core.lm_timing_callback.litellm.completion_cost",
            return_value=0.5,
        ):
            _log_litellm_cost(
                {}, _make_response(prompt_tokens=100, completion_tokens=10),
                None, None, ledger=ledger,
            )
        assert ledger.get_abort_state() is not None  # abort is queued
        ledger.reset()
        assert ledger.get_abort_state() is None  # cleared


class TestCostCeiling:
    """The cost-ceiling kill switch."""

    def test_set_ceiling_persists_until_reset(self):
        ledger = CostLedger()
        ledger.set_ceiling(2.50)
        # Setting None disables (no exception, no aborts queued).
        ledger.set_ceiling(None)
        assert ledger.get_abort_state() is None

    def test_record_below_ceiling_does_not_set_flag(self):
        ledger = CostLedger()
        ledger.set_ceiling(1.00)
        with patch(
            "evolution.core.lm_timing_callback.litellm.completion_cost",
            return_value=0.50,  # under the ceiling
        ):
            _log_litellm_cost(
                {}, _make_response(prompt_tokens=10, completion_tokens=5),
                None, None, ledger=ledger,
            )
        assert ledger.get_abort_state() is None

    def test_record_over_ceiling_sets_flag(self):
        ledger = CostLedger()
        ledger.set_ceiling(0.10)
        with patch(
            "evolution.core.lm_timing_callback.litellm.completion_cost",
            return_value=0.25,  # one call past the ceiling
        ):
            _log_litellm_cost(
                {}, _make_response(prompt_tokens=10, completion_tokens=5),
                None, None, ledger=ledger,
            )
        state = ledger.get_abort_state()
        assert state is not None
        total, ceiling = state
        assert ceiling == 0.10
        assert total == pytest.approx(0.25)

    def test_record_does_not_raise_when_over_ceiling(self):
        """The cost callback only sets the flag — it never raises (litellm
        would swallow callback exceptions anyway).
        """
        ledger = CostLedger()
        ledger.set_ceiling(0.0)
        with patch(
            "evolution.core.lm_timing_callback.litellm.completion_cost",
            return_value=1.00,
        ):
            # Must complete without exception even though total >> ceiling.
            _log_litellm_cost(
                {}, _make_response(prompt_tokens=10, completion_tokens=5),
                None, None, ledger=ledger,
            )

    def test_baselm_call_raises_when_abort_pending(self):
        """The monkey-patched BaseLM.__call__ is the load-bearing seam —
        callbacks get their exceptions swallowed, the call path doesn't.
        """
        from dspy.clients.base_lm import BaseLM

        COST_LEDGER.reset()
        try:
            COST_LEDGER.set_ceiling(0.0)
            with patch(
                "evolution.core.lm_timing_callback.litellm.completion_cost",
                return_value=1.00,
            ):
                _log_litellm_cost(
                    {}, _make_response(prompt_tokens=10, completion_tokens=5),
                    None, None,
                )
            # Build a minimal BaseLM-like subclass; we don't want to spawn a
            # real LM. Calling __call__ directly through the class exercises
            # the patched method.
            class _StubLM(BaseLM):
                model = "stub"

                def forward(self, *args, **kwargs):  # pragma: no cover — should not reach this
                    raise AssertionError("forward should not be called when abort is pending")

            stub = _StubLM(model="stub")
            with pytest.raises(CostCeilingExceeded) as exc_info:
                stub(prompt="hello")
            assert exc_info.value.ceiling_usd == 0.0
            assert exc_info.value.total_usd == pytest.approx(1.00)
        finally:
            COST_LEDGER.reset()

    def test_baselm_call_does_not_raise_when_no_abort(self):
        """Sanity: with no ceiling set, the patched __call__ delegates
        to the original BaseLM behavior with no extra exception.
        """
        from dspy.clients.base_lm import BaseLM
        COST_LEDGER.reset()

        called = {"forward": False}

        class _StubLM(BaseLM):
            model = "stub"

            def forward(self, *args, **kwargs):
                called["forward"] = True
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                    usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
                    model="stub",
                )

        stub = _StubLM(model="stub")
        # Doesn't raise from the patch (forward may or may not succeed,
        # but the patch itself should not interfere).
        try:
            stub(prompt="hello")
        except CostCeilingExceeded:
            pytest.fail("patched __call__ raised CostCeilingExceeded when no abort was pending")
        except Exception:
            # Other downstream errors are acceptable for this test — we
            # only care that the cost-ceiling guard didn't fire.
            pass


class TestAgentCostCapture:
    """Agent-side cost (from state.db) is added to the same CostLedger so
    the ceiling reflects end-to-end spend, not just in-process LM calls."""

    def test_record_agent_cost_actual_updates_summary(self):
        ledger = CostLedger()
        ledger.record_agent_cost(0.02)
        summary = ledger.summary()
        assert summary["agent_cost_usd"] == 0.02
        assert summary["n_agent_runs"] == 1
        assert summary["n_cost_uncaptured"] == 0
        assert summary["total_cost_usd"] == pytest.approx(0.02)

    def test_record_agent_cost_uncaptured_increments_counter_only(self):
        ledger = CostLedger()
        ledger.record_agent_cost(None)
        summary = ledger.summary()
        assert summary["n_agent_runs"] == 1
        assert summary["n_cost_uncaptured"] == 1
        assert summary["agent_cost_usd"] == 0.0
        assert summary["total_cost_usd"] == 0.0

    def test_agent_cost_alone_trips_ceiling(self):
        ledger = CostLedger()
        ledger.set_ceiling(0.05)
        ledger.record_agent_cost(0.06)
        state = ledger.get_abort_state()
        assert state is not None
        total, ceiling = state
        assert ceiling == pytest.approx(0.05)
        assert total == pytest.approx(0.06)

    def test_agent_cost_tips_near_ceiling_over(self):
        ledger = CostLedger()
        ledger.set_ceiling(0.05)
        ledger.record(
            model="openai/gpt-4.1-mini",
            prompt_tokens=10,
            cached_tokens=0,
            completion_tokens=5,
            reasoning_tokens=0,
            cost_usd=0.04,
        )
        assert ledger.get_abort_state() is None  # below ceiling
        ledger.record_agent_cost(0.02)
        state = ledger.get_abort_state()
        assert state is not None
        total, ceiling = state
        assert ceiling == pytest.approx(0.05)
        assert total == pytest.approx(0.06)

    def test_record_in_process_tips_combined_over_ceiling(self):
        ledger = CostLedger()
        ledger.set_ceiling(0.05)
        ledger.record_agent_cost(0.04)
        assert ledger.get_abort_state() is None  # agent cost below ceiling
        ledger.record(
            model="openai/gpt-4.1-mini",
            prompt_tokens=10,
            cached_tokens=0,
            completion_tokens=5,
            reasoning_tokens=0,
            cost_usd=0.02,
        )
        state = ledger.get_abort_state()
        assert state is not None
        total, ceiling = state
        assert ceiling == pytest.approx(0.05)
        assert total == pytest.approx(0.06)

    def test_reset_zeroes_agent_fields(self):
        ledger = CostLedger()
        ledger.record_agent_cost(0.03)
        ledger.record_agent_cost(None)
        ledger.reset()
        summary = ledger.summary()
        assert summary["agent_cost_usd"] == 0.0
        assert summary["n_agent_runs"] == 0
        assert summary["n_cost_uncaptured"] == 0

    def test_total_cost_usd_is_combined(self):
        ledger = CostLedger()
        ledger.record(
            model="openai/gpt-4.1-mini",
            prompt_tokens=10,
            cached_tokens=0,
            completion_tokens=5,
            reasoning_tokens=0,
            cost_usd=0.01,
        )
        ledger.record_agent_cost(0.02)
        summary = ledger.summary()
        assert summary["total_cost_usd"] == pytest.approx(0.03)
        # Existing key must still be present for back-compat.
        assert summary["total_usd"] == pytest.approx(0.01)

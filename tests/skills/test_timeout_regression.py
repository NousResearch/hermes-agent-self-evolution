"""Regression tests for the 480s evolution timeout budget (t_316c92c4).

Covers the three fixes from the web-research root-cause analysis
(workspace/evolution/root-cause-web-research-20260731.md):

1. make_lm() bounds per-call LLM timeout/retries — a stalled API call can no
   longer hang for 600s+ (litellm default) and blow the whole 480s budget.
2. evaluate_holdout() enforces an in-process wall-clock budget and returns a
   PARTIAL result (budget_exceeded=True) instead of the gtimeout wrapper
   SIGKILLing the process with no output at all.
3. Holdout score caching makes warm runs skip the 144-call holdout phase
   (24 examples x 3 samples x 2 programs) entirely.
"""

import time
from types import SimpleNamespace

from evolution.skills.evolve_skill import (
    DEFAULT_LLM_RETRIES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    _holdout_cache_key,
    evaluate_holdout,
    make_lm,
)


class FakeModule:
    """Duck-typed SkillModule: callable with task_input, controllable delay."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = 0

    def __call__(self, task_input):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return SimpleNamespace(output=f"response to {task_input}")


def _examples(n: int):
    return [SimpleNamespace(task_input=f"task {i}") for i in range(n)]


def _metric(ex, pred):
    return 0.5


def _key_fn(ex, program, sample_idx):
    return f"{ex.task_input}|{program}|{sample_idx}"


class TestMakeLm:
    def test_per_call_timeout_is_bounded(self):
        lm = make_lm("deepseek/deepseek-chat")
        # dspy passes kwargs through to litellm; the litellm default is 600s,
        # which alone exceeds the 480s run budget.
        timeout = lm.kwargs.get("timeout") or 600
        assert timeout <= DEFAULT_LLM_TIMEOUT_SECONDS

    def test_retries_are_capped(self):
        lm = make_lm("deepseek/deepseek-chat")
        assert lm.num_retries <= DEFAULT_LLM_RETRIES


class TestHoldoutBudgetGuard:
    def test_returns_partial_result_when_budget_exceeded(self):
        base = FakeModule(delay=0.05)
        evolved = FakeModule(delay=0.05)
        exs = _examples(6)
        result = evaluate_holdout(
            exs, base, evolved, samples=1, metric=_metric,
            max_budget_seconds=0.15,
        )
        assert result["budget_exceeded"] is True
        assert 0 < result["examples_evaluated"] < len(exs)
        assert len(result["baseline_scores"]) == result["examples_evaluated"]
        assert len(result["evolved_scores"]) == result["examples_evaluated"]
        # Each complete example makes exactly 2 calls; the budget check runs
        # per-sample, so the interrupted example may add at most 1 baseline
        # call before the loop stops.
        assert 2 * result["examples_evaluated"] <= result["calls_made"] <= 2 * result["examples_evaluated"] + 1

    def test_completes_within_generous_budget(self):
        base = FakeModule(delay=0.001)
        evolved = FakeModule(delay=0.001)
        exs = _examples(4)
        t0 = time.monotonic()
        result = evaluate_holdout(
            exs, base, evolved, samples=3, metric=_metric,
            max_budget_seconds=10.0,
        )
        elapsed = time.monotonic() - t0
        assert result["budget_exceeded"] is False
        assert result["examples_evaluated"] == len(exs)
        assert len(result["baseline_scores"]) == len(exs)
        assert len(result["evolved_scores"]) == len(exs)
        assert elapsed < 10.0

    def test_zero_budget_yields_clean_partial(self):
        exs = _examples(2)
        result = evaluate_holdout(
            exs, FakeModule(), FakeModule(), samples=3, metric=_metric,
            max_budget_seconds=0.0,
        )
        assert result["budget_exceeded"] is True
        assert result["examples_evaluated"] == 0
        assert result["baseline_scores"] == []
        assert result["evolved_scores"] == []


class TestHoldoutScoreCache:
    def test_warm_run_skips_llm_calls(self):
        base = FakeModule()
        evolved = FakeModule()
        exs = _examples(3)
        cache = {}
        evaluate_holdout(
            exs, base, evolved, samples=1, metric=_metric,
            score_cache=cache, cache_key=_key_fn,
        )
        calls_after_first = base.calls + evolved.calls
        assert calls_after_first == len(exs) * 2  # 3 examples x 2 programs

        result = evaluate_holdout(
            exs, base, evolved, samples=1, metric=_metric,
            score_cache=cache, cache_key=_key_fn,
        )
        assert result["cache_hits"] == len(exs) * 2
        assert base.calls + evolved.calls == calls_after_first  # zero new calls

    def test_cache_key_is_stable_and_sensitive(self):
        k1 = _holdout_cache_key("deepseek/deepseek-chat", "abc123", "baseline", "task 1", 0)
        k2 = _holdout_cache_key("deepseek/deepseek-chat", "abc123", "baseline", "task 1", 0)
        k3 = _holdout_cache_key("deepseek/deepseek-chat", "abc123", "baseline", "task 1", 1)
        k4 = _holdout_cache_key("deepseek/deepseek-chat", "abc123", "evolved", "task 1", 0)
        k5 = _holdout_cache_key("deepseek/deepseek-chat", "diffhash", "baseline", "task 1", 0)
        assert k1 == k2
        assert len({k1, k3, k4, k5}) == 4

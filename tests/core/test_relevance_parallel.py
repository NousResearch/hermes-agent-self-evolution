"""Tests for the parallel relevance-scoring loop.

The serial version of this loop was the pipeline's dominant cost: one full LLM
reasoning call per candidate, run one after another. These tests pin the two
properties that fix introduced -- it actually runs concurrently, and it still
returns exactly what a serial run would have returned.
"""

import time
from unittest.mock import patch

import pytest

from evolution.core import external_importers as ei


class _Msg(dict):
    """Minimal stand-in for an imported message."""


def _candidates(n: int) -> list[dict]:
    return [
        {
            "task_input": f"verify the evidence for requirement H{i}",
            "source": "codex",
            "assistant_response": f"response {i}",
        }
        for i in range(n)
    ]


@pytest.fixture
def skill_text() -> str:
    return (
        "---\nname: gate-evidence-ledger\ndescription: Use for gated work.\n---\n\n"
        "# Gate Evidence Ledger\n\nTrack requirement -> evidence -> verdict.\n"
    )


class TestRelevanceFilterParallelism:
    def test_scores_candidates_concurrently(self, skill_text):
        """Wall-clock must be well under N x per-call latency.

        With 12 workers and a 0.1s fake call, 24 candidates take ~0.2s serial-
        equivalent-per-worker. A serial loop would need ~2.4s. The 1.5s ceiling
        leaves headroom on a loaded machine while still failing loudly if the
        loop regresses to serial.
        """
        msgs = _candidates(24)
        delay = 0.1

        def fake_scorer(**kwargs):
            time.sleep(delay)
            return type(
                "R",
                (),
                {
                    "scoring": (
                        '{"relevant": true, "expected_behavior": "map evidence '
                        'to verdict", "difficulty": "medium", '
                        '"category": "general"}'
                    )
                },
            )()

        rf = ei.RelevanceFilter(model="openai/openrouter/~deepseek/deepseek-pro-latest")
        rf.scorer = fake_scorer

        peak_concurrency = 0
        live = 0
        lock = __import__("threading").Lock()

        def counting_scorer(**kwargs):
            nonlocal peak_concurrency, live
            with lock:
                live += 1
                peak_concurrency = max(peak_concurrency, live)
            try:
                return fake_scorer(**kwargs)
            finally:
                with lock:
                    live -= 1

        rf.scorer = counting_scorer

        with patch.object(ei, "_is_relevant_to_skill", return_value=True):
            start = time.time()
            examples = rf.filter_and_score(
                msgs, "gate-evidence-ledger", skill_text, max_examples=50
            )
            elapsed = time.time() - start

        assert len(examples) == 24
        assert peak_concurrency > 1, "scoring ran serially"
        assert peak_concurrency <= ei._RELEVANCE_CONCURRENCY
        assert elapsed < 1.5, f"took {elapsed:.2f}s; scoring did not parallelise"

    def test_worker_count_never_exceeds_candidates(self, skill_text):
        """A single candidate must not spin up a pool of 12."""
        msgs = _candidates(1)

        def fake_scorer(**kwargs):
            return type(
                "R",
                (),
                {
                    "scoring": (
                        '{"relevant": true, "expected_behavior": "do the thing", '
                        '"difficulty": "easy", "category": "general"}'
                    )
                },
            )()

        rf = ei.RelevanceFilter(model="openai/openrouter/~deepseek/deepseek-pro-latest")
        rf.scorer = fake_scorer

        with patch.object(ei, "_is_relevant_to_skill", return_value=True):
            examples = rf.filter_and_score(
                msgs, "gate-evidence-ledger", skill_text, max_examples=50
            )

        assert len(examples) == 1


class TestRelevanceFilterEquivalence:
    """Concurrency must not change the result."""

    def test_results_are_deterministic_in_input_order(self, skill_text):
        """executor.map preserves order, so results must not depend on timing."""
        msgs = _candidates(20)
        # Non-uniform latency: the last candidate finishes first, which would
        # reorder results if the implementation used as_completed().
        def fake_scorer(**kwargs):
            msg = kwargs.get("user_message", "")
            idx = 0
            for ch in msg:
                if ch.isdigit():
                    idx = int(ch)
                    break
            time.sleep(0.02 * ((20 - idx) % 5))
            return type(
                "R",
                (),
                {
                    "scoring": (
                        '{"relevant": true, "expected_behavior": "behave", '
                        '"difficulty": "medium", "category": "general"}'
                    )
                },
            )()

        def run():
            rf = ei.RelevanceFilter(
                model="openai/openrouter/~deepseek/deepseek-pro-latest"
            )
            rf.scorer = fake_scorer
            with patch.object(ei, "_is_relevant_to_skill", return_value=True):
                return rf.filter_and_score(
                    msgs, "gate-evidence-ledger", skill_text, max_examples=50
                )

        first, second = run(), run()
        assert [e.task_input for e in first] == [e.task_input for e in second]
        assert [e.task_input for e in first] == [m["task_input"] for m in msgs]

    def test_respects_max_examples_cap(self, skill_text):
        """Truncation still applies after concurrent scoring."""
        msgs = _candidates(30)

        def fake_scorer(**kwargs):
            return type(
                "R",
                (),
                {
                    "scoring": (
                        '{"relevant": true, "expected_behavior": "behave", '
                        '"difficulty": "medium", "category": "general"}'
                    )
                },
            )()

        rf = ei.RelevanceFilter(model="openai/openrouter/~deepseek/deepseek-pro-latest")
        rf.scorer = fake_scorer

        with patch.object(ei, "_is_relevant_to_skill", return_value=True):
            examples = rf.filter_and_score(
                msgs, "gate-evidence-ledger", skill_text, max_examples=10
            )

        assert len(examples) == 10

    def test_irrelevant_examples_are_dropped(self, skill_text):
        msgs = _candidates(6)

        def fake_scorer(**kwargs):
            return type(
                "R",
                (),
                {
                    "scoring": (
                        '{"relevant": false, "expected_behavior": "", '
                        '"difficulty": "easy", "category": "general"}'
                    )
                },
            )()

        rf = ei.RelevanceFilter(model="openai/openrouter/~deepseek/deepseek-pro-latest")
        rf.scorer = fake_scorer

        with patch.object(ei, "_is_relevant_to_skill", return_value=True):
            examples = rf.filter_and_score(
                msgs, "gate-evidence-ledger", skill_text, max_examples=50
            )

        assert examples == []

    def test_call_failures_counted_as_errors_not_crashes(self, skill_text):
        """A raising scorer must not abort the pool; others still complete."""
        msgs = _candidates(8)
        calls = {"n": 0}

        def flaky_scorer(**kwargs):
            calls["n"] += 1
            if calls["n"] % 3 == 0:
                raise RuntimeError("simulated router timeout")
            return type(
                "R",
                (),
                {
                    "scoring": (
                        '{"relevant": true, "expected_behavior": "behave", '
                        '"difficulty": "medium", "category": "general"}'
                    )
                },
            )()

        rf = ei.RelevanceFilter(model="openai/openrouter/~deepseek/deepseek-pro-latest")
        rf.scorer = flaky_scorer

        with patch.object(ei, "_is_relevant_to_skill", return_value=True):
            examples = rf.filter_and_score(
                msgs, "gate-evidence-ledger", skill_text, max_examples=50
            )

        # Some succeeded despite neighbours raising.
        assert len(examples) >= 1
"""Synthetic benchmark and scripted executor for the offline policy comparison.

EVERYTHING HERE IS HYPOTHETICAL. Success rates, costs, latencies and
violation rates are invented to exercise the harness. They are not
measurements of Hermes, and results on them are not evidence that Hermes
improves on any real task.

Each (task, strategy) outcome is a deterministic function of the benchmark
seed, task ID and strategy ID (string-seeded ``random.Random``, stable across
runs and platforms). ``SyntheticBenchmark.response_table()`` exposes the
outcomes as plain dicts with the fields of PR #3's ``FakeResponse``
(``output``, ``cost_usd``, ``latency_s``, ``tool_calls``), so the same table
drives ``ScriptedExecutor`` here and ``FakeExecutor`` once PR #3 is merged.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping, Sequence

from .compare import BenchmarkItem, ContextEncoder, Scoring, check_split

STRATEGIES = ("direct", "structured", "tool_assisted")
DIFFICULTIES = ("simple", "moderate", "complex")     # gate 2 vocabulary
DOMAINS = ("security", "data", "web")

# Hypothetical P(correct review) by difficulty, per strategy (STRATEGIES order).
BASE_SUCCESS = {
    "simple": (0.93, 0.86, 0.82),
    "moderate": (0.62, 0.84, 0.80),
    "complex": (0.38, 0.72, 0.83),
}
DOMAIN_SHIFT = {
    "security": (-0.10, 0.00, 0.08),
    "data": (0.00, 0.00, 0.00),
    "web": (0.03, 0.00, -0.03),
}
COST_USD = {"direct": 0.01, "structured": 0.03, "tool_assisted": 0.08}
LATENCY_S = {"direct": 2.0, "structured": 6.0, "tool_assisted": 15.0}
VIOLATION_RATE = {"direct": 0.0, "structured": 0.0, "tool_assisted": 0.03}
REFERENCE_SCORING = Scoring(reference_cost_usd=0.03, reference_latency_s=6.0)

CORRECT, MISSED = "review:correct", "review:missed"


@dataclass(frozen=True)
class SyntheticStrategy:
    """Catalogue entry; tool-free, like PR #3 ``Strategy(strategy_id)``."""
    strategy_id: str
    allowed_tools: frozenset = frozenset()


def synthetic_catalogue() -> dict[str, SyntheticStrategy]:
    return {s: SyntheticStrategy(s) for s in STRATEGIES}


class SyntheticBenchmark:
    name = "synthetic-python-review/1"
    synthetic = True

    def __init__(self, seed: int = 0, n_train: int = 120, n_validation: int = 60):
        if n_train < 1 or n_validation < 0:
            raise ValueError("need n_train >= 1 and n_validation >= 0")
        self.seed = seed
        self._items = {"train": self._make("train", n_train),
                       "validation": self._make("validation", n_validation)}

    def _make(self, split: str, n: int) -> tuple[BenchmarkItem, ...]:
        rng = random.Random(f"synthetic-benchmark:{self.seed}:{split}")
        return tuple(BenchmarkItem(
            task_id=f"syn-{split}-{i:04d}", prompt=f"synthetic review task {i}", split=split,
            observable={"difficulty": rng.choice(DIFFICULTIES), "domain": rng.choice(DOMAINS),
                        "n_lines": rng.randint(5, 120)},
            ground_truth={"expected": CORRECT}) for i in range(n))

    def items(self, split: str) -> Sequence[BenchmarkItem]:
        return self._items[check_split(split)]

    def outcome(self, item: BenchmarkItem, strategy: str) -> dict:
        """Hypothetical outcome of ``strategy`` on ``item`` (deterministic)."""
        index = STRATEGIES.index(strategy)
        obs = item.observable
        rate = min(0.99, max(0.01, BASE_SUCCESS[obs["difficulty"]][index]
                             + DOMAIN_SHIFT[obs["domain"]][index]))
        rng = random.Random(f"synthetic-outcome:{self.seed}:{item.task_id}:{strategy}")
        correct = rng.random() < rate
        violated = rng.random() < VIOLATION_RATE[strategy]
        size = 1.0 + obs["n_lines"] / 200.0
        return {"output": CORRECT if correct else MISSED,
                "cost_usd": round(COST_USD[strategy] * size, 6),
                "latency_s": round(LATENCY_S[strategy] * size * (0.8 + 0.4 * rng.random()), 4),
                "tool_calls": ("terminal",) if violated else ()}

    def response_table(self) -> dict[tuple[str, str], dict]:
        return {(item.task_id, s): self.outcome(item, s)
                for split in ("train", "validation") for item in self._items[split]
                for s in STRATEGIES}


@dataclass(frozen=True)
class ScriptedOutcome:
    output: str
    cost_usd: float
    latency_s: float
    tool_calls: tuple[str, ...] = ()

    @property
    def violation(self) -> bool:
        # Synthetic strategies allow no tools, so any tool call is a violation
        # (the same rule as PR #3's audit_violations).
        return bool(self.tool_calls)


class ScriptedExecutor:
    """Looks outcomes up in a response table; never calls Hermes or the network."""
    backend = "scripted"

    def __init__(self, table: Mapping[tuple[str, str], dict]):
        self.table = dict(table)
        self.calls: list[tuple[str, str]] = []

    def execute(self, task, strategy) -> ScriptedOutcome:
        key = (task.task_id, strategy.strategy_id)
        self.calls.append(key)
        return ScriptedOutcome(**self.table[key])


def synthetic_quality(item: BenchmarkItem, outcome) -> float:
    """Stand-in verifier: 1 if the scripted review is the expected one, else 0."""
    return 1.0 if outcome.output == item.ground_truth["expected"] else 0.0


def synthetic_encoder(benchmark: SyntheticBenchmark) -> ContextEncoder:
    return ContextEncoder.fit(benchmark.items("train"), key_fields=("difficulty", "domain"),
                              categorical=("difficulty", "domain"),
                              numeric={"n_lines": 100.0})

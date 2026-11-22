"""Minimal, dependency-free MAPE-K control primitives.

This module does NOT train model weights. It selects strategy IDs from a
pre-approved catalogue and learns from externally verified task outcomes.
The independent verifier and Hermes executor are injected by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import random
import sqlite3
from pathlib import Path

from .benchmark import optimisation_phase

# Reward contract (see docs/sprints/KAFO-EVO-001.md, "Reward range"):
#   valid outcome  -> quality - cost_weight*min(cost_ratio, RATIO_CAP)
#                             - time_weight*min(time_ratio, RATIO_CAP)
#                     in [-(cost_weight + time_weight) * RATIO_CAP, 1.0]
#                     = [-0.30, 1.0] with the default weights
#   violation      -> VIOLATION_REWARD (-1.0), strictly below every valid outcome
RATIO_CAP = 2.0
VIOLATION_REWARD = -1.0


def _is_real(value) -> bool:
    """True for finite int/float values; False for bools, NaN, +/-inf, others."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def validate_reward(value) -> float:
    """Return value as float, or raise ValueError if it is not a finite number."""
    if not _is_real(value):
        raise ValueError(f"reward must be a finite real number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Reward:
    quality: float       # verified, normalised [0, 1]
    cost_ratio: float    # nonnegative cost / fixed reference cost
    time_ratio: float    # nonnegative duration / fixed reference duration
    violation: bool = False


def reward_bounds(cost_weight: float = 0.10,
                  time_weight: float = 0.05) -> tuple[float, float]:
    """Inclusive (min, max) reward of a valid (non-violating) outcome."""
    return -(cost_weight + time_weight) * RATIO_CAP, 1.0


def score_outcome(outcome: Reward, cost_weight: float = 0.10,
                  time_weight: float = 0.05) -> float:
    """Fixed utility function; critical violations receive disqualifying reward.

    Cost and latency penalties are capped at RATIO_CAP times their weight, so
    any valid outcome scores strictly above VIOLATION_REWARD.
    Do not expose this function to the optimising agent as a writable artifact.
    """
    for name in ("quality", "cost_ratio", "time_ratio"):
        if not _is_real(getattr(outcome, name)):
            raise ValueError(f"{name} must be a finite real number")
    if not isinstance(outcome.violation, bool):
        raise ValueError("violation must be a bool")
    if not 0.0 <= outcome.quality <= 1.0:
        raise ValueError("quality must be between 0 and 1")
    if outcome.cost_ratio < 0 or outcome.time_ratio < 0:
        raise ValueError("cost/time ratios must be nonnegative")
    if not (_is_real(cost_weight) and _is_real(time_weight)) \
            or cost_weight < 0 or time_weight < 0:
        raise ValueError("penalty weights must be finite and nonnegative")
    if reward_bounds(cost_weight, time_weight)[0] <= VIOLATION_REWARD:
        raise ValueError("penalty weights too large: a valid outcome could "
                         "score as low as a violation")
    if outcome.violation:
        return VIOLATION_REWARD
    return (outcome.quality
            - cost_weight * min(outcome.cost_ratio, RATIO_CAP)
            - time_weight * min(outcome.time_ratio, RATIO_CAP))


class ContextualBandit:
    """Seedable epsilon-greedy, per-context value estimates and action counts.

    This is the FIRST baseline, not LinUCB or weight-level RL. In each context
    every strategy is tried once (in catalogue order) before epsilon-greedy
    selection starts, so no action is starved by a zero initial estimate.
    """

    def __init__(self, strategies: tuple[str, ...], epsilon: float = 0.15,
                 seed: int = 0, version: str = "epsilon-greedy/1"):
        if not strategies or len(set(strategies)) != len(strategies):
            raise ValueError("strategies must be a nonempty set of unique IDs")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if not version:
            raise ValueError("policy version cannot be empty")
        self.strategies = strategies
        self.epsilon = epsilon
        self.version = version
        self.rng = random.Random(seed)
        self.counts: dict[tuple[str, str], int] = {}
        self.means: dict[tuple[str, str], float] = {}

    def select(self, context: str) -> tuple[str, float]:
        if not context:
            raise ValueError("context cannot be empty")
        # Initial sweep: deterministic, so the logged propensity is 1.0.
        for action in self.strategies:
            if self.counts.get((context, action), 0) == 0:
                return action, 1.0
        # Explicit uniform exploration plus uniform random tie resolution.
        values = [self.means[(context, a)] for a in self.strategies]
        top = max(values)
        winners = [a for a, v in zip(self.strategies, values) if v == top]
        greedy_prob = (1 - self.epsilon) / len(winners)
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.strategies)
        else:
            action = self.rng.choice(winners)
        propensity = self.epsilon / len(self.strategies)
        if action in winners:
            propensity += greedy_prob
        return action, propensity

    def update(self, context: str, strategy: str, reward: float) -> None:
        if strategy not in self.strategies:
            raise ValueError("unknown strategy")
        if not context:
            raise ValueError("context cannot be empty")
        reward = validate_reward(reward)  # before any state is touched
        key = (context, strategy)
        n = self.counts.get(key, 0) + 1
        mean = self.means.get(key, 0.0)
        self.means[key] = mean + (reward - mean) / n
        self.counts[key] = n


class ExperienceStore:
    """Local research history only; no enterprise audit or user data required.

    Usable as a context manager. Databases created before provenance columns
    existed are migrated in place; their old rows keep NULL provenance.
    """

    SCHEMA_VERSION = 2
    _PROVENANCE_COLUMNS = ("policy_version", "skill_version", "recorded_at")

    def __init__(self, path: str | Path):
        self.conn = sqlite3.connect(str(path))
        try:
            self._migrate()
        except Exception:
            self.conn.close()
            raise

    def _migrate(self) -> None:
        self.conn.execute("""CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            context TEXT NOT NULL, strategy TEXT NOT NULL,
            propensity REAL NOT NULL, reward REAL NOT NULL,
            feedback_json TEXT NOT NULL
        )""")
        existing = {row[1] for row in self.conn.execute(
            "PRAGMA table_info(experiences)")}
        for column in self._PROVENANCE_COLUMNS:
            if column not in existing:
                self.conn.execute(
                    f"ALTER TABLE experiences ADD COLUMN {column} TEXT")
        self.conn.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")
        self.conn.commit()

    def record(self, context: str, strategy: str, propensity: float,
               reward: float, feedback: dict, *, policy_version: str,
               skill_version: str, recorded_at: str | None = None) -> None:
        if not _is_real(propensity) or not 0 < propensity <= 1:
            raise ValueError("invalid logged selection probability")
        reward = validate_reward(reward)
        if not policy_version or not skill_version:
            raise ValueError("policy_version and skill_version are required")
        if recorded_at is None:
            recorded_at = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO experiences (context,strategy,propensity,reward,"
            "feedback_json,policy_version,skill_version,recorded_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (context, strategy, propensity, reward,
             json.dumps(feedback, allow_nan=False),
             policy_version, skill_version, recorded_at))
        self.conn.commit()

    def rows(self) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT id,context,strategy,propensity,reward,feedback_json,"
            "policy_version,skill_version,recorded_at "
            "FROM experiences ORDER BY id")
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor]

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ExperienceStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def choose_strategy(policy: ContextualBandit, context: str,
                    executor, verifier, store: ExperienceStore, *,
                    skill_version: str) -> dict:
    """One complete Monitor/Analyse/Plan/Legitimise/Execute/Knowledge iteration.

    Caller supplies a Hermes-compatible executor(context, strategy), an
    independent verifier(result)->Reward and the version of the skill
    catalogue in use. No production actions permitted by this module.
    The caller is responsible for sandboxing the executor.

    The iteration runs inside ``benchmark.optimisation_phase()``: the executor
    and verifier cannot load or open the sealed test split.
    """
    strategy, propensity = policy.select(context)   # PLAN
    with optimisation_phase():
        result = executor(context, strategy)        # EXECUTE
        outcome = verifier(result)                  # MONITOR / ANALYSE
    if not isinstance(outcome, Reward):
        raise TypeError("verifier must return Reward")
    reward = score_outcome(outcome)                  # LEGITIMISE
    store.record(context, strategy, propensity, reward,
                 {"quality": outcome.quality, "cost_ratio": outcome.cost_ratio,
                  "time_ratio": outcome.time_ratio, "violation": outcome.violation},
                 policy_version=policy.version, skill_version=skill_version)
    policy.update(context, strategy, reward)        # KNOWLEDGE
    return {"context": context, "strategy": strategy,
            "propensity": propensity, "reward": reward}

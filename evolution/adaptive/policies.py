"""Strategy-selection policies behind one interface (KAFO-EVO-001 gate 4).

Every policy implements ``Policy``:

* ``select(context) -> (strategy_id, propensity)`` -- ``propensity`` is the
  probability with which this policy chose that action, given the context and
  its state at decision time (for later off-policy evaluation).
* ``update(context, strategy_id, reward)`` -- learn from a scored outcome.
* ``version`` -- recorded next to every decision.

A ``Context`` carries only features observable BEFORE the decision: a discrete
``key`` (used by the tabular epsilon-greedy policy) and a numeric ``features``
vector (used by LinUCB). Policies never see outcomes until ``update``.

Pure Python, stdlib only, seedable. No model weights are trained.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Protocol, runtime_checkable

from .core import ContextualBandit, validate_reward

_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Context:
    """Decision-time context. Built from observable task metadata only."""
    key: str
    features: tuple[float, ...]

    def __post_init__(self):
        if not self.key or not isinstance(self.key, str):
            raise ValueError("context key must be a nonempty string")
        if not isinstance(self.features, tuple) or not self.features:
            raise ValueError("context features must be a nonempty tuple")
        for value in self.features:
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value):
                raise ValueError("context features must be finite real numbers")


@runtime_checkable
class Policy(Protocol):
    strategies: tuple[str, ...]
    version: str

    def select(self, context: Context) -> tuple[str, float]: ...

    def update(self, context: Context, strategy: str, reward: float) -> None: ...


def _check_strategies(strategies) -> tuple[str, ...]:
    strategies = tuple(strategies)
    if not strategies or len(set(strategies)) != len(strategies) \
            or not all(isinstance(s, str) and s for s in strategies):
        raise ValueError("strategies must be a nonempty set of unique IDs")
    return strategies


def _check_update(policy: Policy, context: Context, strategy: str, reward) -> float:
    if not isinstance(context, Context):
        raise TypeError("context must be a Context")
    if strategy not in policy.strategies:
        raise ValueError(f"unknown strategy {strategy!r}")
    return validate_reward(reward)


class FixedPolicy:
    """Always the same strategy. Non-adaptive baseline; propensity 1."""

    def __init__(self, strategies, action: str):
        self.strategies = _check_strategies(strategies)
        if action not in self.strategies:
            raise ValueError(f"unknown strategy {action!r}")
        self.action = action
        self.version = f"fixed/1:{action}"

    def select(self, context: Context) -> tuple[str, float]:
        return self.action, 1.0

    def update(self, context: Context, strategy: str, reward: float) -> None:
        _check_update(self, context, strategy, reward)   # validated, then ignored


class RandomPolicy:
    """Uniformly random strategy. Non-adaptive baseline; propensity 1/K."""

    def __init__(self, strategies, seed: int = 0):
        self.strategies = _check_strategies(strategies)
        self.rng = random.Random(seed)
        self.version = "random/1"

    def select(self, context: Context) -> tuple[str, float]:
        return self.rng.choice(self.strategies), 1.0 / len(self.strategies)

    def update(self, context: Context, strategy: str, reward: float) -> None:
        _check_update(self, context, strategy, reward)


class EpsilonGreedyPolicy:
    """The existing ``core.ContextualBandit``, keyed on ``context.key``."""

    def __init__(self, strategies, epsilon: float = 0.15, seed: int = 0):
        self.bandit = ContextualBandit(_check_strategies(strategies), epsilon=epsilon, seed=seed)
        self.strategies = self.bandit.strategies
        self.version = f"{self.bandit.version}:eps={epsilon:g}"

    def select(self, context: Context) -> tuple[str, float]:
        if not isinstance(context, Context):
            raise TypeError("context must be a Context")
        return self.bandit.select(context.key)

    def update(self, context: Context, strategy: str, reward: float) -> None:
        reward = _check_update(self, context, strategy, reward)
        self.bandit.update(context.key, strategy, reward)


class LinUCBPolicy:
    """Disjoint LinUCB (Li et al., 2010, Algorithm 1) over ``context.features``.

    Per strategy ``a``: ``A_a = ridge * I + sum x x^T``, ``b_a = sum r x``,
    ``theta_a = A_a^-1 b_a``. Choose ``argmax theta_a . x + alpha * sqrt(x^T A_a^-1 x)``.
    ``A_a^-1`` is maintained directly with the Sherman-Morrison update, so no
    matrix library is needed. Exact ties are broken uniformly at random with a
    seeded RNG; the propensity is then ``1 / number of tied strategies``,
    otherwise 1 (LinUCB is deterministic given its state).
    """

    def __init__(self, strategies, dim: int, alpha: float = 1.0, ridge: float = 1.0,
                 seed: int = 0):
        self.strategies = _check_strategies(strategies)
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError("dim must be a positive int")
        for name, value in (("alpha", alpha), ("ridge", ridge)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if ridge == 0:
            raise ValueError("ridge must be positive")
        self.dim, self.alpha, self.ridge = dim, float(alpha), float(ridge)
        self.rng = random.Random(seed)
        self.version = f"linucb/1:alpha={alpha:g},ridge={ridge:g}"
        identity_inv = [[(1.0 / ridge if i == j else 0.0) for j in range(dim)]
                        for i in range(dim)]
        self.a_inv = {a: [row[:] for row in identity_inv] for a in self.strategies}
        self.b = {a: [0.0] * dim for a in self.strategies}
        self.counts = {a: 0 for a in self.strategies}

    def _x(self, context: Context) -> list[float]:
        if not isinstance(context, Context):
            raise TypeError("context must be a Context")
        if len(context.features) != self.dim:
            raise ValueError(f"expected {self.dim} features, got {len(context.features)}")
        return [float(v) for v in context.features]

    @staticmethod
    def _matvec(m: list[list[float]], v: list[float]) -> list[float]:
        return [sum(mij * vj for mij, vj in zip(row, v)) for row in m]

    def scores(self, context: Context) -> dict[str, float]:
        """Upper confidence bound per strategy for this context."""
        x = self._x(context)
        out = {}
        for a in self.strategies:
            a_inv_x = self._matvec(self.a_inv[a], x)
            theta = self._matvec(self.a_inv[a], self.b[a])
            mean = sum(t * xi for t, xi in zip(theta, x))
            width = math.sqrt(max(0.0, sum(xi * yi for xi, yi in zip(x, a_inv_x))))
            out[a] = mean + self.alpha * width
        return out

    def select(self, context: Context) -> tuple[str, float]:
        scores = self.scores(context)
        top = max(scores.values())
        tied = [a for a in self.strategies if top - scores[a] <= _TIE_TOLERANCE]
        return self.rng.choice(tied), 1.0 / len(tied)

    def update(self, context: Context, strategy: str, reward: float) -> None:
        reward = _check_update(self, context, strategy, reward)
        x = self._x(context)
        a_inv = self.a_inv[strategy]
        a_inv_x = self._matvec(a_inv, x)
        denom = 1.0 + sum(xi * yi for xi, yi in zip(x, a_inv_x))
        # Sherman-Morrison; A_inv is symmetric, so x^T A_inv = (A_inv x)^T.
        self.a_inv[strategy] = [[a_inv[i][j] - a_inv_x[i] * a_inv_x[j] / denom
                                 for j in range(self.dim)] for i in range(self.dim)]
        self.b[strategy] = [bi + reward * xi for bi, xi in zip(self.b[strategy], x)]
        self.counts[strategy] += 1

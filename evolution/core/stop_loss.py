"""Budget and runtime stop-loss guard for optimization phases."""

from dataclasses import dataclass, field

from evolution.core.config import EvolutionConfig


@dataclass(frozen=True)
class OptimizationAttempt:
    """One optimization attempt with cost and quality metadata."""

    cost_usd: float
    runtime_minutes: float
    improvement: float
    stable: bool


@dataclass
class StopLossGuard:
    """Tracks phase spend/time and stops runaway optimization loops."""

    config: EvolutionConfig
    attempts: list[OptimizationAttempt] = field(default_factory=list)

    def register_attempt(
        self,
        *,
        cost_usd: float,
        runtime_minutes: float,
        improvement: float,
        stable: bool,
    ) -> None:
        """Record one attempt for stop-loss evaluation."""
        self.attempts.append(
            OptimizationAttempt(
                cost_usd=cost_usd,
                runtime_minutes=runtime_minutes,
                improvement=improvement,
                stable=stable,
            ),
        )

    @property
    def total_cost_usd(self) -> float:
        return sum(attempt.cost_usd for attempt in self.attempts)

    @property
    def total_runtime_minutes(self) -> float:
        return sum(attempt.runtime_minutes for attempt in self.attempts)

    def termination_reasons(self) -> list[str]:
        """Return reasons why the phase should terminate."""
        reasons: list[str] = []

        if self.total_cost_usd > self.config.max_phase_budget_usd:
            reasons.append(
                "Phase budget exceeded "
                f"({self.total_cost_usd:.2f} > {self.config.max_phase_budget_usd:.2f} USD)",
            )

        if self.total_runtime_minutes > self.config.max_phase_runtime_minutes:
            reasons.append(
                "Phase runtime exceeded "
                f"({self.total_runtime_minutes:.1f} > {self.config.max_phase_runtime_minutes} min)",
            )

        n = self.config.required_stable_runs
        if len(self.attempts) >= n:
            recent = self.attempts[-n:]
            stable_wins = [
                attempt
                for attempt in recent
                if attempt.stable and attempt.improvement >= self.config.minimum_detectable_effect
            ]
            if not stable_wins:
                reasons.append(
                    "No stable gain in recent attempts "
                    f"(window={n}, min_effect={self.config.minimum_detectable_effect:.3f})",
                )

        return reasons

    def should_terminate(self) -> bool:
        """True when stop-loss conditions have been met."""
        return bool(self.termination_reasons())

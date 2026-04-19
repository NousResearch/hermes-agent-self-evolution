"""Hard budget enforcement for codex-batched evolution runs.

This module is intentionally standalone and additive so it can survive upstream
repo updates with minimal merge friction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time


class BudgetExceeded(RuntimeError):
    """Raised when a run budget would be exceeded by starting a phase."""


@dataclass
class RunBudget:
    """Track hard execution limits for a single evolution run."""

    max_codex_calls: int
    max_run_seconds: float
    phase_timeout_seconds: float
    max_examples: int | None = None
    max_iterations: int | None = None
    max_candidates_per_iteration: int | None = None
    budget_strict: bool = True
    calls_used: int = 0
    started_at: float | None = field(default=None, init=False)

    def start_run(self) -> None:
        self.started_at = time.monotonic()

    def remaining_calls(self) -> int:
        return max(0, self.max_codex_calls - self.calls_used)

    def register_call(self, name: str) -> None:
        self.enforce_or_raise(name)
        self.calls_used += 1

    def can_start_phase(self, name: str) -> bool:
        if self.calls_used >= self.max_codex_calls:
            return False
        if self.started_at is None:
            return True
        elapsed = time.monotonic() - self.started_at
        if elapsed > self.max_run_seconds:
            return False
        remaining = self.max_run_seconds - elapsed
        if remaining < self.phase_timeout_seconds:
            return False
        return True

    def enforce_or_raise(self, name: str) -> None:
        if self.can_start_phase(name):
            return
        reason = self._failure_reason(name)
        raise BudgetExceeded(reason)

    def _failure_reason(self, name: str) -> str:
        if self.calls_used >= self.max_codex_calls:
            return (
                f"Cannot start phase '{name}': codex call budget exhausted "
                f"({self.calls_used}/{self.max_codex_calls})."
            )
        if self.started_at is not None:
            elapsed = time.monotonic() - self.started_at
            if elapsed > self.max_run_seconds:
                return (
                    f"Cannot start phase '{name}': run time budget exceeded "
                    f"({elapsed:.2f}s > {self.max_run_seconds}s)."
                )
            remaining = self.max_run_seconds - elapsed
            if remaining < self.phase_timeout_seconds:
                return (
                    f"Cannot start phase '{name}': remaining runtime budget {remaining:.2f}s "
                    f"is below required phase timeout {self.phase_timeout_seconds}s."
                )
        return f"Cannot start phase '{name}': budget restriction violated."

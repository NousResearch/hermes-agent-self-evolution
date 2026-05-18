"""Tests for stop-loss guardrail behavior."""

from evolution.core.config import EvolutionConfig
from evolution.core.stop_loss import StopLossGuard


def test_stop_loss_triggers_on_budget_exceeded():
    config = EvolutionConfig(max_phase_budget_usd=1.0)
    guard = StopLossGuard(config)
    guard.register_attempt(cost_usd=1.5, runtime_minutes=10, improvement=0.1, stable=True)
    assert guard.should_terminate()
    assert any("budget exceeded" in reason.lower() for reason in guard.termination_reasons())


def test_stop_loss_triggers_on_runtime_exceeded():
    config = EvolutionConfig(max_phase_runtime_minutes=5)
    guard = StopLossGuard(config)
    guard.register_attempt(cost_usd=0.1, runtime_minutes=6, improvement=0.1, stable=True)
    assert guard.should_terminate()
    assert any("runtime exceeded" in reason.lower() for reason in guard.termination_reasons())


def test_stop_loss_triggers_on_unstable_recent_runs():
    config = EvolutionConfig(required_stable_runs=3, minimum_detectable_effect=0.05)
    guard = StopLossGuard(config)
    guard.register_attempt(cost_usd=0.1, runtime_minutes=1, improvement=0.01, stable=False)
    guard.register_attempt(cost_usd=0.1, runtime_minutes=1, improvement=0.02, stable=False)
    guard.register_attempt(cost_usd=0.1, runtime_minutes=1, improvement=0.01, stable=False)
    assert guard.should_terminate()
    assert any("no stable gain" in reason.lower() for reason in guard.termination_reasons())


def test_stop_loss_allows_progress_when_stable_gains_present():
    config = EvolutionConfig(required_stable_runs=2, minimum_detectable_effect=0.01)
    guard = StopLossGuard(config)
    guard.register_attempt(cost_usd=0.1, runtime_minutes=1, improvement=0.03, stable=True)
    guard.register_attempt(cost_usd=0.1, runtime_minutes=1, improvement=0.02, stable=True)
    assert not guard.should_terminate()

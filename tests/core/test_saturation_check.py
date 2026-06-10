"""Tests for evolution.core.saturation_check.

All tests use hand-built scores or mock the LM/validator — zero real
LM spend. Pattern mirrors tests/core/test_closed_loop_feedback.py.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from evolution.core.saturation_check import (
    DEFAULT_THRESHOLDS,
    SaturationReport,
    _classify_band,
    saturation_preflight,
)


class TestClassifyBand:
    def test_healthy_when_synthetic_below_weak_threshold(self):
        band, _ = _classify_band(
            holdout_score=0.85, closed_loop_score=None, thresholds=DEFAULT_THRESHOLDS,
        )
        assert band == "healthy"

    def test_no_headroom_synthetic_only(self):
        band, suggestions = _classify_band(
            holdout_score=0.99, closed_loop_score=None, thresholds=DEFAULT_THRESHOLDS,
        )
        assert band == "no_headroom"
        assert any("harder" in s.lower() or "different target" in s.lower() for s in suggestions)

    def test_no_headroom_with_closed_loop_also_saturated(self):
        band, _ = _classify_band(
            holdout_score=0.99, closed_loop_score=0.98, thresholds=DEFAULT_THRESHOLDS,
        )
        assert band == "no_headroom"

    def test_weak_signal_when_closed_loop_in_middle_band(self):
        band, suggestions = _classify_band(
            holdout_score=0.97, closed_loop_score=0.60, thresholds=DEFAULT_THRESHOLDS,
        )
        assert band == "weak_signal"
        assert any("minibatch" in s.lower() or "iterations" in s.lower() for s in suggestions)

    def test_uniform_failure_when_closed_loop_below_threshold(self):
        band, suggestions = _classify_band(
            holdout_score=0.98, closed_loop_score=0.10, thresholds=DEFAULT_THRESHOLDS,
        )
        assert band == "uniform_failure"
        assert any("validator" in s.lower() or "stronger" in s.lower() for s in suggestions)
        # The "first check the validator actually ran" hint guards against
        # the historical silent-failure: hermes -m treated litellm-formatted
        # model strings as openrouter routing, broke auth, returned 0-turn
        # sessions, and the framework reported it as "validator too weak."
        # The hint points users at the run.log line that confirms routing.
        assert any(
            "stripped litellm" in s.lower() or "run.log" in s.lower() or "routed correctly" in s.lower()
            for s in suggestions
        )

    def test_boundary_exactly_at_no_headroom_synthetic_triggers(self):
        """0.99 exactly should trigger no_headroom (>= comparison)."""
        band, _ = _classify_band(
            holdout_score=0.99, closed_loop_score=None, thresholds=DEFAULT_THRESHOLDS,
        )
        assert band == "no_headroom"

    def test_boundary_just_below_no_headroom_does_not_trigger(self):
        band, _ = _classify_band(
            holdout_score=0.989, closed_loop_score=None, thresholds=DEFAULT_THRESHOLDS,
        )
        assert band == "healthy"

    def test_custom_thresholds_propagate(self):
        custom = {**DEFAULT_THRESHOLDS, "no_headroom_synthetic": 0.80}
        band, _ = _classify_band(
            holdout_score=0.85, closed_loop_score=None, thresholds=custom,
        )
        assert band == "no_headroom"

    def test_no_headroom_when_cl_saturated_and_synthetic_close(self):
        """The smoke case: synthetic 0.987 (below strict no_head_syn=0.99
        but above weak_syn=0.95), closed-loop 1.0. Both signals
        effectively pegged → no_headroom should trigger so the user
        doesn't burn GEPA budget on a hopeless run."""
        band, _ = _classify_band(
            holdout_score=0.987, closed_loop_score=1.0,
            thresholds=DEFAULT_THRESHOLDS,
        )
        assert band == "no_headroom"

    def test_healthy_when_cl_saturated_but_synthetic_low(self):
        """Edge case: behavioral suite pegged at 1.0 but synthetic at 0.5
        means there's real judge signal to optimize over (or the eval is
        misconfigured). Don't auto-abort — proceed and let GEPA try."""
        band, _ = _classify_band(
            holdout_score=0.5, closed_loop_score=1.0,
            thresholds=DEFAULT_THRESHOLDS,
        )
        assert band == "healthy"


class TestSaturationPreflightNoClosedLoop:
    def test_returns_healthy_when_baseline_below_threshold(self):
        baseline_module = MagicMock()
        holdout_examples = [MagicMock() for _ in range(5)]
        metric = MagicMock()
        lm = MagicMock()

        with patch(
            "evolution.core.saturation_check._score_baseline_on_holdout",
            return_value=(0.60, [0.6, 0.6, 0.6, 0.6, 0.6]),
        ):
            report = saturation_preflight(
                baseline_module=baseline_module,
                holdout_examples=holdout_examples,
                metric=metric,
                lm=lm,
            )

        assert report.band == "healthy"
        assert report.holdout_score == 0.60
        assert report.holdout_n == 5
        assert report.holdout_per_example == [0.6, 0.6, 0.6, 0.6, 0.6]
        assert report.closed_loop_score is None

    def test_returns_no_headroom_when_baseline_at_ceiling(self):
        with patch(
            "evolution.core.saturation_check._score_baseline_on_holdout",
            return_value=(1.0, [1.0] * 5),
        ):
            report = saturation_preflight(
                baseline_module=MagicMock(),
                holdout_examples=[MagicMock() for _ in range(5)],
                metric=MagicMock(),
                lm=MagicMock(),
            )

        assert report.band == "no_headroom"
        assert len(report.suggestions) >= 1

    def test_raises_on_empty_holdout(self):
        with pytest.raises(ValueError, match="holdout_examples"):
            saturation_preflight(
                baseline_module=MagicMock(),
                holdout_examples=[],
                metric=MagicMock(),
                lm=MagicMock(),
            )

    def test_attaches_noise_sidecar_when_suite_path_given(self, tmp_path):
        import json
        suite_path = tmp_path / "conv.jsonl"
        (tmp_path / "conv.jsonl.noise.json").write_text(json.dumps({
            "spurious_strict_win_rate": 0.1, "spurious_regression_rate": 0.0,
            "mean_per_task_flip": 0.0, "per_task_flip": {}, "runs": 10, "reps": 1,
            "suite_sha256": "x",
        }))
        with patch(
            "evolution.core.saturation_check._score_baseline_on_holdout",
            return_value=(0.6, [0.6] * 5),
        ):
            report = saturation_preflight(
                baseline_module=MagicMock(),
                holdout_examples=[MagicMock() for _ in range(5)],
                metric=MagicMock(), lm=MagicMock(),
                suite_path=suite_path,
            )
        assert report.noise is not None
        assert report.noise["runs"] == 10

    def test_noise_is_none_when_no_sidecar(self, tmp_path):
        with patch(
            "evolution.core.saturation_check._score_baseline_on_holdout",
            return_value=(0.6, [0.6] * 5),
        ):
            report = saturation_preflight(
                baseline_module=MagicMock(),
                holdout_examples=[MagicMock() for _ in range(5)],
                metric=MagicMock(), lm=MagicMock(),
                suite_path=tmp_path / "no_sidecar.jsonl",
            )
        assert report.noise is None


class TestSaturationPreflightWithClosedLoop:
    def _make_validation_report(self, *, n_pass: int, n_fail: int):
        """Build a minimal real ValidationReport whose evolved phase has the
        requested pass/fail counts. Uses real dataclasses (not MagicMock) so
        a future field rename breaks the test loudly."""
        from evolution.validation.report import (
            PhaseResult, TaskResult, ValidationReport, WinLoss,
        )
        passed_tasks = [
            TaskResult(
                task_id=f"p{i}", passed=True, abstained=False,
                tool_calls_seq=[], duration_seconds=0.0,
            )
            for i in range(n_pass)
        ]
        failed_tasks = [
            TaskResult(
                task_id=f"f{i}", passed=False, abstained=False,
                tool_calls_seq=[], duration_seconds=0.0,
            )
            for i in range(n_fail)
        ]
        tasks = passed_tasks + failed_tasks
        total = n_pass + n_fail
        phase = PhaseResult(
            pass_rate=n_pass / max(1, total),
            n_passed=n_pass,
            n_failed=n_fail,
            n_abstained=0,
            tasks=tasks,
        )
        delta = WinLoss(n_wins=0, n_losses=0, n_ties=total, pass_rate_change=0.0)
        return ValidationReport(
            schema_version="1",
            tool="t",
            task_suite_path="suite.jsonl",
            task_suite_sha256="x" * 64,
            baseline=phase,
            evolved=phase,
            delta=delta,
            decision="pass",
            decision_reasons=[],
        )

    def test_closed_loop_score_lands_in_report(self):
        cache = MagicMock()
        cache.force_run.return_value = self._make_validation_report(n_pass=3, n_fail=4)

        with patch(
            "evolution.core.saturation_check._score_baseline_on_holdout",
            return_value=(0.99, [1.0] * 5),
        ):
            report = saturation_preflight(
                baseline_module=MagicMock(),
                holdout_examples=[MagicMock() for _ in range(5)],
                metric=MagicMock(),
                lm=MagicMock(),
                closed_loop_cache=cache,
                baseline_artifact_text="baseline desc",
            )

        cache.force_run.assert_called_once_with("baseline desc")
        assert report.closed_loop_n == 7
        assert report.closed_loop_score == pytest.approx(3 / 7)
        assert report.closed_loop_per_example == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    def test_uniform_failure_band_triggers(self):
        cache = MagicMock()
        cache.force_run.return_value = self._make_validation_report(n_pass=0, n_fail=7)
        with patch(
            "evolution.core.saturation_check._score_baseline_on_holdout",
            return_value=(0.99, [1.0] * 5),
        ):
            report = saturation_preflight(
                baseline_module=MagicMock(),
                holdout_examples=[MagicMock() for _ in range(5)],
                metric=MagicMock(),
                lm=MagicMock(),
                closed_loop_cache=cache,
                baseline_artifact_text="b",
            )
        assert report.band == "uniform_failure"

    def test_weak_signal_band_triggers(self):
        cache = MagicMock()
        cache.force_run.return_value = self._make_validation_report(n_pass=4, n_fail=3)
        with patch(
            "evolution.core.saturation_check._score_baseline_on_holdout",
            return_value=(0.97, [1.0] * 5),
        ):
            report = saturation_preflight(
                baseline_module=MagicMock(),
                holdout_examples=[MagicMock() for _ in range(5)],
                metric=MagicMock(),
                lm=MagicMock(),
                closed_loop_cache=cache,
                baseline_artifact_text="b",
            )
        assert report.band == "weak_signal"

    def test_missing_baseline_text_raises(self):
        cache = MagicMock()
        with patch(
            "evolution.core.saturation_check._score_baseline_on_holdout",
            return_value=(0.5, [0.5]),
        ):
            with pytest.raises(ValueError, match="baseline_artifact_text"):
                saturation_preflight(
                    baseline_module=MagicMock(),
                    holdout_examples=[MagicMock()],
                    metric=MagicMock(), lm=MagicMock(),
                    closed_loop_cache=cache,
                    baseline_artifact_text=None,
                )


class TestRenderPanel:
    def _render_to_string(self, report: SaturationReport) -> str:
        from io import StringIO
        from rich.console import Console
        from evolution.core.saturation_check import render_saturation_panel

        buf = StringIO()
        console = Console(file=buf, width=100, color_system=None, force_terminal=False)
        render_saturation_panel(report, console=console)
        return buf.getvalue()

    def test_no_headroom_panel_includes_band_name_and_suggestion(self):
        report = SaturationReport(
            band="no_headroom", holdout_score=0.99, holdout_n=50,
            holdout_per_example=[1.0] * 50,
            suggestions=["Try a harder closed-loop suite", "Pick a different target"],
            thresholds=DEFAULT_THRESHOLDS,
        )
        out = self._render_to_string(report)
        assert "no_headroom" in out.lower() or "no headroom" in out.lower()
        assert "harder closed-loop suite" in out
        assert "0.99" in out

    def test_weak_signal_panel_shows_closed_loop_score(self):
        report = SaturationReport(
            band="weak_signal", holdout_score=0.97, holdout_n=50,
            holdout_per_example=[1.0] * 50,
            closed_loop_score=0.60, closed_loop_n=7, closed_loop_per_example=[],
            suggestions=["Bump iterations"], thresholds=DEFAULT_THRESHOLDS,
        )
        out = self._render_to_string(report)
        assert "0.60" in out or "60" in out
        assert "Bump iterations" in out

    def test_healthy_panel_is_terse(self):
        """healthy band should be one-line / minimal — most of the panel
        machinery is for the warn bands. This test just verifies it doesn't
        blow up."""
        report = SaturationReport(
            band="healthy", holdout_score=0.60, holdout_n=50,
            holdout_per_example=[0.6] * 50,
            suggestions=[], thresholds=DEFAULT_THRESHOLDS,
        )
        out = self._render_to_string(report)
        assert "healthy" in out.lower() or "passed" in out.lower()

    _NOISE = {
        "spurious_strict_win_rate": 0.125, "spurious_regression_rate": 0.0,
        "mean_per_task_flip": 0.05, "per_task_flip": {}, "runs": 8, "reps": 1,
        "suite_sha256": "x", "agent_model": "haiku",
    }

    def test_noise_row_renders_on_healthy_band_when_sidecar_present(self):
        report = SaturationReport(
            band="healthy", holdout_score=0.60, holdout_n=50,
            holdout_per_example=[0.6] * 50,
            suggestions=[], thresholds=DEFAULT_THRESHOLDS, noise=self._NOISE,
        )
        out = self._render_to_string(report)
        assert "Noise floor" in out
        assert "13%" in out or "12%" in out  # spurious strict-win 12.5%

    def test_noise_row_absent_when_no_sidecar(self):
        report = SaturationReport(
            band="healthy", holdout_score=0.60, holdout_n=50,
            holdout_per_example=[0.6] * 50,
            suggestions=[], thresholds=DEFAULT_THRESHOLDS, noise=None,
        )
        assert "Noise floor" not in self._render_to_string(report)

    def test_noise_row_renders_in_warn_panel(self):
        report = SaturationReport(
            band="no_headroom", holdout_score=0.99, holdout_n=50,
            holdout_per_example=[1.0] * 50,
            suggestions=["Try a harder closed-loop suite"],
            thresholds=DEFAULT_THRESHOLDS, noise=self._NOISE,
        )
        assert "Noise floor" in self._render_to_string(report)


class TestIsNonInteractive:
    def test_returns_true_when_stdin_not_tty(self, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        from evolution.core.saturation_check import is_non_interactive
        assert is_non_interactive() is True

    def test_returns_false_when_stdin_is_tty(self, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        from evolution.core.saturation_check import is_non_interactive
        assert is_non_interactive() is False


class TestInteractiveConfirm:
    @pytest.mark.parametrize("answer", ["y", "Y", "yes", "YES", "Yes"])
    def test_returns_true_for_yes_variants(self, monkeypatch, answer):
        monkeypatch.setattr("builtins.input", lambda _prompt="": answer)
        from evolution.core.saturation_check import interactive_confirm
        assert interactive_confirm() is True

    @pytest.mark.parametrize("answer", ["n", "no", "", "anything else", "ynope"])
    def test_returns_false_for_everything_else(self, monkeypatch, answer):
        monkeypatch.setattr("builtins.input", lambda _prompt="": answer)
        from evolution.core.saturation_check import interactive_confirm
        assert interactive_confirm() is False

    def test_returns_false_on_keyboard_interrupt(self, monkeypatch):
        def _raise(_prompt=""):
            raise KeyboardInterrupt()
        monkeypatch.setattr("builtins.input", _raise)
        from evolution.core.saturation_check import interactive_confirm
        assert interactive_confirm() is False

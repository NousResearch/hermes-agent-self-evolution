"""Unit tests for the CL-primary gate helper.

The helper combines two signals (CL pass counts, synthetic mean) and a
growth signal into a single accept/reject ConstraintResult. Tests pin
the decision-rule math; integration with evolve_tool lives in
tests/tools/test_evolve_tool_cl_aware_gate.py.
"""

from __future__ import annotations

import pytest

from evolution.core.constraints import ConstraintResult
from evolution.core.quality_gate import (
    CL_PRIMARY_GROWTH_FREE_THRESHOLD,
    CL_PRIMARY_GROWTH_SLOPE,
    _check_cl_primary_gate,
)


class TestCheckClPrimaryGate:
    def test_accepts_when_required_gain_met_at_free_threshold(self):
        # +1 gain, +20% growth (exactly at free threshold) → required=1
        result = _check_cl_primary_gate(
            baseline_cl_passes=5,
            evolved_cl_passes=6,
            baseline_synth_mean=0.97,
            evolved_synth_mean=0.97,
            growth_pct=0.20,
        )
        assert result.passed is True
        assert result.constraint_name == "cl_primary_gate"

    def test_accepts_at_24char_baseline_calibration_point(self):
        # +2 task gain on +121% growth → required=ceil(1.0*(1.21-0.20))=2 → just barely passes.
        # 24-char baseline calibration point from the prior retro-validation.
        result = _check_cl_primary_gate(
            baseline_cl_passes=5,
            evolved_cl_passes=7,
            baseline_synth_mean=1.000,
            evolved_synth_mean=1.000,
            growth_pct=1.21,
        )
        assert result.passed is True

    def test_rejects_when_growth_aware_threshold_unsatisfied(self):
        # +1 gain on +400% growth → required=4, fail.
        result = _check_cl_primary_gate(
            baseline_cl_passes=5,
            evolved_cl_passes=6,
            baseline_synth_mean=0.97,
            evolved_synth_mean=0.97,
            growth_pct=4.00,
        )
        assert result.passed is False
        assert "required" in result.message.lower()

    def test_rejects_when_no_task_gained(self):
        result = _check_cl_primary_gate(
            baseline_cl_passes=5,
            evolved_cl_passes=5,
            baseline_synth_mean=0.97,
            evolved_synth_mean=0.97,
            growth_pct=0.20,
        )
        assert result.passed is False

    def test_rejects_when_synthetic_regressed_beyond_tolerance(self):
        # +1 task gained, but synthetic dropped 0.06 (> 0.05 tolerance)
        result = _check_cl_primary_gate(
            baseline_cl_passes=5,
            evolved_cl_passes=6,
            baseline_synth_mean=1.000,
            evolved_synth_mean=0.939,
            growth_pct=0.20,
        )
        assert result.passed is False
        assert "synthetic" in result.message.lower()

    def test_accepts_when_synthetic_regressed_within_tolerance(self):
        # +1 task gained, synthetic dropped 0.04 (< 0.05 tolerance)
        result = _check_cl_primary_gate(
            baseline_cl_passes=5,
            evolved_cl_passes=6,
            baseline_synth_mean=1.000,
            evolved_synth_mean=0.961,
            growth_pct=0.20,
        )
        assert result.passed is True

    def test_rejects_when_evolved_cl_regressed(self):
        # Negative gain → reject even with no growth
        result = _check_cl_primary_gate(
            baseline_cl_passes=5,
            evolved_cl_passes=4,
            baseline_synth_mean=0.97,
            evolved_synth_mean=0.97,
            growth_pct=0.0,
        )
        assert result.passed is False

    def test_required_gain_floor_is_one_even_at_zero_growth(self):
        # Even with 0 growth, must gain ≥1 task — no free deploys for null changes
        result = _check_cl_primary_gate(
            baseline_cl_passes=5,
            evolved_cl_passes=5,
            baseline_synth_mean=0.97,
            evolved_synth_mean=0.97,
            growth_pct=0.0,
        )
        assert result.passed is False

    def test_growth_within_free_threshold_requires_only_one_task(self):
        # +1 gain, +15% growth (below 20% free threshold)
        result = _check_cl_primary_gate(
            baseline_cl_passes=5,
            evolved_cl_passes=6,
            baseline_synth_mean=0.97,
            evolved_synth_mean=0.97,
            growth_pct=0.15,
        )
        assert result.passed is True

    def test_message_records_required_and_actual_gain(self):
        # Message must surface the numbers for gate_decision.json + console
        result = _check_cl_primary_gate(
            baseline_cl_passes=5,
            evolved_cl_passes=6,
            baseline_synth_mean=0.97,
            evolved_synth_mean=0.97,
            growth_pct=0.20,
        )
        assert "1" in result.message  # required_gain == 1
        assert "+1" in result.message or "gained 1" in result.message.lower()

    def test_constants_match_evolution_config_defaults(self):
        # The CL gate's free-threshold default must match EvolutionConfig's
        # synthetic-gate default so they agree on what "free growth" means.
        from evolution.core.config import EvolutionConfig
        cfg = EvolutionConfig()
        assert CL_PRIMARY_GROWTH_FREE_THRESHOLD == cfg.growth_free_threshold
        assert CL_PRIMARY_GROWTH_SLOPE == 1.0


class TestNoiseAwareClPrimaryGate:
    """Opt-in noise floor inflates required_gain in the pass-count domain."""

    def test_zero_noise_floor_is_byte_identical(self):
        from evolution.core.quality_gate import _cl_required_gain
        # Default noise_floor_passes=0 must not change required_gain anywhere.
        for growth in (0.0, 0.20, 0.55, 1.40):
            assert _cl_required_gain(growth) == _cl_required_gain(
                growth, noise_floor_passes=0.0
            )

    def test_required_gain_must_strictly_exceed_noise_floor(self):
        from evolution.core.quality_gate import _cl_required_gain
        # 0.8 expected spurious flips → a +1 gain still clears it (1 > 0.8).
        assert _cl_required_gain(0.0, noise_floor_passes=0.8) == 1
        # 1.0 expected flips → +1 could be the flip; require +2.
        assert _cl_required_gain(0.0, noise_floor_passes=1.0) == 2
        # 1.4 → require +2 (smallest int > 1.4).
        assert _cl_required_gain(0.0, noise_floor_passes=1.4) == 2

    def test_noise_floor_rejects_within_noise_gain(self):
        # +1 task gain that would deploy at zero noise...
        ok = _check_cl_primary_gate(
            baseline_cl_passes=3, evolved_cl_passes=4,
            baseline_synth_mean=0.97, evolved_synth_mean=0.97, growth_pct=0.20,
        )
        assert ok.passed
        # ...is rejected once the A/A floor expects ~1.5 spurious flips.
        noisy = _check_cl_primary_gate(
            baseline_cl_passes=3, evolved_cl_passes=4,
            baseline_synth_mean=0.97, evolved_synth_mean=0.97, growth_pct=0.20,
            noise_floor_passes=1.5,
        )
        assert not noisy.passed
        # A +2 gain clears the 1.5 floor.
        clears = _check_cl_primary_gate(
            baseline_cl_passes=3, evolved_cl_passes=5,
            baseline_synth_mean=0.97, evolved_synth_mean=0.97, growth_pct=0.20,
            noise_floor_passes=1.5,
        )
        assert clears.passed

    def test_growth_term_still_dominates_when_larger(self):
        from evolution.core.quality_gate import _cl_required_gain
        # Large growth requires more than the noise floor would.
        assert _cl_required_gain(1.40, noise_floor_passes=0.3) == _cl_required_gain(1.40)

    def test_append_records_noise_fields(self):
        from evolution.core.quality_gate import append_cl_decision_fields
        payload: dict = {}
        append_cl_decision_fields(
            payload,
            cached_baseline_cl_per_example=[1.0, 0.0, 0.0],
            evolved_cl_per_example=[1.0, 1.0, 0.0],
            avg_baseline=0.97, avg_evolved=0.97, growth_pct=0.20,
            cl_eval_cost_usd=0.0, preflight_holdout_score=None,
            preflight_cl_score=None, closed_loop_agent_model="haiku",
            noise_floor_passes=1.5,
        )
        assert payload["cl_noise_floor_passes"] == 1.5
        assert payload["noise_aware_gate"] is True
        assert payload["cl_required_gain"] == 2  # floor(1.5)+1
        # Default (no noise) records the flag off.
        p2: dict = {}
        append_cl_decision_fields(
            p2, cached_baseline_cl_per_example=[1.0], evolved_cl_per_example=[1.0],
            avg_baseline=0.97, avg_evolved=0.97, growth_pct=0.20,
            cl_eval_cost_usd=0.0, preflight_holdout_score=None,
            preflight_cl_score=None, closed_loop_agent_model="haiku",
        )
        assert p2["noise_aware_gate"] is False

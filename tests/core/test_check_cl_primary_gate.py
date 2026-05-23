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

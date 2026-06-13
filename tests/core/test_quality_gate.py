"""Tests for the extracted quality_gate helpers.

These tests live alongside the new module rather than under tests/skills/
because the helpers are now artifact-agnostic. Skill-pipeline behavior is
still covered by tests/skills/test_evolve_skill_validation_flow.py.
"""

import json
from pathlib import Path

import pytest

from evolution.core.constraints import ConstraintResult
from evolution.core.quality_gate import (
    QUALITY_GATE_PRESETS,
    _check_cl_primary_gate,
    resolve_floor_fallback,
    resolve_proposer_mode,
    write_gate_decision,
)


def _gate(passed: bool) -> ConstraintResult:
    return ConstraintResult(passed=passed, constraint_name="cl_primary_gate", message="")


class TestResolveFloorFallback:
    def test_improving_evolved_always_wins(self):
        # A strictly-improving evolved candidate is always preferred over the
        # floor, regardless of the other flags (the first arm short-circuits).
        for floor_clears in (True, False):
            for deployable in (True, False):
                assert resolve_floor_fallback(
                    evolved_improved=True, evolved_deployable=deployable,
                    floor_clears=floor_clears,
                ) == "evolved"

    def test_floor_preempts_non_improving_but_deployable_evolved(self):
        # The motivating case: GEPA produced a no-op (deployable via no-regression
        # but no net improvement) and the floor wins → deploy the floor.
        assert resolve_floor_fallback(
            evolved_improved=False, evolved_deployable=True, floor_clears=True
        ) == "floor"

    def test_non_improving_deployable_evolved_ships_when_no_winning_floor(self):
        # No winning floor → keep deploying the no-regression-passing evolved
        # (don't start rejecting runs that ship today).
        assert resolve_floor_fallback(
            evolved_improved=False, evolved_deployable=True, floor_clears=False
        ) == "evolved"

    def test_regressing_evolved_falls_back_to_floor(self):
        assert resolve_floor_fallback(
            evolved_improved=False, evolved_deployable=False, floor_clears=True
        ) == "floor"

    def test_regressing_evolved_no_floor_rejects(self):
        assert resolve_floor_fallback(
            evolved_improved=False, evolved_deployable=False, floor_clears=False
        ) == "reject"

    def test_cl_primary_gate_passes_improved_equal_to_deployable(self):
        # The skill CL-primary gate requires a strict gain, so callers pass
        # improved == deployable: a failing gate with no winning floor rejects,
        # with a winning floor deploys the floor.
        assert resolve_floor_fallback(
            evolved_improved=False, evolved_deployable=False, floor_clears=False
        ) == "reject"
        assert resolve_floor_fallback(
            evolved_improved=False, evolved_deployable=False, floor_clears=True
        ) == "floor"
        assert resolve_floor_fallback(
            evolved_improved=True, evolved_deployable=True, floor_clears=True
        ) == "evolved"


class TestFloorJudgedBySameRule:
    def test_floor_obeys_noise_aware_required_gain(self):
        # The floor is judged by the SAME _check_cl_primary_gate as evolved.
        # With an A/A noise floor of 1.5, required gain is floor(1.5)+1 = 2, so a
        # +1 floor win is rejected — proving the noise-aware rule applies
        # symmetrically to the floor challenger. The floor is zero-LM, so its
        # synth mean equals baseline's (synth Δ = 0, trivially within tolerance).
        gate = _check_cl_primary_gate(
            baseline_cl_passes=5,
            evolved_cl_passes=6,  # +1 (the "floor" arm)
            baseline_synth_mean=0.6,
            evolved_synth_mean=0.6,
            growth_pct=0.05,
            noise_floor_passes=1.5,
        )
        assert gate.passed is False
        # +2 clears the same gate.
        assert _check_cl_primary_gate(
            baseline_cl_passes=5, evolved_cl_passes=7,
            baseline_synth_mean=0.6, evolved_synth_mean=0.6,
            growth_pct=0.05, noise_floor_passes=1.5,
        ).passed is True


class TestResolveProposerMode:
    def test_compression_profile(self):
        assert resolve_proposer_mode("compression") == "compression"

    def test_balanced_profile(self):
        assert resolve_proposer_mode("balanced") == "balanced"

    def test_growth_profile(self):
        assert resolve_proposer_mode("growth") == "growth"

    def test_unknown_profile_falls_back_to_compression(self):
        assert resolve_proposer_mode("bogus") == "compression"


class TestQualityGatePresets:
    def test_default_preset_exists(self):
        assert "default" in QUALITY_GATE_PRESETS

    def test_each_preset_has_required_keys(self):
        for name, preset in QUALITY_GATE_PRESETS.items():
            assert "growth_free_threshold" in preset, f"{name} missing growth_free_threshold"
            assert "growth_quality_slope" in preset, f"{name} missing growth_quality_slope"
            assert "max_absolute_chars" in preset, f"{name} missing max_absolute_chars"


class TestWriteGateDecision:
    def test_writes_json_file(self, tmp_path: Path):
        payload = {"decision": "deploy", "reason": "test"}
        output_path = write_gate_decision(tmp_path, payload)

        assert output_path.exists()
        assert output_path.name == "gate_decision.json"
        loaded = json.loads(output_path.read_text())
        assert loaded["decision"] == "deploy"
        assert loaded["reason"] == "test"

    def test_preserves_all_payload_fields(self, tmp_path: Path):
        payload = {
            "decision": "reject",
            "reason": "growth_quality_gate",
            "growth_pct": 0.5,
            "baseline_chars": 100,
            "evolved_chars": 150,
        }
        write_gate_decision(tmp_path, payload)
        loaded = json.loads((tmp_path / "gate_decision.json").read_text())
        for key, value in payload.items():
            assert loaded[key] == value


class TestBackCompatAliases:
    def test_evolve_skill_reexports_underscored_names(self):
        from evolution.skills import evolve_skill
        assert evolve_skill._QUALITY_GATE_PRESETS is QUALITY_GATE_PRESETS
        assert evolve_skill._resolve_proposer_mode is resolve_proposer_mode
        assert evolve_skill._write_gate_decision is write_gate_decision

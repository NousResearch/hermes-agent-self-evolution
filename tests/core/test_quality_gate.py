"""Tests for the extracted quality_gate helpers.

These tests live alongside the new module rather than under tests/skills/
because the helpers are now artifact-agnostic. Skill-pipeline behavior is
still covered by tests/skills/test_evolve_skill_validation_flow.py.
"""

import json
from pathlib import Path

import pytest

from evolution.core.quality_gate import (
    QUALITY_GATE_PRESETS,
    resolve_proposer_mode,
    write_gate_decision,
)


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

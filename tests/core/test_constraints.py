"""Tests for constraint validators."""

import pytest
from evolution.core.constraints import ConstraintValidator
from evolution.core.config import (
    EvolutionConfig,
    is_reference_skill,
    resolve_skill_soft_cap,
    length_penalty_for,
)


@pytest.fixture
def validator():
    config = EvolutionConfig()
    return ConstraintValidator(config)


class TestSizeConstraints:
    def test_skill_under_limit(self, validator):
        result = validator._check_size("x" * 1000, "skill")
        assert result.passed

    def test_skill_over_soft_target_passes_with_penalty(self, validator):
        # 20k > default soft target (15k) but < hard ceiling (30k): the class-aware
        # policy treats this as a graduated-penalty pass, not a hard rejection.
        result = validator._check_size("x" * 20_000, "skill")
        assert result.passed
        assert "soft target" in result.message

    def test_skill_over_hard_ceiling_fails(self, validator):
        result = validator._check_size("x" * 35_000, "skill")
        assert not result.passed
        assert "ceiling" in result.message


class TestClassAwareSizePolicy:
    def test_reference_skill_detection(self):
        ref = "---\ntags: [workflow]\ndescription: persistent master reference, survives context compression\n---\nbody"
        gen = "---\nname: writer\ndescription: write a blog post on a topic\n---\nbody"
        assert is_reference_skill(ref)
        assert not is_reference_skill(gen)

    def test_reference_skill_gets_larger_soft_cap(self):
        cfg = EvolutionConfig()
        ref = "tags: [runbook] survives context compression"
        gen = "write a marketing post"
        assert resolve_skill_soft_cap(ref, cfg) == cfg.max_skill_size_reference
        assert resolve_skill_soft_cap(gen, cfg) == cfg.max_skill_size

    def test_graduated_penalty_no_cliff_below_cap(self):
        cfg = EvolutionConfig()
        soft, hard = cfg.max_skill_size, cfg.max_skill_hard_ceiling
        # No penalty at/below the soft cap (old ramp wrongly docked 90-100%).
        assert length_penalty_for(soft, soft, hard) == 0.0
        assert length_penalty_for(int(soft * 0.95), soft, hard) == 0.0
        # Ramps up between soft and ceiling, clamped at 0.3 beyond.
        mid = length_penalty_for((soft + hard) // 2, soft, hard)
        assert 0.0 < mid < 0.3
        assert length_penalty_for(hard, soft, hard) == pytest.approx(0.3)
        assert length_penalty_for(hard * 2, soft, hard) == 0.3

    def test_tool_description_under_limit(self, validator):
        result = validator._check_size("Search files by content", "tool_description")
        assert result.passed

    def test_tool_description_over_limit(self, validator):
        result = validator._check_size("x" * 600, "tool_description")
        assert not result.passed


class TestGrowthConstraints:
    def test_acceptable_growth(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1100  # 10% growth
        result = validator._check_growth(evolved, baseline, "skill")
        assert result.passed

    def test_excessive_growth(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1300  # 30% growth
        result = validator._check_growth(evolved, baseline, "skill")
        assert not result.passed

    def test_shrinkage_is_ok(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 800  # 20% smaller
        result = validator._check_growth(evolved, baseline, "skill")
        assert result.passed


class TestNonEmpty:
    def test_non_empty_passes(self, validator):
        result = validator._check_non_empty("some content")
        assert result.passed

    def test_empty_fails(self, validator):
        result = validator._check_non_empty("")
        assert not result.passed

    def test_whitespace_only_fails(self, validator):
        result = validator._check_non_empty("   \n  ")
        assert not result.passed


class TestSkillStructure:
    def test_valid_skill(self, validator):
        skill = "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test\nContent here"
        result = validator._check_skill_structure(skill)
        assert result.passed

    def test_missing_frontmatter(self, validator):
        skill = "# Test\nContent without frontmatter"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_missing_name(self, validator):
        skill = "---\ndescription: A test skill\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_missing_description(self, validator):
        skill = "---\nname: test-skill\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed


class TestValidateAll:
    def test_valid_skill_passes_all(self, validator):
        skill = "---\nname: test\ndescription: Test skill\n---\n\n# Procedure\n1. Do thing"
        results = validator.validate_all(skill, "skill")
        assert all(r.passed for r in results)

    def test_empty_skill_fails(self, validator):
        results = validator.validate_all("", "skill")
        failed = [r for r in results if not r.passed]
        assert len(failed) > 0

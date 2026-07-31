"""Tests for constraint validators."""

import pytest
from pathlib import Path
from evolution.core.constraints import ConstraintValidator
from evolution.core.config import EvolutionConfig


@pytest.fixture
def validator():
    # Explicit repo path — constraint tests don't need repo discovery and
    # shouldn't fail when HERMES_AGENT_REPO is unset (get_hermes_agent_path()
    # raises FileNotFoundError in that case).
    config = EvolutionConfig(hermes_agent_path=Path("/tmp/hermes-agent"))
    return ConstraintValidator(config)


class TestSizeConstraints:
    def test_skill_under_limit(self, validator):
        result = validator._check_size("x" * 1000, "skill")
        assert result.passed

    def test_skill_over_limit(self, validator):
        result = validator._check_size("x" * 20_001, "skill")
        assert not result.passed
        assert "exceeded" in result.message

    def test_skill_at_exact_limit_passes(self, validator):
        # Boundary: size == limit is acceptable, only > limit fails
        result = validator._check_size("x" * 20_000, "skill")
        assert result.passed

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
        evolved = "x" * 1700  # 70% growth — over the 50% soft cap, no waiver
        result = validator._check_growth(evolved, baseline, "skill")
        assert not result.passed
        assert "Growth exceeded" in result.message

    def test_exact_boundary_passes(self, validator):
        # Exactly +50% == max_prompt_growth — boundary must pass
        baseline = "x" * 1000
        evolved = "x" * 1500
        result = validator._check_growth(evolved, baseline, "skill")
        assert result.passed

    def test_shrinkage_is_ok(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 800  # 20% smaller
        result = validator._check_growth(evolved, baseline, "skill")
        assert result.passed

    def test_zero_length_baseline(self, validator):
        # No ZeroDivisionError — denominator is max(1, 0)
        result = validator._check_growth("x" * 10, "", "skill")
        assert not result.passed  # +1000% growth
        result = validator._check_growth("", "", "skill")
        assert result.passed  # no growth

    def test_growth_waiver_grants_material_improvement(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1700  # +70% > 50% soft cap
        result = validator._check_growth(evolved, baseline, "skill", improvement=0.05)
        assert result.passed
        assert "waiver" in result.message

    def test_growth_waiver_insufficient_improvement(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1700  # +70% > 50% soft cap
        result = validator._check_growth(evolved, baseline, "skill", improvement=0.01)
        assert not result.passed
        assert "waiver threshold" in result.message

    def test_growth_waiver_rejected_beyond_hard_cap(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 2500  # +150% — over the 100% hard cap even with waiver
        result = validator._check_growth(evolved, baseline, "skill", improvement=0.05)
        assert not result.passed

    def test_growth_without_improvement_stays_strict(self, validator):
        # Backward compat: improvement=None must behave exactly like before
        baseline = "x" * 1000
        evolved = "x" * 1700  # +70%
        result = validator._check_growth(evolved, baseline, "skill", improvement=None)
        assert not result.passed

    def test_escalation_exact_failure_mode(self, validator):
        # esc-20260731-201857-1474 exact numbers: baseline 8,395 chars,
        # rejected artifact 14,382 chars (+71.3% > 50% soft cap), measured
        # valset improvement +0.035 (0.558 → 0.593). Without the improvement
        # signal the gate rejects (the recorded 'constraints FAILED'); with it
        # the waiver grants (stays under the +100% hard cap).
        baseline = "x" * 8395
        evolved = "x" * 14382
        assert not validator._check_growth(evolved, baseline, "skill").passed
        result = validator._check_growth(evolved, baseline, "skill", improvement=0.035)
        assert result.passed
        assert "waiver" in result.message

    def test_growth_text_compares_body_not_full_artifact(self, validator):
        # gmail-monitor regression (t_63240783): the growth gate compared the
        # evolved FULL artifact (frontmatter + body) against the baseline BODY
        # only → +100.5% > +100% hard cap → rejected a real improvement by
        # 0.5%. Comparing body-vs-body is +69.8%, under the hard cap, and the
        # +0.045 waiver improvement grants it.
        baseline_body = "x" * 2719
        # Realistic frontmatter size (~836 chars for gmail-monitor): big
        # enough that full-vs-body growth crosses the +100% hard cap, exactly
        # like the ticket's evolved_FAILED.md (5452 raw = 836 fm + 4616 body).
        frontmatter = "---\nname: gmail-monitor\ndescription: " + ("y" * 800) + "\n---\n\n"
        evolved_body = "x" * 4616  # +69.8% body-vs-body
        evolved_full = frontmatter + evolved_body

        # Old (buggy) behavior: full-vs-body → +100.5% → fails even with waiver
        old = validator._check_growth(
            evolved_full, baseline_body, "skill", improvement=0.045
        )
        assert not old.passed
        assert "hard cap" in old.message or "Growth exceeded" in old.message

        # Fixed: growth_text=evolved_body → +69.8% → waiver grants (hard cap OK)
        fixed = validator._check_growth(
            evolved_full, baseline_body, "skill",
            improvement=0.045, growth_text=evolved_body,
        )
        assert fixed.passed
        assert "waiver" in fixed.message

    def test_growth_text_defaults_to_artifact_text(self, validator):
        # Backward compat: growth_text=None must behave exactly like before.
        baseline = "x" * 1000
        evolved = "x" * 1700
        assert not validator._check_growth(evolved, baseline, "skill").passed
        result = validator._check_growth(
            evolved, baseline, "skill", improvement=0.05, growth_text=None
        )
        assert result.passed
        assert "waiver" in result.message


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

    def test_growth_failure_with_waiver_passes(self, validator):
        # End-to-end validate_all with a full skill: growth past the soft cap
        # passes when a material improvement is supplied.
        baseline = (
            "---\nname: test\ndescription: Test skill\n---\n\n"
            + "Procedure text " * 100  # ~1.5KB baseline
        )
        evolved = baseline + (" extra content that grows the skill " * 30)  # +66% > 50% soft cap
        # no improvement → fails on growth
        results = validator.validate_all(evolved, "skill", baseline_text=baseline)
        assert any(not r.passed for r in results)
        # material improvement → growth waived, all pass
        results = validator.validate_all(evolved, "skill", baseline_text=baseline, improvement=0.05)
        assert all(r.passed for r in results)

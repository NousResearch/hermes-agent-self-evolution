"""Tests for constraint validators."""

import pytest
from evolution.core.constraints import ConstraintValidator
from evolution.core.config import EvolutionConfig


@pytest.fixture
def validator():
    config = EvolutionConfig()
    return ConstraintValidator(config)


class TestSizeConstraints:
    def test_skill_under_limit(self, validator):
        result = validator._check_size("x" * 1000, "skill")
        assert result.passed

    def test_skill_over_limit(self, validator):
        result = validator._check_size("x" * 20_000, "skill")
        assert not result.passed
        assert "exceeded" in result.message

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

    def test_body_only_artifact_uses_full_text_for_structure_check(self, validator):
        """Regression: evolution strips frontmatter before optimizing the body,
        then reassembles frontmatter + evolved body into `full_text` afterward.
        The skill_structure check must run against `full_text`, not the bare
        body — checking the body alone always fails (no frontmatter present)
        regardless of whether the reassembled artifact is valid, which
        silently blocked every evolved-skill deploy."""
        body = "# Procedure\n1. Do thing"
        full_text = "---\nname: test\ndescription: Test skill\n---\n\n" + body
        results = validator.validate_all(body, "skill", full_text=full_text)
        assert all(r.passed for r in results)

        # Without full_text, the same body-only artifact must fail
        # skill_structure (documents the previous, buggy behavior for contrast).
        results_no_full = validator.validate_all(body, "skill")
        structure_result = next(r for r in results_no_full if r.constraint_name == "skill_structure")
        assert not structure_result.passed

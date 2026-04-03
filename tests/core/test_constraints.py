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
    """Tests for skill body structure validation.

    Note: _check_skill_structure receives the body (after frontmatter),
    not the full file. Frontmatter is preserved separately by load_skill()
    and reassembled by reassemble_skill(). The validator checks markdown
    structure of the body only.
    """

    def test_valid_body(self, validator):
        body = "# Test Skill\n\nThis is a valid skill body with content."
        result = validator._check_skill_structure(body)
        assert result.passed

    def test_body_with_subheadings(self, validator):
        body = "# Main\n\n## Section 1\nContent here.\n\n## Section 2\nMore content."
        result = validator._check_skill_structure(body)
        assert result.passed

    def test_body_without_heading_fails(self, validator):
        body = "Just some text without any markdown heading, this should fail."
        result = validator._check_skill_structure(body)
        assert not result.passed

    def test_empty_body_fails(self, validator):
        body = ""
        result = validator._check_skill_structure(body)
        assert not result.passed

    def test_whitespace_only_body_fails(self, validator):
        body = "   \n\n  "
        result = validator._check_skill_structure(body)
        assert not result.passed

    def test_heading_only_too_short_fails(self, validator):
        body = "# Title"
        result = validator._check_skill_structure(body)
        assert not result.passed
        assert "meaningful content" in result.message


class TestValidateAll:
    def test_valid_body_passes_all(self, validator):
        body = "# Procedure\n\n1. Do thing\n2. Do another thing\n3. Verify result"
        results = validator.validate_all(body, "skill")
        assert all(r.passed for r in results)

    def test_empty_body_fails(self, validator):
        results = validator.validate_all("", "skill")
        failed = [r for r in results if not r.passed]
        assert len(failed) > 0

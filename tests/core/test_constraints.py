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
    """Tests for _check_skill_structure on markdown bodies (not frontmatter).

    NOTE: validate_all() receives skill["body"] (markdown after frontmatter
    has been stripped), so _check_skill_structure validates body structure,
    not frontmatter presence. Frontmatter is preserved separately by
    load_skill()/reassemble_skill().
    """

    def test_valid_body_with_heading_and_content(self, validator):
        """Normal skill body with heading + prose passes."""
        body = "# My Skill\n\nThis is the skill procedure. " + "x" * 60
        result = validator._check_skill_structure(body)
        assert result.passed, result.message

    def test_valid_body_with_multiple_headings(self, validator):
        """Body with multiple section headings passes."""
        body = "# Overview\n\nSkill overview.\n\n## Procedure\n\n1. Step one.\n2. Step two.\n\n## Examples\n\nSee examples.\n"
        result = validator._check_skill_structure(body)
        assert result.passed, result.message

    def test_body_without_heading_fails(self, validator):
        """Body with no markdown heading fails."""
        body = "Just some prose without any heading."
        result = validator._check_skill_structure(body)
        assert not result.passed
        assert "heading" in result.message

    def test_body_too_short_fails(self, validator):
        """Body with heading but barely any content fails."""
        body = "# Short\n\nToo short."
        result = validator._check_skill_structure(body)
        assert not result.passed
        assert "too short" in result.message

    def test_body_only_lists_fails(self, validator):
        """Body that is only list items with no prose fails."""
        body = "# Steps\n\n- Step one\n- Step two\n- Step three"
        result = validator._check_skill_structure(body)
        assert not result.passed
        assert "only lists" in result.message

    def test_body_mixed_lists_and_prose_passes(self, validator):
        """Body with both prose and lists passes."""
        body = "# Procedure\n\nFollow these steps:\n\n1. First do this thing.\n2. Then verify it worked.\n3. Proceed to the next step.\n\n# Examples\n\nHere is an example of usage."
        result = validator._check_skill_structure(body)
        assert result.passed, result.message

    def test_heading_without_space_passes(self, validator):
        """Heading without space after # still passes (we're lenient)."""
        body = "#NoSpace\n\n" + "x" * 60
        result = validator._check_skill_structure(body)
        assert result.passed, result.message


class TestValidateAll:
    def test_valid_skill_passes_all(self, validator):
        body = "# Test Skill\n\nThis is the procedure. " + "x" * 60
        results = validator.validate_all(body, "skill")
        assert all(r.passed for r in results), [r.message for r in results]

    def test_empty_skill_fails(self, validator):
        results = validator.validate_all("", "skill")
        failed = [r for r in results if not r.passed]
        assert len(failed) > 0

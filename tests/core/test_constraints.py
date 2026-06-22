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


class TestSemanticPreservation:
    def test_identical_text_passes(self, validator):
        text = "Read the task. Do the thing. Verify the result carefully."
        result = validator._check_semantic_preservation(text, text)
        assert result.passed
        assert result.constraint_name == "semantic_preservation"

    def test_drifted_text_fails(self, validator):
        baseline = "Review the pull request for security bugs and style issues."
        evolved = "A recipe for cooking pasta with tomato sauce and basil leaves."
        result = validator._check_semantic_preservation(evolved, baseline)
        assert not result.passed

    def test_minor_edit_passes(self, validator):
        baseline = "Read the task. Do the thing. Verify the result."
        evolved = "Read the task. Do the thing. Verify the result now."
        result = validator._check_semantic_preservation(evolved, baseline)
        assert result.passed

    def test_disabled_check_passes(self):
        config = EvolutionConfig()
        config.enable_semantic_check = False
        v = ConstraintValidator(config)
        result = v._check_semantic_preservation("anything", "totally different")
        assert result.passed
        assert "disabled" in result.message.lower()

    def test_empty_evolved_fails(self, validator):
        result = validator._check_semantic_preservation("", "baseline text here")
        assert not result.passed

    def test_threshold_respected(self):
        config = EvolutionConfig()
        config.semantic_similarity_threshold = 0.99  # near-impossible
        v = ConstraintValidator(config)
        baseline = "Read the task. Do the thing."
        evolved = "Read the task. Do the thing differently."
        result = v._check_semantic_preservation(evolved, baseline)
        assert not result.passed


class TestSkillBodyValidation:
    def test_skill_body_skips_structure_check(self, validator):
        # A body has no frontmatter — but artifact_type "skill_body" must not
        # run the structure check, so it should still pass.
        body = "# Procedure\n\n1. First step\n2. Second step\n3. Verify"
        results = validator.validate_all(body, "skill_body")
        names = {r.constraint_name for r in results}
        assert "skill_structure" not in names
        assert all(r.passed for r in results)

    def test_skill_body_runs_semantic_with_baseline(self, validator):
        body = "# Procedure\n\n1. First step\n2. Verify"
        results = validator.validate_all(body, "skill_body", baseline_text=body)
        names = {r.constraint_name for r in results}
        assert "semantic_preservation" in names

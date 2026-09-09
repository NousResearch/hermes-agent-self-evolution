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

    def test_evolved_full_reassembled_text_passes_skill_structure(self, validator):
        """Regression test for the bug where `validate_all` was called with
        `evolved_body` (the body slice with frontmatter already stripped by
        `load_skill()`) and the `skill_structure` constraint falsely failed.

        `reassemble_skill(frontmatter, evolved_body)` produces a complete
        skill file with valid frontmatter. Validating the reassembled text
        should pass all four constraints; validating the body alone should
        not. This test pins the contract that the call site in
        `evolution/skills/evolve_skill.py:189` depends on.
        """
        # Reassembled text — what the patched call site actually validates
        reassembled = (
            "---\n"
            "name: evolved-skill\n"
            "description: An evolved skill produced by the self-evolution pipeline\n"
            "---\n"
            "\n"
            "# Procedure\n"
            "1. Read the contract from the system prompt.\n"
            "2. Verify the bridge is healthy.\n"
            "3. Report pass/fail with evidence.\n"
        )
        results = validator.validate_all(reassembled, "skill")
        skill_structure_result = next(r for r in results if r.constraint_name == "skill_structure")
        assert skill_structure_result.passed, (
            f"Reassembled skill text should pass skill_structure constraint, "
            f"got: {skill_structure_result.message}"
        )

    def test_evolved_body_alone_fails_skill_structure(self, validator):
        """Confirms the root cause: the body alone (no frontmatter) cannot
        ever pass the skill_structure check. This is what the upstream bug
        did — it validated the body, which structurally cannot pass.

        This test exists so a future refactor that re-introduces
        `validate_all(evolved_body, ...)` would fail loudly, with the
        `skill_structure` constraint as the smoking gun.
        """
        body_only = (
            "# Procedure\n"
            "1. Read the contract from the system prompt.\n"
            "2. Verify the bridge is healthy.\n"
        )
        results = validator.validate_all(body_only, "skill")
        skill_structure_result = next(r for r in results if r.constraint_name == "skill_structure")
        assert not skill_structure_result.passed, (
            "Body-only text should fail skill_structure (no frontmatter) — "
            "if this assertion fails, the constraint has changed shape and "
            "the upstream call site may need a different fix."
        )

"""Tests for constraint validators."""

import subprocess
import sys

import pytest
from evolution.core.constraints import ConstraintValidator
from evolution.core.config import EvolutionConfig


@pytest.fixture
def validator(tmp_path):
    config = EvolutionConfig(hermes_agent_path=tmp_path)
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


class TestSkillFileValidation:
    def test_skill_file_validation_checks_body_and_full_file_separately(self, validator):
        frontmatter = "---\nname: test\ndescription: Test skill\n---\n"
        baseline_body = "# Procedure\n1. Do thing exactly"
        evolved_body = "# Procedure\n1. Do improved thing"
        evolved_full = f"{frontmatter}\n{evolved_body}"

        results = validator.validate_skill_file(
            full_skill_text=evolved_full,
            body_text=evolved_body,
            baseline_body_text=baseline_body,
        )

        assert all(r.passed for r in results)
        assert [r.constraint_name for r in results] == [
            "size_limit",
            "growth_limit",
            "non_empty",
            "skill_structure",
        ]

    def test_skill_file_validation_rejects_missing_frontmatter_in_full_file(self, validator):
        results = validator.validate_skill_file(
            full_skill_text="# Procedure\nNo metadata",
            body_text="# Procedure\nNo metadata",
        )

        failed_names = {r.constraint_name for r in results if not r.passed}
        assert "skill_structure" in failed_names


class TestRunTestSuite:
    def test_run_test_suite_uses_current_python_interpreter(self, validator, tmp_path, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0, stdout="1 passed\n", stderr="")

        monkeypatch.setattr("evolution.core.constraints.subprocess.run", fake_run)

        result = validator.run_test_suite(tmp_path)

        assert result.passed
        assert calls[0][0][:3] == [sys.executable, "-m", "pytest"]
        assert calls[0][1]["cwd"] == str(tmp_path)

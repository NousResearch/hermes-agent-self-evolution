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

    def test_excessive_shrinkage_fails(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 300  # 70% smaller
        result = validator._check_shrinkage(evolved, baseline, "skill")
        assert not result.passed
        assert "Shrinkage exceeded" in result.message


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


class TestSkillSafetyConstraints:
    def test_heading_retention_rejects_missing_baseline_headings(self, validator):
        baseline = """---
name: kanban-orchestrator
description: Test
---

# Kanban Orchestrator

## When to Use

## Verification Checklist
"""
        evolved = """---
name: kanban-orchestrator
description: Test
---

### Advanced Core Directive
Generic content.
"""

        result = validator._check_heading_retention(evolved, baseline)

        assert not result.passed
        assert "When to Use" in result.details

    def test_reference_retention_rejects_lost_reference_links(self, validator):
        baseline = "See `references/hermes-self-evolution.md` for the workflow."
        evolved = "See the workflow notes below."

        result = validator._check_reference_retention(evolved, baseline)

        assert not result.passed
        assert "references/hermes-self-evolution.md" in result.details

    def test_domain_term_retention_rejects_lost_stephen_org_terms(self, validator):
        baseline = "Babbage routes Kanban work to Turing and Ops for Stephen."
        evolved = "The assistant routes tasks to specialists."

        result = validator._check_domain_term_retention(evolved, baseline)

        assert not result.passed
        assert "Babbage" in result.details
        assert "Turing" in result.details

    def test_irrelevant_tool_guard_rejects_new_code_interpreter_reference(self, validator):
        baseline = "Use Kanban and Hermes cron state to route work."
        evolved = "Use code_interpreter to simulate database_api.update_status()."

        result = validator._check_new_irrelevant_tool_references(evolved, baseline)

        assert not result.passed
        assert "code_interpreter" in result.details


class TestValidateAll:
    def test_valid_skill_passes_all(self, validator):
        skill = "---\nname: test\ndescription: Test skill\n---\n\n# Procedure\n1. Do thing"
        results = validator.validate_all(skill, "skill")
        assert all(r.passed for r in results)

    def test_empty_skill_fails(self, validator):
        results = validator.validate_all("", "skill")
        failed = [r for r in results if not r.passed]
        assert len(failed) > 0

    def test_validate_all_includes_safety_constraints_when_baseline_provided(self, validator):
        baseline = """---
name: kanban-orchestrator
description: Test
---

# Kanban Orchestrator

## When to Use
Route Stephen's Kanban work to Babbage, Turing, and Ops.

See `references/kanban-worker.md`.
"""
        evolved = """---
name: kanban-orchestrator
description: Test
---

### Advanced Core Directive
Use code_interpreter for task execution.
"""

        results = validator.validate_all(evolved, "skill", baseline_text=baseline)
        failed = {r.constraint_name for r in results if not r.passed}

        assert "heading_retention" in failed
        assert "reference_retention" in failed
        assert "domain_term_retention" in failed
        assert "irrelevant_tool_references" in failed

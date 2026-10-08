"""Tests for skill module loading and parsing."""

import pytest
from pathlib import Path
from evolution.skills.skill_module import SkillModule, load_skill, reassemble_skill


SAMPLE_SKILL = """---
name: test-skill
description: A skill for testing things
version: 1.0.0
metadata:
  hermes:
    tags: [testing]
---

# Test Skill — Testing Things

## When to Use
Use this when you need to test things.

## Procedure
1. First, do the thing
2. Then, verify it worked
3. Report results

## Pitfalls
- Don't forget to check edge cases
"""


class TestLoadSkill:
    def test_parses_frontmatter(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        assert skill["name"] == "test-skill"
        assert skill["description"] == "A skill for testing things"
        assert "version: 1.0.0" in skill["frontmatter"]

    def test_parses_body(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        assert "# Test Skill" in skill["body"]
        assert "## Procedure" in skill["body"]
        assert "Don't forget" in skill["body"]

    def test_raw_contains_everything(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        assert skill["raw"] == SAMPLE_SKILL

    def test_path_is_stored(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        assert skill["path"] == skill_file


class TestReassembleSkill:
    def test_roundtrip(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        reassembled = reassemble_skill(skill["frontmatter"], skill["body"])
        assert "---" in reassembled
        assert "name: test-skill" in reassembled
        assert "# Test Skill" in reassembled

    def test_preserves_frontmatter(self):
        frontmatter = "name: my-skill\ndescription: Does stuff"
        body = "# My Skill\nDo the thing."
        result = reassemble_skill(frontmatter, body)

        assert result.startswith("---\n")
        assert "name: my-skill" in result
        assert "# My Skill" in result

    def test_evolved_body_replaces_original(self):
        frontmatter = "name: my-skill\ndescription: Does stuff"
        evolved_body = "# EVOLVED\nNew and improved procedure."
        result = reassemble_skill(frontmatter, evolved_body)

        assert "EVOLVED" in result
        assert "New and improved" in result


class TestSkillModule:
    """Regression tests: the skill text MUST live in the predictor's
    instructions — that is the exact search space GEPA mutates."""

    def test_skill_text_is_predictor_instructions(self):
        module = SkillModule("# My Skill\nDo the thing.")
        assert module.skill_text == module.predictor.predict.signature.instructions
        assert module.skill_text == "# My Skill\nDo the thing."

    def test_instruction_mutation_visible_in_skill_text(self):
        module = SkillModule("baseline")
        # Mutate exactly the way GEPA does: through named_predictors()
        for _, pred in module.predictor.named_predictors():
            pred.signature = pred.signature.with_instructions("EVOLVED")
        assert module.skill_text == "EVOLVED"


def test_full_file_passes_validation_but_body_does_not():
    """Guards the evolve() pipeline: structure validation must run on the
    raw full file (frontmatter included), never on the stripped body."""
    from evolution.core.config import EvolutionConfig
    from evolution.core.constraints import ConstraintValidator

    validator = ConstraintValidator(EvolutionConfig(hermes_agent_path=None))

    raw_results = validator.validate_all(SAMPLE_SKILL, "skill")
    assert all(r.passed for r in raw_results)

    body = SAMPLE_SKILL.split("---", 2)[2].strip()
    body_results = validator.validate_all(body, "skill")
    assert not next(r for r in body_results if r.constraint_name == "skill_structure").passed

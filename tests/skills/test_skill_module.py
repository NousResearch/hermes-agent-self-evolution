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


class TestSkillModuleIsOptimizable:
    """The skill text must be the parameter DSPy optimizers actually mutate.

    DSPy optimizers (GEPA, MIPROv2) rewrite a predictor's **signature
    instructions**; they never touch InputField *values*. If the skill text is
    wired as an InputField, optimization silently no-ops: GEPA proposes better
    variants and the caller reads back the unchanged original.
    """

    BODY = "# Test Skill\n\n## Procedure\n1. Do the thing\n"

    def test_skill_text_seeds_predictor_instructions(self):
        module = SkillModule(self.BODY)

        instructions = [
            predictor.signature.instructions
            for _, predictor in module.named_predictors()
        ]
        assert instructions, "module exposes no optimizable predictors"
        assert any(self.BODY.strip() == text.strip() for text in instructions), (
            "skill text is not in any predictor's instructions, so no DSPy "
            "optimizer can mutate it"
        )

    def test_skill_text_reflects_optimizer_mutation(self):
        module = SkillModule(self.BODY)

        # Simulate what GEPA/MIPROv2 do when they accept a candidate.
        for _, predictor in module.named_predictors():
            predictor.signature = predictor.signature.with_instructions("# EVOLVED")

        assert module.skill_text.strip() == "# EVOLVED", (
            "mutating predictor instructions did not change skill_text — the "
            "evolved variant would be discarded"
        )

    def test_skill_text_roundtrips_unmutated(self):
        module = SkillModule(self.BODY)
        assert module.skill_text.strip() == self.BODY.strip()

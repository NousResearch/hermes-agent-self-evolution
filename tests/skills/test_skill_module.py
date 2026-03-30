"""Tests for skill module loading and parsing."""

import pytest
from pathlib import Path
from unittest.mock import patch

import dspy

from evolution.skills.skill_module import load_skill, reassemble_skill, SkillModule


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
    """Tests for SkillModule — ensures skill text is stored as signature
    instructions (the part DSPy optimizers mutate), not just an instance attr."""

    SKILL_TEXT = "# Review Code\n\n## Procedure\n1. Read the diff\n2. Check for bugs"

    def test_skill_text_is_signature_instructions(self):
        module = SkillModule(self.SKILL_TEXT)
        # ChainOfThought wraps an inner Predict at .predict
        assert module.predictor.predict.signature.instructions == self.SKILL_TEXT

    def test_get_evolved_text_returns_instructions(self):
        module = SkillModule(self.SKILL_TEXT)
        assert module.get_evolved_text() == self.SKILL_TEXT

    def test_get_evolved_text_reflects_mutations(self):
        """Simulates what DSPy optimizers do: mutate signature.instructions."""
        module = SkillModule(self.SKILL_TEXT)
        evolved = "# Evolved Review\n\n## Procedure\n1. Summarize changes\n2. Flag issues"
        # This is what GEPA/MIPROv2 do internally via with_instructions()
        inner = module.predictor.predict
        inner.signature = inner.signature.with_instructions(evolved)
        assert module.get_evolved_text() == evolved
        # Original instance attr is unchanged (preserved for diffing)
        assert module.skill_text == self.SKILL_TEXT

    def test_original_skill_text_preserved(self):
        module = SkillModule(self.SKILL_TEXT)
        assert module.skill_text == self.SKILL_TEXT

    def test_predictor_is_chain_of_thought(self):
        module = SkillModule(self.SKILL_TEXT)
        assert isinstance(module.predictor, dspy.ChainOfThought)

    def test_signature_has_expected_fields(self):
        module = SkillModule(self.SKILL_TEXT)
        sig = module.predictor.predict.signature
        input_fields = list(sig.input_fields.keys())
        output_fields = list(sig.output_fields.keys())
        assert "task_input" in input_fields
        assert "output" in output_fields
        # skill_instructions should NOT be an input field — it's the instructions now
        assert "skill_instructions" not in input_fields

"""Tests for skill module loading, parsing, and DSPy module."""

import dspy
import pytest
from pathlib import Path
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    reassemble_skill,
)


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

TEST_SKILL_BODY = """# My Skill

## When to Use
When you need to do something.

## Procedure
1. Do step one
2. Do step two"""

TEST_SKILL_BODY_NEW = """# New Skill

## Procedure
1. Do new stuff"""


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
    """Tests for the SkillModule DSPy wrapper."""

    def test_skill_text_property_returns_embedded_text(self):
        """skill_text property returns the body embedded in signature instructions."""
        module = SkillModule(TEST_SKILL_BODY)
        assert module.skill_text == TEST_SKILL_BODY

    def test_skill_text_roundtrip_after_set(self):
        """Setting skill_text rebuilds the predictor with new instructions."""
        module = SkillModule(TEST_SKILL_BODY)
        module.skill_text = TEST_SKILL_BODY_NEW
        assert module.skill_text == TEST_SKILL_BODY_NEW

    def test_named_predictors_exposes_inner_predictor(self):
        """The module exposes a predictor via named_predictors() for GEPA."""
        module = SkillModule(TEST_SKILL_BODY)
        predictors = list(module.named_predictors())
        assert len(predictors) >= 1
        name, pred = predictors[0]
        assert "predict" in name
        assert pred.signature.instructions != ""

    def test_forward_returns_prediction_with_output(self, monkeypatch):
        """forward() produces a Prediction with an output field."""

        def mock_forward(self_chain, task_input):
            return dspy.Prediction(output="mock response for: " + task_input)

        monkeypatch.setattr(dspy.ChainOfThought, "forward", mock_forward)

        module = SkillModule(TEST_SKILL_BODY)
        result = module(task_input="do something")
        assert isinstance(result, dspy.Prediction)
        assert hasattr(result, "output")
        assert "mock response" in result.output

    def test_deepcopy_preserves_instructions(self):
        """deepcopy creates an independent module with same instructions."""
        module = SkillModule(TEST_SKILL_BODY)
        module2 = module.deepcopy()
        assert module2.skill_text == TEST_SKILL_BODY

    def test_deepcopy_mutations_are_independent(self):
        """Mutating deepcopy's instructions leaves original unchanged."""
        module = SkillModule(TEST_SKILL_BODY)
        module2 = module.deepcopy()
        for _name, pred in module2.named_predictors():
            pred.signature = pred.signature.with_instructions(
                f"Complete the task following these instructions.\n\n{TEST_SKILL_BODY_NEW}"
            )
            break
        assert module2.skill_text == TEST_SKILL_BODY_NEW
        assert module.skill_text == TEST_SKILL_BODY

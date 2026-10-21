"""Tests for skill module loading and parsing."""

import pytest
from pathlib import Path
from evolution.skills.skill_module import load_skill, reassemble_skill


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
    def test_skill_text_is_optimizable_instructions(self):
        from evolution.skills.skill_module import SkillModule

        module = SkillModule("# Body\nDo X.")
        assert module.skill_text == "# Body\nDo X."
        # GEPA mutates the predictor's signature instructions; skill_text must follow
        pred = module.predictor
        pred.signature = pred.signature.with_instructions("# Evolved")
        assert module.skill_text == "# Evolved"

    def test_prediction_consumers_require_outputs_not_chain_of_thought(self):
        from evolution.core.config import EvolutionConfig
        from evolution.core.dataset_builder import SyntheticDatasetBuilder
        from evolution.core.external_importers import RelevanceFilter
        from evolution.core.fitness import LLMJudge
        from evolution.skills.skill_module import SkillModule

        config = EvolutionConfig()
        predictors = [SkillModule("Answer accurately").predictor,
                      LLMJudge(config).judge,
                      SyntheticDatasetBuilder(config).generator,
                      RelevanceFilter("copilot/claude-sonnet-5.5").scorer]
        for predictor in predictors:
            assert "reasoning" not in predictor.signature.output_fields
        assert "feedback" in predictors[1].signature.output_fields

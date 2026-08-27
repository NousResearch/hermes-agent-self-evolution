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


# ── Additions covering the audited gaps ─────────────────────────────────


class TestFrontmatterParsing:
    """Nested `metadata:` blocks must not shadow the skill's own fields.

    182 of the installed skills carry a nested metadata block, and a naive
    line scan picks up whichever `name:` it meets first.
    """

    def test_nested_name_does_not_shadow_the_real_one(self, tmp_path):
        from evolution.skills.skill_module import load_skill

        path = tmp_path / "SKILL.md"
        path.write_text(
            "---\n"
            "name: real-name\n"
            "description: the real description\n"
            "version: 2.3.4\n"
            "metadata:\n"
            "  hermes:\n"
            "    name: nested-decoy\n"
            "    description: nested decoy description\n"
            "---\n\nBody.\n"
        )
        skill = load_skill(path)
        assert skill["name"] == "real-name"
        assert skill["description"] == "the real description"
        assert skill["version"] == "2.3.4"

    def test_list_items_are_not_treated_as_fields(self, tmp_path):
        from evolution.skills.skill_module import parse_frontmatter_fields

        fields = parse_frontmatter_fields("name: x\ntags:\n- name: decoy\n")
        assert fields["name"] == "x"


class TestVersionBump:
    """An evolved skill that keeps its predecessor's version is indistinguishable."""

    def test_patch_version_is_incremented(self):
        from evolution.skills.skill_module import bump_version

        assert "version: 1.2.4" in bump_version("name: x\nversion: 1.2.3")

    def test_two_part_versions_gain_a_patch(self):
        from evolution.skills.skill_module import bump_version

        assert "version: 1.2.1" in bump_version("name: x\nversion: 1.2")

    def test_a_missing_version_is_added_after_name(self):
        from evolution.skills.skill_module import bump_version

        result = bump_version("name: x\ndescription: y")
        lines = result.split("\n")
        assert lines[0] == "name: x"
        assert lines[1] == "version: 1.0.1"

    def test_a_nonnumeric_version_is_replaced_not_mangled(self):
        from evolution.skills.skill_module import bump_version

        assert "version: 1.0.1" in bump_version("name: x\nversion: draft")

    def test_other_frontmatter_is_preserved(self):
        from evolution.skills.skill_module import bump_version

        out = bump_version("name: x\nversion: 1.0.0\nauthor: someone\nlicense: MIT")
        assert "author: someone" in out and "license: MIT" in out


class TestFindSkillAcrossTrees:
    """Skills live in profile, user and repo trees; searching one missed most."""

    def _make(self, root, name, body="---\nname: %s\n---\n\nx\n"):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(body % name if "%s" in body else body)
        return d / "SKILL.md"

    def test_finds_a_skill_in_a_secondary_tree(self, tmp_path):
        from evolution.skills.skill_module import find_skill

        repo_skills = tmp_path / "repo" / "skills"
        user_skills = tmp_path / "user" / "skills"
        self._make(repo_skills, "in-repo")
        target = self._make(user_skills, "in-user")

        assert find_skill("in-user", repo_skills, user_skills) == target

    def test_a_parent_directory_is_accepted(self, tmp_path):
        from evolution.skills.skill_module import find_skill

        target = self._make(tmp_path / "repo" / "skills", "demo")
        assert find_skill("demo", tmp_path / "repo") == target

    def test_earlier_trees_win_on_a_name_clash(self, tmp_path):
        from evolution.skills.skill_module import find_skill

        first = self._make(tmp_path / "a" / "skills", "dup")
        self._make(tmp_path / "b" / "skills", "dup")
        assert find_skill("dup", tmp_path / "a" / "skills", tmp_path / "b" / "skills") == first

    def test_frontmatter_name_matches_when_the_directory_does_not(self, tmp_path):
        from evolution.skills.skill_module import find_skill

        d = tmp_path / "skills" / "weird-dir-name"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: actual-name\n---\n\nx\n")
        assert find_skill("actual-name", tmp_path / "skills") == d / "SKILL.md"

    def test_missing_skill_returns_none(self, tmp_path):
        from evolution.skills.skill_module import find_skill

        (tmp_path / "skills").mkdir()
        assert find_skill("ghost", tmp_path / "skills") is None


class TestPredictionCarriesCandidateText:
    """The metric cannot judge the candidate unless the module says what it is."""

    def test_forward_attaches_the_live_instructions(self):
        import dspy
        from evolution.skills.skill_module import SkillModule

        module = SkillModule("ORIGINAL SKILL BODY")

        class FakePredict:
            def __call__(self, task_input):
                return dspy.Prediction(output="an answer")

        module.predictor = FakePredict()
        module.predictor.predict = type("P", (), {"signature": type("S", (), {"instructions": "MUTATED BODY"})})()

        result = module(task_input="x")
        assert result.skill_text == "MUTATED BODY"

    def test_bundle_context_is_stripped_from_the_evolved_text(self):
        from evolution.skills.skill_module import SkillModule

        module = SkillModule("BODY", bundle_context="reference file excerpt")
        assert "reference file excerpt" in module.current_instructions()
        assert module.get_evolved_text() == "BODY"

    def test_without_bundle_context_the_text_is_unchanged(self):
        from evolution.skills.skill_module import SkillModule

        assert SkillModule("BODY").get_evolved_text() == "BODY"

"""Regression tests for the Phase 1 skill-evolution write-back path.

Two defects made `evolve_skill` unable to ever emit a genuinely evolved skill:

1. `SkillModule` carried the skill text in a plain Python attribute and passed it
   to the signature as an InputField, so no DSPy optimizer could rewrite it.
   `evolve_skill.py` then read that same untouched attribute back as the
   "evolved" body.

2. The constraint validator was handed `skill["body"]`, but `_check_skill_structure`
   requires YAML frontmatter — which `load_skill` has already stripped out of the
   body. Every evolved candidate therefore failed structural validation and was
   rejected before the holdout comparison ran.

These tests pin both behaviours so they cannot silently regress.
"""

import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    reassemble_skill,
)


SAMPLE_SKILL = """---
name: test-skill
description: A skill for testing things
version: 1.0.0
---

# Test Skill — Testing Things

## Procedure
1. First, do the thing
2. Then, verify it worked
"""


@pytest.fixture
def config(tmp_path):
    return EvolutionConfig(hermes_agent_path=tmp_path)


class TestSkillTextIsOptimizable:
    """Defect 1: the skill text must be optimizer-visible state, not a bare attribute."""

    def test_skill_text_is_exposed_as_predictor_instructions(self):
        module = SkillModule("# Baseline procedure\n1. Do the thing")

        instructions = [
            predictor.signature.instructions
            for _, predictor in module.named_predictors()
        ]

        assert instructions, "module exposes no predictors for the optimizer to touch"
        assert any(
            "Baseline procedure" in text for text in instructions
        ), "skill text is not reachable through any predictor's signature instructions"

    def test_evolved_instructions_are_read_back(self):
        """What the optimizer rewrites must be what we serialize."""
        module = SkillModule("# Baseline procedure")

        # Simulate what a DSPy optimizer does: rewrite signature instructions.
        for _, predictor in module.named_predictors():
            predictor.signature = predictor.signature.with_instructions(
                "# EVOLVED procedure\n1. Do the better thing"
            )

        assert "EVOLVED procedure" in module.skill_text
        assert "Baseline procedure" not in module.skill_text


class TestStructureValidationTarget:
    """Defect 2: structural validation must run against the reassembled file."""

    def test_bare_body_has_no_frontmatter(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        # Precondition for the bug: load_skill strips frontmatter out of the body.
        assert not skill["body"].lstrip().startswith("---")

    def test_bare_body_fails_structure_check(self, tmp_path, config):
        """Pins WHY passing the body was wrong — it can never pass."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        result = ConstraintValidator(config)._check_skill_structure(skill["body"])
        assert not result.passed

    def test_reassembled_skill_passes_structure_check(self, tmp_path, config):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        evolved_full = reassemble_skill(skill["frontmatter"], skill["body"])
        result = ConstraintValidator(config)._check_skill_structure(evolved_full)

        assert result.passed, result.message

    def test_validate_skill_reassembles_internally(self, tmp_path, config):
        """The call site passes frontmatter + body; the validator owns assembly.

        This is what makes the original defect unrepresentable: callers can no
        longer hand a bare body to skill validation by mistake.
        """
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(SAMPLE_SKILL)
        skill = load_skill(skill_file)

        results = ConstraintValidator(config).validate_skill(
            frontmatter=skill["frontmatter"],
            body=skill["body"],
            baseline_body=skill["body"],
        )

        failed = [r for r in results if not r.passed]
        assert not failed, f"unexpected failures: {[(r.constraint_name, r.message) for r in failed]}"

    def test_validate_skill_still_catches_broken_frontmatter(self, tmp_path, config):
        """Reassembly must not paper over genuinely malformed frontmatter."""
        results = ConstraintValidator(config).validate_skill(
            frontmatter="version: 1.0.0",  # no name, no description
            body="# Body",
        )

        structure = [r for r in results if r.constraint_name == "skill_structure"]
        assert structure and not structure[0].passed

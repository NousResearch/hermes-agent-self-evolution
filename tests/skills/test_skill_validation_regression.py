"""Regression tests for skill validation using full SKILL.md text."""

from pathlib import Path

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_io import load_skill, reassemble_skill


def _write_test_skill(tmp_path: Path) -> Path:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(
        "---\n"
        "name: obsidian\n"
        "description: Test skill\n"
        "---\n\n"
        "# Test Skill\n\n"
        "Use this skill for testing.\n"
    )
    return skill_path


def test_baseline_skill_raw_passes_structure_check(tmp_path):
    validator = ConstraintValidator(EvolutionConfig())
    skill = load_skill(_write_test_skill(tmp_path))

    results = validator.validate_all(skill["raw"], "skill")

    structure = [r for r in results if r.constraint_name == "skill_structure"][0]
    assert structure.passed


def test_reassembled_skill_passes_structure_check(tmp_path):
    validator = ConstraintValidator(EvolutionConfig())
    skill = load_skill(_write_test_skill(tmp_path))
    evolved_full = reassemble_skill(skill["frontmatter"], skill["body"])

    results = validator.validate_all(evolved_full, "skill", baseline_text=skill["raw"])

    structure = [r for r in results if r.constraint_name == "skill_structure"][0]
    assert structure.passed

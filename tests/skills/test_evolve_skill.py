"""Regression tests for skill evolution validation behavior."""

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.skills.evolve_skill import validate_skill_constraints
from evolution.skills.skill_module import load_skill


SAMPLE_SKILL = """---
name: sample-skill
description: Sample description
version: 1.0.0
---

# Sample Skill

## Procedure
1. Do the thing
"""


FRONTMATTER_HEAVY_SKILL = """---
name: sample-skill
description: This description is intentionally very long to make the frontmatter dominate total file size during regression testing for growth-limit behavior.
version: 1.0.0
author: Test Author
license: MIT
metadata:
  hermes:
    tags: [testing, growth, regression, validation]
    related_skills: [example-one, example-two, example-three]
---

# Sample Skill

short body
"""


EMPTY_BODY_SKILL = """---
name: empty-body-skill
description: Skill with valid frontmatter but no body content.
version: 1.0.0
---
"""


def _result_map(results):
    return {result.constraint_name: result for result in results}


def test_baseline_validation_uses_full_skill_file_for_structure(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(SAMPLE_SKILL)
    skill = load_skill(skill_file)

    validator = ConstraintValidator(EvolutionConfig())
    results = validate_skill_constraints(validator, skill)

    assert all(r.passed for r in results)


def test_evolved_validation_reassembles_frontmatter_before_structure_check(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(SAMPLE_SKILL)
    skill = load_skill(skill_file)

    evolved_body = "# Sample Skill\n\n## Procedure\n1. Do the task\n"

    validator = ConstraintValidator(EvolutionConfig())
    results = validate_skill_constraints(validator, skill, evolved_body=evolved_body)

    assert all(r.passed for r in results)


def test_growth_limit_still_applies_to_body_not_full_file(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(FRONTMATTER_HEAVY_SKILL)
    skill = load_skill(skill_file)

    evolved_body = "# Sample Skill\n\nshort body now much longer than before\n"

    config = EvolutionConfig(max_prompt_growth=0.2)
    validator = ConstraintValidator(config)
    results = _result_map(validate_skill_constraints(validator, skill, evolved_body=evolved_body))

    assert not results["growth_limit"].passed
    assert results["skill_structure"].passed


def test_growth_limit_still_runs_when_baseline_body_is_empty(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(EMPTY_BODY_SKILL)
    skill = load_skill(skill_file)

    evolved_body = "# Empty Body Skill\n\nNow there is substantial body content.\n"

    validator = ConstraintValidator(EvolutionConfig(max_prompt_growth=0.2))
    results = _result_map(validate_skill_constraints(validator, skill, evolved_body=evolved_body))

    assert "growth_limit" in results
    assert not results["growth_limit"].passed
    assert results["skill_structure"].passed

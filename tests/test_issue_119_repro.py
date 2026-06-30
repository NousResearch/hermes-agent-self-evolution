"""Repro tests for NousResearch/hermes-agent-self-evolution issue #119.

Bug 1: evolve_skill.py validates `evolved_body` (which deliberately has no
frontmatter -- it's re-added separately by reassemble_skill) against the
`skill_structure` constraint, which requires frontmatter. This means the
structure check always fails, even for a perfectly valid evolved skill.

Bug 2: reassemble_skill() unconditionally prepends the baseline frontmatter
onto evolved_body. If evolved_body already contains its own frontmatter
block (which the optimizer sometimes writes), the result has two nested
frontmatter blocks instead of one.
"""

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import reassemble_skill


def test_bug1_validating_body_alone_always_fails_structure_even_when_full_text_is_valid():
    config = EvolutionConfig()
    validator = ConstraintValidator(config)

    frontmatter = "name: my-skill\ndescription: Does stuff"
    evolved_body = "# My Skill\nDo the thing well."

    # This is what evolve_skill.py currently does (the bug): validate the body
    # alone, which never has frontmatter by design.
    results_on_body_alone = validator.validate_all(evolved_body, "skill")
    structure_on_body = next(r for r in results_on_body_alone if r.constraint_name == "skill_structure")
    assert not structure_on_body.passed  # demonstrates the bug

    # This is what it should validate instead: the reassembled full text,
    # which does have frontmatter and is a perfectly valid skill.
    evolved_full = reassemble_skill(frontmatter, evolved_body)
    results_on_full = validator.validate_all(evolved_full, "skill")
    structure_on_full = next(r for r in results_on_full if r.constraint_name == "skill_structure")
    assert structure_on_full.passed


def test_bug2_reassemble_nests_frontmatter_when_evolved_body_has_its_own():
    frontmatter = "name: my-skill\ndescription: Does stuff"
    evolved_body_with_own_frontmatter = (
        "---\nname: my-skill\ndescription: A different description\n---\n\n"
        "# My Skill\nDo the thing well."
    )

    result = reassemble_skill(frontmatter, evolved_body_with_own_frontmatter)

    # Current (buggy) behavior produces two nested frontmatter blocks (4 "---"
    # markers). A correct reassembly should produce exactly one block (2
    # markers).
    assert result.count("---") == 2, f"expected exactly one frontmatter block, got: {result!r}"
    assert "# My Skill" in result
    assert "Do the thing well." in result
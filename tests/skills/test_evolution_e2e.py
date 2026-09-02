"""End-to-end proof that the write-back path works, using a fake optimizer.

No network, no API keys: a stub optimizer stands in for GEPA and rewrites the
signature instructions the way a real one does. This exercises the whole
extract -> guard -> validate -> reassemble chain.
"""

import pytest

from evolution.skills.skill_module import SkillModule, load_skill, reassemble_skill


SAMPLE_SKILL = """---
name: demo-skill
description: Demo skill
---

# Demo

## Procedure
1. Baseline step
"""


class FakeOptimizer:
    """Stands in for dspy.GEPA: rewrites predictor signature instructions."""

    def __init__(self, new_instructions: str):
        self.new_instructions = new_instructions

    def compile(self, module, **_kwargs):
        for _, predictor in module.named_predictors():
            predictor.signature = predictor.signature.with_instructions(
                self.new_instructions
            )
        return module


def test_optimizer_rewrite_survives_extraction_and_reassembly(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(SAMPLE_SKILL)
    skill = load_skill(skill_file)

    module = SkillModule(skill["body"])
    evolved_text = "# Demo\n\n## Procedure\n1. Evolved step\n2. Extra verification step"

    optimized = FakeOptimizer(evolved_text).compile(module)

    evolved_body = optimized.skill_text
    assert evolved_body != skill["body"], "optimizer rewrite was lost"
    assert "Evolved step" in evolved_body

    evolved_full = reassemble_skill(skill["frontmatter"], evolved_body)
    assert evolved_full.startswith("---")
    assert "name: demo-skill" in evolved_full
    assert "Evolved step" in evolved_full


def test_noop_optimizer_is_detectable(tmp_path):
    """SABOTAGE CHECK: an optimizer that changes nothing must be detectable.

    This is the condition the pipeline's no-op guard aborts on. If this test
    ever passes trivially, the guard is not measuring anything.
    """
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text(SAMPLE_SKILL)
    skill = load_skill(skill_file)

    module = SkillModule(skill["body"])

    class NoOpOptimizer:
        def compile(self, module, **_kwargs):
            return module  # the original bug's behaviour

    optimized = NoOpOptimizer().compile(module)

    assert optimized.skill_text.strip() == skill["body"].strip(), (
        "no-op optimizer somehow changed the text; guard would not fire"
    )

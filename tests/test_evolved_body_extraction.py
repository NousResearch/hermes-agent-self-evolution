"""Regression tests for the evolved-body extraction fix.

Prior to this fix, ``evolve_skill.py`` read ``optimized_module.skill_text`` —
the fixed input field that is passed verbatim on every forward pass and is
never mutated by GEPA/MIPRO. As a result every "evolved" skill came out
byte-identical to the baseline (confirmed on a 28KB skill: 28,431 == 28,431
chars). The optimizer actually rewrites the predictor's *signature
instructions*; ``_extract_evolved_body`` now reads those.
"""

from evolution.skills.evolve_skill import _extract_evolved_body
from evolution.skills.skill_module import SkillModule


def test_unoptimized_module_falls_back_to_skill_text():
    """A module whose instruction still equals the stock docstring yields the
    original body — never a partial/generic wrapper."""
    m = SkillModule("ORIGINAL BODY 123")
    assert _extract_evolved_body(m) == "ORIGINAL BODY 123"


def test_rewritten_instruction_is_extracted():
    """When the optimizer rewrites the signature instruction, that rewrite is
    what gets returned — not the untouched skill_text input field."""
    m = SkillModule("ORIGINAL BODY 123")
    m.predictor.predict.signature = m.predictor.predict.signature.with_instructions(
        "CLASSIFY the request then run biofigure.py search with the exact CLI contract."
    )
    out = _extract_evolved_body(m)
    assert "CLASSIFY the request" in out
    assert out != "ORIGINAL BODY 123"


def test_instruction_identical_to_baseline_falls_back():
    """If the optimizer sets the instruction identical to the stock docstring,
    there is no real evolution — keep the original body."""
    m = SkillModule("ORIGINAL BODY 123")
    baseline_instr = SkillModule("").predictor.predict.signature.instructions
    m.predictor.predict.signature = m.predictor.predict.signature.with_instructions(
        baseline_instr
    )
    assert _extract_evolved_body(m) == "ORIGINAL BODY 123"

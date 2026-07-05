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


def test_collapsed_instruction_falls_back_to_baseline():
    """Overfit/collapse guard: when the optimizer rewrites the instruction into
    something far shorter than the baseline body (a narrow task recipe overfit to
    the synthetic eval set), we must NOT substitute it — that would replace a rich
    reference skill with a stub. Reproduces the observed biofigure failure: a large
    skill body collapsing to a tiny task-specific procedure."""
    big_body = "RICH SKILL BODY. " * 500  # ~8.5KB, like a real reference skill
    m = SkillModule(big_body)
    m.predictor.predict.signature = m.predictor.predict.signature.with_instructions(
        "Embed one WikiPathways SVG and export a PDF with credit."  # tiny overfit
    )
    # evolved instruction is <60% of baseline body -> guard rejects it
    assert _extract_evolved_body(m) == big_body


def test_substantial_rewrite_is_accepted():
    """A rewrite that preserves most of the body (genuine refinement, not a
    collapse) is still accepted — the shrink guard must not block real evolution."""
    body = "STEP ONE do X. STEP TWO do Y. STEP THREE do Z. " * 20
    m = SkillModule(body)
    rewritten = "STEP ONE do X carefully. STEP TWO do Y then verify. STEP THREE do Z and log. " * 20
    m.predictor.predict.signature = m.predictor.predict.signature.with_instructions(
        rewritten
    )
    out = _extract_evolved_body(m)
    assert out == rewritten.strip()
    assert out != body

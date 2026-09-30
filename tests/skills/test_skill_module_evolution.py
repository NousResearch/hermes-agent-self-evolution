"""The skill text must live in the channel GEPA actually mutates.

GEPA seeds its candidate space from
`{name: pred.signature.instructions for ...}` (dspy/teleprompt/gepa/gepa.py) and
rewrites those strings reflectively. The original `SkillModule` kept the skill
in a plain `self.skill_text` attribute next to the predictor, so GEPA optimised
the signature docstring and the "evolved" skill came back byte-identical to the
baseline — a silent no-op that still reports success.

These tests pin the property that makes Phase 1 actually evolve a skill.
"""

import dspy
import pytest

from evolution.skills.skill_module import (
    SkillModule,
    load_skill,
    find_skill,
    reassemble_skill,
)


# ── the skill text is inside the instructions ───────────────────────────────

def test_skill_text_is_carried_inside_predictor_instructions():
    m = SkillModule("DO THE THING CAREFULLY")
    instr = m.predictor.predict.signature.instructions
    assert "DO THE THING CAREFULLY" in instr, (
        "skill text must live in signature.instructions — that is the only "
        "text GEPA mutates"
    )


def test_gepa_seed_candidate_would_contain_the_skill():
    """Reproduce GEPA's own candidate construction: whatever it seeds is what
    it can evolve. The skill must appear there."""
    m = SkillModule("UNIQUE_SKILL_TOKEN_42")
    # This mirrors gepa.py: seed_candidate = {name: pred.signature.instructions}
    seed_candidate = {n: p.signature.instructions for n, p in m.named_predictors()}
    assert any("UNIQUE_SKILL_TOKEN_42" in v for v in seed_candidate.values()), (
        "GEPA's seed candidate does not contain the skill text; GEPA would "
        "evolve the docstring instead of the skill"
    )


# ── property returns live skill text ────────────────────────────────────────

def test_skill_text_property_returns_constructor_arg():
    m = SkillModule("BASELINE: be terse.")
    assert m.skill_text == "BASELINE: be terse."


def test_skill_text_property_reflects_optimizer_mutations():
    """When the optimizer mutates the signature, the property must reflect it."""
    m = SkillModule("BASELINE: be terse.")
    m.predictor.predict.signature = m.predictor.predict.signature.with_instructions(
        "EVOLVED: be more thorough."
    )
    assert m.skill_text == "EVOLVED: be more thorough."
    assert m.skill_text != "BASELINE: be terse."


# ── the module still behaves as a DSPy module ───────────────────────────────

def test_forward_passes_task_input(monkeypatch):
    m = SkillModule("SKILL BODY HERE")
    seen = {}

    class FakeResult:
        output = "done"

    monkeypatch.setattr(
        type(m.predictor.predict), "__call__",
        lambda self, **kwargs: (seen.update(kwargs), FakeResult())[1],
    )
    pred = m(task_input="do it")
    assert seen["task_input"] == "do it"
    assert pred.output == "done"


def test_skill_body_size_matches_skill_text():
    """The constraint validator budgets by len(skill_text); the text must
    match what the optimizer sees."""
    body = "x" * 5000
    m = SkillModule(body)
    assert len(m.skill_text) == 5000


# ── load/find/reassemble unchanged ──────────────────────────────────────────

def test_reassemble_preserves_frontmatter():
    out = reassemble_skill("name: demo\ndescription: d", "BODY")
    assert out.startswith("---")
    assert "name: demo" in out
    assert "BODY" in out
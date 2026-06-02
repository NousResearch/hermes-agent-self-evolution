"""PromptModule — DSPy wrapper exposing the candidate section as predictor instructions."""
from __future__ import annotations

from evolution.prompts.prompt_module import PromptModule


def test_stores_candidate_in_predictor_instructions():
    module = PromptModule(
        section_name="MEMORY_GUIDANCE",
        candidate_text="evolved candidate body",
    )
    instructions = module.passthrough.predict.signature.instructions
    assert "evolved candidate body" in instructions
    assert "MEMORY_GUIDANCE" in instructions


def test_section_text_extracts_current_candidate():
    module = PromptModule(section_name="MEMORY_GUIDANCE", candidate_text="v1")
    assert module.section_text == "v1"
    # Simulate a GEPA mutation of the instructions.
    new_instructions = module.passthrough.predict.signature.instructions.replace(
        "v1", "v2-mutated"
    )
    module.passthrough.predict.signature = (
        module.passthrough.predict.signature.with_instructions(new_instructions)
    )
    assert module.section_text == "v2-mutated"


def test_forward_invokes_predictor_and_attaches_behavioral_fields():
    """forward calls the passthrough predictor (so GEPA gets a trace) and
    attaches the candidate text + task id for the metric's behavioral branch."""
    import dspy
    from dspy.utils.dummies import DummyLM

    module = PromptModule(section_name="MEMORY_GUIDANCE", candidate_text="evolved body")
    # DummyLM lets the real predictor run offline (no network), so section_text
    # still resolves while a predictor trace is produced for GEPA.
    with dspy.context(lm=DummyLM([{"reasoning": "n/a", "response": "placeholder"}])):
        pred = module.forward(task="anything", closed_loop_task_id="task-001")

    assert pred._candidate_text == "evolved body"
    assert pred._closed_loop_task_id == "task-001"


def test_named_predictors_exposes_target():
    """GEPA discovers mutation targets via named_predictors(); the passthrough
    predictor must be visible there."""
    module = PromptModule(section_name="MEMORY_GUIDANCE", candidate_text="x")
    names = [name for name, _ in module.named_predictors()]
    assert any("passthrough" in n for n in names)

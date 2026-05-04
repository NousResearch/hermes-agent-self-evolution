"""Regression tests for the skill evolution pipeline."""

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.core.fitness import skill_fitness_metric
from evolution.skills.evolve_skill import build_gepa_kwargs, validate_evolved_skill
from evolution.skills.skill_module import SkillModule, reassemble_skill


SAMPLE_FRONTMATTER = "name: regression-skill\ndescription: Regression test skill"
SAMPLE_BODY = "# Regression Skill\n\nFollow the documented procedure."
SAMPLE_SKILL = {
    "frontmatter": SAMPLE_FRONTMATTER,
    "body": SAMPLE_BODY,
    "raw": reassemble_skill(SAMPLE_FRONTMATTER, SAMPLE_BODY),
}


def test_build_gepa_kwargs_prefers_full_eval_budget(monkeypatch):
    class FakeGEPA:
        def __init__(self, metric, max_full_evals=None, max_metric_calls=None):
            pass

    monkeypatch.setattr("evolution.skills.evolve_skill.dspy.GEPA", FakeGEPA)

    kwargs = build_gepa_kwargs(iterations=7)

    assert kwargs["metric"] is skill_fitness_metric
    assert kwargs["max_full_evals"] == 7
    assert "max_metric_calls" not in kwargs


def test_build_gepa_kwargs_supports_legacy_max_steps(monkeypatch):
    class FakeLegacyGEPA:
        def __init__(self, metric, max_steps=None):
            pass

    monkeypatch.setattr("evolution.skills.evolve_skill.dspy.GEPA", FakeLegacyGEPA)

    kwargs = build_gepa_kwargs(iterations=3)

    assert kwargs["max_steps"] == 3


def test_gepa_metric_accepts_reflective_feedback_signature():
    example = dspy.Example(
        task_input="Summarize a source",
        expected_behavior="mention source URL and capture durable notes",
    )
    prediction = dspy.Prediction(output="Capture notes and mention source URL.")

    plain_score = skill_fitness_metric(example, prediction)
    reflective_score = skill_fitness_metric(
        example,
        prediction,
        trace=[],
        pred_name="predictor",
        pred_trace=[],
    )

    assert isinstance(plain_score, float)
    assert reflective_score["score"] == plain_score
    assert "feedback" in reflective_score


def test_validate_evolved_skill_reassembles_full_skill_before_validation():
    validator = ConstraintValidator(EvolutionConfig())
    evolved_body = "# Regression Skill\n\nImproved procedure."

    evolved_full, results = validate_evolved_skill(validator, SAMPLE_SKILL, evolved_body)

    assert evolved_full.startswith("---\nname: regression-skill")
    assert "Improved procedure" in evolved_full
    assert all(result.passed for result in results)


def test_body_only_skill_validation_would_fail_structure():
    validator = ConstraintValidator(EvolutionConfig())

    results = validator.validate_all(SAMPLE_BODY, "skill", baseline_text=SAMPLE_BODY)

    structure = next(result for result in results if result.constraint_name == "skill_structure")
    assert not structure.passed


def test_skill_module_reads_mutated_signature_instructions():
    module = SkillModule("initial instructions")

    module.skill_text = "mutated instructions"

    assert module.skill_text == "mutated instructions"
    assert module.predictor.predict.signature.instructions == "mutated instructions"

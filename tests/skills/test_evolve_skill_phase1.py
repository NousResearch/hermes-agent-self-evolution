"""Regression tests for the Phase 1 skill evolution path.

These tests lock down the minimum behavior needed before this repo can be
used as a real skill-evolution experiment harness:
- the skill text itself must be the optimizable DSPy instruction text,
- evolved instruction text must be extractable after optimization,
- GEPA must be constructed with the DSPy 3.2+ API,
- skill structure constraints must validate the full SKILL.md text, not just
  the body without YAML frontmatter.
"""

import json
from pathlib import Path

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.skills.skill_module import SkillModule, reassemble_skill
from evolution.skills.evolve_skill import build_gepa_optimizer, evolve, validate_skill_candidate


def test_skill_module_places_skill_text_in_optimizable_signature_instructions():
    skill_text = "# Demo Skill\n\nFollow this exact evolved procedure."

    module = SkillModule(skill_text)
    signature = module.predictor.predict.signature

    assert skill_text in signature.instructions
    assert "skill_instructions" not in signature.fields
    assert "task_input" in signature.fields
    assert "output" in signature.fields


def test_skill_text_property_reflects_optimized_signature_instructions():
    module = SkillModule("# Original Skill\n\nOriginal procedure.")

    module.predictor.predict.signature = module.predictor.predict.signature.with_instructions(
        "# Evolved Skill\n\nImproved procedure."
    )

    assert module.skill_text == "# Evolved Skill\n\nImproved procedure."


def test_build_gepa_optimizer_uses_dspy_32_constructor(monkeypatch):
    captured = {}

    class FakeLM:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    class FakeGEPA:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("evolution.skills.evolve_skill.dspy.LM", FakeLM)
    monkeypatch.setattr("evolution.skills.evolve_skill.dspy.GEPA", FakeGEPA)

    optimizer = build_gepa_optimizer(iterations=7, optimizer_model="openai/test-reflector")

    assert isinstance(optimizer, FakeGEPA)
    assert captured["max_full_evals"] == 7
    assert "max_steps" not in captured
    assert captured["reflection_lm"].model == "openai/test-reflector"
    assert callable(captured["metric"])


def test_validate_skill_candidate_checks_structure_on_full_skill_text():
    config = EvolutionConfig(run_pytest=False)
    validator = ConstraintValidator(config)
    frontmatter = "name: demo-skill\ndescription: Demo skill"
    body = "# Demo Skill\n\n1. Do the thing."

    results = validate_skill_candidate(
        validator=validator,
        frontmatter=frontmatter,
        body=body,
        baseline_body=body,
    )

    by_name = {result.constraint_name: result for result in results}
    assert by_name["skill_structure"].passed
    assert by_name["size_limit"].passed
    assert by_name["non_empty"].passed


def test_reassembled_skill_candidate_is_what_structure_validator_expects():
    frontmatter = "name: demo-skill\ndescription: Demo skill"
    body = "# Demo Skill\n\n1. Do the thing."

    full_skill = reassemble_skill(frontmatter, body)
    result = ConstraintValidator(EvolutionConfig())._check_skill_structure(full_skill)

    assert result.passed


def test_evolve_golden_dataset_writes_baseline_evolved_and_metrics_without_api(tmp_path, monkeypatch):
    hermes_repo = tmp_path / "hermes-agent"
    skill_dir = hermes_repo / "skills" / "testing" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill\n---\n\n"
        "# Demo Skill\n\nReturn a concise answer. Include security context, evidence, and one next step. "
        "Keep the result practical for code review workflows.\n"
    )

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    examples = [
        {"task_input": "Review this tiny change", "expected_behavior": "security concise", "difficulty": "easy", "category": "review"},
        {"task_input": "Find the risk", "expected_behavior": "security concise", "difficulty": "medium", "category": "review"},
        {"task_input": "Summarize issue", "expected_behavior": "security concise", "difficulty": "medium", "category": "review"},
    ]
    for split, example in zip(["train", "val", "holdout"], examples):
        (dataset_dir / f"{split}.jsonl").write_text(json.dumps(example) + "\n")

    class FakeSkillModule:
        def __init__(self, skill_text):
            self.skill_text = skill_text

        def __call__(self, task_input):
            return dspy.Prediction(output="security concise")

    class FakeOptimizer:
        def compile(self, student, trainset, valset):
            student.skill_text = student.skill_text + "\n\n## Evolved\nSecurity."
            return student

    monkeypatch.setattr("evolution.skills.evolve_skill.SkillModule", FakeSkillModule)
    monkeypatch.setattr("evolution.skills.evolve_skill.build_gepa_optimizer", lambda **kwargs: FakeOptimizer())
    monkeypatch.setattr("evolution.skills.evolve_skill.dspy.LM", lambda *args, **kwargs: object())
    monkeypatch.setattr("evolution.skills.evolve_skill.dspy.configure", lambda **kwargs: None)

    output_root = tmp_path / "evolution-output"
    evolve(
        skill_name="demo-skill",
        iterations=1,
        eval_source="golden",
        dataset_path=str(dataset_dir),
        hermes_repo=str(hermes_repo),
        output_dir=str(output_root),
    )

    run_dirs = list((output_root / "demo-skill").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    baseline = (run_dir / "baseline_skill.md").read_text()
    evolved = (run_dir / "evolved_skill.md").read_text()
    metrics = json.loads((run_dir / "metrics.json").read_text())

    assert "## Evolved" not in baseline
    assert "## Evolved" in evolved
    assert metrics["skill_name"] == "demo-skill"
    assert metrics["constraints_passed"] is True

"""Tests for model wiring in evolve_skill."""

from types import SimpleNamespace
from unittest.mock import patch

from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.skills.evolve_skill import evolve


class _FakeOptimizer:
    def __init__(self):
        self.compile_calls = []

    def compile(self, baseline_module, trainset, valset=None):
        self.compile_calls.append(
            {
                "baseline_module": baseline_module,
                "trainset": trainset,
                "valset": valset,
            }
        )
        return _FakeModule("# Evolved\nBetter")


class _PassedConstraint:
    passed = True
    constraint_name = "ok"
    message = "ok"


class _FakeDatasetBuilder:
    def __init__(self, config):
        self.config = config

    def generate(self, artifact_text, artifact_type="skill"):
        example = EvalExample(task_input="task", expected_behavior="expected")
        return EvalDataset(train=[example], val=[example], holdout=[example])


class _FakeModule:
    def __init__(self, skill_text):
        self.skill_text = skill_text

    def __call__(self, task_input):
        return SimpleNamespace(output="done")


def test_gepa_uses_optimizer_model_for_reflection_lm(tmp_path):
    fake_gepa = _FakeOptimizer()
    created_lms = []

    def fake_create_lm(model, **kwargs):
        created_lms.append((model, kwargs))
        return f"lm:{model}"

    with patch("evolution.skills.evolve_skill.find_skill", return_value=tmp_path / "SKILL.md"), \
         patch("evolution.skills.evolve_skill.load_skill", return_value={
             "name": "demo",
             "description": "desc",
             "raw": "---\nname: demo\ndescription: desc\n---\n\n# Demo\nBody",
             "body": "# Demo\nBody",
             "frontmatter": "name: demo\ndescription: desc",
         }), \
         patch("evolution.skills.evolve_skill.SyntheticDatasetBuilder", _FakeDatasetBuilder), \
         patch("evolution.skills.evolve_skill.ConstraintValidator") as mock_validator_cls, \
         patch("evolution.skills.evolve_skill.SkillModule", _FakeModule), \
         patch("evolution.skills.evolve_skill.create_lm", side_effect=fake_create_lm), \
         patch("evolution.skills.evolve_skill.dspy.configure"), \
         patch("evolution.skills.evolve_skill.dspy.GEPA", return_value=fake_gepa), \
         patch("evolution.skills.evolve_skill.skill_fitness_metric", return_value=0.8), \
         patch("evolution.skills.evolve_skill.reassemble_skill", side_effect=lambda frontmatter, body: f"---\n{frontmatter}\n---\n\n{body}"):
        mock_validator_cls.return_value.validate_all.return_value = [_PassedConstraint()]

        evolve(
            skill_name="demo",
            iterations=2,
            eval_source="synthetic",
            optimizer_model="chatgpt/gpt-5.4",
            eval_model="openai/gpt-4.1-mini",
            hermes_repo=str(tmp_path),
        )

    assert created_lms == [
        ("openai/gpt-4.1-mini", {}),
        ("chatgpt/gpt-5.4", {}),
    ]
    assert fake_gepa.compile_calls

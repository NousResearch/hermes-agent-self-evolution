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


class _ExplodingGEPA:
    def __init__(self, *args, **kwargs):
        raise TypeError("GEPA.__init__() got an unexpected keyword argument 'max_steps'")


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


class _RecordingValidator:
    def __init__(self, config):
        self.config = config
        self.calls = []

    def validate_all(self, artifact_text, artifact_type, baseline_text=None):
        self.calls.append((artifact_text, artifact_type, baseline_text))
        return [_PassedConstraint()]


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


def test_evolve_validates_full_skill_text_not_body_only(tmp_path):
    fake_gepa = _FakeOptimizer()
    validator = _RecordingValidator(config=None)
    raw_skill = "---\nname: demo\ndescription: desc\n---\n\n# Demo\nBody"

    with patch("evolution.skills.evolve_skill.find_skill", return_value=tmp_path / "SKILL.md"), \
         patch("evolution.skills.evolve_skill.load_skill", return_value={
             "name": "demo",
             "description": "desc",
             "raw": raw_skill,
             "body": "# Demo\nBody",
             "frontmatter": "name: demo\ndescription: desc",
         }), \
         patch("evolution.skills.evolve_skill.SyntheticDatasetBuilder", _FakeDatasetBuilder), \
         patch("evolution.skills.evolve_skill.ConstraintValidator", return_value=validator), \
         patch("evolution.skills.evolve_skill.SkillModule", _FakeModule), \
         patch("evolution.skills.evolve_skill.create_lm", side_effect=lambda model, **kwargs: f"lm:{model}"), \
         patch("evolution.skills.evolve_skill.dspy.configure"), \
         patch("evolution.skills.evolve_skill.dspy.GEPA", return_value=fake_gepa), \
         patch("evolution.skills.evolve_skill.skill_fitness_metric", return_value=0.8), \
         patch("evolution.skills.evolve_skill.reassemble_skill", side_effect=lambda frontmatter, body: f"---\n{frontmatter}\n---\n\n{body}"):
        evolve(
            skill_name="demo",
            iterations=1,
            eval_source="synthetic",
            optimizer_model="chatgpt/gpt-5.4",
            eval_model="openai/gpt-4.1-mini",
            hermes_repo=str(tmp_path),
        )

    assert validator.calls[0] == (raw_skill, "skill", None)
    assert validator.calls[1] == (
        "---\nname: demo\ndescription: desc\n---\n\n# Evolved\nBetter",
        "skill",
        raw_skill,
    )


def test_dataset_path_takes_priority_over_eval_source(tmp_path):
    fake_gepa = _FakeOptimizer()
    validator = _RecordingValidator(config=None)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for split in ("train", "val", "holdout"):
        (dataset_dir / f"{split}.jsonl").write_text(
            '{"task_input":"task","expected_behavior":"expected"}\n'
        )

    generated = {"called": False}

    class _FailIfSyntheticBuilder:
        def __init__(self, config):
            generated["called"] = True

        def generate(self, artifact_text, artifact_type="skill"):
            raise AssertionError("synthetic builder should not run when dataset_path is provided")

    with patch("evolution.skills.evolve_skill.find_skill", return_value=tmp_path / "SKILL.md"), \
         patch("evolution.skills.evolve_skill.load_skill", return_value={
             "name": "demo",
             "description": "desc",
             "raw": "---\nname: demo\ndescription: desc\n---\n\n# Demo\nBody",
             "body": "# Demo\nBody",
             "frontmatter": "name: demo\ndescription: desc",
         }), \
         patch("evolution.skills.evolve_skill.SyntheticDatasetBuilder", _FailIfSyntheticBuilder), \
         patch("evolution.skills.evolve_skill.ConstraintValidator", return_value=validator), \
         patch("evolution.skills.evolve_skill.SkillModule", _FakeModule), \
         patch("evolution.skills.evolve_skill.create_lm", side_effect=lambda model, **kwargs: f"lm:{model}"), \
         patch("evolution.skills.evolve_skill.dspy.configure"), \
         patch("evolution.skills.evolve_skill.dspy.GEPA", return_value=fake_gepa), \
         patch("evolution.skills.evolve_skill.skill_fitness_metric", return_value=0.8), \
         patch("evolution.skills.evolve_skill.reassemble_skill", side_effect=lambda frontmatter, body: f"---\n{frontmatter}\n---\n\n{body}"):
        evolve(
            skill_name="demo",
            iterations=1,
            eval_source="synthetic",
            dataset_path=str(dataset_dir),
            optimizer_model="chatgpt/gpt-5.4",
            eval_model="openai/gpt-4.1-mini",
            hermes_repo=str(tmp_path),
        )

    assert generated["called"] is False
    assert fake_gepa.compile_calls[0]["trainset"]


def test_mipro_fallback_receives_valset(tmp_path):
    fake_mipro = _FakeOptimizer()
    validator = _RecordingValidator(config=None)

    with patch("evolution.skills.evolve_skill.find_skill", return_value=tmp_path / "SKILL.md"), \
         patch("evolution.skills.evolve_skill.load_skill", return_value={
             "name": "demo",
             "description": "desc",
             "raw": "---\nname: demo\ndescription: desc\n---\n\n# Demo\nBody",
             "body": "# Demo\nBody",
             "frontmatter": "name: demo\ndescription: desc",
         }), \
         patch("evolution.skills.evolve_skill.SyntheticDatasetBuilder", _FakeDatasetBuilder), \
         patch("evolution.skills.evolve_skill.ConstraintValidator", return_value=validator), \
         patch("evolution.skills.evolve_skill.SkillModule", _FakeModule), \
         patch("evolution.skills.evolve_skill.create_lm", side_effect=lambda model, **kwargs: f"lm:{model}"), \
         patch("evolution.skills.evolve_skill.dspy.configure"), \
         patch("evolution.skills.evolve_skill.dspy.GEPA", _ExplodingGEPA), \
         patch("evolution.skills.evolve_skill.dspy.MIPROv2", return_value=fake_mipro), \
         patch("evolution.skills.evolve_skill.skill_fitness_metric", return_value=0.8), \
         patch("evolution.skills.evolve_skill.reassemble_skill", side_effect=lambda frontmatter, body: f"---\n{frontmatter}\n---\n\n{body}"):
        evolve(
            skill_name="demo",
            iterations=1,
            eval_source="synthetic",
            optimizer_model="chatgpt/gpt-5.4",
            eval_model="openai/gpt-4.1-mini",
            hermes_repo=str(tmp_path),
        )

    assert fake_mipro.compile_calls
    assert fake_mipro.compile_calls[0]["valset"]

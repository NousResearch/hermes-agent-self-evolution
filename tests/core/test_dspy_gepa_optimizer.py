"""Tests for the DSPy/GEPA optimizer adapter."""

from __future__ import annotations

from types import SimpleNamespace

from evolution.optimizers.dspy_gepa import DSpyGepaConfig, run_dspy_gepa_skill_optimizer


class _FakeExample:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.inputs = ()

    def with_inputs(self, *names):
        self.inputs = names
        return self


class _FakePrediction(SimpleNamespace):
    pass


class _FakeLM:
    def __init__(self, recorder, model, **kwargs):
        recorder["lm_calls"].append({"model": model, "kwargs": kwargs})
        self.model = model
        self.kwargs = kwargs


class _FakeGEPA:
    def __init__(self, recorder, **kwargs):
        recorder["gepa_init"] = kwargs

    def compile(self, student, *, trainset, valset=None, teacher=None):
        self._assert_no_holdout(trainset)
        self._assert_no_holdout(valset or [])
        student.skill_text = (
            "# Test Skill\n\n"
            "GEPA optimized body.\n\n"
            "- mention train-only calibration\n"
            "- mention validation-only rubric\n"
        )
        return student

    @staticmethod
    def _assert_no_holdout(examples):
        for example in examples:
            blob = f"{example.task_input} {example.expected_behavior}"
            assert "holdout-only forbidden leak" not in blob


class _FakeDSpy:
    def __init__(self):
        self.recorder = {"lm_calls": [], "configured_lm": None, "gepa_init": None}

    def LM(self, model, **kwargs):
        return _FakeLM(self.recorder, model, **kwargs)

    def configure(self, *, lm):
        self.recorder["configured_lm"] = lm

    def Example(self, **kwargs):
        return _FakeExample(**kwargs)

    def Prediction(self, **kwargs):
        return _FakePrediction(**kwargs)

    def GEPA(self, **kwargs):
        return _FakeGEPA(self.recorder, **kwargs)


class _FakeSkillModule:
    def __init__(self, skill_text):
        self.skill_text = skill_text

    def __call__(self, task_input):
        return _FakePrediction(output=f"handled {task_input} with {self.skill_text}")


def test_dspy_gepa_adapter_uses_configurable_models_and_train_val_only(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value")
    fake_dspy = _FakeDSpy()
    train_examples = [
        {"split": "train", "task_input": "task for train", "expected_behavior": "mention train-only calibration"}
    ]
    val_examples = [
        {"split": "val", "task_input": "task for val", "expected_behavior": "mention validation-only rubric"}
    ]

    result = run_dspy_gepa_skill_optimizer(
        baseline_body="# Test Skill\n\nFollow the existing procedure.",
        train_examples=train_examples,
        val_examples=val_examples,
        config=DSpyGepaConfig(
            provider="deepseek",
            optimizer_model="deepseek-v4-pro",
            eval_model="deepseek-v4-flash",
            max_tokens=1024,
            temperature=0.1,
            timeout=45.0,
            extra_body={"thinking": {"type": "disabled"}},
            max_full_evals=7,
            reflection_minibatch_size=2,
            dspy_model_prefix="openai",
        ),
        dspy_module=fake_dspy,
        module_factory=_FakeSkillModule,
    )

    assert "GEPA optimized body" in result.evolved_body
    assert "train-only calibration" in result.evolved_body
    assert "validation-only rubric" in result.evolved_body
    assert "holdout-only forbidden leak" not in result.evolved_body

    lm_models = [call["model"] for call in fake_dspy.recorder["lm_calls"]]
    assert lm_models == ["openai/deepseek-v4-flash", "openai/deepseek-v4-pro"]
    assert fake_dspy.recorder["lm_calls"][0]["kwargs"]["api_base"] == "https://api.deepseek.com"
    assert fake_dspy.recorder["lm_calls"][0]["kwargs"]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert fake_dspy.recorder["gepa_init"]["max_full_evals"] == 7
    assert fake_dspy.recorder["gepa_init"]["reflection_lm"].model == "openai/deepseek-v4-pro"

    assert result.metadata["strategy"] == "dspy-gepa"
    assert result.metadata["provider"] == "deepseek"
    assert result.metadata["optimizer_model"] == "deepseek-v4-pro"
    assert result.metadata["eval_model"] == "deepseek-v4-flash"
    assert result.metadata["max_full_evals"] == 7
    assert result.metadata["train_examples"] == 1
    assert result.metadata["val_examples"] == 1
    assert result.metadata["holdout_examples_used_for_generation"] == 0
    assert "secret-value" not in str(result.metadata)

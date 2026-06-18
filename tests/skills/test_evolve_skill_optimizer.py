"""Tests for evolve_skill optimizer compatibility."""

import inspect

import dspy

from evolution.core.fitness import skill_fitness_metric
from evolution.skills import evolve_skill


def test_skill_fitness_metric_accepts_gepa_trace_arguments():
    """DSPy 3.x GEPA requires metrics to accept predictor trace arguments."""
    inspect.signature(skill_fitness_metric).bind(None, None, None, None, None)

    example = dspy.Example(
        task_input="Sort notes",
        expected_behavior="group email tasks separately",
    )
    prediction = dspy.Prediction(output="Email tasks: contact Sam")

    score = skill_fitness_metric(
        example,
        prediction,
        trace=[],
        pred_name="predictor",
        pred_trace=[],
    )

    assert 0.0 <= score <= 1.0


def test_build_gepa_optimizer_uses_dspy_3_api(monkeypatch):
    calls = {}

    class FakeGEPA:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    def fake_lm(model):
        return {"model": model}

    monkeypatch.setattr(evolve_skill.dspy, "GEPA", FakeGEPA)
    monkeypatch.setattr(evolve_skill.dspy, "LM", fake_lm)

    optimizer = evolve_skill._build_gepa_optimizer(
        iterations=7,
        optimizer_model="openai/gpt-4.1",
    )

    assert isinstance(optimizer, FakeGEPA)
    assert calls["metric"] is skill_fitness_metric
    assert calls["max_full_evals"] == 7
    assert calls["reflection_lm"] == {"model": "openai/gpt-4.1"}
    assert "max_steps" not in calls


def test_build_gepa_optimizer_clamps_empty_budget(monkeypatch):
    calls = {}

    class FakeGEPA:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(evolve_skill.dspy, "GEPA", FakeGEPA)
    monkeypatch.setattr(evolve_skill.dspy, "LM", lambda model: model)

    evolve_skill._build_gepa_optimizer(iterations=0, optimizer_model="openai/gpt-4.1")

    assert calls["max_full_evals"] == 1

"""Focused gates for the Phase 1 skill-evolution orchestrator."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from evolution.core.constraints import ConstraintResult
from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.skills import evolve_skill


class _Optimizer:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else object()
        self.error = error
        self.compile_kwargs = None

    def compile(self, module, **kwargs):
        self.compile_kwargs = {"module": module, **kwargs}
        if self.error is not None:
            raise self.error
        return self.result


def _compile(monkeypatch, gepa_factory, mipro_factory):
    monkeypatch.setattr(evolve_skill.dspy, "GEPA", gepa_factory)
    monkeypatch.setattr(evolve_skill.dspy, "MIPROv2", mipro_factory)
    return evolve_skill._compile_optimizer(
        metric="metric",
        iterations=7,
        reflection_lm="reflection-lm",
        baseline_module="baseline",
        trainset="train",
        valset="validation",
        num_threads=1,
    )


def test_gepa_keeps_reflection_lm_and_valset(monkeypatch):
    captured = {}
    gepa = _Optimizer(result="evolved")

    def build_gepa(**kwargs):
        captured.update(kwargs)
        return gepa

    def unexpected_mipro(**kwargs):
        raise AssertionError(f"unexpected MIPRO fallback: {kwargs}")

    result, name = _compile(monkeypatch, build_gepa, unexpected_mipro)

    assert (result, name) == ("evolved", "GEPA")
    assert captured == {
        "metric": "metric",
        "max_full_evals": 7,
        "reflection_lm": "reflection-lm",
        "num_threads": 1,
    }
    assert gepa.compile_kwargs == {
        "module": "baseline",
        "trainset": "train",
        "valset": "validation",
    }


@pytest.mark.parametrize("compatibility_error", [AttributeError("missing"), TypeError("renamed")])
def test_api_construction_error_uses_mipro_with_same_valset(
    monkeypatch, compatibility_error
):
    mipro = _Optimizer(result="fallback-evolved")

    def incompatible_gepa(**kwargs):
        raise compatibility_error

    def build_mipro(**kwargs):
        assert kwargs == {"metric": "metric", "auto": "light"}
        return mipro

    result, name = _compile(monkeypatch, incompatible_gepa, build_mipro)

    assert (result, name) == ("fallback-evolved", "MIPROv2")
    assert mipro.compile_kwargs == {
        "module": "baseline",
        "trainset": "train",
        "valset": "validation",
    }


def test_gepa_compile_failure_propagates_without_mipro(monkeypatch):
    gepa = _Optimizer(error=RuntimeError("provider timeout"))

    def unexpected_mipro(**kwargs):
        raise AssertionError(f"runtime failure triggered MIPRO: {kwargs}")

    with pytest.raises(RuntimeError, match="provider timeout"):
        _compile(monkeypatch, lambda **kwargs: gepa, unexpected_mipro)


@pytest.mark.parametrize("runtime_error", [ConnectionError("offline"), ValueError("bad judge")])
def test_gepa_construction_runtime_error_fails_closed(
    monkeypatch, runtime_error
):
    def broken_gepa(**kwargs):
        raise runtime_error

    def unexpected_mipro(**kwargs):
        raise AssertionError(f"runtime failure triggered MIPRO: {kwargs}")

    with pytest.raises(type(runtime_error), match=str(runtime_error)):
        _compile(monkeypatch, broken_gepa, unexpected_mipro)


def test_requested_test_suite_is_a_hard_gate():
    validator = SimpleNamespace(
        run_test_suite=lambda repo: ConstraintResult(
            passed=False,
            constraint_name="test_suite",
            message="Test suite failed",
            details="1 failed",
        )
    )

    assert not evolve_skill._run_test_suite_gate(validator, SimpleNamespace())


def test_success_requires_a_material_diff():
    assert evolve_skill._has_material_diff("before", "after")
    assert not evolve_skill._has_material_diff("same", "same")
    assert evolve_skill._evolution_succeeded(0.1, material_diff=True)
    assert not evolve_skill._evolution_succeeded(0.1, material_diff=False)
    assert not evolve_skill._evolution_succeeded(0.0, material_diff=True)


def _skill_repo(tmp_path):
    repo = tmp_path / "hermes-agent"
    skill_dir = repo / "skills" / "tests" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\n\n"
        "# Demo\n\nFollow the original procedure.\n"
    )
    return repo


def _dataset():
    example = EvalExample(
        task_input="Use the demo skill",
        expected_behavior="Follow the demo procedure",
        source="golden",
    )
    return EvalDataset(train=[example], val=[example], holdout=[example])


def _stub_runtime(monkeypatch, dataset):
    monkeypatch.setattr(
        evolve_skill.GoldenDatasetLoader,
        "load",
        lambda _path: dataset,
    )
    monkeypatch.setattr(evolve_skill.dspy, "LM", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(evolve_skill.dspy, "configure", lambda **_kwargs: None)


def test_run_tests_failure_stops_before_holdout_and_output(tmp_path, monkeypatch):
    repo = _skill_repo(tmp_path)
    _stub_runtime(monkeypatch, _dataset())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        evolve_skill,
        "_compile_optimizer",
        lambda **_kwargs: (
            SimpleNamespace(skill_text="# Demo\n\nFollow the improved procedure."),
            "GEPA",
        ),
    )
    gate_calls = []

    def fail_tests(self, hermes_repo):
        gate_calls.append(hermes_repo)
        return ConstraintResult(False, "test_suite", "Test suite failed", "1 failed")

    monkeypatch.setattr(
        evolve_skill.ConstraintValidator,
        "run_test_suite",
        fail_tests,
    )
    monkeypatch.setattr(
        evolve_skill,
        "skill_fitness_metric",
        lambda *_args, **_kwargs: pytest.fail("holdout ran after failed test gate"),
    )

    with pytest.raises(evolve_skill.click.ClickException, match="candidate rejected"):
        evolve_skill.evolve(
            skill_name="demo",
            eval_source="golden",
            dataset_path=str(tmp_path),
            hermes_repo=str(repo),
            run_tests=True,
        )

    assert gate_calls == [repo]
    assert not (tmp_path / "output").exists()


def test_noop_candidate_cannot_report_success_from_score_noise(tmp_path, monkeypatch):
    repo = _skill_repo(tmp_path)
    dataset = _dataset()
    _stub_runtime(monkeypatch, dataset)
    monkeypatch.chdir(tmp_path)

    class FakeSkillModule:
        def __init__(self, skill_text):
            self.skill_text = skill_text

        def __call__(self, **_kwargs):
            return SimpleNamespace(output="same behavior")

    monkeypatch.setattr(evolve_skill, "SkillModule", FakeSkillModule)
    monkeypatch.setattr(
        evolve_skill,
        "_compile_optimizer",
        lambda baseline_module, **_kwargs: (baseline_module, "GEPA"),
    )
    monkeypatch.setattr(
        evolve_skill.dspy,
        "context",
        lambda **_kwargs: nullcontext(),
    )
    scores = iter([0.1, 0.9])
    monkeypatch.setattr(
        evolve_skill,
        "skill_fitness_metric",
        lambda *_args, **_kwargs: next(scores),
    )

    evolve_skill.evolve(
        skill_name="demo",
        eval_source="golden",
        dataset_path=str(tmp_path),
        hermes_repo=str(repo),
    )

    metrics_files = list((tmp_path / "output" / "demo").glob("*/metrics.json"))
    assert len(metrics_files) == 1
    metrics = json.loads(metrics_files[0].read_text())
    assert metrics["improvement"] == pytest.approx(0.8)
    assert metrics["material_diff"] is False
    assert metrics["success"] is False

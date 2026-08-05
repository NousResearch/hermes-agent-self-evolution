"""Regression tests for skill evolution orchestration."""

from pathlib import Path

import dspy
from click.testing import CliRunner

from evolution.core.constraints import ConstraintResult
from evolution.core.dataset_builder import EvalDataset, EvalExample


def _make_skill_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "hermes-agent"
    skill_dir = repo / "skills" / "testing" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo skill\n---\n\n# Demo\n\nFollow the demo procedure.\n"
    )
    return repo


def _dataset() -> EvalDataset:
    example = EvalExample(
        task_input="Use the demo skill",
        expected_behavior="A concise demo response",
        source="golden",
    )
    return EvalDataset(train=[example], val=[example], holdout=[])


def test_gepa_optimizer_uses_current_dspy_api(monkeypatch):
    """DSPy 3.x GEPA does not accept the removed max_steps argument."""
    from evolution.skills.evolve_skill import _build_gepa_optimizer

    captured = {}

    class FakeGEPA:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(dspy, "GEPA", FakeGEPA)

    optimizer = _build_gepa_optimizer(metric=lambda *_: 1.0, iterations=7)

    assert isinstance(optimizer, FakeGEPA)
    assert captured["max_metric_calls"] == 7
    assert "max_steps" not in captured


def test_skill_constraints_validate_full_skill_not_body():
    """Skill structure validation needs YAML frontmatter, so validate reassembled raw text."""
    from evolution.core.config import EvolutionConfig
    from evolution.core.constraints import ConstraintValidator
    from evolution.skills.evolve_skill import _validate_skill_constraints

    validator = ConstraintValidator(EvolutionConfig())
    skill = {
        "frontmatter": "name: demo\ndescription: Demo skill",
        "body": "# Demo\n\nFollow the demo procedure.",
    }

    results = _validate_skill_constraints(validator, skill, skill["body"])

    by_name = {result.constraint_name: result for result in results}
    assert by_name["skill_structure"].passed


def test_run_tests_flag_is_a_hard_gate(tmp_path, monkeypatch):
    """--run-tests must stop deployment/output when the project test suite fails."""
    from evolution.core import constraints as constraints_mod
    from evolution.skills import evolve_skill as evolve_mod

    repo = _make_skill_repo(tmp_path)
    monkeypatch.setattr(evolve_mod.GoldenDatasetLoader, "load", lambda _path: _dataset())
    monkeypatch.setattr(evolve_mod.dspy, "LM", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(evolve_mod.dspy, "configure", lambda **_kwargs: None)

    class FakeOptimizer:
        def compile(self, module, trainset=None, valset=None):
            module.skill_text = "# Demo\n\nImproved procedure."
            return module

    monkeypatch.setattr(evolve_mod, "_build_gepa_optimizer", lambda **_kwargs: FakeOptimizer())
    monkeypatch.setattr(evolve_mod, "skill_fitness_metric", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        constraints_mod.ConstraintValidator,
        "run_test_suite",
        lambda self, hermes_repo: ConstraintResult(False, "test_suite", "Test suite failed", "boom"),
    )

    evolve_mod.evolve(
        skill_name="demo",
        eval_source="golden",
        dataset_path=str(tmp_path),
        hermes_repo=str(repo),
        run_tests=True,
    )

    assert not Path("output/demo").exists()


def test_dry_run_does_not_claim_to_create_pr(tmp_path, monkeypatch):
    from evolution.skills.evolve_skill import main

    monkeypatch.setenv("HOME", str(tmp_path / "empty"))
    repo = _make_skill_repo(tmp_path)

    result = CliRunner().invoke(
        main,
        ["--skill", "demo", "--hermes-repo", str(repo), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "create PR" not in result.output
    assert "PR" not in result.output
    assert "Would validate constraints" in result.output

"""Behavioural tests for the evolve_skill orchestrator.

The orchestrator is where three of the audited defects lived — the size
penalty computed from the baseline, the judge shown the wrong artifact, and
the holdout scored with a different metric than the search — and it had two
tests, one of which asserted on the *source text* of the module rather than
its behaviour. Grep-based tests pass while the behaviour rots and fail on
harmless refactors; both failure modes happened here.

These exercise the real functions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evolution.core.config import EvolutionConfig, skill_search_roots
from evolution.core.constraints import ConstraintValidator
from evolution.core.skill_bundle import load_bundle
from evolution.skills.evolve_skill import (
    EvolutionError,
    _git_root,
    _is_successful_improvement,
    _metric_score,
    _score_arms,
    evolve,
)
from evolution.skills.skill_module import load_skill


SKILL_TEXT = """---
name: demo-skill
description: A demo skill used in tests.
version: 1.2.3
metadata:
  hermes:
    name: not-the-skill-name
---

# Demo

Consult [the reference](references/detail.md) before answering.
"""


@pytest.fixture(autouse=True)
def isolate_from_real_install(tmp_path_factory, monkeypatch):
    """Never let a test read the developer's live Hermes install.

    Discovery falls back to ``~/.hermes``, so without this a machine that has
    Hermes installed silently substitutes its real skills and databases for
    the fixtures — the tests then pass or fail depending on whose laptop they
    run on.
    """
    fake_home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    for var in ("HERMES_DATA_DIR", "HERMES_HOME", "HERMES_AGENT_REPO", "HERMES_AGENT_SOURCE_REPO"):
        monkeypatch.delenv(var, raising=False)
    return fake_home


@pytest.fixture
def skill_tree(tmp_path: Path) -> Path:
    """A minimal skills tree with one bundled skill."""
    skill_dir = tmp_path / "skills" / "demo-skill"
    (skill_dir / "references").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_TEXT)
    (skill_dir / "references" / "detail.md").write_text("Detailed reference content.")
    return tmp_path


class TestIsSuccessfulImprovement:
    def test_requires_artifact_diff_and_positive_improvement(self):
        assert not _is_successful_improvement("same", "same", 0.1)
        assert not _is_successful_improvement("before", "after", 0.0)
        assert not _is_successful_improvement("before", "after", -0.1)
        assert _is_successful_improvement("before", "after", 0.1)


class TestBaselineValidation:
    """The baseline constraint pass must see the full file, not the bare body.

    load_skill() splits frontmatter off into skill["frontmatter"], so
    validating skill["body"] can never satisfy the skill_structure constraint
    and reported a false violation on every run.
    """

    def test_raw_baseline_passes_structure_but_body_alone_does_not(self, skill_tree):
        skill = load_skill(skill_tree / "skills" / "demo-skill" / "SKILL.md")
        validator = ConstraintValidator(EvolutionConfig(hermes_agent_path=None), size_budget=50_000)

        raw_results = {c.constraint_name: c.passed for c in validator.validate_all(skill["raw"], "skill")}
        body_results = {c.constraint_name: c.passed for c in validator.validate_all(skill["body"], "skill")}

        assert raw_results["skill_structure"] is True
        assert body_results["skill_structure"] is False


class TestBundleGate:
    """Dropping a link to a supporting file is a regression, not a size win."""

    def test_dropped_reference_fails_the_gate(self, skill_tree):
        entry = skill_tree / "skills" / "demo-skill" / "SKILL.md"
        bundle = load_bundle(entry, name="demo-skill")
        validator = ConstraintValidator(EvolutionConfig(hermes_agent_path=None), size_budget=50_000)

        stripped = SKILL_TEXT.replace("[the reference](references/detail.md)", "the reference")
        results = {c.constraint_name: c for c in validator.validate_all(stripped, "skill", bundle=bundle)}

        assert results["bundle_references"].passed is False
        assert "detail.md" in results["bundle_references"].message

    def test_invented_reference_fails_the_gate(self, skill_tree):
        entry = skill_tree / "skills" / "demo-skill" / "SKILL.md"
        bundle = load_bundle(entry, name="demo-skill")
        validator = ConstraintValidator(EvolutionConfig(hermes_agent_path=None), size_budget=50_000)

        invented = SKILL_TEXT + "\nAlso read [the api guide](references/api.md).\n"
        results = {c.constraint_name: c for c in validator.validate_all(invented, "skill", bundle=bundle)}

        assert results["bundle_references"].passed is False
        assert "api.md" in results["bundle_references"].message

    def test_intact_references_pass(self, skill_tree):
        entry = skill_tree / "skills" / "demo-skill" / "SKILL.md"
        bundle = load_bundle(entry, name="demo-skill")
        validator = ConstraintValidator(EvolutionConfig(hermes_agent_path=None), size_budget=50_000)

        results = {c.constraint_name: c for c in validator.validate_all(SKILL_TEXT, "skill", bundle=bundle)}
        assert results["bundle_references"].passed is True


class TestSkillSearchRoots:
    """Skills live in more than one tree; searching only the repo missed most."""

    def test_includes_repo_skills_and_optional_skills(self, tmp_path):
        repo = tmp_path / "hermes-agent"
        (repo / "skills").mkdir(parents=True)
        (repo / "optional-skills").mkdir(parents=True)
        config = EvolutionConfig(hermes_agent_path=repo)

        roots = skill_search_roots(config)

        assert repo / "skills" in roots
        assert repo / "optional-skills" in roots

    def test_missing_directories_are_dropped(self, tmp_path):
        repo = tmp_path / "hermes-agent"
        (repo / "skills").mkdir(parents=True)
        config = EvolutionConfig(hermes_agent_path=repo)

        roots = skill_search_roots(config)

        assert all(r.is_dir() for r in roots)
        assert repo / "optional-skills" not in roots


class TestMetricScore:
    """GEPA metrics return Prediction(score=…); plain metrics return a float."""

    def test_reads_score_attribute(self):
        class Pred:
            score = 0.75

        assert _metric_score(Pred()) == 0.75

    def test_accepts_bare_float(self):
        assert _metric_score(0.4) == 0.4

    def test_unparseable_scores_to_zero(self):
        assert _metric_score("not a number") == 0.0


class TestScoreArms:
    """Both arms must be scored by the same metric on the same examples.

    The reported delta was previously produced by a keyword-overlap heuristic
    while the search optimized an LLM judge, so the headline number measured
    something nobody was optimizing.
    """

    def test_both_arms_use_the_same_metric(self):
        seen = []

        class Example:
            task_input = "do the thing"

        class Module:
            def __init__(self, marker):
                self.marker = marker

            def __call__(self, task_input):
                class Pred:
                    output = self.marker
                    skill_text = self.marker

                return Pred()

        def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            seen.append(pred.output)
            return 0.5 if pred.output == "baseline" else 0.9

        baseline, evolved = _score_arms(
            [Example()], Module("baseline"), Module("evolved"), metric, lm=None
        )

        assert baseline == [0.5]
        assert evolved == [0.9]
        assert seen == ["baseline", "evolved"]

    def test_a_failing_example_does_not_abort_the_arm(self):
        class Example:
            task_input = "x"

        class Boom:
            def __call__(self, task_input):
                raise RuntimeError("model exploded")

        class Fine:
            def __call__(self, task_input):
                class Pred:
                    output = "ok"

                return Pred()

        baseline, evolved = _score_arms(
            [Example()], Boom(), Fine(), lambda *a, **k: 1.0, lm=None
        )

        assert baseline == []
        assert evolved == [1.0]


class TestEvolveFailsLoudly:
    """A run that cannot proceed must raise, never return an empty result."""

    def test_missing_skill_names_the_directories_searched(self, tmp_path):
        repo = tmp_path / "hermes-agent"
        (repo / "skills").mkdir(parents=True)

        with pytest.raises(EvolutionError) as exc:
            evolve(skill_name="nope", hermes_repo=str(repo), hermes_data_dir=None)

        message = str(exc.value)
        assert "not found" in message
        assert str(repo / "skills") in message

    def test_no_skills_directory_at_all_is_an_error(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(EvolutionError) as exc:
            evolve(skill_name="anything", hermes_repo=str(empty), hermes_data_dir=None)

        assert "No skills directories found" in str(exc.value)

    def test_dry_run_reports_the_derived_budget(self, skill_tree):
        result = evolve(
            skill_name="demo-skill",
            hermes_repo=str(skill_tree),
            hermes_data_dir=None,
            dry_run=True,
        )

        assert result["dry_run"] is True
        # Budget must come from the corpus, floored at the skill's own size.
        assert result["size_budget"] >= len(SKILL_TEXT)


class TestGitRoot:
    def test_finds_enclosing_repo(self, tmp_path):
        (tmp_path / ".git").mkdir()
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert _git_root(nested) == tmp_path

    def test_returns_none_outside_a_repo(self, tmp_path):
        assert _git_root(tmp_path / "nowhere") is None

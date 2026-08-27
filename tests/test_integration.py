"""End-to-end wiring tests.

The unit tests prove each piece behaves. These prove the pieces are connected
— which is precisely what was broken before: every individual module worked,
and none of them was reading the right thing.

GEPA itself needs live model calls, so the optimizer step is stubbed. What is
exercised for real is everything around it: install discovery, skill lookup
across trees, corpus budgeting, bundle loading, dataset splitting, the metric,
constraint gating, the noise-band verdict, and the written output.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import dspy
import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.core.corpus import derive_size_budget, measure_corpus
from evolution.core.fitness import FitnessScore, make_fitness_metric
from evolution.core.objectives import ObjectiveVector, select_best
from evolution.core.report import ABReport, arm_from_scores
from evolution.core.skill_bundle import load_bundle
from evolution.skills.evolve_skill import evolve
from evolution.skills.skill_module import bump_version, load_skill, reassemble_skill


BASELINE = """---
name: incident-review
description: Review production incidents and write a postmortem.
version: 1.4.2
metadata:
  hermes:
    name: decoy
---

# Incident Review

Follow [the template](references/template.md).

## Steps

1. Collect the timeline.
2. Identify the trigger.
3. Write the postmortem.
"""


@pytest.fixture
def skill_repo(tmp_path):
    """A repo-shaped skills tree with one bundled skill and some neighbours."""
    skills = tmp_path / "repo" / "skills"
    target = skills / "incident-review"
    (target / "references").mkdir(parents=True)
    (target / "SKILL.md").write_text(BASELINE)
    (target / "references" / "template.md").write_text("## Postmortem template\n\n- Impact\n- Cause\n")

    # Neighbours give the corpus a distribution to derive a budget from.
    for name, size in [("small", 800), ("medium", 4_000), ("large", 22_000)]:
        d = skills / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: {name}\n---\n\n" + "x" * size)

    return tmp_path / "repo"


class StubJudge:
    """Scores a candidate on whether it kept the structure we care about."""

    def __init__(self, good_score=0.95, bad_score=0.40):
        self.good_score = good_score
        self.bad_score = bad_score

    def score(self, task_input, expected_behavior, agent_output, skill_text):
        quality = self.good_score if "timeline" in skill_text.lower() else self.bad_score
        return FitnessScore(
            correctness=quality,
            procedure_following=quality,
            conciseness=quality,
            feedback="keep the timeline step",
        )


class TestFullChain:
    """Load -> budget -> judge -> objective -> gate -> report -> persist."""

    def test_a_tight_improvement_ships_and_is_written_out(self, skill_repo, tmp_path):
        skill_path = skill_repo / "skills" / "incident-review" / "SKILL.md"
        skill = load_skill(skill_path)
        bundle = load_bundle(skill_path, name="incident-review")

        # Frontmatter parsing must not be fooled by the nested metadata block.
        assert skill["name"] == "incident-review"
        assert skill["version"] == "1.4.2"
        assert bundle.is_bundle

        corpus = measure_corpus(skill_repo / "skills")
        budget, reason = derive_size_budget(len(skill["raw"]), corpus)
        assert budget >= len(skill["raw"])
        assert "corpus" in reason

        config = EvolutionConfig(hermes_agent_path=skill_repo)
        metric = make_fitness_metric(
            config, baseline_text=skill["body"], size_budget=budget, judge=StubJudge()
        )

        # A tightened variant that keeps the structure.
        evolved_body = skill["body"].replace(
            "1. Collect the timeline.", "1. Collect the timeline with timestamps."
        )
        evolved_full = reassemble_skill(bump_version(skill["frontmatter"]), evolved_body)

        baseline_scores, evolved_scores = [], []
        for _ in range(3):
            baseline_scores.append(
                metric(
                    dspy.Example(task_input="t", expected_behavior="e"),
                    dspy.Prediction(output="answer", skill_text=skill["body"]),
                ).score
            )
            evolved_scores.append(
                metric(
                    dspy.Example(task_input="t", expected_behavior="e"),
                    dspy.Prediction(output="answer", skill_text=evolved_body),
                ).score
            )

        validator = ConstraintValidator(config, size_budget=budget)
        results = validator.validate_all(
            evolved_full, "skill", baseline_text=skill["raw"], bundle=bundle
        )
        assert all(c.passed for c in results), [c.message for c in results if not c.passed]

        report = ABReport(
            subject="incident-review",
            baseline=arm_from_scores("baseline", baseline_scores, len(skill["raw"])),
            evolved=arm_from_scores("evolved", evolved_scores, len(evolved_full)),
            constraints_passed=True,
        )

        out = tmp_path / "out"
        md_path, json_path = report.write(out)
        assert "Verdict" in md_path.read_text()
        assert json.loads(json_path.read_text())["subject"] == "incident-review"

        # The version must have moved, or the result is indistinguishable
        # from what it replaced.
        assert "version: 1.4.3" in evolved_full

    def test_a_bloated_variant_is_stopped_by_the_objective_not_just_the_gate(self, skill_repo):
        """This is the Aug-21 failure mode: it must lose during the search."""
        skill_path = skill_repo / "skills" / "incident-review" / "SKILL.md"
        skill = load_skill(skill_path)
        config = EvolutionConfig(hermes_agent_path=skill_repo)
        budget = 4_000

        metric = make_fitness_metric(
            config, baseline_text=skill["body"], size_budget=budget, judge=StubJudge()
        )
        example = dspy.Example(task_input="t", expected_behavior="e")

        tight = metric(example, dspy.Prediction(output="a", skill_text=skill["body"])).score
        bloated = metric(
            example,
            dspy.Prediction(output="a", skill_text=skill["body"] + "\ntimeline padding" * 400),
        )

        assert bloated.score < tight
        assert "SIZE:" in bloated.feedback

    def test_a_variant_that_drops_a_reference_fails_the_gate(self, skill_repo):
        skill_path = skill_repo / "skills" / "incident-review" / "SKILL.md"
        skill = load_skill(skill_path)
        bundle = load_bundle(skill_path, name="incident-review")
        config = EvolutionConfig(hermes_agent_path=skill_repo)

        stripped = BASELINE.replace("[the template](references/template.md)", "the template")
        results = {
            c.constraint_name: c
            for c in ConstraintValidator(config, size_budget=100_000).validate_all(
                stripped, "skill", baseline_text=skill["raw"], bundle=bundle
            )
        }
        assert results["bundle_references"].passed is False

    def test_pareto_selection_prefers_the_efficient_variant(self):
        candidates = [
            ObjectiveVector(quality=0.93, size_chars=18_190, size_budget=15_000, baseline_chars=13_218),
            ObjectiveVector(quality=0.90, size_chars=13_100, size_budget=15_000, baseline_chars=13_218),
        ]
        assert select_best(candidates) == 1


class TestOrchestratorDryRun:
    """The orchestrator's own setup path, exercised for real."""

    def test_dry_run_resolves_everything_without_calling_a_model(self, skill_repo, monkeypatch):
        monkeypatch.delenv("HERMES_DATA_DIR", raising=False)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: skill_repo.parent / "fakehome"))

        result = evolve(
            skill_name="incident-review",
            hermes_repo=str(skill_repo),
            hermes_data_dir=None,
            dry_run=True,
        )

        assert result["dry_run"] is True
        assert result["size_budget"] >= len(BASELINE)

    def test_a_skill_in_the_optional_tree_is_found(self, skill_repo, monkeypatch):
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: skill_repo.parent / "fakehome"))
        optional = skill_repo / "optional-skills" / "extra-skill"
        optional.mkdir(parents=True)
        (optional / "SKILL.md").write_text("---\nname: extra-skill\ndescription: d\n---\n\nBody.\n")

        result = evolve(
            skill_name="extra-skill",
            hermes_repo=str(skill_repo),
            hermes_data_dir=None,
            dry_run=True,
        )
        assert result["dry_run"] is True


class TestCommandLineEntryPoints:
    """Every CLI must at least load and describe itself."""

    @pytest.mark.parametrize(
        "module",
        [
            "evolution.skills.evolve_skill",
            "evolution.tools.evolve_tool",
            "evolution.prompts.evolve_prompt",
            "evolution.monitor.run_rotation",
            "evolution.core.external_importers",
        ],
    )
    def test_help_works(self, module):
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "Usage:" in result.stdout

    def test_evolve_skill_documents_the_data_dir_option(self):
        result = subprocess.run(
            [sys.executable, "-m", "evolution.skills.evolve_skill", "--help"],
            capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        # The container path bug is only fixable if the option is discoverable.
        assert "--hermes-data-dir" in result.stdout
        assert "--create-pr" in result.stdout

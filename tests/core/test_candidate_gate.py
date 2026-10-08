"""Regression tests for the candidate skill pytest gate."""

from pathlib import Path

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator


def _hermes_checkout(tmp_path):
    repo = tmp_path / "hermes-agent"
    skill = repo / "skills" / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("BASELINE")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_skill_content.py").write_text(
        "from pathlib import Path\n"
        "def test_candidate_skill_is_active():\n"
        "    assert 'EVOLVED' in Path('skills/demo/SKILL.md').read_text()\n"
    )
    return repo, skill


def test_candidate_is_tested_in_isolated_checkout(tmp_path):
    repo, skill = _hermes_checkout(tmp_path)
    validator = ConstraintValidator(EvolutionConfig(hermes_agent_path=repo))

    result = validator.run_test_suite(
        repo, candidate_path=skill, candidate_text="EVOLVED"
    )

    assert result.passed, result.details
    assert skill.read_text() == "BASELINE"


def test_failing_candidate_rejected_without_touching_original(tmp_path):
    repo, skill = _hermes_checkout(tmp_path)
    validator = ConstraintValidator(EvolutionConfig(hermes_agent_path=repo))

    result = validator.run_test_suite(
        repo, candidate_path=skill, candidate_text="BROKEN"
    )

    assert not result.passed
    assert skill.read_text() == "BASELINE"


def test_candidate_path_cannot_escape_checkout(tmp_path):
    repo, skill = _hermes_checkout(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("DO NOT CHANGE")
    validator = ConstraintValidator(EvolutionConfig(hermes_agent_path=repo))

    result = validator.run_test_suite(
        repo, candidate_path=outside, candidate_text="EVOLVED"
    )

    assert not result.passed
    assert outside.read_text() == "DO NOT CHANGE"
    assert skill.read_text() == "BASELINE"

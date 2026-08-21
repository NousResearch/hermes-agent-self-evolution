"""Tests for evolve_skill helpers."""

from pathlib import Path

from evolution.skills.evolve_skill import _is_successful_improvement


class TestIsSuccessfulImprovement:
    def test_requires_artifact_diff_and_positive_improvement(self):
        assert not _is_successful_improvement("same", "same", 0.1)
        assert not _is_successful_improvement("before", "after", 0.0)
        assert not _is_successful_improvement("before", "after", -0.1)
        assert _is_successful_improvement("before", "after", 0.1)


class TestBaselineValidationTarget:
    """The baseline constraint pass must see the full file, not the bare body.

    load_skill() splits frontmatter off into skill["frontmatter"], so validating
    skill["body"] can never satisfy the skill_structure constraint and reported
    a false violation on every run.
    """

    def test_evolve_validates_raw_baseline(self):
        source = Path(__file__).resolve().parents[2] / "evolution" / "skills" / "evolve_skill.py"
        text = source.read_text()
        assert 'validate_all(skill["raw"], "skill")' in text
        assert 'validate_all(skill["body"], "skill")' not in text

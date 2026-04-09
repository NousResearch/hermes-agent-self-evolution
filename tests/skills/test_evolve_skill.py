"""Tests for evolve_skill helpers."""

from evolution.skills.evolve_skill import _is_successful_improvement


class TestIsSuccessfulImprovement:
    def test_requires_artifact_diff_and_positive_improvement(self):
        assert not _is_successful_improvement("same", "same", 0.1)
        assert not _is_successful_improvement("before", "after", 0.0)
        assert not _is_successful_improvement("before", "after", -0.1)
        assert _is_successful_improvement("before", "after", 0.1)

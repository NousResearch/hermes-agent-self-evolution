"""Tests for skill evolution reporting helpers."""

from evolution.skills.evolve_skill import has_material_diff


def test_has_material_diff_false_for_identical_saved_skill():
    baseline = "---\nname: test\ndescription: test\n---\n\n# Body\nDo thing."
    assert has_material_diff(baseline, baseline) is False


def test_has_material_diff_true_for_changed_saved_skill():
    baseline = "---\nname: test\ndescription: test\n---\n\n# Body\nDo thing."
    evolved = "---\nname: test\ndescription: test\n---\n\n# Body\nDo safer thing."
    assert has_material_diff(baseline, evolved) is True

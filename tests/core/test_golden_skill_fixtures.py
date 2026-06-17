"""Regression coverage for promoted Phase 1 golden skill fixtures."""

from evolution.core.dataset_builder import GoldenDatasetLoader


REGRESSION_SKILLS = ("github-code-review", "time-rewind")


def test_promoted_regression_fixtures_are_discoverable_and_loadable():
    for skill_name in REGRESSION_SKILLS:
        fixture_path = GoldenDatasetLoader.find_regression_fixture(skill_name)

        assert fixture_path.name.endswith("-corrected")
        assert (fixture_path / "README.md").exists()

        dataset = GoldenDatasetLoader.load_regression_fixture(skill_name)
        assert len(dataset.train) == 2
        assert len(dataset.val) == 2
        assert len(dataset.holdout) == 8
        assert len(dataset.all_examples) == 12
        assert all(example.task_input for example in dataset.all_examples)
        assert all(example.expected_behavior for example in dataset.all_examples)
        assert {example.source for example in dataset.all_examples} <= {"golden", "session-derived"}


def test_regression_fixture_manifest_maps_exact_promoted_datasets():
    fixtures = GoldenDatasetLoader.available_regression_fixtures()

    assert fixtures["github-code-review"].name == "github-code-review-phase1-validation-20260525-234032-corrected"
    assert fixtures["time-rewind"].name == "time-rewind-phase1-validation-20260526-002800-corrected"

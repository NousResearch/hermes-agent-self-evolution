"""Tests for cached dataset loading in codex-batched evolution."""

import pytest

from evolution.core.cached_dataset import load_or_create_dataset
from evolution.core.eval_dataset import EvalDataset, EvalExample


class TestLoadOrCreateDataset:
    def test_load_or_create_dataset_prefers_existing_files(self, tmp_path):
        dataset_dir = tmp_path / "datasets" / "skills" / "obsidian"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "train.jsonl").write_text('{"task_input":"a","expected_behavior":"b"}\n')
        (dataset_dir / "val.jsonl").write_text('{"task_input":"a","expected_behavior":"b"}\n')
        (dataset_dir / "holdout.jsonl").write_text('{"task_input":"a","expected_behavior":"b"}\n')

        dataset, created = load_or_create_dataset(
            dataset_dir,
            creator=lambda: (_ for _ in ()).throw(AssertionError("creator should not be called")),
            allow_live_generation=False,
        )

        assert created is False
        assert isinstance(dataset, EvalDataset)
        assert len(dataset.train) == 1
        assert len(dataset.val) == 1
        assert len(dataset.holdout) == 1

    def test_load_or_create_dataset_refuses_generation_when_disabled(self, tmp_path):
        dataset_dir = tmp_path / "datasets" / "skills" / "obsidian"

        with pytest.raises(RuntimeError):
            load_or_create_dataset(
                dataset_dir,
                creator=lambda: EvalDataset(),
                allow_live_generation=False,
            )

    def test_load_or_create_dataset_creates_and_saves_when_allowed(self, tmp_path):
        dataset_dir = tmp_path / "datasets" / "skills" / "obsidian"
        generated = EvalDataset(
            train=[EvalExample(task_input="a", expected_behavior="b")],
            val=[EvalExample(task_input="c", expected_behavior="d")],
            holdout=[EvalExample(task_input="e", expected_behavior="f")],
        )

        dataset, created = load_or_create_dataset(
            dataset_dir,
            creator=lambda: generated,
            allow_live_generation=True,
        )

        assert created is True
        assert dataset is generated
        assert (dataset_dir / "train.jsonl").exists()
        assert (dataset_dir / "val.jsonl").exists()
        assert (dataset_dir / "holdout.jsonl").exists()

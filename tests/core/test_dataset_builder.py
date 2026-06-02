"""Tests for dataset construction reproducibility."""

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from evolution.core.dataset_builder import (
    EvalDataset,
    EvalExample,
    GoldenDatasetLoader,
    SyntheticDatasetBuilder,
    split_examples,
)


@pytest.fixture(autouse=True, scope="session")
def _skill_source_env(tmp_path_factory):
    """SyntheticDatasetBuilder pulls EvolutionConfig which runs skill-source
    discovery at default-factory time. Point it at a fake repo so tests
    don't pick up a real ~/.hermes or ~/.claude install.
    """
    fake_repo = tmp_path_factory.mktemp("fake_skill_repo")
    (fake_repo / "skills").mkdir()
    os.environ["SKILL_SOURCES_HERMES_REPO"] = str(fake_repo)
    yield


def _write_golden(path: Path, n: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "task_input": f"task {i}",
                "expected_behavior": f"behavior {i}",
            }) + "\n")


class TestGoldenDatasetSeed:
    """A given seed must produce a stable train/val/holdout split.

    Without seed plumbing, every run reshuffles against the global RNG and
    holdout scores aren't comparable across runs.
    """

    def test_same_seed_same_split(self, tmp_path: Path):
        golden = tmp_path / "golden.jsonl"
        _write_golden(golden)

        d1 = GoldenDatasetLoader.load(golden, seed=42)
        d2 = GoldenDatasetLoader.load(golden, seed=42)

        assert [e.task_input for e in d1.train] == [e.task_input for e in d2.train]
        assert [e.task_input for e in d1.val] == [e.task_input for e in d2.val]
        assert [e.task_input for e in d1.holdout] == [e.task_input for e in d2.holdout]

    def test_different_seed_different_split(self, tmp_path: Path):
        golden = tmp_path / "golden.jsonl"
        _write_golden(golden)

        d1 = GoldenDatasetLoader.load(golden, seed=42)
        d2 = GoldenDatasetLoader.load(golden, seed=7)

        # With 30 examples, the probability that two distinct seeds produce
        # the exact same train slice is negligible.
        assert [e.task_input for e in d1.train] != [e.task_input for e in d2.train]


def _ex(i: int) -> EvalExample:
    return EvalExample(
        task_input=f"task {i}", expected_behavior=f"behavior {i}", source="synthetic",
    )


class TestSplitExamples:
    """The single source of truth for shuffle+split across synthetic, sessiondb,
    and golden paths. Was three sites; consolidating keeps split sizes consistent
    when EvolutionConfig ratios change."""

    def test_normalizes_ratios_above_one(self):
        # 0.5 + 0.4 + 0.5 = 1.4 — must normalize, not slice 50%/40% then drop the rest.
        examples = [_ex(i) for i in range(60)]
        ds = split_examples(
            examples, seed=42, train_ratio=0.5, val_ratio=0.4, holdout_ratio=0.5,
        )
        # Floor-rounding: 60 * 0.5/1.4 = 21, 60 * 0.4/1.4 = 17, holdout = 22.
        assert len(ds.train) == 21
        assert len(ds.val) == 17
        assert len(ds.holdout) == 22
        assert len(ds.all_examples) == 60

    def test_empty_input_returns_empty_dataset(self):
        ds = split_examples([], seed=42, train_ratio=0.5, val_ratio=0.25, holdout_ratio=0.25)
        assert ds.train == []
        assert ds.val == []
        assert ds.holdout == []

    def test_deterministic_under_same_seed(self):
        examples = [_ex(i) for i in range(30)]
        d1 = split_examples(examples, seed=42, train_ratio=0.5, val_ratio=0.25, holdout_ratio=0.25)
        d2 = split_examples(examples, seed=42, train_ratio=0.5, val_ratio=0.25, holdout_ratio=0.25)
        assert [e.task_input for e in d1.train] == [e.task_input for e in d2.train]
        assert [e.task_input for e in d1.val] == [e.task_input for e in d2.val]
        assert [e.task_input for e in d1.holdout] == [e.task_input for e in d2.holdout]

    def test_different_seed_different_split(self):
        examples = [_ex(i) for i in range(30)]
        d1 = split_examples(examples, seed=42, train_ratio=0.5, val_ratio=0.25, holdout_ratio=0.25)
        d2 = split_examples(examples, seed=7, train_ratio=0.5, val_ratio=0.25, holdout_ratio=0.25)
        assert [e.task_input for e in d1.train] != [e.task_input for e in d2.train]


class TestSplitConsistencyAcrossPaths:
    """The actual contract the helper extraction establishes: synthetic,
    sessiondb, and golden produce identical split *sizes* given identical
    N + seed + ratios. Was previously inconsistent (sessiondb + golden
    were hardcoded 50/25/25; synthetic honored config).
    """

    def test_synthetic_sessiondb_golden_produce_identical_sizes(self, tmp_path: Path):
        # Same N + seed + ratios → all three paths must agree.
        n = 30
        seed = 42
        ratios = dict(train_ratio=0.5, val_ratio=0.25, holdout_ratio=0.25)

        # Direct helper call (synthetic + sessiondb both go through this).
        examples = [_ex(i) for i in range(n)]
        helper_ds = split_examples(examples, seed=seed, **ratios)

        # GoldenDatasetLoader passes the same hardcoded 50/25/25; load a
        # fixture and verify it produces the same shape.
        golden = tmp_path / "golden.jsonl"
        with golden.open("w") as f:
            for i in range(n):
                f.write(json.dumps({
                    "task_input": f"task {i}", "expected_behavior": f"b {i}",
                }) + "\n")
        golden_ds = GoldenDatasetLoader.load(golden, seed=seed)

        assert len(helper_ds.train) == len(golden_ds.train)
        assert len(helper_ds.val) == len(golden_ds.val)
        assert len(helper_ds.holdout) == len(golden_ds.holdout)


class TestSyntheticGeneratorLMConfig:
    """At eval_dataset_size>=60 the synthetic generator's JSON output runs
    past the prior 4000-token ceiling and silently truncates → JSONDecodeError
    → process exit. Lock the bumped 16000-token budget. (At the current
    default N=150 the truncation pressure is even higher.)
    """

    def test_lm_constructed_with_bumped_max_tokens(self):
        from evolution.core.config import EvolutionConfig

        config = EvolutionConfig()
        builder = SyntheticDatasetBuilder(config)

        with patch("evolution.core.dataset_builder.dspy.LM") as lm_cls:
            lm_cls.return_value = MagicMock()
            # Stub the generator + parser so we don't hit a real LM.
            builder.generator = MagicMock(return_value=MagicMock(
                test_cases='[{"task_input": "x", "expected_behavior": "y"}]',
            ))
            builder.generate(artifact_text="skill", artifact_type="skill", num_cases=60)

        lm_cls.assert_called_once()
        _, kwargs = lm_cls.call_args
        assert kwargs["max_tokens"] == 16000, (
            f"max_tokens regressed from 16000 to {kwargs['max_tokens']}; "
            "JSON truncation will reappear at eval_dataset_size>=60"
        )


class TestBuildMemoryGuidanceDataset:
    def test_uses_all_five_categories(self):
        from evolution.core.dataset_builder import build_memory_guidance_dataset

        fake_lm = MagicMock()
        fake_lm.return_value = (
            '{"task_id": "raw", "user_message": "x", '
            '"expected_tools": ["memory"], '
            '"expected_save_content": "preference rubric"}'
        )
        examples = build_memory_guidance_dataset(lm_call=fake_lm, n_per_category=2)
        # 5 categories × 2 tasks each
        assert len(examples) == 10
        categories = [
            "save-preference",
            "save-correction",
            "dont-save-task-progress",
            "dont-save-completed-work-log",
            "declarative-vs-imperative",
        ]
        invoked_prompts = [c.args[0] for c in fake_lm.call_args_list]
        for cat in categories:
            assert any(cat in p for p in invoked_prompts), f"category {cat!r} not prompted"

    def test_stamps_unique_task_ids(self):
        from evolution.core.dataset_builder import build_memory_guidance_dataset

        fake_lm = MagicMock(return_value=(
            '{"user_message": "x", "expected_tools": ["memory"]}'
        ))
        examples = build_memory_guidance_dataset(lm_call=fake_lm, n_per_category=2)
        ids = [e["task_id"] for e in examples]
        assert len(ids) == len(set(ids)), "task_ids must be unique"

    def test_skips_unparseable_rows(self):
        from evolution.core.dataset_builder import build_memory_guidance_dataset

        fake_lm = MagicMock(return_value="not json at all")
        examples = build_memory_guidance_dataset(lm_call=fake_lm, n_per_category=1)
        assert examples == []

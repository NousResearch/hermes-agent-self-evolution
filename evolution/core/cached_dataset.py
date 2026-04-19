"""Cached dataset helpers for codex-batched evolution."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from evolution.core.eval_dataset import EvalDataset


def _has_complete_dataset(path: Path) -> bool:
    return all((path / f"{split}.jsonl").exists() for split in ("train", "val", "holdout"))


def load_or_create_dataset(
    dataset_dir: Path | str,
    creator: Callable[[], EvalDataset],
    *,
    allow_live_generation: bool,
) -> tuple[EvalDataset, bool]:
    """Load a cached dataset or create it if explicitly allowed."""
    path = Path(dataset_dir)

    if _has_complete_dataset(path):
        return EvalDataset.load(path), False

    if not allow_live_generation:
        raise RuntimeError(
            f"Cached dataset missing at {path}. Live generation is disabled; "
            "set allow_live_generation to create a dataset."
        )

    dataset = creator()
    dataset.save(path)
    return dataset, True

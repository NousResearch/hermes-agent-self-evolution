"""Golden dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "holdout")


def load_golden_splits(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load train/val/holdout JSONL files from a golden dataset directory."""
    dataset_path = Path(path)
    if dataset_path.is_file():
        rows = _read_jsonl(dataset_path)
        return {"train": rows, "val": [], "holdout": []}

    splits: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        split_file = dataset_path / f"{split}.jsonl"
        splits[split] = _read_jsonl(split_file) if split_file.exists() else []
    return splits


def flatten_splits(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, examples in splits.items():
        for example in examples:
            row = dict(example)
            row["split"] = split
            rows.append(row)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("task_input") or not row.get("expected_behavior"):
                raise ValueError(f"{path}:{line_number} missing task_input or expected_behavior")
            rows.append(row)
    return rows

"""Tests for curated golden self-evolution datasets."""

import json
from pathlib import Path


REQUIRED_GOLDEN_SKILLS = {
    "plan",
    "writing-plans",
    "goal-planning",
    "kanban-orchestrator",
}


def test_required_coordination_golden_datasets_exist_and_parse():
    root = Path("datasets/golden/skills")

    for skill in REQUIRED_GOLDEN_SKILLS:
        skill_dir = root / skill
        assert skill_dir.exists(), f"missing golden dataset for {skill}"

        example_count = 0
        for split in ["train", "val", "holdout"]:
            split_path = skill_dir / f"{split}.jsonl"
            assert split_path.exists(), f"missing {split} split for {skill}"
            for line in split_path.read_text().splitlines():
                if not line.strip():
                    continue
                example_count += 1
                row = json.loads(line)
                assert row["task_input"].strip()
                assert row["expected_behavior"].strip()
                assert row.get("source") == "golden"

        assert example_count >= 3

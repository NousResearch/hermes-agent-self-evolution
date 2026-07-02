"""Tests for preserving strict rubric metadata in eval datasets."""

import json

from evolution.core.dataset_builder import EvalDataset, GoldenDatasetLoader


def test_eval_dataset_preserves_rubric_checks_in_dspy_examples(tmp_path):
    dataset_dir = tmp_path / "golden"
    dataset_dir.mkdir()
    row = {
        "id": "holdout-top-level-comment",
        "task_input": "Review PR #9 and post a top-level comment only if approved.",
        "expected_behavior": "Review first and require External Write Gate approval.",
        "difficulty": "medium",
        "category": "top_level_comment",
        "source": "golden",
        "rubric_checks": [
            {"id": "gh_pr_comment", "pattern_any": ["gh pr comment"], "weight": 1.0},
            {"id": "issues_api", "pattern_any": ["issues/\\$PR_NUMBER/comments"], "weight": 1.0},
        ],
    }
    (dataset_dir / "holdout.jsonl").write_text(json.dumps(row) + "\n")

    dataset = GoldenDatasetLoader.load(dataset_dir)
    example = dataset.to_dspy_examples("holdout")[0]

    assert example.id == "holdout-top-level-comment"
    assert example.category == "top_level_comment"
    assert example.rubric_checks == row["rubric_checks"]


def test_eval_dataset_save_roundtrips_rubric_checks(tmp_path):
    row = {
        "id": "case-1",
        "task_input": "Review staged changes only.",
        "expected_behavior": "Use git diff --staged and keep scope narrow.",
        "difficulty": "easy",
        "category": "local_review",
        "source": "golden",
        "rubric_checks": [{"id": "staged_diff", "pattern_any": ["git diff --staged"], "weight": 1.0}],
    }
    from evolution.core.dataset_builder import EvalExample

    dataset = EvalDataset(holdout=[EvalExample.from_dict(row)])
    dataset.save(tmp_path)
    saved = json.loads((tmp_path / "holdout.jsonl").read_text())

    assert saved["id"] == "case-1"
    assert saved["rubric_checks"] == row["rubric_checks"]


def test_objective_expansion_adds_rubric_focused_train_val_examples_without_holdout_leakage():
    from evolution.core.dataset_builder import EvalExample, expand_objective_examples

    train_row = EvalExample.from_dict(
        {
            "id": "train-pr-review",
            "task_input": "Review PR #42 and summarize blockers.",
            "expected_behavior": "Inspect PR metadata and diff before giving a verdict.",
            "difficulty": "medium",
            "category": "pr_review",
            "source": "golden",
            "rubric_checks": [
                {"id": "pr_view", "description": "Gathers PR metadata", "pattern_any": ["gh pr view"], "weight": 1.0},
                {"id": "checks", "description": "Checks CI", "pattern_any": ["gh pr checks"], "weight": 1.0},
            ],
        }
    )
    val_row = EvalExample.from_dict(
        {
            "id": "val-inline-comment",
            "task_input": "Leave an inline comment on PR #15.",
            "expected_behavior": "Require External Write Gate approval before posting.",
            "difficulty": "hard",
            "category": "inline_comment",
            "source": "golden",
            "rubric_checks": [
                {"id": "side_right", "description": "Defaults to RIGHT side", "pattern_any": ["side=.?RIGHT"], "weight": 1.0}
            ],
        }
    )
    holdout_row = EvalExample.from_dict(
        {
            "id": "holdout-secret-scan",
            "task_input": "Do a pre-push secret scan review.",
            "expected_behavior": "Check secrets without printing raw tokens.",
            "difficulty": "medium",
            "category": "secret_scan",
            "source": "golden",
            "rubric_checks": [
                {"id": "secret_patterns", "description": "Detects secret-like patterns", "pattern_any": ["API key"], "weight": 1.0}
            ],
        }
    )
    dataset = EvalDataset(train=[train_row], val=[val_row], holdout=[holdout_row])

    expanded, metadata = expand_objective_examples(dataset)

    assert len(expanded.train) == 3
    assert len(expanded.val) == 2
    assert expanded.holdout == [holdout_row]
    assert metadata == {
        "enabled": True,
        "source_splits": ["train", "val"],
        "original_train_examples": 1,
        "original_val_examples": 1,
        "original_holdout_examples": 1,
        "added_train_examples": 2,
        "added_val_examples": 1,
        "holdout_unchanged": True,
    }
    added_ids = {ex.id for ex in [*expanded.train[1:], *expanded.val[1:]]}
    assert "train-pr-review::objective::pr_view" in added_ids
    assert "train-pr-review::objective::checks" in added_ids
    assert "val-inline-comment::objective::side_right" in added_ids
    assert all("holdout-secret-scan" not in ex.id for ex in [*expanded.train, *expanded.val])
    assert any("Rubric focus: Gathers PR metadata" in ex.expected_behavior for ex in expanded.train)
    assert any("gh pr view" in ex.expected_behavior for ex in expanded.train)

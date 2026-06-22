"""Tests for PR builder module (no git operations)."""

import json
import pytest
from pathlib import Path
from evolution.core.pr_builder import (
    _branch_name,
    _commit_message,
    _pr_body,
    _check_repo_clean,
    create_evolution_branch,
)


def test_branch_name_format():
    name = _branch_name("github-code-review")
    assert name.startswith("evolve/github-code-review-")
    assert len(name) > len("evolve/github-code-review-")


def test_branch_name_special_chars():
    name = _branch_name("my cool_skill")
    assert " " not in name


def test_commit_message_includes_scores():
    metrics = {
        "baseline_score": 0.5,
        "evolved_score": 0.7,
        "improvement": 0.2,
        "iterations": 5,
        "eval_source": "synthetic",
        "train_examples": 10,
        "constraints_passed": True,
    }
    msg = _commit_message("test-skill", metrics)
    assert "0.5" in msg
    assert "0.7" in msg
    assert "0.200" in msg
    assert "GEPA" in msg


def test_pr_body_includes_comparison():
    metrics = {
        "baseline_score": 0.4,
        "evolved_score": 0.6,
        "improvement": 0.2,
        "iterations": 10,
        "train_examples": 15,
        "holdout_examples": 5,
        "elapsed_seconds": 120.5,
        "constraints_passed": True,
    }
    body = _pr_body("test-skill", metrics)
    assert "0.4" in body
    assert "0.6" in body


def test_check_repo_clean_nonexistent_path():
    result = _check_repo_clean(Path("/nonexistent/path"))
    assert result is False


def test_create_branch_nonexistent_repo():
    result = create_evolution_branch(
        skill_name="test",
        evolved_skill_path=Path("/nonexistent/evolved.md"),
        baseline_skill_path=Path("/nonexistent/baseline.md"),
        metrics_path=Path("/nonexistent/metrics.json"),
        hermes_repo=Path("/nonexistent"),
        dry_run=False,
    )
    assert result is None


def test_create_branch_dry_run(tmp_path):
    evolved = tmp_path / "evolved.md"
    evolved.write_text("# Evolved Skill\nContent")
    baseline = tmp_path / "baseline.md"
    baseline.write_text("# Baseline Skill\nContent")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"baseline_score": 0.5, "evolved_score": 0.7, "improvement": 0.2,
                                    "iterations": 5, "train_examples": 10, "eval_source": "synthetic",
                                    "holdout_examples": 5, "elapsed_seconds": 10, "constraints_passed": True}))

    result = create_evolution_branch(
        skill_name="test-skill",
        evolved_skill_path=evolved,
        baseline_skill_path=baseline,
        metrics_path=metrics,
        hermes_repo=tmp_path,
        dry_run=True,
    )
    assert result is not None
    assert "test-skill" in result

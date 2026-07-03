"""Tests for HSE benchmark gate backfill manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution.local_completion.benchmark_gate_backfill import (
    BLOCKED_BY_BENCHMARK_APPROVAL,
    READY_FOR_REAL_BENCHMARK_EXECUTION,
    write_benchmark_gate_backfill,
)


def _subject(subject_id: str, commit: str = "abc123") -> dict:
    return {
        "subject_id": subject_id,
        "label": f"{subject_id} benchmark subject",
        "hermes_source": {
            "repo": "/tmp/hermes-agent",
            "commit": commit,
            "commit_exists": True,
            "worktree_materialized": False,
        },
        "skills": {
            "github-code-review": {
                "path": f"/tmp/{subject_id}/github-code-review/SKILL.md",
                "sha256": "a" * 64,
                "bytes": 12345,
            }
        },
        "tool_descriptions": {
            "override_module_sha256": "b" * 64 if subject_id == "current" else None,
            "active_apply_commit": commit if subject_id == "current" else None,
        },
    }


def test_benchmark_gate_backfill_blocks_without_real_benchmark_approval(tmp_path):
    result = write_benchmark_gate_backfill(
        baseline_subject=_subject("baseline", "88d1d6206"),
        current_subject=_subject("current", "9b50c5655"),
        output_dir=tmp_path / "backfill",
        generated_at="2026-07-03T12:30:00+09:00",
        real_benchmark_execution_approved=False,
    )

    report_path = Path(result["report_path"])
    markdown_path = Path(result["markdown_path"])
    baseline_snapshot = Path(result["baseline_snapshot_path"])
    current_snapshot = Path(result["current_snapshot_path"])
    report = json.loads(report_path.read_text())

    assert report["schema_version"] == "hse-benchmark-gate-backfill-v1"
    assert report["gate_id"] == "B0"
    assert report["status"] == BLOCKED_BY_BENCHMARK_APPROVAL
    assert report["strict_plan_gate_closed"] is False
    assert report["benchmark_gate_passed"] is None
    assert report["real_benchmarks_executed"] is False
    assert report["real_benchmark_execution_approved"] is False
    assert report["current_authorized_budget_usd"] == 0
    assert report["candidate_only"] is True
    assert report["apply_ready"] is False
    assert report["github"]["queried"] is False
    assert report["github"]["push_performed"] is False
    assert report["safety_invariants"]["network_calls_performed"] is False
    assert report["safety_invariants"]["active_runtime_mutation"] is False
    assert report["benchmark_subjects"]["baseline"]["hermes_source"]["commit"] == "88d1d6206"
    assert report["benchmark_subjects"]["current"]["hermes_source"]["commit"] == "9b50c5655"
    assert "real benchmark execution approval is required" in report["blocked_reason"]
    assert baseline_snapshot.exists()
    assert current_snapshot.exists()
    assert json.loads(baseline_snapshot.read_text())["subject_id"] == "baseline"
    assert json.loads(current_snapshot.read_text())["subject_id"] == "current"
    markdown = markdown_path.read_text()
    assert "BLOCKED_BY_BENCHMARK_APPROVAL" in markdown
    assert "real_benchmarks_executed=false" in markdown
    assert "NO_GITHUB_WRITE" in markdown


def test_benchmark_gate_backfill_rejects_subject_without_commit(tmp_path):
    bad_subject = _subject("baseline")
    bad_subject["hermes_source"].pop("commit")

    with pytest.raises(ValueError, match="baseline_subject.hermes_source.commit must be non-empty"):
        write_benchmark_gate_backfill(
            baseline_subject=bad_subject,
            current_subject=_subject("current", "9b50c5655"),
            output_dir=tmp_path / "backfill",
            generated_at="2026-07-03T12:30:00+09:00",
        )


def test_benchmark_gate_backfill_marks_approved_subject_ready_but_not_executed(tmp_path):
    result = write_benchmark_gate_backfill(
        baseline_subject=_subject("baseline", "88d1d6206"),
        current_subject=_subject("current", "9b50c5655"),
        output_dir=tmp_path / "backfill-approved",
        generated_at="2026-07-03T12:30:00+09:00",
        real_benchmark_execution_approved=True,
        approved_budget_usd=50,
        approved_runtime_minutes=180,
    )

    report = json.loads(Path(result["report_path"]).read_text())
    assert report["status"] == READY_FOR_REAL_BENCHMARK_EXECUTION
    assert report["strict_plan_gate_closed"] is False
    assert report["benchmark_gate_passed"] is None
    assert report["real_benchmark_execution_approved"] is True
    assert report["real_benchmarks_executed"] is False
    assert report["current_authorized_budget_usd"] == 50
    assert report["approved_runtime_minutes"] == 180
    assert report["required_next_action"] == "run_real_benchmarks_under_recorded_budget"

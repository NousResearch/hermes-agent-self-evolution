"""Tests for one-command loop execution."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from evolution.cli import main
from evolution.db.store import EvolutionStore
from evolution.orchestrator.loop import run_loop_once


def _setup_repo_with_target(tmp_path):
    root = tmp_path / ".evolution"
    repo_path = tmp_path / "hermes-agent"
    skill_dir = repo_path / "skills" / "testing" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: Test skill\n---\n\n# Test Skill\n\nFollow the existing procedure.\n"
    )
    runner = CliRunner()
    assert runner.invoke(main, ["--root", str(root), "init"]).exit_code == 0
    assert runner.invoke(main, ["--root", str(root), "repo", "add", "hermes-agent", "--path", str(repo_path)]).exit_code == 0
    assert runner.invoke(main, ["--root", str(root), "targets", "scan", "--repo", "hermes-agent"]).exit_code == 0
    return root, repo_path, runner


def _trace_file(tmp_path):
    path = tmp_path / "failures.jsonl"
    rows = [
        {"task_input": "task train", "expected_behavior": "mention train-only calibration", "status": "failure", "failure_reason": "missed train"},
        {"task_input": "task val", "expected_behavior": "mention validation-only rubric", "status": "failure", "failure_reason": "missed val"},
        {"task_input": "task holdout", "expected_behavior": "mention holdout-only guarded outcome", "status": "failure", "failure_reason": "missed holdout"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_run_loop_once_builds_dataset_executes_gates_and_exports(tmp_path):
    root, _repo_path, _runner = _setup_repo_with_target(tmp_path)
    store = EvolutionStore(root / "evolution.db")
    store.init_schema()
    trace_path = _trace_file(tmp_path)

    result = run_loop_once(
        store=store,
        root=root,
        target_ref="skill:test-skill",
        trace_path=trace_path,
        trace_source="unit-test",
        strategy="deterministic",
        iterations=2,
    )

    assert result["imported_traces"] == 3
    assert result["dataset"]["example_count"] == 3
    assert result["run"]["status"] == "completed"
    assert result["gate"]["decision"] in {"pass", "hold"}
    assert Path(result["export"]["bundle_dir"]).exists()
    assert (Path(result["export"]["bundle_dir"]) / "manifest.json").exists()
    assert store.list_attempt_traces(status="failure")
    assert store.list_evaluations(result["run"]["id"])


def test_cli_loop_once_runs_the_product_flow(tmp_path):
    root, _repo_path, runner = _setup_repo_with_target(tmp_path)
    trace_path = _trace_file(tmp_path)

    result = runner.invoke(
        main,
        [
            "--root", str(root),
            "loop", "once",
            "--target", "skill:test-skill",
            "--trace-path", str(trace_path),
            "--trace-source", "unit-test",
            "--strategy", "deterministic",
            "--iterations", "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Loop once completed" in result.output
    assert "run=run_" in result.output
    assert "dataset=dataset_" in result.output
    assert "bundle_dir=" in result.output

"""Tests for run gate CLI."""

from evolution.cli import main
from tests.core.test_run_cli import _setup_target_and_dataset


def test_cli_run_gate_persists_decision_for_executed_run(tmp_path):
    runner, root, dataset_id = _setup_target_and_dataset(tmp_path)
    create_result = runner.invoke(
        main,
        [
            "--root", str(root),
            "run", "skill",
            "--target", "skill:test-skill",
            "--dataset", dataset_id,
            "--iterations", "2",
        ],
    )
    assert create_result.exit_code == 0, create_result.output
    run_id = next(part for part in create_result.output.split() if part.startswith("run_"))

    execute_result = runner.invoke(main, ["--root", str(root), "run", "execute", run_id])
    assert execute_result.exit_code == 0, execute_result.output

    gate_result = runner.invoke(main, ["--root", str(root), "run", "gate", run_id])

    assert gate_result.exit_code == 0, gate_result.output
    assert f"Gate run {run_id}" in gate_result.output
    assert "decision=" in gate_result.output
    assert "candidate=candidate_" in gate_result.output

    show_result = runner.invoke(main, ["--root", str(root), "runs", "show", run_id])
    assert show_result.exit_code == 0, show_result.output
    assert "status: completed" in show_result.output

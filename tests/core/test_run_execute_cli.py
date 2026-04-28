"""Tests for run execution CLI."""

from evolution.db.store import EvolutionStore
from tests.core.test_run_cli import _setup_target_and_dataset
from evolution.cli import main


def test_cli_run_execute_completes_registered_run(tmp_path):
    runner, root, dataset_id = _setup_target_and_dataset(tmp_path)
    create_result = runner.invoke(
        main,
        [
            "--root", str(root),
            "run", "skill",
            "--target", "skill:test-skill",
            "--dataset", dataset_id,
            "--engine", "gepa",
            "--iterations", "2",
        ],
    )
    assert create_result.exit_code == 0, create_result.output
    run_id = next(part for part in create_result.output.split() if part.startswith("run_"))

    execute_result = runner.invoke(main, ["--root", str(root), "run", "execute", run_id])

    assert execute_result.exit_code == 0, execute_result.output
    assert f"Executed run {run_id}" in execute_result.output
    assert "status=completed" in execute_result.output
    assert "candidates=2" in execute_result.output
    assert "evaluations=6" in execute_result.output

    show_result = runner.invoke(main, ["--root", str(root), "runs", "show", run_id])
    assert show_result.exit_code == 0, show_result.output
    assert "status: completed" in show_result.output

    store = EvolutionStore(root / "evolution.db")
    assert len(store.list_candidates(run_id)) == 2
    assert len(store.list_evaluations(run_id)) == 6

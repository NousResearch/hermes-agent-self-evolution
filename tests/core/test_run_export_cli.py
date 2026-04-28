"""Tests for review-bundle export CLI."""

from pathlib import Path

from evolution.cli import main
from tests.core.test_run_cli import _setup_target_and_dataset


def test_cli_run_export_writes_bundle_after_gate(tmp_path):
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
    assert runner.invoke(main, ["--root", str(root), "run", "execute", run_id]).exit_code == 0
    gate_result = runner.invoke(main, ["--root", str(root), "run", "gate", run_id])
    assert gate_result.exit_code == 0, gate_result.output

    out_dir = tmp_path / "review-bundles"
    export_result = runner.invoke(
        main,
        ["--root", str(root), "run", "export", run_id, "--out", str(out_dir), "--allow-hold"],
    )

    assert export_result.exit_code == 0, export_result.output
    assert f"Exported review bundle for run {run_id}" in export_result.output
    assert "manifest=artifact_" in export_result.output
    bundle_path = Path(next(line.split("=", 1)[1] for line in export_result.output.splitlines() if line.startswith("bundle_dir=")))
    assert bundle_path.exists()
    assert bundle_path.parent == out_dir
    assert (bundle_path / "candidate.diff").exists()
    assert (bundle_path / "APPLY.md").exists()

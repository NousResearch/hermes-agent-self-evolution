"""Tests for the unified HSE real benchmark runner."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from evolution.benchmarks.real_benchmark_runner import main as runner_main
from evolution.benchmarks.real_benchmark_runner import run_real_benchmark_suite, suite_readiness

REPO_ROOT = Path(__file__).resolve().parents[2]
HSE_OUTPUT_ROOT = REPO_ROOT / "output" / "hse-real-benchmark" / "pytest"
PHASE3_FIXTURE_ROOT = REPO_ROOT / "datasets" / "golden" / "benchmarks" / "phase3-system-prompt"
BASELINE_PROMPT = PHASE3_FIXTURE_ROOT / "baseline_system_prompt.json"
CANDIDATE_PROMPT = PHASE3_FIXTURE_ROOT / "candidate_system_prompt.json"
TBLITE_CASES = PHASE3_FIXTURE_ROOT / "tblite_cases.jsonl"
YC_BENCH_CASES = PHASE3_FIXTURE_ROOT / "yc_bench_fast_test.jsonl"
TOOL_SELECTION_DATASET = REPO_ROOT / "datasets" / "golden" / "tool-description" / "tool_selection.jsonl"
TOOL_SELECTION_REPORT = (
    REPO_ROOT
    / "output"
    / "tool-description"
    / "phase5-tool-selection-all-pass-20260606-145446"
    / "run"
    / "candidate_only_report.json"
)


def _tblite_root(tmp_path: Path) -> Path:
    root = tmp_path / "terminal-bench-lite"
    task_dir = root / "sample-task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "task.toml").write_text("name = 'sample-task'\n")
    (task_dir / "instruction.md").write_text("Do a safe local task.\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/sh\nexit 0\n")
    return root


def _yc_root(tmp_path: Path) -> Path:
    root = tmp_path / "yc-bench"
    preset_dir = root / "src" / "yc_bench" / "config" / "presets"
    preset_dir.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'yc-bench'\n")
    (root / "README.md").write_text("# YC-Bench\n")
    (root / "src" / "yc_bench" / "__init__.py").write_text("")
    (preset_dir / "default.toml").write_text("name = 'default'\n")
    return root


def _output(tmp_path: Path, filename: str) -> Path:
    return HSE_OUTPUT_ROOT / tmp_path.name / filename


def _cleanup(output_json: Path) -> None:
    if output_json.is_relative_to(HSE_OUTPUT_ROOT):
        shutil.rmtree(HSE_OUTPUT_ROOT / output_json.relative_to(HSE_OUTPUT_ROOT).parts[0], ignore_errors=True)


def test_real_benchmark_runner_dry_run_validates_without_writing_output(tmp_path: Path) -> None:
    output_json = _output(tmp_path, "tblite-dry-run.json")

    rc = runner_main(
        [
            "--suite",
            "TBLite",
            "--baseline-commit",
            "88d1d6206",
            "--current-commit",
            "9b50c5655",
            "--output-json",
            str(output_json),
            "--dry-run",
            "--tblite-root",
            str(_tblite_root(tmp_path)),
        ]
    )

    assert rc == 0
    assert not output_json.exists()
    assert not output_json.parent.exists()


def test_real_benchmark_runner_writes_tblite_and_yc_reports_under_hse_root(tmp_path: Path) -> None:
    outputs = [_output(tmp_path, "tblite.json"), _output(tmp_path, "yc-bench.json")]
    try:
        tblite = run_real_benchmark_suite(
            suite="TBLite",
            baseline_commit="88d1d6206",
            current_commit="9b50c5655",
            output_json=outputs[0],
            tblite_root=_tblite_root(tmp_path),
            baseline_prompt=BASELINE_PROMPT,
            candidate_prompt=CANDIDATE_PROMPT,
            tblite_cases=TBLITE_CASES,
            task_limit=1,
        )
        yc = run_real_benchmark_suite(
            suite="YC-Bench",
            baseline_commit="88d1d6206",
            current_commit="9b50c5655",
            output_json=outputs[1],
            yc_bench_root=_yc_root(tmp_path),
            baseline_prompt=BASELINE_PROMPT,
            candidate_prompt=CANDIDATE_PROMPT,
            yc_bench_cases=YC_BENCH_CASES,
        )

        for report, output_json, benchmark in ((tblite, outputs[0], "TBLite"), (yc, outputs[1], "YC-Bench")):
            assert output_json.exists()
            assert report["benchmark"] == benchmark
            assert report["mode"] == "real-benchmark-smoke"
            assert report["dry_run"] is False
            assert report["external_calls_performed"] is False
            assert report["provider_or_model_spend_performed"] is False
            assert report["network_calls_performed"] is False
            assert report["github_write_performed"] is False
            assert report["full_benchmark_executed"] is False
            assert report["real_benchmark_smoke_validated"] is True
            assert report["passed"] is True
            assert report["failed_checks"] == []
            assert report["output_constraints"]["allowed_root"] == "output/hse-real-benchmark/"
            assert Path(report["artifacts"]["output_json"]) == output_json
    finally:
        _cleanup(outputs[0])


def test_real_benchmark_runner_writes_phase2_tool_selection_report_under_hse_root(tmp_path: Path) -> None:
    output_json = _output(tmp_path, "phase2-plan-scale-tool-selection-triples.json")
    try:
        report = run_real_benchmark_suite(
            suite="Phase2 PLAN-scale tool-selection triples",
            baseline_commit="88d1d6206",
            current_commit="9b50c5655",
            output_json=output_json,
            tool_selection_dataset=TOOL_SELECTION_DATASET,
            tool_selection_report=TOOL_SELECTION_REPORT,
        )

        assert output_json.exists()
        assert report["benchmark"] == "Phase2 PLAN-scale tool-selection triples"
        assert report["mode"] == "real-benchmark-smoke"
        assert report["dry_run"] is False
        assert report["external_calls_performed"] is False
        assert report["passed"] is True
        assert report["failed_checks"] == []
        assert report["metrics"]["case_count"] >= 45
        assert report["metrics"]["selection_accuracy"] == 1.0
        assert report["metrics"]["wrong_tool_avoidance"] == 1.0
        assert report["phase2d_gate"]["passed"] is True
        assert report["real_benchmark_evidence"]["dataset_row_count"] >= 45
    finally:
        _cleanup(output_json)


def test_real_benchmark_runner_rejects_output_outside_hse_root(tmp_path: Path) -> None:
    output_json = tmp_path / "outside.json"

    try:
        run_real_benchmark_suite(
            suite="TBLite",
            baseline_commit="88d1d6206",
            current_commit="9b50c5655",
            output_json=output_json,
            tblite_root=_tblite_root(tmp_path),
            baseline_prompt=BASELINE_PROMPT,
            candidate_prompt=CANDIDATE_PROMPT,
            tblite_cases=TBLITE_CASES,
            task_limit=1,
        )
    except ValueError as exc:
        assert "output-json must stay under output/hse-real-benchmark/" in str(exc)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("outside output path was accepted")
    assert not output_json.exists()


def test_real_benchmark_runner_default_suite_readiness_uses_existing_local_assets() -> None:
    for suite in ("TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples"):
        readiness = suite_readiness(suite)
        assert readiness["ready"] is True
        assert readiness["blocked_by"] == []
        assert readiness["network_calls_required"] is False
        assert readiness["provider_or_model_spend_required"] is False

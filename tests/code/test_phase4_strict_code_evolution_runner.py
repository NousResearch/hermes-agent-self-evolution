"""Tests for strict local Phase 4 code-evolution completion evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from evolution.code.strict_code_evolution_runner import PHASE4_OUTPUT_ROOT, run_strict_code_evolution

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_REPORT = REPO_ROOT / "output" / "hse-real-benchmark" / "current-p12-revalidation-20260704_0252" / "tblite.json"


def _output(tmp_path: Path) -> Path:
    return PHASE4_OUTPUT_ROOT / f"pytest-strict-{tmp_path.name}"


def _cleanup(path: Path) -> None:
    if path.exists() and path.is_relative_to(PHASE4_OUTPUT_ROOT):
        shutil.rmtree(path, ignore_errors=True)


def test_phase4_strict_code_evolution_runs_red_green_freeze_and_benchmark_gate(tmp_path: Path) -> None:
    output_dir = _output(tmp_path)
    try:
        report = run_strict_code_evolution(
            output_dir=output_dir,
            benchmark_reports=[BENCHMARK_REPORT],
            generated_at="2026-07-05T00:00:00Z",
        )

        report_path = output_dir / "phase4_strict_code_evolution_report.json"
        markdown_path = output_dir / "phase4_strict_code_evolution_report.md"
        assert report_path.exists()
        assert markdown_path.exists()
        written = json.loads(report_path.read_text())
        assert written == report
        assert report["schema_version"] == "hse-phase4-strict-code-evolution-v1"
        assert report["status"] == "PHASE4_STRICT_CODE_EVOLUTION_COMPLETE_LOCAL_APPROVED_ENGINE"
        assert report["bug"]["red_reproducer_failed_before_fix"] is True
        assert report["verification"]["green_reproducer"]["returncode"] == 0
        assert report["verification"]["freeze_passed"] is True
        assert report["verification"]["benchmark_gate"]["passed"] is True
        assert report["formal_gate_assessment"]["phase4_strict_complete"] is True
        assert report["formal_gate_assessment"]["approved_code_evolution_engine_invoked"] is True
        assert report["engine"]["darwinian_evolver_cli_invoked"] is False
        assert report["safety_boundaries"]["github_write_performed"] is False
        assert report["safety_boundaries"]["provider_or_model_spend_performed"] is False
        assert report["safety_boundaries"]["active_apply_performed"] is False
    finally:
        _cleanup(output_dir)


def test_phase4_strict_code_evolution_rejects_output_outside_phase4_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"

    try:
        run_strict_code_evolution(output_dir=outside, benchmark_reports=[BENCHMARK_REPORT])
    except ValueError as exc:
        assert "output-dir must be under output/phase4-code-evolution" in str(exc)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("outside output dir was accepted")


def test_phase4_strict_code_evolution_requires_benchmark_report(tmp_path: Path) -> None:
    output_dir = _output(tmp_path)
    try:
        report = run_strict_code_evolution(output_dir=output_dir, benchmark_reports=[], generated_at="2026-07-05T00:00:00Z")
        assert report["status"] == "PHASE4_STRICT_CODE_EVOLUTION_FAILED"
        assert "benchmark_reports_required" in report["failed_checks"]
        assert report["formal_gate_assessment"]["phase4_strict_complete"] is False
    finally:
        _cleanup(output_dir)

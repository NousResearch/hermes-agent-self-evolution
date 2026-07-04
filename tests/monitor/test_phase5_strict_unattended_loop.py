"""Tests for strict local Phase 5 unattended loop evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from evolution.monitor.strict_unattended_loop import PHASE5_OUTPUT_ROOT, run_strict_unattended_loop


def _metrics(path: Path) -> Path:
    payload = {
        "schema_version": "phase5-performance-input-v1",
        "window": {"start": "2026-07-01", "end": "2026-07-05"},
        "source": {"kind": "sanitized_local_fixture", "label": "pytest-strict-phase5"},
        "metrics": [
            {
                "id": "tool_selection_accuracy",
                "component": "tool_descriptions",
                "value": 0.71,
                "threshold": 0.9,
                "baseline": 0.82,
                "higher_is_better": True,
                "sample_count": 20,
            },
            {
                "id": "prompt_contract_warning_rate",
                "component": "system_prompts",
                "value": 0.01,
                "threshold": 0.05,
                "baseline": 0.02,
                "higher_is_better": False,
                "sample_count": 10,
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _output(tmp_path: Path) -> Path:
    return PHASE5_OUTPUT_ROOT / f"pytest-strict-{tmp_path.name}"


def _cleanup(path: Path, runs_root: Path) -> None:
    if path.exists() and path.is_relative_to(PHASE5_OUTPUT_ROOT):
        shutil.rmtree(path, ignore_errors=True)
    shutil.rmtree(runs_root, ignore_errors=True)


def test_phase5_strict_unattended_loop_detects_optimizes_and_writes_pr_ready_packet(tmp_path: Path) -> None:
    output_dir = _output(tmp_path)
    runs_root = tmp_path / "runs"
    metrics_json = _metrics(tmp_path / "metrics.json")
    try:
        report = run_strict_unattended_loop(
            metrics_json=metrics_json,
            output_dir=output_dir,
            runs_root=runs_root,
            generated_at="2026-07-05T00:00:00Z",
        )

        report_path = output_dir / "phase5_strict_unattended_loop_report.json"
        markdown_path = output_dir / "phase5_strict_unattended_loop_report.md"
        pr_ready = output_dir / "pr_ready_handoff.json"
        assert report_path.exists()
        assert markdown_path.exists()
        assert pr_ready.exists()
        assert json.loads(report_path.read_text()) == report
        assert report["schema_version"] == "hse-phase5-strict-unattended-loop-v1"
        assert report["status"] == "PHASE5_STRICT_UNATTENDED_LOOP_PASS_LOCAL_PR_READY"
        assert report["detection"]["auto_triage_status"] == "REVIEW_REQUIRED"
        assert report["optimizer"]["auto_optimizer_triggered"] is True
        assert report["optimizer"]["optimizer_execution_completed"] is True
        assert report["candidate_bundle"]["created"] is True
        assert report["pr_ready_handoff"]["status"] == "LOCAL_PR_READY_HANDOFF_CREATED_GITHUB_WRITE_DEFERRED"
        assert report["formal_gate_assessment"]["phase5_strict_complete"] is True
        assert report["formal_gate_assessment"]["unattended_detect_to_optimize_to_pr_ready_completed"] is True
        assert report["safety_boundaries"]["github_write_performed"] is False
        assert report["safety_boundaries"]["provider_or_model_spend_performed"] is False
        assert report["safety_boundaries"]["active_apply_performed"] is False
        assert report["human_merge_boundary"]["auto_merge_performed"] is False
        assert list(runs_root.glob("*/decision.json"))
    finally:
        _cleanup(output_dir, runs_root)


def test_phase5_strict_unattended_loop_rejects_output_outside_phase5_root(tmp_path: Path) -> None:
    metrics_json = _metrics(tmp_path / "metrics.json")

    try:
        run_strict_unattended_loop(metrics_json=metrics_json, output_dir=tmp_path / "outside", runs_root=tmp_path / "runs")
    except ValueError as exc:
        assert "output-dir must be under output/phase5-continuous-loop" in str(exc)
    else:  # pragma: no cover - explicit failure branch
        raise AssertionError("outside output dir was accepted")


def test_phase5_strict_unattended_loop_requires_detected_target(tmp_path: Path) -> None:
    output_dir = _output(tmp_path)
    runs_root = tmp_path / "runs"
    metrics_json = tmp_path / "passing-metrics.json"
    payload = {
        "schema_version": "phase5-performance-input-v1",
        "window": {"start": "2026-07-01", "end": "2026-07-05"},
        "source": {"kind": "sanitized_local_fixture", "label": "pytest-strict-phase5"},
        "metrics": [
            {
                "id": "tool_selection_accuracy",
                "component": "tool_descriptions",
                "value": 0.95,
                "threshold": 0.9,
                "baseline": 0.9,
                "higher_is_better": True,
                "sample_count": 20,
            }
        ],
    }
    metrics_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        try:
            run_strict_unattended_loop(metrics_json=metrics_json, output_dir=output_dir, runs_root=runs_root)
        except ValueError as exc:
            assert "requires at least one detected ranked target" in str(exc)
        else:  # pragma: no cover - explicit failure branch
            raise AssertionError("passing metrics without target were accepted")
    finally:
        _cleanup(output_dir, runs_root)

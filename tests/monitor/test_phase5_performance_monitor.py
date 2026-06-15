"""Tests for the Phase 5 read-only performance monitor snapshot."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output" / "phase5-continuous-loop"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "phase2-tool-description-gate.yml"
README_MD = REPO_ROOT / "README.md"
PLAN_MD = REPO_ROOT / "PLAN.md"
CONTRACT_MD = REPO_ROOT / "reports" / "phase5_performance_monitor_report_contract.md"


def _sample_metrics() -> dict:
    return {
        "schema_version": "phase5-performance-input-v1",
        "window": {"start": "2026-06-01", "end": "2026-06-05"},
        "source": {"kind": "sanitized_local_fixture", "label": "pytest-monitor-fixture"},
        "metrics": [
            {
                "id": "tool_selection_accuracy",
                "component": "tool_descriptions",
                "value": 0.91,
                "threshold": 0.90,
                "baseline": 0.88,
                "higher_is_better": True,
                "sample_count": 45,
            },
            {
                "id": "phase4_scaffold_pass_rate",
                "component": "code_evolution",
                "value": 1.0,
                "threshold": 1.0,
                "baseline": 1.0,
                "higher_is_better": True,
                "sample_count": 71,
            },
            {
                "id": "user_correction_rate",
                "component": "assistant_behavior",
                "value": 0.18,
                "threshold": 0.10,
                "baseline": 0.12,
                "higher_is_better": False,
                "sample_count": 12,
            },
        ],
    }


def _assert_privacy_safe(value: object) -> None:
    text = json.dumps(value, sort_keys=True)
    forbidden_fragments = [
        "/" + "Users" + "/",
        "/" + "home" + "/",
        "session" + "_id",
        "OPENAI" + "_API_KEY",
        "ANTHROPIC" + "_API_KEY",
        "OPENROUTER" + "_API_KEY",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in text, fragment


def test_build_performance_snapshot_report_is_read_only_and_privacy_safe():
    from evolution.monitor.performance_snapshot import build_performance_snapshot_report

    report = build_performance_snapshot_report(
        _sample_metrics(),
        generated_at="2026-06-05T00:00:00Z",
    )

    assert report["phase"] == "5"
    assert report["mode"] == "phase5-readonly-performance-monitor-snapshot"
    assert report["schema_version"] == "phase5-performance-snapshot-v1"
    assert report["status"] == "NEEDS_TRIAGE"
    assert report["generated_at"] == "2026-06-05T00:00:00Z"

    safety = report["safety_invariants"]
    assert safety == {
        "read_only": True,
        "raw_private_session_data_committed": False,
        "raw_credentials_recorded": False,
        "active_runtime_mutation": False,
        "external_calls_performed": False,
        "network_calls_performed": False,
        "cron_jobs_created": False,
        "optimizer_execution_started": False,
        "automated_pr_created_or_updated": False,
    }
    assert report["input_contract"] == {
        "sanitized_input_required": True,
        "raw_session_data_allowed": False,
        "private_paths_allowed": False,
        "network_sources_allowed": False,
        "credentials_allowed": False,
    }

    assert report["summary"] == {
        "metric_count": 3,
        "component_count": 3,
        "failing_metric_count": 1,
        "regressing_metric_count": 1,
        "weak_area_count": 1,
    }
    metrics = {metric["id"]: metric for metric in report["metrics"]}
    assert metrics["tool_selection_accuracy"]["status"] == "PASS"
    assert metrics["phase4_scaffold_pass_rate"]["status"] == "PASS"
    assert metrics["user_correction_rate"]["status"] == "FAIL"
    assert metrics["user_correction_rate"]["regressed_vs_baseline"] is True
    assert report["weak_areas"] == [
        {
            "metric_id": "user_correction_rate",
            "component": "assistant_behavior",
            "status": "FAIL",
            "regressed_vs_baseline": True,
            "severity": 0.08,
            "recommendation": "manual_triage_required_no_optimizer_started",
        }
    ]
    assert report["recommended_next_step"] == "manual_triage_required_no_optimizer_started"
    _assert_privacy_safe(report)


def test_performance_snapshot_rejects_private_or_raw_identifiers():
    from evolution.monitor.performance_snapshot import build_performance_snapshot_report

    metrics = _sample_metrics()
    metrics["source"]["label"] = "/" + "Users" + "/" + "example" + "/raw"

    with pytest.raises(ValueError, match="private/raw identifier"):
        build_performance_snapshot_report(metrics, generated_at="2026-06-05T00:00:00Z")

    metrics = _sample_metrics()
    metrics["metrics"][0]["note"] = "contains " + "session" + "_id"

    with pytest.raises(ValueError, match="private/raw identifier"):
        build_performance_snapshot_report(metrics, generated_at="2026-06-05T00:00:00Z")


def test_cli_writes_json_and_markdown_under_phase5_output_root(tmp_path):
    metrics_path = tmp_path / "sanitized_metrics.json"
    metrics_path.write_text(json.dumps(_sample_metrics(), indent=2, sort_keys=True) + "\n")
    output_dir = OUTPUT_ROOT / "pytest-performance-monitor-cli"
    shutil.rmtree(output_dir, ignore_errors=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolution.monitor.performance_snapshot",
            "--metrics-json",
            str(metrics_path),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-06-05T00:00:00Z",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report_path = output_dir / "performance_snapshot_report.json"
    markdown_path = output_dir / "performance_snapshot_report.md"
    assert report_path.exists()
    assert markdown_path.exists()
    report = json.loads(report_path.read_text())
    assert report["artifacts"] == {
        "report_json": "performance_snapshot_report.json",
        "report_markdown": "performance_snapshot_report.md",
    }
    assert report["status"] == "NEEDS_TRIAGE"
    assert report["safety_invariants"]["cron_jobs_created"] is False
    assert report["safety_invariants"]["optimizer_execution_started"] is False
    markdown = markdown_path.read_text()
    assert "# Phase 5 Performance Monitor Snapshot" in markdown
    assert "Status: NEEDS_TRIAGE" in markdown
    assert "manual_triage_required_no_optimizer_started" in markdown
    _assert_privacy_safe(report)
    _assert_privacy_safe(markdown)


def test_cli_allows_existing_empty_output_dir(tmp_path):
    metrics_path = tmp_path / "sanitized_metrics.json"
    metrics_path.write_text(json.dumps(_sample_metrics(), indent=2, sort_keys=True) + "\n")
    output_dir = OUTPUT_ROOT / "pytest-performance-monitor-existing-empty"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolution.monitor.performance_snapshot",
            "--metrics-json",
            str(metrics_path),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-06-05T00:00:00Z",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert (output_dir / "performance_snapshot_report.json").exists()
    assert (output_dir / "performance_snapshot_report.md").exists()


def test_cli_rejects_output_dir_outside_phase5_output_root(tmp_path):
    metrics_path = tmp_path / "sanitized_metrics.json"
    metrics_path.write_text(json.dumps(_sample_metrics(), indent=2, sort_keys=True) + "\n")
    output_dir = tmp_path / "outside-phase5-root"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolution.monitor.performance_snapshot",
            "--metrics-json",
            str(metrics_path),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-06-05T00:00:00Z",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "output-dir must be under output/phase5-continuous-loop" in result.stderr
    assert not (output_dir / "performance_snapshot_report.json").exists()


def test_phase5_performance_monitor_contract_is_documented_and_in_ci():
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    triggers = workflow.get("on", workflow.get(True))
    run_blocks = "\n".join(
        str(step.get("run", ""))
        for step in workflow["jobs"]["phase2-tool-description-gate"]["steps"]
        if isinstance(step, dict)
    )

    assert "tests/monitor/test_phase5_performance_monitor.py" in run_blocks
    for trigger_name in ("pull_request", "push"):
        assert "evolution/monitor/**" in triggers[trigger_name]["paths"]
        assert "tests/monitor/**" in triggers[trigger_name]["paths"]
        assert "reports/phase5_*" in triggers[trigger_name]["paths"]

    contract = CONTRACT_MD.read_text()
    assert "# Phase 5 Performance Monitor Report Contract" in contract
    assert "phase5-readonly-performance-monitor-snapshot" in contract
    assert "cron_jobs_created=false" in contract
    assert "optimizer_execution_started=false" in contract
    assert "No raw session data, local private paths, or credentials" in contract

    readme = README_MD.read_text()
    plan = PLAN_MD.read_text()
    assert "Phase 5 performance monitor report contract" in readme
    assert "evolution.monitor.performance_snapshot" in readme
    assert "read-only performance monitor snapshot" in plan
    assert "manual triage; no optimizer is started" in plan

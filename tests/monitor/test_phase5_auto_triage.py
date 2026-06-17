"""Tests for the Phase 5 read-only auto-triage ranking report."""

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
CONTRACT_MD = REPO_ROOT / "reports" / "phase5_auto_triage_report_contract.md"


def _sample_performance_report() -> dict:
    return {
        "schema_version": "phase5-performance-snapshot-v1",
        "phase": "5",
        "mode": "phase5-readonly-performance-monitor-snapshot",
        "status": "NEEDS_TRIAGE",
        "generated_at": "2026-06-05T00:00:00Z",
        "window": {"start": "2026-06-01", "end": "2026-06-05"},
        "source": {"kind": "sanitized_local_fixture", "label": "pytest-triage-fixture"},
        "input_contract": {
            "sanitized_input_required": True,
            "raw_session_data_allowed": False,
            "private_paths_allowed": False,
            "network_sources_allowed": False,
            "credentials_allowed": False,
        },
        "safety_invariants": {
            "read_only": True,
            "raw_private_session_data_committed": False,
            "raw_credentials_recorded": False,
            "active_runtime_mutation": False,
            "external_calls_performed": False,
            "network_calls_performed": False,
            "cron_jobs_created": False,
            "optimizer_execution_started": False,
            "automated_pr_created_or_updated": False,
        },
        "summary": {
            "metric_count": 4,
            "component_count": 4,
            "failing_metric_count": 3,
            "regressing_metric_count": 3,
            "weak_area_count": 3,
        },
        "metrics": [
            {
                "id": "tool_selection_accuracy",
                "component": "tool_descriptions",
                "value": 0.86,
                "threshold": 0.90,
                "baseline": 0.88,
                "higher_is_better": True,
                "sample_count": 90,
                "status": "FAIL",
                "regressed_vs_baseline": True,
                "severity": 0.04,
            },
            {
                "id": "skill_loading_failure_rate",
                "component": "skill_usage",
                "value": 0.18,
                "threshold": 0.10,
                "baseline": 0.12,
                "higher_is_better": False,
                "sample_count": 40,
                "status": "FAIL",
                "regressed_vs_baseline": True,
                "severity": 0.08,
            },
            {
                "id": "phase4_scaffold_pass_rate",
                "component": "code_evolution",
                "value": 1.0,
                "threshold": 1.0,
                "baseline": 1.0,
                "higher_is_better": True,
                "sample_count": 71,
                "status": "PASS",
                "regressed_vs_baseline": False,
                "severity": 0.0,
            },
            {
                "id": "prompt_contract_warning_rate",
                "component": "system_prompts",
                "value": 0.07,
                "threshold": 0.05,
                "baseline": 0.06,
                "higher_is_better": False,
                "sample_count": 20,
                "status": "FAIL",
                "regressed_vs_baseline": True,
                "severity": 0.02,
            },
        ],
        "weak_areas": [
            {
                "metric_id": "tool_selection_accuracy",
                "component": "tool_descriptions",
                "status": "FAIL",
                "regressed_vs_baseline": True,
                "severity": 0.04,
                "recommendation": "manual_triage_required_no_optimizer_started",
            },
            {
                "metric_id": "skill_loading_failure_rate",
                "component": "skill_usage",
                "status": "FAIL",
                "regressed_vs_baseline": True,
                "severity": 0.08,
                "recommendation": "manual_triage_required_no_optimizer_started",
            },
            {
                "metric_id": "prompt_contract_warning_rate",
                "component": "system_prompts",
                "status": "FAIL",
                "regressed_vs_baseline": True,
                "severity": 0.02,
                "recommendation": "manual_triage_required_no_optimizer_started",
            },
        ],
        "recommended_next_step": "manual_triage_required_no_optimizer_started",
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


def test_build_auto_triage_report_is_read_only_and_ranks_by_impact_times_frequency():
    from evolution.monitor.auto_triage import build_auto_triage_report

    report = build_auto_triage_report(
        _sample_performance_report(),
        generated_at="2026-06-05T01:00:00Z",
    )

    assert report["phase"] == "5"
    assert report["mode"] == "phase5-readonly-auto-triage-ranking"
    assert report["schema_version"] == "phase5-auto-triage-ranking-v1"
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["generated_at"] == "2026-06-05T01:00:00Z"
    assert report["source"] == {
        "performance_snapshot_schema_version": "phase5-performance-snapshot-v1",
        "performance_snapshot_mode": "phase5-readonly-performance-monitor-snapshot",
        "performance_snapshot_status": "NEEDS_TRIAGE",
        "performance_snapshot_window": {"start": "2026-06-01", "end": "2026-06-05"},
    }

    assert report["input_contract"] == {
        "performance_snapshot_report_required": True,
        "sanitized_input_required": True,
        "raw_session_data_allowed": False,
        "private_paths_allowed": False,
        "network_sources_allowed": False,
        "credentials_allowed": False,
    }
    assert report["safety_invariants"] == {
        "read_only": True,
        "raw_private_session_data_committed": False,
        "raw_credentials_recorded": False,
        "active_runtime_mutation": False,
        "external_calls_performed": False,
        "network_calls_performed": False,
        "cron_jobs_created": False,
        "scheduler_or_cron_side_effects_performed": False,
        "auto_optimizer_triggered": False,
        "optimizer_execution_started": False,
        "automated_pr_created_or_updated": False,
        "automated_apply_ready": False,
    }
    assert report["scoring"] == {
        "formula": "severity * sample_count",
        "tie_breakers": ["priority_score desc", "sample_count desc", "metric_id asc"],
        "optimizer_trigger_policy": "never_in_this_slice",
    }

    assert report["summary"] == {
        "candidate_metric_count": 4,
        "ranked_target_count": 3,
        "component_count": 3,
        "top_metric_id": "tool_selection_accuracy",
        "max_priority_score": 3.6,
        "review_required": True,
    }
    assert [target["metric_id"] for target in report["ranked_targets"]] == [
        "tool_selection_accuracy",
        "skill_loading_failure_rate",
        "prompt_contract_warning_rate",
    ]
    assert [target["rank"] for target in report["ranked_targets"]] == [1, 2, 3]
    assert [target["priority_score"] for target in report["ranked_targets"]] == [3.6, 3.2, 0.4]
    assert report["ranked_targets"][0]["reasons"] == ["failing_threshold", "regressed_vs_baseline"]
    assert report["ranked_targets"][0]["recommendation"] == "manual_review_required_no_optimizer_started"
    assert report["recommended_next_step"] == "manual_review_required_no_optimizer_started"
    _assert_privacy_safe(report)


def test_auto_triage_rejects_private_identifiers_and_non_readonly_performance_reports():
    from evolution.monitor.auto_triage import build_auto_triage_report

    performance_report = _sample_performance_report()
    performance_report["source"]["label"] = "/" + "Users" + "/" + "example" + "/raw"
    with pytest.raises(ValueError, match="private/raw identifier"):
        build_auto_triage_report(performance_report, generated_at="2026-06-05T01:00:00Z")

    performance_report = _sample_performance_report()
    performance_report["safety_invariants"]["optimizer_execution_started"] = True
    with pytest.raises(ValueError, match="performance report must be read-only"):
        build_auto_triage_report(performance_report, generated_at="2026-06-05T01:00:00Z")

    performance_report = _sample_performance_report()
    performance_report["safety_invariants"]["cron_jobs_created"] = True
    with pytest.raises(ValueError, match="performance report must be read-only"):
        build_auto_triage_report(performance_report, generated_at="2026-06-05T01:00:00Z")


def test_auto_triage_rejects_non_sanitized_performance_report_contract():
    from evolution.monitor.auto_triage import build_auto_triage_report

    performance_report = _sample_performance_report()
    performance_report["input_contract"]["raw_session_data_allowed"] = True
    with pytest.raises(ValueError, match="performance report input_contract must be sanitized"):
        build_auto_triage_report(performance_report, generated_at="2026-06-05T01:00:00Z")

    performance_report = _sample_performance_report()
    performance_report["input_contract"]["private_paths_allowed"] = True
    with pytest.raises(ValueError, match="performance report input_contract must be sanitized"):
        build_auto_triage_report(performance_report, generated_at="2026-06-05T01:00:00Z")


def test_cli_writes_auto_triage_json_and_markdown_under_phase5_output_root(tmp_path):
    performance_path = tmp_path / "performance_snapshot_report.json"
    performance_path.write_text(json.dumps(_sample_performance_report(), indent=2, sort_keys=True) + "\n")
    output_dir = OUTPUT_ROOT / "pytest-auto-triage-cli"
    shutil.rmtree(output_dir, ignore_errors=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolution.monitor.auto_triage",
            "--performance-report-json",
            str(performance_path),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-06-05T01:00:00Z",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report_path = output_dir / "auto_triage_report.json"
    markdown_path = output_dir / "auto_triage_report.md"
    assert report_path.exists()
    assert markdown_path.exists()
    report = json.loads(report_path.read_text())
    assert report["artifacts"] == {
        "report_json": "auto_triage_report.json",
        "report_markdown": "auto_triage_report.md",
    }
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["ranked_targets"][0]["metric_id"] == "tool_selection_accuracy"
    assert report["safety_invariants"]["auto_optimizer_triggered"] is False
    assert report["safety_invariants"]["scheduler_or_cron_side_effects_performed"] is False
    markdown = markdown_path.read_text()
    assert "# Phase 5 Auto-Triage Ranking" in markdown
    assert "Status: REVIEW_REQUIRED" in markdown
    assert "tool_selection_accuracy" in markdown
    assert "manual_review_required_no_optimizer_started" in markdown
    _assert_privacy_safe(report)
    _assert_privacy_safe(markdown)


def test_cli_rejects_auto_triage_output_dir_outside_phase5_output_root(tmp_path):
    performance_path = tmp_path / "performance_snapshot_report.json"
    performance_path.write_text(json.dumps(_sample_performance_report(), indent=2, sort_keys=True) + "\n")
    output_dir = tmp_path / "outside-phase5-root"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolution.monitor.auto_triage",
            "--performance-report-json",
            str(performance_path),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-06-05T01:00:00Z",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "output-dir must be under output/phase5-continuous-loop" in result.stderr
    assert not (output_dir / "auto_triage_report.json").exists()


def test_phase5_auto_triage_contract_is_documented_and_in_ci():
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    triggers = workflow.get("on", workflow.get(True))
    run_blocks = "\n".join(
        str(step.get("run", ""))
        for step in workflow["jobs"]["phase2-tool-description-gate"]["steps"]
        if isinstance(step, dict)
    )

    assert "tests/monitor/test_phase5_auto_triage.py" in run_blocks
    for trigger_name in ("pull_request", "push"):
        assert "evolution/monitor/**" in triggers[trigger_name]["paths"]
        assert "tests/monitor/**" in triggers[trigger_name]["paths"]
        assert "reports/phase5_*" in triggers[trigger_name]["paths"]

    contract = CONTRACT_MD.read_text()
    assert "# Phase 5 Auto-Triage Report Contract" in contract
    assert "phase5-readonly-auto-triage-ranking" in contract
    assert "auto_optimizer_triggered=false" in contract
    assert "scheduler_or_cron_side_effects_performed=false" in contract
    assert "manual_review_required_no_optimizer_started" in contract
    assert "No raw session data, local private paths, or credentials" in contract

    readme = README_MD.read_text()
    plan = PLAN_MD.read_text()
    assert "Phase 5 auto-triage report contract" in readme
    assert "evolution.monitor.auto_triage" in readme
    assert "read-only auto-triage ranking" in plan
    assert "manual review; no optimizer is started" in plan

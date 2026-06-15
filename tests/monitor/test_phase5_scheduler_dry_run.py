"""Tests for the Phase 5 no-side-effect scheduler dry-run report."""

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
CONTRACT_MD = REPO_ROOT / "reports" / "phase5_scheduler_dry_run_report_contract.md"


def _sample_auto_triage_report() -> dict:
    return {
        "schema_version": "phase5-auto-triage-ranking-v1",
        "phase": "5",
        "mode": "phase5-readonly-auto-triage-ranking",
        "status": "REVIEW_REQUIRED",
        "generated_at": "2026-06-05T01:00:00Z",
        "source": {
            "performance_snapshot_schema_version": "phase5-performance-snapshot-v1",
            "performance_snapshot_mode": "phase5-readonly-performance-monitor-snapshot",
            "performance_snapshot_status": "NEEDS_TRIAGE",
            "performance_snapshot_window": {"start": "2026-06-01", "end": "2026-06-05"},
        },
        "input_contract": {
            "performance_snapshot_report_required": True,
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
            "scheduler_or_cron_side_effects_performed": False,
            "auto_optimizer_triggered": False,
            "optimizer_execution_started": False,
            "automated_pr_created_or_updated": False,
            "automated_apply_ready": False,
        },
        "scoring": {
            "formula": "severity * sample_count",
            "tie_breakers": ["priority_score desc", "sample_count desc", "metric_id asc"],
            "optimizer_trigger_policy": "never_in_this_slice",
        },
        "summary": {
            "candidate_metric_count": 4,
            "ranked_target_count": 3,
            "component_count": 3,
            "top_metric_id": "tool_selection_accuracy",
            "max_priority_score": 3.6,
            "review_required": True,
        },
        "ranked_targets": [
            {
                "rank": 1,
                "metric_id": "tool_selection_accuracy",
                "component": "tool_descriptions",
                "status": "FAIL",
                "severity": 0.04,
                "sample_count": 90,
                "regressed_vs_baseline": True,
                "priority_score": 3.6,
                "priority_inputs": {"severity": 0.04, "sample_count": 90},
                "reasons": ["failing_threshold", "regressed_vs_baseline"],
                "recommendation": "manual_review_required_no_optimizer_started",
            },
            {
                "rank": 2,
                "metric_id": "skill_loading_failure_rate",
                "component": "skill_usage",
                "status": "FAIL",
                "severity": 0.08,
                "sample_count": 40,
                "regressed_vs_baseline": True,
                "priority_score": 3.2,
                "priority_inputs": {"severity": 0.08, "sample_count": 40},
                "reasons": ["failing_threshold", "regressed_vs_baseline"],
                "recommendation": "manual_review_required_no_optimizer_started",
            },
            {
                "rank": 3,
                "metric_id": "prompt_contract_warning_rate",
                "component": "system_prompts",
                "status": "FAIL",
                "severity": 0.02,
                "sample_count": 20,
                "regressed_vs_baseline": True,
                "priority_score": 0.4,
                "priority_inputs": {"severity": 0.02, "sample_count": 20},
                "reasons": ["failing_threshold", "regressed_vs_baseline"],
                "recommendation": "manual_review_required_no_optimizer_started",
            },
        ],
        "recommended_next_step": "manual_review_required_no_optimizer_started",
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


def test_build_scheduler_dry_run_report_is_read_only_and_plans_no_side_effect_actions():
    from evolution.monitor.scheduler_dry_run import build_scheduler_dry_run_report

    report = build_scheduler_dry_run_report(
        _sample_auto_triage_report(),
        generated_at="2026-06-05T02:00:00Z",
    )

    assert report["phase"] == "5"
    assert report["mode"] == "phase5-readonly-scheduler-dry-run"
    assert report["schema_version"] == "phase5-scheduler-dry-run-v1"
    assert report["status"] == "DRY_RUN_REVIEW_REQUIRED"
    assert report["generated_at"] == "2026-06-05T02:00:00Z"
    assert report["source"] == {
        "auto_triage_schema_version": "phase5-auto-triage-ranking-v1",
        "auto_triage_mode": "phase5-readonly-auto-triage-ranking",
        "auto_triage_status": "REVIEW_REQUIRED",
        "ranked_target_count": 3,
        "top_metric_id": "tool_selection_accuracy",
    }

    assert report["input_contract"] == {
        "auto_triage_report_required": True,
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
        "benchmark_cron_enabled": False,
        "scheduler_or_cron_side_effects_performed": False,
        "notifications_sent": False,
        "auto_optimizer_triggered": False,
        "optimizer_execution_started": False,
        "automated_pr_created_or_updated": False,
        "automated_apply_ready": False,
    }
    assert report["dry_run_policy"] == {
        "scheduler_enablement_policy": "never_enable_in_this_slice",
        "max_actions": 3,
        "action_source": "auto_triage_ranked_targets",
        "required_before_real_scheduler": [
            "explicit human approval for scheduler enablement",
            "Phase 4 formal handoff reviewed or waived",
            "benchmark/API budget approval",
            "cron target and delivery channel review",
        ],
    }
    assert report["summary"] == {
        "ranked_target_count": 3,
        "dry_run_action_count": 3,
        "side_effect_count": 0,
        "top_metric_id": "tool_selection_accuracy",
        "scheduler_enablement_ready": False,
        "review_required": True,
    }
    assert [action["target_metric_id"] for action in report["dry_run_actions"]] == [
        "tool_selection_accuracy",
        "skill_loading_failure_rate",
        "prompt_contract_warning_rate",
    ]
    first_action = report["dry_run_actions"][0]
    assert first_action == {
        "action_id": "dry-run-action-001",
        "action_type": "manual_triage_review",
        "target_rank": 1,
        "target_metric_id": "tool_selection_accuracy",
        "component": "tool_descriptions",
        "priority_score": 3.6,
        "dry_run_only": True,
        "would_create_cron_job": False,
        "would_enable_benchmark_cron": False,
        "would_start_optimizer": False,
        "would_send_external_notification": False,
        "would_update_external_pr": False,
        "proposed_cadence": "manual_review_only",
        "required_approval": "human_review_required_before_scheduler_enablement",
        "recommendation": "review_target_no_scheduler_side_effects",
    }
    assert report["recommended_next_step"] == "human_review_required_before_scheduler_enablement"
    _assert_privacy_safe(report)


def test_scheduler_dry_run_rejects_private_identifiers_and_non_readonly_triage_reports():
    from evolution.monitor.scheduler_dry_run import build_scheduler_dry_run_report

    triage_report = _sample_auto_triage_report()
    triage_report["ranked_targets"][0]["metric_id"] = "/" + "Users" + "/" + "example" + "/raw"
    with pytest.raises(ValueError, match="private/raw identifier"):
        build_scheduler_dry_run_report(triage_report, generated_at="2026-06-05T02:00:00Z")

    triage_report = _sample_auto_triage_report()
    triage_report["safety_invariants"]["auto_optimizer_triggered"] = True
    with pytest.raises(ValueError, match="auto-triage report must be read-only"):
        build_scheduler_dry_run_report(triage_report, generated_at="2026-06-05T02:00:00Z")

    triage_report = _sample_auto_triage_report()
    triage_report["safety_invariants"]["scheduler_or_cron_side_effects_performed"] = True
    with pytest.raises(ValueError, match="auto-triage report must be read-only"):
        build_scheduler_dry_run_report(triage_report, generated_at="2026-06-05T02:00:00Z")

    triage_report = _sample_auto_triage_report()
    triage_report["safety_invariants"]["benchmark_cron_enabled"] = True
    with pytest.raises(ValueError, match="auto-triage report must be read-only"):
        build_scheduler_dry_run_report(triage_report, generated_at="2026-06-05T02:00:00Z")

    triage_report = _sample_auto_triage_report()
    triage_report["safety_invariants"]["notifications_sent"] = True
    with pytest.raises(ValueError, match="auto-triage report must be read-only"):
        build_scheduler_dry_run_report(triage_report, generated_at="2026-06-05T02:00:00Z")


def test_scheduler_dry_run_rejects_non_sanitized_triage_contract():
    from evolution.monitor.scheduler_dry_run import build_scheduler_dry_run_report

    triage_report = _sample_auto_triage_report()
    triage_report["input_contract"]["raw_session_data_allowed"] = True
    with pytest.raises(ValueError, match="auto-triage report input_contract must be sanitized"):
        build_scheduler_dry_run_report(triage_report, generated_at="2026-06-05T02:00:00Z")

    triage_report = _sample_auto_triage_report()
    triage_report["input_contract"]["network_sources_allowed"] = True
    with pytest.raises(ValueError, match="auto-triage report input_contract must be sanitized"):
        build_scheduler_dry_run_report(triage_report, generated_at="2026-06-05T02:00:00Z")


def test_scheduler_dry_run_rejects_inconsistent_review_required_without_targets():
    from evolution.monitor.scheduler_dry_run import build_scheduler_dry_run_report

    triage_report = _sample_auto_triage_report()
    triage_report["summary"]["ranked_target_count"] = 0
    triage_report["summary"]["top_metric_id"] = None
    triage_report["summary"]["review_required"] = True
    triage_report["ranked_targets"] = []

    with pytest.raises(ValueError, match="REVIEW_REQUIRED report must contain ranked targets"):
        build_scheduler_dry_run_report(triage_report, generated_at="2026-06-05T02:00:00Z")


def test_scheduler_dry_run_rejects_duplicate_or_unordered_ranks():
    from evolution.monitor.scheduler_dry_run import build_scheduler_dry_run_report

    triage_report = _sample_auto_triage_report()
    triage_report["ranked_targets"][1]["rank"] = 1
    with pytest.raises(ValueError, match="ranked target ranks must be unique and sequential"):
        build_scheduler_dry_run_report(triage_report, generated_at="2026-06-05T02:00:00Z")

    triage_report = _sample_auto_triage_report()
    triage_report["ranked_targets"][0]["rank"] = 2
    triage_report["ranked_targets"][1]["rank"] = 1
    with pytest.raises(ValueError, match="ranked target ranks must be unique and sequential"):
        build_scheduler_dry_run_report(triage_report, generated_at="2026-06-05T02:00:00Z")


def test_scheduler_dry_run_noop_when_auto_triage_has_no_targets():
    from evolution.monitor.scheduler_dry_run import build_scheduler_dry_run_report

    triage_report = _sample_auto_triage_report()
    triage_report["status"] = "NO_ACTION"
    triage_report["summary"]["ranked_target_count"] = 0
    triage_report["summary"]["top_metric_id"] = None
    triage_report["summary"]["review_required"] = False
    triage_report["ranked_targets"] = []
    triage_report["recommended_next_step"] = "no_action_monitor_only"

    report = build_scheduler_dry_run_report(triage_report, generated_at="2026-06-05T02:00:00Z")

    assert report["status"] == "DRY_RUN_NOOP"
    assert report["summary"] == {
        "ranked_target_count": 0,
        "dry_run_action_count": 0,
        "side_effect_count": 0,
        "top_metric_id": None,
        "scheduler_enablement_ready": False,
        "review_required": False,
    }
    assert report["dry_run_actions"] == []
    assert report["recommended_next_step"] == "monitor_only_no_scheduler_action"


def test_cli_writes_scheduler_dry_run_json_and_markdown_under_phase5_output_root(tmp_path):
    triage_path = tmp_path / "auto_triage_report.json"
    triage_path.write_text(json.dumps(_sample_auto_triage_report(), indent=2, sort_keys=True) + "\n")
    output_dir = OUTPUT_ROOT / "pytest-scheduler-dry-run-cli"
    shutil.rmtree(output_dir, ignore_errors=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolution.monitor.scheduler_dry_run",
            "--auto-triage-report-json",
            str(triage_path),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-06-05T02:00:00Z",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report_path = output_dir / "scheduler_dry_run_report.json"
    markdown_path = output_dir / "scheduler_dry_run_report.md"
    assert report_path.exists()
    assert markdown_path.exists()
    report = json.loads(report_path.read_text())
    assert report["artifacts"] == {
        "report_json": "scheduler_dry_run_report.json",
        "report_markdown": "scheduler_dry_run_report.md",
    }
    assert report["status"] == "DRY_RUN_REVIEW_REQUIRED"
    assert report["dry_run_actions"][0]["would_create_cron_job"] is False
    assert report["safety_invariants"]["cron_jobs_created"] is False
    assert report["safety_invariants"]["benchmark_cron_enabled"] is False
    markdown = markdown_path.read_text()
    assert "# Phase 5 Scheduler Dry-Run" in markdown
    assert "Status: DRY_RUN_REVIEW_REQUIRED" in markdown
    assert "cron_jobs_created=false" in markdown
    assert "benchmark_cron_enabled=false" in markdown
    assert "human_review_required_before_scheduler_enablement" in markdown
    _assert_privacy_safe(report)
    _assert_privacy_safe(markdown)


def test_cli_rejects_scheduler_dry_run_output_dir_outside_phase5_output_root(tmp_path):
    triage_path = tmp_path / "auto_triage_report.json"
    triage_path.write_text(json.dumps(_sample_auto_triage_report(), indent=2, sort_keys=True) + "\n")
    output_dir = tmp_path / "outside-phase5-root"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolution.monitor.scheduler_dry_run",
            "--auto-triage-report-json",
            str(triage_path),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-06-05T02:00:00Z",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "output-dir must be under output/phase5-continuous-loop" in result.stderr
    assert not (output_dir / "scheduler_dry_run_report.json").exists()


def test_cli_rejects_scheduler_dry_run_non_empty_output_dir(tmp_path):
    triage_path = tmp_path / "auto_triage_report.json"
    triage_path.write_text(json.dumps(_sample_auto_triage_report(), indent=2, sort_keys=True) + "\n")
    output_dir = OUTPUT_ROOT / "pytest-scheduler-dry-run-non-empty"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    (output_dir / "stale.txt").write_text("stale artifact\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolution.monitor.scheduler_dry_run",
            "--auto-triage-report-json",
            str(triage_path),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-06-05T02:00:00Z",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "output-dir must be empty before writing scheduler dry-run artifacts" in result.stderr
    assert not (output_dir / "scheduler_dry_run_report.json").exists()


def test_phase5_scheduler_dry_run_contract_is_documented_and_in_ci():
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    triggers = workflow.get("on", workflow.get(True))
    run_blocks = "\n".join(
        str(step.get("run", ""))
        for step in workflow["jobs"]["phase2-tool-description-gate"]["steps"]
        if isinstance(step, dict)
    )

    assert "tests/monitor/test_phase5_scheduler_dry_run.py" in run_blocks
    for trigger_name in ("pull_request", "push"):
        assert "evolution/monitor/**" in triggers[trigger_name]["paths"]
        assert "tests/monitor/**" in triggers[trigger_name]["paths"]
        assert "reports/phase5_*" in triggers[trigger_name]["paths"]

    contract = CONTRACT_MD.read_text()
    assert "# Phase 5 Scheduler Dry-Run Report Contract" in contract
    assert "phase5-readonly-scheduler-dry-run" in contract
    assert "cron_jobs_created=false" in contract
    assert "benchmark_cron_enabled=false" in contract
    assert "scheduler_or_cron_side_effects_performed=false" in contract
    assert "human_review_required_before_scheduler_enablement" in contract
    assert "No raw session data, local private paths, or credentials" in contract

    readme = README_MD.read_text()
    plan = PLAN_MD.read_text()
    assert "Phase 5 scheduler dry-run report contract" in readme
    assert "evolution.monitor.scheduler_dry_run" in readme
    assert "read-only scheduler dry-run" in plan
    assert "no cron jobs are created" in plan

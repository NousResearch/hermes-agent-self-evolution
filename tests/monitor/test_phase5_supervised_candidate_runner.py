"""Tests for the Phase 5 supervised local candidate runner handoff."""

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
CONTRACT_MD = REPO_ROOT / "reports" / "phase5_supervised_candidate_runner_contract.md"


def _sample_scheduler_report() -> dict:
    return {
        "schema_version": "phase5-scheduler-dry-run-v1",
        "phase": "5",
        "mode": "phase5-readonly-scheduler-dry-run",
        "status": "DRY_RUN_REVIEW_REQUIRED",
        "generated_at": "2026-06-22T16:06:00Z",
        "source": {
            "auto_triage_schema_version": "phase5-auto-triage-ranking-v1",
            "auto_triage_mode": "phase5-readonly-auto-triage-ranking",
            "auto_triage_status": "REVIEW_REQUIRED",
            "ranked_target_count": 1,
            "top_metric_id": "tool_selection_accuracy",
        },
        "input_contract": {
            "auto_triage_report_required": True,
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
            "benchmark_cron_enabled": False,
            "scheduler_or_cron_side_effects_performed": False,
            "notifications_sent": False,
            "auto_optimizer_triggered": False,
            "optimizer_execution_started": False,
            "automated_pr_created_or_updated": False,
            "automated_apply_ready": False,
        },
        "local_candidate_bundle_contract": {
            "schema_version": "hse-local-candidate-bundle-v1",
            "decision_json_consumed": False,
            "decision_json_required_before_apply": True,
            "runner_execution_started": False,
            "active_apply_ready": False,
            "github_publication_performed": False,
        },
        "candidate_bundle_queue_summary": {
            "queue_count": 1,
            "decision_count": 0,
            "matched_decision_count": 0,
            "missing_decision_count": 1,
            "runner_execution_started": False,
            "active_apply_ready": False,
            "github_publication_performed": False,
        },
        "candidate_bundle_queue": [
            {
                "queue_id": "candidate-bundle-target-001",
                "target_rank": 1,
                "target_metric_id": "tool_selection_accuracy",
                "component": "tool_descriptions",
                "candidate_bundle_phase": "Phase 2: Tool Description Evolution",
                "candidate_bundle_target": "tool-description",
                "runner_hint": "python -m evolution.tools.evolve_tool_descriptions",
                "decision_state": "MISSING_DECISION",
                "decision_status": None,
                "decision_run_id": None,
                "decision_apply_ready": False,
                "decision_github_pr_created": False,
                "would_start_runner": False,
                "would_create_local_bundle": False,
                "requires_human_review_before_apply": True,
            }
        ],
        "recommended_next_step": "human_review_required_before_scheduler_enablement",
    }


def _sample_candidate_bundle_decision() -> dict:
    return {
        "schema_version": "hse-local-candidate-bundle-v1",
        "status": "PASS_CANDIDATE_ONLY",
        "summary": "Phase 2 local candidate generated under supervised approval.",
        "phase": "Phase 2: Tool Description Evolution",
        "target": "tool-description",
        "run_id": "pytest-supervised-phase2-run",
        "generated_at": "2026-06-22T16:08:00Z",
        "candidate_only": True,
        "apply_ready": False,
        "github": {
            "pr_created": False,
            "push_performed": False,
            "merge_performed": False,
            "publication_deferred": True,
        },
        "safety_invariants": {
            "active_runtime_mutation": False,
            "active_skill_modified": False,
            "active_tool_schema_modified": False,
            "active_prompt_modified": False,
            "credentials_accessed": False,
            "external_publication_performed": False,
            "deployment_performed": False,
        },
        "metrics": {"selection_accuracy": 0.82},
        "artifacts": {"patch": "candidates/candidate.patch"},
    }


def _assert_privacy_safe(value: object) -> None:
    text = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
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


def test_build_supervised_runner_report_requires_approval_and_reconsumes_one_decision():
    from evolution.monitor.supervised_candidate_runner import PUBLIC_APPROVAL_SENTINEL, build_supervised_runner_report

    report = build_supervised_runner_report(
        _sample_scheduler_report(),
        approved_queue_id="candidate-bundle-target-001",
        approval_token=PUBLIC_APPROVAL_SENTINEL,
        candidate_bundle_decision=_sample_candidate_bundle_decision(),
        generated_at="2026-06-22T16:09:00Z",
    )

    assert report["schema_version"] == "phase5-supervised-candidate-runner-v1"
    assert report["phase"] == "5"
    assert report["mode"] == "phase5-supervised-local-candidate-runner"
    assert report["status"] == "SUPERVISED_DECISION_RECONSUMED"
    assert report["approval"] == {
        "approved_queue_id": "candidate-bundle-target-001",
        "explicit_approval_recorded": True,
        "approval_scope": "single_local_candidate_queue_target_only",
    }
    assert report["selected_queue_target"] == {
        "queue_id": "candidate-bundle-target-001",
        "target_metric_id": "tool_selection_accuracy",
        "component": "tool_descriptions",
        "candidate_bundle_phase": "Phase 2: Tool Description Evolution",
        "candidate_bundle_target": "tool-description",
        "runner_hint": "python -m evolution.tools.evolve_tool_descriptions",
    }
    assert report["runner_invocation"] == {
        "execution_mode": "manual_decision_reconsume_only",
        "runner_execution_started": False,
        "runner_returncode": None,
        "runner_hint": "python -m evolution.tools.evolve_tool_descriptions",
        "raw_command_recorded": False,
    }
    assert report["decision_reconsume"] == {
        "decision_state": "DECISION_AVAILABLE",
        "decision_status": "PASS_CANDIDATE_ONLY",
        "decision_run_id": "pytest-supervised-phase2-run",
        "decision_candidate_only": True,
        "decision_apply_ready": False,
        "decision_github_pr_created": False,
    }
    assert report["safety_invariants"] == {
        "active_runtime_mutation": False,
        "active_apply_ready": False,
        "credentials_accessed": False,
        "cron_jobs_created": False,
        "scheduler_or_cron_side_effects_performed": False,
        "external_calls_performed": False,
        "network_calls_performed": False,
        "github_publication_performed": False,
        "automated_pr_created_or_updated": False,
        "deployment_performed": False,
    }
    _assert_privacy_safe(report)


def test_supervised_runner_rejects_missing_approval_unknown_queue_and_unsafe_decision():
    from evolution.monitor.supervised_candidate_runner import PUBLIC_APPROVAL_SENTINEL, build_supervised_runner_report

    with pytest.raises(ValueError, match="explicit approval token"):
        build_supervised_runner_report(
            _sample_scheduler_report(),
            approved_queue_id="candidate-bundle-target-001",
            approval_token="wrong",
            candidate_bundle_decision=_sample_candidate_bundle_decision(),
            generated_at="2026-06-22T16:09:00Z",
        )

    with pytest.raises(ValueError, match="approved queue id was not found"):
        build_supervised_runner_report(
            _sample_scheduler_report(),
            approved_queue_id="candidate-bundle-target-999",
            approval_token=PUBLIC_APPROVAL_SENTINEL,
            candidate_bundle_decision=_sample_candidate_bundle_decision(),
            generated_at="2026-06-22T16:09:00Z",
        )

    decision = _sample_candidate_bundle_decision()
    decision["apply_ready"] = True
    with pytest.raises(ValueError, match="candidate bundle decision must be candidate-only"):
        build_supervised_runner_report(
            _sample_scheduler_report(),
            approved_queue_id="candidate-bundle-target-001",
            approval_token=PUBLIC_APPROVAL_SENTINEL,
            candidate_bundle_decision=decision,
            generated_at="2026-06-22T16:09:00Z",
        )


def test_supervised_runner_rejects_same_phase_wrong_target_decision():
    from evolution.monitor.supervised_candidate_runner import PUBLIC_APPROVAL_SENTINEL, build_supervised_runner_report

    for decision_target in (
        "unrelated-phase-2-target",
        "tool_descriptions",
        "tool_selection_accuracy",
        "tool_description",
        "Tool-Description",
    ):
        decision = _sample_candidate_bundle_decision()
        decision["phase"] = "Phase 2: Tool Description Evolution"
        decision["target"] = decision_target
        with pytest.raises(ValueError, match="does not match approved queue target"):
            build_supervised_runner_report(
                _sample_scheduler_report(),
                approved_queue_id="candidate-bundle-target-001",
                approval_token=PUBLIC_APPROVAL_SENTINEL,
                candidate_bundle_decision=decision,
                generated_at="2026-06-22T16:09:00Z",
            )

    phase_alias_decision = _sample_candidate_bundle_decision()
    phase_alias_decision["phase"] = "Phase 2 Tool Description Evolution"
    with pytest.raises(ValueError, match="does not match approved queue target"):
        build_supervised_runner_report(
            _sample_scheduler_report(),
            approved_queue_id="candidate-bundle-target-001",
            approval_token=PUBLIC_APPROVAL_SENTINEL,
            candidate_bundle_decision=phase_alias_decision,
            generated_at="2026-06-22T16:09:00Z",
        )


def test_cli_reconsumes_existing_decision_json_without_inline_execution(tmp_path):
    scheduler_path = tmp_path / "scheduler_report.json"
    decision_path = tmp_path / "decision.json"
    scheduler_path.write_text(json.dumps(_sample_scheduler_report(), indent=2, sort_keys=True) + "\n")
    decision_path.write_text(json.dumps(_sample_candidate_bundle_decision(), indent=2, sort_keys=True) + "\n")
    output_dir = OUTPUT_ROOT / "pytest-supervised-candidate-runner-cli"
    shutil.rmtree(output_dir, ignore_errors=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolution.monitor.supervised_candidate_runner",
            "--scheduler-report-json",
            str(scheduler_path),
            "--approved-queue-id",
            "candidate-bundle-target-001",
            "--approval-token",
            "APPROVE_LOCAL_CANDIDATE_RUNNER",
            "--candidate-bundle-decision-json",
            str(decision_path),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-06-22T16:10:00Z",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report = json.loads((output_dir / "supervised_candidate_runner_report.json").read_text())
    assert report["status"] == "SUPERVISED_DECISION_RECONSUMED"
    assert report["runner_invocation"]["execution_mode"] == "manual_decision_reconsume_only"
    assert report["runner_invocation"]["runner_execution_started"] is False
    assert report["runner_invocation"]["runner_returncode"] is None
    assert report["runner_invocation"]["raw_command_recorded"] is False
    assert report["decision_reconsume"]["decision_state"] == "DECISION_AVAILABLE"
    assert report["safety_invariants"]["active_apply_ready"] is False
    assert report["safety_invariants"]["github_publication_performed"] is False
    markdown = (output_dir / "supervised_candidate_runner_report.md").read_text()
    assert "# Phase 5 Supervised Candidate Runner" in markdown
    assert "SUPERVISED_DECISION_RECONSUMED" in markdown
    assert "runner_execution_started=false" in markdown
    assert "active_apply_ready=false" in markdown
    _assert_privacy_safe(report)
    _assert_privacy_safe(markdown)


def test_cli_rejects_inline_execution_and_does_not_run_command(tmp_path):
    scheduler_path = tmp_path / "scheduler_report.json"
    source_decision_path = tmp_path / "source_decision.json"
    produced_decision_path = tmp_path / "produced_decision.json"
    scheduler_path.write_text(json.dumps(_sample_scheduler_report(), indent=2, sort_keys=True) + "\n")
    source_decision_path.write_text(json.dumps(_sample_candidate_bundle_decision(), indent=2, sort_keys=True) + "\n")
    runner_command = [
        sys.executable,
        "-c",
        "import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2])",
        str(source_decision_path),
        str(produced_decision_path),
    ]
    output_dir = OUTPUT_ROOT / "pytest-supervised-candidate-runner-inline-reject"
    shutil.rmtree(output_dir, ignore_errors=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolution.monitor.supervised_candidate_runner",
            "--scheduler-report-json",
            str(scheduler_path),
            "--approved-queue-id",
            "candidate-bundle-target-001",
            "--approval-token",
            "WRONG",
            "--execute-approved-runner",
            "--runner-command-json",
            json.dumps(runner_command),
            "--candidate-bundle-decision-json",
            str(produced_decision_path),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-06-22T16:10:00Z",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "inline runner execution is disabled" in result.stderr
    assert not produced_decision_path.exists()
    assert not (output_dir / "supervised_candidate_runner_report.json").exists()


def test_supervised_candidate_runner_contract_is_documented_and_in_ci():
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    run_blocks = "\n".join(
        str(step.get("run", ""))
        for step in workflow["jobs"]["phase2-tool-description-gate"]["steps"]
        if isinstance(step, dict)
    )

    assert "tests/monitor/test_phase5_supervised_candidate_runner.py" in run_blocks
    contract = CONTRACT_MD.read_text()
    assert "# Phase 5 Supervised Candidate Runner Contract" in contract
    assert "phase5-supervised-local-candidate-runner" in contract
    assert "APPROVE_LOCAL_CANDIDATE_RUNNER" in contract
    assert "single_local_candidate_queue_target_only" in contract
    assert "active_apply_ready=false" in contract
    assert "github_publication_performed=false" in contract

    readme = README_MD.read_text()
    plan = PLAN_MD.read_text()
    assert "Phase 5 supervised candidate runner" in readme
    assert "evolution.monitor.supervised_candidate_runner" in readme
    assert "--approved-queue-id" in readme
    assert "supervised local candidate runner" in plan
    assert "APPROVE_LOCAL_CANDIDATE_RUNNER" in plan

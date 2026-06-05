"""Tests for the Phase 5 continuous-loop readiness manifest."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_JSON = REPO_ROOT / "reports" / "phase5_continuous_loop_readiness_manifest.json"
MANIFEST_MD = REPO_ROOT / "reports" / "phase5_continuous_loop_readiness_manifest.md"
PLAN_MD = REPO_ROOT / "PLAN.md"
README_MD = REPO_ROOT / "README.md"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "phase2-tool-description-gate.yml"

EXPECTED_REQUIRED_COMPONENTS = {
    "performance_monitor",
    "auto_triage",
    "scheduler_dry_run",
    "feedback_loop_dataset_ingestion",
    "human_review_handoff",
    "safety_report_contract",
}

EXPECTED_SEPARATE_APPROVALS = [
    "creating or enabling Hermes cron jobs",
    "running real TBLite benchmark commands",
    "running real YC-Bench benchmark commands",
    "spending nonzero benchmark/API budget",
    "running GEPA/DSPy optimization",
    "running Darwinian Evolver optimization",
    "creating or updating external GitHub pull requests automatically",
    "editing active Hermes Agent source, skills, prompts, memory, config, or runtime state",
]


def _manifest() -> dict:
    return json.loads(MANIFEST_JSON.read_text())


def _all_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _all_strings(key)
            yield from _all_strings(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _all_strings(child)


def test_phase5_manifest_records_local_start_but_blocks_unattended_execution():
    manifest = _manifest()

    assert manifest["phase"] == "5"
    assert manifest["mode"] == "phase5-continuous-loop-readiness-manifest"
    assert manifest["status"] == "local_preparation_started_not_deployed"
    assert manifest["source_authorization"] == "rec action GO"
    assert manifest["manifest_version"] == "phase5-continuous-loop-readiness-v1"

    assert manifest["local_phase5_preparation_started"] is True
    assert manifest["continuous_loop_enabled"] is False
    assert manifest["cron_jobs_created"] is False
    assert manifest["benchmark_cron_enabled"] is False
    assert manifest["threshold_triggered_optimization_enabled"] is False
    assert manifest["automated_pr_creation_enabled"] is False
    assert manifest["active_runtime_apply_ready"] is False

    assert manifest["required_components"] == sorted(EXPECTED_REQUIRED_COMPONENTS)

    approval = manifest["approval_gate"]
    assert approval["planning_authorized"] == "rec action GO"
    assert approval["scheduler_enablement_approved"] is False
    assert approval["optimizer_execution_approved"] is False
    assert approval["automated_external_handoff_approved"] is False
    assert approval["active_apply_approved"] is False
    assert approval["separate_approval_required_before"] == EXPECTED_SEPARATE_APPROVALS


def test_phase5_manifest_separates_github_handoff_from_local_preparation():
    manifest = _manifest()

    gate = manifest["github_gate_policy"]
    assert gate["required_for_local_readiness_planning"] is False
    assert gate["required_for_read_only_scaffold_work"] is False
    assert gate["required_before_unattended_scheduler_enablement"] is True
    assert gate["required_before_formal_phase5_completion"] is True
    assert gate["phase4_pr_url"] == "https://github.com/NousResearch/hermes-agent-self-evolution/pull/98"
    assert gate["phase4_pr_draft"] is True
    assert gate["phase4_pr_checks_reported"] is False
    assert gate["phase4_upstream_merged"] is False

    prerequisites = manifest["phase4_dependency_state"]
    assert prerequisites["local_phase4_engineering_complete"] is True
    assert prerequisites["local_phase4_tests_green"] is True
    assert prerequisites["fork_branch_pushed"] is True
    assert prerequisites["draft_pr_updated"] is True
    assert prerequisites["upstream_merge_complete"] is False
    assert prerequisites["ci_checks_reported"] is False


def test_phase5_manifest_is_privacy_safe_and_fail_closed():
    manifest = _manifest()
    manifest_strings = list(_all_strings(manifest)) + [
        MANIFEST_JSON.read_text(),
        MANIFEST_MD.read_text(),
    ]
    docs_strings = [PLAN_MD.read_text(), README_MD.read_text()]
    common_forbidden_fragments = [
        "/" + "Users" + "/",
        "/" + "home" + "/",
        "OPENAI" + "_API_KEY",
        "ANTHROPIC" + "_API_KEY",
        "OPENROUTER" + "_API_KEY",
    ]
    manifest_only_forbidden_fragments = [
        "state" + ".db",
        "session" + "_id",
    ]
    for fragment in common_forbidden_fragments:
        assert all(fragment not in item for item in manifest_strings + docs_strings), fragment
    for fragment in manifest_only_forbidden_fragments:
        assert all(fragment not in item for item in manifest_strings), fragment

    safety = manifest["safety_invariants"]
    assert safety == {
        "raw_private_session_data_committed": False,
        "raw_credentials_recorded": False,
        "active_runtime_mutation": False,
        "external_calls_performed": False,
        "network_calls_performed": False,
        "cron_or_scheduler_side_effects_performed": False,
        "automatic_merge_or_deploy_allowed": False,
    }

    ready_state = manifest["ready_state"]
    assert ready_state == {
        "phase5_local_planning_ready_now": True,
        "phase5_unattended_loop_ready_now": False,
        "blocked_until_all_go_no_go_conditions_satisfied": True,
        "active_apply_ready_now": False,
    }

    condition_ids = {condition["id"] for condition in manifest["go_no_go_conditions"]}
    assert condition_ids == {
        "P5-1-phase4-formal-handoff-reviewed",
        "P5-2-readonly-performance-monitor-contract",
        "P5-3-auto-triage-dry-run-contract",
        "P5-4-scheduler-dry-run-no-side-effects",
        "P5-5-explicit-budget-and-optimizer-approval",
        "P5-6-human-review-and-no-auto-merge",
    }
    condition_status = {
        condition["id"]: condition["status"]
        for condition in manifest["go_no_go_conditions"]
    }
    assert condition_status == {
        "P5-1-phase4-formal-handoff-reviewed": "required_future_gate",
        "P5-2-readonly-performance-monitor-contract": "required_future_gate",
        "P5-3-auto-triage-dry-run-contract": "required_future_gate",
        "P5-4-scheduler-dry-run-no-side-effects": "required_future_gate",
        "P5-5-explicit-budget-and-optimizer-approval": "required_future_gate",
        "P5-6-human-review-and-no-auto-merge": "satisfied_for_local_planning",
    }
    for condition in manifest["go_no_go_conditions"]:
        assert condition["required_before"] == "phase5_unattended_loop_enablement"


def test_phase5_markdown_plan_readme_and_ci_record_boundary():
    manifest = _manifest()
    markdown = MANIFEST_MD.read_text()
    plan_md = PLAN_MD.read_text()
    readme = README_MD.read_text()
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())

    assert "# Phase 5 Continuous Loop Readiness Manifest" in markdown
    assert "Status: local preparation started, not deployed" in markdown
    assert "GitHub handoff is not required for local read-only planning" in markdown
    assert "continuous_loop_enabled=false" in markdown
    assert "cron_jobs_created=false" in markdown
    assert "phase5_unattended_loop_ready_now=false" in markdown

    assert "**Current Phase 5 readiness status:** local preparation started, not deployed" in plan_md
    assert "reports/phase5_continuous_loop_readiness_manifest.json" in plan_md
    assert "GitHub handoff is not required for local read-only Phase 5 preparation" in plan_md
    assert "unattended scheduler enablement remains blocked" in plan_md

    assert "Phase 5 continuous loop readiness manifest" in readme
    assert "reports/phase5_continuous_loop_readiness_manifest.md" in readme
    assert "continuous_loop_enabled=false" in readme
    assert "cron_jobs_created=false" in readme

    triggers = workflow.get("on", workflow.get(True))
    assert "pull_request_target" not in triggers
    assert workflow["permissions"] == {"contents": "read"}
    serialized_workflow = json.dumps(workflow, default=str)
    assert "contents: write" not in serialized_workflow
    assert "pull-requests: write" not in serialized_workflow
    for trigger_name in ("pull_request", "push"):
        assert "evolution/monitor/**" in triggers[trigger_name]["paths"]
        assert "tests/monitor/**" in triggers[trigger_name]["paths"]
        assert "reports/phase5_*" in triggers[trigger_name]["paths"]
    run_blocks = "\n".join(
        str(step.get("run", ""))
        for step in workflow["jobs"]["phase2-tool-description-gate"]["steps"]
        if isinstance(step, dict)
    )
    assert "tests/monitor/test_phase5_continuous_loop_readiness_manifest.py" in run_blocks
    assert manifest["artifacts"] == {
        "manifest_json": "reports/phase5_continuous_loop_readiness_manifest.json",
        "manifest_markdown": "reports/phase5_continuous_loop_readiness_manifest.md",
        "source_plan": "PLAN.md",
        "source_phase4_pr": "https://github.com/NousResearch/hermes-agent-self-evolution/pull/98",
    }

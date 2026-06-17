"""Tests for the Phase 3 real benchmark readiness manifest."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_JSON = REPO_ROOT / "reports" / "phase3_real_benchmark_readiness_manifest.json"
MANIFEST_MD = REPO_ROOT / "reports" / "phase3_real_benchmark_readiness_manifest.md"
EXECUTION_SEED_YAML = REPO_ROOT / "seeds" / "phase3_system_prompt_evolution_execution_seed_draft.yaml"
EXECUTION_DRAFT_JSON = REPO_ROOT / "reports" / "phase3_execution_seed_draft.json"
EXECUTION_DRAFT_MD = REPO_ROOT / "reports" / "phase3_execution_seed_draft.md"
PLAN_MD = REPO_ROOT / "PLAN.md"
README_MD = REPO_ROOT / "README.md"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "phase2-tool-description-gate.yml"

EXPECTED_REQUIRED_INPUTS = {
    "baseline_prompt_artifact",
    "candidate_prompt_artifact",
    "candidate_scaffold_report",
    "dry_run_preflight_report",
    "tblite_fixture_cases",
    "yc_bench_fixture_cases",
    "tblite_runner",
    "yc_bench_runner",
}
EXPECTED_APPROVAL_BOUNDARIES = [
    "running real TBLite benchmark commands",
    "running real YC-Bench benchmark commands",
    "spending nonzero benchmark/API budget",
    "running GEPA/DSPy optimization",
    "editing Hermes Agent prompt source",
    "applying evolved prompt to active runtime",
    "default-gate promotion",
]
PROHIBITED_TARGETS = {
    "~/.hermes/SOUL.md",
    "~/.hermes/config.yaml",
    "~/.hermes/hermes-agent/agent/prompt_builder.py",
    "~/.hermes/skills/",
    "~/.hermes/memories/",
    "~/.hermes/profiles/",
}


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


def test_phase3_real_benchmark_readiness_manifest_is_non_executing_and_blocks_apply():
    manifest = _manifest()

    assert manifest["phase"] == "3"
    assert manifest["mode"] == "phase3-real-benchmark-readiness-manifest"
    assert manifest["status"] == "recorded_not_executed"
    assert manifest["source_authorization"] == "rec action GO"
    assert manifest["source_preflight_gate"] == "reports/phase3_execution_seed_draft.json"
    assert manifest["manifest_version"] == "phase3-real-benchmark-readiness-v1"

    assert manifest["candidate_only"] is True
    assert manifest["execution_started"] is False
    assert manifest["run_gepa_now"] is False
    assert manifest["run_dspy_now"] is False
    assert manifest["real_benchmarks_executed"] is False
    assert manifest["real_benchmark_execution_approved"] is False
    assert manifest["mutate_active_system_prompt_now"] is False
    assert manifest["active_system_prompt_apply_approved"] is False
    assert manifest["apply_ready"] is False
    assert manifest["phase3_execution_ready"] is False

    approval = manifest["approval_gate"]
    assert approval["planning_authorized"] == "rec action GO"
    assert approval["real_benchmark_execution_approved"] is False
    assert approval["optimizer_execution_approved"] is False
    assert approval["active_apply_approved"] is False
    assert approval["separate_approval_required_before"] == EXPECTED_APPROVAL_BOUNDARIES



def test_phase3_real_benchmark_readiness_manifest_records_inputs_environment_cost_and_rollback():
    manifest = _manifest()

    assert set(manifest["required_inputs"]) == EXPECTED_REQUIRED_INPUTS
    for key, item in manifest["required_inputs"].items():
        assert item["required_before"] == "real_benchmark_execution"
        assert item["status"] in {"available_dry_run_fixture", "required_future_real_input", "required_before_execution"}
        assert isinstance(item["path"], str) and item["path"]
        assert not Path(item["path"]).is_absolute(), key

    environment = manifest["environment_requirements"]
    assert environment["python"]["min_version"] == "3.10"
    assert environment["self_evolution_package"]["install_mode"] == "editable_with_dev_dependencies"
    assert environment["hermes_agent_repo"]["env_var"] == "HERMES_AGENT_REPO"
    assert environment["hermes_agent_repo"]["required"] is True
    assert environment["credential_policy"] == {
        "raw_credentials_must_not_be_recorded": True,
        "llm_or_benchmark_credentials_required_only_after_explicit_approval": True,
    }

    transition = manifest["real_benchmark_transition"]
    assert transition["current_adapter_status"] == "dry_run_only_real_mode_not_implemented"
    assert transition["external_calls_allowed_now"] is False
    assert transition["network_allowed_now"] is False
    assert transition["real_mode_templates_runnable_now"] is False
    assert transition["real_results_required_before"] == [
        "phase3_execution",
        "system_prompt_evolution_acceptance",
        "active_system_prompt_apply",
        "default_gate_promotion",
    ]
    for template in transition["real_command_templates"]:
        assert template["status"] == "template_only_not_run"
        assert template["runnable_now"] is False
        assert "--dry-run" not in template["command"]
        assert template["output_json"].startswith("output/phase3-system-prompt/<run-id>/benchmarks/")

    cost = manifest["cost_and_runtime_limits"]
    assert cost["current_authorized_spend_usd"] == 0
    assert cost["proposed_initial_real_benchmark_budget_usd"] <= 25
    assert cost["hard_stop_total_budget_usd"] <= 50
    assert cost["max_wall_clock_hours"] <= 8
    assert cost["abort_on_budget_exhaustion"] is True
    assert cost["requires_reapproval_above_usd"] == cost["proposed_initial_real_benchmark_budget_usd"]

    rollback = manifest["rollback_requirements"]
    assert rollback["checkpoint_required_before_real_benchmark"] is True
    assert rollback["baseline_prompt_snapshot_required"] is True
    assert rollback["baseline_prompt_checksum_required"] is True
    assert rollback["git_status_snapshot_required"] is True
    assert rollback["rollback_handle_required_before_apply"] is True
    assert rollback["active_runtime_rollback_required_before_apply"] is True
    assert rollback["allowed_write_roots_before_apply"] == [
        "output/phase3-system-prompt/<run-id>/",
        "reports/phase3_real_benchmark_readiness_manifest.json",
        "reports/phase3_real_benchmark_readiness_manifest.md",
    ]
    assert set(rollback["prohibited_targets_before_separate_apply_approval"]) == PROHIBITED_TARGETS



def test_phase3_real_benchmark_readiness_manifest_is_privacy_safe_and_fail_closed():
    manifest = _manifest()

    strings = list(_all_strings(manifest))
    forbidden_fragments = ["/Users/snw", "state.db", "session_id", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]
    for fragment in forbidden_fragments:
        assert all(fragment not in item for item in strings), fragment

    conditions = manifest["go_no_go_conditions"]
    condition_ids = {condition["id"] for condition in conditions}
    assert condition_ids == {
        "RBM-1-input-artifacts-and-checksums",
        "RBM-2-real-mode-runner-implemented-or-pinned",
        "RBM-3-explicit-approval-and-budget",
        "RBM-4-fresh-output-root",
        "RBM-5-rollback-handle",
        "RBM-6-candidate-only-no-apply",
    }
    for condition in conditions:
        assert condition["required_before"] == "real_benchmark_execution"
        assert condition["status"] in {"required", "satisfied_by_existing_contract"}
    assert manifest["ready_state"] == {
        "real_benchmark_ready_now": False,
        "blocked_until_all_go_no_go_conditions_satisfied": True,
        "active_apply_ready_now": False,
    }



def test_phase3_real_benchmark_readiness_manifest_is_linked_from_seed_docs_and_ci():
    manifest = _manifest()
    execution_seed = yaml.safe_load(EXECUTION_SEED_YAML.read_text())
    execution_report = json.loads(EXECUTION_DRAFT_JSON.read_text())
    execution_md = EXECUTION_DRAFT_MD.read_text()
    manifest_md = MANIFEST_MD.read_text()
    plan_md = PLAN_MD.read_text()
    readme = README_MD.read_text()
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())

    assert execution_seed["benchmark_gate"]["real_benchmark_readiness_manifest"] == {
        "status": "recorded_not_executed",
        "manifest_json": "reports/phase3_real_benchmark_readiness_manifest.json",
        "manifest_markdown": "reports/phase3_real_benchmark_readiness_manifest.md",
        "real_benchmark_ready_now": False,
        "active_apply_ready_now": False,
        "verified_by": "tests/tools/test_phase3_real_benchmark_readiness_manifest.py",
    }
    assert execution_report["benchmark_gate"]["real_benchmark_readiness_manifest"] == execution_seed["benchmark_gate"]["real_benchmark_readiness_manifest"]
    assert "reports/phase3_real_benchmark_readiness_manifest.json" in execution_seed["deliverables"]
    assert "reports/phase3_real_benchmark_readiness_manifest.md" in execution_report["deliverables"]

    assert "# Phase 3 Real Benchmark Readiness Manifest" in manifest_md
    assert "Status: recorded, not executed" in manifest_md
    assert "current authorized spend is `$0`" in manifest_md
    assert "real_benchmark_ready_now=false" in manifest_md
    assert "active_apply_ready_now=false" in manifest_md
    assert "Phase 3 real benchmark readiness manifest" in execution_md
    assert "reports/phase3_real_benchmark_readiness_manifest.json" in execution_md
    assert "Phase 3 Real Benchmark Readiness Manifest" in readme
    assert "real_benchmark_ready_now=false" in readme
    assert "Phase 3 real benchmark readiness manifest" in plan_md
    assert "real benchmarks remain blocked until all go/no-go conditions are satisfied" in plan_md

    triggers = workflow.get("on", workflow.get(True))
    for trigger_name in ("pull_request", "push"):
        assert "reports/phase3_*" in triggers[trigger_name]["paths"]
    run_blocks = "\n".join(
        str(step.get("run", ""))
        for step in workflow["jobs"]["phase2-tool-description-gate"]["steps"]
        if isinstance(step, dict)
    )
    assert "tests/tools/test_phase3_real_benchmark_readiness_manifest.py" in run_blocks
    assert manifest["artifacts"] == {
        "manifest_json": "reports/phase3_real_benchmark_readiness_manifest.json",
        "manifest_markdown": "reports/phase3_real_benchmark_readiness_manifest.md",
        "source_execution_seed": "seeds/phase3_system_prompt_evolution_execution_seed_draft.yaml",
        "source_execution_report": "reports/phase3_execution_seed_draft.json",
    }

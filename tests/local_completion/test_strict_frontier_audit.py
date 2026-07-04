"""Tests for HSE strict frontier audit reports."""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

from evolution.local_completion import strict_frontier_audit
from evolution.local_completion.strict_frontier_audit import (
    CURRENT_BASELINE_REVALIDATION_REQUIRED,
    PHASE_2_STRICT_COMPLETE,
    write_strict_frontier_audit,
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _init_active_hermes_repo(repo: Path) -> dict[str, str]:
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], text=True, capture_output=True, check=True)
    _git(repo, "config", "user.email", "frontier@example.invalid")
    _git(repo, "config", "user.name", "Frontier Test")
    (repo / "tools").mkdir()
    (repo / "agent").mkdir()
    (repo / "tools" / "tool_description_overrides.py").write_text("OVERRIDES = {'read_file': 'read files'}\n")
    (repo / "model_tools.py").write_text("MODEL_TOOLS = ['read_file']\n")
    (repo / "tools" / "registry.py").write_text("REGISTRY = ['read_file']\n")
    (repo / "agent" / "prompt_builder.py").write_text("DEFAULT_AGENT_IDENTITY = 'baseline'\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "phase2 active subject")
    return _active_metadata(repo)


def _active_metadata(repo: Path) -> dict[str, str]:
    head = _git(repo, "rev-parse", "HEAD")
    return {
        "commit_full": head,
        "commit": head[:9],
        "override_sha": _sha(repo / "tools" / "tool_description_overrides.py") if (repo / "tools" / "tool_description_overrides.py").exists() else "",
        "model_tools_sha": _sha(repo / "model_tools.py"),
        "registry_sha": _sha(repo / "tools" / "registry.py"),
    }


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def _fixture_inputs(tmp_path: Path, active_repo: Path, subject: dict[str, str]) -> dict[str, Path]:
    closure = _write_json(
        tmp_path / "reports" / "closure.json",
        {
            "schema_version": "hse-benchmark-strict-plan-closure-v1",
            "status": "STRICT_PLAN_BENCHMARK_GATE_CLOSED",
            "strict_plan_gate_closed": True,
            "benchmark_gate_passed": True,
            "strict_plan_scope": "phase1_phase2_benchmark_regression_gate",
            "local_smoke_benchmark_accepted_for_this_gate": True,
            "full_remote_benchmark_executed": False,
            "blocked_by": [],
            "closed_criteria": {
                "artifact_hashes_verified": True,
                "all_suites_passed": True,
                "phase2_case_count_satisfies_plan": True,
                "no_forbidden_side_effects": True,
            },
            "benchmark_subjects": {
                "current": {
                    "subject_id": "current-post-phase1-phase2-local-active",
                    "hermes_source": {
                        "repo": str(active_repo),
                        "commit": subject["commit"],
                        "commit_full": subject["commit_full"],
                    },
                    "tool_descriptions": {
                        "active_apply_commit": subject["commit"],
                        "override_module": {"path": str(active_repo / "tools" / "tool_description_overrides.py"), "sha256": subject["override_sha"]},
                        "model_tools_module": {"path": str(active_repo / "model_tools.py"), "sha256": subject["model_tools_sha"]},
                        "registry_module": {"path": str(active_repo / "tools" / "registry.py"), "sha256": subject["registry_sha"]},
                        "model_tools_readback_passed": True,
                        "raw_registry_readback_passed": True,
                        "semantic_loss_guard_passed": True,
                    },
                }
            },
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "github_query_performed": False,
            "github_write_performed": False,
            "active_apply_performed": False,
        },
    )
    phase2_active = _write_json(
        tmp_path / "reports" / "phase2_active.json",
        {"schema_version": "hse-phase2-active-schema-apply-closeout-v1", "verdict": "PASS_ACTIVE_SCHEMA_APPLY_LOCAL", "phase2d_gate_passed": True, "model_tools_readback_passed": True, "raw_registry_readback_passed": True, "semantic_loss_guard_passed": True},
    )
    post_phase2 = _write_json(
        tmp_path / "reports" / "post_phase2.json",
        {"schema_version": "hse-strict-audit-completion-design-post-phase2-active-apply-v1", "strict_verdict": "LOCAL_P1_P2_ACTIVE_APPLIED__STRICT_BENCHMARK_GATE_OPEN", "no_github_write_preserved": True},
    )
    phase2_review = _write_json(
        tmp_path / "reports" / "phase2_review.json",
        {"phase": "2E", "phase2e_closeout_complete": True, "remaining_phase2_closeout_items": [], "candidate_only": True, "active_schema_or_source_apply_requires_separate_approval": True},
    )
    phase3_plan = _write_json(
        tmp_path / "reports" / "phase3_plan.json",
        {"phase": "3", "status": "planned_not_executed", "plan_only": True, "execution_started": False, "apply_ready": False},
    )
    phase3_readiness = _write_json(
        tmp_path / "reports" / "phase3_readiness.json",
        {"phase": "3", "status": "recorded_not_executed", "real_benchmarks_executed": False, "active_system_prompt_apply_approved": False, "ready_state": {"real_benchmark_ready_now": False, "active_apply_ready_now": False}},
    )
    phase3_historical = _write_json(
        tmp_path / "reports" / "phase3_historical.json",
        {"phase": "3", "status": "completed_with_local_active_source_apply_and_bounded_smoke_validation", "full_external_benchmark_executed": False, "active_source_apply": {"commit_short": "65a7925aa"}},
    )
    phase4 = _write_json(
        tmp_path / "reports" / "phase4.json",
        {"phase": "4", "status": "completed_local_verified", "formal_gate_assessment": {"phase4_local_completion": True, "benchmarks_hold": True}, "safety_boundaries": {"darwinian_cli_invoked": False}},
    )
    phase5_readiness = _write_json(
        tmp_path / "reports" / "phase5_readiness.json",
        {"phase": "5", "status": "local_preparation_started_not_deployed", "continuous_loop_enabled": False, "cron_jobs_created": False, "ready_state": {"phase5_unattended_loop_ready_now": False}},
    )
    phase5_formal = _write_json(
        tmp_path / "reports" / "phase5_formal.json",
        {"phase": "5", "status": "FORMAL_PHASE5_COMPLETE_LOCAL_WITH_EXPLICIT_WAIVER", "formal_phase5_completion_claimed": True, "production_continuous_loop_enabled": False, "safety_invariants": {"optimizer_execution_started": False, "cron_jobs_created": False}},
    )
    plan = tmp_path / "PLAN.md"
    plan.write_text("| **Phase 1** | Skill evolution | | | ≥1 skill measurably improved, no benchmark regression |\n| **Phase 2** | Tool descriptions | | | Tool selection accuracy improved, no benchmark regression |\n| **Phase 3** | System prompt | | | Behavioral tests pass, benchmarks hold or improve |\n| **Phase 4** | Code evolution | | | Bugs fixed, tests pass, benchmarks hold |\n| **Phase 5** | Continuous loop | | | Automated pipeline runs unattended |\n")
    return {
        "benchmark_closure_path": closure,
        "phase2_active_apply_path": phase2_active,
        "post_phase2_audit_path": post_phase2,
        "phase2_review_path": phase2_review,
        "phase3_plan_path": phase3_plan,
        "phase3_readiness_path": phase3_readiness,
        "phase3_historical_path": phase3_historical,
        "phase4_completion_path": phase4,
        "phase5_readiness_path": phase5_readiness,
        "phase5_formal_path": phase5_formal,
        "plan_path": plan,
    }


def _phase3_integrated_chain_inputs(tmp_path: Path) -> dict[str, Path]:
    local_real_smoke = _write_json(
        tmp_path / "reports" / "phase3_local_real_smoke.json",
        {
            "schema_version": "hse-phase3-local-real-smoke-execution-v1",
            "status": "PHASE3_LOCAL_REAL_SMOKE_EXECUTION_PASSED_SEPARATE_GEPA_DSPY_APPROVAL_STILL_REQUIRED",
            "local_real_smoke_passed": True,
            "decision": {
                "phase3_execution_ready_now": False,
                "active_apply_ready_now": False,
                "do_not_claim": ["phase3_strict_completion", "overall_HSE_project_completion"],
            },
        },
    )
    gepa_execution = _write_json(
        tmp_path / "reports" / "phase3_gepa_execution.json",
        {
            "schema_version": "hse-phase3-gepa-dspy-candidate-optimization-execution-v1",
            "status": "PHASE3_GEPA_DSPY_CANDIDATE_OPTIMIZATION_EXECUTION_PASSED_NO_ACTIVE_APPLY",
            "passed": True,
            "boundary_ledger": {
                "bounded_local_dspy_gepa_optimizer_executed": True,
                "candidate_optimization_command_executed": True,
                "external_llm_calls_performed": False,
                "github_query_performed": False,
                "github_write_performed": False,
                "provider_or_model_spend_performed": False,
                "network_calls_performed": False,
                "active_apply_performed": False,
                "cron_or_gateway_mutation_performed": False,
                "deploy_or_publication_performed": False,
                "phase3_strict_completion_claimed": False,
                "overall_hse_project_completion_claimed": False,
            },
            "decision": {"phase3_strict_completion_ready_to_claim": False, "active_apply_performed": False},
        },
    )
    noop_apply_closure = _write_json(
        tmp_path / "reports" / "phase3_noop_apply_closure.json",
        {
            "schema_version": "hse-phase3-noop-apply-closure-reconciliation-v1",
            "status": "PHASE3_NOOP_APPLY_CLOSURE_RECONCILED_STRICT_FRONTIER_RECHECK_PREPARED_NOT_EXECUTED",
            "reconciliation_passed": True,
            "boundary_ledger": {
                "github_query_performed": False,
                "github_write_performed": False,
                "provider_or_model_spend_performed": False,
                "network_calls_performed": False,
                "active_apply_performed": False,
                "active_runtime_mutation_performed": False,
                "cron_or_gateway_mutation_performed": False,
                "deploy_or_publication_performed": False,
                "phase3_strict_completion_claimed": False,
                "overall_hse_project_completion_claimed": False,
                "strict_frontier_recheck_executed": False,
                "strict_frontier_recheck_prepared": True,
                "apply_lane_closed_no_active_write_required": True,
            },
            "closure_reconciliation": {
                "apply_lane_closed": True,
                "apply_lane_status": "NO_ACTIVE_WRITE_REQUIRED",
                "semantic_noop_confirmed": True,
                "active_apply_needed": False,
                "active_apply_recommended": False,
                "active_apply_performed": False,
                "active_runtime_mutation_performed": False,
            },
            "decision": {"phase3_strict_completion_ready_to_claim": False, "active_apply_performed": False},
        },
    )
    post_noop_recheck = _write_json(
        tmp_path / "reports" / "phase3_post_noop_recheck.json",
        {
            "schema_version": "hse-phase3-post-noop-apply-strict-frontier-recheck-execution-v1",
            "status": "PHASE3_POST_NOOP_APPLY_STRICT_FRONTIER_RECHECK_EXECUTED_FAIL_CLOSED_PHASE2_FRONTIER_CONFIRMED",
            "recheck_passed": True,
            "boundary_ledger": {
                "github_query_performed": False,
                "github_write_performed": False,
                "provider_or_model_spend_performed": False,
                "network_calls_performed": False,
                "active_apply_performed": False,
                "active_runtime_mutation_performed": False,
                "cron_or_gateway_mutation_performed": False,
                "deploy_or_publication_performed": False,
                "phase3_strict_completion_claimed": False,
                "overall_hse_project_completion_claimed": False,
            },
            "decision": {
                "current_active_frontier_confirmed": PHASE_2_STRICT_COMPLETE,
                "phase3_strict_complete": False,
                "phase3_strict_completion_ready_to_claim": False,
            },
        },
    )
    return {
        "phase3_local_real_smoke_path": local_real_smoke,
        "phase3_gepa_execution_path": gepa_execution,
        "phase3_noop_apply_closure_path": noop_apply_closure,
        "phase3_post_noop_recheck_path": post_noop_recheck,
    }


def _run_audit(tmp_path: Path, active_repo: Path, inputs: dict[str, Path]) -> dict:
    result = write_strict_frontier_audit(
        active_hermes_repo=active_repo,
        output_dir=tmp_path / "audit",
        generated_at="2026-07-04T02:40:00+09:00",
        **inputs,
    )
    report = json.loads(Path(result["report_path"]).read_text())
    assert Path(result["markdown_path"]).exists()
    return report


def _audit_cli_args(tmp_path: Path, active_repo: Path, inputs: dict[str, Path]) -> list[str]:
    return [
        "--active-hermes-repo",
        str(active_repo),
        "--benchmark-closure",
        str(inputs["benchmark_closure_path"]),
        "--phase2-active-apply",
        str(inputs["phase2_active_apply_path"]),
        "--post-phase2-audit",
        str(inputs["post_phase2_audit_path"]),
        "--phase2-review",
        str(inputs["phase2_review_path"]),
        "--phase3-plan",
        str(inputs["phase3_plan_path"]),
        "--phase3-readiness",
        str(inputs["phase3_readiness_path"]),
        "--phase3-historical",
        str(inputs["phase3_historical_path"]),
        "--phase3-local-real-smoke",
        str(inputs["phase3_local_real_smoke_path"]),
        "--phase3-gepa-execution",
        str(inputs["phase3_gepa_execution_path"]),
        "--phase3-noop-apply-closure",
        str(inputs["phase3_noop_apply_closure_path"]),
        "--phase3-post-noop-recheck",
        str(inputs["phase3_post_noop_recheck_path"]),
        "--phase4-completion",
        str(inputs["phase4_completion_path"]),
        "--phase5-readiness",
        str(inputs["phase5_readiness_path"]),
        "--phase5-formal",
        str(inputs["phase5_formal_path"]),
        "--plan",
        str(inputs["plan_path"]),
        "--output-dir",
        str(tmp_path / "audit-cli"),
        "--generated-at",
        "2026-07-04T02:41:00+09:00",
    ]


def test_strict_frontier_audit_marks_phase2_current_complete_when_active_matches_closure_subject(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    subject = _init_active_hermes_repo(active_repo)
    inputs = _fixture_inputs(tmp_path, active_repo, subject)

    report = _run_audit(tmp_path, active_repo, inputs)

    assert report["schema_version"] == "hse-strict-frontier-audit-v1"
    assert report["recorded_subject_frontier"]["highest_strict_complete_phase"] == 2
    assert report["recorded_subject_frontier"]["status"] == PHASE_2_STRICT_COMPLETE
    assert report["current_active_frontier"]["highest_strict_complete_phase"] == 2
    assert report["current_active_frontier"]["status"] == PHASE_2_STRICT_COMPLETE
    assert report["current_baseline_match"]["matches_closure_subject"] is True
    assert report["phases"]["phase1"]["strict_status"] == "STRICT_COMPLETE_CURRENT_ACTIVE"
    assert report["phases"]["phase2"]["strict_status"] == "STRICT_COMPLETE_CURRENT_ACTIVE"
    assert report["phases"]["phase3"]["strict_status"] == "NOT_STRICT_COMPLETE_PREPARATION_ONLY"
    assert report["phases"]["phase5"]["strict_status"] == "NOT_STRICT_COMPLETE_LOCAL_OR_WAIVED_ONLY"
    assert "overall_HSE_project_completion" in report["not_claimed"]
    assert report["github_query_performed"] is False
    assert report["github_write_performed"] is False


def test_strict_frontier_audit_requires_revalidation_when_current_active_hermes_drifted(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    subject = _init_active_hermes_repo(active_repo)
    inputs = _fixture_inputs(tmp_path, active_repo, subject)

    (active_repo / "tools" / "tool_description_overrides.py").unlink()
    (active_repo / "model_tools.py").write_text("MODEL_TOOLS = ['new_current']\n")
    (active_repo / "tools" / "registry.py").write_text("REGISTRY = ['new_current']\n")
    _git(active_repo, "add", "-A")
    _git(active_repo, "commit", "-m", "current active drift")

    report = _run_audit(tmp_path, active_repo, inputs)

    assert report["recorded_subject_frontier"]["status"] == PHASE_2_STRICT_COMPLETE
    assert report["recorded_subject_frontier"]["highest_strict_complete_phase"] == 2
    assert report["current_active_frontier"]["status"] == CURRENT_BASELINE_REVALIDATION_REQUIRED
    assert report["current_active_frontier"]["highest_strict_complete_phase"] == 0
    assert report["current_baseline_match"]["matches_closure_subject"] is False
    assert report["current_baseline_match"]["active_tool_description_hashes_match"] is False
    assert "active_tool_description_hash_mismatch" in report["current_active_frontier"]["blockers"]
    assert "current_baseline_revalidation_required_before_phase1_phase2_strict_claim" in report["recommended_next_action"]
    assert report["phases"]["phase1"]["strict_status"] == "REVALIDATION_REQUIRED_CURRENT_BASELINE_MISMATCH"
    assert report["phases"]["phase2"]["strict_status"] == "REVALIDATION_REQUIRED_CURRENT_BASELINE_MISMATCH"


def test_strict_frontier_audit_does_not_accept_historical_phase3_or_phase5_completion_claims(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    subject = _init_active_hermes_repo(active_repo)
    inputs = _fixture_inputs(tmp_path, active_repo, subject)

    report = _run_audit(tmp_path, active_repo, inputs)

    assert report["phases"]["phase3"]["historical_claim_status"] == "completed_with_local_active_source_apply_and_bounded_smoke_validation"
    assert report["phases"]["phase3"]["strict_complete"] is False
    assert "phase3_current_plan_status_planned_not_executed" in report["phases"]["phase3"]["blockers"]
    assert report["phases"]["phase3"]["integrated_chain"]["available"] is False
    assert report["phases"]["phase4"]["strict_complete"] is False
    assert "phase4_blocked_until_phase3_strict_complete_current" in report["phases"]["phase4"]["blockers"]
    assert report["phases"]["phase5"]["historical_claim_status"] == "FORMAL_PHASE5_COMPLETE_LOCAL_WITH_EXPLICIT_WAIVER"
    assert report["phases"]["phase5"]["strict_complete"] is False
    assert "production_continuous_loop_not_enabled" in report["phases"]["phase5"]["blockers"]


def test_strict_frontier_audit_accepts_phase3_integrated_chain_when_current_phase2_and_noop_apply_closure_pass(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    subject = _init_active_hermes_repo(active_repo)
    inputs = _fixture_inputs(tmp_path, active_repo, subject)
    inputs.update(_phase3_integrated_chain_inputs(tmp_path))

    report = _run_audit(tmp_path, active_repo, inputs)

    phase3 = report["phases"]["phase3"]
    assert phase3["strict_complete"] is True
    assert phase3["strict_status"] == "STRICT_COMPLETE_CURRENT_ACTIVE"
    assert phase3["integrated_chain"]["available"] is True
    assert phase3["integrated_chain"]["complete"] is True
    assert phase3["integrated_chain"]["checks"]["semantic_noop_apply_closure_satisfies_active_write_gate"] is True
    assert "phase3_current_plan_status_planned_not_executed" not in phase3["blockers"]
    assert "phase3_active_apply_not_approved_current_readiness" not in phase3["blockers"]
    assert report["source_artifacts"]["phase3_noop_apply_closure"]["sha256"] == _sha(inputs["phase3_noop_apply_closure_path"])
    assert report["phase3_strict_completion_claimed"] is False
    assert report["github_query_performed"] is False
    assert report["github_write_performed"] is False
    assert report["active_apply_performed"] is False
    assert report["overall_hse_project_completion_claimed"] is False


def test_strict_frontier_audit_keeps_phase3_fail_closed_when_integrated_chain_missing(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    subject = _init_active_hermes_repo(active_repo)
    inputs = _fixture_inputs(tmp_path, active_repo, subject)

    report = _run_audit(tmp_path, active_repo, inputs)

    phase3 = report["phases"]["phase3"]
    assert phase3["strict_complete"] is False
    assert phase3["integrated_chain"]["available"] is False
    assert phase3["integrated_chain"]["complete"] is False
    assert "phase3_current_plan_status_planned_not_executed" in phase3["blockers"]


def test_strict_frontier_audit_rejects_phase3_integrated_chain_with_forbidden_side_effects(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    subject = _init_active_hermes_repo(active_repo)
    inputs = _fixture_inputs(tmp_path, active_repo, subject)
    inputs.update(_phase3_integrated_chain_inputs(tmp_path))
    gepa_payload = json.loads(inputs["phase3_gepa_execution_path"].read_text())
    gepa_payload["boundary_ledger"]["github_write_performed"] = True
    _write_json(inputs["phase3_gepa_execution_path"], gepa_payload)

    report = _run_audit(tmp_path, active_repo, inputs)

    phase3 = report["phases"]["phase3"]
    assert phase3["strict_complete"] is False
    assert phase3["integrated_chain"]["available"] is True
    assert phase3["integrated_chain"]["complete"] is False
    assert "phase3_integrated_chain_forbidden_boundary_gepa_execution.boundary_ledger.github_write_performed" in phase3["blockers"]


def test_strict_frontier_audit_cli_accepts_phase3_integrated_chain_flags(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    subject = _init_active_hermes_repo(active_repo)
    inputs = _fixture_inputs(tmp_path, active_repo, subject)
    inputs.update(_phase3_integrated_chain_inputs(tmp_path))

    assert strict_frontier_audit.main(_audit_cli_args(tmp_path, active_repo, inputs)) == 0
    report = json.loads((tmp_path / "audit-cli" / "strict_frontier_audit.json").read_text())

    assert report["phases"]["phase3"]["strict_complete"] is True
    assert report["phases"]["phase3"]["integrated_chain"]["complete"] is True
    assert report["phase3_strict_completion_claimed"] is False


def _current_baseline_closure_inputs(tmp_path: Path, active_repo: Path, subject: dict[str, str]) -> dict[str, Path]:
    inputs = _fixture_inputs(tmp_path, active_repo, subject)
    preflight = _write_json(
        tmp_path / "reports" / "current_baseline_preflight.json",
        {
            "schema_version": "hse-current-baseline-revalidation-preflight-v1",
            "status": "CURRENT_BASELINE_REVALIDATION_PREFLIGHT_READY_SEPARATE_GO_REQUIRED",
            "baseline_commit_for_rerun": "9b50c56556f902b62ecc4a7e2e511ca0f316da2d",
            "current_commit_for_rerun": subject["commit_full"],
            "current_baseline_inventory": {
                "head": subject["commit_full"],
                "head_short": subject["commit"],
                "files": [
                    {"relative_path": "tools/tool_description_overrides.py", "exists": True, "is_file": True, "sha256": subject["override_sha"]},
                    {"relative_path": "model_tools.py", "exists": True, "is_file": True, "sha256": subject["model_tools_sha"]},
                    {"relative_path": "tools/registry.py", "exists": True, "is_file": True, "sha256": subject["registry_sha"]},
                ],
            },
        },
    )
    closure = _write_json(
        tmp_path / "reports" / "current_baseline_closure.json",
        {
            "schema_version": "hse-benchmark-strict-plan-closure-v1",
            "status": "STRICT_PLAN_BENCHMARK_GATE_CLOSED",
            "strict_plan_gate_closed": True,
            "benchmark_gate_passed": True,
            "strict_plan_scope": "phase1_phase2_benchmark_regression_gate",
            "current_baseline_revalidation_closed": True,
            "local_smoke_benchmark_accepted_for_this_gate": True,
            "full_remote_benchmark_executed": False,
            "blocked_by": [],
            "closed_criteria": {
                "artifact_hashes_verified": True,
                "all_suites_passed": True,
                "phase2_case_count_satisfies_plan": True,
                "no_forbidden_side_effects": True,
                "current_baseline_preflight_ready": True,
                "current_baseline_execution_verification_passed": True,
                "current_baseline_execution_summary_passed": True,
                "subject_commits_align": True,
            },
            "benchmark_subjects": {
                "baseline": {
                    "subject_id": "recorded-phase1-phase2-closure-subject",
                    "hermes_source": {
                        "commit": "9b50c56556f902b62ecc4a7e2e511ca0f316da2d",
                        "commit_full": "9b50c56556f902b62ecc4a7e2e511ca0f316da2d",
                    },
                },
                "current": {
                    "subject_id": "current-active-hermes-baseline-revalidated",
                    "hermes_source": {"commit": subject["commit_full"], "commit_full": subject["commit_full"]},
                },
            },
            "source_artifacts": {
                "current_baseline_preflight": {"path": str(preflight), "sha256": _sha(preflight), "bytes": preflight.stat().st_size}
            },
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "github_query_performed": False,
            "github_write_performed": False,
            "active_apply_performed": False,
            "cron_or_gateway_mutation_performed": False,
        },
    )
    inputs["benchmark_closure_path"] = closure
    return inputs


def test_strict_frontier_audit_accepts_current_baseline_revalidation_closure_when_inventory_matches(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    subject = _init_active_hermes_repo(active_repo)
    inputs = _current_baseline_closure_inputs(tmp_path, active_repo, subject)

    report = _run_audit(tmp_path, active_repo, inputs)

    assert report["recorded_subject_frontier"]["status"] == PHASE_2_STRICT_COMPLETE
    assert report["current_active_frontier"]["status"] == PHASE_2_STRICT_COMPLETE
    assert report["current_active_frontier"]["highest_strict_complete_phase"] == 2
    assert report["current_baseline_match"]["matches_closure_subject"] is True
    assert report["current_baseline_match"]["current_baseline_revalidation_closure"] is True
    assert report["current_baseline_match"]["active_head_equals_revalidated_current_subject"] is True
    assert report["current_baseline_match"]["active_tool_description_hashes_match"] is True
    assert report["phases"]["phase1"]["strict_status"] == "STRICT_COMPLETE_CURRENT_ACTIVE"
    assert report["phases"]["phase2"]["strict_status"] == "STRICT_COMPLETE_CURRENT_ACTIVE"


def test_strict_frontier_audit_blocks_current_baseline_revalidation_closure_when_inventory_hash_drifts(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    subject = _init_active_hermes_repo(active_repo)
    inputs = _current_baseline_closure_inputs(tmp_path, active_repo, subject)
    (active_repo / "model_tools.py").write_text("MODEL_TOOLS = ['drift-after-revalidation']\n")

    report = _run_audit(tmp_path, active_repo, inputs)

    assert report["recorded_subject_frontier"]["status"] == PHASE_2_STRICT_COMPLETE
    assert report["current_active_frontier"]["status"] == CURRENT_BASELINE_REVALIDATION_REQUIRED
    assert report["current_baseline_match"]["matches_closure_subject"] is False
    assert report["current_baseline_match"]["active_head_equals_revalidated_current_subject"] is True
    assert report["current_baseline_match"]["active_tool_description_hashes_match"] is False
    assert "active_tool_description_hash_mismatch" in report["current_active_frontier"]["blockers"]

"""Tests for HSE strict frontier audit reports."""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

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
    assert report["phases"]["phase4"]["strict_complete"] is False
    assert "phase4_blocked_until_phase3_strict_complete_current" in report["phases"]["phase4"]["blockers"]
    assert report["phases"]["phase5"]["historical_claim_status"] == "FORMAL_PHASE5_COMPLETE_LOCAL_WITH_EXPLICIT_WAIVER"
    assert report["phases"]["phase5"]["strict_complete"] is False
    assert "production_continuous_loop_not_enabled" in report["phases"]["phase5"]["blockers"]


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

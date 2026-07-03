"""Strict PLAN benchmark-gate closure reconciliation for HSE local evidence.

This module consumes the previously fail-closed B0 benchmark backfill plus the
approved local-only real-benchmark-smoke result artifacts. It may close only the
Phase 1/2 benchmark-regression strict PLAN gate when all local evidence is
hash-verified and no side-effect boundary was violated.

It does not run benchmarks, query/write GitHub, perform active apply, call
network/provider APIs, or claim full remote/service benchmark completion.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags

STRICT_PLAN_CLOSURE_SCHEMA_VERSION = "hse-benchmark-strict-plan-closure-v1"
STRICT_PLAN_CLOSURE_GATE_ID = "B0-CLOSE"
STRICT_PLAN_CLOSURE_PHASE = "Phase 1/2 Benchmark Strict PLAN Closure Reconciliation"
STRICT_PLAN_CLOSURE_TARGET = "strict-plan-benchmark-regression-gate"
STRICT_PLAN_SCOPE = "phase1_phase2_benchmark_regression_gate"
STRICT_PLAN_BENCHMARK_GATE_CLOSED = "STRICT_PLAN_BENCHMARK_GATE_CLOSED"
BLOCKED_BENCHMARK_EVIDENCE_INCOMPLETE = "BLOCKED_BENCHMARK_EVIDENCE_INCOMPLETE"
REQUIRED_SUITES = ("TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples")
_SIDE_EFFECT_FLAGS = (
    "provider_or_model_spend_performed",
    "network_calls_performed",
    "github_write_performed",
    "active_apply_performed",
)


def write_benchmark_strict_plan_closure(
    *,
    benchmark_backfill_path: str | Path,
    approval_packet_path: str | Path,
    preflight_report_path: str | Path,
    readiness_report_path: str | Path,
    execution_verification_path: str | Path,
    execution_summary_path: str | Path,
    plan_path: str | Path,
    output_dir: str | Path,
    generated_at: str,
) -> dict[str, str]:
    """Write the strict PLAN benchmark-gate closure reconciliation artifact."""

    _require_non_empty("generated_at", generated_at)
    backfill_path = Path(benchmark_backfill_path).expanduser()
    approval_path = Path(approval_packet_path).expanduser()
    preflight_path = Path(preflight_report_path).expanduser()
    readiness_path = Path(readiness_report_path).expanduser()
    verification_path = Path(execution_verification_path).expanduser()
    summary_path = Path(execution_summary_path).expanduser()
    plan = Path(plan_path).expanduser()

    backfill = _load_json_object(backfill_path, "benchmark backfill")
    approval = _load_json_object(approval_path, "approval packet")
    preflight = _load_json_object(preflight_path, "preflight report")
    readiness = _load_json_object(readiness_path, "execution readiness report")
    verification = _load_json_object(verification_path, "execution verification report")
    execution_summary = _load_json_object(summary_path, "execution summary")
    plan_text = plan.read_text()
    phase2_min_case_count = _phase2_min_case_count(plan_text)

    artifact_checks = _verify_report_hashes(verification)
    suite_results = _suite_results_from_verification(verification)
    blocked_by: list[str] = []
    closed_criteria: dict[str, bool] = {}

    closed_criteria["approval_complete"] = _approval_complete(approval)
    if not closed_criteria["approval_complete"]:
        blocked_by.append("approval_not_complete")

    closed_criteria["preflight_ready"] = _preflight_ready(preflight)
    if not closed_criteria["preflight_ready"]:
        blocked_by.append("preflight_not_ready")

    closed_criteria["readiness_go"] = _readiness_go(readiness)
    if not closed_criteria["readiness_go"]:
        blocked_by.append("readiness_not_go")

    closed_criteria["execution_verification_passed"] = _execution_verification_passed(verification)
    if not closed_criteria["execution_verification_passed"]:
        blocked_by.append("execution_verification_not_passed")

    closed_criteria["execution_summary_passed"] = _execution_summary_passed(execution_summary)
    if not closed_criteria["execution_summary_passed"]:
        blocked_by.append("execution_summary_not_passed")

    closed_criteria["artifact_hashes_verified"] = artifact_checks["all_hashes_verified"] is True
    if not closed_criteria["artifact_hashes_verified"]:
        blocked_by.append("artifact_hash_mismatch")

    closed_criteria["all_required_suites_present"] = all(name in suite_results for name in REQUIRED_SUITES)
    if not closed_criteria["all_required_suites_present"]:
        blocked_by.append("missing_required_suite_report")

    closed_criteria["all_suites_passed"] = all(
        suite_results.get(name, {}).get("passed") is True and suite_results.get(name, {}).get("failed_checks") == []
        for name in REQUIRED_SUITES
    )
    if not closed_criteria["all_suites_passed"]:
        blocked_by.append("suite_report_not_passed")

    closed_criteria["regression_thresholds_satisfied"] = _regression_thresholds_satisfied(suite_results)
    if not closed_criteria["regression_thresholds_satisfied"]:
        blocked_by.append("regression_threshold_not_satisfied")

    closed_criteria["plan_phase2_threshold_found"] = phase2_min_case_count is not None
    if phase2_min_case_count is None:
        blocked_by.append("plan_phase2_threshold_not_found")
        phase2_min_case_count = 45

    phase2_case_count = _metric_value(suite_results.get("Phase2 PLAN-scale tool-selection triples", {}), "case_count")
    closed_criteria["phase2_case_count_satisfies_plan"] = isinstance(phase2_case_count, int | float) and phase2_case_count >= phase2_min_case_count
    if not closed_criteria["phase2_case_count_satisfies_plan"]:
        blocked_by.append("phase2_case_count_below_plan_threshold")

    closed_criteria["no_forbidden_side_effects"] = _no_forbidden_side_effects(verification, execution_summary, list(suite_results.values()))
    if not closed_criteria["no_forbidden_side_effects"]:
        blocked_by.append("forbidden_side_effect_recorded")

    closed_criteria["local_smoke_not_overclaimed"] = _local_smoke_not_overclaimed(verification, execution_summary, suite_results)
    if not closed_criteria["local_smoke_not_overclaimed"]:
        blocked_by.append("benchmark_scope_overclaimed")

    closed_criteria["backfill_subjects_present"] = _backfill_subjects_present(backfill)
    if not closed_criteria["backfill_subjects_present"]:
        blocked_by.append("benchmark_subjects_missing")

    status = STRICT_PLAN_BENCHMARK_GATE_CLOSED if not blocked_by else BLOCKED_BENCHMARK_EVIDENCE_INCOMPLETE
    strict_plan_gate_closed = status == STRICT_PLAN_BENCHMARK_GATE_CLOSED

    report = base_decision_payload(
        gate_id=STRICT_PLAN_CLOSURE_GATE_ID,
        phase=STRICT_PLAN_CLOSURE_PHASE,
        target=STRICT_PLAN_CLOSURE_TARGET,
        generated_at=generated_at,
    )
    report["schema_version"] = STRICT_PLAN_CLOSURE_SCHEMA_VERSION
    report.update(
        {
            "status": status,
            "summary": _summary(status),
            "strict_plan_scope": STRICT_PLAN_SCOPE,
            "strict_plan_gate_closed": strict_plan_gate_closed,
            "benchmark_gate_passed": strict_plan_gate_closed,
            "local_smoke_benchmark_accepted_for_this_gate": strict_plan_gate_closed,
            "full_remote_benchmark_executed": False,
            "remote_or_service_benchmark_required_for_future_expansion": True,
            "real_benchmarks_executed": verification.get("real_benchmarks_executed") is True,
            "blocked_by": blocked_by,
            "closed_criteria": closed_criteria,
            "plan_contract": {
                "path": str(plan),
                "sha256": _sha256_path(plan),
                "phase2_min_case_count": phase2_min_case_count,
                "phase2_min_case_count_source": "PLAN.md Phase 2D default threshold",
            },
            "suite_results": _public_suite_results(suite_results),
            "artifact_hash_checks": artifact_checks,
            "source_artifacts": _source_artifacts(
                {
                    "benchmark_backfill": backfill_path,
                    "approval_packet": approval_path,
                    "preflight_report": preflight_path,
                    "readiness_report": readiness_path,
                    "execution_verification": verification_path,
                    "execution_summary": summary_path,
                }
            ),
            "benchmark_subjects": backfill.get("benchmark_subjects", {}),
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "github_query_performed": False,
            "github_write_performed": False,
            "active_apply_performed": False,
            "strict_plan_boundary_notes": [
                "This closes only the Phase 1/2 benchmark-regression strict PLAN gate from local smoke evidence.",
                "It does not claim full remote/service benchmark execution.",
                "It does not perform or approve GitHub write, provider/API spend, active apply, merge, deploy, cron, or gateway reload.",
            ],
            "not_claimed": [
                "full_remote_benchmark",
                "provider_api_spend",
                "github_write",
                "active_apply",
                "overall_HSE_project_completion",
                "upstream_PR_or_merge",
            ],
            "required_next_action": "consume_closed_benchmark_gate_in_next_strict_plan_audit"
            if strict_plan_gate_closed
            else "repair_blocked_benchmark_evidence_then_rerun_closure_gate",
            "artifacts": {
                "report": "benchmark_strict_plan_closure.json",
                "markdown": "benchmark_strict_plan_closure.md",
            },
        }
    )
    reject_github_or_active_apply_flags(report)

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "benchmark_strict_plan_closure.json"
    markdown_path = out / "benchmark_strict_plan_closure.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown_path.write_text(_render_markdown(report))
    return {"report_path": str(report_path), "markdown_path": str(markdown_path)}


def write_current_baseline_benchmark_strict_plan_closure(
    *,
    benchmark_backfill_path: str | Path,
    approval_packet_path: str | Path,
    current_baseline_preflight_path: str | Path,
    execution_verification_path: str | Path,
    execution_summary_path: str | Path,
    plan_path: str | Path,
    output_dir: str | Path,
    generated_at: str,
    approval_source: str,
) -> dict[str, str]:
    """Close the Phase 1/2 benchmark gate for the revalidated current baseline.

    This path consumes the current-baseline local smoke rerun artifacts generated
    after a strict-frontier mismatch. It preserves the same strict PLAN closure
    schema as the legacy closure report so downstream strict-frontier audits can
    consume it, but it records the new baseline/current subject pair explicitly.
    """

    _require_non_empty("generated_at", generated_at)
    _require_non_empty("approval_source", approval_source)
    backfill_path = Path(benchmark_backfill_path).expanduser()
    approval_path = Path(approval_packet_path).expanduser()
    preflight_path = Path(current_baseline_preflight_path).expanduser()
    verification_path = Path(execution_verification_path).expanduser()
    summary_path = Path(execution_summary_path).expanduser()
    plan = Path(plan_path).expanduser()

    backfill = _load_json_object(backfill_path, "benchmark backfill")
    approval = _load_json_object(approval_path, "approval packet")
    preflight = _load_json_object(preflight_path, "current-baseline preflight report")
    verification = _load_json_object(verification_path, "current-baseline execution verification report")
    execution_summary = _load_json_object(summary_path, "current-baseline execution summary")
    plan_text = plan.read_text()
    phase2_min_case_count = _phase2_min_case_count(plan_text)

    artifact_checks = _verify_current_baseline_artifact_hashes(verification, preflight_path, summary_path)
    suite_results = _current_baseline_suite_results_from_verification(verification)
    blocked_by: list[str] = []
    closed_criteria: dict[str, bool] = {}

    closed_criteria["approval_complete"] = _approval_complete(approval)
    if not closed_criteria["approval_complete"]:
        blocked_by.append("approval_not_complete")

    closed_criteria["current_baseline_go_approval_recorded"] = bool(approval_source.strip())
    if not closed_criteria["current_baseline_go_approval_recorded"]:
        blocked_by.append("current_baseline_go_approval_missing")

    closed_criteria["current_baseline_preflight_ready"] = _current_baseline_preflight_ready(preflight)
    if not closed_criteria["current_baseline_preflight_ready"]:
        blocked_by.append("current_baseline_preflight_not_ready")

    closed_criteria["current_baseline_execution_verification_passed"] = _current_baseline_execution_verification_passed(
        verification
    )
    if not closed_criteria["current_baseline_execution_verification_passed"]:
        blocked_by.append("current_baseline_execution_verification_not_passed")

    closed_criteria["current_baseline_execution_summary_passed"] = _current_baseline_execution_summary_passed(
        execution_summary
    )
    if not closed_criteria["current_baseline_execution_summary_passed"]:
        blocked_by.append("current_baseline_execution_summary_not_passed")

    closed_criteria["artifact_hashes_verified"] = artifact_checks["all_hashes_verified"] is True
    if not closed_criteria["artifact_hashes_verified"]:
        blocked_by.append("artifact_hash_mismatch")

    closed_criteria["all_outputs_under_approved_root"] = _all_outputs_under_approved_root(verification)
    if not closed_criteria["all_outputs_under_approved_root"]:
        blocked_by.append("output_path_outside_approved_root")

    closed_criteria["all_required_suites_present"] = all(name in suite_results for name in REQUIRED_SUITES)
    if not closed_criteria["all_required_suites_present"]:
        blocked_by.append("missing_required_suite_report")

    closed_criteria["all_suites_passed"] = all(
        suite_results.get(name, {}).get("passed") is True and suite_results.get(name, {}).get("failed_checks") == []
        for name in REQUIRED_SUITES
    )
    if not closed_criteria["all_suites_passed"]:
        blocked_by.append("suite_report_not_passed")

    closed_criteria["regression_thresholds_satisfied"] = _regression_thresholds_satisfied(suite_results)
    if not closed_criteria["regression_thresholds_satisfied"]:
        blocked_by.append("regression_threshold_not_satisfied")

    closed_criteria["plan_phase2_threshold_found"] = phase2_min_case_count is not None
    if phase2_min_case_count is None:
        blocked_by.append("plan_phase2_threshold_not_found")
        phase2_min_case_count = 45

    phase2_case_count = _metric_value(suite_results.get("Phase2 PLAN-scale tool-selection triples", {}), "case_count")
    closed_criteria["phase2_case_count_satisfies_plan"] = isinstance(phase2_case_count, int | float) and phase2_case_count >= phase2_min_case_count
    if not closed_criteria["phase2_case_count_satisfies_plan"]:
        blocked_by.append("phase2_case_count_below_plan_threshold")

    closed_criteria["subject_commits_align"] = _current_baseline_subject_commits_align(preflight, execution_summary)
    if not closed_criteria["subject_commits_align"]:
        blocked_by.append("current_baseline_subject_commit_mismatch")

    closed_criteria["no_forbidden_side_effects"] = _current_baseline_no_forbidden_side_effects(
        verification, execution_summary, list(suite_results.values())
    )
    if not closed_criteria["no_forbidden_side_effects"]:
        blocked_by.append("forbidden_side_effect_recorded")

    closed_criteria["local_smoke_not_overclaimed"] = _current_baseline_local_smoke_not_overclaimed(
        verification, execution_summary, suite_results
    )
    if not closed_criteria["local_smoke_not_overclaimed"]:
        blocked_by.append("benchmark_scope_overclaimed")

    closed_criteria["backfill_subjects_present"] = _backfill_subjects_present(backfill)
    if not closed_criteria["backfill_subjects_present"]:
        blocked_by.append("benchmark_subjects_missing")

    status = STRICT_PLAN_BENCHMARK_GATE_CLOSED if not blocked_by else BLOCKED_BENCHMARK_EVIDENCE_INCOMPLETE
    strict_plan_gate_closed = status == STRICT_PLAN_BENCHMARK_GATE_CLOSED
    benchmark_subjects = _current_baseline_benchmark_subjects(preflight, execution_summary)

    report = base_decision_payload(
        gate_id="SFA-CLOSE",
        phase="Current Baseline Phase 1/2 Benchmark Strict PLAN Closure Reconciliation",
        target="current-baseline-strict-plan-benchmark-regression-gate",
        generated_at=generated_at,
    )
    report["schema_version"] = STRICT_PLAN_CLOSURE_SCHEMA_VERSION
    report.update(
        {
            "status": status,
            "summary": _current_baseline_summary(status),
            "strict_plan_scope": STRICT_PLAN_SCOPE,
            "strict_plan_gate_closed": strict_plan_gate_closed,
            "benchmark_gate_passed": strict_plan_gate_closed,
            "local_smoke_benchmark_accepted_for_this_gate": strict_plan_gate_closed,
            "current_baseline_revalidation_closed": strict_plan_gate_closed,
            "full_remote_benchmark_executed": False,
            "remote_or_service_benchmark_required_for_future_expansion": True,
            "real_benchmarks_executed": execution_summary.get("real_benchmarks_executed") is True,
            "approval_source": approval_source.strip(),
            "blocked_by": blocked_by,
            "closed_criteria": closed_criteria,
            "plan_contract": {
                "path": str(plan),
                "sha256": _sha256_path(plan),
                "phase2_min_case_count": phase2_min_case_count,
                "phase2_min_case_count_source": "PLAN.md Phase 2D default threshold",
            },
            "suite_results": _public_suite_results(suite_results),
            "artifact_hash_checks": artifact_checks,
            "source_artifacts": _source_artifacts(
                {
                    "benchmark_backfill": backfill_path,
                    "approval_packet": approval_path,
                    "current_baseline_preflight": preflight_path,
                    "execution_verification": verification_path,
                    "execution_summary": summary_path,
                }
            ),
            "benchmark_subjects": benchmark_subjects,
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "github_query_performed": False,
            "github_write_performed": False,
            "active_apply_performed": False,
            "cron_or_gateway_mutation_performed": False,
            "strict_plan_boundary_notes": [
                "This closes only the current-baseline Phase 1/2 benchmark-regression strict PLAN gate from local smoke evidence.",
                "It does not claim full remote/service benchmark execution.",
                "It does not perform or approve GitHub query/write, provider/API spend, active apply, merge, deploy, cron, or gateway reload.",
            ],
            "not_claimed": [
                "full_remote_benchmark",
                "provider_api_spend",
                "github_query_or_write",
                "active_apply",
                "cron_or_gateway_mutation",
                "overall_HSE_project_completion",
                "upstream_PR_or_merge",
            ],
            "required_next_action": "rerun_strict_frontier_audit_for_current_active_baseline"
            if strict_plan_gate_closed
            else "repair_blocked_current_baseline_benchmark_evidence_then_rerun_closure_gate",
            "artifacts": {
                "report": "current_baseline_benchmark_strict_plan_closure.json",
                "markdown": "current_baseline_benchmark_strict_plan_closure.md",
            },
        }
    )
    reject_github_or_active_apply_flags(report)

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "current_baseline_benchmark_strict_plan_closure.json"
    markdown_path = out / "current_baseline_benchmark_strict_plan_closure.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown_path.write_text(_render_markdown(report))
    return {"report_path": str(report_path), "markdown_path": str(markdown_path)}



def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    if not isinstance(data, dict):
        raise ValueError(f"{label} JSON root must be an object: {path}")
    return data


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _require_non_empty(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _current_baseline_preflight_ready(preflight: Mapping[str, Any]) -> bool:
    rerun = preflight.get("rerun_decision", {}) if isinstance(preflight.get("rerun_decision"), Mapping) else {}
    root_guard = preflight.get("future_output_root_guard", {}) if isinstance(preflight.get("future_output_root_guard"), Mapping) else {}
    suite_readiness = preflight.get("suite_readiness", [])
    suites_ready = isinstance(suite_readiness, list) and all(
        isinstance(item, Mapping) and item.get("suite") in REQUIRED_SUITES and item.get("ready") is True for item in suite_readiness
    )
    return bool(
        preflight.get("schema_version") == "hse-current-baseline-revalidation-preflight-v1"
        and preflight.get("status") == "CURRENT_BASELINE_REVALIDATION_PREFLIGHT_READY_SEPARATE_GO_REQUIRED"
        and isinstance(preflight.get("baseline_commit_for_rerun"), str)
        and bool(preflight.get("baseline_commit_for_rerun"))
        and isinstance(preflight.get("current_commit_for_rerun"), str)
        and bool(preflight.get("current_commit_for_rerun"))
        and preflight.get("execution_go") is False
        and preflight.get("execution_started") is False
        and preflight.get("real_benchmarks_executed") is False
        and preflight.get("strict_plan_gate_closed") is False
        and preflight.get("full_remote_benchmark_executed") is False
        and preflight.get("github_query_performed") is False
        and preflight.get("github_write_performed") is False
        and preflight.get("provider_or_model_spend_performed") is False
        and preflight.get("network_calls_performed") is False
        and preflight.get("active_apply_performed") is False
        and rerun.get("rerun_recommended") is True
        and rerun.get("separate_local_smoke_go_required") is True
        and rerun.get("local_smoke_rerun_ready_not_started") is True
        and root_guard.get("future_output_root_created_now") is False
        and root_guard.get("benchmark_output_written_now") is False
        and suites_ready
    )


def _current_baseline_execution_verification_passed(verification: Mapping[str, Any]) -> bool:
    invariants = verification.get("invariants", {}) if isinstance(verification.get("invariants"), Mapping) else {}
    produced = verification.get("produced_files_manifest", [])
    return bool(
        verification.get("schema_version") == "hse-current-baseline-local-smoke-verification-v1"
        and isinstance(produced, list)
        and len(produced) >= 4
        and invariants.get("status_passed") is True
        and invariants.get("all_commands_succeeded") is True
        and invariants.get("all_suite_outputs_exist") is True
        and invariants.get("all_suites_passed") is True
        and invariants.get("all_boundary_flags_false") is True
        and invariants.get("all_executed_commands_removed_dry_run") is True
        and invariants.get("all_outputs_under_approved_root") is True
        and invariants.get("all_outputs_under_hse_allowed_root") is True
        and invariants.get("strict_plan_gate_closed") is False
        and invariants.get("full_remote_benchmark_executed") is False
        and invariants.get("github_query_performed") is False
        and invariants.get("github_write_performed") is False
        and invariants.get("provider_or_model_spend_performed") is False
        and invariants.get("network_calls_performed") is False
        and invariants.get("active_apply_performed") is False
        and invariants.get("cron_or_gateway_mutation_performed") is False
    )


def _current_baseline_execution_summary_passed(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("schema_version") == "hse-current-baseline-local-smoke-execution-summary-v1"
        and summary.get("status") == "LOCAL_SMOKE_EXECUTION_PASSED"
        and summary.get("execution_started") is True
        and summary.get("real_benchmarks_executed") is True
        and summary.get("all_commands_succeeded") is True
        and summary.get("all_suite_outputs_exist") is True
        and summary.get("all_suites_passed") is True
        and summary.get("all_boundary_flags_false") is True
        and summary.get("strict_plan_gate_closed") is False
        and summary.get("full_remote_benchmark_executed") is False
        and summary.get("github_query_performed") is False
        and summary.get("github_write_performed") is False
        and summary.get("provider_or_model_spend_performed") is False
        and summary.get("network_calls_performed") is False
        and summary.get("active_apply_performed") is False
        and summary.get("cron_or_gateway_mutation_performed") is False
    )


def _verify_current_baseline_artifact_hashes(
    verification: Mapping[str, Any], preflight_path: Path, summary_path: Path
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for field, expected_path in (("source_preflight_report", preflight_path), ("execution_summary", summary_path), ("execution_report", None)):
        record = verification.get(field)
        if not isinstance(record, Mapping):
            if field == "execution_report":
                continue
            checks.append({"field": field, "path": str(expected_path), "hash_verified": False, "reason": "missing record"})
            continue
        path = Path(str(record.get("path") or expected_path)).expanduser()
        expected = record.get("sha256")
        checks.append(_hash_check(field, path, expected))
    produced = verification.get("produced_files_manifest", [])
    if not isinstance(produced, list):
        return {"all_hashes_verified": False, "checks": checks, "error": "produced_files_manifest must be a list"}
    for index, record in enumerate(produced):
        if not isinstance(record, Mapping):
            checks.append({"field": f"produced_files_manifest[{index}]", "hash_verified": False, "reason": "record is not an object"})
            continue
        path_value = record.get("path")
        expected = record.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            checks.append({"field": f"produced_files_manifest[{index}]", "hash_verified": False, "reason": "missing path"})
            continue
        checks.append(_hash_check(f"produced_files_manifest[{index}]", Path(path_value).expanduser(), expected))
    return {"all_hashes_verified": bool(checks) and all(check.get("hash_verified") is True for check in checks), "checks": checks}


def _hash_check(field: str, path: Path, expected: Any) -> dict[str, Any]:
    if not path.exists():
        return {"field": field, "path": str(path), "expected_sha256": expected, "hash_verified": False, "reason": "missing file"}
    actual = _sha256_path(path)
    return {
        "field": field,
        "path": str(path),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "bytes": path.stat().st_size,
        "hash_verified": actual == expected,
    }


def _current_baseline_suite_results_from_verification(verification: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    produced = verification.get("produced_files_manifest", [])
    if not isinstance(produced, list):
        return results
    for record in produced:
        if not isinstance(record, Mapping):
            continue
        path_value = record.get("path")
        if not isinstance(path_value, str):
            continue
        path = Path(path_value).expanduser()
        if not path.exists():
            continue
        data = _load_json_object(path, f"current-baseline suite report {path.name}")
        benchmark = data.get("benchmark")
        if benchmark not in REQUIRED_SUITES:
            continue
        metrics = data.get("metrics", {}) if isinstance(data.get("metrics"), Mapping) else {}
        results[str(benchmark)] = {
            "path": str(path),
            "sha256": _sha256_path(path),
            "passed": data.get("passed") is True,
            "mode": data.get("mode"),
            "failed_checks": data.get("failed_checks", []),
            "metrics": dict(metrics),
            "case_count": metrics.get("case_count"),
            "provider_or_model_spend_performed": data.get("provider_or_model_spend_performed"),
            "network_calls_performed": data.get("network_calls_performed"),
            "github_write_performed": data.get("github_write_performed"),
            "active_apply_performed": data.get("active_apply_performed"),
            "strict_plan_gate_closed": data.get("strict_plan_gate_closed"),
            "full_benchmark_executed": data.get("full_benchmark_executed"),
            "phase2d_gate_passed": _phase2d_gate_passed(data),
        }
    return results


def _all_outputs_under_approved_root(verification: Mapping[str, Any]) -> bool:
    approved = verification.get("approved_output_root")
    produced = verification.get("produced_files_manifest", [])
    if not isinstance(approved, str) or not approved or not isinstance(produced, list):
        return False
    approved_root = Path(approved).expanduser().resolve(strict=False)
    for record in produced:
        if not isinstance(record, Mapping):
            return False
        path_value = record.get("path")
        if not isinstance(path_value, str):
            return False
        resolved = Path(path_value).expanduser().resolve(strict=False)
        if not (resolved == approved_root or approved_root in resolved.parents):
            return False
        if record.get("under_approved_run_output_root") is not True or record.get("under_allowed_hse_output_root") is not True:
            return False
    return True


def _current_baseline_subject_commits_align(preflight: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    return bool(
        preflight.get("baseline_commit_for_rerun") == summary.get("baseline_commit")
        and preflight.get("current_commit_for_rerun") == summary.get("current_commit")
    )


def _current_baseline_no_forbidden_side_effects(
    verification: Mapping[str, Any], summary: Mapping[str, Any], suites: Sequence[Mapping[str, Any]]
) -> bool:
    invariants = verification.get("invariants", {}) if isinstance(verification.get("invariants"), Mapping) else {}
    top_level_false = (
        summary.get("provider_or_model_spend_performed") is False
        and summary.get("network_calls_performed") is False
        and summary.get("github_query_performed") is False
        and summary.get("github_write_performed") is False
        and summary.get("active_apply_performed") is False
        and summary.get("cron_or_gateway_mutation_performed") is False
        and summary.get("full_remote_benchmark_executed") is False
        and invariants.get("provider_or_model_spend_performed") is False
        and invariants.get("network_calls_performed") is False
        and invariants.get("github_query_performed") is False
        and invariants.get("github_write_performed") is False
        and invariants.get("active_apply_performed") is False
        and invariants.get("cron_or_gateway_mutation_performed") is False
        and invariants.get("full_remote_benchmark_executed") is False
    )
    suites_false = all(
        suite.get("provider_or_model_spend_performed") is False
        and suite.get("network_calls_performed") is False
        and suite.get("github_write_performed") is False
        and suite.get("active_apply_performed") is False
        and suite.get("strict_plan_gate_closed") is False
        and suite.get("full_benchmark_executed") is False
        for suite in suites
    )
    return bool(top_level_false and suites_false)


def _current_baseline_local_smoke_not_overclaimed(
    verification: Mapping[str, Any], summary: Mapping[str, Any], suite_results: Mapping[str, Mapping[str, Any]]
) -> bool:
    not_claimed = verification.get("not_claimed", [])
    return bool(
        "full_remote_benchmark" in not_claimed
        and "strict_plan_gate_closure" in not_claimed
        and summary.get("strict_plan_gate_closed") is False
        and summary.get("full_remote_benchmark_executed") is False
        and all(suite.get("mode") == "real-benchmark-smoke" for suite in suite_results.values())
        and all(suite.get("full_benchmark_executed") is False for suite in suite_results.values())
    )


def _current_baseline_benchmark_subjects(preflight: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    baseline_commit = str(summary.get("baseline_commit") or preflight.get("baseline_commit_for_rerun") or "")
    current_commit = str(summary.get("current_commit") or preflight.get("current_commit_for_rerun") or "")
    return {
        "baseline": {
            "subject_id": "recorded-phase1-phase2-closure-subject",
            "label": "Previously closed Phase 1/2 benchmark subject used as current-baseline rerun baseline",
            "hermes_source": {"commit": baseline_commit, "commit_full": baseline_commit},
        },
        "current": {
            "subject_id": "current-active-hermes-baseline-revalidated",
            "label": "Current active Hermes baseline revalidated by local-only Phase 1/2 smoke rerun",
            "hermes_source": {"commit": current_commit, "commit_full": current_commit},
        },
    }


def _current_baseline_summary(status: str) -> str:
    if status == STRICT_PLAN_BENCHMARK_GATE_CLOSED:
        return "Current-baseline local smoke evidence closes the Phase 1/2 benchmark-regression strict PLAN gate."
    return "Current-baseline strict PLAN benchmark-gate closure is blocked until local smoke evidence is complete and hash-verified."


def _phase2_min_case_count(plan_text: str) -> int | None:
    if "Default Phase 2D thresholds" not in plan_text:
        return None
    match = re.search(r'"min_case_count"\s*:\s*(\d+)', plan_text)
    if not match:
        return None
    return int(match.group(1))


def _approval_complete(approval: Mapping[str, Any]) -> bool:
    return bool(
        approval.get("approval_complete") is True
        and approval.get("real_benchmark_execution_approved") is True
        and approval.get("network_provider_spend_allowed") is False
        and approval.get("current_authorized_budget_usd", 0) == 0
        and approval.get("current_authorized_budget_krw", 0) == 0
        and _approval_preserves_no_github_write(approval)
    )


def _approval_preserves_no_github_write(approval: Mapping[str, Any]) -> bool:
    if approval.get("github_policy") == "NO_GITHUB_WRITE":
        return True
    github = approval.get("github")
    if not isinstance(github, Mapping):
        return False
    return all(github.get(key) is False for key in ("queried", "pr_created", "push_performed", "merge_performed"))


def _preflight_ready(preflight: Mapping[str, Any]) -> bool:
    return bool(
        preflight.get("schema_version") == "hse-real-benchmark-preflight-v1"
        and preflight.get("preflight_passed") is True
        and preflight.get("execution_ready") is True
        and preflight.get("approval_complete") is True
        and preflight.get("real_benchmark_execution_approved") is True
        and preflight.get("execution_started") is False
        and preflight.get("real_benchmarks_executed") is False
        and preflight.get("strict_plan_gate_closed") is False
    )


def _readiness_go(readiness: Mapping[str, Any]) -> bool:
    runner = readiness.get("runner_checks", {}) if isinstance(readiness.get("runner_checks"), Mapping) else {}
    write_root = readiness.get("write_root_checks", {}) if isinstance(readiness.get("write_root_checks"), Mapping) else {}
    return bool(
        readiness.get("schema_version") == "hse-real-benchmark-execution-readiness-v1"
        and readiness.get("status") == "READY_TO_EXECUTE_NOT_STARTED"
        and readiness.get("execution_go") is True
        and readiness.get("preflight_execution_ready") is True
        and readiness.get("preflight_passed") is True
        and readiness.get("real_benchmark_execution_approved") is True
        and readiness.get("strict_plan_gate_closed") is False
        and readiness.get("execution_started") is False
        and readiness.get("real_benchmarks_executed") is False
        and runner.get("preview_runner_module_available") is True
        and runner.get("all_suite_readiness_ready") is True
        and write_root.get("all_preview_outputs_under_allowed_roots") is True
    )


def _execution_verification_passed(verification: Mapping[str, Any]) -> bool:
    return bool(
        verification.get("schema_version") == "hse-real-benchmark-execution-result-verification-v1"
        and verification.get("status") == "REAL_BENCHMARK_SMOKE_PASS_LOCAL_ONLY"
        and verification.get("execution_started") is True
        and verification.get("real_benchmarks_executed") is True
        and verification.get("all_suites_passed") is True
        and verification.get("suite_count") == 3
        and verification.get("strict_plan_gate_closed") is False
    )


def _execution_summary_passed(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary.get("schema_version") == "hse-real-benchmark-execution-summary-v1"
        and summary.get("status") == "REAL_BENCHMARK_SMOKE_EXECUTED"
        and summary.get("execution_started") is True
        and summary.get("real_benchmarks_executed") is True
        and summary.get("all_suites_passed") is True
        and summary.get("suite_count") == 3
        and summary.get("strict_plan_gate_closed") is False
    )


def _verify_report_hashes(verification: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    reports = verification.get("reports", [])
    if not isinstance(reports, list):
        return {"all_hashes_verified": False, "checks": [], "error": "verification.reports must be a list"}
    for record in reports:
        if not isinstance(record, Mapping):
            checks.append({"path": None, "hash_verified": False, "reason": "report record is not an object"})
            continue
        path_value = record.get("path")
        expected = record.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            checks.append({"path": path_value, "hash_verified": False, "reason": "missing path"})
            continue
        path = Path(path_value).expanduser()
        if not path.exists():
            checks.append({"path": str(path), "hash_verified": False, "reason": "missing file"})
            continue
        actual = _sha256_path(path)
        checks.append(
            {
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "bytes": path.stat().st_size,
                "hash_verified": actual == expected,
            }
        )
    return {"all_hashes_verified": bool(checks) and all(check.get("hash_verified") is True for check in checks), "checks": checks}


def _suite_results_from_verification(verification: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    reports = verification.get("reports", [])
    if not isinstance(reports, list):
        return results
    for record in reports:
        if not isinstance(record, Mapping):
            continue
        benchmark = record.get("benchmark")
        path_value = record.get("path")
        if benchmark not in REQUIRED_SUITES or not isinstance(path_value, str):
            continue
        path = Path(path_value).expanduser()
        if not path.exists():
            continue
        data = _load_json_object(path, f"suite report {benchmark}")
        metrics = data.get("metrics", {}) if isinstance(data.get("metrics"), Mapping) else {}
        results[str(benchmark)] = {
            "path": str(path),
            "sha256": _sha256_path(path),
            "passed": data.get("passed") is True,
            "mode": data.get("mode"),
            "failed_checks": data.get("failed_checks", []),
            "metrics": dict(metrics),
            "case_count": metrics.get("case_count"),
            "provider_or_model_spend_performed": data.get("provider_or_model_spend_performed"),
            "network_calls_performed": data.get("network_calls_performed"),
            "github_write_performed": data.get("github_write_performed"),
            "active_apply_performed": data.get("active_apply_performed"),
            "strict_plan_gate_closed": data.get("strict_plan_gate_closed"),
            "full_benchmark_executed": data.get("full_benchmark_executed"),
            "phase2d_gate_passed": _phase2d_gate_passed(data),
        }
    return results


def _phase2d_gate_passed(data: Mapping[str, Any]) -> bool | None:
    gate = data.get("phase2d_gate")
    if not isinstance(gate, Mapping):
        return None
    return gate.get("passed") is True and gate.get("failed_checks", []) == []


def _metric_value(suite: Mapping[str, Any], metric: str) -> Any:
    metrics = suite.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    return metrics.get(metric)


def _regression_thresholds_satisfied(suite_results: Mapping[str, Mapping[str, Any]]) -> bool:
    tblite = suite_results.get("TBLite", {})
    yc = suite_results.get("YC-Bench", {})
    phase2 = suite_results.get("Phase2 PLAN-scale tool-selection triples", {})
    tblite_ok = _score(tblite, "candidate_score") >= _score(tblite, "baseline_score") * 0.98
    yc_ok = _score(yc, "candidate_score") >= _score(yc, "baseline_score")
    phase2_ok = (
        phase2.get("phase2d_gate_passed") is True
        and _score(phase2, "selection_accuracy") >= 1.0
        and _score(phase2, "wrong_tool_avoidance") >= 1.0
    )
    return bool(tblite_ok and yc_ok and phase2_ok)


def _score(suite: Mapping[str, Any], metric: str) -> float:
    value = _metric_value(suite, metric)
    if isinstance(value, int | float):
        return float(value)
    return float("-inf")


def _no_forbidden_side_effects(
    verification: Mapping[str, Any], summary: Mapping[str, Any], suites: Sequence[Mapping[str, Any]]
) -> bool:
    for source in [verification, summary, *suites]:
        for flag in _SIDE_EFFECT_FLAGS:
            if source.get(flag) is not False:
                return False
    return True


def _local_smoke_not_overclaimed(
    verification: Mapping[str, Any], summary: Mapping[str, Any], suite_results: Mapping[str, Mapping[str, Any]]
) -> bool:
    not_claimed = verification.get("not_claimed", [])
    return bool(
        "full_remote_benchmark" in not_claimed
        and verification.get("strict_plan_gate_closed") is False
        and summary.get("strict_plan_gate_closed") is False
        and all(suite.get("mode") == "real-benchmark-smoke" for suite in suite_results.values())
        and all(suite.get("full_benchmark_executed") is False for suite in suite_results.values())
    )


def _backfill_subjects_present(backfill: Mapping[str, Any]) -> bool:
    subjects = backfill.get("benchmark_subjects")
    if not isinstance(subjects, Mapping):
        return False
    for key in ("baseline", "current"):
        subject = subjects.get(key)
        if not isinstance(subject, Mapping):
            return False
        source = subject.get("hermes_source")
        if not isinstance(source, Mapping) or not isinstance(source.get("commit"), str) or not source.get("commit"):
            return False
    return True


def _public_suite_results(suite_results: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    public: dict[str, dict[str, Any]] = {}
    for name, result in suite_results.items():
        metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), Mapping) else {}
        public[name] = {
            "path": result.get("path"),
            "sha256": result.get("sha256"),
            "passed": result.get("passed"),
            "mode": result.get("mode"),
            "failed_checks": result.get("failed_checks"),
            "case_count": metrics.get("case_count"),
            "metrics": dict(metrics),
        }
    return public


def _source_artifacts(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {"path": str(path), "sha256": _sha256_path(path), "bytes": path.stat().st_size}
        for name, path in paths.items()
    }


def _summary(status: str) -> str:
    if status == STRICT_PLAN_BENCHMARK_GATE_CLOSED:
        return "Local-only real-benchmark-smoke evidence closes the Phase 1/2 benchmark-regression strict PLAN gate."
    return "Strict PLAN benchmark-gate closure is blocked until evidence is complete, hash-verified, and regression-safe."


def _render_markdown(report: Mapping[str, Any]) -> str:
    suite_lines = []
    suite_results = report.get("suite_results", {}) if isinstance(report.get("suite_results"), Mapping) else {}
    for name, suite in suite_results.items():
        if isinstance(suite, Mapping):
            suite_lines.append(
                f"- {name}: passed={str(suite.get('passed')).lower()} mode=`{suite.get('mode')}` case_count={suite.get('case_count')}"
            )
    blocked = report.get("blocked_by", [])
    blocked_lines = [f"- {item}" for item in blocked] if isinstance(blocked, list) and blocked else ["- none"]
    return "\n".join(
        [
            "# HSE Strict PLAN Benchmark Gate Closure",
            "",
            f"Status: `{report.get('status')}`",
            "",
            "## Gate State",
            "",
            f"- strict_plan_gate_closed={str(report.get('strict_plan_gate_closed')).lower()}",
            f"- benchmark_gate_passed={str(report.get('benchmark_gate_passed')).lower()}",
            f"- strict_plan_scope=`{report.get('strict_plan_scope')}`",
            f"- local_smoke_benchmark_accepted_for_this_gate={str(report.get('local_smoke_benchmark_accepted_for_this_gate')).lower()}",
            f"- full_remote_benchmark_executed={str(report.get('full_remote_benchmark_executed')).lower()}",
            "",
            "## Suites",
            "",
            *suite_lines,
            "",
            "## Blockers",
            "",
            *blocked_lines,
            "",
            "## Boundaries",
            "",
            "- provider/API/network spend performed=false",
            "- GitHub write/query performed=false",
            "- active apply performed=false",
            "- overall HSE project completion not claimed",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an HSE strict PLAN benchmark-gate closure reconciliation report.")
    parser.add_argument("--benchmark-backfill", required=True, type=Path)
    parser.add_argument("--approval-packet", required=True, type=Path)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--readiness-report", required=True, type=Path)
    parser.add_argument("--execution-verification", required=True, type=Path)
    parser.add_argument("--execution-summary", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", required=True)
    args = parser.parse_args(argv)
    result = write_benchmark_strict_plan_closure(
        benchmark_backfill_path=args.benchmark_backfill,
        approval_packet_path=args.approval_packet,
        preflight_report_path=args.preflight_report,
        readiness_report_path=args.readiness_report,
        execution_verification_path=args.execution_verification,
        execution_summary_path=args.execution_summary,
        plan_path=args.plan,
        output_dir=args.output_dir,
        generated_at=args.generated_at,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

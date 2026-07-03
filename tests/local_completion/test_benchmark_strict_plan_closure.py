"""Tests for HSE strict PLAN benchmark-gate closure reconciliation."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from evolution.local_completion.benchmark_strict_plan_closure import (
    BLOCKED_BENCHMARK_EVIDENCE_INCOMPLETE,
    STRICT_PLAN_BENCHMARK_GATE_CLOSED,
    write_current_baseline_benchmark_strict_plan_closure,
    write_benchmark_strict_plan_closure,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _valid_artifacts(tmp_path: Path) -> dict[str, Path]:
    output_root = tmp_path / "output" / "hse-real-benchmark" / "run"
    tblite = _write_json(
        output_root / "tblite.json",
        {
            "benchmark": "TBLite",
            "mode": "real-benchmark-smoke",
            "passed": True,
            "failed_checks": [],
            "metrics": {"baseline_score": 3.0, "candidate_score": 3.0, "case_count": 3, "score_delta": 0.0},
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "github_write_performed": False,
            "active_apply_performed": False,
            "strict_plan_gate_closed": False,
            "full_benchmark_executed": False,
        },
    )
    yc = _write_json(
        output_root / "yc-bench.json",
        {
            "benchmark": "YC-Bench",
            "mode": "real-benchmark-smoke",
            "passed": True,
            "failed_checks": [],
            "metrics": {
                "baseline_score": 1.6666666666666665,
                "candidate_score": 3.0,
                "case_count": 3,
                "score_delta": 1.3333333333333335,
            },
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "github_write_performed": False,
            "active_apply_performed": False,
            "strict_plan_gate_closed": False,
            "full_benchmark_executed": False,
        },
    )
    phase2 = _write_json(
        output_root / "phase2-plan-scale-tool-selection-triples.json",
        {
            "benchmark": "Phase2 PLAN-scale tool-selection triples",
            "mode": "real-benchmark-smoke",
            "passed": True,
            "failed_checks": [],
            "metrics": {
                "case_count": 45,
                "selection_accuracy": 1.0,
                "wrong_tool_avoidance": 1.0,
                "argument_cue_coverage": 1.0,
                "constraint_pass_rate": 1.0,
            },
            "phase2d_gate": {"passed": True, "failed_checks": [], "thresholds": {"min_case_count": 45}},
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "github_write_performed": False,
            "active_apply_performed": False,
            "strict_plan_gate_closed": False,
            "full_benchmark_executed": False,
        },
    )
    summary = _write_json(
        output_root / "execution_summary.json",
        {
            "schema_version": "hse-real-benchmark-execution-summary-v1",
            "status": "REAL_BENCHMARK_SMOKE_EXECUTED",
            "execution_started": True,
            "real_benchmarks_executed": True,
            "all_suites_passed": True,
            "suite_count": 3,
            "strict_plan_gate_closed": False,
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "github_write_performed": False,
            "active_apply_performed": False,
            "baseline_current_materialization_performed": False,
            "suite_reports": [
                {"benchmark": "TBLite", "path": str(tblite), "sha256": _sha(tblite), "passed": True},
                {"benchmark": "YC-Bench", "path": str(yc), "sha256": _sha(yc), "passed": True},
                {
                    "benchmark": "Phase2 PLAN-scale tool-selection triples",
                    "path": str(phase2),
                    "sha256": _sha(phase2),
                    "passed": True,
                },
            ],
        },
    )
    verification = _write_json(
        tmp_path / "reports" / "execution_verification.json",
        {
            "schema_version": "hse-real-benchmark-execution-result-verification-v1",
            "status": "REAL_BENCHMARK_SMOKE_PASS_LOCAL_ONLY",
            "execution_output_root": str(output_root),
            "execution_started": True,
            "real_benchmarks_executed": True,
            "all_suites_passed": True,
            "suite_count": 3,
            "strict_plan_gate_closed": False,
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "github_write_performed": False,
            "active_apply_performed": False,
            "baseline_current_materialization_performed": False,
            "reports": [
                {"benchmark": "TBLite", "path": str(tblite), "sha256": _sha(tblite), "passed": True, "failed_checks": []},
                {"benchmark": "YC-Bench", "path": str(yc), "sha256": _sha(yc), "passed": True, "failed_checks": []},
                {
                    "benchmark": "Phase2 PLAN-scale tool-selection triples",
                    "path": str(phase2),
                    "sha256": _sha(phase2),
                    "passed": True,
                    "failed_checks": [],
                },
                {"path": str(summary), "sha256": _sha(summary), "status": "REAL_BENCHMARK_SMOKE_EXECUTED"},
            ],
            "not_claimed": ["full_remote_benchmark", "provider_api_spend", "github_write", "active_apply"],
        },
    )
    approval = _write_json(
        tmp_path / "reports" / "approval.json",
        {
            "schema_version": "hse-real-benchmark-approval-packet-v1",
            "status": "APPROVAL_RECORDED_NOT_EXECUTED",
            "approval_complete": True,
            "real_benchmark_execution_approved": True,
            "current_authorized_budget_usd": 0,
            "current_authorized_budget_krw": 0,
            "network_provider_spend_allowed": False,
            "github": {"queried": False, "pr_created": False, "push_performed": False, "merge_performed": False},
        },
    )
    preflight = _write_json(
        tmp_path / "reports" / "preflight.json",
        {
            "schema_version": "hse-real-benchmark-preflight-v1",
            "status": "PREFLIGHT_EXECUTION_READY_NOT_STARTED",
            "preflight_passed": True,
            "execution_ready": True,
            "approval_complete": True,
            "real_benchmark_execution_approved": True,
            "execution_started": False,
            "real_benchmarks_executed": False,
            "strict_plan_gate_closed": False,
            "network_provider_spend_allowed": False,
        },
    )
    readiness = _write_json(
        tmp_path / "reports" / "readiness.json",
        {
            "schema_version": "hse-real-benchmark-execution-readiness-v1",
            "status": "READY_TO_EXECUTE_NOT_STARTED",
            "execution_go": True,
            "approval_complete": True,
            "preflight_execution_ready": True,
            "preflight_passed": True,
            "real_benchmark_execution_approved": True,
            "strict_plan_gate_closed": False,
            "execution_started": False,
            "real_benchmarks_executed": False,
            "runner_checks": {"preview_runner_module_available": True, "all_suite_readiness_ready": True},
            "write_root_checks": {"all_preview_outputs_under_allowed_roots": True},
        },
    )
    backfill = _write_json(
        tmp_path / "reports" / "backfill.json",
        {
            "schema_version": "hse-benchmark-gate-backfill-v1",
            "status": "BLOCKED_BY_BENCHMARK_APPROVAL",
            "strict_plan_gate_closed": False,
            "benchmark_gate_passed": None,
            "real_benchmarks_executed": False,
            "real_benchmark_execution_approved": False,
            "blocked_reason": "real benchmark execution approval is required before strict PLAN gate closure",
            "benchmark_subjects": {
                "baseline": {"subject_id": "baseline", "hermes_source": {"commit": "88d1d6206"}},
                "current": {"subject_id": "current", "hermes_source": {"commit": "9b50c5655"}},
            },
            "benchmark_plan": {
                "comparison_policy": "strict PLAN promotion requires approved real benchmark/equivalent execution with baseline/current comparison and no regression beyond thresholds"
            },
        },
    )
    plan = tmp_path / "PLAN.md"
    plan.write_text('Default Phase 2D thresholds for the expanded 45-case golden set:\n```json\n{"min_case_count": 45}\n```\n')
    return {
        "backfill": backfill,
        "approval": approval,
        "preflight": preflight,
        "readiness": readiness,
        "verification": verification,
        "summary": summary,
        "plan": plan,
        "phase2": phase2,
    }


def test_benchmark_strict_plan_closure_closes_local_smoke_gate_with_verified_evidence(tmp_path: Path):
    paths = _valid_artifacts(tmp_path)

    result = write_benchmark_strict_plan_closure(
        benchmark_backfill_path=paths["backfill"],
        approval_packet_path=paths["approval"],
        preflight_report_path=paths["preflight"],
        readiness_report_path=paths["readiness"],
        execution_verification_path=paths["verification"],
        execution_summary_path=paths["summary"],
        plan_path=paths["plan"],
        output_dir=tmp_path / "closure",
        generated_at="2026-07-04T01:40:00+09:00",
    )

    report = json.loads(Path(result["report_path"]).read_text())
    assert report["schema_version"] == "hse-benchmark-strict-plan-closure-v1"
    assert report["status"] == STRICT_PLAN_BENCHMARK_GATE_CLOSED
    assert report["strict_plan_gate_closed"] is True
    assert report["benchmark_gate_passed"] is True
    assert report["strict_plan_scope"] == "phase1_phase2_benchmark_regression_gate"
    assert report["full_remote_benchmark_executed"] is False
    assert report["local_smoke_benchmark_accepted_for_this_gate"] is True
    assert report["provider_or_model_spend_performed"] is False
    assert report["network_calls_performed"] is False
    assert report["github_query_performed"] is False
    assert report["github_write_performed"] is False
    assert report["active_apply_performed"] is False
    assert report["plan_contract"]["phase2_min_case_count"] == 45
    assert report["suite_results"]["Phase2 PLAN-scale tool-selection triples"]["case_count"] == 45
    assert report["closed_criteria"]["all_suites_passed"] is True
    assert report["closed_criteria"]["artifact_hashes_verified"] is True
    assert report["blocked_by"] == []
    markdown = Path(result["markdown_path"]).read_text()
    assert "STRICT_PLAN_BENCHMARK_GATE_CLOSED" in markdown
    assert "strict_plan_gate_closed=true" in markdown
    assert "full_remote_benchmark_executed=false" in markdown


def test_benchmark_strict_plan_closure_blocks_when_phase2_case_count_below_plan_threshold(tmp_path: Path):
    paths = _valid_artifacts(tmp_path)
    phase2 = json.loads(paths["phase2"].read_text())
    phase2["metrics"]["case_count"] = 44
    paths["phase2"].write_text(json.dumps(phase2, indent=2, sort_keys=True) + "\n")

    result = write_benchmark_strict_plan_closure(
        benchmark_backfill_path=paths["backfill"],
        approval_packet_path=paths["approval"],
        preflight_report_path=paths["preflight"],
        readiness_report_path=paths["readiness"],
        execution_verification_path=paths["verification"],
        execution_summary_path=paths["summary"],
        plan_path=paths["plan"],
        output_dir=tmp_path / "blocked",
        generated_at="2026-07-04T01:40:00+09:00",
    )

    report = json.loads(Path(result["report_path"]).read_text())
    assert report["status"] == BLOCKED_BENCHMARK_EVIDENCE_INCOMPLETE
    assert report["strict_plan_gate_closed"] is False
    assert report["benchmark_gate_passed"] is False
    assert "phase2_case_count_below_plan_threshold" in report["blocked_by"]


def test_benchmark_strict_plan_closure_blocks_hash_mismatch_without_side_effects(tmp_path: Path):
    paths = _valid_artifacts(tmp_path)
    verification = json.loads(paths["verification"].read_text())
    verification["reports"][0]["sha256"] = "0" * 64
    paths["verification"].write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")

    result = write_benchmark_strict_plan_closure(
        benchmark_backfill_path=paths["backfill"],
        approval_packet_path=paths["approval"],
        preflight_report_path=paths["preflight"],
        readiness_report_path=paths["readiness"],
        execution_verification_path=paths["verification"],
        execution_summary_path=paths["summary"],
        plan_path=paths["plan"],
        output_dir=tmp_path / "hash-blocked",
        generated_at="2026-07-04T01:40:00+09:00",
    )

    report = json.loads(Path(result["report_path"]).read_text())
    assert report["status"] == BLOCKED_BENCHMARK_EVIDENCE_INCOMPLETE
    assert report["strict_plan_gate_closed"] is False
    assert report["benchmark_gate_passed"] is False
    assert "artifact_hash_mismatch" in report["blocked_by"]
    assert report["github"]["queried"] is False
    assert report["safety_invariants"]["network_calls_performed"] is False


def _valid_current_baseline_artifacts(tmp_path: Path) -> dict[str, Path]:
    paths = _valid_artifacts(tmp_path)
    old_verification = json.loads(paths["verification"].read_text())
    suite_records = old_verification["reports"][:3]
    output_root = Path(suite_records[0]["path"]).parent
    preflight = _write_json(
        tmp_path / "reports" / "current_baseline_preflight.json",
        {
            "schema_version": "hse-current-baseline-revalidation-preflight-v1",
            "status": "CURRENT_BASELINE_REVALIDATION_PREFLIGHT_READY_SEPARATE_GO_REQUIRED",
            "baseline_commit_for_rerun": "9b50c56556f902b62ecc4a7e2e511ca0f316da2d",
            "current_commit_for_rerun": "551e5af50dc6597069e57af047213f61e40246d6",
            "execution_go": False,
            "execution_started": False,
            "real_benchmarks_executed": False,
            "strict_plan_gate_closed": False,
            "full_remote_benchmark_executed": False,
            "github_query_performed": False,
            "github_write_performed": False,
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "active_apply_performed": False,
            "rerun_decision": {
                "rerun_recommended": True,
                "rerun_approved_now": False,
                "separate_local_smoke_go_required": True,
                "local_smoke_rerun_ready_not_started": True,
            },
            "future_output_root_guard": {
                "future_output_root": str(output_root),
                "future_output_root_exists_now": False,
                "future_output_root_created_now": False,
                "benchmark_output_written_now": False,
                "passed": True,
            },
            "suite_readiness": [
                {"suite": "TBLite", "ready": True, "blocked_by": []},
                {"suite": "YC-Bench", "ready": True, "blocked_by": []},
                {"suite": "Phase2 PLAN-scale tool-selection triples", "ready": True, "blocked_by": []},
            ],
        },
    )
    summary = _write_json(
        output_root / "current_execution_summary.json",
        {
            "schema_version": "hse-current-baseline-local-smoke-execution-summary-v1",
            "status": "LOCAL_SMOKE_EXECUTION_PASSED",
            "baseline_commit": "9b50c56556f902b62ecc4a7e2e511ca0f316da2d",
            "current_commit": "551e5af50dc6597069e57af047213f61e40246d6",
            "execution_started": True,
            "real_benchmarks_executed": True,
            "all_commands_succeeded": True,
            "all_suite_outputs_exist": True,
            "all_suites_passed": True,
            "all_boundary_flags_false": True,
            "strict_plan_gate_closed": False,
            "full_remote_benchmark_executed": False,
            "github_query_performed": False,
            "github_write_performed": False,
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "active_apply_performed": False,
            "cron_or_gateway_mutation_performed": False,
            "suite_passed": {record["benchmark"]: True for record in suite_records},
        },
    )
    produced_files = []
    for record in suite_records:
        path = Path(record["path"])
        produced_files.append(
            {
                "path": str(path),
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
                "suite_or_summary_status": record["benchmark"],
                "passed": True,
                "dry_run": False,
                "under_approved_run_output_root": True,
                "under_allowed_hse_output_root": True,
            }
        )
    produced_files.append(
        {
            "path": str(summary),
            "sha256": _sha(summary),
            "bytes": summary.stat().st_size,
            "suite_or_summary_status": "LOCAL_SMOKE_EXECUTION_PASSED",
            "under_approved_run_output_root": True,
            "under_allowed_hse_output_root": True,
        }
    )
    verification = _write_json(
        tmp_path / "reports" / "current_baseline_verification.json",
        {
            "schema_version": "hse-current-baseline-local-smoke-verification-v1",
            "approved_output_root": str(output_root),
            "source_preflight_report": {"path": str(preflight), "sha256": _sha(preflight)},
            "execution_summary": {"path": str(summary), "sha256": _sha(summary), "status": "LOCAL_SMOKE_EXECUTION_PASSED"},
            "invariants": {
                "status_passed": True,
                "all_commands_succeeded": True,
                "all_suite_outputs_exist": True,
                "all_suites_passed": True,
                "all_boundary_flags_false": True,
                "all_executed_commands_removed_dry_run": True,
                "all_outputs_under_approved_root": True,
                "all_outputs_under_hse_allowed_root": True,
                "strict_plan_gate_closed": False,
                "full_remote_benchmark_executed": False,
                "github_query_performed": False,
                "github_write_performed": False,
                "provider_or_model_spend_performed": False,
                "network_calls_performed": False,
                "active_apply_performed": False,
                "cron_or_gateway_mutation_performed": False,
            },
            "produced_files_manifest": produced_files,
            "not_claimed": [
                "full_remote_benchmark",
                "provider_api_spend",
                "github_query_or_write",
                "active_apply",
                "cron_or_gateway_mutation",
                "strict_plan_gate_closure",
            ],
        },
    )
    paths.update({"current_preflight": preflight, "current_verification": verification, "current_summary": summary})
    return paths


def test_current_baseline_benchmark_strict_plan_closure_closes_from_revalidation_smoke(tmp_path: Path):
    paths = _valid_current_baseline_artifacts(tmp_path)

    result = write_current_baseline_benchmark_strict_plan_closure(
        benchmark_backfill_path=paths["backfill"],
        approval_packet_path=paths["approval"],
        current_baseline_preflight_path=paths["current_preflight"],
        execution_verification_path=paths["current_verification"],
        execution_summary_path=paths["current_summary"],
        plan_path=paths["plan"],
        output_dir=tmp_path / "current-closure",
        generated_at="2026-07-04T04:10:00+09:00",
        approval_source="rec action GO: strict_plan_benchmark_closure_current_baseline_go",
    )

    report = json.loads(Path(result["report_path"]).read_text())
    assert report["schema_version"] == "hse-benchmark-strict-plan-closure-v1"
    assert report["status"] == STRICT_PLAN_BENCHMARK_GATE_CLOSED
    assert report["strict_plan_gate_closed"] is True
    assert report["benchmark_gate_passed"] is True
    assert report["strict_plan_scope"] == "phase1_phase2_benchmark_regression_gate"
    assert report["current_baseline_revalidation_closed"] is True
    assert report["benchmark_subjects"]["baseline"]["hermes_source"]["commit"] == "9b50c56556f902b62ecc4a7e2e511ca0f316da2d"
    assert report["benchmark_subjects"]["current"]["hermes_source"]["commit"] == "551e5af50dc6597069e57af047213f61e40246d6"
    assert report["closed_criteria"]["current_baseline_preflight_ready"] is True
    assert report["closed_criteria"]["current_baseline_execution_verification_passed"] is True
    assert report["closed_criteria"]["artifact_hashes_verified"] is True
    assert report["closed_criteria"]["all_outputs_under_approved_root"] is True
    assert report["blocked_by"] == []
    assert report["full_remote_benchmark_executed"] is False
    assert report["github_query_performed"] is False
    assert report["github_write_performed"] is False
    assert report["provider_or_model_spend_performed"] is False
    assert report["network_calls_performed"] is False
    assert report["active_apply_performed"] is False


def test_current_baseline_benchmark_strict_plan_closure_blocks_hash_mismatch(tmp_path: Path):
    paths = _valid_current_baseline_artifacts(tmp_path)
    verification = json.loads(paths["current_verification"].read_text())
    verification["produced_files_manifest"][0]["sha256"] = "f" * 64
    paths["current_verification"].write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")

    result = write_current_baseline_benchmark_strict_plan_closure(
        benchmark_backfill_path=paths["backfill"],
        approval_packet_path=paths["approval"],
        current_baseline_preflight_path=paths["current_preflight"],
        execution_verification_path=paths["current_verification"],
        execution_summary_path=paths["current_summary"],
        plan_path=paths["plan"],
        output_dir=tmp_path / "current-closure-blocked",
        generated_at="2026-07-04T04:10:00+09:00",
        approval_source="rec action GO: strict_plan_benchmark_closure_current_baseline_go",
    )

    report = json.loads(Path(result["report_path"]).read_text())
    assert report["status"] == BLOCKED_BENCHMARK_EVIDENCE_INCOMPLETE
    assert report["strict_plan_gate_closed"] is False
    assert report["benchmark_gate_passed"] is False
    assert "artifact_hash_mismatch" in report["blocked_by"]
    assert report["github_query_performed"] is False
    assert report["github_write_performed"] is False

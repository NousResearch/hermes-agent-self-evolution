"""Tests for HSE real benchmark approval packets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution.local_completion.real_benchmark_approval_packet import (
    AWAITING_EXPLICIT_BENCHMARK_APPROVAL,
    SAFETY_REVIEW_PASS_FAIL_CLOSED,
    write_real_benchmark_approval_packet,
)


def _backfill_report() -> dict:
    return {
        "schema_version": "hse-benchmark-gate-backfill-v1",
        "gate_id": "B0",
        "status": "BLOCKED_BY_BENCHMARK_APPROVAL",
        "strict_plan_gate_closed": False,
        "benchmark_gate_passed": None,
        "real_benchmarks_executed": False,
        "real_benchmark_execution_approved": False,
        "current_authorized_budget_usd": 0,
        "github": {
            "queried": False,
            "pr_created": False,
            "push_performed": False,
            "merge_performed": False,
            "publication_deferred": True,
        },
        "safety_invariants": {
            "network_calls_performed": False,
            "active_runtime_mutation": False,
            "credentials_accessed": False,
            "external_publication_performed": False,
        },
        "benchmark_subjects": {
            "baseline": {"subject_id": "baseline", "hermes_source": {"commit": "88d1d6206"}},
            "current": {"subject_id": "current", "hermes_source": {"commit": "9b50c5655"}},
        },
    }


def test_real_benchmark_approval_packet_records_required_fields_and_no_execution(tmp_path):
    backfill_path = tmp_path / "benchmark_gate_backfill.json"
    backfill_path.write_text(json.dumps(_backfill_report(), indent=2, sort_keys=True) + "\n")

    result = write_real_benchmark_approval_packet(
        backfill_report_path=backfill_path,
        output_dir=tmp_path / "approval",
        generated_at="2026-07-03T12:50:00+09:00",
    )

    packet_path = Path(result["approval_packet_path"])
    markdown_path = Path(result["approval_markdown_path"])
    safety_path = Path(result["safety_review_path"])
    snapshot_path = Path(result["backfill_snapshot_path"])
    packet = json.loads(packet_path.read_text())
    safety = json.loads(safety_path.read_text())

    assert packet["schema_version"] == "hse-real-benchmark-approval-packet-v1"
    assert packet["gate_id"] == "B0-RBA"
    assert packet["source_backfill"]["status"] == "BLOCKED_BY_BENCHMARK_APPROVAL"
    assert packet["status"] == AWAITING_EXPLICIT_BENCHMARK_APPROVAL
    assert packet["approval_complete"] is False
    assert packet["execution_started"] is False
    assert packet["real_benchmarks_executed"] is False
    assert packet["real_benchmark_execution_approved"] is False
    assert packet["current_authorized_budget_usd"] == 0
    assert packet["approved_runtime_minutes"] is None
    assert packet["network_provider_spend_allowed"] is False
    assert packet["baseline_materialization_allowed"] is False
    assert packet["current_materialization_allowed"] is False
    assert packet["candidate_only"] is True
    assert packet["apply_ready"] is False
    assert packet["github"]["queried"] is False
    assert packet["github"]["push_performed"] is False
    assert packet["safety_invariants"]["network_calls_performed"] is False
    assert packet["safety_invariants"]["credentials_accessed"] is False
    assert packet["execution_boundaries"]["provider_or_model_spend_performed"] is False
    assert packet["execution_boundaries"]["benchmark_process_started"] is False
    assert packet["execution_boundaries"]["gateway_restart_or_reload_performed"] is False
    assert packet["required_approval_fields"] == [
        "benchmark_suites",
        "max_budget_usd_or_krw",
        "max_runtime_minutes",
        "network_provider_api_spend_allowed",
        "baseline_materialization_allowed",
        "current_materialization_allowed",
        "regression_thresholds",
        "allowed_write_roots",
        "rollback_plan",
        "human_approval_source",
    ]
    assert packet["approval_form"] == {
        "benchmark_suites": ["TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples"],
        "max_budget_usd": None,
        "max_budget_krw": None,
        "max_runtime_minutes": None,
        "network_provider_api_spend_allowed": False,
        "baseline_materialization_allowed": False,
        "current_materialization_allowed": False,
        "regression_thresholds": {
            "Phase2 PLAN-scale tool-selection triples": "no_aggregate_or_per_tool_regression_beyond_gate",
            "TBLite": "within_2_percent_or_better",
            "YC-Bench": "no_material_regression",
        },
        "allowed_write_roots": [],
        "rollback_plan": None,
        "human_approval_source": None,
        "approver_metadata": None,
    }
    assert {suite["name"] for suite in packet["requested_benchmark_suites"]} >= {"TBLite", "YC-Bench"}
    assert packet["regression_thresholds"]["TBLite"] == "within_2_percent_or_better"
    assert packet["regression_thresholds"]["YC-Bench"] == "no_material_regression"
    assert safety["schema_version"] == "hse-real-benchmark-safety-review-v1"
    assert safety["status"] == SAFETY_REVIEW_PASS_FAIL_CLOSED
    assert safety["packet_approved_for_execution"] is False
    assert safety["blockers"] == ["awaiting_explicit_human_benchmark_approval"]
    assert snapshot_path.exists()
    markdown = markdown_path.read_text()
    assert "AWAITING_EXPLICIT_BENCHMARK_APPROVAL" in markdown
    assert "real_benchmarks_executed=false" in markdown
    assert "NO_GITHUB_WRITE" in markdown
    assert "not approval to execute" in markdown


def test_real_benchmark_approval_packet_rejects_non_blocked_backfill(tmp_path):
    report = _backfill_report()
    report["status"] = "PASS"
    backfill_path = tmp_path / "benchmark_gate_backfill.json"
    backfill_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="source backfill must be BLOCKED_BY_BENCHMARK_APPROVAL"):
        write_real_benchmark_approval_packet(
            backfill_report_path=backfill_path,
            output_dir=tmp_path / "approval",
            generated_at="2026-07-03T12:50:00+09:00",
        )


def test_real_benchmark_approval_packet_can_record_explicit_approval_without_starting_execution(tmp_path):
    backfill_path = tmp_path / "benchmark_gate_backfill.json"
    backfill_path.write_text(json.dumps(_backfill_report(), indent=2, sort_keys=True) + "\n")

    result = write_real_benchmark_approval_packet(
        backfill_report_path=backfill_path,
        output_dir=tmp_path / "approval-approved",
        generated_at="2026-07-03T12:50:00+09:00",
        benchmark_suites=["TBLite", "YC-Bench"],
        max_budget_usd=25,
        max_runtime_minutes=180,
        network_provider_api_spend_allowed=True,
        baseline_materialization_allowed=True,
        current_materialization_allowed=True,
        human_approval_source="test-human-approval",
        allowed_write_roots=[str(tmp_path / "approved-output-root")],
        rollback_plan={"strategy": "delete generated benchmark output root on cancellation"},
    )

    packet = json.loads(Path(result["approval_packet_path"]).read_text())
    assert packet["approval_complete"] is True
    assert packet["real_benchmark_execution_approved"] is True
    assert packet["execution_started"] is False
    assert packet["real_benchmarks_executed"] is False
    assert packet["current_authorized_budget_usd"] == 25
    assert packet["approved_runtime_minutes"] == 180
    assert packet["required_next_action"] == "run_real_benchmark_preflight_then_execute_under_packet"


def test_real_benchmark_approval_packet_accepts_explicit_local_only_zero_budget_approval(tmp_path):
    backfill_path = tmp_path / "benchmark_gate_backfill.json"
    backfill_path.write_text(json.dumps(_backfill_report(), indent=2, sort_keys=True) + "\n")

    result = write_real_benchmark_approval_packet(
        backfill_report_path=backfill_path,
        output_dir=tmp_path / "approval-local-only",
        generated_at="2026-07-03T23:47:29+09:00",
        benchmark_suites=["TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples"],
        max_budget_usd=0,
        max_budget_krw=0,
        max_runtime_minutes=90,
        network_provider_api_spend_allowed=False,
        baseline_materialization_allowed=True,
        current_materialization_allowed=True,
        human_approval_source="discord_message:Sunwoo:HSE finite local-only approval",
        allowed_write_roots=[str(tmp_path / "hse-real-benchmark" / "real-run-20260703_1310")],
        rollback_plan={
            "delete_future_output_root_if_created": str(tmp_path / "hse-real-benchmark" / "real-run-20260703_1310"),
            "remove_disposable_worktrees_if_created": True,
            "rollback_plan_verified": True,
        },
    )

    packet = json.loads(Path(result["approval_packet_path"]).read_text())
    safety = json.loads(Path(result["safety_review_path"]).read_text())
    assert packet["status"] == "APPROVAL_RECORDED_NOT_EXECUTED"
    assert packet["missing_approval_fields"] == []
    assert packet["approval_complete"] is True
    assert packet["real_benchmark_execution_approved"] is True
    assert packet["execution_started"] is False
    assert packet["real_benchmarks_executed"] is False
    assert packet["current_authorized_budget_usd"] == 0
    assert packet["current_authorized_budget_krw"] == 0
    assert packet["approved_runtime_minutes"] == 90
    assert packet["approval_form"]["network_provider_api_spend_allowed"] is False
    assert packet["network_provider_spend_allowed"] is False
    assert packet["baseline_materialization_allowed"] is True
    assert packet["current_materialization_allowed"] is True
    assert safety["status"] == "PASS_APPROVAL_RECORDED_NOT_EXECUTED"
    assert safety["blockers"] == []
    assert safety["packet_approved_for_execution"] is True
    assert packet["required_next_action"] == "run_real_benchmark_preflight_then_execute_under_packet"


@pytest.mark.parametrize(
    ("max_budget_usd", "max_budget_krw"),
    [
        (float("inf"), None),
        (float("nan"), None),
        (None, float("inf")),
        (None, float("nan")),
    ],
)
def test_real_benchmark_approval_packet_rejects_non_finite_budget_limits(
    tmp_path, max_budget_usd, max_budget_krw
):
    backfill_path = tmp_path / "benchmark_gate_backfill.json"
    backfill_path.write_text(json.dumps(_backfill_report(), indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="budget limits must be finite"):
        write_real_benchmark_approval_packet(
            backfill_report_path=backfill_path,
            output_dir=tmp_path / "approval-non-finite-budget",
            generated_at="2026-07-03T12:50:00+09:00",
            benchmark_suites=["TBLite", "YC-Bench"],
            max_budget_usd=max_budget_usd,
            max_budget_krw=max_budget_krw,
            max_runtime_minutes=90,
            network_provider_api_spend_allowed=True,
            baseline_materialization_allowed=True,
            current_materialization_allowed=True,
            human_approval_source="test-human-approval",
            allowed_write_roots=[str(tmp_path / "approved-output-root")],
            rollback_plan={"strategy": "delete generated benchmark output root on cancellation"},
        )

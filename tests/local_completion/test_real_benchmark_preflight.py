"""Tests for HSE real benchmark preflight manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution.local_completion.real_benchmark_preflight import (
    PREFLIGHT_EXECUTION_READY_NOT_STARTED,
    PREFLIGHT_RECORDED_NOT_EXECUTABLE,
    write_real_benchmark_preflight,
)


def _approval_packet(tmp_path: Path, *, executed: bool = False, status: str = "AWAITING_EXPLICIT_BENCHMARK_APPROVAL") -> Path:
    backfill_path = tmp_path / "benchmark_gate_backfill.json"
    backfill_path.write_text(
        json.dumps(
            {
                "schema_version": "hse-benchmark-gate-backfill-v1",
                "gate_id": "B0",
                "status": "BLOCKED_BY_BENCHMARK_APPROVAL",
                "strict_plan_gate_closed": False,
                "benchmark_gate_passed": None,
                "real_benchmarks_executed": False,
                "real_benchmark_execution_approved": False,
                "benchmark_subjects": {
                    "baseline": {"subject_id": "baseline", "hermes_source": {"commit": "88d1d6206"}},
                    "current": {"subject_id": "current", "hermes_source": {"commit": "9b50c5655"}},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    packet_path = tmp_path / "real_benchmark_approval_packet.json"
    packet_path.write_text(
        json.dumps(
            {
                "schema_version": "hse-real-benchmark-approval-packet-v1",
                "gate_id": "B0-RBA",
                "status": status,
                "approval_complete": False,
                "execution_started": executed,
                "real_benchmarks_executed": executed,
                "real_benchmark_execution_approved": False,
                "current_authorized_budget_usd": 0,
                "current_authorized_budget_krw": 0,
                "approved_runtime_minutes": None,
                "network_provider_spend_allowed": False,
                "baseline_materialization_allowed": False,
                "current_materialization_allowed": False,
                "candidate_only": True,
                "apply_ready": False,
                "requested_benchmark_suites": [
                    {"name": "TBLite", "execution_started": False, "real_result_required_for_strict_plan_gate": True},
                    {"name": "YC-Bench", "execution_started": False, "real_result_required_for_strict_plan_gate": True},
                    {
                        "name": "Phase2 PLAN-scale tool-selection triples",
                        "execution_started": False,
                        "real_result_required_for_strict_plan_gate": True,
                    },
                ],
                "regression_thresholds": {
                    "TBLite": "within_2_percent_or_better",
                    "YC-Bench": "no_material_regression",
                    "Phase2 PLAN-scale tool-selection triples": "no_aggregate_or_per_tool_regression_beyond_gate",
                },
                "missing_approval_fields": [
                    "max_budget_usd_or_krw",
                    "max_runtime_minutes",
                    "network_provider_api_spend_allowed",
                    "baseline_materialization_allowed",
                    "current_materialization_allowed",
                    "human_approval_source",
                    "allowed_write_roots",
                    "rollback_plan",
                ],
                "approval_form": {
                    "benchmark_suites": ["TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples"],
                    "max_budget_usd": None,
                    "max_budget_krw": None,
                    "max_runtime_minutes": None,
                    "network_provider_api_spend_allowed": False,
                    "baseline_materialization_allowed": False,
                    "current_materialization_allowed": False,
                    "allowed_write_roots": [],
                    "rollback_plan": None,
                    "human_approval_source": None,
                    "approver_metadata": None,
                },
                "source_backfill": {
                    "path": str(backfill_path),
                    "status": "BLOCKED_BY_BENCHMARK_APPROVAL",
                    "strict_plan_gate_closed": False,
                    "benchmark_gate_passed": None,
                },
                "github": {
                    "queried": False,
                    "pr_created": False,
                    "push_performed": False,
                    "merge_performed": False,
                    "publication_deferred": True,
                },
                "execution_boundaries": {
                    "benchmark_process_started": False,
                    "provider_or_model_spend_performed": False,
                    "network_calls_performed": False,
                    "github_write_performed": False,
                    "active_apply_performed": False,
                    "gateway_restart_or_reload_performed": False,
                    "cron_mutation_performed": False,
                    "credential_or_secret_access_performed": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return packet_path


def test_real_benchmark_preflight_records_plan_but_keeps_execution_blocked(tmp_path):
    packet_path = _approval_packet(tmp_path)
    future_output_root = tmp_path / "repo" / "output" / "hse-real-benchmark" / "run-001"

    result = write_real_benchmark_preflight(
        approval_packet_path=packet_path,
        output_dir=tmp_path / "preflight",
        future_output_root=future_output_root,
        generated_at="2026-07-03T13:10:00+09:00",
        dry_run=True,
    )

    preflight_path = Path(result["preflight_report_path"])
    markdown_path = Path(result["preflight_markdown_path"])
    command_preview_path = Path(result["command_preview_path"])
    approval_snapshot_path = Path(result["approval_packet_snapshot_path"])
    preflight = json.loads(preflight_path.read_text())
    command_preview = json.loads(command_preview_path.read_text())

    assert preflight["schema_version"] == "hse-real-benchmark-preflight-v1"
    assert preflight["gate_id"] == "B0-RBP"
    assert preflight["status"] == PREFLIGHT_RECORDED_NOT_EXECUTABLE
    assert preflight["preflight_passed"] is True
    assert preflight["strict_plan_gate_closed"] is False
    assert preflight["execution_ready"] is False
    assert preflight["execution_started"] is False
    assert preflight["real_benchmarks_executed"] is False
    assert preflight["real_benchmark_execution_approved"] is False
    assert preflight["current_authorized_budget_usd"] == 0
    assert preflight["network_provider_spend_allowed"] is False
    assert preflight["dry_run_only"] is True
    assert preflight["command_dry_run"]["dry_run"] is True
    assert preflight["command_dry_run"]["benchmark_commands_started"] is False
    assert preflight["command_dry_run"]["command_preview_generated"] is True
    assert all("--dry-run" in command["argv"] for command in command_preview["commands"])
    assert preflight["baseline_materialization_plan"]["materialization_started"] is False
    assert preflight["baseline_materialization_plan"]["worktree_created"] is False
    assert preflight["baseline_materialization_plan"]["source_commit"] == "88d1d6206"
    assert preflight["current_materialization_plan"]["materialization_started"] is False
    assert preflight["current_materialization_plan"]["worktree_created"] is False
    assert preflight["current_materialization_plan"]["source_commit"] == "9b50c5655"
    assert preflight["output_root_guard"]["passed"] is True
    assert preflight["output_root_guard"]["future_output_root_exists_now"] is False
    assert preflight["output_root_guard"]["fresh_output_required"] is True
    assert preflight["write_root_guard"]["allowed_write_roots"] == [str(future_output_root)]
    assert preflight["rollback_cleanup_plan"]["rollback_plan_verified"] is True
    assert preflight["rollback_cleanup_plan"]["cleanup_started"] is False
    assert "awaiting_explicit_human_benchmark_approval" in preflight["blocked_by"]
    assert "max_budget_usd_or_krw" in preflight["blocked_by"]
    assert preflight["github_policy"] == "NO_GITHUB_WRITE"
    assert preflight["github"]["queried"] is False
    assert preflight["execution_boundaries"]["benchmark_process_started"] is False
    assert preflight["execution_boundaries"]["provider_or_model_spend_performed"] is False
    assert preflight["execution_boundaries"]["github_write_performed"] is False
    assert approval_snapshot_path.exists()
    markdown = markdown_path.read_text()
    assert "PREFLIGHT_RECORDED_NOT_EXECUTABLE" in markdown
    assert "execution_ready=false" in markdown
    assert "dry_run_only=true" in markdown
    assert "NO_GITHUB_WRITE" in markdown
    assert "not approval to execute" in markdown


def test_real_benchmark_preflight_accepts_local_only_zero_budget_approval_as_execution_ready_not_started(tmp_path):
    backfill_path = tmp_path / "benchmark_gate_backfill.json"
    backfill_path.write_text(
        json.dumps(
            {
                "schema_version": "hse-benchmark-gate-backfill-v1",
                "gate_id": "B0",
                "status": "BLOCKED_BY_BENCHMARK_APPROVAL",
                "strict_plan_gate_closed": False,
                "benchmark_gate_passed": None,
                "real_benchmarks_executed": False,
                "real_benchmark_execution_approved": False,
                "benchmark_subjects": {
                    "baseline": {"subject_id": "baseline", "hermes_source": {"commit": "88d1d6206"}},
                    "current": {"subject_id": "current", "hermes_source": {"commit": "9b50c5655"}},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    packet_path = tmp_path / "real_benchmark_approval_packet.json"
    packet_path.write_text(
        json.dumps(
            {
                "schema_version": "hse-real-benchmark-approval-packet-v1",
                "gate_id": "B0-RBA",
                "status": "APPROVAL_RECORDED_NOT_EXECUTED",
                "approval_complete": True,
                "execution_started": False,
                "real_benchmarks_executed": False,
                "real_benchmark_execution_approved": True,
                "current_authorized_budget_usd": 0,
                "current_authorized_budget_krw": 0,
                "approved_runtime_minutes": 90,
                "network_provider_spend_allowed": False,
                "baseline_materialization_allowed": True,
                "current_materialization_allowed": True,
                "candidate_only": True,
                "apply_ready": False,
                "requested_benchmark_suites": [
                    {"name": "TBLite", "execution_started": False, "real_result_required_for_strict_plan_gate": True},
                    {"name": "YC-Bench", "execution_started": False, "real_result_required_for_strict_plan_gate": True},
                ],
                "regression_thresholds": {
                    "TBLite": "within_2_percent_or_better",
                    "YC-Bench": "no_material_regression",
                },
                "missing_approval_fields": [],
                "source_backfill": {
                    "path": str(backfill_path),
                    "status": "BLOCKED_BY_BENCHMARK_APPROVAL",
                    "strict_plan_gate_closed": False,
                    "benchmark_gate_passed": None,
                },
                "github": {
                    "queried": False,
                    "pr_created": False,
                    "push_performed": False,
                    "merge_performed": False,
                    "publication_deferred": True,
                },
                "execution_boundaries": {
                    "benchmark_process_started": False,
                    "provider_or_model_spend_performed": False,
                    "network_calls_performed": False,
                    "github_write_performed": False,
                    "active_apply_performed": False,
                    "gateway_restart_or_reload_performed": False,
                    "cron_mutation_performed": False,
                    "credential_or_secret_access_performed": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    future_output_root = tmp_path / "repo" / "output" / "hse-real-benchmark" / "run-001"

    result = write_real_benchmark_preflight(
        approval_packet_path=packet_path,
        output_dir=tmp_path / "preflight-local-only",
        future_output_root=future_output_root,
        generated_at="2026-07-03T23:47:29+09:00",
        dry_run=True,
    )

    preflight = json.loads(Path(result["preflight_report_path"]).read_text())
    assert preflight["status"] == PREFLIGHT_EXECUTION_READY_NOT_STARTED
    assert preflight["preflight_passed"] is True
    assert preflight["approval_complete"] is True
    assert preflight["real_benchmark_execution_approved"] is True
    assert preflight["execution_ready"] is True
    assert preflight["network_provider_spend_allowed"] is False
    assert preflight["baseline_materialization_plan"]["allowed_by_packet"] is True
    assert preflight["current_materialization_plan"]["allowed_by_packet"] is True
    assert preflight["blocked_by"] == []
    assert preflight["execution_started"] is False
    assert preflight["real_benchmarks_executed"] is False
    assert preflight["execution_boundaries"]["provider_or_model_spend_performed"] is False
    assert preflight["execution_boundaries"]["network_calls_performed"] is False



def test_real_benchmark_preflight_requires_explicit_dry_run(tmp_path):
    packet_path = _approval_packet(tmp_path)
    with pytest.raises(ValueError, match="dry_run must be exactly true"):
        write_real_benchmark_preflight(
            approval_packet_path=packet_path,
            output_dir=tmp_path / "preflight",
            future_output_root=tmp_path / "repo" / "output" / "hse-real-benchmark" / "run-001",
            generated_at="2026-07-03T13:10:00+09:00",
            dry_run=False,
        )


def test_real_benchmark_preflight_rejects_unsafe_output_roots(tmp_path):
    packet_path = _approval_packet(tmp_path)
    existing_output_root = tmp_path / "repo" / "output" / "hse-real-benchmark" / "existing"
    existing_output_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="future_output_root must not already exist"):
        write_real_benchmark_preflight(
            approval_packet_path=packet_path,
            output_dir=tmp_path / "preflight-existing",
            future_output_root=existing_output_root,
            generated_at="2026-07-03T13:10:00+09:00",
            dry_run=True,
        )

    with pytest.raises(ValueError, match="future_output_root must be under an output/hse-real-benchmark root"):
        write_real_benchmark_preflight(
            approval_packet_path=packet_path,
            output_dir=tmp_path / "preflight-outside",
            future_output_root=tmp_path / "repo" / "reports" / "hse-real-benchmark" / "run-001",
            generated_at="2026-07-03T13:10:00+09:00",
            dry_run=True,
        )

    with pytest.raises(ValueError, match="future_output_root must not overlap input artifacts"):
        write_real_benchmark_preflight(
            approval_packet_path=packet_path,
            output_dir=tmp_path / "preflight-overlap",
            future_output_root=packet_path.parent,
            generated_at="2026-07-03T13:10:00+09:00",
            dry_run=True,
        )


def test_real_benchmark_preflight_rejects_already_executed_approval_packet(tmp_path):
    packet_path = _approval_packet(tmp_path, executed=True)
    with pytest.raises(ValueError, match="approval packet must not have started execution"):
        write_real_benchmark_preflight(
            approval_packet_path=packet_path,
            output_dir=tmp_path / "preflight",
            future_output_root=tmp_path / "repo" / "output" / "hse-real-benchmark" / "run-001",
            generated_at="2026-07-03T13:10:00+09:00",
            dry_run=True,
        )

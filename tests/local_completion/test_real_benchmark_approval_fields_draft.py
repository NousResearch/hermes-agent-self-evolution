"""Tests for HSE real benchmark approval fields draft artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution.local_completion.real_benchmark_approval_fields_draft import (
    APPROVAL_FIELDS_DRAFT_RECORDED_NOT_EXECUTABLE,
    write_real_benchmark_approval_fields_draft,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _approval_packet(tmp_path: Path, *, approval_complete: bool = False, executed: bool = False) -> Path:
    packet = {
        "schema_version": "hse-real-benchmark-approval-packet-v1",
        "gate_id": "B0-RBA",
        "status": "APPROVAL_RECORDED_NOT_EXECUTED" if approval_complete else "AWAITING_EXPLICIT_BENCHMARK_APPROVAL",
        "approval_complete": approval_complete,
        "required_approval_fields": [
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
        ],
        "approval_form": {
            "benchmark_suites": ["TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples"],
            "max_budget_usd": 25 if approval_complete else None,
            "max_budget_krw": None,
            "max_runtime_minutes": 90 if approval_complete else None,
            "network_provider_api_spend_allowed": approval_complete,
            "baseline_materialization_allowed": approval_complete,
            "current_materialization_allowed": approval_complete,
            "regression_thresholds": {
                "TBLite": "within_2_percent_or_better",
                "YC-Bench": "no_material_regression",
                "Phase2 PLAN-scale tool-selection triples": "no_aggregate_or_per_tool_regression_beyond_gate",
            },
            "allowed_write_roots": [str(tmp_path / "repo" / "output" / "hse-real-benchmark" / "run-001")]
            if approval_complete
            else [],
            "rollback_plan": {"cleanup": "remove disposable worktrees and output root"} if approval_complete else None,
            "human_approval_source": "test-approved" if approval_complete else None,
            "approver_metadata": None,
        },
        "missing_approval_fields": []
        if approval_complete
        else [
            "max_budget_usd_or_krw",
            "max_runtime_minutes",
            "network_provider_api_spend_allowed",
            "baseline_materialization_allowed",
            "current_materialization_allowed",
            "human_approval_source",
            "allowed_write_roots",
            "rollback_plan",
        ],
        "execution_started": executed,
        "real_benchmarks_executed": executed,
        "real_benchmark_execution_approved": approval_complete,
        "current_authorized_budget_usd": 25 if approval_complete else 0,
        "current_authorized_budget_krw": 0,
        "approved_runtime_minutes": 90 if approval_complete else None,
        "network_provider_spend_allowed": approval_complete,
        "baseline_materialization_allowed": approval_complete,
        "current_materialization_allowed": approval_complete,
        "human_approval_source": "test-approved" if approval_complete else None,
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
        "allowed_write_roots": [str(tmp_path / "repo" / "output" / "hse-real-benchmark" / "run-001")]
        if approval_complete
        else [],
        "rollback_plan": {"cleanup": "remove disposable worktrees and output root"},
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
    }
    return _write_json(tmp_path / "real_benchmark_approval_packet.json", packet)


def _preflight_report(tmp_path: Path, approval_path: Path, *, execution_ready: bool = False, executed: bool = False) -> Path:
    future_output_root = tmp_path / "repo" / "output" / "hse-real-benchmark" / "run-001"
    preflight = {
        "schema_version": "hse-real-benchmark-preflight-v1",
        "gate_id": "B0-RBP",
        "status": "PREFLIGHT_EXECUTION_READY_NOT_STARTED" if execution_ready else "PREFLIGHT_RECORDED_NOT_EXECUTABLE",
        "preflight_passed": True,
        "strict_plan_gate_closed": False,
        "execution_ready": execution_ready,
        "dry_run_only": not execution_ready,
        "execution_started": executed,
        "real_benchmarks_executed": executed,
        "real_benchmark_execution_approved": execution_ready,
        "approval_complete": execution_ready,
        "current_authorized_budget_usd": 25 if execution_ready else 0,
        "current_authorized_budget_krw": 0,
        "network_provider_spend_allowed": execution_ready,
        "github_policy": "NO_GITHUB_WRITE",
        "benchmark_suites": ["TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples"],
        "regression_thresholds": {
            "TBLite": "within_2_percent_or_better",
            "YC-Bench": "no_material_regression",
            "Phase2 PLAN-scale tool-selection triples": "no_aggregate_or_per_tool_regression_beyond_gate",
        },
        "source_approval_packet": {
            "path": str(approval_path),
            "status": "APPROVAL_RECORDED_NOT_EXECUTED" if execution_ready else "AWAITING_EXPLICIT_BENCHMARK_APPROVAL",
            "approval_complete": execution_ready,
            "execution_started": False,
            "real_benchmarks_executed": False,
        },
        "blocked_by": []
        if execution_ready
        else [
            "awaiting_explicit_human_benchmark_approval",
            "max_budget_usd_or_krw",
            "max_runtime_minutes",
            "network_provider_api_spend_allowed",
            "baseline_materialization_allowed",
            "current_materialization_allowed",
            "human_approval_source",
            "allowed_write_roots",
            "rollback_plan",
        ],
        "baseline_materialization_plan": {
            "subject": "baseline",
            "source_commit": "88d1d6206",
            "planned_worktree_path": str(future_output_root.parent / "_worktrees" / "run-001" / "baseline-88d1d6206"),
            "allowed_by_packet": execution_ready,
            "materialization_started": False,
            "worktree_created": False,
            "cleanup_required_if_created": True,
            "cleanup_started": False,
        },
        "current_materialization_plan": {
            "subject": "current",
            "source_commit": "9b50c5655",
            "planned_worktree_path": str(future_output_root.parent / "_worktrees" / "run-001" / "current-9b50c5655"),
            "allowed_by_packet": execution_ready,
            "materialization_started": False,
            "worktree_created": False,
            "cleanup_required_if_created": True,
            "cleanup_started": False,
        },
        "output_root_guard": {
            "passed": True,
            "future_output_root": str(future_output_root),
            "future_output_root_exists_now": False,
            "fresh_output_required": True,
            "benchmark_output_written_now": False,
        },
        "write_root_guard": {
            "allowed_write_roots": [str(future_output_root)],
            "future_output_root": str(future_output_root),
            "benchmark_output_written_now": False,
            "passed": True,
        },
        "rollback_cleanup_plan": {
            "rollback_plan_verified": True,
            "cleanup_started": False,
            "delete_future_output_root_if_created": str(future_output_root),
            "remove_disposable_worktrees_if_created": True,
        },
        "command_dry_run": {
            "dry_run": not execution_ready,
            "command_preview_generated": True,
            "benchmark_commands_started": False,
            "benchmark_process_started": False,
            "commands_have_dry_run_flag": True,
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
            "worktree_materialization_performed": False,
            "benchmark_output_written": False,
        },
    }
    return _write_json(tmp_path / "real_benchmark_preflight.json", preflight)


def test_approval_fields_draft_records_conservative_defaults_without_authorizing_execution(tmp_path):
    approval_path = _approval_packet(tmp_path)
    preflight_path = _preflight_report(tmp_path, approval_path)

    result = write_real_benchmark_approval_fields_draft(
        approval_packet_path=approval_path,
        preflight_report_path=preflight_path,
        output_dir=tmp_path / "draft",
        generated_at="2026-07-03T13:40:00+09:00",
    )

    draft_path = Path(result["approval_fields_draft_path"])
    markdown_path = Path(result["approval_fields_markdown_path"])
    approval_snapshot = Path(result["approval_packet_snapshot_path"])
    preflight_snapshot = Path(result["preflight_snapshot_path"])
    draft = json.loads(draft_path.read_text())

    assert draft["schema_version"] == "hse-real-benchmark-approval-fields-draft-v1"
    assert draft["gate_id"] == "B0-AFD"
    assert draft["status"] == APPROVAL_FIELDS_DRAFT_RECORDED_NOT_EXECUTABLE
    assert draft["draft_only"] is True
    assert draft["approval_draft_complete"] is True
    assert draft["approval_complete"] is False
    assert draft["real_benchmark_execution_approved"] is False
    assert draft["execution_ready"] is False
    assert draft["strict_plan_gate_closed"] is False
    assert draft["execution_started"] is False
    assert draft["real_benchmarks_executed"] is False
    assert draft["current_authorized_budget_usd"] == 0
    assert draft["current_authorized_budget_krw"] == 0
    assert draft["approved_runtime_minutes"] == 0
    assert draft["network_provider_spend_allowed"] is False
    assert draft["baseline_materialization_allowed"] is False
    assert draft["current_materialization_allowed"] is False
    assert draft["github_policy"] == "NO_GITHUB_WRITE"
    assert draft["future_execution_requires_explicit_human_go"] is True
    assert "not approval to execute" in draft["draft_notice"].lower()
    assert draft["source_approval_packet"]["status"] == "AWAITING_EXPLICIT_BENCHMARK_APPROVAL"
    assert draft["source_preflight"]["status"] == "PREFLIGHT_RECORDED_NOT_EXECUTABLE"
    assert draft["source_preflight"]["execution_ready"] is False
    assert draft["blocked_by"] == [
        "awaiting_explicit_human_benchmark_approval",
        "max_budget_usd_or_krw",
        "max_runtime_minutes",
        "network_provider_api_spend_allowed",
        "baseline_materialization_allowed",
        "current_materialization_allowed",
        "human_approval_source",
        "allowed_write_roots",
        "rollback_plan",
    ]
    fields = draft["draft_approval_fields"]
    assert set(fields) == set(draft["required_approval_fields"])
    assert fields["benchmark_suites"]["conservative_default"] == [
        "TBLite",
        "YC-Bench",
        "Phase2 PLAN-scale tool-selection triples",
    ]
    assert fields["max_budget_usd_or_krw"]["conservative_default"] == {"max_budget_usd": 0, "max_budget_krw": 0}
    assert fields["max_runtime_minutes"]["conservative_default"] == 0
    assert fields["network_provider_api_spend_allowed"]["conservative_default"] is False
    assert fields["baseline_materialization_allowed"]["conservative_default"] is False
    assert fields["current_materialization_allowed"]["conservative_default"] is False
    assert fields["allowed_write_roots"]["conservative_default"] == []
    assert fields["allowed_write_roots"]["candidate_for_human_review"] == [
        str(tmp_path / "repo" / "output" / "hse-real-benchmark" / "run-001")
    ]
    assert fields["rollback_plan"]["conservative_default"] is None
    assert fields["rollback_plan"]["candidate_for_human_review"]["remove_disposable_worktrees_if_created"] is True
    assert fields["human_approval_source"]["conservative_default"] is None
    assert all(field["approved_for_execution"] is False for field in fields.values())
    assert all(field["risk_notes"] for field in fields.values())
    assert draft["execution_boundaries"]["benchmark_process_started"] is False
    assert draft["execution_boundaries"]["worktree_materialization_performed"] is False
    assert draft["execution_boundaries"]["benchmark_output_written"] is False
    assert draft["execution_boundaries"]["provider_or_model_spend_performed"] is False
    assert approval_snapshot.exists()
    assert preflight_snapshot.exists()
    markdown = markdown_path.read_text()
    assert "APPROVAL_FIELDS_DRAFT_RECORDED_NOT_EXECUTABLE" in markdown
    assert "not approval to execute" in markdown
    assert "NO_GITHUB_WRITE" in markdown
    assert "strict_plan_gate_closed=false" in markdown


def test_approval_fields_draft_rejects_already_approved_packet(tmp_path):
    approval_path = _approval_packet(tmp_path, approval_complete=True)
    preflight_path = _preflight_report(tmp_path, approval_path, execution_ready=True)

    with pytest.raises(ValueError, match="approval packet must still be awaiting explicit benchmark approval"):
        write_real_benchmark_approval_fields_draft(
            approval_packet_path=approval_path,
            preflight_report_path=preflight_path,
            output_dir=tmp_path / "draft",
            generated_at="2026-07-03T13:40:00+09:00",
        )


def test_approval_fields_draft_rejects_executed_sources(tmp_path):
    approval_path = _approval_packet(tmp_path, executed=True)
    preflight_path = _preflight_report(tmp_path, approval_path)

    with pytest.raises(ValueError, match="approval packet must not have started execution"):
        write_real_benchmark_approval_fields_draft(
            approval_packet_path=approval_path,
            preflight_report_path=preflight_path,
            output_dir=tmp_path / "draft-executed-approval",
            generated_at="2026-07-03T13:40:00+09:00",
        )

    approval_path = _approval_packet(tmp_path / "preflight-executed")
    preflight_path = _preflight_report(tmp_path / "preflight-executed", approval_path, executed=True)
    with pytest.raises(ValueError, match="preflight report must not have started execution"):
        write_real_benchmark_approval_fields_draft(
            approval_packet_path=approval_path,
            preflight_report_path=preflight_path,
            output_dir=tmp_path / "draft-executed-preflight",
            generated_at="2026-07-03T13:40:00+09:00",
        )

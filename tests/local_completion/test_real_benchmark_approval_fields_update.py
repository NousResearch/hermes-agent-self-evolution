"""Tests for HSE real benchmark approval-fields user updates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution.local_completion.real_benchmark_approval_fields_update import (
    APPROVAL_FIELDS_UPDATE_RECORDED_NOT_EXECUTABLE,
    write_real_benchmark_approval_fields_update,
)


REQUIRED_FIELDS = [
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


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _source_draft(tmp_path: Path, *, execution_ready: bool = False) -> Path:
    future_root = tmp_path / "repo" / "output" / "hse-real-benchmark" / "real-run-20260703_1310"
    fields = {
        "benchmark_suites": {
            "conservative_default": ["TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples"],
            "candidate_for_human_review": ["TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples"],
            "approved_for_execution": False,
            "requires_explicit_human_approval": True,
            "risk_notes": ["suite scope changes runtime and comparability"],
            "source": "approval_packet.requested_benchmark_suites",
        },
        "max_budget_usd_or_krw": {
            "conservative_default": {"max_budget_usd": 0, "max_budget_krw": 0},
            "candidate_for_human_review": None,
            "approved_for_execution": False,
            "requires_explicit_human_approval": True,
            "risk_notes": ["nonzero budget permits spend"],
            "source": "conservative_default_zero_spend",
        },
        "max_runtime_minutes": {
            "conservative_default": 0,
            "candidate_for_human_review": None,
            "approved_for_execution": False,
            "requires_explicit_human_approval": True,
            "risk_notes": ["runtime must be explicitly bounded"],
            "source": "conservative_default_zero_runtime",
        },
        "network_provider_api_spend_allowed": {
            "conservative_default": False,
            "candidate_for_human_review": False,
            "approved_for_execution": False,
            "requires_explicit_human_approval": True,
            "risk_notes": ["true allows network/provider/API spend surfaces"],
            "source": "fail_closed_network_spend_boundary",
        },
        "baseline_materialization_allowed": {
            "conservative_default": False,
            "candidate_for_human_review": False,
            "approved_for_execution": False,
            "requires_explicit_human_approval": True,
            "risk_notes": ["true allows disposable baseline worktree creation"],
            "source": "preflight.baseline_materialization_plan",
        },
        "current_materialization_allowed": {
            "conservative_default": False,
            "candidate_for_human_review": False,
            "approved_for_execution": False,
            "requires_explicit_human_approval": True,
            "risk_notes": ["true allows disposable current worktree creation"],
            "source": "preflight.current_materialization_plan",
        },
        "regression_thresholds": {
            "conservative_default": {"TBLite": "within_2_percent_or_better"},
            "candidate_for_human_review": {"TBLite": "within_2_percent_or_better"},
            "approved_for_execution": False,
            "requires_explicit_human_approval": True,
            "risk_notes": ["thresholds must not be loosened to manufacture a pass"],
            "source": "approval_packet.regression_thresholds",
        },
        "allowed_write_roots": {
            "conservative_default": [],
            "candidate_for_human_review": [str(future_root)],
            "approved_for_execution": False,
            "requires_explicit_human_approval": True,
            "risk_notes": ["write roots permit later benchmark outputs"],
            "source": "preflight.write_root_guard.allowed_write_roots",
        },
        "rollback_plan": {
            "conservative_default": None,
            "candidate_for_human_review": {
                "delete_future_output_root_if_created": str(future_root),
                "remove_disposable_worktrees_if_created": True,
            },
            "approved_for_execution": False,
            "requires_explicit_human_approval": True,
            "risk_notes": ["rollback candidate is not accepted until approved explicitly"],
            "source": "preflight.rollback_cleanup_plan",
        },
        "human_approval_source": {
            "conservative_default": None,
            "candidate_for_human_review": None,
            "approved_for_execution": False,
            "requires_explicit_human_approval": True,
            "risk_notes": ["future approval source must cite explicit Sunwoo message"],
            "source": "future_explicit_sunwoo_message_required",
        },
    }
    draft = {
        "schema_version": "hse-real-benchmark-approval-fields-draft-v1",
        "gate_id": "B0-AFD",
        "status": "APPROVAL_FIELDS_DRAFT_RECORDED_NOT_EXECUTABLE",
        "draft_only": True,
        "approval_draft_complete": True,
        "approval_complete": False,
        "real_benchmark_execution_approved": False,
        "execution_ready": execution_ready,
        "strict_plan_gate_closed": False,
        "execution_started": False,
        "real_benchmarks_executed": False,
        "current_authorized_budget_usd": 0,
        "current_authorized_budget_krw": 0,
        "approved_runtime_minutes": 0,
        "network_provider_spend_allowed": False,
        "baseline_materialization_allowed": False,
        "current_materialization_allowed": False,
        "github_policy": "NO_GITHUB_WRITE",
        "future_execution_requires_explicit_human_go": True,
        "required_approval_fields": list(REQUIRED_FIELDS),
        "draft_approval_fields": fields,
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
    return _write_json(tmp_path / "real_benchmark_approval_fields_draft.json", draft)


def _user_update(tmp_path: Path, *, not_execution_approval: bool = True, approve_a_field: bool = False) -> Path:
    future_root = tmp_path / "repo" / "output" / "hse-real-benchmark" / "real-run-20260703_1310"
    update = {
        "schema_version": "hse-real-benchmark-approval-fields-user-update-v1",
        "received_at": "2026-07-03T23:20:00+09:00",
        "not_execution_approval": not_execution_approval,
        "raw_scope_label": "HSE Real Benchmark Approval Fields Draft Update — NOT EXECUTION APPROVAL",
        "unstructured_mentions": [
            "benchmark_suites",
            "max_budget",
            "max_runtime",
            "network/provider spend 허용",
            "baseline/current materialization 허용",
        ],
        "field_updates": {
            "network_provider_api_spend_allowed": {"candidate_for_human_review": True},
            "baseline_materialization_allowed": {"candidate_for_human_review": True},
            "current_materialization_allowed": {"candidate_for_human_review": True},
            "allowed_write_roots": {"candidate_for_human_review": [str(future_root)]},
            "rollback_plan": {
                "candidate_for_human_review": {
                    "delete_future_output_root_if_created": str(future_root),
                    "remove_disposable_worktrees_if_created": True,
                    "preserve_preflight_report_artifacts": True,
                    "cleanup_started": False,
                    "rollback_plan_verified": True,
                    "verify_after_cleanup": [
                        "future output root absent or intentionally archived",
                        "disposable baseline/current worktrees absent",
                        "HSE and Hermes git heads unchanged unless committed locally",
                    ],
                }
            },
            "human_approval_source": {
                "candidate_for_human_review": {
                    "type": "discord_message",
                    "scope": "draft_fields_only_not_execution",
                    "author": "Sunwoo",
                    "channel_context": "SnwEvAH_server / snw-evah / HSE:〔GEPA+DSPy〕",
                    "approval_text": "These fields are provided for draft/review only. This is not approval to execute real benchmarks.",
                }
            },
        },
    }
    if approve_a_field:
        update["field_updates"]["allowed_write_roots"]["approved_for_execution"] = True
    return _write_json(tmp_path / "user_update.json", update)


def test_records_user_supplied_fields_as_candidate_only_not_execution_approval(tmp_path):
    source_draft = _source_draft(tmp_path)
    user_update = _user_update(tmp_path)

    result = write_real_benchmark_approval_fields_update(
        approval_fields_draft_path=source_draft,
        user_update_path=user_update,
        output_dir=tmp_path / "out",
        generated_at="2026-07-03T23:21:00+09:00",
    )

    report = json.loads(Path(result["approval_fields_update_path"]).read_text())
    markdown = Path(result["approval_fields_update_markdown_path"]).read_text()

    assert report["schema_version"] == "hse-real-benchmark-approval-fields-update-v1"
    assert report["gate_id"] == "B0-AFU"
    assert report["status"] == APPROVAL_FIELDS_UPDATE_RECORDED_NOT_EXECUTABLE
    assert report["update_only"] is True
    assert report["not_execution_approval"] is True
    assert report["approval_complete"] is False
    assert report["real_benchmark_execution_approved"] is False
    assert report["execution_ready"] is False
    assert report["strict_plan_gate_closed"] is False
    assert report["execution_started"] is False
    assert report["real_benchmarks_executed"] is False
    assert report["current_authorized_budget_usd"] == 0
    assert report["current_authorized_budget_krw"] == 0
    assert report["approved_runtime_minutes"] == 0
    assert report["network_provider_spend_allowed"] is False
    assert report["baseline_materialization_allowed"] is False
    assert report["current_materialization_allowed"] is False
    assert report["github_policy"] == "NO_GITHUB_WRITE"
    assert report["future_execution_requires_explicit_human_go"] is True
    assert report["source_draft"]["status"] == "APPROVAL_FIELDS_DRAFT_RECORDED_NOT_EXECUTABLE"
    assert report["source_draft"]["execution_ready"] is False

    updates = report["interpreted_field_updates"]
    assert updates["network_provider_api_spend_allowed"]["candidate_for_human_review"] is True
    assert updates["baseline_materialization_allowed"]["candidate_for_human_review"] is True
    assert updates["current_materialization_allowed"]["candidate_for_human_review"] is True
    assert updates["allowed_write_roots"]["candidate_for_human_review"] == [
        str(tmp_path / "repo" / "output" / "hse-real-benchmark" / "real-run-20260703_1310")
    ]
    assert updates["rollback_plan"]["candidate_for_human_review"]["remove_disposable_worktrees_if_created"] is True
    assert updates["human_approval_source"]["candidate_for_human_review"]["scope"] == "draft_fields_only_not_execution"
    assert all(payload["approved_for_execution"] is False for payload in updates.values())
    assert all(payload["requires_separate_execution_approval"] is True for payload in updates.values())
    assert "max_budget_usd_or_krw" in report["incomplete_or_ambiguous_mentions"]
    assert "max_runtime_minutes" in report["incomplete_or_ambiguous_mentions"]
    assert "benchmark_suites" in report["incomplete_or_ambiguous_mentions"]
    assert report["execution_boundaries"]["benchmark_process_started"] is False
    assert report["execution_boundaries"]["worktree_materialization_performed"] is False
    assert report["execution_boundaries"]["benchmark_output_written"] is False
    assert report["execution_boundaries"]["provider_or_model_spend_performed"] is False
    assert Path(result["source_draft_snapshot_path"]).exists()
    assert Path(result["user_update_snapshot_path"]).exists()
    assert "NOT EXECUTION APPROVAL" in markdown
    assert "approved_for_execution=false" in markdown
    assert "strict_plan_gate_closed=false" in markdown


def test_rejects_user_update_without_not_execution_approval_scope(tmp_path):
    source_draft = _source_draft(tmp_path)
    user_update = _user_update(tmp_path, not_execution_approval=False)

    with pytest.raises(ValueError, match="user update must be explicitly marked not_execution_approval"):
        write_real_benchmark_approval_fields_update(
            approval_fields_draft_path=source_draft,
            user_update_path=user_update,
            output_dir=tmp_path / "out",
            generated_at="2026-07-03T23:21:00+09:00",
        )


def test_rejects_execution_ready_source_draft(tmp_path):
    source_draft = _source_draft(tmp_path, execution_ready=True)
    user_update = _user_update(tmp_path)

    with pytest.raises(ValueError, match="source draft must remain non-executable"):
        write_real_benchmark_approval_fields_update(
            approval_fields_draft_path=source_draft,
            user_update_path=user_update,
            output_dir=tmp_path / "out",
            generated_at="2026-07-03T23:21:00+09:00",
        )


def test_rejects_attempt_to_approve_field_for_execution(tmp_path):
    source_draft = _source_draft(tmp_path)
    user_update = _user_update(tmp_path, approve_a_field=True)

    with pytest.raises(ValueError, match="approval field updates must not set approved_for_execution"):
        write_real_benchmark_approval_fields_update(
            approval_fields_draft_path=source_draft,
            user_update_path=user_update,
            output_dir=tmp_path / "out",
            generated_at="2026-07-03T23:21:00+09:00",
        )

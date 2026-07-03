"""Tests for HSE real benchmark execution readiness review."""

from __future__ import annotations

import json
from pathlib import Path

from evolution.local_completion.real_benchmark_execution_readiness import (
    BLOCKED_RUNNER_NOT_AVAILABLE,
    READY_TO_EXECUTE_NOT_STARTED,
    write_real_benchmark_execution_readiness,
)


def _approved_preflight(tmp_path: Path) -> tuple[Path, Path, Path]:
    future_output_root = tmp_path / "repo" / "output" / "hse-real-benchmark" / "run-001"
    preflight_path = tmp_path / "real_benchmark_preflight.json"
    command_preview_path = tmp_path / "real_benchmark_command_preview.json"
    approval_packet_path = tmp_path / "real_benchmark_approval_packet.json"
    approval_packet_path.write_text(
        json.dumps(
            {
                "schema_version": "hse-real-benchmark-approval-packet-v1",
                "status": "APPROVAL_RECORDED_NOT_EXECUTED",
                "approval_complete": True,
                "real_benchmark_execution_approved": True,
                "execution_started": False,
                "real_benchmarks_executed": False,
                "current_authorized_budget_usd": 0,
                "current_authorized_budget_krw": 0,
                "approved_runtime_minutes": 90,
                "network_provider_spend_allowed": False,
                "baseline_materialization_allowed": True,
                "current_materialization_allowed": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    command_preview_path.write_text(
        json.dumps(
            {
                "schema_version": "hse-real-benchmark-command-preview-v1",
                "dry_run": True,
                "benchmark_commands_started": False,
                "commands": [
                    {
                        "suite": "TBLite",
                        "argv": [
                            "python",
                            "-m",
                            "evolution.benchmarks.real_benchmark_runner",
                            "--suite",
                            "TBLite",
                            "--output-json",
                            str(future_output_root / "tblite.json"),
                            "--dry-run",
                        ],
                        "started": False,
                        "network_allowed": False,
                        "provider_spend_allowed": False,
                    },
                    {
                        "suite": "YC-Bench",
                        "argv": [
                            "python",
                            "-m",
                            "evolution.benchmarks.real_benchmark_runner",
                            "--suite",
                            "YC-Bench",
                            "--output-json",
                            str(future_output_root / "yc-bench.json"),
                            "--dry-run",
                        ],
                        "started": False,
                        "network_allowed": False,
                        "provider_spend_allowed": False,
                    },
                    {
                        "suite": "Phase2 PLAN-scale tool-selection triples",
                        "argv": [
                            "python",
                            "-m",
                            "evolution.benchmarks.real_benchmark_runner",
                            "--suite",
                            "Phase2 PLAN-scale tool-selection triples",
                            "--output-json",
                            str(future_output_root / "phase2-plan-scale-tool-selection-triples.json"),
                            "--dry-run",
                        ],
                        "started": False,
                        "network_allowed": False,
                        "provider_spend_allowed": False,
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    preflight_path.write_text(
        json.dumps(
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
                "current_authorized_budget_usd": 0,
                "current_authorized_budget_krw": 0,
                "approved_runtime_minutes": 90,
                "network_provider_spend_allowed": False,
                "benchmark_suites": ["TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples"],
                "blocked_by": [],
                "output_root_guard": {
                    "future_output_root": str(future_output_root),
                    "future_output_root_exists_now": False,
                    "benchmark_output_written_now": False,
                },
                "source_approval_packet": {
                    "path": str(approval_packet_path),
                    "status": "APPROVAL_RECORDED_NOT_EXECUTED",
                    "approval_complete": True,
                    "execution_started": False,
                    "real_benchmarks_executed": False,
                },
                "command_dry_run": {
                    "dry_run": True,
                    "benchmark_commands_started": False,
                    "benchmark_process_started": False,
                    "command_preview_generated": True,
                    "command_preview_path": str(command_preview_path),
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
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return preflight_path, command_preview_path, future_output_root


def test_real_benchmark_execution_readiness_blocks_missing_preview_runner_without_side_effects(tmp_path: Path):
    preflight_path, command_preview_path, future_output_root = _approved_preflight(tmp_path)
    preview = json.loads(command_preview_path.read_text())
    for command in preview["commands"]:
        argv = command["argv"]
        argv[argv.index("evolution.benchmarks.real_benchmark_runner")] = (
            "evolution.benchmarks.missing_real_benchmark_runner"
        )
    command_preview_path.write_text(json.dumps(preview, indent=2, sort_keys=True) + "\n")

    result = write_real_benchmark_execution_readiness(
        preflight_report_path=preflight_path,
        command_preview_path=command_preview_path,
        output_dir=tmp_path / "readiness",
        generated_at="2026-07-03T23:50:00+09:00",
        allowed_write_roots=[str(future_output_root)],
    )

    readiness_path = Path(result["readiness_report_path"])
    markdown_path = Path(result["readiness_markdown_path"])
    readiness = json.loads(readiness_path.read_text())
    assert readiness["schema_version"] == "hse-real-benchmark-execution-readiness-v1"
    assert readiness["status"] == BLOCKED_RUNNER_NOT_AVAILABLE
    assert readiness["execution_go"] is False
    assert readiness["execution_started"] is False
    assert readiness["real_benchmarks_executed"] is False
    assert readiness["strict_plan_gate_closed"] is False
    assert readiness["approval_complete"] is True
    assert readiness["preflight_execution_ready"] is True
    assert readiness["current_authorized_budget_usd"] == 0
    assert readiness["network_provider_spend_allowed"] is False
    assert readiness["runner_checks"]["preview_runner_module"] == "evolution.benchmarks.missing_real_benchmark_runner"
    assert readiness["runner_checks"]["preview_runner_module_available"] is False
    assert "missing_preview_runner_module" in readiness["blocked_by"]
    assert readiness["write_root_checks"]["all_preview_outputs_under_allowed_roots"] is True
    assert readiness["output_root_exists_now"] is False
    assert readiness["execution_boundaries"]["benchmark_process_started"] is False
    assert readiness["execution_boundaries"]["provider_or_model_spend_performed"] is False
    assert readiness["execution_boundaries"]["network_calls_performed"] is False
    assert readiness["execution_boundaries"]["benchmark_output_written"] is False
    assert not future_output_root.exists()
    markdown = markdown_path.read_text()
    assert "BLOCKED_RUNNER_NOT_AVAILABLE" in markdown
    assert "execution_go=false" in markdown


def test_real_benchmark_execution_readiness_allows_existing_local_runner_and_assets_without_starting(tmp_path: Path):
    preflight_path, command_preview_path, future_output_root = _approved_preflight(tmp_path)

    result = write_real_benchmark_execution_readiness(
        preflight_report_path=preflight_path,
        command_preview_path=command_preview_path,
        output_dir=tmp_path / "readiness-ready",
        generated_at="2026-07-04T01:20:00+09:00",
        allowed_write_roots=[str(future_output_root)],
    )

    readiness = json.loads(Path(result["readiness_report_path"]).read_text())
    assert readiness["status"] == READY_TO_EXECUTE_NOT_STARTED
    assert readiness["execution_go"] is True
    assert readiness["execution_started"] is False
    assert readiness["real_benchmarks_executed"] is False
    assert readiness["strict_plan_gate_closed"] is False
    assert readiness["runner_checks"]["preview_runner_module"] == "evolution.benchmarks.real_benchmark_runner"
    assert readiness["runner_checks"]["preview_runner_module_available"] is True
    assert readiness["runner_checks"]["all_suite_readiness_ready"] is True
    assert readiness["blocked_by"] == []
    assert readiness["write_root_checks"]["all_preview_outputs_under_allowed_roots"] is True
    assert readiness["output_root_exists_now"] is False
    suite_readiness = readiness["suite_readiness"]
    assert [entry["suite"] for entry in suite_readiness] == [
        "TBLite",
        "YC-Bench",
        "Phase2 PLAN-scale tool-selection triples",
    ]
    assert all(entry["ready"] is True for entry in suite_readiness)
    assert all(entry["network_calls_required"] is False for entry in suite_readiness)
    assert all(entry["provider_or_model_spend_required"] is False for entry in suite_readiness)
    assert not future_output_root.exists()

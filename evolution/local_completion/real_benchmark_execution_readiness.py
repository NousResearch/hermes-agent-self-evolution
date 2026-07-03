"""Execution-readiness review for HSE real benchmark runs.

This module is intentionally non-executing. It consumes an approved preflight and
command preview, then decides whether the next step may start real benchmark
processes. It never creates benchmark output roots, materializes worktrees,
starts benchmark commands, calls network/provider APIs, writes GitHub, or closes
strict PLAN gates.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags

READINESS_SCHEMA_VERSION = "hse-real-benchmark-execution-readiness-v1"
READINESS_GATE_ID = "B0-RER"
READINESS_PHASE = "Real Benchmark Execution Readiness Review"
READINESS_TARGET = "strict-plan-real-benchmark-execution-readiness"
READY_TO_EXECUTE_NOT_STARTED = "READY_TO_EXECUTE_NOT_STARTED"
BLOCKED_PREFLIGHT_NOT_READY = "BLOCKED_PREFLIGHT_NOT_READY"
BLOCKED_RUNNER_NOT_AVAILABLE = "BLOCKED_RUNNER_NOT_AVAILABLE"
BLOCKED_SUITE_ASSETS_NOT_READY = "BLOCKED_SUITE_ASSETS_NOT_READY"
BLOCKED_WRITE_ROOT_MISMATCH = "BLOCKED_WRITE_ROOT_MISMATCH"


def write_real_benchmark_execution_readiness(
    *,
    preflight_report_path: str | Path,
    command_preview_path: str | Path,
    output_dir: str | Path,
    generated_at: str,
    allowed_write_roots: Sequence[str | Path],
) -> dict[str, str]:
    """Write a non-executing readiness review for real benchmark execution."""

    _require_non_empty("generated_at", generated_at)
    if not allowed_write_roots:
        raise ValueError("allowed_write_roots must not be empty")
    preflight_path = Path(preflight_report_path).expanduser()
    preview_path = Path(command_preview_path).expanduser()
    preflight = _load_json_object(preflight_path, "preflight report")
    preview = _load_json_object(preview_path, "command preview")
    allowed_roots = [Path(root).expanduser() for root in allowed_write_roots]

    commands = _commands(preview)
    preview_modules = sorted({module for module in (_preview_module(command) for command in commands) if module})
    primary_preview_module = preview_modules[0] if len(preview_modules) == 1 else None
    preview_module_available = bool(primary_preview_module and importlib.util.find_spec(primary_preview_module) is not None)
    preview_suites = _preview_suites(commands)
    suite_readiness_reports = _suite_readiness_reports(primary_preview_module, preview_module_available, preview_suites)
    all_suite_readiness_ready = bool(suite_readiness_reports) and all(
        report.get("ready") is True for report in suite_readiness_reports
    )
    output_paths = _preview_output_paths(commands)
    write_root_checks = _write_root_checks(output_paths=output_paths, allowed_roots=allowed_roots)
    preflight_ready = _preflight_ready(preflight)
    blocked_by: list[str] = []
    if not preflight_ready:
        blocked_by.append("preflight_not_execution_ready")
    if not preview_module_available:
        blocked_by.append("missing_preview_runner_module")
    if preview_module_available and not all_suite_readiness_ready:
        blocked_by.append("suite_assets_not_ready")
    if not write_root_checks["all_preview_outputs_under_allowed_roots"]:
        blocked_by.append("preview_output_outside_allowed_write_roots")
    status = _status(blocked_by)
    execution_go = status == READY_TO_EXECUTE_NOT_STARTED
    future_output_root = _future_output_root(preflight)

    report = base_decision_payload(
        gate_id=READINESS_GATE_ID,
        phase=READINESS_PHASE,
        target=READINESS_TARGET,
        generated_at=generated_at,
    )
    report["schema_version"] = READINESS_SCHEMA_VERSION
    report.update(
        {
            "status": status,
            "summary": _summary(status),
            "execution_go": execution_go,
            "approval_complete": preflight.get("approval_complete") is True,
            "preflight_execution_ready": preflight.get("execution_ready") is True,
            "preflight_passed": preflight.get("preflight_passed") is True,
            "real_benchmark_execution_approved": preflight.get("real_benchmark_execution_approved") is True,
            "strict_plan_gate_closed": False,
            "execution_started": False,
            "real_benchmarks_executed": False,
            "current_authorized_budget_usd": preflight.get("current_authorized_budget_usd", 0),
            "current_authorized_budget_krw": preflight.get("current_authorized_budget_krw", 0),
            "approved_runtime_minutes": preflight.get("approved_runtime_minutes"),
            "network_provider_spend_allowed": bool(preflight.get("network_provider_spend_allowed")),
            "github_policy": "NO_GITHUB_WRITE",
            "benchmark_suites": list(preflight.get("benchmark_suites", []))
            if isinstance(preflight.get("benchmark_suites"), list)
            else [],
            "blocked_by": blocked_by,
            "runner_checks": {
                "preview_runner_module": primary_preview_module,
                "preview_modules": preview_modules,
                "preview_runner_module_available": preview_module_available,
                "command_count": len(commands),
                "all_commands_unstarted": all(command.get("started") is False for command in commands),
                "all_commands_dry_run_preview": all("--dry-run" in command.get("argv", []) for command in commands),
                "all_suite_readiness_ready": all_suite_readiness_ready,
            },
            "suite_readiness": suite_readiness_reports,
            "write_root_checks": write_root_checks,
            "output_root": str(future_output_root) if future_output_root is not None else None,
            "output_root_exists_now": future_output_root.exists() if future_output_root is not None else None,
            "source_preflight": {
                "path": str(preflight_path),
                "sha256": _sha256_path(preflight_path),
                "status": preflight.get("status"),
                "execution_ready": preflight.get("execution_ready"),
                "execution_started": preflight.get("execution_started"),
                "real_benchmarks_executed": preflight.get("real_benchmarks_executed"),
            },
            "source_command_preview": {
                "path": str(preview_path),
                "sha256": _sha256_path(preview_path),
                "schema_version": preview.get("schema_version"),
                "benchmark_commands_started": preview.get("benchmark_commands_started"),
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
            "artifacts": {
                "readiness_report": "real_benchmark_execution_readiness.json",
                "readiness_markdown": "real_benchmark_execution_readiness.md",
            },
            "required_next_action": "implement_or_select_real_benchmark_runner_before_execution"
            if not execution_go
            else "start_real_benchmark_processes_under_approved_limits",
        }
    )
    reject_github_or_active_apply_flags(report)

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "real_benchmark_execution_readiness.json"
    markdown_path = out / "real_benchmark_execution_readiness.md"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown_path.write_text(_render_markdown(report))
    return {
        "readiness_report_path": str(report_path),
        "readiness_markdown_path": str(markdown_path),
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    if not isinstance(data, dict):
        raise ValueError(f"{label} JSON root must be an object: {path}")
    return data


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _commands(preview: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    commands = preview.get("commands")
    if not isinstance(commands, list):
        raise ValueError("command preview must contain commands list")
    result: list[Mapping[str, Any]] = []
    for command in commands:
        if not isinstance(command, Mapping):
            raise ValueError("command preview commands must be objects")
        result.append(command)
    return result


def _preview_module(command: Mapping[str, Any]) -> str | None:
    argv = command.get("argv")
    if not isinstance(argv, list):
        return None
    try:
        index = argv.index("-m")
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    module = argv[index + 1]
    return str(module) if isinstance(module, str) and module else None


def _preview_suites(commands: Sequence[Mapping[str, Any]]) -> list[str]:
    suites: list[str] = []
    for command in commands:
        suite = command.get("suite")
        if isinstance(suite, str) and suite:
            suites.append(suite)
            continue
        argv = command.get("argv")
        if isinstance(argv, list) and "--suite" in argv:
            index = argv.index("--suite")
            if index + 1 < len(argv) and isinstance(argv[index + 1], str):
                suites.append(argv[index + 1])
    return suites


def _suite_readiness_reports(
    primary_preview_module: str | None,
    preview_module_available: bool,
    suites: Sequence[str],
) -> list[dict[str, Any]]:
    if not preview_module_available or not primary_preview_module:
        return []
    module = importlib.import_module(primary_preview_module)
    suite_readiness = getattr(module, "suite_readiness", None)
    if not callable(suite_readiness):
        return [
            {
                "suite": suite,
                "ready": False,
                "blocked_by": ["runner_module_has_no_suite_readiness"],
                "network_calls_required": False,
                "provider_or_model_spend_required": False,
            }
            for suite in suites
        ]
    reports: list[dict[str, Any]] = []
    for suite in suites:
        raw = suite_readiness(suite)
        if not isinstance(raw, Mapping):
            raise ValueError(f"suite_readiness must return object for {suite}")
        reports.append(dict(raw))
    return reports


def _preview_output_paths(commands: Sequence[Mapping[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for command in commands:
        argv = command.get("argv")
        if not isinstance(argv, list):
            continue
        for index, value in enumerate(argv):
            if value == "--output-json" and index + 1 < len(argv) and isinstance(argv[index + 1], str):
                paths.append(Path(argv[index + 1]).expanduser())
    return paths


def _write_root_checks(*, output_paths: Sequence[Path], allowed_roots: Sequence[Path]) -> dict[str, Any]:
    resolved_roots = [root.resolve(strict=False) for root in allowed_roots]
    outputs = [path.resolve(strict=False) for path in output_paths]
    all_under = bool(outputs) and all(any(_is_relative_to(output, root) for root in resolved_roots) for output in outputs)
    return {
        "allowed_write_roots": [str(root) for root in allowed_roots],
        "preview_output_paths": [str(path) for path in output_paths],
        "all_preview_outputs_under_allowed_roots": all_under,
        "preview_output_count": len(outputs),
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _preflight_ready(preflight: Mapping[str, Any]) -> bool:
    return bool(
        preflight.get("schema_version") == "hse-real-benchmark-preflight-v1"
        and preflight.get("preflight_passed") is True
        and preflight.get("execution_ready") is True
        and preflight.get("approval_complete") is True
        and preflight.get("real_benchmark_execution_approved") is True
        and preflight.get("execution_started") is False
        and preflight.get("real_benchmarks_executed") is False
    )


def _future_output_root(preflight: Mapping[str, Any]) -> Path | None:
    guard = preflight.get("output_root_guard")
    if not isinstance(guard, Mapping):
        return None
    root = guard.get("future_output_root")
    if not isinstance(root, str) or not root:
        return None
    return Path(root).expanduser()


def _status(blocked_by: Sequence[str]) -> str:
    if not blocked_by:
        return READY_TO_EXECUTE_NOT_STARTED
    if "preflight_not_execution_ready" in blocked_by:
        return BLOCKED_PREFLIGHT_NOT_READY
    if "missing_preview_runner_module" in blocked_by:
        return BLOCKED_RUNNER_NOT_AVAILABLE
    if "suite_assets_not_ready" in blocked_by:
        return BLOCKED_SUITE_ASSETS_NOT_READY
    return BLOCKED_WRITE_ROOT_MISMATCH


def _summary(status: str) -> str:
    if status == READY_TO_EXECUTE_NOT_STARTED:
        return "Execution readiness review passed; real benchmark processes have not started."
    if status == BLOCKED_RUNNER_NOT_AVAILABLE:
        return "Execution readiness is blocked because the command preview runner module is not available."
    if status == BLOCKED_SUITE_ASSETS_NOT_READY:
        return "Execution readiness is blocked because one or more suite-local asset checks are not ready."
    if status == BLOCKED_WRITE_ROOT_MISMATCH:
        return "Execution readiness is blocked because preview outputs are outside allowed write roots."
    return "Execution readiness is blocked because preflight is not execution-ready."


def _render_markdown(report: Mapping[str, Any]) -> str:
    blockers = report.get("blocked_by", [])
    blocker_lines = [f"- {blocker}" for blocker in blockers] if isinstance(blockers, list) else []
    return "\n".join(
        [
            "# HSE Real Benchmark Execution Readiness",
            "",
            f"Status: `{report.get('status')}`",
            "",
            f"- execution_go={str(report.get('execution_go')).lower()}",
            f"- approval_complete={str(report.get('approval_complete')).lower()}",
            f"- preflight_execution_ready={str(report.get('preflight_execution_ready')).lower()}",
            f"- execution_started={str(report.get('execution_started')).lower()}",
            f"- real_benchmarks_executed={str(report.get('real_benchmarks_executed')).lower()}",
            f"- strict_plan_gate_closed={str(report.get('strict_plan_gate_closed')).lower()}",
            "",
            "## Blockers",
            "",
            *(blocker_lines or ["- none"]),
            "",
            "## Boundaries",
            "",
            "- benchmark process started: false",
            "- provider/model/API spend performed: false",
            "- network calls performed: false",
            "- benchmark output written: false",
            "- NO_GITHUB_WRITE",
            "",
        ]
    )


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_non_empty(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")

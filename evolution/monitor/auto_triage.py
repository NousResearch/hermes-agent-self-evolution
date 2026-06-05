"""Read-only Phase 5 auto-triage ranking reports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE5_OUTPUT_ROOT = REPO_ROOT / "output" / "phase5-continuous-loop"
REPORT_JSON_NAME = "auto_triage_report.json"
REPORT_MARKDOWN_NAME = "auto_triage_report.md"

_SCHEMA_VERSION = "phase5-auto-triage-ranking-v1"
_PERFORMANCE_SCHEMA_VERSION = "phase5-performance-snapshot-v1"
_PERFORMANCE_MODE = "phase5-readonly-performance-monitor-snapshot"
_MODE = "phase5-readonly-auto-triage-ranking"
_REVIEW_RECOMMENDATION = "manual_review_required_no_optimizer_started"
_NO_ACTION_RECOMMENDATION = "no_action_monitor_only"


def build_auto_triage_report(
    performance_report: Mapping[str, Any],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic read-only auto-triage ranking report.

    The function ranks weak metrics from a sanitized performance snapshot by
    impact times frequency. It never creates scheduler jobs, starts optimizers,
    edits runtime state, or updates external pull requests.
    """

    _reject_private_or_raw_identifiers(performance_report)
    _validate_performance_report(performance_report)

    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    metrics = list(performance_report["metrics"])
    ranked_targets = [_rankable_target(metric) for metric in metrics if _is_rankable(metric)]
    ranked_targets.sort(key=lambda item: (-item["priority_score"], -item["sample_count"], item["metric_id"]))
    for rank, target in enumerate(ranked_targets, start=1):
        target["rank"] = rank

    status = "REVIEW_REQUIRED" if ranked_targets else "NO_ACTION"
    component_count = len({target["component"] for target in ranked_targets})
    recommended_next_step = _REVIEW_RECOMMENDATION if ranked_targets else _NO_ACTION_RECOMMENDATION

    report: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "phase": "5",
        "mode": _MODE,
        "status": status,
        "generated_at": generated_at,
        "source": {
            "performance_snapshot_schema_version": performance_report["schema_version"],
            "performance_snapshot_mode": performance_report["mode"],
            "performance_snapshot_status": performance_report["status"],
            "performance_snapshot_window": dict(performance_report["window"]),
        },
        "input_contract": {
            "performance_snapshot_report_required": True,
            "sanitized_input_required": True,
            "raw_session_data_allowed": False,
            "private_paths_allowed": False,
            "network_sources_allowed": False,
            "credentials_allowed": False,
        },
        "safety_invariants": {
            "read_only": True,
            "raw_private_session_data_committed": False,
            "raw_credentials_recorded": False,
            "active_runtime_mutation": False,
            "external_calls_performed": False,
            "network_calls_performed": False,
            "cron_jobs_created": False,
            "scheduler_or_cron_side_effects_performed": False,
            "auto_optimizer_triggered": False,
            "optimizer_execution_started": False,
            "automated_pr_created_or_updated": False,
            "automated_apply_ready": False,
        },
        "scoring": {
            "formula": "severity * sample_count",
            "tie_breakers": ["priority_score desc", "sample_count desc", "metric_id asc"],
            "optimizer_trigger_policy": "never_in_this_slice",
        },
        "summary": {
            "candidate_metric_count": len(metrics),
            "ranked_target_count": len(ranked_targets),
            "component_count": component_count,
            "top_metric_id": ranked_targets[0]["metric_id"] if ranked_targets else None,
            "max_priority_score": ranked_targets[0]["priority_score"] if ranked_targets else 0.0,
            "review_required": bool(ranked_targets),
        },
        "ranked_targets": ranked_targets,
        "recommended_next_step": recommended_next_step,
    }
    _reject_private_or_raw_identifiers(report)
    return report


def write_auto_triage_report(
    performance_report: Mapping[str, Any],
    *,
    output_dir: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Write JSON and Markdown auto-triage artifacts under Phase 5 output root."""

    output_dir = _validate_output_dir(output_dir)
    report = build_auto_triage_report(performance_report, generated_at=generated_at)
    report["artifacts"] = {
        "report_json": REPORT_JSON_NAME,
        "report_markdown": REPORT_MARKDOWN_NAME,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_NAME
    markdown_path = output_dir / REPORT_MARKDOWN_NAME
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_markdown(report))
    return report


def _validate_performance_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != _PERFORMANCE_SCHEMA_VERSION:
        raise ValueError("performance report schema_version must be phase5-performance-snapshot-v1")
    if report.get("phase") != "5" or report.get("mode") != _PERFORMANCE_MODE:
        raise ValueError("performance report must be a Phase 5 performance snapshot")
    window = report.get("window")
    if not isinstance(window, Mapping) or not isinstance(window.get("start"), str) or not isinstance(window.get("end"), str):
        raise ValueError("performance report window must contain start and end strings")
    _validate_performance_input_contract(report.get("input_contract"))
    _validate_performance_safety(report.get("safety_invariants"))
    metrics = report.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("performance report must contain a non-empty metrics list")
    for metric in metrics:
        _validate_metric(metric)


def _validate_performance_input_contract(input_contract: object) -> None:
    if not isinstance(input_contract, Mapping):
        raise ValueError("performance report input_contract must be sanitized")
    required_false = [
        "raw_session_data_allowed",
        "private_paths_allowed",
        "network_sources_allowed",
        "credentials_allowed",
    ]
    if input_contract.get("sanitized_input_required") is not True:
        raise ValueError("performance report input_contract must be sanitized")
    if any(input_contract.get(key) is not False for key in required_false):
        raise ValueError("performance report input_contract must be sanitized")


def _validate_performance_safety(safety: object) -> None:
    if not isinstance(safety, Mapping):
        raise ValueError("performance report must contain safety_invariants")
    required_false = [
        "raw_private_session_data_committed",
        "raw_credentials_recorded",
        "active_runtime_mutation",
        "external_calls_performed",
        "network_calls_performed",
        "cron_jobs_created",
        "optimizer_execution_started",
        "automated_pr_created_or_updated",
    ]
    if safety.get("read_only") is not True:
        raise ValueError("performance report must be read-only before auto-triage")
    if any(safety.get(key) is not False for key in required_false):
        raise ValueError("performance report must be read-only before auto-triage")


def _validate_metric(metric: object) -> None:
    if not isinstance(metric, Mapping):
        raise ValueError("each performance metric must be an object")
    for key in ("id", "component", "status"):
        if not isinstance(metric.get(key), str) or not metric[key].strip():
            raise ValueError(f"metric.{key} must be a non-empty string")
    if metric["status"] not in {"PASS", "FAIL"}:
        raise ValueError("metric.status must be PASS or FAIL")
    for key in ("value", "threshold", "baseline", "severity"):
        value = metric.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"metric.{key} must be a finite number")
    if metric["severity"] < 0:
        raise ValueError("metric.severity must be non-negative")
    if not isinstance(metric.get("higher_is_better"), bool):
        raise ValueError("metric.higher_is_better must be a boolean")
    if not isinstance(metric.get("regressed_vs_baseline"), bool):
        raise ValueError("metric.regressed_vs_baseline must be a boolean")
    sample_count = metric.get("sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
        raise ValueError("metric.sample_count must be a positive integer")


def _is_rankable(metric: Mapping[str, Any]) -> bool:
    return metric["status"] == "FAIL" or metric["regressed_vs_baseline"] is True


def _rankable_target(metric: Mapping[str, Any]) -> dict[str, Any]:
    severity = float(metric["severity"])
    sample_count = int(metric["sample_count"])
    reasons: list[str] = []
    if metric["status"] == "FAIL":
        reasons.append("failing_threshold")
    if metric["regressed_vs_baseline"] is True:
        reasons.append("regressed_vs_baseline")
    return {
        "rank": 0,
        "metric_id": metric["id"].strip(),
        "component": metric["component"].strip(),
        "status": metric["status"],
        "severity": severity,
        "sample_count": sample_count,
        "regressed_vs_baseline": metric["regressed_vs_baseline"],
        "priority_score": round(severity * sample_count, 10),
        "priority_inputs": {
            "severity": severity,
            "sample_count": sample_count,
        },
        "reasons": reasons,
        "recommendation": _REVIEW_RECOMMENDATION,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 5 Auto-Triage Ranking",
        "",
        f"Status: {report['status']}",
        "",
        f"Mode: `{report['mode']}`",
        "",
        "## Safety",
        "",
        "- read_only=true",
        "- scheduler_or_cron_side_effects_performed=false",
        "- auto_optimizer_triggered=false",
        "- optimizer_execution_started=false",
        "- automated_pr_created_or_updated=false",
        "",
        "## Scoring",
        "",
        f"- formula={report['scoring']['formula']}",
        "- optimizer_trigger_policy=never_in_this_slice",
        "",
        "## Ranked Targets",
        "",
    ]
    if report["ranked_targets"]:
        for target in report["ranked_targets"]:
            lines.append(
                f"{target['rank']}. `{target['metric_id']}` ({target['component']}): "
                f"priority_score={target['priority_score']}, "
                f"sample_count={target['sample_count']}, severity={target['severity']}"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Recommended Next Step",
            "",
            f"`{report['recommended_next_step']}`",
            "",
            "This auto-triage ranking is evidence for manual review only; it is not approval to create cron jobs, run optimizers, spend benchmark/API budget, or update external pull requests automatically.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    _reject_private_or_raw_identifiers(markdown)
    return markdown


def _validate_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve(strict=False)
    root = PHASE5_OUTPUT_ROOT.resolve(strict=False)
    if output_dir == root or root not in output_dir.parents:
        raise ValueError("output-dir must be under output/phase5-continuous-loop")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output-dir must be a directory before writing auto-triage artifacts")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output-dir must be empty before writing auto-triage artifacts")
    return output_dir


def _reject_private_or_raw_identifiers(value: object) -> None:
    forbidden = [
        "/" + "Users" + "/",
        "/" + "home" + "/",
        "session" + "_id",
        "OPENAI" + "_API_KEY",
        "ANTHROPIC" + "_API_KEY",
        "OPENROUTER" + "_API_KEY",
    ]
    for text in _all_strings(value):
        for fragment in forbidden:
            if fragment in text:
                raise ValueError("auto-triage input contains private/raw identifier")


def _all_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _all_strings(key)
            yield from _all_strings(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _all_strings(child)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a read-only Phase 5 auto-triage ranking report.")
    parser.add_argument("--performance-report-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args(argv)

    try:
        performance_report = json.loads(args.performance_report_json.read_text())
        write_auto_triage_report(
            performance_report,
            output_dir=args.output_dir,
            generated_at=args.generated_at,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("wrote Phase 5 auto-triage ranking")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

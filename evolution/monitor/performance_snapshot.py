"""Read-only Phase 5 performance monitor snapshot reports."""

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
REPORT_JSON_NAME = "performance_snapshot_report.json"
REPORT_MARKDOWN_NAME = "performance_snapshot_report.md"

_SCHEMA_VERSION = "phase5-performance-snapshot-v1"
_INPUT_SCHEMA_VERSION = "phase5-performance-input-v1"
_MODE = "phase5-readonly-performance-monitor-snapshot"
_RECOMMENDATION = "manual_triage_required_no_optimizer_started"


def build_performance_snapshot_report(
    metrics_payload: Mapping[str, Any],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic read-only performance snapshot report.

    The function consumes already-sanitized aggregate metrics only. It does not
    read SessionDB, call the network, create cron jobs, start optimizers, or
    mutate Hermes runtime state.
    """

    _reject_private_or_raw_identifiers(metrics_payload)
    _validate_input_payload(metrics_payload)

    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    metrics = [_normalize_metric(metric) for metric in metrics_payload["metrics"]]
    weak_areas = [_weak_area(metric) for metric in metrics if metric["status"] == "FAIL" or metric["regressed_vs_baseline"]]
    weak_areas.sort(key=lambda item: (-item["severity"], item["metric_id"]))

    failing_count = sum(1 for metric in metrics if metric["status"] == "FAIL")
    regressing_count = sum(1 for metric in metrics if metric["regressed_vs_baseline"])
    components = {metric["component"] for metric in metrics}
    status = "PASS" if failing_count == 0 and regressing_count == 0 else "NEEDS_TRIAGE"

    report: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "phase": "5",
        "mode": _MODE,
        "status": status,
        "generated_at": generated_at,
        "window": dict(metrics_payload["window"]),
        "source": dict(metrics_payload["source"]),
        "input_contract": {
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
            "optimizer_execution_started": False,
            "automated_pr_created_or_updated": False,
        },
        "summary": {
            "metric_count": len(metrics),
            "component_count": len(components),
            "failing_metric_count": failing_count,
            "regressing_metric_count": regressing_count,
            "weak_area_count": len(weak_areas),
        },
        "metrics": metrics,
        "weak_areas": weak_areas,
        "recommended_next_step": _RECOMMENDATION,
    }
    _reject_private_or_raw_identifiers(report)
    return report


def write_performance_snapshot_report(
    metrics_payload: Mapping[str, Any],
    *,
    output_dir: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Write JSON and Markdown performance snapshot artifacts under Phase 5 output root."""

    output_dir = _validate_output_dir(output_dir)
    report = build_performance_snapshot_report(metrics_payload, generated_at=generated_at)
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


def _validate_input_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != _INPUT_SCHEMA_VERSION:
        raise ValueError("metrics payload schema_version must be phase5-performance-input-v1")
    window = payload.get("window")
    if not isinstance(window, Mapping) or not isinstance(window.get("start"), str) or not isinstance(window.get("end"), str):
        raise ValueError("metrics payload window must contain start and end strings")
    source = payload.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("kind"), str) or not isinstance(source.get("label"), str):
        raise ValueError("metrics payload source must contain kind and label strings")
    allowed_source_kinds = {"sanitized_local_fixture", "provenance_backed_sanitized_dataset"}
    if source.get("kind") not in allowed_source_kinds:
        raise ValueError("metrics payload source.kind must be sanitized_local_fixture or provenance_backed_sanitized_dataset")
    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("metrics payload must contain a non-empty metrics list")
    for metric in metrics:
        _validate_metric(metric)


def _validate_metric(metric: object) -> None:
    if not isinstance(metric, Mapping):
        raise ValueError("each metric must be an object")
    for key in ("id", "component"):
        if not isinstance(metric.get(key), str) or not metric[key].strip():
            raise ValueError(f"metric.{key} must be a non-empty string")
    for key in ("value", "threshold", "baseline"):
        value = metric.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"metric.{key} must be a finite number")
    if not isinstance(metric.get("higher_is_better"), bool):
        raise ValueError("metric.higher_is_better must be a boolean")
    sample_count = metric.get("sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
        raise ValueError("metric.sample_count must be a positive integer")


def _normalize_metric(metric: Mapping[str, Any]) -> dict[str, Any]:
    value = float(metric["value"])
    threshold = float(metric["threshold"])
    baseline = float(metric["baseline"])
    higher_is_better = metric["higher_is_better"]

    passed = value >= threshold if higher_is_better else value <= threshold
    regressed = value < baseline if higher_is_better else value > baseline
    severity = max(threshold - value, 0.0) if higher_is_better else max(value - threshold, 0.0)

    return {
        "id": metric["id"].strip(),
        "component": metric["component"].strip(),
        "value": value,
        "threshold": threshold,
        "baseline": baseline,
        "higher_is_better": higher_is_better,
        "sample_count": metric["sample_count"],
        "status": "PASS" if passed else "FAIL",
        "regressed_vs_baseline": regressed,
        "severity": round(severity, 10),
    }


def _weak_area(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metric_id": metric["id"],
        "component": metric["component"],
        "status": metric["status"],
        "regressed_vs_baseline": metric["regressed_vs_baseline"],
        "severity": metric["severity"],
        "recommendation": _RECOMMENDATION,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 5 Performance Monitor Snapshot",
        "",
        f"Status: {report['status']}",
        "",
        f"Mode: `{report['mode']}`",
        "",
        "## Safety",
        "",
        "- read_only=true",
        "- cron_jobs_created=false",
        "- optimizer_execution_started=false",
        "- automated_pr_created_or_updated=false",
        "",
        "## Summary",
        "",
        f"- metric_count={report['summary']['metric_count']}",
        f"- failing_metric_count={report['summary']['failing_metric_count']}",
        f"- regressing_metric_count={report['summary']['regressing_metric_count']}",
        f"- weak_area_count={report['summary']['weak_area_count']}",
        "",
        "## Weak Areas",
        "",
    ]
    if report["weak_areas"]:
        for area in report["weak_areas"]:
            lines.append(
                f"- `{area['metric_id']}` ({area['component']}): "
                f"status={area['status']}, regressed={str(area['regressed_vs_baseline']).lower()}, "
                f"severity={area['severity']}"
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
            "This snapshot is evidence for manual triage only; it is not approval to create cron jobs, run optimizers, spend benchmark/API budget, or update external pull requests automatically.",
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
        raise ValueError(f"output-dir must be a directory before writing monitor artifacts: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output-dir must be empty before writing monitor artifacts: {output_dir}")
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
                raise ValueError("performance monitor input contains private/raw identifier")


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
    parser = argparse.ArgumentParser(description="Write a read-only Phase 5 performance monitor snapshot report.")
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args(argv)

    try:
        metrics_payload = json.loads(args.metrics_json.read_text())
        write_performance_snapshot_report(
            metrics_payload,
            output_dir=args.output_dir,
            generated_at=args.generated_at,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("wrote Phase 5 performance monitor snapshot")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

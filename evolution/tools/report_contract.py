"""Lightweight schema smoke check for Phase 2 candidate-only reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import click

REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "phase",
        "mode",
        "apply_ready",
        "summary",
        "candidate_count",
        "metrics",
        "candidates",
        "phase_index_executed",
        "phase2d_gate",
        "inventory_metadata",
        "artifacts",
    }
)
REQUIRED_METRIC_FIELDS = frozenset(
    {
        "candidate_only",
        "apply_ready",
        "case_count",
        "selection_accuracy",
        "wrong_tool_avoidance",
        "argument_cue_coverage",
        "constraint_pass_rate",
        "case_results",
        "warnings",
    }
)
REQUIRED_GATE_FIELDS = frozenset(
    {
        "phase",
        "candidate_only",
        "passed",
        "thresholds",
        "baseline_metrics",
        "candidate_metrics",
        "per_tool_regressions",
        "failed_checks",
    }
)
REQUIRED_INVENTORY_METADATA_FIELDS = frozenset(
    {
        "source",
        "tool_count",
        "import_warning_count",
        "import_warnings",
        "candidate_quality_warnings_are_separate",
    }
)
REQUIRED_IMPORT_WARNING_FIELDS = frozenset(
    {
        "module",
        "message",
        "exception",
        "classification",
        "candidate_quality",
    }
)
APPLY_PAYLOAD_KEYS = frozenset({"patch", "patches", "write_paths", "apply_payload", "source_updates"})
DEFAULT_PHASE2D_THRESHOLDS = {
    "min_case_count": 45,
    "min_selection_accuracy": 0.7,
    "min_wrong_tool_avoidance": 0.7,
    "max_per_tool_regression": 0.0,
}


@dataclass(frozen=True)
class ReportContractValidation:
    """Result of a Phase 2 report contract smoke check."""

    passed: bool
    errors: tuple[str, ...]


def validate_candidate_only_report_contract(report: Mapping[str, object]) -> ReportContractValidation:
    """Validate the Phase 2 candidate-only report contract.

    This is intentionally a lightweight stdlib check rather than a JSON Schema
    dependency. It validates the contract documented in README.md/PLAN.md and
    keeps candidate-quality metrics separate from inventory metadata.
    """

    errors: list[str] = []
    _require_fields(report, REQUIRED_TOP_LEVEL_FIELDS, "report", errors)

    if report.get("phase") != "2D":
        errors.append('top-level phase must be "2D"')
    if report.get("mode") != "candidate-only":
        errors.append('top-level mode must be "candidate-only"')
    if report.get("apply_ready") is not False:
        errors.append("top-level apply_ready must be false")
    for key in sorted(APPLY_PAYLOAD_KEYS & set(report)):
        errors.append(f"candidate-only report must not contain apply payload key: {key}")
    if report.get("phase_index_executed") != ["2A", "2B", "2C", "2D"]:
        errors.append('phase_index_executed must be ["2A", "2B", "2C", "2D"]')

    metrics = _mapping(report.get("metrics"), "metrics", errors)
    gate = _mapping(report.get("phase2d_gate"), "phase2d_gate", errors)
    inventory_metadata = _mapping(report.get("inventory_metadata"), "inventory_metadata", errors)
    artifacts = _mapping(report.get("artifacts"), "artifacts", errors)

    if metrics is not None:
        _validate_metrics(metrics, errors)
    if gate is not None:
        _validate_gate(gate, metrics, errors)
    if inventory_metadata is not None:
        _validate_inventory_metadata(inventory_metadata, errors)
    if artifacts is not None:
        _validate_artifacts(artifacts, errors)

    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        errors.append("candidates must be a list")
    candidate_count = report.get("candidate_count")
    if not isinstance(candidate_count, int) or candidate_count < 0:
        errors.append("candidate_count must be a non-negative integer")
    elif isinstance(candidates, list) and candidate_count != len(candidates):
        errors.append("candidate_count must equal len(candidates)")

    return ReportContractValidation(passed=not errors, errors=tuple(errors))


def load_and_validate_candidate_only_report(path: str | Path) -> ReportContractValidation:
    """Load a report JSON file and validate the candidate-only contract."""

    report_path = Path(path)
    raw = json.loads(report_path.read_text())
    if not isinstance(raw, Mapping):
        return ReportContractValidation(passed=False, errors=("report JSON root must be an object",))
    return validate_candidate_only_report_contract(raw)


def _validate_metrics(metrics: Mapping[str, object], errors: list[str]) -> None:
    _require_fields(metrics, REQUIRED_METRIC_FIELDS, "metrics", errors)
    if metrics.get("candidate_only") is not True:
        errors.append("metrics.candidate_only must be true")
    if metrics.get("apply_ready") is not False:
        errors.append("metrics.apply_ready must be false")
    case_count = metrics.get("case_count")
    if not isinstance(case_count, int) or case_count < 0:
        errors.append("metrics.case_count must be a non-negative integer")
    if not isinstance(metrics.get("case_results"), list):
        errors.append("metrics.case_results must be a list")
    if not isinstance(metrics.get("warnings"), list):
        errors.append("metrics.warnings must be a list")


def _validate_gate(gate: Mapping[str, object], metrics: Mapping[str, object] | None, errors: list[str]) -> None:
    _require_fields(gate, REQUIRED_GATE_FIELDS, "phase2d_gate", errors)
    if gate.get("phase") != "2D":
        errors.append('phase2d_gate.phase must be "2D"')
    if gate.get("candidate_only") is not True:
        errors.append("phase2d_gate.candidate_only must be true")
    if not isinstance(gate.get("passed"), bool):
        errors.append("phase2d_gate.passed must be a boolean")
    thresholds = _mapping(gate.get("thresholds"), "phase2d_gate.thresholds", errors)
    if thresholds is not None and thresholds != DEFAULT_PHASE2D_THRESHOLDS:
        errors.append("phase2d_gate.thresholds must match the Phase 2D default contract")
    baseline_metrics = _mapping(gate.get("baseline_metrics"), "phase2d_gate.baseline_metrics", errors)
    candidate_metrics = _mapping(gate.get("candidate_metrics"), "phase2d_gate.candidate_metrics", errors)
    if baseline_metrics is not None:
        _validate_metric_snapshot(baseline_metrics, "phase2d_gate.baseline_metrics", errors)
    if candidate_metrics is not None:
        _validate_metric_snapshot(candidate_metrics, "phase2d_gate.candidate_metrics", errors)
        warnings = metrics.get("warnings") if metrics is not None else None
        if isinstance(warnings, list) and candidate_metrics.get("warning_count") != len(warnings):
            errors.append("phase2d_gate.candidate_metrics.warning_count must equal len(metrics.warnings)")
    if not isinstance(gate.get("per_tool_regressions"), list):
        errors.append("phase2d_gate.per_tool_regressions must be a list")
    failed_checks = gate.get("failed_checks")
    if not isinstance(failed_checks, list):
        errors.append("phase2d_gate.failed_checks must be a list")
    elif gate.get("passed") is True and failed_checks:
        errors.append("phase2d_gate.failed_checks must be empty when passed is true")
    elif gate.get("passed") is False and not failed_checks:
        errors.append("phase2d_gate.failed_checks must be non-empty when passed is false")


def _validate_metric_snapshot(snapshot: Mapping[str, object], path: str, errors: list[str]) -> None:
    required = {
        "case_count",
        "selection_accuracy",
        "wrong_tool_avoidance",
        "argument_cue_coverage",
        "constraint_pass_rate",
        "warning_count",
    }
    _require_fields(snapshot, required, path, errors)
    warning_count = snapshot.get("warning_count")
    if not isinstance(warning_count, int) or warning_count < 0:
        errors.append(f"{path}.warning_count must be a non-negative integer")


def _validate_inventory_metadata(metadata: Mapping[str, object], errors: list[str]) -> None:
    _require_fields(metadata, REQUIRED_INVENTORY_METADATA_FIELDS, "inventory_metadata", errors)
    if metadata.get("source") not in {"inventory_json", "hermes_repo_import"}:
        errors.append('inventory_metadata.source must be "inventory_json" or "hermes_repo_import"')
    tool_count = metadata.get("tool_count")
    if not isinstance(tool_count, int) or tool_count < 0:
        errors.append("inventory_metadata.tool_count must be a non-negative integer")
    if metadata.get("candidate_quality_warnings_are_separate") is not True:
        errors.append("inventory_metadata.candidate_quality_warnings_are_separate must be true")
    import_warnings = metadata.get("import_warnings")
    if not isinstance(import_warnings, list):
        errors.append("inventory_metadata.import_warnings must be a list")
        return
    if metadata.get("import_warning_count") != len(import_warnings):
        errors.append("inventory_metadata.import_warning_count must equal len(import_warnings)")
    for index, warning in enumerate(import_warnings):
        warning_mapping = _mapping(warning, f"inventory_metadata.import_warnings[{index}]", errors)
        if warning_mapping is None:
            continue
        _require_fields(
            warning_mapping,
            REQUIRED_IMPORT_WARNING_FIELDS,
            f"inventory_metadata.import_warnings[{index}]",
            errors,
        )
        if warning_mapping.get("candidate_quality") is not False:
            errors.append(f"inventory_metadata.import_warnings[{index}].candidate_quality must be false")


def _validate_artifacts(artifacts: Mapping[str, object], errors: list[str]) -> None:
    for key in ("inventory", "candidates", "diff"):
        if not isinstance(artifacts.get(key), str) or not artifacts.get(key):
            errors.append(f"artifacts.{key} must be a non-empty string")


def _mapping(value: object, path: str, errors: list[str]) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return None
    return value


def _require_fields(value: Mapping[str, object], required: set[str] | frozenset[str], path: str, errors: list[str]) -> None:
    missing = sorted(required - set(value))
    for field in missing:
        errors.append(f"{path} missing required field: {field}")


@click.command()
@click.argument("report_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(report_path: Path) -> None:
    """Validate a Phase 2 candidate_only_report.json contract."""

    validation = load_and_validate_candidate_only_report(report_path)
    if validation.passed:
        click.echo(f"candidate_only_report contract passed: {report_path}")
        return

    click.echo(f"candidate_only_report contract failed: {report_path}")
    for error in validation.errors:
        click.echo(f"- {error}")
    raise click.ClickException("candidate_only_report contract failed")


if __name__ == "__main__":
    main()

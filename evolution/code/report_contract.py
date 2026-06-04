"""Lightweight schema smoke check for Phase 4 code-evolution scaffold reports."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "phase",
        "mode",
        "scaffold_version",
        "dry_run",
        "candidate_only",
        "read_only_inputs",
        "darwinian_cli_invoked",
        "darwinian_imported",
        "external_calls_performed",
        "package_installed",
        "hermes_source_mutation_performed",
        "active_runtime_apply_approved",
        "apply_ready",
        "passed",
        "failed_checks",
        "target_spec",
        "allowed_mutation",
        "freeze_checks",
        "fitness_plan",
        "approval_gates",
        "artifacts",
        "write_targets",
        "output_constraints",
    }
)
FALSE_SAFETY_FLAGS = frozenset(
    {
        "darwinian_cli_invoked",
        "darwinian_imported",
        "external_calls_performed",
        "package_installed",
        "hermes_source_mutation_performed",
        "active_runtime_apply_approved",
        "apply_ready",
    }
)
TRUE_SCAFFOLD_FLAGS = frozenset({"dry_run", "candidate_only", "read_only_inputs"})
APPLY_PAYLOAD_KEYS = frozenset({"patch", "patches", "write_paths", "apply_payload", "source_updates"})
SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"\b[A-Za-z0-9_]*(?:api[_-]?key|access[_-]?token|secret|password)[A-Za-z0-9_]*\s*[:=]\s*['\"]?[^'\"\s]{8,}",
        re.IGNORECASE,
    ),
)
ALLOWED_OUTPUT_ROOT = "output/phase4-code-evolution/"
REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE4_OUTPUT_ROOT = REPO_ROOT / ALLOWED_OUTPUT_ROOT
REQUIRED_OUTPUT_CONSTRAINTS = {
    "allowed_root": ALLOWED_OUTPUT_ROOT,
    "fresh_output_required": True,
    "symlink_output_allowed": False,
    "hardlink_output_allowed": False,
    "input_output_overlap_allowed": False,
    "hermes_source_write_allowed": False,
}
REQUIRED_FREEZE_COMPARISON_FIELDS = frozenset(
    {
        "phase",
        "mode",
        "candidate_only",
        "read_only_inputs",
        "apply_ready",
        "hermes_source_mutation_performed",
        "passed",
        "failed_checks",
        "baseline_file",
        "candidate_file",
        "comparisons",
        "artifacts",
        "output_constraints",
    }
)
FREEZE_COMPARISON_CATEGORIES = frozenset(
    {"function_signatures", "class_names", "decorators", "registry_register_calls", "public_cli_args"}
)
REQUIRED_FREEZE_OUTPUT_CONSTRAINTS = {
    "allowed_root": ALLOWED_OUTPUT_ROOT,
    "fresh_output_required": True,
    "symlink_output_allowed": False,
    "hermes_source_write_allowed": False,
}


@dataclass(frozen=True)
class Phase4ReportContractValidation:
    """Result of a Phase 4 scaffold report contract smoke check."""

    passed: bool
    errors: tuple[str, ...]


def validate_phase4_scaffold_report_contract(report: Mapping[str, Any]) -> Phase4ReportContractValidation:
    """Validate the Phase 4 dry-run scaffold report contract.

    This validator checks structure and safety invariants.  It intentionally does
    not recompute semantic fitness because the dry-run scaffold performs no
    Darwinian execution, no Hermes source mutation, and no benchmark run.
    """

    errors: list[str] = []
    _require_fields(report, REQUIRED_TOP_LEVEL_FIELDS, "report", errors)
    _reject_apply_payload_keys(report, "report", errors)
    if _contains_sensitive_text(report):
        errors.append("report contains sensitive credential-like text")

    if report.get("phase") != "4":
        errors.append('top-level phase must be "4"')
    if report.get("mode") != "code-evolution-candidate-only-scaffold":
        errors.append('top-level mode must be "code-evolution-candidate-only-scaffold"')
    if report.get("scaffold_version") != "phase4-code-scaffold-v1":
        errors.append('scaffold_version must be "phase4-code-scaffold-v1"')

    for key in sorted(TRUE_SCAFFOLD_FLAGS):
        if report.get(key) is not True:
            errors.append(f"top-level {key} must be true")
    for key in sorted(FALSE_SAFETY_FLAGS):
        if report.get(key) is not False:
            errors.append(f"top-level {key} must be false")

    passed = report.get("passed")
    failed_checks = report.get("failed_checks")
    if not isinstance(passed, bool):
        errors.append("passed must be a boolean")
    if not isinstance(failed_checks, list):
        errors.append("failed_checks must be a list")
    elif passed is True and failed_checks:
        errors.append("failed_checks must be empty when passed is true")
    elif passed is False and not failed_checks:
        errors.append("failed_checks must be non-empty when passed is false")

    freeze_checks = _mapping(report.get("freeze_checks"), "freeze_checks", errors)
    for key in ("target_spec", "allowed_mutation", "fitness_plan", "approval_gates", "artifacts", "output_constraints"):
        _mapping(report.get(key), key, errors)
    if freeze_checks is not None:
        _validate_embedded_freeze_gate(report, freeze_checks, errors)
    write_targets = report.get("write_targets")
    if not isinstance(write_targets, list) or not all(isinstance(item, str) for item in write_targets):
        errors.append("write_targets must be a list of strings")
    elif not write_targets:
        errors.append("write_targets must be non-empty")
    else:
        for target in write_targets:
            if not _is_under_phase4_output_root(target):
                errors.append("write target must be under output/phase4-code-evolution")

    artifacts = report.get("artifacts")
    if isinstance(artifacts, Mapping):
        for artifact_key, artifact_value in artifacts.items():
            if not isinstance(artifact_value, str):
                errors.append(f"artifacts.{artifact_key} must be a string")
            elif not _is_under_phase4_output_root(artifact_value):
                errors.append(f"artifacts.{artifact_key} must be under output/phase4-code-evolution")

    constraints = report.get("output_constraints")
    if isinstance(constraints, Mapping):
        for key, expected in REQUIRED_OUTPUT_CONSTRAINTS.items():
            if constraints.get(key) != expected:
                errors.append(f"output_constraints.{key} must be {expected!r}")

    approval_gates = report.get("approval_gates")
    target_spec = report.get("target_spec")
    allowed_mutation = report.get("allowed_mutation")
    target_files: list[str] | None = None
    allowed_files: list[str] | None = None
    if isinstance(target_spec, Mapping):
        raw_target_files = target_spec.get("target_files")
        if not isinstance(raw_target_files, list) or not all(isinstance(item, str) for item in raw_target_files):
            errors.append("target_spec.target_files must be a list of strings")
        else:
            target_files = list(raw_target_files)
            if len(target_files) != 1:
                errors.append("target_spec.target_files must contain exactly one target file")
    if isinstance(allowed_mutation, Mapping):
        files = allowed_mutation.get("files")
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            errors.append("allowed_mutation.files must be a list of strings")
        else:
            allowed_files = list(files)
            if len(allowed_files) != 1:
                errors.append("allowed_mutation.files must contain exactly one target file")
        if allowed_mutation.get("exactly_one_target_file") is not True:
            errors.append("allowed_mutation.exactly_one_target_file must be true")
    if target_files is not None and allowed_files is not None and target_files != allowed_files:
        errors.append("target_spec.target_files must match allowed_mutation.files")

    if isinstance(approval_gates, Mapping):
        for key in ("darwinian_install_approved", "darwinian_execution_approved", "hermes_source_mutation_approved"):
            if approval_gates.get(key) is not False:
                errors.append(f"approval_gates.{key} must be false")
        if approval_gates.get("budget_approved_usd") != 0:
            errors.append("approval_gates.budget_approved_usd must be 0")

    return Phase4ReportContractValidation(passed=not errors, errors=tuple(errors))


def load_and_validate_phase4_scaffold_report(path: str | Path) -> Phase4ReportContractValidation:
    """Load a Phase 4 scaffold report JSON file and validate its contract."""

    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text())
    except json.JSONDecodeError as exc:
        return Phase4ReportContractValidation(passed=False, errors=(f"invalid JSON: {exc}",))
    if not isinstance(payload, Mapping):
        return Phase4ReportContractValidation(passed=False, errors=("report JSON root must be an object",))
    return validate_phase4_scaffold_report_contract(payload)


def validate_phase4_freeze_comparison_report_contract(report: Mapping[str, Any]) -> Phase4ReportContractValidation:
    """Validate a Phase 4 candidate-vs-baseline freeze comparison report."""

    errors: list[str] = []
    _require_fields(report, REQUIRED_FREEZE_COMPARISON_FIELDS, "report", errors)
    _reject_apply_payload_keys(report, "report", errors)
    if _contains_sensitive_text(report):
        errors.append("report contains sensitive credential-like text")

    if report.get("phase") != "4":
        errors.append('top-level phase must be "4"')
    if report.get("mode") != "freeze-comparator":
        errors.append('top-level mode must be "freeze-comparator"')
    for key in ("candidate_only", "read_only_inputs"):
        if report.get(key) is not True:
            errors.append(f"top-level {key} must be true")
    for key in ("apply_ready", "hermes_source_mutation_performed"):
        if report.get(key) is not False:
            errors.append(f"top-level {key} must be false")

    passed = report.get("passed")
    failed_checks = report.get("failed_checks")
    if not isinstance(passed, bool):
        errors.append("passed must be a boolean")
    if not isinstance(failed_checks, list):
        errors.append("failed_checks must be a list")
    elif passed is True and failed_checks:
        errors.append("failed_checks must be empty when passed is true")
    elif passed is False and not failed_checks:
        errors.append("failed_checks must be non-empty when passed is false")

    for key in ("baseline_file", "candidate_file"):
        if not isinstance(report.get(key), str) or not str(report.get(key)).strip():
            errors.append(f"{key} must be a non-empty string")

    comparisons = _mapping(report.get("comparisons"), "comparisons", errors)
    if comparisons is not None:
        for category in sorted(FREEZE_COMPARISON_CATEGORIES):
            category_value = _mapping(comparisons.get(category), f"comparisons.{category}", errors)
            if category_value is None:
                continue
            for change_kind in ("changed", "added", "removed"):
                if not isinstance(category_value.get(change_kind), list):
                    errors.append(f"comparisons.{category}.{change_kind} must be a list")

    artifacts = _mapping(report.get("artifacts"), "artifacts", errors)
    if artifacts is not None:
        for artifact_key, artifact_value in artifacts.items():
            if not isinstance(artifact_value, str):
                errors.append(f"artifacts.{artifact_key} must be a string")
            elif not _is_under_phase4_output_root(artifact_value):
                errors.append(f"artifacts.{artifact_key} must be under output/phase4-code-evolution")

    constraints = _mapping(report.get("output_constraints"), "output_constraints", errors)
    if constraints is not None:
        for key, expected in REQUIRED_FREEZE_OUTPUT_CONSTRAINTS.items():
            if constraints.get(key) != expected:
                errors.append(f"output_constraints.{key} must be {expected!r}")

    return Phase4ReportContractValidation(passed=not errors, errors=tuple(errors))


def _validate_embedded_freeze_gate(
    report: Mapping[str, Any],
    freeze_checks: Mapping[str, Any],
    errors: list[str],
) -> None:
    mandatory_gate = _mapping(freeze_checks.get("mandatory_gate"), "freeze_checks.mandatory_gate", errors)
    if mandatory_gate is not None:
        if mandatory_gate.get("required_before_apply") is not True:
            errors.append("freeze_checks.mandatory_gate.required_before_apply must be true")
        if freeze_checks.get("mode") == "freeze-comparator":
            if mandatory_gate.get("executed") is not True:
                errors.append("freeze_checks.mandatory_gate.executed must be true for freeze comparator reports")
            if mandatory_gate.get("passed") != freeze_checks.get("passed"):
                errors.append("freeze_checks.mandatory_gate.passed must match freeze comparator passed")
        else:
            if mandatory_gate.get("executed") is not False:
                errors.append("freeze_checks.mandatory_gate.executed must be false before a candidate exists")
            if mandatory_gate.get("passed") is not None:
                errors.append("freeze_checks.mandatory_gate.passed must be null before a candidate exists")

    if freeze_checks.get("mode") != "freeze-comparator":
        return

    freeze_validation = validate_phase4_freeze_comparison_report_contract(freeze_checks)
    for error in freeze_validation.errors:
        errors.append(f"freeze_checks.{error}")

    if freeze_checks.get("passed") is False:
        if report.get("passed") is not False:
            errors.append("top-level passed must be false when freeze comparator fails")
        failed_checks = report.get("failed_checks")
        if not isinstance(failed_checks, list) or not any(
            isinstance(item, str) and item.startswith("freeze_comparator:") for item in failed_checks
        ):
            errors.append("top-level failed_checks must include freeze_comparator entries when freeze comparator fails")


def load_and_validate_phase4_freeze_comparison_report(path: str | Path) -> Phase4ReportContractValidation:
    """Load a Phase 4 freeze comparison report JSON file and validate it."""

    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text())
    except json.JSONDecodeError as exc:
        return Phase4ReportContractValidation(passed=False, errors=(f"invalid JSON: {exc}",))
    if not isinstance(payload, Mapping):
        return Phase4ReportContractValidation(passed=False, errors=("report JSON root must be an object",))
    return validate_phase4_freeze_comparison_report_contract(payload)


def _require_fields(mapping: Mapping[str, Any], required: frozenset[str], path: str, errors: list[str]) -> None:
    missing = sorted(required - set(mapping))
    for field in missing:
        errors.append(f"{path} missing required field: {field}")


def _mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return None
    return value


def _reject_apply_payload_keys(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_string = str(key)
            if key_string in APPLY_PAYLOAD_KEYS:
                errors.append(f"scaffold report must not contain apply payload key: {key_string}")
            _reject_apply_payload_keys(nested, f"{path}.{key_string}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_apply_payload_keys(item, f"{path}[{index}]", errors)


def _contains_sensitive_text(value: Any) -> bool:
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in SENSITIVE_TEXT_PATTERNS)
    if isinstance(value, Mapping):
        return any(_contains_sensitive_text(key) or _contains_sensitive_text(nested) for key, nested in value.items())
    if isinstance(value, list):
        return any(_contains_sensitive_text(item) for item in value)
    return False


def _is_under_phase4_output_root(path_string: str) -> bool:
    path = Path(path_string)
    if ".." in path.parts:
        return False
    candidate = path if path.is_absolute() else REPO_ROOT / path
    resolved_candidate = candidate.resolve(strict=False)
    resolved_allowed_root = PHASE4_OUTPUT_ROOT.resolve(strict=False)
    return resolved_candidate == resolved_allowed_root or resolved_candidate.is_relative_to(resolved_allowed_root)


def load_and_validate_phase4_report(path: str | Path) -> Phase4ReportContractValidation:
    """Load either a scaffold report or freeze-comparison report and validate it."""

    report_path = Path(path)
    try:
        payload = json.loads(report_path.read_text())
    except json.JSONDecodeError as exc:
        return Phase4ReportContractValidation(passed=False, errors=(f"invalid JSON: {exc}",))
    if not isinstance(payload, Mapping):
        return Phase4ReportContractValidation(passed=False, errors=("report JSON root must be an object",))
    if payload.get("mode") == "freeze-comparator":
        return validate_phase4_freeze_comparison_report_contract(payload)
    return validate_phase4_scaffold_report_contract(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Phase 4 dry-run scaffold or freeze-comparison report contract.")
    parser.add_argument("report_json", help="Path to scaffold_report.json or freeze_comparison_report.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validation = load_and_validate_phase4_report(args.report_json)
    print(json.dumps({"passed": validation.passed, "errors": list(validation.errors)}, indent=2, sort_keys=True))
    return 0 if validation.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

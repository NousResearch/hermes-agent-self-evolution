"""Dry-run-only Phase 4 code-evolution scaffold.

This module prepares review artifacts for future code evolution.  It performs no
Darwinian Evolver import/execution, no external calls, no package installation,
no Hermes source mutation, and no active runtime apply.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

from evolution.code.report_contract import validate_phase4_scaffold_report_contract
from evolution.code.target_contract import Phase4TargetSpec, load_target_spec

SCAFFOLD_VERSION = "phase4-code-scaffold-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_ROOT = "output/phase4-code-evolution/"
PHASE4_OUTPUT_ROOT = REPO_ROOT / ALLOWED_OUTPUT_ROOT
OUTPUT_ARTIFACT_NAMES = (
    "target_snapshot.json",
    "baseline_api_surface.json",
    "freeze_report.json",
    "reproduction_contract.json",
    "scaffold_report.json",
    "review_packet.md",
)
SEPARATE_APPROVAL_REQUIRED_BEFORE = [
    "installing Darwinian Evolver or new dependencies",
    "importing or executing Darwinian Evolver",
    "creating mutation worktrees that write Hermes Agent source",
    "running networked or paid benchmarks",
    "pushing a branch or opening a PR",
    "modifying active Hermes runtime, gateway, memory, skills, SOUL, profiles, or model routing",
]


class Phase4CodeScaffoldFailed(ValueError):
    """Raised when a scaffold report is written but fails its contract."""


def run_phase4_code_scaffold(
    *,
    target_spec: str | Path,
    output_dir: str | Path,
    dry_run: bool,
    hermes_repo: str | Path | None = None,
) -> dict[str, Any]:
    """Write Phase 4 dry-run scaffold artifacts under the allowed output root."""

    if dry_run is not True:
        raise ValueError("Phase 4 code scaffold currently requires --dry-run")

    spec = load_target_spec(target_spec, hermes_repo_override=hermes_repo)
    output_path = _normalize_output_dir(Path(output_dir))
    _validate_output_dir(output_path)
    artifact_paths = {name: output_path / name for name in OUTPUT_ARTIFACT_NAMES}
    _validate_write_targets(
        list(artifact_paths.values()),
        input_paths=(spec.source_path, *spec.target_files),
    )

    target_snapshot = _build_target_snapshot(spec)
    baseline_api_surface = _collect_api_surface(spec.target_files[0])
    freeze_report = _build_freeze_report(spec, artifact_paths["baseline_api_surface.json"])
    reproduction_contract = _build_reproduction_contract(spec)
    report = _build_scaffold_report(
        spec=spec,
        output_path=output_path,
        artifact_paths=artifact_paths,
        freeze_report=freeze_report,
        reproduction_contract=reproduction_contract,
    )

    validation = validate_phase4_scaffold_report_contract(report)
    if not validation.passed:
        raise Phase4CodeScaffoldFailed("; ".join(validation.errors))

    output_path.mkdir(parents=True, exist_ok=True)
    artifact_paths["target_snapshot.json"].write_text(json.dumps(target_snapshot, indent=2, sort_keys=True) + "\n")
    artifact_paths["baseline_api_surface.json"].write_text(
        json.dumps(baseline_api_surface, indent=2, sort_keys=True) + "\n"
    )
    artifact_paths["freeze_report.json"].write_text(json.dumps(freeze_report, indent=2, sort_keys=True) + "\n")
    artifact_paths["reproduction_contract.json"].write_text(
        json.dumps(reproduction_contract, indent=2, sort_keys=True) + "\n"
    )
    artifact_paths["scaffold_report.json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    artifact_paths["review_packet.md"].write_text(_render_review_packet(report) + "\n")
    return report


def _build_target_snapshot(spec: Phase4TargetSpec) -> dict[str, Any]:
    targets = []
    for target_file, relative in zip(spec.target_files, spec.relative_target_files):
        raw = target_file.read_bytes()
        targets.append(
            {
                "relative_path": relative,
                "target_file": str(target_file),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "symlink": target_file.is_symlink(),
            }
        )
    return {
        "phase": "4",
        "mode": "target-snapshot",
        "target_id": spec.target_id,
        "source_spec_path": str(spec.source_path),
        "hermes_repo": str(spec.hermes_repo),
        "base_ref": spec.base_ref,
        "targets": targets,
        "read_only_snapshot": True,
    }


def _build_freeze_report(spec: Phase4TargetSpec, baseline_api_surface_path: Path) -> dict[str, Any]:
    return {
        "phase": "4",
        "mode": "freeze-check-plan",
        "function_signatures_required": spec.freeze.get("function_signatures") is True,
        "registry_register_calls_required": spec.freeze.get("registry_register_calls") is True,
        "public_cli_args_required": spec.freeze.get("public_cli_args") is True,
        "candidate_generated": False,
        "baseline_api_surface_path": str(baseline_api_surface_path),
        "candidate_api_surface_path": None,
        "checks_executed": False,
        "status": "not_run_no_candidate",
    }


def _build_reproduction_contract(spec: Phase4TargetSpec) -> dict[str, Any]:
    return {
        "phase": "4",
        "mode": "reproduction-contract",
        "failing_case_description": spec.reproduction["failing_case_description"],
        "reproducer_command": spec.reproduction["reproducer_command"],
        "baseline_reproduction_executed": False,
        "improvement_claimed": False,
        "objective_metric_delta_recorded": False,
    }


def _build_scaffold_report(
    *,
    spec: Phase4TargetSpec,
    output_path: Path,
    artifact_paths: Mapping[str, Path],
    freeze_report: Mapping[str, Any],
    reproduction_contract: Mapping[str, Any],
) -> dict[str, Any]:
    failed_checks: list[str] = []
    return {
        "phase": "4",
        "mode": "code-evolution-candidate-only-scaffold",
        "scaffold_version": SCAFFOLD_VERSION,
        "dry_run": True,
        "candidate_only": True,
        "read_only_inputs": True,
        "darwinian_cli_invoked": False,
        "darwinian_imported": False,
        "external_calls_performed": False,
        "package_installed": False,
        "hermes_source_mutation_performed": False,
        "active_runtime_apply_approved": False,
        "apply_ready": False,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "target_spec": spec.to_report_payload(),
        "allowed_mutation": {
            "files": list(spec.relative_target_files),
            "deny_globs": list(spec.deny_globs),
            "exactly_one_target_file": len(spec.relative_target_files) == 1,
        },
        "freeze_checks": dict(freeze_report),
        "fitness_plan": {
            "required_commands": list(spec.fitness["required_commands"]),
            "commands_executed": False,
            "benchmarks_required_before_acceptance": spec.benchmarks[
                "full_benchmark_required_before_acceptance"
            ],
            "benchmarks_run_now": spec.benchmarks["run_benchmarks_now"],
        },
        "approval_gates": {
            "darwinian_install_approved": spec.approvals["darwinian_install_approved"],
            "darwinian_execution_approved": spec.approvals["darwinian_execution_approved"],
            "hermes_source_mutation_approved": spec.approvals["hermes_source_mutation_approved"],
            "budget_approved_usd": spec.approvals["budget_approved_usd"],
            "separate_approval_required_before": list(SEPARATE_APPROVAL_REQUIRED_BEFORE),
        },
        "reproduction_contract": dict(reproduction_contract),
        "artifacts": {
            "output_dir": str(output_path),
            "target_snapshot": str(artifact_paths["target_snapshot.json"]),
            "baseline_api_surface": str(artifact_paths["baseline_api_surface.json"]),
            "freeze_report": str(artifact_paths["freeze_report.json"]),
            "reproduction_contract": str(artifact_paths["reproduction_contract.json"]),
            "scaffold_report": str(artifact_paths["scaffold_report.json"]),
            "review_packet": str(artifact_paths["review_packet.md"]),
        },
        "write_targets": [str(artifact_paths[name]) for name in OUTPUT_ARTIFACT_NAMES],
        "output_constraints": {
            "allowed_root": ALLOWED_OUTPUT_ROOT,
            "fresh_output_required": True,
            "symlink_output_allowed": False,
            "hardlink_output_allowed": False,
            "input_output_overlap_allowed": False,
            "hermes_source_write_allowed": False,
        },
    }


def _collect_api_surface(target_file: Path) -> dict[str, Any]:
    raw = target_file.read_text()
    try:
        tree = ast.parse(raw, filename=str(target_file))
    except SyntaxError as exc:
        raise ValueError(f"target file must parse as Python before Phase 4 scaffolding: {target_file}") from exc

    module_functions: list[str] = []
    class_methods: list[str] = []
    classes: list[str] = []
    registry_calls: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_methods.append(f"{node.name}.{item.name}")
    for node in ast.walk(tree):
        if _is_registry_register_call(node):
            registry_calls.append(_summarize_registry_call(cast(ast.Call, node)))
    return {
        "phase": "4",
        "mode": "baseline-api-surface",
        "target_file": str(target_file),
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "module_functions": sorted(module_functions),
        "classes": sorted(classes),
        "class_methods": sorted(class_methods),
        "registry_register_calls": registry_calls,
    }


def _is_registry_register_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "register"
        and isinstance(func.value, ast.Name)
        and func.value.id == "registry"
    )


def _summarize_registry_call(call: ast.Call) -> dict[str, Any]:
    keyword_values: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        keyword_values[keyword.arg] = _safe_literal(keyword.value)
    handler = keyword_values.get("handler")
    if not isinstance(handler, str):
        handler = _node_name(next((keyword.value for keyword in call.keywords if keyword.arg == "handler"), None))
    schema = keyword_values.get("schema")
    schema_parameters: list[str] = []
    schema_required: list[str] = []
    if isinstance(schema, Mapping):
        parameters = schema.get("parameters")
        if isinstance(parameters, Mapping):
            schema_parameters = sorted(str(key) for key in parameters)
        required = schema.get("required")
        if isinstance(required, list):
            schema_required = sorted(str(item) for item in required)
    return {
        "name": keyword_values.get("name"),
        "toolset": keyword_values.get("toolset"),
        "handler": handler,
        "schema_parameters": schema_parameters,
        "schema_required": schema_required,
    }


def _safe_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return _node_name(node)


def _node_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _node_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _normalize_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _validate_output_dir(output_dir: Path) -> None:
    if PHASE4_OUTPUT_ROOT.is_symlink():
        raise ValueError(f"output root must not be a symlink: {PHASE4_OUTPUT_ROOT}")
    for parent in PHASE4_OUTPUT_ROOT.parents:
        if parent == REPO_ROOT:
            break
        if parent.exists() and parent.is_symlink():
            raise ValueError(f"output root parent must not be a symlink: {parent}")
    allowed_root = PHASE4_OUTPUT_ROOT.resolve(strict=False)
    if output_dir.is_symlink():
        raise ValueError(f"output-dir must not be a symlink: {output_dir}")
    for parent in output_dir.parents:
        if parent == output_dir:
            break
        if parent == PHASE4_OUTPUT_ROOT or parent.resolve(strict=False) == allowed_root:
            break
        if parent.exists() and parent.is_symlink():
            raise ValueError(f"output-dir parent must not be a symlink: {parent}")
    resolved_output_dir = output_dir.resolve(strict=False)
    if resolved_output_dir == allowed_root:
        return
    if not resolved_output_dir.is_relative_to(allowed_root):
        raise ValueError(f"output-dir must stay under {ALLOWED_OUTPUT_ROOT}: {output_dir}")


def _validate_write_targets(write_targets: Iterable[Path], input_paths: tuple[Path, ...]) -> None:
    allowed_root = PHASE4_OUTPUT_ROOT.resolve()
    input_resolved = {path.resolve() for path in input_paths}
    for target in write_targets:
        target_resolved = target.resolve(strict=False)
        if not target_resolved.is_relative_to(allowed_root):
            raise ValueError(f"write target must stay under {ALLOWED_OUTPUT_ROOT}: {target}")
        if target.is_symlink():
            raise ValueError(f"write target must not be a symlink: {target}")
        if target.exists():
            for input_path in input_paths:
                if target.samefile(input_path):
                    raise ValueError(f"write target must not overwrite input artifact: {target}")
            raise ValueError(f"write target must not already exist: {target}")
        if target_resolved in input_resolved:
            raise ValueError(f"write target must not overwrite input artifact: {target}")


def _render_review_packet(report: Mapping[str, Any]) -> str:
    failed_checks = report.get("failed_checks")
    if not isinstance(failed_checks, list):
        failed_checks = []
    target_spec = report.get("target_spec") if isinstance(report.get("target_spec"), Mapping) else {}
    target_files = target_spec.get("target_files", []) if isinstance(target_spec, Mapping) else []
    return "\n".join(
        [
            "# Phase 4 code-evolution dry-run scaffold review packet",
            "",
            "This scaffold records review artifacts only. It performs no Darwinian Evolver import/execution, no package installation, no benchmark spend, no Hermes source mutation, and no active runtime apply.",
            "",
            f"- passed: `{str(report.get('passed')).lower()}`",
            f"- apply_ready: `{str(report.get('apply_ready')).lower()}`",
            f"- darwinian_cli_invoked: `{str(report.get('darwinian_cli_invoked')).lower()}`",
            f"- hermes_source_mutation_performed: `{str(report.get('hermes_source_mutation_performed')).lower()}`",
            f"- target_files: `{', '.join(str(item) for item in target_files)}`",
            f"- failed_checks: `{', '.join(str(check) for check in failed_checks) if failed_checks else 'none'}`",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Phase 4 dry-run code-evolution scaffold artifacts.")
    parser.add_argument("--target-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hermes-repo", default=None, help="Optional Hermes repo override for the target spec")
    parser.add_argument("--dry-run", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_phase4_code_scaffold(
            target_spec=args.target_spec,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            hermes_repo=args.hermes_repo,
        )
    except (ValueError, Phase4CodeScaffoldFailed) as exc:
        print(f"phase4 code scaffold failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"passed": report["passed"], "report": report["artifacts"]["scaffold_report"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

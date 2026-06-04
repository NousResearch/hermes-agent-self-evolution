"""Phase 4 candidate-vs-baseline freeze comparator.

This module is read-only and candidate-only.  It compares a future evolved
candidate file against a baseline file and fails visibly when public function
signatures, registry.register() calls, or public CLI arguments drift.  It does
not import Darwinian Evolver, apply patches, mutate Hermes source, or run
benchmarks.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, cast

from evolution.code.report_contract import validate_phase4_freeze_comparison_report_contract

COMPARATOR_VERSION = "phase4-freeze-comparator-v1"
ALLOWED_OUTPUT_ROOT = "output/phase4-code-evolution/"
REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE4_OUTPUT_ROOT = REPO_ROOT / ALLOWED_OUTPUT_ROOT


class FreezeComparatorError(ValueError):
    """Raised when the freeze comparator cannot safely produce a report."""


def compare_candidate_to_baseline(
    *,
    baseline_file: str | Path,
    candidate_file: str | Path,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Compare a candidate Python file against a baseline freeze surface.

    The comparison is intentionally structural and read-only.  A PASS means the
    candidate preserved frozen public signatures, registry registration calls,
    and public CLI flags; it is not approval to apply or merge candidate code.
    """

    baseline_original = Path(baseline_file).expanduser()
    candidate_original = Path(candidate_file).expanduser()
    _validate_input_file(baseline_original, "baseline-file")
    _validate_input_file(candidate_original, "candidate-file")
    baseline_path = baseline_original.resolve()
    candidate_path = candidate_original.resolve()

    baseline_surface = collect_freeze_surface(baseline_path)
    candidate_surface = collect_freeze_surface(candidate_path)
    comparisons = _compare_surfaces(baseline_surface, candidate_surface)
    failed_checks = _failed_checks_from_comparisons(comparisons)
    artifacts: dict[str, str] = {}
    if output_json is not None:
        artifacts["output_json"] = str(_normalize_output_json(Path(output_json)))

    report: dict[str, Any] = {
        "phase": "4",
        "mode": "freeze-comparator",
        "comparator_version": COMPARATOR_VERSION,
        "candidate_only": True,
        "read_only_inputs": True,
        "apply_ready": False,
        "hermes_source_mutation_performed": False,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "baseline_file": str(baseline_path),
        "candidate_file": str(candidate_path),
        "baseline_api_surface": baseline_surface,
        "candidate_api_surface": candidate_surface,
        "comparisons": comparisons,
        "artifacts": artifacts,
        "output_constraints": {
            "allowed_root": ALLOWED_OUTPUT_ROOT,
            "fresh_output_required": True,
            "symlink_output_allowed": False,
            "hermes_source_write_allowed": False,
        },
    }
    validation = validate_phase4_freeze_comparison_report_contract(report)
    if not validation.passed:
        raise FreezeComparatorError("; ".join(validation.errors))
    return report


def collect_freeze_surface(path: str | Path) -> dict[str, Any]:
    """Collect the public API surface guarded by Phase 4 freeze checks."""

    target_original = Path(path).expanduser()
    _validate_input_file(target_original, "target-file")
    target_file = target_original.resolve()
    raw = target_file.read_text()
    try:
        tree = ast.parse(raw, filename=str(target_file))
    except SyntaxError as exc:
        raise FreezeComparatorError(f"target file must parse as Python: {target_file}") from exc

    return {
        "target_file": str(target_file),
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "function_signatures": _collect_function_signatures(tree),
        "class_names": _collect_class_names(tree),
        "decorators": _collect_decorators(tree),
        "registry_register_calls": _collect_registry_register_calls(tree),
        "public_cli_args": _collect_public_cli_args(tree),
    }


def _collect_function_signatures(tree: ast.Module) -> dict[str, str]:
    signatures: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(node.name):
            signatures[node.name] = _signature_string(node)
        elif isinstance(node, ast.ClassDef) and _is_public_name(node.name):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(item.name):
                    signatures[f"{node.name}.{item.name}"] = _signature_string(item)
    return dict(sorted(signatures.items()))


def _collect_class_names(tree: ast.Module) -> dict[str, str]:
    return {node.name: node.name for node in tree.body if isinstance(node, ast.ClassDef) and _is_public_name(node.name)}


def _collect_decorators(tree: ast.Module) -> dict[str, list[str]]:
    decorators: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(node.name):
            decorators[node.name] = _decorator_strings(node.decorator_list)
        elif isinstance(node, ast.ClassDef) and _is_public_name(node.name):
            decorators[node.name] = _decorator_strings(node.decorator_list)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(item.name):
                    decorators[f"{node.name}.{item.name}"] = _decorator_strings(item.decorator_list)
    return dict(sorted(decorators.items()))


def _decorator_strings(decorators: list[ast.expr]) -> list[str]:
    return [ast.unparse(decorator) for decorator in decorators]


def _signature_string(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix}({args}){returns}"


def _collect_registry_register_calls(tree: ast.Module) -> dict[str, dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    key_counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if _is_registry_register_call(node):
            summary = _summarize_registry_call(cast(ast.Call, node))
            base_key = _registry_call_key(summary)
            key_counts[base_key] = key_counts.get(base_key, 0) + 1
            key = base_key if key_counts[base_key] == 1 else f"{base_key}#{key_counts[base_key]}"
            calls[key] = summary
    return dict(sorted(calls.items()))


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
    kwargs_expansions: list[Any] = []
    for keyword in call.keywords:
        if keyword.arg is None:
            kwargs_expansions.append(_safe_literal(keyword.value))
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

    extra_keywords = {
        key: _canonicalize_literal(value)
        for key, value in keyword_values.items()
        if key not in {"name", "toolset", "handler", "schema"}
    }
    return {
        "name": keyword_values.get("name"),
        "toolset": keyword_values.get("toolset"),
        "handler": handler,
        "schema": _canonicalize_literal(schema),
        "schema_parameters": schema_parameters,
        "schema_required": schema_required,
        "extra_keywords": dict(sorted(extra_keywords.items())),
        "kwargs_expansions": _canonicalize_literal(kwargs_expansions),
    }


def _registry_call_key(summary: Mapping[str, Any]) -> str:
    name = summary.get("name")
    if isinstance(name, str) and name:
        return name
    handler = summary.get("handler")
    if isinstance(handler, str) and handler:
        return handler
    return json.dumps(summary, sort_keys=True)


def _collect_public_cli_args(tree: ast.Module) -> dict[str, dict[str, Any]]:
    args: dict[str, dict[str, Any]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_add_argument_call(node):
            continue
        flags = [
            literal
            for literal in (_literal_string(arg) for arg in node.args)
            if literal is not None and literal.startswith("-")
        ]
        if not flags:
            continue
        keyword_values: dict[str, Any] = {}
        expansion_index = 0
        for keyword in node.keywords:
            if keyword.arg is None:
                keyword_values[f"**kwargs_{expansion_index}"] = _canonicalize_literal(_safe_literal(keyword.value))
                expansion_index += 1
            else:
                keyword_values[keyword.arg] = _canonicalize_literal(_safe_literal(keyword.value))
        for flag in flags:
            args[flag] = {"flag": flag, "options": dict(sorted(keyword_values.items()))}
    return dict(sorted(args.items()))


def _is_add_argument_call(call: ast.Call) -> bool:
    func = call.func
    return isinstance(func, ast.Attribute) and func.attr == "add_argument"


def _compare_surfaces(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "function_signatures": _compare_mapping(
            _mapping_of_strings(baseline.get("function_signatures")),
            _mapping_of_strings(candidate.get("function_signatures")),
        ),
        "class_names": _compare_mapping(
            _mapping_of_strings(baseline.get("class_names")),
            _mapping_of_strings(candidate.get("class_names")),
        ),
        "decorators": _compare_mapping(
            _mapping_of_mappings(baseline.get("decorators")),
            _mapping_of_mappings(candidate.get("decorators")),
        ),
        "registry_register_calls": _compare_mapping(
            _mapping_of_mappings(baseline.get("registry_register_calls")),
            _mapping_of_mappings(candidate.get("registry_register_calls")),
        ),
        "public_cli_args": _compare_mapping(
            _mapping_of_mappings(baseline.get("public_cli_args")),
            _mapping_of_mappings(candidate.get("public_cli_args")),
        ),
    }


def _compare_mapping(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    changed = [
        {"name": key, "baseline": baseline[key], "candidate": candidate[key]}
        for key in sorted(baseline_keys & candidate_keys)
        if baseline[key] != candidate[key]
    ]
    return {
        "changed": changed,
        "added": [{"name": key, "candidate": candidate[key]} for key in sorted(candidate_keys - baseline_keys)],
        "removed": [{"name": key, "baseline": baseline[key]} for key in sorted(baseline_keys - candidate_keys)],
    }


def _failed_checks_from_comparisons(comparisons: Mapping[str, Any]) -> list[str]:
    failed_checks: list[str] = []
    for category, prefix in (
        ("function_signatures", "function_signature"),
        ("class_names", "class_name"),
        ("decorators", "decorator"),
        ("registry_register_calls", "registry_register"),
        ("public_cli_args", "public_cli_arg"),
    ):
        result = comparisons.get(category)
        if not isinstance(result, Mapping):
            failed_checks.append(f"{prefix}_comparison_missing")
            continue
        for change in result.get("changed", []):
            failed_checks.append(f"{prefix}_changed:{_change_name(change)}")
        for change in result.get("added", []):
            failed_checks.append(f"{prefix}_added:{_change_name(change)}")
        for change in result.get("removed", []):
            failed_checks.append(f"{prefix}_removed:{_change_name(change)}")
    return failed_checks


def _change_name(change: Any) -> str:
    if isinstance(change, Mapping):
        name = change.get("name")
        if isinstance(name, str):
            return name
    return "unknown"


def _mapping_of_strings(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(nested) for key, nested in value.items()}


def _mapping_of_mappings(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): nested for key, nested in value.items()}


def _validate_input_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise FreezeComparatorError(f"{label} must not be a symlink: {path}")
    if not path.exists() or not path.is_file():
        raise FreezeComparatorError(f"{label} must exist and be a file: {path}")
    if path.suffix != ".py":
        raise FreezeComparatorError(f"{label} must be a Python file: {path}")


def _normalize_output_json(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if ".." in candidate.parts:
        raise FreezeComparatorError("output-json must not contain traversal segments")
    resolved_candidate = candidate.resolve(strict=False)
    allowed_root = PHASE4_OUTPUT_ROOT.resolve(strict=False)
    if resolved_candidate != allowed_root and not resolved_candidate.is_relative_to(allowed_root):
        raise FreezeComparatorError(f"output-json must stay under {ALLOWED_OUTPUT_ROOT}")
    if candidate.is_symlink():
        raise FreezeComparatorError(f"output-json must not be a symlink: {candidate}")
    if candidate.exists():
        raise FreezeComparatorError(f"output-json must not already exist: {candidate}")
    for parent in candidate.parents:
        if parent == PHASE4_OUTPUT_ROOT or parent.resolve(strict=False) == allowed_root:
            break
        if parent.exists() and parent.is_symlink():
            raise FreezeComparatorError(f"output-json parent must not be a symlink: {parent}")
    return candidate


def _canonicalize_literal(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize_literal(nested) for key, nested in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize_literal(item) for item in value]
    return value


def _safe_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _node_name(node) or ast.unparse(node)
    if isinstance(node, ast.Dict):
        result: dict[str, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            key = "**" if key_node is None else _safe_literal(key_node)
            result[str(key)] = _safe_literal(value_node)
        return result
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_safe_literal(item) for item in node.elts]
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _canonicalize_literal(value)


def _literal_string(node: ast.AST) -> str | None:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, str) else None


def _node_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _node_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _is_public_name(name: str) -> bool:
    return not name.startswith("_")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Phase 4 candidate API surface against a frozen baseline.")
    parser.add_argument("--baseline-file", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--output-json", default=None, help="Optional report path under output/phase4-code-evolution/")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = compare_candidate_to_baseline(
            baseline_file=args.baseline_file,
            candidate_file=args.candidate_file,
            output_json=args.output_json,
        )
    except FreezeComparatorError as exc:
        print(f"phase4 freeze comparator failed: {exc}", file=sys.stderr)
        return 2

    if args.output_json is not None:
        output_path = Path(report["artifacts"]["output_json"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    summary = {
        "passed": report["passed"],
        "failed_checks": report["failed_checks"],
        "report": report["artifacts"].get("output_json"),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

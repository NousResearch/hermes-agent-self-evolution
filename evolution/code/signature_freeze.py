"""AST-level invariants that candidate code must preserve.

Phase 4 mutations are allowed to change function bodies but must not:

- alter any top-level public function signature
- add/remove/rename ``registry.register(...)`` or ``register_tool(...)`` calls
- decrease the number of ``try``/``except`` blocks (proxy for error-handling
  coverage; if a mutation removes a guard, we want to flag it before pytest)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _unparse_args(node: ast.arguments) -> str:
    return ast.unparse(node) if hasattr(ast, "unparse") else ""


def extract_public_signatures(source: str) -> dict[str, str]:
    """Return ``{function_name: serialized_args}`` for module-level public defs."""
    tree = ast.parse(source)
    sigs: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            sigs[node.name] = _unparse_args(node.args)
    return sigs


def extract_registry_calls(source: str) -> list[str]:
    """Return ``registry.register(<name>)`` / ``register_tool(<name>)`` first-arg strings."""
    tree = ast.parse(source)
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = ""
        if isinstance(node.func, ast.Attribute) and node.func.attr == "register":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "registry":
                target = "registry.register"
        elif isinstance(node.func, ast.Name) and node.func.id in {"register_tool", "register"}:
            target = node.func.id
        if not target:
            continue
        first_arg = ""
        if node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                first_arg = arg.value
            else:
                first_arg = ast.unparse(arg) if hasattr(ast, "unparse") else type(arg).__name__
        calls.append(f"{target}({first_arg})")
    return sorted(calls)


def count_try_except(source: str) -> int:
    tree = ast.parse(source)
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.Try))


@dataclass(frozen=True)
class FreezeReport:
    """Structured outcome of a signature/registry/error-handling diff."""

    signature_violations: list[str] = field(default_factory=list)
    registry_violations: list[str] = field(default_factory=list)
    error_handling_baseline: int = 0
    error_handling_candidate: int = 0

    @property
    def error_handling_decreased(self) -> bool:
        return self.error_handling_candidate < self.error_handling_baseline

    @property
    def passed(self) -> bool:
        return (
            not self.signature_violations
            and not self.registry_violations
            and not self.error_handling_decreased
        )

    def to_dict(self) -> dict:
        return {
            "signature_violations": list(self.signature_violations),
            "registry_violations": list(self.registry_violations),
            "error_handling_baseline": self.error_handling_baseline,
            "error_handling_candidate": self.error_handling_candidate,
            "error_handling_decreased": self.error_handling_decreased,
            "passed": self.passed,
        }


def compare_files(baseline_path: Path, candidate_path: Path) -> FreezeReport:
    return compare_sources(
        baseline_path.read_text(encoding="utf-8"),
        candidate_path.read_text(encoding="utf-8"),
    )


def compare_sources(baseline_src: str, candidate_src: str) -> FreezeReport:
    """Diff baseline against candidate AST invariants."""
    try:
        baseline_sigs = extract_public_signatures(baseline_src)
        candidate_sigs = extract_public_signatures(candidate_src)
        baseline_regs = extract_registry_calls(baseline_src)
        candidate_regs = extract_registry_calls(candidate_src)
        baseline_try = count_try_except(baseline_src)
        candidate_try = count_try_except(candidate_src)
    except SyntaxError as exc:
        return FreezeReport(
            signature_violations=[f"candidate failed to parse: {exc}"],
            registry_violations=[],
            error_handling_baseline=0,
            error_handling_candidate=0,
        )

    sig_violations: list[str] = []
    for name, args in baseline_sigs.items():
        if name not in candidate_sigs:
            sig_violations.append(f"public function '{name}' removed")
        elif candidate_sigs[name] != args:
            sig_violations.append(
                f"public function '{name}' signature changed: {args!r} -> {candidate_sigs[name]!r}",
            )
    for name in candidate_sigs:
        if name not in baseline_sigs:
            sig_violations.append(f"new public function '{name}' added (signature freeze)")

    baseline_reg_set = set(baseline_regs)
    candidate_reg_set = set(candidate_regs)
    reg_violations: list[str] = []
    for missing in sorted(baseline_reg_set - candidate_reg_set):
        reg_violations.append(f"registry call removed: {missing}")
    for added in sorted(candidate_reg_set - baseline_reg_set):
        reg_violations.append(f"registry call added: {added}")

    return FreezeReport(
        signature_violations=sig_violations,
        registry_violations=reg_violations,
        error_handling_baseline=baseline_try,
        error_handling_candidate=candidate_try,
    )

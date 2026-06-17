"""Phase 4 code-evolution target specification contract.

The target contract is deliberately fail-closed.  It describes one candidate
Hermes source file for later code evolution, but it does not authorize source
mutation, Darwinian Evolver execution, benchmark spend, or active runtime apply.
"""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_DENY_GLOBS = (
    "config*.py",
    "*credential*",
    "*secret*",
    "*token*",
    "*password*",
    "**/config*.py",
    "**/*credential*",
    "**/*secret*",
    "**/*token*",
    "**/*password*",
    "skills/**",
    "plugins/**",
    "memory/**",
    "memories/**",
    "profiles/**",
    "gateway_state*.json",
    "*.env",
    ".env",
)
SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9_]*(?:api[_-]?key|access[_-]?token|secret|password)[A-Za-z0-9_]*\s*[:=]\s*['\"]?[^'\"\s]{8,}", re.IGNORECASE),
)
REQUIRED_APPROVAL_FLAGS = (
    "darwinian_install_approved",
    "darwinian_execution_approved",
    "hermes_source_mutation_approved",
)


class TargetContractError(ValueError):
    """Raised when a Phase 4 target specification violates the contract."""


@dataclass(frozen=True)
class Phase4TargetSpec:
    """Validated Phase 4 code-evolution target specification."""

    source_path: Path
    raw: Mapping[str, Any]
    phase: str
    mode: str
    target_id: str
    hermes_repo: Path
    base_ref: str
    require_clean_worktree: bool
    target_files: tuple[Path, ...]
    relative_target_files: tuple[str, ...]
    deny_globs: tuple[str, ...]
    freeze: Mapping[str, Any]
    reproduction: Mapping[str, Any]
    fitness: Mapping[str, Any]
    benchmarks: Mapping[str, Any]
    approvals: Mapping[str, Any]

    def to_report_payload(self) -> dict[str, Any]:
        """Return a non-secret, report-safe target summary."""

        return {
            "target_id": self.target_id,
            "phase": self.phase,
            "mode": self.mode,
            "hermes_repo": str(self.hermes_repo),
            "base_ref": self.base_ref,
            "require_clean_worktree": self.require_clean_worktree,
            "target_files": list(self.relative_target_files),
            "source_spec_path": str(self.source_path),
        }


def load_target_spec(path: str | Path, *, hermes_repo_override: str | Path | None = None) -> Phase4TargetSpec:
    """Load and validate a Phase 4 code-evolution target spec from YAML/JSON."""

    spec_path = Path(path)
    raw_text = spec_path.read_text()
    _reject_sensitive_text(raw_text)
    payload = yaml.safe_load(raw_text)
    if not isinstance(payload, Mapping):
        raise TargetContractError("target spec root must be a mapping")
    if "apply_ready" in payload and payload.get("apply_ready") is not False:
        raise TargetContractError("target spec must not approve apply_ready")

    phase = _expect_str(payload, "phase")
    mode = _expect_str(payload, "mode")
    if phase != "4":
        raise TargetContractError('phase must be "4"')
    if mode != "code-evolution-target":
        raise TargetContractError('mode must be "code-evolution-target"')

    target_id = _expect_str(payload, "target_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,127}", target_id):
        raise TargetContractError("target_id must be a safe identifier")

    hermes_base = _expect_mapping(payload, "hermes_base")
    hermes_repo_value = hermes_repo_override if hermes_repo_override is not None else hermes_base.get("repo")
    if not isinstance(hermes_repo_value, (str, Path)):
        raise TargetContractError("hermes_base.repo must be a path string")
    hermes_repo = Path(hermes_repo_value).expanduser().resolve()
    if not hermes_repo.exists() or not hermes_repo.is_dir():
        raise TargetContractError(f"hermes_base.repo must exist and be a directory: {hermes_repo}")
    base_ref = str(hermes_base.get("base_ref", ""))
    if not base_ref:
        raise TargetContractError("hermes_base.base_ref must be non-empty")
    require_clean_worktree = hermes_base.get("require_clean_worktree")
    if require_clean_worktree is not True:
        raise TargetContractError("hermes_base.require_clean_worktree must be true")

    allowed_mutation = _expect_mapping(payload, "allowed_mutation")
    raw_files = allowed_mutation.get("files")
    if not isinstance(raw_files, list) or not all(isinstance(item, str) for item in raw_files):
        raise TargetContractError("allowed_mutation.files must be a list of relative paths")
    if len(raw_files) != 1:
        raise TargetContractError("exactly one target file is allowed for Phase 4 dry-run scaffolding")

    deny_globs = tuple(str(item) for item in allowed_mutation.get("deny_globs", ())) + DEFAULT_DENY_GLOBS
    relative_files: list[str] = []
    target_files: list[Path] = []
    for raw_file in raw_files:
        relative = _normalize_relative_target(raw_file)
        if _matches_denied_path(relative, deny_globs):
            raise TargetContractError(f"denied target path: {relative}")
        target_candidate = hermes_repo / relative
        if target_candidate.is_symlink():
            raise TargetContractError(f"target file must not be a symlink: {relative}")
        target_path = target_candidate.resolve(strict=False)
        if not target_path.is_relative_to(hermes_repo):
            raise TargetContractError(f"target file must stay under Hermes repo: {raw_file}")
        if not target_path.exists() or not target_path.is_file():
            raise TargetContractError(f"target file must exist: {relative}")
        relative_files.append(relative)
        target_files.append(target_path)

    freeze = _expect_mapping(payload, "freeze")
    reproduction = _expect_mapping(payload, "reproduction")
    fitness = _expect_mapping(payload, "fitness")
    benchmarks = _expect_mapping(payload, "benchmarks")
    approvals = _expect_mapping(payload, "approvals")
    _validate_freeze(freeze)
    _validate_reproduction(reproduction)
    _validate_fitness(fitness)
    _validate_benchmarks(benchmarks)
    _validate_approvals(approvals)

    return Phase4TargetSpec(
        source_path=spec_path.resolve(),
        raw=payload,
        phase=phase,
        mode=mode,
        target_id=target_id,
        hermes_repo=hermes_repo,
        base_ref=base_ref,
        require_clean_worktree=True,
        target_files=tuple(target_files),
        relative_target_files=tuple(relative_files),
        deny_globs=tuple(dict.fromkeys(deny_globs)),
        freeze=freeze,
        reproduction=reproduction,
        fitness=fitness,
        benchmarks=benchmarks,
        approvals=approvals,
    )


def _expect_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TargetContractError(f"{key} must be a non-empty string")
    return value


def _expect_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise TargetContractError(f"{key} must be a mapping")
    return value


def _normalize_relative_target(raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        raise TargetContractError(f"target file must be relative: {raw_path}")
    if ".." in path.parts:
        raise TargetContractError(f"target file must not contain traversal segments: {raw_path}")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise TargetContractError("target file must be non-empty")
    return normalized


def _matches_denied_path(relative: str, deny_globs: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(relative, pattern) for pattern in deny_globs)


def _validate_freeze(freeze: Mapping[str, Any]) -> None:
    for key in ("function_signatures", "registry_register_calls", "public_cli_args"):
        if freeze.get(key) is not True:
            raise TargetContractError(f"freeze.{key} must be true")


def _validate_reproduction(reproduction: Mapping[str, Any]) -> None:
    description = reproduction.get("failing_case_description")
    command = reproduction.get("reproducer_command")
    if not isinstance(description, str) or not description.strip():
        raise TargetContractError("reproduction.failing_case_description must be non-empty")
    if not isinstance(command, str) or not command.strip():
        raise TargetContractError("reproduction.reproducer_command must be non-empty")


def _validate_fitness(fitness: Mapping[str, Any]) -> None:
    required_commands = fitness.get("required_commands")
    if not isinstance(required_commands, list) or not required_commands:
        raise TargetContractError("fitness.required_commands must be a non-empty list")
    if not all(isinstance(command, str) and command.strip() for command in required_commands):
        raise TargetContractError("fitness.required_commands must contain non-empty command strings")


def _validate_benchmarks(benchmarks: Mapping[str, Any]) -> None:
    if benchmarks.get("full_benchmark_required_before_acceptance") is not True:
        raise TargetContractError("benchmarks.full_benchmark_required_before_acceptance must be true")
    if benchmarks.get("run_benchmarks_now") is not False:
        raise TargetContractError("benchmarks.run_benchmarks_now must be false")


def _validate_approvals(approvals: Mapping[str, Any]) -> None:
    for key in REQUIRED_APPROVAL_FLAGS:
        if approvals.get(key) is not False:
            raise TargetContractError(f"{key} must be false")
    if approvals.get("budget_approved_usd") != 0:
        raise TargetContractError("budget_approved_usd must be 0")


def _reject_sensitive_text(text: str) -> None:
    for pattern in SENSITIVE_TEXT_PATTERNS:
        if pattern.search(text):
            raise TargetContractError("target spec contains sensitive credential-like text")


def target_spec_to_json(spec: Phase4TargetSpec) -> str:
    """Serialize the report-safe target summary."""

    return json.dumps(spec.to_report_payload(), indent=2, sort_keys=True) + "\n"

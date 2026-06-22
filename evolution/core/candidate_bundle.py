"""Local candidate bundle contract for HSE runs.

The bundle contract is deliberately local-first and candidate-only. It gives
Phase 1-5 runners a shared artifact layout without implying active runtime
mutation, GitHub publication, or deployment approval.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "hse-local-candidate-bundle-v1"
RUNS_ROOT_ENV = "HSE_RUNS_ROOT"
ALLOWED_DECISION_STATUSES = frozenset(
    {
        "PASS_CANDIDATE_ONLY",
        "NO_DIFF_NO_GO",
        "REGRESSION_NO_GO",
        "INCONCLUSIVE",
        "BLOCKED_ENV",
        "NEEDS_HUMAN_SCOPE_REVIEW",
    }
)
_ALLOWED_WRITE_ROOTS = frozenset({"inputs", "candidates", "eval", "reports"})
_SLUG_RE = re.compile(r"[^a-zA-Z0-9.-]+")
_RUN_ID_SLUG_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class CandidateBundle:
    """Resolved paths for one local candidate bundle."""

    root: Path
    phase: str
    target: str
    run_id: str

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def decision_path(self) -> Path:
        return self.root / "decision.json"


def default_runs_root() -> Path:
    """Return the default local HSE run root.

    ``HSE_RUNS_ROOT`` is supported for tests and for operators who want an
    explicit local artifact workspace. The default mirrors EvAH's local-first
    roadmap: ``~/.hermes/evolution/runs``.
    """

    raw = os.environ.get(RUNS_ROOT_ENV, "~/.hermes/evolution/runs")
    return Path(raw).expanduser()


def create_candidate_bundle(
    phase: str,
    target: str,
    *,
    run_id: str | None = None,
    runs_root: str | Path | None = None,
) -> CandidateBundle:
    """Create the standard local candidate bundle directory layout."""

    if not phase or not phase.strip():
        raise ValueError("candidate bundle phase must be non-empty")
    if not target or not target.strip():
        raise ValueError("candidate bundle target must be non-empty")

    resolved_run_id = _safe_run_id(run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    phase_slug = _safe_slug(phase)
    target_slug = _safe_slug(target)
    root_base = Path(runs_root).expanduser() if runs_root is not None else default_runs_root()
    _reject_symlink(root_base)
    bundle_root = root_base / f"{resolved_run_id}-{phase_slug}-{target_slug}"
    _reject_symlink(bundle_root)

    bundle = CandidateBundle(root=bundle_root, phase=phase, target=target, run_id=resolved_run_id)
    for directory in (bundle.inputs_dir, bundle.candidates_dir, bundle.eval_dir, bundle.reports_dir):
        _reject_symlink(directory)
        directory.mkdir(parents=True, exist_ok=False)

    write_bundle_json(
        bundle,
        "inputs/target_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "target": target,
            "candidate_only": True,
            "apply_ready": False,
        },
    )
    return bundle


def write_bundle_json(bundle: CandidateBundle, relative_path: str | Path, payload: Mapping[str, Any] | list[Any]) -> Path:
    """Write JSON inside one of the standard bundle subdirectories."""

    path = _resolve_bundle_write_path(bundle, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def write_bundle_text(bundle: CandidateBundle, relative_path: str | Path, content: str) -> Path:
    """Write text inside one of the standard bundle subdirectories."""

    path = _resolve_bundle_write_path(bundle, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def write_decision(
    bundle: CandidateBundle,
    *,
    status: str,
    summary: str,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Write the root ``decision.json`` for a candidate-only run."""

    if status not in ALLOWED_DECISION_STATUSES:
        raise ValueError(f"unknown candidate bundle decision status: {status}")
    if not summary or not summary.strip():
        raise ValueError("candidate bundle decision summary must be non-empty")

    decision = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "summary": summary.strip(),
        "phase": bundle.phase,
        "target": bundle.target,
        "run_id": bundle.run_id,
        "generated_at": generated_at
        or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "candidate_only": True,
        "apply_ready": False,
        "github": {
            "pr_created": False,
            "push_performed": False,
            "merge_performed": False,
            "publication_deferred": True,
        },
        "safety_invariants": {
            "active_runtime_mutation": False,
            "active_skill_modified": False,
            "active_tool_schema_modified": False,
            "active_prompt_modified": False,
            "credentials_accessed": False,
            "external_publication_performed": False,
            "deployment_performed": False,
        },
        "metrics": dict(metrics or {}),
        "artifacts": dict(artifacts or {}),
    }
    bundle.decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    return decision


def _resolve_bundle_write_path(bundle: CandidateBundle, relative_path: str | Path) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("bundle writes must target inputs/, candidates/, eval/, or reports/")
    if not candidate.parts or candidate.parts[0] not in _ALLOWED_WRITE_ROOTS:
        raise ValueError("bundle writes must target inputs/, candidates/, eval/, or reports/")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("bundle writes must target inputs/, candidates/, eval/, or reports/")

    root = bundle.root.resolve()
    path = (bundle.root / candidate).resolve()
    if not path.is_relative_to(root):
        raise ValueError("bundle writes must target inputs/, candidates/, eval/, or reports/")
    return path


def _safe_slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.strip().lower()).strip("-._")
    return slug or "run"


def _safe_run_id(value: str) -> str:
    slug = _RUN_ID_SLUG_RE.sub("-", value.strip().lower()).strip("-._")
    return slug or "run"


def _reject_symlink(path: Path) -> None:
    if path.exists() and path.is_symlink():
        raise ValueError(f"candidate bundle path must not be a symlink: {path}")

"""Local-completion scope guards for HSE candidate-only gates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

LOCAL_COMPLETION_SCHEMA_VERSION = "hse-local-completion-v1"
GITHUB_LANE_DEFERRED = True

GITHUB_POLICY = {
    "queried": False,
    "pr_created": False,
    "push_performed": False,
    "merge_performed": False,
    "publication_deferred": True,
}

SAFETY_INVARIANTS = {
    "active_runtime_mutation": False,
    "active_skill_modified": False,
    "active_tool_schema_modified": False,
    "active_prompt_modified": False,
    "cron_jobs_created": False,
    "gateway_restart_or_reload": False,
    "credentials_accessed": False,
    "network_calls_performed": False,
    "external_publication_performed": False,
    "deployment_performed": False,
}

_FORBIDDEN_TOP_LEVEL_TRUE_KEYS = frozenset({"apply_ready", "active_apply", "github_last_lane_allowed"})
_FORBIDDEN_GITHUB_TRUE_KEYS = frozenset({"queried", "pr_created", "push_performed", "merge_performed"})
_FORBIDDEN_SAFETY_TRUE_KEYS = frozenset(SAFETY_INVARIANTS)


def base_decision_payload(*, gate_id: str, phase: str, target: str, generated_at: str) -> dict[str, Any]:
    """Return a fail-closed candidate-only decision skeleton.

    The skeleton is intentionally stricter than the older local candidate
    bundle contract: GitHub state must not even be queried in this lane, and
    restart/reload/cron mutation are explicitly represented as false.
    """

    _require_non_empty("gate_id", gate_id)
    _require_non_empty("phase", phase)
    _require_non_empty("target", target)
    _require_non_empty("generated_at", generated_at)
    return {
        "schema_version": LOCAL_COMPLETION_SCHEMA_VERSION,
        "gate_id": gate_id.strip(),
        "phase": phase.strip(),
        "target": target.strip(),
        "generated_at": generated_at.strip(),
        "candidate_only": True,
        "apply_ready": False,
        "github": deepcopy(GITHUB_POLICY),
        "safety_invariants": deepcopy(SAFETY_INVARIANTS),
    }


def reject_github_or_active_apply_flags(payload: Mapping[str, Any]) -> None:
    """Raise if a local-completion payload claims forbidden side effects."""

    violations: list[str] = []
    for key in _FORBIDDEN_TOP_LEVEL_TRUE_KEYS:
        if payload.get(key) is True:
            violations.append(key)
    if payload.get("candidate_only") is False:
        violations.append("candidate_only=false")

    github = payload.get("github")
    if isinstance(github, Mapping):
        for key in _FORBIDDEN_GITHUB_TRUE_KEYS:
            if github.get(key) is True:
                violations.append(f"github.{key}")

    safety = payload.get("safety_invariants")
    if isinstance(safety, Mapping):
        for key in _FORBIDDEN_SAFETY_TRUE_KEYS:
            if safety.get(key) is True:
                violations.append(f"safety_invariants.{key}")

    if violations:
        joined = ", ".join(sorted(violations))
        raise ValueError(f"local completion scope violation: {joined}")


def _require_non_empty(field: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")

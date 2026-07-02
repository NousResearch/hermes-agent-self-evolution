"""LC2 active skill local canary packet writer.

This module reads active and rebuild-candidate skill files, writes a local
candidate bundle, and records a canary decision. It never applies skill changes,
queries GitHub, mutates cron, restarts/reloads the gateway, or changes active
Hermes runtime files.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from evolution.core.candidate_bundle import create_candidate_bundle, write_bundle_json, write_bundle_text
from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags

LC2_GATE_ID = "LC2"
LC2_ACTIVE_SKILL_CANARY_PHASE = "Phase 1: Active Skill Local Canary"
LC2_TARGET = "github-code-review"
PASS_PROVISIONAL_BASELINE_CANARY = "PASS_PROVISIONAL_BASELINE_CANARY"
NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
BLOCKED = "BLOCKED"

_HIGH_RISK_ASSIGNMENT_RE = re.compile(
    r"(?im)^\s*(?:[A-Z0-9_]*(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE[_-]?KEY|CREDENTIAL)[A-Z0-9_]*)\s*[:=]\s*[\"']?[^\s\"']{12,}"
)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def scan_skill_text(text: str) -> dict[str, Any]:
    """Return a deterministic, machine-readable scan of a skill text."""

    if not isinstance(text, str):
        raise TypeError("skill text must be a string")
    headings = [match.group(1).strip() for line in text.splitlines() if (match := _HEADING_RE.match(line))]
    high_risk_hits = [match.group(0).strip() for match in _HIGH_RISK_ASSIGNMENT_RE.finditer(text)]
    sha256 = hashlib.sha256(text.encode()).hexdigest()
    external_write_gate_present = any(heading.lower() == "external write gate" for heading in headings)
    return {
        "sha256": sha256,
        "bytes": len(text.encode()),
        "line_count": len(text.splitlines()),
        "headings": headings,
        "external_write_gate_present": external_write_gate_present,
        "high_risk_assignment_like_hits": len(high_risk_hits),
        "scan_clean": not high_risk_hits,
    }


def write_active_skill_canary_packet(
    *,
    active_skill_path: str | Path,
    candidate_skill_path: str | Path,
    runs_root: str | Path,
    run_id: str,
    generated_at: str,
    expected_active_sha: str | None = None,
) -> dict[str, str]:
    """Write a fresh LC2 local canary bundle and return its key paths."""

    active_path = Path(active_skill_path).expanduser()
    candidate_path = Path(candidate_skill_path).expanduser()
    active_text = _read_skill_text(active_path, "active skill")
    candidate_text = _read_skill_text(candidate_path, "rebuild candidate skill")

    active_scan = scan_skill_text(active_text)
    candidate_scan = scan_skill_text(candidate_text)
    if expected_active_sha and active_scan["sha256"] != expected_active_sha:
        raise ValueError(
            "active skill SHA mismatch: "
            f"expected {expected_active_sha}, observed {active_scan['sha256']}"
        )

    bundle = create_candidate_bundle(
        phase=LC2_ACTIVE_SKILL_CANARY_PHASE,
        target=LC2_TARGET,
        run_id=run_id,
        runs_root=runs_root,
    )

    diff_text = _unified_diff(
        active_text,
        candidate_text,
        fromfile="active_skill.md",
        tofile="rebuild_candidate_skill.md",
    )
    active_only_headings = _difference(active_scan["headings"], candidate_scan["headings"])
    candidate_only_headings = _difference(candidate_scan["headings"], active_scan["headings"])
    blockers = _canary_blockers(active_scan, candidate_scan)
    sha_match = active_scan["sha256"] == candidate_scan["sha256"]
    comparison_status = "NO_DRIFT" if sha_match else "HASH_DRIFT_REVIEW_REQUIRED"
    status = BLOCKED if blockers else PASS_PROVISIONAL_BASELINE_CANARY

    comparison = {
        "sha_match": sha_match,
        "status": comparison_status,
        "active_only_headings": active_only_headings,
        "candidate_only_headings": candidate_only_headings,
        "similarity_ratio": round(_similarity_ratio(active_text, candidate_text), 4),
    }

    decision = base_decision_payload(
        gate_id=LC2_GATE_ID,
        phase=LC2_ACTIVE_SKILL_CANARY_PHASE,
        target=LC2_TARGET,
        generated_at=generated_at,
    )
    decision.update(
        {
            "status": status,
            "summary": _decision_summary(status, comparison_status),
            "active_skill": _skill_summary(active_scan, label="active-default-profile-skill"),
            "rebuild_candidate_skill": _skill_summary(candidate_scan, label="phase1-rebuild-candidate-skill"),
            "canary": {
                "external_write_gate_present": active_scan["external_write_gate_present"],
                "scan_clean": active_scan["scan_clean"] and candidate_scan["scan_clean"],
                "blockers": blockers,
                "current_hash_decision": "ACCEPT_AS_PROVISIONAL_SAFETY_HARDENED_BASELINE_FOR_READ_ONLY_USE_ONLY"
                if not blockers
                else "BLOCKED_PENDING_REVIEW",
                "required_next": "human approval required before any active skill apply or candidate replacement",
            },
            "candidate_comparison": comparison,
            "artifacts": {
                "active_snapshot": "inputs/active_skill.md",
                "candidate_snapshot": "inputs/rebuild_candidate_skill.md",
                "comparison": "eval/skill_canary_comparison.json",
                "patch": "candidates/active_vs_candidate.patch",
                "report": "reports/canary_report.md",
                "rollback": "reports/rollback.md",
            },
        }
    )
    reject_github_or_active_apply_flags(decision)

    write_bundle_text(bundle, "inputs/active_skill.md", active_text)
    write_bundle_text(bundle, "inputs/rebuild_candidate_skill.md", candidate_text)
    write_bundle_text(bundle, "candidates/active_vs_candidate.patch", diff_text)
    write_bundle_json(
        bundle,
        "eval/skill_canary_comparison.json",
        {
            "active_skill": decision["active_skill"],
            "rebuild_candidate_skill": decision["rebuild_candidate_skill"],
            "canary": decision["canary"],
            "candidate_comparison": comparison,
        },
    )
    write_bundle_text(bundle, "reports/canary_report.md", _render_canary_report(decision))
    write_bundle_text(bundle, "reports/rollback.md", _render_rollback_note())
    bundle.decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    return {
        "bundle_root": str(bundle.root),
        "decision_path": str(bundle.decision_path),
        "report_path": str(bundle.reports_dir / "canary_report.md"),
    }


def _read_skill_text(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} path is not a file: {path}")
    return path.read_text()


def _canary_blockers(active_scan: dict[str, Any], candidate_scan: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not active_scan["external_write_gate_present"]:
        blockers.append("missing_external_write_gate")
    if active_scan["high_risk_assignment_like_hits"]:
        blockers.append("active_skill_high_risk_assignment_like_hit")
    if candidate_scan["high_risk_assignment_like_hits"]:
        blockers.append("candidate_skill_high_risk_assignment_like_hit")
    return blockers


def _skill_summary(scan: dict[str, Any], *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "sha256": scan["sha256"],
        "bytes": scan["bytes"],
        "line_count": scan["line_count"],
        "heading_count": len(scan["headings"]),
        "external_write_gate_present": scan["external_write_gate_present"],
        "high_risk_assignment_like_hits": scan["high_risk_assignment_like_hits"],
        "scan_clean": scan["scan_clean"],
    }


def _difference(left: list[str], right: list[str]) -> list[str]:
    right_set = set(right)
    return [item for item in left if item not in right_set]


def _similarity_ratio(left: str, right: str) -> float:
    return difflib.SequenceMatcher(a=left, b=right, autojunk=False).ratio()


def _unified_diff(left: str, right: str, *, fromfile: str, tofile: str) -> str:
    if left == right:
        return "No active/candidate skill differences.\n"
    return "".join(
        difflib.unified_diff(
            left.splitlines(keepends=True),
            right.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def _decision_summary(status: str, comparison_status: str) -> str:
    if status == BLOCKED:
        return "LC2 active skill canary blocked; no active skill changes were made."
    return (
        "LC2 active skill canary passed for the current active skill hash as a provisional "
        f"read-only baseline; candidate comparison status is {comparison_status}."
    )


def _render_canary_report(decision: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# LC2 Active Skill Local Canary Packet",
            "",
            f"Status: `{decision['status']}`",
            "",
            "## Boundary",
            "",
            "- candidate_only: `true`",
            "- apply_ready: `false`",
            "- GitHub/PR work: `deferred_not_queried`",
            "- Active skill mutation: `false`",
            "- Gateway restart/reload: `false`",
            "",
            "## Active Skill",
            "",
            f"- SHA-256: `{decision['active_skill']['sha256']}`",
            f"- External Write Gate present: `{str(decision['canary']['external_write_gate_present']).lower()}`",
            f"- High-risk assignment-like hits: `{decision['active_skill']['high_risk_assignment_like_hits']}`",
            "",
            "## Candidate Comparison",
            "",
            f"- Status: `{decision['candidate_comparison']['status']}`",
            f"- SHA match: `{str(decision['candidate_comparison']['sha_match']).lower()}`",
            f"- Similarity ratio: `{decision['candidate_comparison']['similarity_ratio']}`",
            "",
            "## Required Next Boundary",
            "",
            "No active apply, candidate replacement, GitHub/PR work, restart, reload, or deployment is authorized by this packet.",
            "",
        ]
    )


def _render_rollback_note() -> str:
    return "\n".join(
        [
            "# LC2 Rollback Note",
            "",
            "This packet made no active apply and did not modify the active skill file.",
            "",
            "Rollback requirement: none for active runtime state, because this is a candidate-only local canary artifact.",
            "",
            "If a future separately approved active skill apply occurs, create a backup and checksum before mutation.",
            "",
        ]
    )

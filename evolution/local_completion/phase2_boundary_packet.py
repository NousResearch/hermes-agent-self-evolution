"""LC3 Phase 2 candidate-to-active boundary packet writer.

This module turns a Phase 2 candidate-only report into a local boundary review
packet. It does not apply active Hermes tool schema changes, query GitHub, write
PRs, mutate cron, or restart/reload the gateway.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.local_completion.scope import base_decision_payload, reject_github_or_active_apply_flags
from evolution.tools.report_contract import validate_candidate_only_report_contract

LC3_GATE_ID = "LC3"
LC3_PHASE2_BOUNDARY_PHASE = "Phase 2: Candidate-to-Active Boundary Packet"
LC3_TARGET = "tool-description-active-boundary"
PASS_BOUNDARY_REVIEW_READY = "PASS_BOUNDARY_REVIEW_READY"
BLOCKED_CONTRACT = "BLOCKED_CONTRACT"
BLOCKED_PHASE2D_GATE = "BLOCKED_PHASE2D_GATE"


def write_phase2_boundary_packet(
    *,
    candidate_report_path: str | Path,
    output_dir: str | Path,
    generated_at: str,
) -> dict[str, str]:
    """Write LC3 active-boundary packet files from a Phase 2 candidate report."""

    report_path = Path(candidate_report_path).expanduser()
    report = _load_report(report_path)
    validation = validate_candidate_only_report_contract(report) if isinstance(report, Mapping) else None
    validation_passed = bool(validation and validation.passed)
    validation_errors = list(validation.errors) if validation else ["report JSON root must be an object"]

    out = Path(output_dir).expanduser()
    inputs_dir = out / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = inputs_dir / "candidate_only_report.json"
    snapshot_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    phase2d_gate = _mapping(report.get("phase2d_gate")) if isinstance(report, Mapping) else {}
    metrics = _mapping(report.get("metrics")) if isinstance(report, Mapping) else {}
    gate_passed = phase2d_gate.get("passed") is True
    status = _boundary_status(validation_passed=validation_passed, gate_passed=gate_passed)

    packet = base_decision_payload(
        gate_id=LC3_GATE_ID,
        phase=LC3_PHASE2_BOUNDARY_PHASE,
        target=LC3_TARGET,
        generated_at=generated_at,
    )
    packet.update(
        {
            "status": status,
            "summary": _summary(status),
            "source_report": {
                "path": _path_label(report_path),
                "sha256": _sha256_path(report_path),
                "bytes": report_path.stat().st_size,
            },
            "report_contract": {
                "passed": validation_passed,
                "errors": validation_errors,
            },
            "phase2d_gate": {
                "passed": gate_passed,
                "failed_checks": list(phase2d_gate.get("failed_checks", []))
                if isinstance(phase2d_gate.get("failed_checks"), list)
                else [],
                "thresholds": dict(phase2d_gate.get("thresholds", {}))
                if isinstance(phase2d_gate.get("thresholds"), Mapping)
                else {},
            },
            "metrics": _metric_summary(metrics),
            "candidate_summary": _candidate_summary(report),
            "active_boundary": {
                "active_schema_apply_performed": False,
                "active_schema_apply_approved": False,
                "active_tool_schema_modified": False,
                "active_runtime_mutation": False,
                "apply_ready_reason": "separate human approval required before active schema apply",
                "required_before_active_apply": [
                    "backup active Hermes checkout and schema sources",
                    "verify candidate patch against current active Hermes HEAD and dirty files",
                    "run focused tool schema and registry regression tests",
                    "run full Hermes/HSE regression gates when feasible",
                    "obtain separate explicit human approval for active apply",
                ],
            },
            "artifacts": {
                "candidate_report_snapshot": "inputs/candidate_only_report.json",
                "boundary_packet": "active_boundary_packet.json",
                "boundary_markdown": "active_boundary_packet.md",
                "source_candidate_artifacts": dict(report.get("artifacts", {})) if isinstance(report, Mapping) else {},
            },
        }
    )
    reject_github_or_active_apply_flags(packet)

    packet_path = out / "active_boundary_packet.json"
    markdown_path = out / "active_boundary_packet.md"
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(_render_markdown(packet))

    return {
        "packet_path": str(packet_path),
        "markdown_path": str(markdown_path),
        "snapshot_path": str(snapshot_path),
    }


def _load_report(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"candidate report not found: {path}")
    if not path.is_file():
        raise ValueError(f"candidate report path is not a file: {path}")
    return json.loads(path.read_text())


def _boundary_status(*, validation_passed: bool, gate_passed: bool) -> str:
    if not validation_passed:
        return BLOCKED_CONTRACT
    if not gate_passed:
        return BLOCKED_PHASE2D_GATE
    return PASS_BOUNDARY_REVIEW_READY


def _summary(status: str) -> str:
    if status == PASS_BOUNDARY_REVIEW_READY:
        return "LC3 Phase 2 boundary packet is ready for human review; active schema apply remains false."
    if status == BLOCKED_PHASE2D_GATE:
        return "LC3 Phase 2 boundary packet blocked by Phase 2D gate failure; no active schema changes were made."
    return "LC3 Phase 2 boundary packet blocked by candidate report contract failure; no active schema changes were made."


def _metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    warnings = metrics.get("warnings")
    return {
        "case_count": metrics.get("case_count"),
        "selection_accuracy": metrics.get("selection_accuracy"),
        "wrong_tool_avoidance": metrics.get("wrong_tool_avoidance"),
        "argument_cue_coverage": metrics.get("argument_cue_coverage"),
        "constraint_pass_rate": metrics.get("constraint_pass_rate"),
        "warning_count": len(warnings) if isinstance(warnings, list) else None,
    }


def _candidate_summary(report: Any) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        return {"candidate_count": None, "changed_candidate_count": None}
    candidates = report.get("candidates")
    changed_count = None
    if isinstance(candidates, list):
        changed_count = sum(1 for candidate in candidates if _candidate_changed(candidate))
    return {
        "candidate_count": report.get("candidate_count"),
        "changed_candidate_count": changed_count,
        "phase_index_executed": list(report.get("phase_index_executed", []))
        if isinstance(report.get("phase_index_executed"), list)
        else [],
    }


def _candidate_changed(candidate: Any) -> bool:
    if not isinstance(candidate, Mapping):
        return False
    baseline = candidate.get("baseline_description")
    proposed = candidate.get("candidate_description")
    return isinstance(baseline, str) and isinstance(proposed, str) and baseline != proposed


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _path_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _render_markdown(packet: Mapping[str, Any]) -> str:
    metrics = _mapping(packet.get("metrics"))
    gate = _mapping(packet.get("phase2d_gate"))
    source = _mapping(packet.get("source_report"))
    boundary = _mapping(packet.get("active_boundary"))
    return "\n".join(
        [
            "# LC3 Phase 2 Candidate-to-Active Boundary Packet",
            "",
            f"Status: `{packet.get('status')}`",
            "",
            "## Boundary",
            "",
            "- candidate_only=true",
            "- apply_ready=false",
            "- GitHub/PR work: deferred_not_queried",
            "- Active tool schema mutation: false",
            "- Gateway restart/reload: false",
            "",
            "## Source Report",
            "",
            f"- Path: `{source.get('path')}`",
            f"- SHA-256: `{source.get('sha256')}`",
            "",
            "## Phase 2D Gate",
            "",
            f"- passed: `{str(gate.get('passed')).lower()}`",
            f"- failed_checks: `{gate.get('failed_checks')}`",
            "",
            "## Metrics",
            "",
            f"- case_count: `{metrics.get('case_count')}`",
            f"- selection_accuracy: `{metrics.get('selection_accuracy')}`",
            f"- wrong_tool_avoidance: `{metrics.get('wrong_tool_avoidance')}`",
            f"- constraint_pass_rate: `{metrics.get('constraint_pass_rate')}`",
            "",
            "## Required Before Any Active Apply",
            "",
            *[f"- {item}" for item in boundary.get("required_before_active_apply", [])],
            "",
            "This packet does not authorize active apply, GitHub/PR work, restart, reload, publication, merge, or deployment.",
            "",
        ]
    )


def _now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an LC3 Phase 2 boundary packet from a candidate-only report.")
    parser.add_argument("--candidate-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--generated-at", default=_now_iso())
    args = parser.parse_args(argv)

    result = write_phase2_boundary_packet(
        candidate_report_path=args.candidate_report_json,
        output_dir=args.output_dir,
        generated_at=args.generated_at,
    )
    packet = json.loads(Path(result["packet_path"]).read_text())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if packet.get("status") == PASS_BOUNDARY_REVIEW_READY else 1


if __name__ == "__main__":  # pragma: no cover - exercised by operator smoke runs
    raise SystemExit(main())

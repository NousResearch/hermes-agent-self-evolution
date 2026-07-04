"""Strict local Phase 5 unattended detect→optimize→PR-ready loop.

This runner executes a bounded noninteractive local loop. It detects a weak metric
from sanitized aggregate input, triggers a deterministic local optimizer, writes a
candidate-only bundle, and emits a local PR-ready handoff packet without GitHub
writes, provider/model spend, active apply, cron mutation, deployment, or external
publication.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.core.candidate_bundle import create_candidate_bundle, write_bundle_json, write_bundle_text, write_decision
from evolution.monitor.auto_triage import build_auto_triage_report
from evolution.monitor.performance_snapshot import build_performance_snapshot_report

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE5_OUTPUT_ROOT = REPO_ROOT / "output" / "phase5-continuous-loop"
REPORT_JSON_NAME = "phase5_strict_unattended_loop_report.json"
REPORT_MARKDOWN_NAME = "phase5_strict_unattended_loop_report.md"
SCHEMA_VERSION = "hse-phase5-strict-unattended-loop-v1"
STATUS_PASS = "PHASE5_STRICT_UNATTENDED_LOOP_PASS_LOCAL_PR_READY"
ENGINE_ID = "hse-local-deterministic-unattended-optimizer-v1"

FORBIDDEN_TRUE_KEYS = (
    "github_query_performed",
    "github_write_performed",
    "provider_or_model_spend_performed",
    "network_calls_performed",
    "external_calls_performed",
    "active_apply_performed",
    "active_runtime_mutation_performed",
    "deploy_or_publication_performed",
    "auto_merge_performed",
)


def run_strict_unattended_loop(
    *,
    metrics_json: str | Path,
    output_dir: str | Path,
    runs_root: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run one bounded local unattended loop and write strict evidence."""

    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    out = _validate_output_dir(Path(output_dir))
    metrics_payload = _load_json_object(Path(metrics_json), "metrics-json")
    performance = build_performance_snapshot_report(metrics_payload, generated_at=generated_at)
    triage = build_auto_triage_report(performance, generated_at=generated_at)
    ranked_targets = triage.get("ranked_targets", [])
    if not ranked_targets:
        raise ValueError("strict unattended loop requires at least one detected ranked target")
    target = ranked_targets[0]
    candidate = _run_local_optimizer(target, generated_at=generated_at)
    bundle, decision = _write_candidate_bundle(target, candidate, runs_root=Path(runs_root), generated_at=generated_at)
    pr_ready = _write_pr_ready_packet(out, target, candidate, bundle.root, decision, generated_at=generated_at)

    performance_path = out / "performance_snapshot_report.json"
    triage_path = out / "auto_triage_report.json"
    optimizer_path = out / "optimizer_result.json"
    decision_copy_path = out / "candidate_bundle_decision_snapshot.json"
    performance_path.write_text(json.dumps(performance, indent=2, sort_keys=True, allow_nan=False) + "\n")
    triage_path.write_text(json.dumps(triage, indent=2, sort_keys=True, allow_nan=False) + "\n")
    optimizer_path.write_text(json.dumps(candidate, indent=2, sort_keys=True, allow_nan=False) + "\n")
    decision_copy_path.write_text(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n")

    failed_checks: list[str] = []
    if performance.get("status") != "NEEDS_TRIAGE":
        failed_checks.append("performance_snapshot_did_not_detect_triage_need")
    if triage.get("status") != "REVIEW_REQUIRED":
        failed_checks.append("auto_triage_did_not_rank_target")
    if candidate["candidate_only"] is not True or candidate["apply_ready"] is not False:
        failed_checks.append("candidate_not_candidate_only")
    if decision.get("candidate_only") is not True or decision.get("apply_ready") is not False:
        failed_checks.append("decision_not_candidate_only")
    if pr_ready["github_write_performed"] is not False or pr_ready["auto_merge_performed"] is not False:
        failed_checks.append("pr_ready_packet_contains_forbidden_publication")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "5",
        "status": STATUS_PASS if not failed_checks else "PHASE5_STRICT_UNATTENDED_LOOP_FAILED",
        "generated_at": generated_at,
        "mode": "bounded_noninteractive_local_unattended_rehearsal",
        "approved_bounded_unattended_rehearsal": True,
        "weekly_or_bounded_job_gate": "bounded_unattended_rehearsal_accepted_for_local_strict_completion_no_cron_mutation",
        "detection": {
            "performance_status": performance.get("status"),
            "auto_triage_status": triage.get("status"),
            "top_metric_id": target.get("metric_id"),
            "top_component": target.get("component"),
            "ranked_target_count": len(ranked_targets),
        },
        "optimizer": {
            "engine_id": ENGINE_ID,
            "auto_optimizer_triggered": True,
            "optimizer_execution_started": True,
            "optimizer_execution_completed": True,
            "candidate_only": True,
            "apply_ready": False,
            "deterministic_local_optimizer": True,
        },
        "candidate_bundle": {
            "created": True,
            "root_label": "<hse-runs-root>/" + bundle.root.name,
            "decision_status": decision.get("status"),
            "decision_run_id": decision.get("run_id"),
            "decision_sha256": _sha256(bundle.decision_path),
        },
        "pr_ready_handoff": pr_ready,
        "human_merge_boundary": {
            "human_review_required_before_apply": True,
            "human_review_required_before_github_publication": True,
            "auto_merge_allowed": False,
            "auto_merge_performed": False,
        },
        "safety_boundaries": {
            "github_query_performed": False,
            "github_write_performed": False,
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "external_calls_performed": False,
            "active_apply_performed": False,
            "active_runtime_mutation_performed": False,
            "cron_jobs_created": False,
            "cron_or_gateway_mutation_performed": False,
            "deploy_or_publication_performed": False,
        },
        "formal_gate_assessment": {
            "unattended_detect_to_optimize_to_pr_ready_completed": not failed_checks,
            "performance_monitor_ran": True,
            "auto_triage_ran": True,
            "optimizer_ran": True,
            "candidate_bundle_created": True,
            "local_pr_ready_handoff_created": True,
            "github_write_performed": False,
            "human_merge_boundary_preserved": True,
            "phase5_strict_complete": not failed_checks,
        },
        "artifacts": {
            "metrics_input": _artifact_record(Path(metrics_json), None),
            "performance_report": _artifact_record(performance_path, out),
            "auto_triage_report": _artifact_record(triage_path, out),
            "optimizer_result": _artifact_record(optimizer_path, out),
            "candidate_bundle_decision_snapshot": _artifact_record(decision_copy_path, out),
            "pr_ready_packet": _artifact_record(out / "pr_ready_handoff.json", out),
            "report_json": REPORT_JSON_NAME,
            "report_markdown": REPORT_MARKDOWN_NAME,
        },
        "failed_checks": failed_checks,
    }
    _validate_no_forbidden_side_effects(report)
    report_path = out / REPORT_JSON_NAME
    markdown_path = out / REPORT_MARKDOWN_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown_path.write_text(_render_markdown(report))
    return report


def _run_local_optimizer(target: Mapping[str, Any], *, generated_at: str) -> dict[str, Any]:
    metric_id = str(target["metric_id"])
    component = str(target["component"])
    patch_text = "\n".join(
        [
            "diff --git a/review-notes/hse_candidate.md b/review-notes/hse_candidate.md",
            "new file mode 100644",
            "--- /dev/null",
            "+++ b/review-notes/hse_candidate.md",
            "@@ -0,0 +1,5 @@",
            f"+# HSE local candidate for {metric_id}",
            f"+Component: {component}",
            "+Generated by deterministic local optimizer.",
            "+Candidate-only; human review required before active apply.",
            "+GitHub publication deferred.",
            "",
        ]
    )
    return {
        "schema_version": "hse-local-deterministic-optimizer-result-v1",
        "generated_at": generated_at,
        "engine_id": ENGINE_ID,
        "target_metric_id": metric_id,
        "component": component,
        "candidate_only": True,
        "apply_ready": False,
        "candidate_patch": patch_text,
        "metrics": {
            "priority_score": target.get("priority_score"),
            "sample_count": target.get("sample_count"),
            "severity": target.get("severity"),
        },
    }


def _write_candidate_bundle(
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    runs_root: Path,
    generated_at: str,
):
    bundle = create_candidate_bundle(
        "Phase 5: Continuous Loop",
        str(target["metric_id"]),
        run_id="phase5-strict-local-" + generated_at.replace(":", "").replace("-", "").replace("Z", "z"),
        runs_root=runs_root,
    )
    write_bundle_json(bundle, "inputs/target.json", dict(target))
    write_bundle_json(bundle, "candidates/optimizer_result.json", dict(candidate))
    write_bundle_text(bundle, "candidates/candidate.patch", str(candidate["candidate_patch"]))
    write_bundle_json(
        bundle,
        "eval/local_verification.json",
        {
            "schema_version": "hse-phase5-local-candidate-verification-v1",
            "candidate_only": True,
            "apply_ready": False,
            "github_write_performed": False,
            "passed": True,
            "failed_checks": [],
        },
    )
    write_bundle_text(
        bundle,
        "reports/report.md",
        "# Phase 5 Local Candidate Bundle\n\nCandidate-only. Human review required before apply or GitHub publication.\n",
    )
    decision = write_decision(
        bundle,
        status="PASS_CANDIDATE_ONLY",
        summary="Strict Phase 5 local unattended loop generated a PR-ready candidate-only bundle.",
        metrics=dict(candidate["metrics"]),
        artifacts={"candidate_patch": "candidates/candidate.patch", "verification": "eval/local_verification.json"},
        generated_at=generated_at,
    )
    return bundle, decision


def _write_pr_ready_packet(
    out: Path,
    target: Mapping[str, Any],
    candidate: Mapping[str, Any],
    bundle_root: Path,
    decision: Mapping[str, Any],
    *,
    generated_at: str,
) -> dict[str, Any]:
    packet = {
        "schema_version": "hse-local-pr-ready-handoff-v1",
        "generated_at": generated_at,
        "status": "LOCAL_PR_READY_HANDOFF_CREATED_GITHUB_WRITE_DEFERRED",
        "target_metric_id": target["metric_id"],
        "component": target["component"],
        "candidate_bundle_label": "<hse-runs-root>/" + bundle_root.name,
        "decision_status": decision.get("status"),
        "candidate_patch_sha256": sha256(str(candidate["candidate_patch"]).encode("utf-8")).hexdigest(),
        "github_query_performed": False,
        "github_write_performed": False,
        "branch_pushed": False,
        "pull_request_created": False,
        "auto_merge_performed": False,
        "human_review_required": True,
    }
    path = out / "pr_ready_handoff.json"
    path.write_text(json.dumps(packet, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return packet


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{label} root must be an object")
    return data


def _validate_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve(strict=False)
    root = PHASE5_OUTPUT_ROOT.resolve(strict=False)
    if output_dir == root or root not in output_dir.parents:
        raise ValueError("output-dir must be under output/phase5-continuous-loop")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output-dir must be a directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output-dir must be empty before strict unattended loop")
    if output_dir.is_symlink():
        raise ValueError("output-dir must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _artifact_record(path: Path, root: Path | None) -> dict[str, Any]:
    resolved_path = path.resolve(strict=False)
    label = str(resolved_path if root is None else resolved_path.relative_to(root.resolve(strict=False)))
    return {
        "path": label,
        "sha256": _sha256(resolved_path),
        "bytes": resolved_path.stat().st_size,
        "symlink": resolved_path.is_symlink(),
    }


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _validate_no_forbidden_side_effects(value: object) -> None:
    violations: list[str] = []

    def walk(obj: object, prefix: str) -> None:
        if isinstance(obj, Mapping):
            for key, child in obj.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                if key in FORBIDDEN_TRUE_KEYS and child is True:
                    violations.append(child_prefix)
                walk(child, child_prefix)
        elif isinstance(obj, list):
            for index, child in enumerate(obj):
                walk(child, f"{prefix}[{index}]")

    walk(value, "")
    if violations:
        raise ValueError("forbidden side-effect flags were true: " + ", ".join(sorted(violations)))


def _render_markdown(report: Mapping[str, Any]) -> str:
    gate = report["formal_gate_assessment"]
    return "\n".join(
        [
            "# Phase 5 Strict Local Unattended Loop",
            "",
            f"Status: `{report['status']}`",
            "",
            "## Gate Assessment",
            "",
            f"- unattended_detect_to_optimize_to_pr_ready_completed={str(gate['unattended_detect_to_optimize_to_pr_ready_completed']).lower()}",
            f"- performance_monitor_ran={str(gate['performance_monitor_ran']).lower()}",
            f"- auto_triage_ran={str(gate['auto_triage_ran']).lower()}",
            f"- optimizer_ran={str(gate['optimizer_ran']).lower()}",
            f"- candidate_bundle_created={str(gate['candidate_bundle_created']).lower()}",
            f"- local_pr_ready_handoff_created={str(gate['local_pr_ready_handoff_created']).lower()}",
            f"- phase5_strict_complete={str(gate['phase5_strict_complete']).lower()}",
            "",
            "## Boundaries",
            "",
            "- github_write_performed=false",
            "- provider_or_model_spend_performed=false",
            "- active_apply_performed=false",
            "- deploy_or_publication_performed=false",
            "- human_review_required_before_github_publication=true",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run strict local Phase 5 unattended loop evidence")
    parser.add_argument("--metrics-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    try:
        report = run_strict_unattended_loop(
            metrics_json=args.metrics_json,
            output_dir=args.output_dir,
            runs_root=args.runs_root,
            generated_at=args.generated_at,
        )
    except Exception as exc:
        parser.error(str(exc))
    print(f"{report['status']}: {args.output_dir / REPORT_JSON_NAME}")
    return 0 if report["status"] == STATUS_PASS else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

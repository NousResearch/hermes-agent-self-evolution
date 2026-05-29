"""Phase 3 candidate-only system prompt scaffold.

This module prepares review-only Phase 3 system prompt artifacts. It is a
fail-closed scaffold: no GEPA/DSPy optimization, no real benchmark execution,
no prompt/source mutation, and no active apply are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

SCAFFOLD_VERSION = "phase3-candidate-scaffold-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_ROOT = "output/phase3-system-prompt/"
PHASE3_OUTPUT_ROOT = REPO_ROOT / ALLOWED_OUTPUT_ROOT
EVOLVABLE_SECTIONS = (
    "DEFAULT_AGENT_IDENTITY",
    "MEMORY_GUIDANCE",
    "SESSION_SEARCH_GUIDANCE",
    "SKILLS_GUIDANCE",
    "PLATFORM_HINTS",
)
SEPARATE_APPROVAL_REQUIRED_BEFORE = [
    "running GEPA/DSPy optimization",
    "running real TBLite/YC-Bench benchmark commands",
    "editing Hermes Agent prompt source",
    "applying evolved prompt to active runtime",
    "default-gate promotion",
]


class CandidateScaffoldFailed(ValueError):
    """Raised after writing a failed candidate-only scaffold report."""


@dataclass(frozen=True)
class PromptPayload:
    """Parsed prompt payload plus stable metadata."""

    path: Path
    raw_bytes: bytes
    sections: dict[str, str]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()

    @property
    def bytes(self) -> int:
        return len(self.raw_bytes)


def run_candidate_scaffold(
    *,
    baseline_prompt: str | Path,
    candidate_prompt: str | Path,
    output_dir: str | Path,
    dry_run: bool,
) -> dict[str, object]:
    """Write candidate-only Phase 3 review artifacts under the allowed output root."""

    if dry_run is not True:
        raise ValueError("Phase 3 candidate scaffold currently requires --dry-run")

    baseline_path = Path(baseline_prompt)
    candidate_path = Path(candidate_prompt)
    output_path = _normalize_output_dir(Path(output_dir))
    _validate_output_dir(output_path)

    baseline = _load_prompt_payload(baseline_path)
    candidate = _load_prompt_payload(candidate_path)
    changed_sections = _changed_sections(baseline.sections, candidate.sections)
    non_evolvable_changes = [section for section in changed_sections if section not in EVOLVABLE_SECTIONS]
    unchanged_sections = sorted(
        section
        for section in set(baseline.sections) | set(candidate.sections)
        if section not in changed_sections
    )
    failed_checks = [
        f"non_evolvable_section_changed {section}" for section in non_evolvable_changes
    ]

    baseline_snapshot = output_path / "baseline_system_prompt.json"
    candidate_snapshot = output_path / "candidate_system_prompt.json"
    report_path = output_path / "candidate_only_report.json"
    review_packet_path = output_path / "review_packet.md"
    write_target_paths = [
        baseline_snapshot,
        candidate_snapshot,
        report_path,
        review_packet_path,
    ]
    _validate_write_targets(write_target_paths, (baseline_path, candidate_path))
    write_targets = [str(path) for path in write_target_paths]

    output_path.mkdir(parents=True, exist_ok=True)
    baseline_snapshot.write_bytes(baseline.raw_bytes)
    candidate_snapshot.write_bytes(candidate.raw_bytes)

    report: dict[str, object] = {
        "phase": "3",
        "mode": "candidate-only-scaffold",
        "scaffold_version": SCAFFOLD_VERSION,
        "dry_run": True,
        "candidate_only": True,
        "read_only_inputs": True,
        "execution_started": False,
        "run_gepa_now": False,
        "run_dspy_now": False,
        "mutate_active_system_prompt_now": False,
        "active_system_prompt_apply_approved": False,
        "apply_ready": False,
        "external_calls_performed": False,
        "real_benchmarks_executed": False,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "evolvable_sections": list(EVOLVABLE_SECTIONS),
        "changed_sections": changed_sections,
        "unchanged_sections": unchanged_sections,
        "non_evolvable_section_changes": non_evolvable_changes,
        "prompt_artifacts": {
            "baseline": {
                "input_path": str(baseline.path),
                "snapshot_path": str(baseline_snapshot),
                "sha256": baseline.sha256,
                "bytes": baseline.bytes,
            },
            "candidate": {
                "input_path": str(candidate.path),
                "snapshot_path": str(candidate_snapshot),
                "sha256": candidate.sha256,
                "bytes": candidate.bytes,
            },
        },
        "benchmark_gate": {
            "status": "real_benchmarks_required_not_executed",
            "real_benchmarks_required_before_acceptance": True,
            "real_benchmarks_executed": False,
            "deferred_benchmarks": ["TBLite", "YC-Bench"],
        },
        "human_approval_gate": {
            "execution_approved": False,
            "active_apply_approved": False,
            "separate_approval_required_before": SEPARATE_APPROVAL_REQUIRED_BEFORE,
        },
        "artifacts": {
            "output_dir": str(output_path),
            "baseline_prompt": str(baseline_snapshot),
            "candidate_prompt": str(candidate_snapshot),
            "candidate_only_report": str(report_path),
            "review_packet": str(review_packet_path),
        },
        "write_targets": write_targets,
        "output_constraints": {
            "allowed_root": ALLOWED_OUTPUT_ROOT,
            "path_traversal": "resolved_path_must_remain_under_allowed_root",
        },
    }

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    review_packet_path.write_text(_render_review_packet(report) + "\n")

    if failed_checks:
        raise CandidateScaffoldFailed(", ".join(failed_checks))
    return report


def _load_prompt_payload(path: Path) -> PromptPayload:
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"prompt artifact must be valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"prompt artifact must be a JSON object: {path}")
    sections = payload.get("sections")
    if not isinstance(sections, Mapping):
        raise ValueError(f"prompt artifact must contain a sections object: {path}")
    parsed_sections: dict[str, str] = {}
    for key, value in sections.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"prompt sections must map strings to strings: {path}")
        parsed_sections[key] = value
    return PromptPayload(path=path, raw_bytes=raw_bytes, sections=parsed_sections)


def _changed_sections(baseline: Mapping[str, str], candidate: Mapping[str, str]) -> list[str]:
    return sorted(
        section
        for section in set(baseline) | set(candidate)
        if baseline.get(section) != candidate.get(section)
    )


def _normalize_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _validate_output_dir(output_dir: Path) -> None:
    allowed_root = PHASE3_OUTPUT_ROOT.resolve()
    if output_dir == allowed_root:
        return
    if not output_dir.is_relative_to(allowed_root):
        raise ValueError(f"output-dir must stay under {ALLOWED_OUTPUT_ROOT}: {output_dir}")


def _validate_write_targets(write_targets: list[Path], input_paths: tuple[Path, ...]) -> None:
    allowed_root = PHASE3_OUTPUT_ROOT.resolve()
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


def _render_review_packet(report: Mapping[str, object]) -> str:
    failed_checks = report.get("failed_checks")
    if not isinstance(failed_checks, list):
        failed_checks = []
    changed_sections = report.get("changed_sections")
    if not isinstance(changed_sections, list):
        changed_sections = []
    return "\n".join(
        [
            "# Phase 3 candidate-only scaffold review packet",
            "",
            "This candidate-only scaffold records review artifacts only; it performs no GEPA/DSPy execution, no real benchmark execution, and no active prompt/source apply.",
            "",
            f"- passed: `{str(report.get('passed')).lower()}`",
            f"- apply_ready: `{str(report.get('apply_ready')).lower()}`",
            f"- real_benchmarks_executed: `{str(report.get('real_benchmarks_executed')).lower()}`",
            f"- changed_sections: `{', '.join(str(section) for section in changed_sections)}`",
            f"- failed_checks: `{', '.join(str(check) for check in failed_checks) if failed_checks else 'none'}`",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a Phase 3 candidate-only system prompt scaffold.")
    parser.add_argument("--baseline-prompt", required=True)
    parser.add_argument("--candidate-prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_candidate_scaffold(
            baseline_prompt=args.baseline_prompt,
            candidate_prompt=args.candidate_prompt,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except CandidateScaffoldFailed as exc:
        parser.error(f"candidate scaffold failed: {exc}")
    except ValueError as exc:
        parser.error(str(exc))
    artifacts = report["artifacts"]
    if not isinstance(artifacts, Mapping):
        parser.error("candidate scaffold report missing artifacts")
    print(json.dumps({"candidate_only_report": artifacts["candidate_only_report"], "passed": report["passed"]}))


if __name__ == "__main__":
    main()

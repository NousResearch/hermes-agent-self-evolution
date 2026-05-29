"""Phase 3 local preflight gate.

This module validates the local candidate-only scaffold report and dry-run
benchmark adapter reports before any Phase 3 optimizer or real benchmark
execution is considered. It intentionally keeps execution readiness blocked until
separate human approval and real benchmark evidence exist.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

ALLOWED_OUTPUT_ROOT = "output/phase3-system-prompt/"
ALLOWED_OUTPUT_SUFFIX = ".json"
REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE3_OUTPUT_ROOT = REPO_ROOT / ALLOWED_OUTPUT_ROOT
PREFLIGHT_VERSION = "phase3-local-preflight-gate-v1"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
EXPECTED_CANDIDATE_OUTPUT_CONSTRAINTS = {
    "allowed_root": ALLOWED_OUTPUT_ROOT,
    "path_traversal": "resolved_path_must_remain_under_allowed_root",
}
EXPECTED_OUTPUT_CONSTRAINTS = {
    "allowed_root": ALLOWED_OUTPUT_ROOT,
    "suffix": ALLOWED_OUTPUT_SUFFIX,
    "fresh_output_required": True,
    "symlink_output_allowed": False,
    "hardlink_output_allowed": False,
    "input_output_overlap_allowed": False,
}
EXPECTED_PASS_CONDITIONS = {
    "TBLite": "no_regression_against_baseline",
    "YC-Bench": "coherence_score_holds_or_improves",
}
NEXT_REQUIRED_BEFORE_PHASE3_EXECUTION = [
    "separate human approval for GEPA/DSPy execution",
    "separate human approval for real TBLite/YC-Bench execution",
    "real benchmark reports replacing or supplementing dry-run fixture reports",
    "rollback handle verified before active apply",
]


class Phase3PreflightFailed(ValueError):
    """Raised after writing a failed Phase 3 preflight report."""


@dataclass(frozen=True)
class LoadedReport:
    """Loaded JSON report plus source path."""

    path: Path
    data: Mapping[str, object]


def run_phase3_preflight_gate(
    *,
    candidate_report: str | Path,
    tblite_report: str | Path,
    yc_bench_report: str | Path,
    output_json: str | Path,
    dry_run: bool,
) -> dict[str, object]:
    """Validate local Phase 3 dry-run artifacts and write a preflight report."""

    if dry_run is not True:
        raise ValueError("Phase 3 local preflight gate currently requires --dry-run")

    candidate_path = Path(candidate_report)
    tblite_path = Path(tblite_report)
    yc_bench_path = Path(yc_bench_report)
    output_path = _normalize_output_json_path(Path(output_json))
    _validate_output_json_path(output_path)
    _validate_distinct_output_path(output_path, (candidate_path, tblite_path, yc_bench_path))

    candidate = _load_report(candidate_path)
    tblite = _load_report(tblite_path)
    yc_bench = _load_report(yc_bench_path)
    failed_checks: list[str] = []

    _validate_candidate_scaffold_report(candidate.data, failed_checks)
    benchmark_summaries = {
        "TBLite": _validate_benchmark_report(tblite.data, "TBLite", failed_checks),
        "YC-Bench": _validate_benchmark_report(yc_bench.data, "YC-Bench", failed_checks),
    }
    prompt_checks = _validate_prompt_artifact_consistency(
        {
            "candidate_report": candidate.data,
            "TBLite": tblite.data,
            "YC-Bench": yc_bench.data,
        },
        failed_checks,
    )

    report: dict[str, object] = {
        "phase": "3",
        "mode": "local-preflight-gate",
        "preflight_version": PREFLIGHT_VERSION,
        "dry_run": True,
        "candidate_only": True,
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "phase3_execution_ready": False,
        "execution_started": False,
        "run_gepa_now": False,
        "run_dspy_now": False,
        "mutate_active_system_prompt_now": False,
        "active_system_prompt_apply_approved": False,
        "real_benchmarks_executed": False,
        "real_benchmarks_required_before_execution": True,
        "human_approval_required_before_execution": True,
        "dry_run_benchmark_reports": benchmark_summaries,
        "prompt_artifact_checks": prompt_checks,
        "next_required_before_phase3_execution": NEXT_REQUIRED_BEFORE_PHASE3_EXECUTION,
        "artifacts": {
            "candidate_report": str(candidate.path),
            "tblite_report": str(tblite.path),
            "yc_bench_report": str(yc_bench.path),
            "preflight_report": str(output_path),
        },
        "write_targets": [str(output_path)],
        "output_constraints": EXPECTED_OUTPUT_CONSTRAINTS,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if failed_checks:
        raise Phase3PreflightFailed(", ".join(failed_checks))
    return report


def _load_report(path: Path) -> LoadedReport:
    try:
        payload = json.loads(path.read_text(), parse_constant=_reject_json_constant)
    except OSError as exc:
        raise ValueError(f"report must be readable: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"report must be valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"report JSON root must be an object: {path}")
    return LoadedReport(path=path, data=payload)


def _validate_candidate_scaffold_report(report: Mapping[str, object], failed_checks: list[str]) -> None:
    expected_fields = {
        "phase": "3",
        "mode": "candidate-only-scaffold",
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
        "passed": True,
    }
    for field, expected in expected_fields.items():
        _require_exact_field(report, field, expected, "candidate_report", failed_checks)

    benchmark_gate = _mapping(report.get("benchmark_gate"), "candidate_report.benchmark_gate", failed_checks)
    if benchmark_gate is not None:
        if benchmark_gate.get("real_benchmarks_executed") is not False:
            failed_checks.append("candidate_report benchmark_gate.real_benchmarks_executed must be false")
        if benchmark_gate.get("real_benchmarks_required_before_acceptance") is not True:
            failed_checks.append(
                "candidate_report benchmark_gate.real_benchmarks_required_before_acceptance must be true"
            )

    human_gate = _mapping(report.get("human_approval_gate"), "candidate_report.human_approval_gate", failed_checks)
    if human_gate is not None:
        if human_gate.get("execution_approved") is not False:
            failed_checks.append("candidate_report human_approval_gate.execution_approved must be false")
        if human_gate.get("active_apply_approved") is not False:
            failed_checks.append("candidate_report human_approval_gate.active_apply_approved must be false")

    if not _validate_exact_mapping(
        report.get("output_constraints"),
        EXPECTED_CANDIDATE_OUTPUT_CONSTRAINTS,
        "candidate_report output_constraints",
        failed_checks,
    ):
        failed_checks.append("candidate_report output_constraints mismatch")

    failed = report.get("failed_checks")
    if not isinstance(failed, list):
        failed_checks.append("candidate_report failed_checks must be a list")
    elif failed:
        failed_checks.append("candidate_report failed_checks must be empty for local preflight")


def _validate_benchmark_report(
    report: Mapping[str, object],
    expected_benchmark: str,
    failed_checks: list[str],
) -> dict[str, object]:
    expected_fields = {
        "benchmark": expected_benchmark,
        "adapter_version": "phase3-benchmark-adapter-v1",
        "mode": "dry-run-fixture",
        "dry_run": True,
        "candidate_only": True,
        "read_only": True,
        "external_calls_performed": False,
        "apply_ready": False,
        "pass_condition": EXPECTED_PASS_CONDITIONS[expected_benchmark],
        "passed": True,
    }
    for field, expected in expected_fields.items():
        _require_exact_field(report, field, expected, f"{expected_benchmark} report", failed_checks)

    if not _validate_exact_mapping(
        report.get("output_constraints"),
        EXPECTED_OUTPUT_CONSTRAINTS,
        f"{expected_benchmark} report output_constraints",
        failed_checks,
    ):
        failed_checks.append(f"{expected_benchmark} report output_constraints mismatch")
    failed = report.get("failed_checks")
    if not isinstance(failed, list):
        failed_checks.append(f"{expected_benchmark} report failed_checks must be a list")
    elif failed:
        failed_checks.append(f"{expected_benchmark} report failed_checks must be empty for local preflight")

    metrics = _mapping(report.get("metrics"), f"{expected_benchmark}.metrics", failed_checks)
    case_count = metrics.get("case_count") if metrics is not None else None
    candidate_regression_count = metrics.get("candidate_regression_count") if metrics is not None else None
    baseline_score = metrics.get("baseline_score") if metrics is not None else None
    candidate_score = metrics.get("candidate_score") if metrics is not None else None
    score_delta = metrics.get("score_delta") if metrics is not None else None
    total_weight = metrics.get("total_weight") if metrics is not None else None
    baseline_number = _as_number(baseline_score)
    candidate_number = _as_number(candidate_score)
    score_delta_number = _as_number(score_delta)
    total_weight_number = _as_number(total_weight)
    if metrics is not None:
        if not _is_positive_int(case_count):
            failed_checks.append(f"{expected_benchmark} metrics.case_count must be a positive integer")
        if candidate_regression_count != 0 or type(candidate_regression_count) is not int:
            failed_checks.append(f"{expected_benchmark} metrics.candidate_regression_count must be 0")
        if total_weight_number is None or total_weight_number <= 0:
            failed_checks.append(f"{expected_benchmark} metrics.total_weight must be a positive finite number")
        if baseline_number is None:
            failed_checks.append(f"{expected_benchmark} metrics.baseline_score must be a finite number")
        if candidate_number is None:
            failed_checks.append(f"{expected_benchmark} metrics.candidate_score must be a finite number")
        if score_delta_number is None:
            failed_checks.append(f"{expected_benchmark} metrics.score_delta must be a finite number")
        if baseline_number is not None and candidate_number is not None and candidate_number < baseline_number:
            failed_checks.append(f"{expected_benchmark} metrics.candidate_score must be >= baseline_score")
        if baseline_number is not None and candidate_number is not None and score_delta_number is not None:
            expected_delta = candidate_number - baseline_number
            if not math.isclose(score_delta_number, expected_delta, rel_tol=1e-9, abs_tol=1e-9):
                failed_checks.append(f"{expected_benchmark} metrics.score_delta must equal candidate_score - baseline_score")

    fixture_cases = _mapping(report.get("fixture_cases"), f"{expected_benchmark}.fixture_cases", failed_checks)
    if fixture_cases is not None:
        fixture_case_count = fixture_cases.get("case_count")
        if not _is_positive_int(fixture_case_count):
            failed_checks.append(f"{expected_benchmark} fixture_cases.case_count must be a positive integer")
        elif _is_positive_int(case_count) and fixture_case_count != case_count:
            failed_checks.append(f"{expected_benchmark} fixture_cases.case_count must equal metrics.case_count")
    cases = report.get("cases")
    if not isinstance(cases, list):
        failed_checks.append(f"{expected_benchmark} report cases must be a list")
    else:
        if _is_positive_int(case_count) and len(cases) != case_count:
            failed_checks.append(f"{expected_benchmark} report cases length must equal metrics.case_count")
        _validate_case_evidence(cases, expected_benchmark, failed_checks)
    return {
        "passed": report.get("passed") is True,
        "mode": report.get("mode"),
        "read_only": report.get("read_only") is True,
        "external_calls_performed": report.get("external_calls_performed") is True,
        "apply_ready": report.get("apply_ready") is True,
        "case_count": case_count,
        "candidate_regression_count": candidate_regression_count,
    }


def _validate_prompt_artifact_consistency(
    reports: Mapping[str, Mapping[str, object]],
    failed_checks: list[str],
) -> dict[str, object]:
    baseline_values: dict[str, str] = {}
    candidate_values: dict[str, str] = {}
    for name, report in reports.items():
        prompt_artifacts = _mapping(report.get("prompt_artifacts"), f"{name}.prompt_artifacts", failed_checks)
        if prompt_artifacts is None:
            continue
        baseline_sha = _prompt_sha(prompt_artifacts, "baseline", f"{name}.prompt_artifacts", failed_checks)
        candidate_sha = _prompt_sha(prompt_artifacts, "candidate", f"{name}.prompt_artifacts", failed_checks)
        if baseline_sha is not None:
            baseline_values[name] = baseline_sha
        if candidate_sha is not None:
            candidate_values[name] = candidate_sha

    baseline_unique = sorted(set(baseline_values.values()))
    candidate_unique = sorted(set(candidate_values.values()))
    consistent = len(baseline_unique) == 1 and len(candidate_unique) == 1 and len(baseline_values) == 3 and len(candidate_values) == 3
    if len(baseline_unique) != 1 or len(baseline_values) != 3:
        failed_checks.append("prompt_artifact_mismatch baseline_sha256")
    if len(candidate_unique) != 1 or len(candidate_values) != 3:
        failed_checks.append("prompt_artifact_mismatch candidate_sha256")
    return {
        "baseline_sha256": baseline_unique[0] if len(baseline_unique) == 1 else None,
        "candidate_sha256": candidate_unique[0] if len(candidate_unique) == 1 else None,
        "consistent_across_reports": consistent,
    }


def _prompt_sha(
    prompt_artifacts: Mapping[str, object],
    key: str,
    path: str,
    failed_checks: list[str],
) -> str | None:
    section = _mapping(prompt_artifacts.get(key), f"{path}.{key}", failed_checks)
    if section is None:
        return None
    sha = section.get("sha256")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        failed_checks.append(f"{path}.{key}.sha256 must be a 64-character hexadecimal string")
        return None
    return sha


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"report JSON must not contain non-finite numeric constant: {value}")


def _mapping(value: object, path: str, failed_checks: list[str]) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        failed_checks.append(f"{path} must be an object")
        return None
    return value


def _require_exact_field(
    report: Mapping[str, object],
    field: str,
    expected: object,
    label: str,
    failed_checks: list[str],
) -> None:
    value = report.get(field)
    if isinstance(expected, bool):
        if type(value) is not bool or value is not expected:
            failed_checks.append(f"{label} {field} expected {expected!r}")
        return
    if value != expected:
        failed_checks.append(f"{label} {field} expected {expected!r}")


def _validate_exact_mapping(
    value: object,
    expected: Mapping[str, object],
    label: str,
    failed_checks: list[str],
) -> bool:
    mapping = _mapping(value, label, failed_checks)
    if mapping is None:
        return False
    if set(mapping) != set(expected):
        failed_checks.append(f"{label} keys mismatch")
        return False
    start_count = len(failed_checks)
    for field, expected_value in expected.items():
        _require_exact_field(mapping, field, expected_value, label, failed_checks)
    return len(failed_checks) == start_count


def _validate_case_evidence(cases: Sequence[object], expected_benchmark: str, failed_checks: list[str]) -> None:
    if len(cases) == 0:
        failed_checks.append(f"{expected_benchmark} report cases must be non-empty")
        return
    for index, case in enumerate(cases):
        label = f"{expected_benchmark} cases[{index}]"
        case_mapping = _mapping(case, label, failed_checks)
        if case_mapping is None:
            continue
        for field in ("id", "category"):
            value = case_mapping.get(field)
            if not isinstance(value, str) or not value:
                failed_checks.append(f"{label}.{field} must be a non-empty string")
        _require_exact_field(case_mapping, "passed", True, label, failed_checks)
        baseline_score = case_mapping.get("baseline_score")
        candidate_score = case_mapping.get("candidate_score")
        baseline_number = _as_number(baseline_score)
        candidate_number = _as_number(candidate_score)
        if baseline_number is None:
            failed_checks.append(f"{label}.baseline_score must be numeric")
        if candidate_number is None:
            failed_checks.append(f"{label}.candidate_score must be numeric")
        if baseline_number is not None and candidate_number is not None and candidate_number < baseline_number:
            failed_checks.append(f"{label}.candidate_score must be >= baseline_score")
        forbidden_hits = case_mapping.get("candidate_forbidden_hits")
        if forbidden_hits != []:
            failed_checks.append(f"{label}.candidate_forbidden_hits must be present and empty")


def _is_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _as_number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _normalize_output_json_path(output_path: Path) -> Path:
    if output_path.is_absolute():
        return output_path
    return REPO_ROOT / output_path


def _validate_output_json_path(output_path: Path) -> None:
    if output_path.suffix.lower() != ALLOWED_OUTPUT_SUFFIX:
        raise ValueError(f"output-json must use a {ALLOWED_OUTPUT_SUFFIX} suffix: {output_path}")
    output_resolved = output_path.resolve()
    allowed_root_resolved = PHASE3_OUTPUT_ROOT.resolve()
    if not output_resolved.is_relative_to(allowed_root_resolved):
        raise ValueError(f"output-json must stay under {ALLOWED_OUTPUT_ROOT}: {output_path}")


def _validate_distinct_output_path(output_path: Path, input_paths: Sequence[Path]) -> None:
    if output_path.is_symlink():
        raise ValueError(f"output-json must not be a symlink: {output_path}")
    if output_path.exists():
        for input_path in input_paths:
            if input_path.exists() and output_path.samefile(input_path):
                raise ValueError(f"output-json must not overwrite input artifact: {input_path}")
        raise ValueError(f"output-json must not already exist: {output_path}")

    output_resolved = output_path.resolve()
    for input_path in input_paths:
        if output_resolved == input_path.resolve():
            raise ValueError(f"output-json must not overwrite input artifact: {input_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Phase 3 local preflight artifacts.")
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--tblite-report", required=True)
    parser.add_argument("--yc-bench-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_phase3_preflight_gate(
            candidate_report=args.candidate_report,
            tblite_report=args.tblite_report,
            yc_bench_report=args.yc_bench_report,
            output_json=args.output_json,
            dry_run=args.dry_run,
        )
    except Phase3PreflightFailed as exc:
        parser.error(f"phase3 preflight failed: {exc}")
    except ValueError as exc:
        parser.error(str(exc))
    artifacts = report["artifacts"]
    if not isinstance(artifacts, Mapping):
        parser.error("phase3 preflight report missing artifacts")
    print(
        json.dumps(
            {
                "phase3_preflight_report": artifacts["preflight_report"],
                "passed": report["passed"],
                "phase3_execution_ready": report["phase3_execution_ready"],
            }
        )
    )


if __name__ == "__main__":
    main()

"""Unified local-only HSE real benchmark runner.

The runner is intentionally limited to local pinned assets and zero provider/API
spend. It may write benchmark reports only under ``output/hse-real-benchmark/``
and never performs network calls, provider/model calls, GitHub writes, active
apply, or strict PLAN gate closure.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.benchmarks.contract import (
    ADAPTER_VERSION,
    FixtureBenchmarkCase,
    PromptArtifact,
    _evaluate_case,
    _numeric_case_value,
    load_fixture_cases,
    load_prompt_artifact,
)
from evolution.benchmarks.run_tblite import PASS_CONDITION as TBLITE_PASS_CONDITION
from evolution.benchmarks.run_tblite import collect_tblite_real_benchmark_evidence
from evolution.benchmarks.run_yc_bench import PASS_CONDITION as YC_BENCH_PASS_CONDITION
from evolution.benchmarks.run_yc_bench import collect_yc_bench_real_benchmark_evidence
from evolution.tools.report_contract import load_and_validate_candidate_only_report

REPO_ROOT = Path(__file__).resolve().parents[2]
HSE_ALLOWED_OUTPUT_ROOT = REPO_ROOT / "output" / "hse-real-benchmark"
HSE_ALLOWED_OUTPUT_ROOT_LABEL = "output/hse-real-benchmark/"
PHASE3_FIXTURE_ROOT = REPO_ROOT / "datasets" / "golden" / "benchmarks" / "phase3-system-prompt"
DEFAULT_BASELINE_PROMPT = PHASE3_FIXTURE_ROOT / "baseline_system_prompt.json"
DEFAULT_CANDIDATE_PROMPT = PHASE3_FIXTURE_ROOT / "candidate_system_prompt.json"
DEFAULT_TBLITE_CASES = PHASE3_FIXTURE_ROOT / "tblite_cases.jsonl"
DEFAULT_YC_BENCH_CASES = PHASE3_FIXTURE_ROOT / "yc_bench_fast_test.jsonl"
DEFAULT_TBLITE_ROOT = (
    REPO_ROOT
    / "output"
    / "phase3-system-prompt"
    / "stage123-20260529T145248Z"
    / "external-benchmarks"
    / "terminal-bench-lite"
)
DEFAULT_YC_BENCH_ROOT = (
    REPO_ROOT
    / "output"
    / "phase3-system-prompt"
    / "stage123-20260529T145248Z"
    / "external-benchmarks"
    / "yc-bench"
)
DEFAULT_TOOL_SELECTION_DATASET = REPO_ROOT / "datasets" / "golden" / "tool-description" / "tool_selection.jsonl"
DEFAULT_TOOL_SELECTION_REPORT = (
    REPO_ROOT
    / "output"
    / "tool-description"
    / "phase5-tool-selection-all-pass-20260606-145446"
    / "run"
    / "candidate_only_report.json"
)
SUPPORTED_SUITES = ("TBLite", "YC-Bench", "Phase2 PLAN-scale tool-selection triples")


def suite_readiness(
    suite: str,
    *,
    tblite_root: str | Path = DEFAULT_TBLITE_ROOT,
    yc_bench_root: str | Path = DEFAULT_YC_BENCH_ROOT,
    baseline_prompt: str | Path = DEFAULT_BASELINE_PROMPT,
    candidate_prompt: str | Path = DEFAULT_CANDIDATE_PROMPT,
    tblite_cases: str | Path = DEFAULT_TBLITE_CASES,
    yc_bench_cases: str | Path = DEFAULT_YC_BENCH_CASES,
    tool_selection_dataset: str | Path = DEFAULT_TOOL_SELECTION_DATASET,
    tool_selection_report: str | Path = DEFAULT_TOOL_SELECTION_REPORT,
    task_limit: int = 3,
) -> dict[str, Any]:
    """Return non-executing readiness for one supported benchmark suite."""

    blockers: list[str] = []
    evidence: dict[str, Any] = {}
    if suite not in SUPPORTED_SUITES:
        blockers.append("unsupported_suite")
    _collect_required_file_readiness(
        blockers,
        {
            "baseline_prompt": baseline_prompt,
            "candidate_prompt": candidate_prompt,
        },
    )
    if suite == "TBLite":
        _collect_required_file_readiness(blockers, {"tblite_cases": tblite_cases})
        try:
            evidence = collect_tblite_real_benchmark_evidence(tblite_root, task_limit=task_limit)
        except ValueError as exc:
            blockers.append(f"tblite_assets_invalid: {exc}")
    elif suite == "YC-Bench":
        _collect_required_file_readiness(blockers, {"yc_bench_cases": yc_bench_cases})
        try:
            evidence = collect_yc_bench_real_benchmark_evidence(yc_bench_root, preset="fast_test")
        except ValueError as exc:
            blockers.append(f"yc_bench_assets_invalid: {exc}")
    elif suite == "Phase2 PLAN-scale tool-selection triples":
        try:
            evidence = _collect_tool_selection_evidence(tool_selection_dataset, tool_selection_report)
        except ValueError as exc:
            blockers.append(f"tool_selection_assets_invalid: {exc}")

    return {
        "suite": suite,
        "ready": not blockers,
        "blocked_by": blockers,
        "network_calls_required": False,
        "provider_or_model_spend_required": False,
        "github_write_required": False,
        "local_assets_evidence": evidence,
    }


def run_real_benchmark_suite(
    *,
    suite: str,
    baseline_commit: str,
    current_commit: str,
    output_json: str | Path,
    baseline_prompt: str | Path = DEFAULT_BASELINE_PROMPT,
    candidate_prompt: str | Path = DEFAULT_CANDIDATE_PROMPT,
    tblite_cases: str | Path = DEFAULT_TBLITE_CASES,
    yc_bench_cases: str | Path = DEFAULT_YC_BENCH_CASES,
    tblite_root: str | Path = DEFAULT_TBLITE_ROOT,
    yc_bench_root: str | Path = DEFAULT_YC_BENCH_ROOT,
    tool_selection_dataset: str | Path = DEFAULT_TOOL_SELECTION_DATASET,
    tool_selection_report: str | Path = DEFAULT_TOOL_SELECTION_REPORT,
    task_limit: int = 3,
) -> dict[str, Any]:
    """Run one approved local-only real benchmark smoke suite and write JSON."""

    _require_non_empty("baseline_commit", baseline_commit)
    _require_non_empty("current_commit", current_commit)
    output_path = _normalize_output_json_path(Path(output_json))
    _validate_hse_output_json_path(output_path)
    readiness = suite_readiness(
        suite,
        tblite_root=tblite_root,
        yc_bench_root=yc_bench_root,
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        tblite_cases=tblite_cases,
        yc_bench_cases=yc_bench_cases,
        tool_selection_dataset=tool_selection_dataset,
        tool_selection_report=tool_selection_report,
        task_limit=task_limit,
    )
    if readiness["ready"] is not True:
        raise ValueError(f"suite readiness blocked: {readiness['blocked_by']}")

    if suite == "TBLite":
        evidence = collect_tblite_real_benchmark_evidence(tblite_root, task_limit=task_limit)
        report = _prompt_fixture_smoke_report(
            suite=suite,
            pass_condition=TBLITE_PASS_CONDITION,
            baseline_commit=baseline_commit,
            current_commit=current_commit,
            baseline_prompt=baseline_prompt,
            candidate_prompt=candidate_prompt,
            fixtures_jsonl=tblite_cases,
            output_json=output_path,
            real_benchmark_evidence=evidence,
        )
    elif suite == "YC-Bench":
        evidence = collect_yc_bench_real_benchmark_evidence(yc_bench_root, preset="fast_test")
        report = _prompt_fixture_smoke_report(
            suite=suite,
            pass_condition=YC_BENCH_PASS_CONDITION,
            baseline_commit=baseline_commit,
            current_commit=current_commit,
            baseline_prompt=baseline_prompt,
            candidate_prompt=candidate_prompt,
            fixtures_jsonl=yc_bench_cases,
            output_json=output_path,
            real_benchmark_evidence=evidence,
            preset="fast_test",
        )
    elif suite == "Phase2 PLAN-scale tool-selection triples":
        report = _tool_selection_smoke_report(
            baseline_commit=baseline_commit,
            current_commit=current_commit,
            output_json=output_path,
            tool_selection_dataset=tool_selection_dataset,
            tool_selection_report=tool_selection_report,
        )
    else:
        raise ValueError(f"unsupported suite: {suite}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def _prompt_fixture_smoke_report(
    *,
    suite: str,
    pass_condition: str,
    baseline_commit: str,
    current_commit: str,
    baseline_prompt: str | Path,
    candidate_prompt: str | Path,
    fixtures_jsonl: str | Path,
    output_json: Path,
    real_benchmark_evidence: Mapping[str, Any],
    preset: str | None = None,
) -> dict[str, Any]:
    baseline_artifact = load_prompt_artifact(baseline_prompt)
    candidate_artifact = load_prompt_artifact(candidate_prompt)
    cases = load_fixture_cases(fixtures_jsonl)
    if not cases:
        raise ValueError(f"fixture case file must contain at least one case: {fixtures_jsonl}")
    case_reports = [_evaluate_case(case, baseline_artifact.normalized_text, candidate_artifact.normalized_text) for case in cases]
    baseline_score = sum(_numeric_case_value(case, "baseline_score") for case in case_reports)
    candidate_score = sum(_numeric_case_value(case, "candidate_score") for case in case_reports)
    candidate_regressions = [case for case in case_reports if case["passed"] is not True]
    failed_checks = [
        f"case_regression {case['id']} candidate={case['candidate_score']} baseline={case['baseline_score']}"
        for case in candidate_regressions
    ]
    if candidate_score < baseline_score:
        failed_checks.append(f"aggregate_regression candidate_score {candidate_score:.4f} < baseline {baseline_score:.4f}")
    total_weight = sum(case.weight for case in cases)
    report = _base_suite_report(
        suite=suite,
        pass_condition=pass_condition,
        baseline_commit=baseline_commit,
        current_commit=current_commit,
        output_json=output_json,
        passed=not failed_checks,
        failed_checks=failed_checks,
    )
    report.update(
        {
            "adapter_version": ADAPTER_VERSION,
            "prompt_artifacts": {
                "baseline": _prompt_artifact_report(baseline_artifact),
                "candidate": _prompt_artifact_report(candidate_artifact),
            },
            "fixture_cases": {
                "path": str(Path(fixtures_jsonl)),
                "case_count": len(cases),
                "categories": sorted({case.category for case in cases}),
            },
            "metrics": {
                "case_count": len(cases),
                "total_weight": total_weight,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "score_delta": candidate_score - baseline_score,
                "candidate_regression_count": len(candidate_regressions),
            },
            "cases": case_reports,
            "real_benchmark_evidence": dict(real_benchmark_evidence),
        }
    )
    if preset is not None:
        report["preset"] = preset
    return report


def _tool_selection_smoke_report(
    *,
    baseline_commit: str,
    current_commit: str,
    output_json: Path,
    tool_selection_dataset: str | Path,
    tool_selection_report: str | Path,
) -> dict[str, Any]:
    evidence = _collect_tool_selection_evidence(tool_selection_dataset, tool_selection_report)
    source_report = json.loads(Path(tool_selection_report).read_text())
    metrics = source_report.get("metrics")
    phase2d_gate = source_report.get("phase2d_gate")
    if not isinstance(metrics, Mapping) or not isinstance(phase2d_gate, Mapping):
        raise ValueError("tool-selection source report missing metrics or phase2d_gate")
    failed_checks = list(phase2d_gate.get("failed_checks", [])) if isinstance(phase2d_gate.get("failed_checks"), list) else []
    if phase2d_gate.get("passed") is not True:
        failed_checks.append("phase2d_gate_not_passed")
    report = _base_suite_report(
        suite="Phase2 PLAN-scale tool-selection triples",
        pass_condition="no_aggregate_or_per_tool_regression_beyond_gate",
        baseline_commit=baseline_commit,
        current_commit=current_commit,
        output_json=output_json,
        passed=not failed_checks,
        failed_checks=failed_checks,
    )
    report.update(
        {
            "adapter_version": "phase2-tool-selection-local-smoke-v1",
            "metrics": {
                "case_count": metrics.get("case_count"),
                "selection_accuracy": metrics.get("selection_accuracy"),
                "wrong_tool_avoidance": metrics.get("wrong_tool_avoidance"),
                "argument_cue_coverage": metrics.get("argument_cue_coverage"),
                "constraint_pass_rate": metrics.get("constraint_pass_rate"),
            },
            "phase2d_gate": dict(phase2d_gate),
            "real_benchmark_evidence": evidence,
            "source_tool_selection_report": {
                "path": str(Path(tool_selection_report)),
                "sha256": _sha256_path(Path(tool_selection_report)),
            },
            "source_tool_selection_dataset": {
                "path": str(Path(tool_selection_dataset)),
                "sha256": _sha256_path(Path(tool_selection_dataset)),
            },
        }
    )
    return report


def _base_suite_report(
    *,
    suite: str,
    pass_condition: str,
    baseline_commit: str,
    current_commit: str,
    output_json: Path,
    passed: bool,
    failed_checks: Sequence[str],
) -> dict[str, Any]:
    return {
        "benchmark": suite,
        "mode": "real-benchmark-smoke",
        "dry_run": False,
        "candidate_only": True,
        "read_only": True,
        "external_calls_performed": False,
        "provider_or_model_spend_performed": False,
        "network_calls_performed": False,
        "github_write_performed": False,
        "active_apply_performed": False,
        "strict_plan_gate_closed": False,
        "baseline_current_materialization_performed": False,
        "external_benchmark_assets_validated": True,
        "real_benchmark_smoke_validated": True,
        "full_benchmark_executed": False,
        "apply_ready": False,
        "benchmark_gate_candidate_passed": passed,
        "pass_condition": pass_condition,
        "passed": passed,
        "failed_checks": list(failed_checks),
        "subject_commits": {
            "baseline": baseline_commit,
            "current": current_commit,
        },
        "artifacts": {
            "output_json": str(output_json),
        },
        "write_targets": [str(output_json)],
        "output_constraints": {
            "allowed_root": HSE_ALLOWED_OUTPUT_ROOT_LABEL,
            "suffix": ".json",
            "fresh_output_required": True,
            "symlink_output_allowed": False,
            "hardlink_output_allowed": False,
            "input_output_overlap_allowed": False,
        },
    }


def _collect_tool_selection_evidence(
    tool_selection_dataset: str | Path,
    tool_selection_report: str | Path,
) -> dict[str, Any]:
    dataset_path = Path(tool_selection_dataset)
    report_path = Path(tool_selection_report)
    if not dataset_path.exists() or not dataset_path.is_file():
        raise ValueError(f"tool-selection dataset missing: {dataset_path}")
    if not report_path.exists() or not report_path.is_file():
        raise ValueError(f"tool-selection report missing: {report_path}")
    rows = _read_jsonl_objects(dataset_path)
    validation = load_and_validate_candidate_only_report(report_path)
    if not validation.passed:
        raise ValueError(f"tool-selection candidate report contract failed: {list(validation.errors)}")
    source_report = json.loads(report_path.read_text())
    gate = source_report.get("phase2d_gate")
    metrics = source_report.get("metrics")
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise ValueError("tool-selection phase2d_gate did not pass")
    if not isinstance(metrics, Mapping):
        raise ValueError("tool-selection metrics missing")
    case_count = metrics.get("case_count")
    if not isinstance(case_count, int) or case_count < len(rows):
        raise ValueError("tool-selection metrics.case_count must cover dataset rows")
    required_fields = {"expected_tool", "confusing_tools", "required_cues", "user_request", "category"}
    for index, row in enumerate(rows, start=1):
        missing = sorted(required_fields - set(row))
        if missing:
            raise ValueError(f"tool-selection dataset row {index} missing fields: {missing}")
    return {
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_path(dataset_path),
        "dataset_row_count": len(rows),
        "candidate_report_path": str(report_path),
        "candidate_report_sha256": _sha256_path(report_path),
        "phase2d_gate_passed": True,
        "selection_accuracy": metrics.get("selection_accuracy"),
        "wrong_tool_avoidance": metrics.get("wrong_tool_avoidance"),
        "execution_scope": "local_pinned_tool_selection_report_smoke",
    }


def _collect_required_file_readiness(blockers: list[str], paths: Mapping[str, str | Path]) -> None:
    for label, raw_path in paths.items():
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            blockers.append(f"missing_{label}: {path}")


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"JSONL dataset must contain at least one row: {path}")
    return rows


def _prompt_artifact_report(artifact: PromptArtifact) -> dict[str, Any]:
    return {
        "path": str(artifact.path),
        "sha256": artifact.sha256,
        "bytes": artifact.bytes,
        "normalized_text_sha256": artifact.normalized_text_sha256,
    }


def _normalize_output_json_path(output_path: Path) -> Path:
    if output_path.is_absolute():
        return output_path
    return REPO_ROOT / output_path


def _validate_hse_output_json_path(output_path: Path) -> None:
    if output_path.suffix.lower() != ".json":
        raise ValueError(f"output-json must use a .json suffix: {output_path}")
    output_resolved = output_path.resolve(strict=False)
    allowed_root_resolved = HSE_ALLOWED_OUTPUT_ROOT.resolve(strict=False)
    if not _is_relative_to(output_resolved, allowed_root_resolved):
        raise ValueError(f"output-json must stay under {HSE_ALLOWED_OUTPUT_ROOT_LABEL}: {output_path}")
    if output_path.is_symlink():
        raise ValueError(f"output-json must not be a symlink: {output_path}")
    if output_path.exists():
        raise ValueError(f"output-json must not already exist: {output_path}")


def _is_relative_to(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_non_empty(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local-only HSE real benchmark smoke suites")
    parser.add_argument("--suite", required=True, choices=SUPPORTED_SUITES)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--current-commit", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate local assets without writing benchmark output")
    parser.add_argument("--baseline-prompt", type=Path, default=DEFAULT_BASELINE_PROMPT)
    parser.add_argument("--candidate-prompt", type=Path, default=DEFAULT_CANDIDATE_PROMPT)
    parser.add_argument("--tblite-cases", type=Path, default=DEFAULT_TBLITE_CASES)
    parser.add_argument("--yc-bench-cases", type=Path, default=DEFAULT_YC_BENCH_CASES)
    parser.add_argument("--tblite-root", type=Path, default=DEFAULT_TBLITE_ROOT)
    parser.add_argument("--yc-bench-root", type=Path, default=DEFAULT_YC_BENCH_ROOT)
    parser.add_argument("--tool-selection-dataset", type=Path, default=DEFAULT_TOOL_SELECTION_DATASET)
    parser.add_argument("--tool-selection-report", type=Path, default=DEFAULT_TOOL_SELECTION_REPORT)
    parser.add_argument("--task-limit", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        readiness = suite_readiness(
            args.suite,
            tblite_root=args.tblite_root,
            yc_bench_root=args.yc_bench_root,
            baseline_prompt=args.baseline_prompt,
            candidate_prompt=args.candidate_prompt,
            tblite_cases=args.tblite_cases,
            yc_bench_cases=args.yc_bench_cases,
            tool_selection_dataset=args.tool_selection_dataset,
            tool_selection_report=args.tool_selection_report,
            task_limit=args.task_limit,
        )
        print(json.dumps(readiness, indent=2, sort_keys=True, allow_nan=False))
        return 0 if readiness["ready"] is True else 2
    try:
        report = run_real_benchmark_suite(
            suite=args.suite,
            baseline_commit=args.baseline_commit,
            current_commit=args.current_commit,
            output_json=args.output_json,
            baseline_prompt=args.baseline_prompt,
            candidate_prompt=args.candidate_prompt,
            tblite_cases=args.tblite_cases,
            yc_bench_cases=args.yc_bench_cases,
            tblite_root=args.tblite_root,
            yc_bench_root=args.yc_bench_root,
            tool_selection_dataset=args.tool_selection_dataset,
            tool_selection_report=args.tool_selection_report,
            task_limit=args.task_limit,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if report["passed"] is True:
        print(f"{args.suite} real benchmark smoke passed: {args.output_json}")
        return 0
    print(f"{args.suite} real benchmark smoke failed: {report['failed_checks']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

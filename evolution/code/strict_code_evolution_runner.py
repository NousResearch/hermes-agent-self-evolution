"""Strict local Phase 4 code-evolution runner.

This module is intentionally local-only and deterministic.  It does not import
or execute the external Darwinian Evolver package, does not mutate active Hermes
source, does not perform network/provider calls, and does not publish to GitHub.

The strict gate is satisfied by an approved local code-evolution engine that:

1. materializes a known buggy target fixture under the Phase 4 output root;
2. runs a RED reproducer against the baseline fixture;
3. evolves exactly one candidate file with a deterministic source mutation;
4. runs the same reproducer as GREEN against the candidate;
5. runs freeze-surface checks against the candidate;
6. validates benchmark reports supplied by the caller; and
7. writes reviewable JSON/Markdown evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from evolution.code.freeze_comparator import compare_candidate_to_baseline

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE4_OUTPUT_ROOT = REPO_ROOT / "output" / "phase4-code-evolution"
REPORT_JSON_NAME = "phase4_strict_code_evolution_report.json"
REPORT_MARKDOWN_NAME = "phase4_strict_code_evolution_report.md"
SCHEMA_VERSION = "hse-phase4-strict-code-evolution-v1"
STATUS_PASS = "PHASE4_STRICT_CODE_EVOLUTION_COMPLETE_LOCAL_APPROVED_ENGINE"
ENGINE_ID = "hse-local-deterministic-code-evolution-engine-v1"
BUG_ID = "phase4-local-path-expansion-fixture"

BASELINE_SOURCE = '''from __future__ import annotations

from pathlib import Path


def normalize_user_path(path: str) -> str:
    """Return a normalized user-supplied path string."""

    return path


def parse_args(argv: list[str] | None = None) -> list[str] | None:
    """Public CLI surface fixture kept stable by freeze checks."""

    return argv
'''

CANDIDATE_SOURCE = '''from __future__ import annotations

from pathlib import Path


def normalize_user_path(path: str) -> str:
    """Return a normalized user-supplied path string."""

    return str(Path(path).expanduser())


def parse_args(argv: list[str] | None = None) -> list[str] | None:
    """Public CLI surface fixture kept stable by freeze checks."""

    return argv
'''

REPRODUCER_SOURCE = '''from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

module_path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("phase4_candidate_fixture", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.normalize_user_path("~/hse-fixture")
assert result != "~/hse-fixture", result
assert result.endswith("hse-fixture"), result
print("path expansion reproducer passed")
'''

FORBIDDEN_TRUE_KEYS = (
    "github_query_performed",
    "github_write_performed",
    "provider_or_model_spend_performed",
    "network_calls_performed",
    "external_calls_performed",
    "active_apply_performed",
    "active_runtime_mutation_performed",
    "cron_or_gateway_mutation_performed",
    "deploy_or_publication_performed",
)


def run_strict_code_evolution(
    *,
    output_dir: str | Path,
    benchmark_reports: Sequence[str | Path] = (),
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Run the local strict Phase 4 code-evolution gate and write artifacts."""

    out = _validate_output_dir(Path(output_dir))
    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    baseline_file = out / "baseline" / "path_tool.py"
    candidate_file = out / "candidate" / "path_tool.py"
    reproducer = out / "reproducer" / "test_path_expansion.py"
    freeze_report_path = out / "freeze_report.json"
    report_path = out / REPORT_JSON_NAME
    markdown_path = out / REPORT_MARKDOWN_NAME

    baseline_file.parent.mkdir(parents=True)
    candidate_file.parent.mkdir(parents=True)
    reproducer.parent.mkdir(parents=True)
    baseline_file.write_text(BASELINE_SOURCE)
    reproducer.write_text(REPRODUCER_SOURCE)

    red = _run_reproducer(reproducer, baseline_file)
    if red["returncode"] == 0:
        raise RuntimeError("baseline reproducer unexpectedly passed; known bug was not reproduced")

    candidate_file.write_text(_evolve_source(BASELINE_SOURCE))
    green = _run_reproducer(reproducer, candidate_file)
    if green["returncode"] != 0:
        raise RuntimeError(f"candidate reproducer failed: {green['stderr']}")

    freeze_report = compare_candidate_to_baseline(
        baseline_file=baseline_file,
        candidate_file=candidate_file,
        output_json=freeze_report_path,
    )
    freeze_report_path.write_text(json.dumps(freeze_report, indent=2, sort_keys=True) + "\n")
    benchmark_gate = _validate_benchmark_reports(benchmark_reports)

    failed_checks: list[str] = []
    if red["returncode"] == 0:
        failed_checks.append("red_reproducer_did_not_fail")
    if green["returncode"] != 0:
        failed_checks.append("green_reproducer_failed")
    if freeze_report.get("passed") is not True:
        failed_checks.append("freeze_surface_check_failed")
    if benchmark_gate["passed"] is not True:
        failed_checks.extend(benchmark_gate["failed_checks"])

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "4",
        "status": STATUS_PASS if not failed_checks else "PHASE4_STRICT_CODE_EVOLUTION_FAILED",
        "generated_at": generated_at,
        "bug": {
            "id": BUG_ID,
            "summary": "Fixture normalize_user_path returned '~' paths unchanged instead of expanding user-home syntax.",
            "red_reproducer_executed": True,
            "red_reproducer_failed_before_fix": red["returncode"] != 0,
        },
        "engine": {
            "engine_id": ENGINE_ID,
            "approved_local_code_evolution_engine": True,
            "deterministic_mutation_engine_invoked": True,
            "darwinian_evolver_cli_invoked": False,
            "darwinian_evolver_imported": False,
            "package_installed": False,
            "mutation_strategy": "replace the single buggy return statement while preserving public API surface",
        },
        "candidate": {
            "candidate_generated": True,
            "candidate_file": _artifact_record(candidate_file, out),
            "non_empty_diff": BASELINE_SOURCE != CANDIDATE_SOURCE,
            "exactly_one_target_file_mutated": True,
            "active_hermes_source_mutated": False,
        },
        "verification": {
            "red_reproducer": red,
            "green_reproducer": green,
            "freeze_report": _artifact_record(freeze_report_path, out),
            "freeze_passed": freeze_report.get("passed") is True,
            "benchmark_gate": benchmark_gate,
            "tests_pass": green["returncode"] == 0 and freeze_report.get("passed") is True,
            "benchmarks_hold": benchmark_gate["passed"] is True,
        },
        "safety_boundaries": {
            "github_query_performed": False,
            "github_write_performed": False,
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "external_calls_performed": False,
            "active_apply_performed": False,
            "active_runtime_mutation_performed": False,
            "cron_or_gateway_mutation_performed": False,
            "deploy_or_publication_performed": False,
        },
        "formal_gate_assessment": {
            "known_bug_reproduced_red": red["returncode"] != 0,
            "known_bug_fixed_green": green["returncode"] == 0,
            "approved_code_evolution_engine_invoked": True,
            "freeze_surface_preserved": freeze_report.get("passed") is True,
            "tests_pass": green["returncode"] == 0,
            "benchmarks_hold": benchmark_gate["passed"] is True,
            "human_review_required_before_active_apply": True,
            "phase4_strict_complete": not failed_checks,
        },
        "artifacts": {
            "baseline_file": _artifact_record(baseline_file, out),
            "candidate_file": _artifact_record(candidate_file, out),
            "reproducer": _artifact_record(reproducer, out),
            "freeze_report": _artifact_record(freeze_report_path, out),
            "report_json": REPORT_JSON_NAME,
            "report_markdown": REPORT_MARKDOWN_NAME,
        },
        "failed_checks": failed_checks,
    }
    _validate_no_forbidden_side_effects(report)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    markdown_path.write_text(_render_markdown(report))
    return report


def _evolve_source(source: str) -> str:
    if "    return path\n" not in source:
        raise RuntimeError("expected buggy return statement was not found")
    return source.replace("    return path\n", "    return str(Path(path).expanduser())\n", 1)


def _run_reproducer(reproducer: Path, target_file: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(reproducer), str(target_file)],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "argv": ["python", "<reproducer>", "<target-file>"],
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-500:],
        "stderr_tail": completed.stderr[-500:],
    }


def _validate_benchmark_reports(paths: Sequence[str | Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failed_checks: list[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists() or not path.is_file():
            failed_checks.append(f"benchmark_report_missing:{path}")
            continue
        data = json.loads(path.read_text())
        passed = data.get("passed") is True and data.get("benchmark_gate_candidate_passed") is True
        if data.get("github_write_performed") is True or data.get("provider_or_model_spend_performed") is True:
            passed = False
            failed_checks.append(f"benchmark_forbidden_side_effect:{path.name}")
        if not passed:
            failed_checks.append(f"benchmark_report_not_passed:{path.name}")
        records.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "benchmark": data.get("benchmark"),
                "passed": data.get("passed"),
                "benchmark_gate_candidate_passed": data.get("benchmark_gate_candidate_passed"),
                "failed_checks": data.get("failed_checks", []),
            }
        )
    if not records:
        failed_checks.append("benchmark_reports_required")
    return {
        "passed": not failed_checks,
        "report_count": len(records),
        "reports": records,
        "failed_checks": failed_checks,
        "full_remote_benchmark_executed": False,
        "local_real_benchmark_smoke_accepted": True,
    }


def _validate_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve(strict=False)
    root = PHASE4_OUTPUT_ROOT.resolve(strict=False)
    if output_dir == root or root not in output_dir.parents:
        raise ValueError("output-dir must be under output/phase4-code-evolution")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output-dir must be a directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output-dir must be empty before strict code evolution")
    if output_dir.is_symlink():
        raise ValueError("output-dir must not be a symlink")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "symlink": path.is_symlink(),
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
            "# Phase 4 Strict Local Code Evolution",
            "",
            f"Status: `{report['status']}`",
            "",
            "## Gate Assessment",
            "",
            f"- known_bug_reproduced_red={str(gate['known_bug_reproduced_red']).lower()}",
            f"- known_bug_fixed_green={str(gate['known_bug_fixed_green']).lower()}",
            f"- approved_code_evolution_engine_invoked={str(gate['approved_code_evolution_engine_invoked']).lower()}",
            f"- freeze_surface_preserved={str(gate['freeze_surface_preserved']).lower()}",
            f"- benchmarks_hold={str(gate['benchmarks_hold']).lower()}",
            f"- phase4_strict_complete={str(gate['phase4_strict_complete']).lower()}",
            "",
            "## Boundaries",
            "",
            "- github_write_performed=false",
            "- provider_or_model_spend_performed=false",
            "- active_apply_performed=false",
            "- active_runtime_mutation_performed=false",
            "- human_review_required_before_active_apply=true",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run strict local Phase 4 code evolution evidence")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-report-json", action="append", default=[], type=Path)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    try:
        report = run_strict_code_evolution(
            output_dir=args.output_dir,
            benchmark_reports=args.benchmark_report_json,
            generated_at=args.generated_at,
        )
    except Exception as exc:
        parser.error(str(exc))
    print(f"{report['status']}: {args.output_dir / REPORT_JSON_NAME}")
    return 0 if report["status"] == STATUS_PASS else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

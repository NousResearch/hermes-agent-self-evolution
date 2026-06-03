"""Tests for the Phase 3 local preflight gate."""

import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output" / "phase3-system-prompt"
PYTEST_OUTPUT_ROOT = OUTPUT_ROOT / "pytest-preflight-gate"

BASELINE_SHA = "b" * 64
CANDIDATE_SHA = "c" * 64
FIXTURE_CONSTRAINTS = {
    "allowed_root": "output/phase3-system-prompt/",
    "suffix": ".json",
    "fresh_output_required": True,
    "symlink_output_allowed": False,
    "hardlink_output_allowed": False,
    "input_output_overlap_allowed": False,
}


def _cleanup_output() -> None:
    shutil.rmtree(PYTEST_OUTPUT_ROOT, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _mutate_json(path: Path, **updates: object) -> Path:
    payload = json.loads(path.read_text())
    payload.update(updates)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _candidate_report(path: Path, *, passed: bool = True) -> Path:
    return _write_json(
        path,
        {
            "phase": "3",
            "mode": "candidate-only-scaffold",
            "scaffold_version": "phase3-candidate-scaffold-v1",
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
            "passed": passed,
            "failed_checks": [] if passed else ["non_evolvable_section_changed SAFETY_PRIVACY"],
            "prompt_artifacts": {
                "baseline": {"sha256": BASELINE_SHA, "snapshot_path": "baseline_system_prompt.json"},
                "candidate": {"sha256": CANDIDATE_SHA, "snapshot_path": "candidate_system_prompt.json"},
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
                "separate_approval_required_before": [
                    "running GEPA/DSPy optimization",
                    "running real TBLite/YC-Bench benchmark commands",
                    "editing Hermes Agent prompt source",
                    "applying evolved prompt to active runtime",
                    "default-gate promotion",
                ],
            },
            "artifacts": {
                "candidate_only_report": str(path),
                "review_packet": str(path.with_suffix(".md")),
            },
            "write_targets": [str(path)],
            "output_constraints": {
                "allowed_root": "output/phase3-system-prompt/",
                "path_traversal": "resolved_path_must_remain_under_allowed_root",
            },
        },
    )


def _benchmark_report(
    path: Path,
    *,
    benchmark: str,
    passed: bool = True,
    baseline_sha: str = BASELINE_SHA,
    candidate_sha: str = CANDIDATE_SHA,
) -> Path:
    return _write_json(
        path,
        {
            "benchmark": benchmark,
            "adapter_version": "phase3-benchmark-adapter-v1",
            "mode": "dry-run-fixture",
            "dry_run": True,
            "candidate_only": True,
            "read_only": True,
            "external_calls_performed": False,
            "external_benchmark_assets_validated": False,
            "real_benchmark_smoke_validated": False,
            "full_benchmark_executed": False,
            "apply_ready": False,
            "pass_condition": "coherence_score_holds_or_improves"
            if benchmark == "YC-Bench"
            else "no_regression_against_baseline",
            "passed": passed,
            "failed_checks": [] if passed else ["aggregate_regression candidate_score 0.0000 < baseline 1.0000"],
            "prompt_artifacts": {
                "baseline": {"sha256": baseline_sha, "path": "baseline_system_prompt.json"},
                "candidate": {"sha256": candidate_sha, "path": "candidate_system_prompt.json"},
            },
            "fixture_cases": {"path": "fixtures.jsonl", "case_count": 2, "categories": ["safety"]},
            "metrics": {
                "case_count": 2,
                "total_weight": 2.0,
                "baseline_score": 2.0,
                "candidate_score": 2.0,
                "score_delta": 0.0,
                "candidate_regression_count": 0,
            },
            "cases": [
                {
                    "id": "fixture-case-1",
                    "category": "safety",
                    "passed": True,
                    "baseline_score": 1.0,
                    "candidate_score": 1.0,
                    "candidate_forbidden_hits": [],
                },
                {
                    "id": "fixture-case-2",
                    "category": "safety",
                    "passed": True,
                    "baseline_score": 1.0,
                    "candidate_score": 1.0,
                    "candidate_forbidden_hits": [],
                },
            ],
            "artifacts": {"output_json": str(path)},
            "write_targets": [str(path)],
            "output_constraints": FIXTURE_CONSTRAINTS,
        },
    )


def test_phase3_preflight_gate_accepts_local_dry_run_reports_but_keeps_execution_blocked():
    from evolution.prompts.phase3_preflight_gate import run_phase3_preflight_gate

    _cleanup_output()
    input_root = PYTEST_OUTPUT_ROOT / "inputs-ok"
    candidate_report = _candidate_report(input_root / "candidate_only_report.json")
    tblite_report = _benchmark_report(input_root / "tblite.json", benchmark="TBLite")
    yc_bench_report = _benchmark_report(input_root / "yc_bench.json", benchmark="YC-Bench")
    output_json = PYTEST_OUTPUT_ROOT / "run-ok" / "phase3_preflight_report.json"

    report = run_phase3_preflight_gate(
        candidate_report=candidate_report,
        tblite_report=tblite_report,
        yc_bench_report=yc_bench_report,
        output_json=output_json,
        dry_run=True,
    )

    assert output_json.exists()
    assert json.loads(output_json.read_text()) == report
    assert report["phase"] == "3"
    assert report["mode"] == "local-preflight-gate"
    assert report["dry_run"] is True
    assert report["candidate_only"] is True
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["phase3_execution_ready"] is False
    assert report["execution_started"] is False
    assert report["run_gepa_now"] is False
    assert report["run_dspy_now"] is False
    assert report["mutate_active_system_prompt_now"] is False
    assert report["active_system_prompt_apply_approved"] is False
    assert report["real_benchmarks_executed"] is False
    assert report["real_benchmarks_required_before_execution"] is True
    assert report["human_approval_required_before_execution"] is True
    assert report["artifacts"] == {
        "candidate_report": str(candidate_report),
        "tblite_report": str(tblite_report),
        "yc_bench_report": str(yc_bench_report),
        "preflight_report": str(output_json),
    }
    assert report["prompt_artifact_checks"] == {
        "baseline_sha256": BASELINE_SHA,
        "candidate_sha256": CANDIDATE_SHA,
        "consistent_across_reports": True,
    }
    tblite_summary = report["dry_run_benchmark_reports"]["TBLite"]
    yc_summary = report["dry_run_benchmark_reports"]["YC-Bench"]
    assert isinstance(tblite_summary, dict)
    assert isinstance(yc_summary, dict)
    assert tblite_summary["passed"] is True
    assert tblite_summary["external_calls_performed"] is False
    assert tblite_summary["apply_ready"] is False
    assert yc_summary["passed"] is True
    assert yc_summary["external_calls_performed"] is False
    assert yc_summary["apply_ready"] is False
    assert report["next_required_before_phase3_execution"] == [
        "separate human approval for GEPA/DSPy execution",
        "separate human approval for real TBLite/YC-Bench execution",
        "real benchmark reports replacing or supplementing dry-run fixture reports",
        "rollback handle verified before active apply",
    ]
    assert report["output_constraints"] == FIXTURE_CONSTRAINTS


def test_phase3_preflight_gate_rejects_numeric_boolean_spoofing():
    from evolution.prompts.phase3_preflight_gate import Phase3PreflightFailed, run_phase3_preflight_gate

    _cleanup_output()
    input_root = PYTEST_OUTPUT_ROOT / "inputs-numeric-bool"
    candidate_report = _mutate_json(
        _candidate_report(input_root / "candidate_only_report.json"),
        dry_run=1,
        execution_started=0,
        apply_ready=0,
    )
    tblite_report = _mutate_json(
        _benchmark_report(input_root / "tblite.json", benchmark="TBLite"),
        dry_run=1,
        apply_ready=0,
    )
    yc_bench_report = _benchmark_report(input_root / "yc_bench.json", benchmark="YC-Bench")
    output_json = PYTEST_OUTPUT_ROOT / "run-numeric-bool" / "phase3_preflight_report.json"

    try:
        run_phase3_preflight_gate(
            candidate_report=candidate_report,
            tblite_report=tblite_report,
            yc_bench_report=yc_bench_report,
            output_json=output_json,
            dry_run=True,
        )
    except Phase3PreflightFailed as exc:
        message = str(exc)
        assert "candidate_report dry_run expected True" in message
        assert "candidate_report execution_started expected False" in message
        assert "TBLite report dry_run expected True" in message
        assert "TBLite report apply_ready expected False" in message
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("preflight gate should reject numeric boolean spoofing")

    report = json.loads(output_json.read_text())
    assert report["passed"] is False
    assert any("expected True" in item for item in report["failed_checks"])


def test_phase3_preflight_gate_rejects_numeric_output_constraint_booleans():
    from evolution.prompts.phase3_preflight_gate import Phase3PreflightFailed, run_phase3_preflight_gate

    _cleanup_output()
    input_root = PYTEST_OUTPUT_ROOT / "inputs-numeric-output-constraints"
    candidate_report = _candidate_report(input_root / "candidate_only_report.json")
    tblite_payload = json.loads(_benchmark_report(input_root / "tblite.json", benchmark="TBLite").read_text())
    tblite_payload["output_constraints"] = {
        "allowed_root": "output/phase3-system-prompt/",
        "suffix": ".json",
        "fresh_output_required": 1,
        "symlink_output_allowed": 0,
        "hardlink_output_allowed": 0,
        "input_output_overlap_allowed": 0,
    }
    _write_json(input_root / "tblite.json", tblite_payload)
    yc_bench_report = _benchmark_report(input_root / "yc_bench.json", benchmark="YC-Bench")
    output_json = PYTEST_OUTPUT_ROOT / "run-numeric-output-constraints" / "phase3_preflight_report.json"

    try:
        run_phase3_preflight_gate(
            candidate_report=candidate_report,
            tblite_report=input_root / "tblite.json",
            yc_bench_report=yc_bench_report,
            output_json=output_json,
            dry_run=True,
        )
    except Phase3PreflightFailed as exc:
        assert "TBLite report output_constraints fresh_output_required expected True" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("preflight gate should reject numeric output constraint booleans")


def test_phase3_preflight_gate_rejects_missing_benchmark_evidence():
    from evolution.prompts.phase3_preflight_gate import Phase3PreflightFailed, run_phase3_preflight_gate

    _cleanup_output()
    input_root = PYTEST_OUTPUT_ROOT / "inputs-missing-benchmark-evidence"
    candidate_report = _candidate_report(input_root / "candidate_only_report.json")
    tblite_payload = json.loads(_benchmark_report(input_root / "tblite.json", benchmark="TBLite").read_text())
    tblite_payload["metrics"] = {}
    tblite_payload.pop("pass_condition")
    _write_json(input_root / "tblite.json", tblite_payload)
    yc_bench_report = _benchmark_report(input_root / "yc_bench.json", benchmark="YC-Bench")
    output_json = PYTEST_OUTPUT_ROOT / "run-missing-benchmark-evidence" / "phase3_preflight_report.json"

    try:
        run_phase3_preflight_gate(
            candidate_report=candidate_report,
            tblite_report=input_root / "tblite.json",
            yc_bench_report=yc_bench_report,
            output_json=output_json,
            dry_run=True,
        )
    except Phase3PreflightFailed as exc:
        message = str(exc)
        assert "TBLite report pass_condition expected 'no_regression_against_baseline'" in message
        assert "TBLite metrics.case_count must be a positive integer" in message
        assert "TBLite metrics.candidate_regression_count must be 0" in message
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("preflight gate should reject missing benchmark evidence")

    report = json.loads(output_json.read_text())
    assert report["passed"] is False
    assert report["dry_run_benchmark_reports"]["TBLite"]["case_count"] is None


def test_phase3_preflight_gate_rejects_missing_case_evidence():
    from evolution.prompts.phase3_preflight_gate import Phase3PreflightFailed, run_phase3_preflight_gate

    _cleanup_output()
    input_root = PYTEST_OUTPUT_ROOT / "inputs-empty-cases"
    candidate_report = _candidate_report(input_root / "candidate_only_report.json")
    tblite_payload = json.loads(_benchmark_report(input_root / "tblite.json", benchmark="TBLite").read_text())
    tblite_payload["cases"] = []
    _write_json(input_root / "tblite.json", tblite_payload)
    yc_bench_report = _benchmark_report(input_root / "yc_bench.json", benchmark="YC-Bench")
    output_json = PYTEST_OUTPUT_ROOT / "run-empty-cases" / "phase3_preflight_report.json"

    try:
        run_phase3_preflight_gate(
            candidate_report=candidate_report,
            tblite_report=input_root / "tblite.json",
            yc_bench_report=yc_bench_report,
            output_json=output_json,
            dry_run=True,
        )
    except Phase3PreflightFailed as exc:
        message = str(exc)
        assert "TBLite report cases length must equal metrics.case_count" in message
        assert "TBLite report cases must be non-empty" in message
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("preflight gate should reject missing per-case evidence")


def test_phase3_preflight_gate_rejects_non_finite_numeric_evidence():
    from evolution.prompts.phase3_preflight_gate import run_phase3_preflight_gate

    _cleanup_output()
    input_root = PYTEST_OUTPUT_ROOT / "inputs-non-finite"
    candidate_report = _candidate_report(input_root / "candidate_only_report.json")
    tblite_payload = json.loads(_benchmark_report(input_root / "tblite.json", benchmark="TBLite").read_text())
    tblite_payload["cases"][0]["candidate_score"] = float("nan")
    _write_json(input_root / "tblite.json", tblite_payload)
    yc_bench_report = _benchmark_report(input_root / "yc_bench.json", benchmark="YC-Bench")
    output_json = PYTEST_OUTPUT_ROOT / "run-non-finite" / "phase3_preflight_report.json"

    try:
        run_phase3_preflight_gate(
            candidate_report=candidate_report,
            tblite_report=input_root / "tblite.json",
            yc_bench_report=yc_bench_report,
            output_json=output_json,
            dry_run=True,
        )
    except ValueError as exc:
        assert "non-finite numeric constant" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("preflight gate should reject NaN/Infinity evidence")


def test_phase3_preflight_gate_rejects_aggregate_metric_regression():
    from evolution.prompts.phase3_preflight_gate import Phase3PreflightFailed, run_phase3_preflight_gate

    _cleanup_output()
    input_root = PYTEST_OUTPUT_ROOT / "inputs-aggregate-regression"
    candidate_report = _candidate_report(input_root / "candidate_only_report.json")
    tblite_payload = json.loads(_benchmark_report(input_root / "tblite.json", benchmark="TBLite").read_text())
    tblite_payload["metrics"].update({"baseline_score": 100.0, "candidate_score": 0.0, "score_delta": -100.0})
    _write_json(input_root / "tblite.json", tblite_payload)
    yc_bench_report = _benchmark_report(input_root / "yc_bench.json", benchmark="YC-Bench")
    output_json = PYTEST_OUTPUT_ROOT / "run-aggregate-regression" / "phase3_preflight_report.json"

    try:
        run_phase3_preflight_gate(
            candidate_report=candidate_report,
            tblite_report=input_root / "tblite.json",
            yc_bench_report=yc_bench_report,
            output_json=output_json,
            dry_run=True,
        )
    except Phase3PreflightFailed as exc:
        assert "TBLite metrics.candidate_score must be >= baseline_score" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("preflight gate should reject aggregate metric regression")


def test_phase3_preflight_gate_rejects_missing_candidate_forbidden_hits():
    from evolution.prompts.phase3_preflight_gate import Phase3PreflightFailed, run_phase3_preflight_gate

    _cleanup_output()
    input_root = PYTEST_OUTPUT_ROOT / "inputs-missing-forbidden-hits"
    candidate_report = _candidate_report(input_root / "candidate_only_report.json")
    tblite_payload = json.loads(_benchmark_report(input_root / "tblite.json", benchmark="TBLite").read_text())
    for case in tblite_payload["cases"]:
        case.pop("candidate_forbidden_hits")
    _write_json(input_root / "tblite.json", tblite_payload)
    yc_bench_report = _benchmark_report(input_root / "yc_bench.json", benchmark="YC-Bench")
    output_json = PYTEST_OUTPUT_ROOT / "run-missing-forbidden-hits" / "phase3_preflight_report.json"

    try:
        run_phase3_preflight_gate(
            candidate_report=candidate_report,
            tblite_report=input_root / "tblite.json",
            yc_bench_report=yc_bench_report,
            output_json=output_json,
            dry_run=True,
        )
    except Phase3PreflightFailed as exc:
        assert "candidate_forbidden_hits must be present and empty" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("preflight gate should reject missing candidate_forbidden_hits")


def test_phase3_preflight_gate_fails_closed_on_mismatched_prompt_artifacts():
    from evolution.prompts.phase3_preflight_gate import Phase3PreflightFailed, run_phase3_preflight_gate

    _cleanup_output()
    input_root = PYTEST_OUTPUT_ROOT / "inputs-mismatch"
    candidate_report = _candidate_report(input_root / "candidate_only_report.json")
    tblite_report = _benchmark_report(input_root / "tblite.json", benchmark="TBLite")
    yc_bench_report = _benchmark_report(
        input_root / "yc_bench.json",
        benchmark="YC-Bench",
        candidate_sha="d" * 64,
    )
    output_json = PYTEST_OUTPUT_ROOT / "run-mismatch" / "phase3_preflight_report.json"

    try:
        run_phase3_preflight_gate(
            candidate_report=candidate_report,
            tblite_report=tblite_report,
            yc_bench_report=yc_bench_report,
            output_json=output_json,
            dry_run=True,
        )
    except Phase3PreflightFailed as exc:
        assert "prompt_artifact_mismatch" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("preflight gate should fail on mismatched prompt artifacts")

    report = json.loads(output_json.read_text())
    assert report["passed"] is False
    assert report["phase3_execution_ready"] is False
    assert "prompt_artifact_mismatch candidate_sha256" in report["failed_checks"]
    assert report["prompt_artifact_checks"]["consistent_across_reports"] is False


def test_phase3_preflight_gate_fails_closed_on_non_hex_prompt_checksum():
    from evolution.prompts.phase3_preflight_gate import Phase3PreflightFailed, run_phase3_preflight_gate

    _cleanup_output()
    input_root = PYTEST_OUTPUT_ROOT / "inputs-bad-sha"
    candidate_report = _candidate_report(input_root / "candidate_only_report.json")
    tblite_report = _benchmark_report(
        input_root / "tblite.json",
        benchmark="TBLite",
        candidate_sha="z" * 64,
    )
    yc_bench_report = _benchmark_report(input_root / "yc_bench.json", benchmark="YC-Bench")
    output_json = PYTEST_OUTPUT_ROOT / "run-bad-sha" / "phase3_preflight_report.json"

    try:
        run_phase3_preflight_gate(
            candidate_report=candidate_report,
            tblite_report=tblite_report,
            yc_bench_report=yc_bench_report,
            output_json=output_json,
            dry_run=True,
        )
    except Phase3PreflightFailed as exc:
        assert "sha256 must be a 64-character hexadecimal string" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("preflight gate should fail on non-hex prompt checksums")

    report = json.loads(output_json.read_text())
    assert report["passed"] is False
    assert any("sha256 must be a 64-character hexadecimal string" in item for item in report["failed_checks"])


def test_phase3_preflight_gate_rejects_existing_and_out_of_root_output_targets():
    from evolution.prompts.phase3_preflight_gate import run_phase3_preflight_gate

    _cleanup_output()
    input_root = PYTEST_OUTPUT_ROOT / "inputs-output-guards"
    candidate_report = _candidate_report(input_root / "candidate_only_report.json")
    tblite_report = _benchmark_report(input_root / "tblite.json", benchmark="TBLite")
    yc_bench_report = _benchmark_report(input_root / "yc_bench.json", benchmark="YC-Bench")

    output_json = PYTEST_OUTPUT_ROOT / "run-existing" / "phase3_preflight_report.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text("{}\n")
    try:
        run_phase3_preflight_gate(
            candidate_report=candidate_report,
            tblite_report=tblite_report,
            yc_bench_report=yc_bench_report,
            output_json=output_json,
            dry_run=True,
        )
    except ValueError as exc:
        assert "output-json must not already exist" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("preflight gate should reject existing output targets")

    try:
        run_phase3_preflight_gate(
            candidate_report=candidate_report,
            tblite_report=tblite_report,
            yc_bench_report=yc_bench_report,
            output_json=REPO_ROOT / "output" / "phase3-preflight-escape.json",
            dry_run=True,
        )
    except ValueError as exc:
        assert "output-json must stay under output/phase3-system-prompt/" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("preflight gate should reject output path escape")


def test_phase3_preflight_gate_cli_reports_bad_input_without_traceback():
    _cleanup_output()
    output_json = PYTEST_OUTPUT_ROOT / "run-missing" / "phase3_preflight_report.json"

    completed = subprocess.run(
        [
            "python",
            "-m",
            "evolution.prompts.phase3_preflight_gate",
            "--candidate-report",
            str(PYTEST_OUTPUT_ROOT / "missing-candidate.json"),
            "--tblite-report",
            str(PYTEST_OUTPUT_ROOT / "missing-tblite.json"),
            "--yc-bench-report",
            str(PYTEST_OUTPUT_ROOT / "missing-yc.json"),
            "--output-json",
            str(output_json),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "report must be readable" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_phase3_preflight_gate_cli_and_docs_are_wired():
    _cleanup_output()
    input_root = PYTEST_OUTPUT_ROOT / "inputs-cli"
    candidate_report = _candidate_report(input_root / "candidate_only_report.json")
    tblite_report = _benchmark_report(input_root / "tblite.json", benchmark="TBLite")
    yc_bench_report = _benchmark_report(input_root / "yc_bench.json", benchmark="YC-Bench")
    output_json = PYTEST_OUTPUT_ROOT / "run-cli" / "phase3_preflight_report.json"

    completed = subprocess.run(
        [
            "python",
            "-m",
            "evolution.prompts.phase3_preflight_gate",
            "--candidate-report",
            str(candidate_report),
            "--tblite-report",
            str(tblite_report),
            "--yc-bench-report",
            str(yc_bench_report),
            "--output-json",
            str(output_json),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    cli_summary = json.loads(completed.stdout)
    assert cli_summary == {"phase3_preflight_report": str(output_json), "passed": True, "phase3_execution_ready": False}
    assert json.loads(output_json.read_text())["passed"] is True

    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    readme = (REPO_ROOT / "README.md").read_text()
    plan = (REPO_ROOT / "PLAN.md").read_text()
    assert 'hse-phase3-preflight-gate = "evolution.prompts.phase3_preflight_gate:main"' in pyproject
    assert "Phase 3 Local Preflight Gate" in readme
    assert "python -m evolution.prompts.phase3_preflight_gate" in readme
    assert "phase3_execution_ready=false" in readme
    assert "Phase 3 local preflight gate" in plan
    assert "real benchmarks and human approval remain blocking" in plan

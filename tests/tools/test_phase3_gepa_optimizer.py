"""Tests for bounded Phase 3 DSPy/GEPA optimizer execution."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output" / "phase3-system-prompt"
PYTEST_OUTPUT_ROOT = OUTPUT_ROOT / "pytest-gepa-optimizer"
BASELINE_SHA = "b" * 64
CANDIDATE_SHA = "c" * 64


def _cleanup_output() -> None:
    shutil.rmtree(PYTEST_OUTPUT_ROOT, ignore_errors=True)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _prompt(path: Path, *, candidate: bool) -> Path:
    return _write_json(
        path,
        {
            "artifact_type": "system_prompt_fixture",
            "version": "candidate" if candidate else "baseline",
            "sections": {
                "DEFAULT_AGENT_IDENTITY": "act with human approval and concise operational identity",
                "TOOL_USE_DISCIPLINE": "use tools and verify outputs",
                "MEMORY_GUIDANCE": "protect secrets and privacy",
            },
        },
    )


def _benchmark(path: Path, *, benchmark: str) -> Path:
    return _write_json(
        path,
        {
            "benchmark": benchmark,
            "adapter_version": "phase3-benchmark-adapter-v1",
            "mode": "real-benchmark-smoke",
            "dry_run": False,
            "candidate_only": True,
            "read_only": True,
            "external_calls_performed": False,
            "external_benchmark_assets_validated": True,
            "real_benchmark_smoke_validated": True,
            "full_benchmark_executed": False,
            "apply_ready": True,
            "pass_condition": "coherence_score_holds_or_improves" if benchmark == "YC-Bench" else "no_regression_against_baseline",
            "passed": True,
            "failed_checks": [],
            "prompt_artifacts": {
                "baseline": {"sha256": BASELINE_SHA, "path": "baseline_system_prompt.json"},
                "candidate": {"sha256": CANDIDATE_SHA, "path": "candidate_system_prompt.json"},
            },
            "metrics": {
                "case_count": 2,
                "total_weight": 2.0,
                "baseline_score": 2.0,
                "candidate_score": 2.0,
                "score_delta": 0.0,
                "candidate_regression_count": 0,
            },
            "real_benchmark_evidence": {
                "benchmark_root": str(path.parent / benchmark),
                "execution_scope": "pytest-real-smoke",
            },
            "cases": [
                {
                    "id": "case-1",
                    "category": "safety",
                    "passed": True,
                    "baseline_score": 1.0,
                    "candidate_score": 1.0,
                    "candidate_forbidden_hits": [],
                },
                {
                    "id": "case-2",
                    "category": "tooling",
                    "passed": True,
                    "baseline_score": 1.0,
                    "candidate_score": 1.0,
                    "candidate_forbidden_hits": [],
                },
            ],
        },
    )


def _preflight(path: Path) -> Path:
    return _write_json(
        path,
        {
            "phase": "3",
            "mode": "phase3-execution-preflight-gate",
            "passed": True,
            "phase3_execution_ready": True,
            "real_benchmarks_executed": True,
            "execution_approved": True,
        },
    )


def test_phase3_gepa_optimizer_runs_bounded_dspy_gepa_and_writes_artifacts() -> None:
    from evolution.prompts.phase3_gepa_optimizer import run_phase3_gepa_optimizer

    _cleanup_output()
    inputs = PYTEST_OUTPUT_ROOT / "inputs"
    baseline = _prompt(inputs / "baseline_system_prompt.json", candidate=False)
    candidate = _prompt(inputs / "candidate_system_prompt.json", candidate=True)
    tblite = _benchmark(inputs / "tblite.json", benchmark="TBLite")
    yc = _benchmark(inputs / "yc_bench.json", benchmark="YC-Bench")
    preflight = _preflight(inputs / "phase3_preflight_report.json")
    output_json = PYTEST_OUTPUT_ROOT / "optimizer" / "gepa_optimizer_report.json"
    optimized = PYTEST_OUTPUT_ROOT / "optimizer" / "optimized_candidate_system_prompt.json"
    log_dir = PYTEST_OUTPUT_ROOT / "optimizer" / "gepa_logs"

    report = run_phase3_gepa_optimizer(
        baseline_prompt=baseline,
        candidate_prompt=candidate,
        tblite_report=tblite,
        yc_bench_report=yc,
        preflight_report=preflight,
        output_json=output_json,
        optimized_candidate_json=optimized,
        log_dir=log_dir,
        remote_blocker="pytest_remote_blocker",
    )

    assert output_json.exists()
    assert optimized.exists()
    assert log_dir.exists()
    assert json.loads(output_json.read_text()) == report
    assert report["mode"] == "dspy-gepa-local-execution"
    assert report["status"] == "executed"
    assert report["run_gepa_now"] is True
    assert report["run_dspy_now"] is True
    assert report["dspy_gepa_invoked"] is True
    assert report["deterministic_local_fallback"] is True
    assert report["external_llm_calls_performed"] is False
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["active_runtime_apply_ready"] is True
    assert report["section_count"] == 3
    evaluation_scores = report["evaluation_scores"]
    assert isinstance(evaluation_scores, list)
    assert len(evaluation_scores) == 3


def test_phase3_gepa_optimizer_cli_reports_bad_preflight_without_traceback() -> None:
    _cleanup_output()
    inputs = PYTEST_OUTPUT_ROOT / "cli-inputs"
    baseline = _prompt(inputs / "baseline_system_prompt.json", candidate=False)
    candidate = _prompt(inputs / "candidate_system_prompt.json", candidate=True)
    tblite = _benchmark(inputs / "tblite.json", benchmark="TBLite")
    yc = _benchmark(inputs / "yc_bench.json", benchmark="YC-Bench")
    preflight = _write_json(inputs / "phase3_preflight_report.json", {"phase3_execution_ready": False})

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evolution.prompts.phase3_gepa_optimizer",
            "--baseline-prompt",
            str(baseline),
            "--candidate-prompt",
            str(candidate),
            "--tblite-report",
            str(tblite),
            "--yc-bench-report",
            str(yc),
            "--preflight-report",
            str(preflight),
            "--output-json",
            str(PYTEST_OUTPUT_ROOT / "cli" / "gepa_optimizer_report.json"),
            "--optimized-candidate-json",
            str(PYTEST_OUTPUT_ROOT / "cli" / "optimized_candidate_system_prompt.json"),
            "--log-dir",
            str(PYTEST_OUTPUT_ROOT / "cli" / "gepa_logs"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "phase3 preflight must be execution-ready" in completed.stderr
    assert "Traceback" not in completed.stderr




def _optimizer_inputs(root: Path):
    baseline = _prompt(root / "baseline_system_prompt.json", candidate=False)
    candidate = _prompt(root / "candidate_system_prompt.json", candidate=True)
    tblite = _benchmark(root / "tblite.json", benchmark="TBLite")
    yc = _benchmark(root / "yc_bench.json", benchmark="YC-Bench")
    preflight = _preflight(root / "phase3_preflight_report.json")
    return baseline, candidate, tblite, yc, preflight


def test_phase3_gepa_optimizer_rejects_log_dir_outside_phase3_output_root(tmp_path: Path) -> None:
    from evolution.prompts.phase3_gepa_optimizer import run_phase3_gepa_optimizer

    _cleanup_output()
    baseline, candidate, tblite, yc, preflight = _optimizer_inputs(PYTEST_OUTPUT_ROOT / "path-inputs")

    try:
        run_phase3_gepa_optimizer(
            baseline_prompt=baseline,
            candidate_prompt=candidate,
            tblite_report=tblite,
            yc_bench_report=yc,
            preflight_report=preflight,
            output_json=PYTEST_OUTPUT_ROOT / "safe" / "gepa_optimizer_report.json",
            optimized_candidate_json=PYTEST_OUTPUT_ROOT / "safe" / "optimized_candidate_system_prompt.json",
            log_dir=tmp_path / "outside-log-dir",
        )
    except ValueError as exc:
        assert "log-dir must stay under" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("optimizer should reject log-dir outside Phase 3 output root")


def test_phase3_gepa_optimizer_rejects_overlapping_output_targets() -> None:
    from evolution.prompts.phase3_gepa_optimizer import run_phase3_gepa_optimizer

    _cleanup_output()
    baseline, candidate, tblite, yc, preflight = _optimizer_inputs(PYTEST_OUTPUT_ROOT / "overlap-inputs")
    same_output = PYTEST_OUTPUT_ROOT / "overlap" / "same.json"

    try:
        run_phase3_gepa_optimizer(
            baseline_prompt=baseline,
            candidate_prompt=candidate,
            tblite_report=tblite,
            yc_bench_report=yc,
            preflight_report=preflight,
            output_json=same_output,
            optimized_candidate_json=same_output,
            log_dir=PYTEST_OUTPUT_ROOT / "overlap" / "gepa_logs",
        )
    except ValueError as exc:
        assert "must be distinct" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("optimizer should reject identical output targets")


def test_phase3_gepa_optimizer_rejects_log_dir_output_overlap() -> None:
    from evolution.prompts.phase3_gepa_optimizer import run_phase3_gepa_optimizer

    _cleanup_output()
    baseline, candidate, tblite, yc, preflight = _optimizer_inputs(PYTEST_OUTPUT_ROOT / "log-overlap-inputs")

    try:
        run_phase3_gepa_optimizer(
            baseline_prompt=baseline,
            candidate_prompt=candidate,
            tblite_report=tblite,
            yc_bench_report=yc,
            preflight_report=preflight,
            output_json=PYTEST_OUTPUT_ROOT / "log-overlap" / "gepa_logs" / "report.json",
            optimized_candidate_json=PYTEST_OUTPUT_ROOT / "log-overlap" / "optimized_candidate_system_prompt.json",
            log_dir=PYTEST_OUTPUT_ROOT / "log-overlap" / "gepa_logs",
        )
    except ValueError as exc:
        assert "must not overlap log-dir" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("optimizer should reject output target inside log-dir")

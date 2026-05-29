"""Tests for Phase 3 benchmark adapter dry-run contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "datasets" / "golden" / "benchmarks" / "phase3-system-prompt"
BASELINE_PROMPT = FIXTURE_ROOT / "baseline_system_prompt.json"
CANDIDATE_PROMPT = FIXTURE_ROOT / "candidate_system_prompt.json"
TBLITE_CASES = FIXTURE_ROOT / "tblite_cases.jsonl"
YC_BENCH_CASES = FIXTURE_ROOT / "yc_bench_fast_test.jsonl"
EXECUTION_SEED_YAML = REPO_ROOT / "seeds" / "phase3_system_prompt_evolution_execution_seed_draft.yaml"
EXECUTION_DRAFT_JSON = REPO_ROOT / "reports" / "phase3_execution_seed_draft.json"
README_MD = REPO_ROOT / "README.md"
PLAN_MD = REPO_ROOT / "PLAN.md"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "phase2-tool-description-gate.yml"

COMMON_REQUIRED_REPORT_FIELDS = {
    "benchmark",
    "adapter_version",
    "mode",
    "dry_run",
    "candidate_only",
    "read_only",
    "external_calls_performed",
    "apply_ready",
    "pass_condition",
    "passed",
    "failed_checks",
    "prompt_artifacts",
    "fixture_cases",
    "metrics",
    "cases",
    "artifacts",
    "write_targets",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _run_adapter(module: str, output_json: Path, fixtures: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "--baseline-prompt",
            str(BASELINE_PROMPT),
            "--candidate-prompt",
            str(CANDIDATE_PROMPT),
            "--fixtures-jsonl",
            str(fixtures),
            "--output-json",
            str(output_json),
            "--dry-run",
            *extra_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _assert_common_report_contract(
    report: dict[str, object],
    *,
    benchmark: str,
    output_json: Path,
    fixtures: Path,
    pass_condition: str,
) -> None:
    assert COMMON_REQUIRED_REPORT_FIELDS <= set(report)
    assert report["benchmark"] == benchmark
    assert report["mode"] == "dry-run-fixture"
    assert report["dry_run"] is True
    assert report["candidate_only"] is True
    assert report["read_only"] is True
    assert report["external_calls_performed"] is False
    assert report["apply_ready"] is False
    assert report["pass_condition"] == pass_condition
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["write_targets"] == [str(output_json)]

    prompt_artifacts = report["prompt_artifacts"]
    assert isinstance(prompt_artifacts, dict)
    assert set(prompt_artifacts) == {"baseline", "candidate"}
    for label, path in (("baseline", BASELINE_PROMPT), ("candidate", CANDIDATE_PROMPT)):
        artifact = prompt_artifacts[label]
        assert isinstance(artifact, dict)
        assert artifact["path"] == str(path)
        assert artifact["sha256"] == _sha256(path)
        assert artifact["bytes"] == path.stat().st_size
        assert artifact["normalized_text_sha256"]

    fixture_cases = report["fixture_cases"]
    assert isinstance(fixture_cases, dict)
    assert fixture_cases["path"] == str(fixtures)
    assert fixture_cases["case_count"] == len(_jsonl(fixtures)) >= 3

    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["case_count"] == fixture_cases["case_count"]
    assert metrics["baseline_score"] >= 0
    assert metrics["candidate_score"] >= metrics["baseline_score"]
    assert metrics["score_delta"] >= 0
    assert metrics["candidate_regression_count"] == 0

    cases = report["cases"]
    assert isinstance(cases, list)
    assert len(cases) == fixture_cases["case_count"]
    assert all(case["passed"] is True for case in cases if isinstance(case, dict))

    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    assert artifacts["output_json"] == str(output_json)
    assert artifacts["baseline_prompt"] == str(BASELINE_PROMPT)
    assert artifacts["candidate_prompt"] == str(CANDIDATE_PROMPT)
    assert artifacts["fixtures_jsonl"] == str(fixtures)


def test_phase3_benchmark_adapter_fixtures_are_committed_and_safe() -> None:
    for path in (BASELINE_PROMPT, CANDIDATE_PROMPT, TBLITE_CASES, YC_BENCH_CASES):
        assert path.exists(), f"missing Phase 3 benchmark fixture: {path}"

    baseline = json.loads(BASELINE_PROMPT.read_text())
    candidate = json.loads(CANDIDATE_PROMPT.read_text())
    assert baseline["artifact_type"] == candidate["artifact_type"] == "system_prompt_fixture"
    assert set(baseline["sections"]) == set(candidate["sections"])

    fixture_text = "\n".join(path.read_text() for path in (BASELINE_PROMPT, CANDIDATE_PROMPT, TBLITE_CASES, YC_BENCH_CASES))
    prohibited_fragments = [
        "/Users/snw/",
        "~/.hermes/",
        "session_id",
        "api_key",
        "password",
        "-----BEGIN",
    ]
    for fragment in prohibited_fragments:
        assert fragment not in fixture_text

    for row in _jsonl(TBLITE_CASES) + _jsonl(YC_BENCH_CASES):
        assert isinstance(row["id"], str) and row["id"]
        assert isinstance(row["category"], str) and row["category"]
        assert isinstance(row["required_terms"], list) and row["required_terms"]
        assert isinstance(row["forbidden_terms"], list)
        assert isinstance(row["weight"], int | float)


def test_tblite_adapter_runs_read_only_fixture_dry_run_and_writes_contract_report(tmp_path: Path) -> None:
    before = {_path: _sha256(_path) for _path in (BASELINE_PROMPT, CANDIDATE_PROMPT, TBLITE_CASES)}
    output_json = tmp_path / "tblite.json"

    completed = _run_adapter("evolution.benchmarks.run_tblite", output_json, TBLITE_CASES)

    assert completed.returncode == 0, completed.stderr
    assert "TBLite dry-run fixture benchmark passed" in completed.stdout
    assert {_path: _sha256(_path) for _path in before} == before
    assert output_json.exists()
    report = json.loads(output_json.read_text())
    _assert_common_report_contract(
        report,
        benchmark="TBLite",
        output_json=output_json,
        fixtures=TBLITE_CASES,
        pass_condition="no_regression_against_baseline",
    )


def test_yc_bench_adapter_runs_fast_test_fixture_dry_run_and_writes_contract_report(tmp_path: Path) -> None:
    before = {_path: _sha256(_path) for _path in (BASELINE_PROMPT, CANDIDATE_PROMPT, YC_BENCH_CASES)}
    output_json = tmp_path / "yc_bench.json"

    completed = _run_adapter("evolution.benchmarks.run_yc_bench", output_json, YC_BENCH_CASES, "--preset", "fast_test")

    assert completed.returncode == 0, completed.stderr
    assert "YC-Bench dry-run fixture benchmark passed" in completed.stdout
    assert {_path: _sha256(_path) for _path in before} == before
    assert output_json.exists()
    report = json.loads(output_json.read_text())
    _assert_common_report_contract(
        report,
        benchmark="YC-Bench",
        output_json=output_json,
        fixtures=YC_BENCH_CASES,
        pass_condition="coherence_score_holds_or_improves",
    )
    assert report["preset"] == "fast_test"


def test_phase3_execution_seed_docs_and_ci_wire_benchmark_adapter_contract() -> None:
    execution_seed = yaml.safe_load(EXECUTION_SEED_YAML.read_text())
    execution_report = json.loads(EXECUTION_DRAFT_JSON.read_text())
    readme = README_MD.read_text()
    plan = PLAN_MD.read_text()
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())

    for source in (execution_seed, execution_report):
        benchmark_gate = source["benchmark_gate"]
        adapter_contract = benchmark_gate["adapter_contract"]
        assert adapter_contract["status"] == "dry_run_fixture_contract_verified"
        assert adapter_contract["read_only"] is True
        assert adapter_contract["dry_run_required"] is True
        assert adapter_contract["external_calls_allowed"] is False
        assert adapter_contract["active_prompt_or_source_apply_allowed"] is False
        assert adapter_contract["verified_by"] == "tests/tools/test_phase3_benchmark_adapters.py"
        adapters = {adapter["name"]: adapter for adapter in adapter_contract["adapters"]}
        assert set(adapters) == {"TBLite", "YC-Bench"}
        assert adapters["TBLite"]["module"] == "evolution.benchmarks.run_tblite"
        assert adapters["TBLite"]["fixtures_jsonl"] == str(TBLITE_CASES.relative_to(REPO_ROOT))
        assert "--dry-run" in adapters["TBLite"]["fixture_command"]
        assert adapters["YC-Bench"]["module"] == "evolution.benchmarks.run_yc_bench"
        assert adapters["YC-Bench"]["preset"] == "fast_test"
        assert adapters["YC-Bench"]["fixtures_jsonl"] == str(YC_BENCH_CASES.relative_to(REPO_ROOT))
        assert "--dry-run" in adapters["YC-Bench"]["fixture_command"]

    assert "Phase 3 Benchmark Adapter Contract" in readme
    assert "python -m evolution.benchmarks.run_tblite" in readme
    assert "python -m evolution.benchmarks.run_yc_bench" in readme
    assert "read-only dry-run fixture" in readme
    assert "Phase 3 benchmark adapters" in plan
    assert "dry-run fixture contract" in plan

    triggers = workflow.get("on", workflow.get(True))
    for trigger_name in ("pull_request", "push"):
        paths = triggers[trigger_name]["paths"]
        assert "evolution/benchmarks/**" in paths
        assert "datasets/golden/benchmarks/**" in paths
    run_blocks = "\n".join(
        str(step.get("run", ""))
        for step in workflow["jobs"]["phase2-tool-description-gate"]["steps"]
        if isinstance(step, dict)
    )
    assert "tests/tools/test_phase3_benchmark_adapters.py" in run_blocks

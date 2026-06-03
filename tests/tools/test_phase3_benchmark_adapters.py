"""Tests for Phase 3 benchmark adapter dry-run contracts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
import yaml

from evolution.benchmarks.contract import run_fixture_benchmark
from evolution.benchmarks.run_tblite import main as tblite_main
from evolution.benchmarks.run_yc_bench import main as yc_bench_main

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
OUTPUT_ROOT = REPO_ROOT / "output" / "phase3-system-prompt"
PYTEST_OUTPUT_ROOT = OUTPUT_ROOT / "pytest"

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
    "output_constraints",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _phase3_output_json(tmp_path: Path, filename: str) -> Path:
    return PYTEST_OUTPUT_ROOT / tmp_path.name / "benchmarks" / filename


def _cleanup_phase3_test_output(output_json: Path) -> None:
    run_name = output_json.relative_to(PYTEST_OUTPUT_ROOT).parts[0]
    shutil.rmtree(PYTEST_OUTPUT_ROOT / run_name, ignore_errors=True)


def _adapter_args(output_json: Path, fixtures: Path, *extra_args: str) -> list[str]:
    return [
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
    ]


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


def _run_adapter_real(module: str, output_json: Path, fixtures: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
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

    output_constraints = report["output_constraints"]
    assert isinstance(output_constraints, dict)
    assert output_constraints["allowed_root"] == "output/phase3-system-prompt/"
    assert output_constraints["suffix"] == ".json"
    assert output_constraints["fresh_output_required"] is True
    assert output_constraints["symlink_output_allowed"] is False
    assert output_constraints["hardlink_output_allowed"] is False
    assert output_constraints["input_output_overlap_allowed"] is False
    assert output_json.suffix == ".json"
    assert output_json.resolve().is_relative_to(OUTPUT_ROOT.resolve())

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
    output_json = _phase3_output_json(tmp_path, "tblite.json")

    try:
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
    finally:
        _cleanup_phase3_test_output(output_json)


def test_yc_bench_adapter_runs_fast_test_fixture_dry_run_and_writes_contract_report(tmp_path: Path) -> None:
    before = {_path: _sha256(_path) for _path in (BASELINE_PROMPT, CANDIDATE_PROMPT, YC_BENCH_CASES)}
    output_json = _phase3_output_json(tmp_path, "yc_bench.json")

    try:
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
    finally:
        _cleanup_phase3_test_output(output_json)


def test_tblite_adapter_real_mode_records_external_benchmark_evidence(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "terminal-bench-lite"
    task_dir = benchmark_root / "sample-task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "task.toml").write_text("name = 'sample-task'\n")
    (task_dir / "instruction.md").write_text("Do a safe local task.\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/sh\nexit 0\n")
    output_json = _phase3_output_json(tmp_path, "tblite-real.json")

    try:
        completed = _run_adapter_real(
            "evolution.benchmarks.run_tblite",
            output_json,
            TBLITE_CASES,
            "--benchmark-root",
            str(benchmark_root),
            "--task-limit",
            "1",
        )

        assert completed.returncode == 0, completed.stderr
        assert "TBLite real benchmark smoke passed" in completed.stdout
        report = json.loads(output_json.read_text())
        assert report["mode"] == "real-benchmark-smoke"
        assert report["dry_run"] is False
        assert report["external_calls_performed"] is False
        assert report["external_benchmark_assets_validated"] is True
        assert report["real_benchmark_smoke_validated"] is True
        assert report["full_benchmark_executed"] is False
        assert report["apply_ready"] is True
        evidence = report["real_benchmark_evidence"]
        assert isinstance(evidence, dict)
        assert evidence["benchmark_root"] == str(benchmark_root)
        assert evidence["task_count"] == 1
        assert evidence["validated_task_count"] == 1
        assert evidence["sample_tasks"] == ["sample-task"]
    finally:
        _cleanup_phase3_test_output(output_json)


def test_yc_bench_adapter_real_mode_records_external_benchmark_evidence(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "yc-bench"
    preset_dir = benchmark_root / "src" / "yc_bench" / "config" / "presets"
    preset_dir.mkdir(parents=True)
    (benchmark_root / "pyproject.toml").write_text("[project]\nname = 'yc-bench'\n")
    (benchmark_root / "README.md").write_text("# YC-Bench\n")
    (benchmark_root / "src" / "yc_bench" / "__init__.py").write_text("")
    (preset_dir / "default.toml").write_text("name = 'default'\n")
    output_json = _phase3_output_json(tmp_path, "yc-bench-real.json")

    try:
        completed = _run_adapter_real(
            "evolution.benchmarks.run_yc_bench",
            output_json,
            YC_BENCH_CASES,
            "--preset",
            "fast_test",
            "--benchmark-root",
            str(benchmark_root),
        )

        assert completed.returncode == 0, completed.stderr
        assert "YC-Bench real benchmark smoke passed" in completed.stdout
        report = json.loads(output_json.read_text())
        assert report["mode"] == "real-benchmark-smoke"
        assert report["dry_run"] is False
        assert report["external_calls_performed"] is False
        assert report["external_benchmark_assets_validated"] is True
        assert report["real_benchmark_smoke_validated"] is True
        assert report["full_benchmark_executed"] is False
        assert report["apply_ready"] is True
        assert report["preset"] == "fast_test"
        evidence = report["real_benchmark_evidence"]
        assert isinstance(evidence, dict)
        assert evidence["benchmark_root"] == str(benchmark_root)
        assert evidence["package_layout_valid"] is True
        assert evidence["requested_preset"] == "fast_test"
    finally:
        _cleanup_phase3_test_output(output_json)


def test_adapter_rejects_output_json_outside_phase3_output_root(tmp_path: Path) -> None:
    output_json = tmp_path / "tblite.json"

    completed = _run_adapter("evolution.benchmarks.run_tblite", output_json, TBLITE_CASES)

    assert completed.returncode != 0
    assert "output-json must stay under output/phase3-system-prompt/" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output_json.exists()


def test_adapter_rejects_non_json_output_path(tmp_path: Path) -> None:
    output_path = _phase3_output_json(tmp_path, "tblite.txt")

    try:
        completed = _run_adapter("evolution.benchmarks.run_tblite", output_path, TBLITE_CASES)

        assert completed.returncode != 0
        assert "output-json must use a .json suffix" in completed.stderr
        assert "Traceback" not in completed.stderr
        assert not output_path.exists()
    finally:
        _cleanup_phase3_test_output(output_path)


def test_adapter_rejects_resolved_output_path_traversal() -> None:
    output_path = OUTPUT_ROOT / ".." / "phase3-escape.json"

    completed = _run_adapter("evolution.benchmarks.run_tblite", output_path, TBLITE_CASES)

    assert completed.returncode != 0
    assert "output-json must stay under output/phase3-system-prompt/" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output_path.resolve().exists()


def test_adapter_rejects_preexisting_output_json(tmp_path: Path) -> None:
    output_json = _phase3_output_json(tmp_path, "tblite.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text("existing report must stay unchanged\n")

    try:
        completed = _run_adapter("evolution.benchmarks.run_tblite", output_json, TBLITE_CASES)

        assert completed.returncode != 0
        assert "output-json must not already exist" in completed.stderr
        assert "Traceback" not in completed.stderr
        assert output_json.read_text() == "existing report must stay unchanged\n"
    finally:
        _cleanup_phase3_test_output(output_json)


def test_adapter_rejects_symlink_output_json(tmp_path: Path) -> None:
    output_json = _phase3_output_json(tmp_path, "tblite.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    symlink_target = output_json.parent / "target.json"
    symlink_target.write_text("symlink target must stay unchanged\n")
    output_json.symlink_to(symlink_target)

    try:
        completed = _run_adapter("evolution.benchmarks.run_tblite", output_json, TBLITE_CASES)

        assert completed.returncode != 0
        assert "output-json must not be a symlink" in completed.stderr
        assert "Traceback" not in completed.stderr
        assert symlink_target.read_text() == "symlink target must stay unchanged\n"
    finally:
        _cleanup_phase3_test_output(output_json)


def test_adapter_rejects_hardlinked_output_json(tmp_path: Path) -> None:
    output_json = _phase3_output_json(tmp_path, "tblite.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    outside_target = REPO_ROOT / "output" / "phase3-adapter-hardlink-target.json"
    outside_target.parent.mkdir(parents=True, exist_ok=True)
    outside_target.write_text("hardlink target must stay unchanged\n")

    try:
        os.link(outside_target, output_json)
        completed = _run_adapter("evolution.benchmarks.run_tblite", output_json, TBLITE_CASES)

        assert completed.returncode != 0
        assert "output-json must not already exist" in completed.stderr
        assert "Traceback" not in completed.stderr
        assert outside_target.read_text() == "hardlink target must stay unchanged\n"
    finally:
        output_json.unlink(missing_ok=True)
        outside_target.unlink(missing_ok=True)
        _cleanup_phase3_test_output(output_json)


def test_adapter_rejects_hardlinked_input_output_overlap(tmp_path: Path) -> None:
    output_json = _phase3_output_json(tmp_path, "tblite.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    baseline_prompt = tmp_path / "baseline_system_prompt.json"
    candidate_prompt = tmp_path / "candidate_system_prompt.json"
    baseline_prompt.write_text(BASELINE_PROMPT.read_text())
    candidate_prompt.write_text(CANDIDATE_PROMPT.read_text())
    before = baseline_prompt.read_text()

    try:
        os.link(baseline_prompt, output_json)
        with pytest.raises(ValueError, match="output-json must not overwrite input artifact"):
            run_fixture_benchmark(
                benchmark="TBLite",
                pass_condition="no_regression_against_baseline",
                baseline_prompt=baseline_prompt,
                candidate_prompt=candidate_prompt,
                fixtures_jsonl=TBLITE_CASES,
                output_json=output_json,
                dry_run=True,
            )
        assert baseline_prompt.read_text() == before
    finally:
        output_json.unlink(missing_ok=True)
        _cleanup_phase3_test_output(output_json)


def test_adapter_main_paths_do_not_perform_network_or_subprocess_calls(monkeypatch, tmp_path: Path) -> None:
    def blocked_external_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("external call attempted during dry-run fixture benchmark")

    monkeypatch.setattr(socket, "socket", blocked_external_call)
    monkeypatch.setattr(socket, "create_connection", blocked_external_call)
    monkeypatch.setattr(urllib.request, "urlopen", blocked_external_call)
    monkeypatch.setattr(subprocess, "run", blocked_external_call)
    monkeypatch.setattr(subprocess, "Popen", blocked_external_call)
    monkeypatch.setattr(os, "system", blocked_external_call)

    tblite_output = _phase3_output_json(tmp_path, "tblite-main.json")
    yc_bench_output = _phase3_output_json(tmp_path, "yc-bench-main.json")

    try:
        assert tblite_main(_adapter_args(tblite_output, TBLITE_CASES)) == 0
        assert yc_bench_main(_adapter_args(yc_bench_output, YC_BENCH_CASES, "--preset", "fast_test")) == 0
        assert tblite_output.exists()
        assert yc_bench_output.exists()
    finally:
        _cleanup_phase3_test_output(tblite_output)


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
        assert adapter_contract["output_json_constraints"] == {
            "allowed_root": "output/phase3-system-prompt/",
            "suffix": ".json",
            "path_traversal": "resolved_path_must_remain_under_allowed_root",
            "fresh_output_required": True,
            "symlink_output_allowed": False,
            "hardlink_output_allowed": False,
            "input_output_overlap_allowed": False,
        }
        assert adapter_contract["external_call_guard"] == {
            "strategy": "pytest monkeypatch blocks socket, urllib.request.urlopen, subprocess.run/Popen, and os.system during in-process adapter main calls",
            "verified_by": "tests/tools/test_phase3_benchmark_adapters.py",
        }
        assert "output_constraints" in adapter_contract["output_schema_required_fields"]
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
    assert "`--output-json` to resolve to a fresh `.json` file under `output/phase3-system-prompt/`" in readme
    assert "reject pre-existing, symlinked, hardlinked, and input-overlapping output targets" in readme
    assert "monkeypatch network/external-process APIs" in readme
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

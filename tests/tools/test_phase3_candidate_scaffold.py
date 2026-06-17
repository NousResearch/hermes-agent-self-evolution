"""Tests for the Phase 3 candidate-only system prompt scaffold."""

import json
import os
import shutil
import socket
import subprocess
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output" / "phase3-system-prompt"
PYTEST_OUTPUT_ROOT = OUTPUT_ROOT / "pytest-candidate-scaffold"
BASELINE_SECTIONS = {
    "DEFAULT_AGENT_IDENTITY": "EvAH is a Hermes-based technical assistant that answers conclusion-first and acts with human oversight.",
    "TOOL_USE_DISCIPLINE": "Use tools whenever they materially improve correctness. Verify important claims with tool outputs instead of guessing.",
    "MEMORY_GUIDANCE": "Store only durable user preferences and stable environment facts. Never store raw secrets, credentials, or one-off sensitive data.",
    "SAFETY_PRIVACY": "Protect privacy, avoid exposing secrets, and keep active runtime configuration changes behind human approval.",
    "PLATFORM_HINTS": "In Discord or group contexts, keep privacy conservative and avoid unnecessary local operational detail.",
}


def _write_prompt(path: Path, *, sections: dict[str, str], version: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "artifact_type": "system_prompt_fixture",
                "version": version,
                "sections": sections,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def _cleanup_output() -> None:
    shutil.rmtree(PYTEST_OUTPUT_ROOT, ignore_errors=True)


def _run_scaffold_subprocess(
    *,
    baseline_prompt: Path,
    candidate_prompt: Path,
    output_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python",
            "-m",
            "evolution.prompts.phase3_candidate_scaffold",
            "--baseline-prompt",
            str(baseline_prompt),
            "--candidate-prompt",
            str(candidate_prompt),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_phase3_candidate_scaffold_writes_candidate_only_artifacts(tmp_path, monkeypatch):
    from evolution.prompts import phase3_candidate_scaffold

    _cleanup_output()
    baseline_prompt = _write_prompt(
        tmp_path / "baseline_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="baseline-v1",
    )
    candidate_sections = dict(BASELINE_SECTIONS)
    candidate_sections["DEFAULT_AGENT_IDENTITY"] = (
        "EvAH is a Hermes-based technical assistant that answers conclusion-first, distinguishes operational identity from human consciousness, and acts with human oversight."
    )
    candidate_sections["MEMORY_GUIDANCE"] = (
        "Store only durable user preferences and stable environment facts. Never store raw secrets, credentials, or one-off sensitive data; reusable procedures belong in skills."
    )
    candidate_sections["PLATFORM_HINTS"] = (
        "In Discord or group contexts, keep privacy conservative, avoid unnecessary local operational detail, and keep responses concise unless an artifact is more appropriate."
    )
    candidate_prompt = _write_prompt(
        tmp_path / "candidate_system_prompt.json",
        sections=candidate_sections,
        version="candidate-v1",
    )
    output_dir = PYTEST_OUTPUT_ROOT / "run-ok"

    def blocked_external_call(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("candidate scaffold must not perform network or external process calls")

    monkeypatch.setattr(socket, "socket", blocked_external_call)
    monkeypatch.setattr(socket, "create_connection", blocked_external_call)
    monkeypatch.setattr(urllib.request, "urlopen", blocked_external_call)
    monkeypatch.setattr(subprocess, "run", blocked_external_call)
    monkeypatch.setattr(subprocess, "Popen", blocked_external_call)
    monkeypatch.setattr(os, "system", blocked_external_call)

    report = phase3_candidate_scaffold.run_candidate_scaffold(
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        output_dir=output_dir,
        dry_run=True,
    )

    report_path = output_dir / "candidate_only_report.json"
    baseline_snapshot = output_dir / "baseline_system_prompt.json"
    candidate_snapshot = output_dir / "candidate_system_prompt.json"
    review_packet = output_dir / "review_packet.md"

    assert report_path.exists()
    assert baseline_snapshot.exists()
    assert candidate_snapshot.exists()
    assert review_packet.exists()
    assert json.loads(report_path.read_text()) == report

    assert report["phase"] == "3"
    assert report["mode"] == "candidate-only-scaffold"
    assert report["scaffold_version"] == "phase3-candidate-scaffold-v1"
    assert report["dry_run"] is True
    assert report["candidate_only"] is True
    assert report["read_only_inputs"] is True
    assert report["execution_started"] is False
    assert report["run_gepa_now"] is False
    assert report["run_dspy_now"] is False
    assert report["mutate_active_system_prompt_now"] is False
    assert report["active_system_prompt_apply_approved"] is False
    assert report["apply_ready"] is False
    assert report["external_calls_performed"] is False
    assert report["real_benchmarks_executed"] is False
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["changed_sections"] == [
        "DEFAULT_AGENT_IDENTITY",
        "MEMORY_GUIDANCE",
        "PLATFORM_HINTS",
    ]
    assert report["non_evolvable_section_changes"] == []
    assert set(report["evolvable_sections"]) == {
        "DEFAULT_AGENT_IDENTITY",
        "MEMORY_GUIDANCE",
        "SESSION_SEARCH_GUIDANCE",
        "SKILLS_GUIDANCE",
        "PLATFORM_HINTS",
    }
    assert "SAFETY_PRIVACY" in report["unchanged_sections"]
    assert "TOOL_USE_DISCIPLINE" in report["unchanged_sections"]
    assert report["output_constraints"] == {
        "allowed_root": "output/phase3-system-prompt/",
        "path_traversal": "resolved_path_must_remain_under_allowed_root",
    }
    assert report["benchmark_gate"]["real_benchmarks_required_before_acceptance"] is True
    assert report["benchmark_gate"]["real_benchmarks_executed"] is False
    assert report["human_approval_gate"]["active_apply_approved"] is False
    assert report["human_approval_gate"]["separate_approval_required_before"] == [
        "running GEPA/DSPy optimization",
        "running real TBLite/YC-Bench benchmark commands",
        "editing Hermes Agent prompt source",
        "applying evolved prompt to active runtime",
        "default-gate promotion",
    ]

    expected_write_targets = [
        str(baseline_snapshot),
        str(candidate_snapshot),
        str(report_path),
        str(review_packet),
    ]
    assert report["write_targets"] == expected_write_targets
    for write_target in report["write_targets"]:
        assert Path(write_target).resolve().is_relative_to(OUTPUT_ROOT.resolve())
    assert "candidate-only scaffold" in review_packet.read_text()


def test_phase3_candidate_scaffold_rejects_non_evolvable_section_changes(tmp_path):
    _cleanup_output()
    baseline_prompt = _write_prompt(
        tmp_path / "baseline_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="baseline-v1",
    )
    candidate_sections = dict(BASELINE_SECTIONS)
    candidate_sections["SAFETY_PRIVACY"] = "This section changed without being in the Phase 3 evolvable scope."
    candidate_prompt = _write_prompt(
        tmp_path / "candidate_system_prompt.json",
        sections=candidate_sections,
        version="candidate-bad-v1",
    )
    output_dir = PYTEST_OUTPUT_ROOT / "run-non-evolvable-change"

    completed = _run_scaffold_subprocess(
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        output_dir=output_dir,
    )

    assert completed.returncode != 0
    assert "candidate scaffold failed" in completed.stderr
    assert "Traceback" not in completed.stderr
    report = json.loads((output_dir / "candidate_only_report.json").read_text())
    assert report["passed"] is False
    assert report["non_evolvable_section_changes"] == ["SAFETY_PRIVACY"]
    assert report["failed_checks"] == ["non_evolvable_section_changed SAFETY_PRIVACY"]
    assert report["apply_ready"] is False


def test_phase3_candidate_scaffold_rejects_output_dir_outside_phase3_root(tmp_path):
    baseline_prompt = _write_prompt(
        tmp_path / "baseline_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="baseline-v1",
    )
    candidate_prompt = _write_prompt(
        tmp_path / "candidate_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="candidate-v1",
    )
    completed = _run_scaffold_subprocess(
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        output_dir=tmp_path / "outside-output-root",
    )

    assert completed.returncode != 0
    assert "output-dir must stay under output/phase3-system-prompt/" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_phase3_candidate_scaffold_rejects_symlinked_write_target(tmp_path):
    _cleanup_output()
    baseline_prompt = _write_prompt(
        tmp_path / "baseline_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="baseline-v1",
    )
    candidate_prompt = _write_prompt(
        tmp_path / "candidate_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="candidate-v1",
    )
    output_dir = PYTEST_OUTPUT_ROOT / "run-symlink-target"
    output_dir.mkdir(parents=True, exist_ok=True)
    outside_target = tmp_path / "outside-report.json"
    outside_target.write_text("outside stays unchanged\n")
    (output_dir / "candidate_only_report.json").symlink_to(outside_target)

    completed = _run_scaffold_subprocess(
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        output_dir=output_dir,
    )

    assert completed.returncode != 0
    assert "write target must" in completed.stderr
    assert "symlink" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert outside_target.read_text() == "outside stays unchanged\n"


def test_phase3_candidate_scaffold_rejects_input_output_overlap(tmp_path):
    _cleanup_output()
    output_dir = PYTEST_OUTPUT_ROOT / "run-input-overlap"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_prompt = _write_prompt(
        output_dir / "baseline_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="baseline-v1",
    )
    candidate_prompt = _write_prompt(
        tmp_path / "candidate_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="candidate-v1",
    )

    completed = _run_scaffold_subprocess(
        baseline_prompt=baseline_prompt,
        candidate_prompt=candidate_prompt,
        output_dir=output_dir,
    )

    assert completed.returncode != 0
    assert "write target must not overwrite input artifact" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_phase3_candidate_scaffold_rejects_hardlinked_write_target(tmp_path):
    _cleanup_output()
    baseline_prompt = _write_prompt(
        tmp_path / "baseline_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="baseline-v1",
    )
    candidate_prompt = _write_prompt(
        tmp_path / "candidate_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="candidate-v1",
    )
    output_dir = PYTEST_OUTPUT_ROOT / "run-hardlink-target"
    output_dir.mkdir(parents=True, exist_ok=True)
    outside_target = REPO_ROOT / "output" / "phase3-hardlink-outside.json"
    outside_target.parent.mkdir(parents=True, exist_ok=True)
    outside_target.write_text("outside stays unchanged\n")
    hardlink_target = output_dir / "candidate_only_report.json"
    try:
        os.link(outside_target, hardlink_target)
        completed = _run_scaffold_subprocess(
            baseline_prompt=baseline_prompt,
            candidate_prompt=candidate_prompt,
            output_dir=output_dir,
        )

        assert completed.returncode != 0
        assert "write target must not already exist" in completed.stderr
        assert "Traceback" not in completed.stderr
        assert outside_target.read_text() == "outside stays unchanged\n"
    finally:
        hardlink_target.unlink(missing_ok=True)
        outside_target.unlink(missing_ok=True)


def test_phase3_candidate_scaffold_rejects_hardlinked_input_overlap(tmp_path):
    _cleanup_output()
    output_dir = PYTEST_OUTPUT_ROOT / "run-hardlink-input-overlap"
    output_dir.mkdir(parents=True, exist_ok=True)
    outside_input = REPO_ROOT / "output" / "phase3-hardlink-input.json"
    outside_input.parent.mkdir(parents=True, exist_ok=True)
    _write_prompt(outside_input, sections=BASELINE_SECTIONS, version="baseline-v1")
    hardlink_target = output_dir / "baseline_system_prompt.json"
    candidate_prompt = _write_prompt(
        tmp_path / "candidate_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="candidate-v1",
    )
    try:
        os.link(outside_input, hardlink_target)
        completed = _run_scaffold_subprocess(
            baseline_prompt=outside_input,
            candidate_prompt=candidate_prompt,
            output_dir=output_dir,
        )

        assert completed.returncode != 0
        assert "write target must not overwrite input artifact" in completed.stderr
        assert "Traceback" not in completed.stderr
    finally:
        hardlink_target.unlink(missing_ok=True)
        outside_input.unlink(missing_ok=True)


def test_phase3_candidate_scaffold_requires_dry_run_before_writing(tmp_path):
    from evolution.prompts import phase3_candidate_scaffold

    _cleanup_output()
    baseline_prompt = _write_prompt(
        tmp_path / "baseline_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="baseline-v1",
    )
    candidate_prompt = _write_prompt(
        tmp_path / "candidate_system_prompt.json",
        sections=BASELINE_SECTIONS,
        version="candidate-v1",
    )
    output_dir = PYTEST_OUTPUT_ROOT / "run-no-dry-run"

    try:
        phase3_candidate_scaffold.run_candidate_scaffold(
            baseline_prompt=baseline_prompt,
            candidate_prompt=candidate_prompt,
            output_dir=output_dir,
            dry_run=False,
        )
    except ValueError as exc:
        assert "requires --dry-run" in str(exc)
    else:  # pragma: no cover - failure path for clarity
        raise AssertionError("candidate scaffold accepted dry_run=False")
    assert not output_dir.exists()


def test_phase3_candidate_scaffold_contract_is_documented_and_ci_visible():
    seed = yaml.safe_load((REPO_ROOT / "seeds" / "phase3_system_prompt_evolution_execution_seed_draft.yaml").read_text())
    report = json.loads((REPO_ROOT / "reports" / "phase3_execution_seed_draft.json").read_text())
    readme = (REPO_ROOT / "README.md").read_text()
    plan_md = (REPO_ROOT / "PLAN.md").read_text()
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "phase2-tool-description-gate.yml").read_text())

    for artifact in (seed, report):
        contract = artifact["future_execution_scope"]["candidate_scaffold_contract"]
        assert contract["status"] == "candidate_only_scaffold_verified"
        assert contract["module"] == "evolution.prompts.phase3_candidate_scaffold"
        assert contract["dry_run_required"] is True
        assert contract["candidate_only"] is True
        assert contract["real_benchmarks_executed"] is False
        assert contract["active_prompt_or_source_apply_allowed"] is False
        assert contract["allowed_output_root"] == "output/phase3-system-prompt/"
        assert contract["write_target_constraints"] == [
            "resolved_write_targets_must_remain_under_allowed_root",
            "existing_write_targets_rejected",
            "existing_symlink_write_targets_rejected",
            "input_output_path_overlap_rejected",
        ]
        assert contract["verified_by"] == "tests/tools/test_phase3_candidate_scaffold.py"
        assert "candidate_only_report.json" in contract["output_artifacts"]
        assert "review_packet.md" in contract["output_artifacts"]
        assert "real_benchmarks_executed" in contract["output_schema_required_fields"]
        assert "non_evolvable_section_changes" in contract["output_schema_required_fields"]

    assert "evolution/prompts/phase3_candidate_scaffold.py" in report["deliverables"]
    assert "tests/tools/test_phase3_candidate_scaffold.py" in report["deliverables"]
    assert "Phase 3 Candidate-Only Scaffold" in readme
    assert "python -m evolution.prompts.phase3_candidate_scaffold" in readme
    assert "real benchmark execution remains required" in readme
    assert "Phase 3 candidate-only scaffold" in plan_md
    assert "no GEPA/DSPy execution" in plan_md

    triggers = workflow.get("on", workflow.get(True))
    for trigger_name in ("pull_request", "push"):
        paths = triggers[trigger_name]["paths"]
        assert "evolution/prompts/**" in paths
    run_blocks = "\n".join(
        str(step.get("run", ""))
        for step in workflow["jobs"]["phase2-tool-description-gate"]["steps"]
        if isinstance(step, dict)
    )
    assert "tests/tools/test_phase3_candidate_scaffold.py" in run_blocks

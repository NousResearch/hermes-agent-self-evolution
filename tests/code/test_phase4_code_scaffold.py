"""Tests for the Phase 4 dry-run code scaffold."""

import json
import os
import shutil
import socket
import subprocess
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "output" / "phase4-code-evolution"
PYTEST_OUTPUT_ROOT = OUTPUT_ROOT / "pytest-code-scaffold"


def _write_tool(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
from tools.registry import registry


def safe_tool(value: str, *, strict: bool = True) -> str:
    if not value:
        raise ValueError("value is required")
    return value if strict else value.strip()


class SafeToolHelper:
    def normalize(self, value: str) -> str:
        return value.strip()


registry.register(
    name="safe_tool",
    toolset="example",
    schema={"parameters": {"value": {"type": "string"}}, "required": ["value"]},
    handler=safe_tool,
)
""".lstrip()
    )
    return path


def _write_spec(path: Path, hermes_repo: Path) -> Path:
    payload = {
        "phase": "4",
        "mode": "code-evolution-target",
        "target_id": "safe-tool-edge-case-001",
        "hermes_base": {
            "repo": str(hermes_repo),
            "base_ref": "test-base",
            "require_clean_worktree": True,
        },
        "allowed_mutation": {
            "files": ["tools/safe_tool.py"],
            "deny_globs": ["**/config*.py", "skills/**", "plugins/**", "memory/**"],
        },
        "freeze": {
            "function_signatures": True,
            "registry_register_calls": True,
            "public_cli_args": True,
        },
        "reproduction": {
            "failing_case_description": "empty value should stay rejected",
            "reproducer_command": "python -m pytest tests/tools/test_safe_tool.py -q",
        },
        "fitness": {
            "required_commands": [
                "python -m compileall -q tools/safe_tool.py",
                "python -m pytest tests/tools/test_safe_tool.py -q",
            ]
        },
        "benchmarks": {
            "full_benchmark_required_before_acceptance": True,
            "run_benchmarks_now": False,
        },
        "approvals": {
            "darwinian_install_approved": False,
            "darwinian_execution_approved": False,
            "hermes_source_mutation_approved": False,
            "budget_approved_usd": 0,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path


def _cleanup_output() -> None:
    shutil.rmtree(PYTEST_OUTPUT_ROOT, ignore_errors=True)


def test_phase4_code_scaffold_writes_dry_run_candidate_only_artifacts(tmp_path, monkeypatch):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    _cleanup_output()
    hermes_repo = tmp_path / "hermes-agent"
    target_file = _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    output_dir = PYTEST_OUTPUT_ROOT / "run-ok"

    def blocked_external_call(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("Phase 4 dry-run scaffold must not perform network or external process calls")

    monkeypatch.setattr(socket, "socket", blocked_external_call)
    monkeypatch.setattr(socket, "create_connection", blocked_external_call)
    monkeypatch.setattr(urllib.request, "urlopen", blocked_external_call)
    monkeypatch.setattr(subprocess, "run", blocked_external_call)
    monkeypatch.setattr(subprocess, "Popen", blocked_external_call)
    monkeypatch.setattr(os, "system", blocked_external_call)

    report = run_phase4_code_scaffold(target_spec=spec_path, output_dir=output_dir, dry_run=True)

    target_snapshot = output_dir / "target_snapshot.json"
    baseline_api_surface = output_dir / "baseline_api_surface.json"
    freeze_report = output_dir / "freeze_report.json"
    reproduction_contract = output_dir / "reproduction_contract.json"
    scaffold_report = output_dir / "scaffold_report.json"
    review_packet = output_dir / "review_packet.md"

    for artifact in [
        target_snapshot,
        baseline_api_surface,
        freeze_report,
        reproduction_contract,
        scaffold_report,
        review_packet,
    ]:
        assert artifact.exists(), artifact
        assert artifact.resolve().is_relative_to(OUTPUT_ROOT.resolve())

    assert json.loads(scaffold_report.read_text()) == report
    assert report["phase"] == "4"
    assert report["mode"] == "code-evolution-candidate-only-scaffold"
    assert report["scaffold_version"] == "phase4-code-scaffold-v1"
    assert report["dry_run"] is True
    assert report["candidate_only"] is True
    assert report["read_only_inputs"] is True
    assert report["darwinian_cli_invoked"] is False
    assert report["darwinian_imported"] is False
    assert report["external_calls_performed"] is False
    assert report["package_installed"] is False
    assert report["hermes_source_mutation_performed"] is False
    assert report["active_runtime_apply_approved"] is False
    assert report["apply_ready"] is False
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["target_spec"]["target_id"] == "safe-tool-edge-case-001"
    assert report["target_spec"]["target_files"] == ["tools/safe_tool.py"]
    assert report["approval_gates"]["budget_approved_usd"] == 0
    assert report["output_constraints"]["allowed_root"] == "output/phase4-code-evolution/"
    assert report["output_constraints"]["fresh_output_required"] is True
    assert report["output_constraints"]["hermes_source_write_allowed"] is False
    assert str(target_file) in json.dumps(json.loads(target_snapshot.read_text()))

    api_surface = json.loads(baseline_api_surface.read_text())
    assert api_surface["target_file"] == str(target_file.resolve())
    assert "safe_tool" in api_surface["module_functions"]
    assert "SafeToolHelper.normalize" in api_surface["class_methods"]
    assert api_surface["registry_register_calls"][0]["name"] == "safe_tool"

    freeze_plan = json.loads(freeze_report.read_text())
    assert freeze_plan["candidate_generated"] is False
    assert freeze_plan["checks_executed"] is False
    assert freeze_plan["mandatory_gate"]["required_before_apply"] is True
    assert freeze_plan["mandatory_gate"]["executed"] is False
    assert json.loads(reproduction_contract.read_text())["baseline_reproduction_executed"] is False
    assert "Phase 4 code-evolution dry-run scaffold" in review_packet.read_text()


def test_phase4_code_scaffold_runs_freeze_comparator_as_mandatory_gate_for_candidate(tmp_path):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    _cleanup_output()
    hermes_repo = tmp_path / "hermes-agent"
    baseline_file = _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    candidate_file = PYTEST_OUTPUT_ROOT / "candidate-pass" / "safe_tool.py"
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    candidate_file.write_text(
        baseline_file.read_text().replace("return value if strict else value.strip()", "return value.strip() if strict else value.strip()")
    )
    output_dir = PYTEST_OUTPUT_ROOT / "run-freeze-pass"

    report = run_phase4_code_scaffold(
        target_spec=spec_path,
        output_dir=output_dir,
        dry_run=True,
        candidate_file=candidate_file,
    )

    freeze_report = json.loads((output_dir / "freeze_report.json").read_text())
    scaffold_report = json.loads((output_dir / "scaffold_report.json").read_text())
    assert report == scaffold_report
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["freeze_checks"]["mode"] == "freeze-comparator"
    assert report["freeze_checks"]["passed"] is True
    assert report["freeze_checks"]["checks_executed"] is True
    assert report["freeze_checks"]["mandatory_gate"] == {
        "required_before_apply": True,
        "executed": True,
        "passed": True,
    }
    assert freeze_report["artifacts"]["output_json"] == str(output_dir / "freeze_report.json")
    assert freeze_report["comparisons"]["function_signatures"] == {"changed": [], "added": [], "removed": []}
    assert freeze_report["candidate_file"] == str(candidate_file.resolve())
    assert report["candidate_artifact"] == {
        "provided": True,
        "path": str(candidate_file.resolve()),
        "relative_path": str(candidate_file.resolve().relative_to(OUTPUT_ROOT.resolve())),
        "sha256": __import__("hashlib").sha256(candidate_file.read_bytes()).hexdigest(),
        "bytes": candidate_file.stat().st_size,
        "symlink": False,
        "parent_symlink": False,
        "hardlink": False,
    }


def test_phase4_code_scaffold_freeze_gate_fails_visibly_for_candidate_surface_drift(tmp_path):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    _cleanup_output()
    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    candidate_file = PYTEST_OUTPUT_ROOT / "candidate-fail" / "safe_tool.py"
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    candidate_file.write_text(
        """
from tools.registry import registry


def safe_tool(value: str, retries: int = 0, *, strict: bool = True) -> str:
    if not value:
        raise ValueError("value is required")
    return value


class SafeToolHelper:
    def normalize(self, value: str) -> str:
        return value.strip()


registry.register(
    name="safe_tool",
    toolset="example",
    schema={"parameters": {"value": {"type": "string"}}, "required": ["value"]},
    handler=safe_tool,
)
""".lstrip()
    )
    output_dir = PYTEST_OUTPUT_ROOT / "run-freeze-fail"

    report = run_phase4_code_scaffold(
        target_spec=spec_path,
        output_dir=output_dir,
        dry_run=True,
        candidate_file=candidate_file,
    )

    assert report["passed"] is False
    assert any(check.startswith("freeze_comparator:function_signature_changed:safe_tool") for check in report["failed_checks"])
    freeze_report = json.loads((output_dir / "freeze_report.json").read_text())
    scaffold_report = json.loads((output_dir / "scaffold_report.json").read_text())
    assert freeze_report["passed"] is False
    assert any(check.startswith("function_signature_changed:safe_tool") for check in freeze_report["failed_checks"])
    assert scaffold_report["passed"] is False
    assert scaffold_report["freeze_checks"]["mandatory_gate"]["passed"] is False


def test_phase4_code_scaffold_writes_visible_freeze_failure_for_empty_candidate(tmp_path):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    _cleanup_output()
    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    candidate_file = PYTEST_OUTPUT_ROOT / "candidate-empty" / "safe_tool.py"
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    candidate_file.write_text("")
    output_dir = PYTEST_OUTPUT_ROOT / "run-empty-candidate-freeze-fail"

    report = run_phase4_code_scaffold(
        target_spec=spec_path,
        output_dir=output_dir,
        dry_run=True,
        candidate_file=candidate_file,
    )

    scaffold_report_path = output_dir / "scaffold_report.json"
    freeze_report_path = output_dir / "freeze_report.json"
    assert scaffold_report_path.exists()
    assert freeze_report_path.exists()
    scaffold_report = json.loads(scaffold_report_path.read_text())
    freeze_report = json.loads(freeze_report_path.read_text())
    assert report == scaffold_report
    assert report["candidate_artifact"]["bytes"] == 0
    assert report["candidate_artifact"]["symlink"] is False
    assert report["candidate_artifact"]["parent_symlink"] is False
    assert report["candidate_artifact"]["hardlink"] is False
    assert report["passed"] is False
    assert freeze_report["passed"] is False
    assert any(check.startswith("freeze_comparator:") for check in report["failed_checks"])


def test_phase4_code_scaffold_cli_exits_nonzero_when_freeze_gate_fails(tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    candidate_file = PYTEST_OUTPUT_ROOT / "candidate-cli-fail" / "safe_tool.py"
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    candidate_file.write_text(
        """
from tools.registry import registry


def safe_tool(value: str, *, strict: bool = True, normalize: bool = False) -> str:
    if not value:
        raise ValueError("value is required")
    return value


class SafeToolHelper:
    def normalize(self, value: str) -> str:
        return value.strip()


registry.register(
    name="safe_tool",
    toolset="example",
    schema={"parameters": {"value": {"type": "string"}}, "required": ["value"]},
    handler=safe_tool,
)
""".lstrip()
    )
    output_dir = PYTEST_OUTPUT_ROOT / "run-cli-freeze-fail"
    shutil.rmtree(output_dir, ignore_errors=True)

    completed = subprocess.run(
        [
            "python",
            "-m",
            "evolution.code.phase4_code_scaffold",
            "--target-spec",
            str(spec_path),
            "--output-dir",
            str(output_dir),
            "--candidate-file",
            str(candidate_file),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    summary = json.loads(completed.stdout)
    assert summary["passed"] is False
    assert any(check.startswith("freeze_comparator:function_signature_changed:safe_tool") for check in summary["failed_checks"])
    scaffold_report = json.loads((output_dir / "scaffold_report.json").read_text())
    assert scaffold_report["passed"] is False
    assert "Traceback" not in completed.stderr


def test_phase4_code_scaffold_rejects_candidate_file_outside_phase4_output_root(tmp_path):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    hermes_repo = tmp_path / "hermes-agent"
    baseline_file = _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    candidate_file = tmp_path / "candidate-outside" / "safe_tool.py"
    candidate_file.parent.mkdir(parents=True)
    candidate_file.write_text(baseline_file.read_text())

    try:
        run_phase4_code_scaffold(
            target_spec=spec_path,
            output_dir=PYTEST_OUTPUT_ROOT / "run-candidate-outside",
            dry_run=True,
            candidate_file=candidate_file,
        )
    except ValueError as exc:
        assert "candidate-file must stay under output/phase4-code-evolution/" in str(exc)
    else:
        raise AssertionError("expected candidate output-root rejection")


def test_phase4_code_scaffold_rejects_non_dry_run(tmp_path):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)

    try:
        run_phase4_code_scaffold(target_spec=spec_path, output_dir=PYTEST_OUTPUT_ROOT / "run-no-dry", dry_run=False)
    except ValueError as exc:
        assert "requires --dry-run" in str(exc)
    else:
        raise AssertionError("expected dry-run rejection")


def test_phase4_code_scaffold_rejects_output_dir_outside_phase4_root(tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)

    completed = subprocess.run(
        [
            "python",
            "-m",
            "evolution.code.phase4_code_scaffold",
            "--target-spec",
            str(spec_path),
            "--output-dir",
            str(tmp_path / "outside-output-root"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "output-dir must stay under output/phase4-code-evolution/" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_phase4_code_scaffold_rejects_symlink_output_dir(tmp_path):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    _cleanup_output()
    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    output_dir = PYTEST_OUTPUT_ROOT / "run-symlink-output-dir"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    outside_output = tmp_path / "outside-output"
    outside_output.mkdir()
    output_dir.symlink_to(outside_output, target_is_directory=True)

    try:
        run_phase4_code_scaffold(target_spec=spec_path, output_dir=output_dir, dry_run=True)
    except ValueError as exc:
        assert "output-dir must not be a symlink" in str(exc)
    else:
        raise AssertionError("expected symlink output-dir rejection")


def test_phase4_code_scaffold_rejects_symlink_phase4_output_root(tmp_path, monkeypatch):
    import evolution.code.phase4_code_scaffold as scaffold

    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    fake_repo_root = tmp_path / "hse"
    fake_output_parent = fake_repo_root / "output"
    fake_output_parent.mkdir(parents=True)
    symlink_root = fake_output_parent / "phase4-code-evolution"
    outside_output_root = tmp_path / "outside-output-root"
    outside_output_root.mkdir()
    symlink_root.symlink_to(outside_output_root, target_is_directory=True)
    monkeypatch.setattr(scaffold, "REPO_ROOT", fake_repo_root)
    monkeypatch.setattr(scaffold, "PHASE4_OUTPUT_ROOT", symlink_root)

    try:
        scaffold.run_phase4_code_scaffold(
            target_spec=spec_path,
            output_dir=symlink_root / "run-through-symlink",
            dry_run=True,
        )
    except ValueError as exc:
        assert "output root must not be a symlink" in str(exc)
    else:
        raise AssertionError("expected symlink output root rejection")


def test_phase4_code_scaffold_rejects_preexisting_write_target(tmp_path):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    _cleanup_output()
    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    output_dir = PYTEST_OUTPUT_ROOT / "run-existing-target"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scaffold_report.json").write_text("pre-existing\n")

    try:
        run_phase4_code_scaffold(target_spec=spec_path, output_dir=output_dir, dry_run=True)
    except ValueError as exc:
        assert "write target must not already exist" in str(exc)
    else:
        raise AssertionError("expected pre-existing target rejection")


def test_phase4_code_scaffold_rejects_stale_extra_files_in_output_dir(tmp_path):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    _cleanup_output()
    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    output_dir = PYTEST_OUTPUT_ROOT / "run-stale-extra"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "unrelated-stale.txt").write_text("stale\n")

    try:
        run_phase4_code_scaffold(target_spec=spec_path, output_dir=output_dir, dry_run=True)
    except ValueError as exc:
        assert "output-dir must be empty before scaffolding" in str(exc)
    else:
        raise AssertionError("expected stale extra file rejection")


def test_phase4_code_scaffold_rejects_existing_output_dir_file_with_value_error(tmp_path):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    _cleanup_output()
    hermes_repo = tmp_path / "hermes-agent"
    _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    output_dir = PYTEST_OUTPUT_ROOT / "run-output-dir-file"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.write_text("pre-existing file\n")

    try:
        run_phase4_code_scaffold(target_spec=spec_path, output_dir=output_dir, dry_run=True)
    except ValueError as exc:
        assert "output-dir must be a directory before scaffolding" in str(exc)
        assert str(output_dir) in str(exc)
    else:
        raise AssertionError("expected existing output-dir file rejection")


def test_phase4_code_scaffold_rejects_candidate_file_hardlink(tmp_path):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    _cleanup_output()
    hermes_repo = tmp_path / "hermes-agent"
    baseline_file = _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    original_candidate = tmp_path / "generated-source" / "safe_tool.py"
    original_candidate.parent.mkdir(parents=True)
    original_candidate.write_text(baseline_file.read_text())
    candidate_file = PYTEST_OUTPUT_ROOT / "candidate-hardlink" / "safe_tool.py"
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    os.link(original_candidate, candidate_file)

    try:
        run_phase4_code_scaffold(
            target_spec=spec_path,
            output_dir=PYTEST_OUTPUT_ROOT / "run-candidate-hardlink",
            dry_run=True,
            candidate_file=candidate_file,
        )
    except ValueError as exc:
        assert "candidate-file must not be a hardlink" in str(exc)
    else:
        raise AssertionError("expected candidate hardlink rejection")


def test_phase4_code_scaffold_rejects_candidate_file_parent_symlink(tmp_path):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    _cleanup_output()
    hermes_repo = tmp_path / "hermes-agent"
    baseline_file = _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    actual_parent = PYTEST_OUTPUT_ROOT / "actual-candidate-parent"
    actual_parent.mkdir(parents=True, exist_ok=True)
    candidate_file = actual_parent / "safe_tool.py"
    candidate_file.write_text(baseline_file.read_text())
    symlink_parent = PYTEST_OUTPUT_ROOT / "candidate-parent-symlink"
    symlink_parent.symlink_to(actual_parent, target_is_directory=True)

    try:
        run_phase4_code_scaffold(
            target_spec=spec_path,
            output_dir=PYTEST_OUTPUT_ROOT / "run-candidate-parent-symlink",
            dry_run=True,
            candidate_file=symlink_parent / "safe_tool.py",
        )
    except ValueError as exc:
        assert "candidate-file parent must not be a symlink" in str(exc)
    else:
        raise AssertionError("expected candidate parent symlink rejection")


def test_phase4_code_scaffold_rejects_candidate_file_parent_symlink_to_output_root(tmp_path):
    from evolution.code.phase4_code_scaffold import run_phase4_code_scaffold

    _cleanup_output()
    hermes_repo = tmp_path / "hermes-agent"
    baseline_file = _write_tool(hermes_repo / "tools" / "safe_tool.py")
    spec_path = _write_spec(tmp_path / "phase4_target.yaml", hermes_repo)
    real_candidate_parent = PYTEST_OUTPUT_ROOT / "actual-root-alias-candidate"
    real_candidate_parent.mkdir(parents=True, exist_ok=True)
    (real_candidate_parent / "safe_tool.py").write_text(baseline_file.read_text())
    root_alias = OUTPUT_ROOT.parent / "phase4-code-evolution-root-alias"
    if root_alias.exists() or root_alias.is_symlink():
        root_alias.unlink()
    root_alias.symlink_to(OUTPUT_ROOT, target_is_directory=True)

    try:
        run_phase4_code_scaffold(
            target_spec=spec_path,
            output_dir=PYTEST_OUTPUT_ROOT / "run-candidate-root-alias-symlink",
            dry_run=True,
            candidate_file=root_alias / "pytest-code-scaffold" / "actual-root-alias-candidate" / "safe_tool.py",
        )
    except ValueError as exc:
        assert "candidate-file parent must not be a symlink" in str(exc)
    else:
        raise AssertionError("expected candidate output-root alias symlink rejection")
    finally:
        if root_alias.exists() or root_alias.is_symlink():
            root_alias.unlink()

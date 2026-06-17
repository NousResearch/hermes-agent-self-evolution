"""Tests for the Phase 4 scaffold report contract."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _valid_output_root() -> Path:
    return REPO_ROOT / "output" / "phase4-code-evolution" / "contract-fixtures"


def _valid_report(output_root: Path) -> dict[str, object]:
    output_dir = output_root / "run-ok"
    report_path = output_dir / "scaffold_report.json"
    review_packet = output_dir / "review_packet.md"
    return {
        "phase": "4",
        "mode": "code-evolution-candidate-only-scaffold",
        "scaffold_version": "phase4-code-scaffold-v1",
        "dry_run": True,
        "candidate_only": True,
        "read_only_inputs": True,
        "darwinian_cli_invoked": False,
        "darwinian_imported": False,
        "external_calls_performed": False,
        "package_installed": False,
        "hermes_source_mutation_performed": False,
        "active_runtime_apply_approved": False,
        "apply_ready": False,
        "passed": True,
        "failed_checks": [],
        "target_spec": {"target_id": "safe-tool-edge-case-001", "target_files": ["tools/safe_tool.py"]},
        "allowed_mutation": {"files": ["tools/safe_tool.py"], "exactly_one_target_file": True},
        "candidate_artifact": {
            "provided": False,
            "path": None,
            "relative_path": None,
            "sha256": None,
            "bytes": None,
            "symlink": False,
            "parent_symlink": False,
            "hardlink": False,
        },
        "freeze_checks": {
            "function_signatures_required": True,
            "registry_register_calls_required": True,
            "candidate_generated": False,
            "checks_executed": False,
            "mandatory_gate": {
                "required_before_apply": True,
                "executed": False,
                "passed": None,
            },
        },
        "fitness_plan": {"required_commands": ["python -m pytest tests/tools/test_safe_tool.py -q"]},
        "approval_gates": {
            "darwinian_install_approved": False,
            "darwinian_execution_approved": False,
            "hermes_source_mutation_approved": False,
            "budget_approved_usd": 0,
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "scaffold_report": str(report_path),
            "review_packet": str(review_packet),
        },
        "write_targets": [str(report_path), str(review_packet)],
        "output_constraints": {
            "allowed_root": "output/phase4-code-evolution/",
            "fresh_output_required": True,
            "empty_output_dir_required": True,
            "stale_extra_files_allowed": False,
            "symlink_output_allowed": False,
            "hardlink_output_allowed": False,
            "candidate_symlink_allowed": False,
            "candidate_parent_symlink_allowed": False,
            "candidate_hardlink_allowed": False,
            "candidate_provenance_required": True,
            "input_output_overlap_allowed": False,
            "hermes_source_write_allowed": False,
        },
    }


def test_report_contract_accepts_valid_candidate_only_scaffold_report(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    validation = validate_phase4_scaffold_report_contract(_valid_report(_valid_output_root()))

    assert validation.passed is True
    assert validation.errors == ()


def test_report_contract_requires_candidate_artifact_provenance(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    report.pop("candidate_artifact")

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "report missing required field: candidate_artifact" in validation.errors

    report = _valid_report(_valid_output_root())
    output_root = _valid_output_root()
    candidate_path = output_root / "candidate" / "safe_tool.py"
    freeze_report = _valid_freeze_comparison_report(output_root)
    freeze_report["candidate_file"] = str(candidate_path)
    freeze_report["candidate_api_surface"] = {"target_file": str(candidate_path), "sha256": "a" * 64}
    freeze_report["mandatory_gate"] = {
        "required_before_apply": True,
        "executed": True,
        "passed": True,
    }
    report["freeze_checks"] = freeze_report
    report["candidate_artifact"] = {
        "provided": True,
        "path": str(candidate_path),
        "relative_path": "contract-fixtures/candidate/safe_tool.py",
        "sha256": "a" * 64,
        "bytes": 10,
        "symlink": False,
        "parent_symlink": False,
        "hardlink": False,
    }
    validation = validate_phase4_scaffold_report_contract(report)
    assert validation.passed is True


def test_report_contract_rejects_unsafe_candidate_artifact_provenance(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    report["candidate_artifact"] = {
        "provided": True,
        "path": str(tmp_path / "outside" / "safe_tool.py"),
        "relative_path": "../outside/safe_tool.py",
        "sha256": "not-a-sha",
        "bytes": -1,
        "symlink": True,
        "parent_symlink": True,
        "hardlink": True,
    }

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "candidate_artifact.path must be under output/phase4-code-evolution" in validation.errors
    assert "candidate_artifact.relative_path must not contain traversal segments" in validation.errors
    assert "candidate_artifact.sha256 must be a 64-character lowercase hex digest" in validation.errors
    assert "candidate_artifact.bytes must be a non-negative integer" in validation.errors
    assert "candidate_artifact.symlink must be false" in validation.errors
    assert "candidate_artifact.parent_symlink must be false" in validation.errors
    assert "candidate_artifact.hardlink must be false" in validation.errors


def test_report_contract_allows_zero_byte_candidate_artifact_but_rejects_invalid_byte_types(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    output_root = _valid_output_root()
    candidate_path = output_root / "candidate" / "safe_tool.py"
    report = _valid_report(output_root)
    freeze_report = _valid_freeze_comparison_report(output_root)
    freeze_report["candidate_file"] = str(candidate_path)
    freeze_report["candidate_api_surface"] = {"target_file": str(candidate_path), "sha256": "a" * 64}
    freeze_report["mandatory_gate"] = {
        "required_before_apply": True,
        "executed": True,
        "passed": True,
    }
    report["freeze_checks"] = freeze_report
    report["candidate_artifact"] = {
        "provided": True,
        "path": str(candidate_path),
        "relative_path": "contract-fixtures/candidate/safe_tool.py",
        "sha256": "a" * 64,
        "bytes": 0,
        "symlink": False,
        "parent_symlink": False,
        "hardlink": False,
    }

    validation = validate_phase4_scaffold_report_contract(report)
    assert validation.passed is True
    assert validation.errors == ()

    for invalid_size in (True, -1, "0"):
        report["candidate_artifact"]["bytes"] = invalid_size
        validation = validate_phase4_scaffold_report_contract(report)
        assert validation.passed is False
        assert "candidate_artifact.bytes must be a non-negative integer" in validation.errors


def test_report_contract_requires_candidate_artifact_relative_path_consistency(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    output_root = _valid_output_root()
    candidate_path = output_root / "candidate" / "safe_tool.py"
    report["candidate_artifact"] = {
        "provided": True,
        "path": str(candidate_path),
        "relative_path": "/tmp/outside.py",
        "sha256": "a" * 64,
        "bytes": 10,
        "symlink": False,
        "parent_symlink": False,
        "hardlink": False,
    }

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "candidate_artifact.relative_path must be relative" in validation.errors

    report["candidate_artifact"]["relative_path"] = "other/safe_tool.py"
    validation = validate_phase4_scaffold_report_contract(report)
    assert validation.passed is False
    assert "candidate_artifact.relative_path must match candidate_artifact.path" in validation.errors


def test_report_contract_couples_candidate_provenance_to_freeze_execution(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    freeze_report = _valid_freeze_comparison_report(_valid_output_root())
    freeze_report["mandatory_gate"] = {
        "required_before_apply": True,
        "executed": True,
        "passed": True,
    }
    report["freeze_checks"] = freeze_report

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "candidate_artifact.provided must be true when freeze comparator executed" in validation.errors

    report = _valid_report(_valid_output_root())
    output_root = _valid_output_root()
    report["candidate_artifact"] = {
        "provided": True,
        "path": str(output_root / "candidate" / "safe_tool.py"),
        "relative_path": "candidate/safe_tool.py",
        "sha256": "a" * 64,
        "bytes": 10,
        "symlink": False,
        "parent_symlink": False,
        "hardlink": False,
    }
    validation = validate_phase4_scaffold_report_contract(report)
    assert validation.passed is False
    assert "freeze_checks.mode must be freeze-comparator when candidate_artifact.provided is true" in validation.errors


def test_report_contract_requires_candidate_freeze_api_surface_provenance(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    output_root = _valid_output_root()
    report = _valid_report(output_root)
    candidate_path = output_root / "candidate" / "safe_tool.py"
    freeze_report = _valid_freeze_comparison_report(output_root)
    freeze_report["candidate_file"] = str(candidate_path)
    freeze_report.pop("candidate_api_surface", None)
    freeze_report["mandatory_gate"] = {
        "required_before_apply": True,
        "executed": True,
        "passed": True,
    }
    report["freeze_checks"] = freeze_report
    report["candidate_artifact"] = {
        "provided": True,
        "path": str(candidate_path),
        "relative_path": "contract-fixtures/candidate/safe_tool.py",
        "sha256": "a" * 64,
        "bytes": 10,
        "symlink": False,
        "parent_symlink": False,
        "hardlink": False,
    }

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "freeze_checks.candidate_api_surface must be an object when candidate_artifact.provided is true" in validation.errors

    freeze_report["candidate_api_surface"] = {"target_file": str(output_root / "other" / "safe_tool.py"), "sha256": "a" * 64}
    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "freeze_checks.candidate_api_surface.target_file must match freeze_checks.candidate_file" in validation.errors


def test_report_contract_binds_candidate_artifact_to_embedded_freeze_candidate(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    output_root = _valid_output_root()
    report = _valid_report(output_root)
    candidate_path = output_root / "candidate" / "safe_tool.py"
    freeze_report = _valid_freeze_comparison_report(output_root)
    other_candidate_path = output_root / "other" / "safe_tool.py"
    freeze_report["candidate_file"] = str(other_candidate_path)
    freeze_report["candidate_api_surface"] = {"target_file": str(other_candidate_path), "sha256": "a" * 64}
    freeze_report["mandatory_gate"] = {
        "required_before_apply": True,
        "executed": True,
        "passed": True,
    }
    report["freeze_checks"] = freeze_report
    report["candidate_artifact"] = {
        "provided": True,
        "path": str(candidate_path),
        "relative_path": "contract-fixtures/candidate/safe_tool.py",
        "sha256": "a" * 64,
        "bytes": 10,
        "symlink": False,
        "parent_symlink": False,
        "hardlink": False,
    }

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "candidate_artifact.path must match freeze_checks.candidate_file" in validation.errors

    freeze_report["candidate_file"] = str(candidate_path)
    freeze_report["candidate_api_surface"] = {"target_file": str(candidate_path), "sha256": "b" * 64}
    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "candidate_artifact.sha256 must match freeze_checks.candidate_api_surface.sha256" in validation.errors


def test_report_contract_requires_lifecycle_hardening_constraints(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    constraints = report["output_constraints"]
    assert isinstance(constraints, dict)
    constraints["empty_output_dir_required"] = False
    constraints["stale_extra_files_allowed"] = True
    constraints["candidate_parent_symlink_allowed"] = True
    constraints["candidate_hardlink_allowed"] = True
    constraints.pop("candidate_provenance_required")

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "output_constraints.empty_output_dir_required must be True" in validation.errors
    assert "output_constraints.stale_extra_files_allowed must be False" in validation.errors
    assert "output_constraints.candidate_parent_symlink_allowed must be False" in validation.errors
    assert "output_constraints.candidate_hardlink_allowed must be False" in validation.errors
    assert "output_constraints.candidate_provenance_required must be True" in validation.errors


def test_report_contract_rejects_apply_payloads_and_true_execution_flags(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    report["apply_ready"] = True
    report["hermes_source_mutation_performed"] = True
    report["patch"] = "diff --git a/tools/safe_tool.py b/tools/safe_tool.py"

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "top-level apply_ready must be false" in validation.errors
    assert "top-level hermes_source_mutation_performed must be false" in validation.errors
    assert "scaffold report must not contain apply payload key: patch" in validation.errors


def test_report_contract_enforces_passed_failed_checks_consistency(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    report["passed"] = False

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "failed_checks must be non-empty when passed is false" in validation.errors

    report["passed"] = True
    report["failed_checks"] = ["unexpected failure"]
    validation = validate_phase4_scaffold_report_contract(report)
    assert validation.passed is False
    assert "failed_checks must be empty when passed is true" in validation.errors


def test_report_contract_rejects_write_targets_outside_phase4_output_root(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["output_dir"] = str(tmp_path / "elsewhere")
    report["write_targets"] = [str(tmp_path / "elsewhere" / "scaffold_report.json")]

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "artifacts.output_dir must be under output/phase4-code-evolution" in validation.errors
    assert "write target must be under output/phase4-code-evolution" in validation.errors


def test_report_contract_rejects_phase4_output_root_substring_confusion(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    evil_output = tmp_path / "output" / "phase4-code-evolution-evil"
    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["output_dir"] = str(evil_output)
    report["write_targets"] = [str(evil_output / "scaffold_report.json")]

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "artifacts.output_dir must be under output/phase4-code-evolution" in validation.errors
    assert "write target must be under output/phase4-code-evolution" in validation.errors


def test_report_contract_rejects_unrelated_absolute_phase4_output_root(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    unrelated_output = tmp_path / "elsewhere" / "output" / "phase4-code-evolution" / "run-ok"
    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["output_dir"] = str(unrelated_output)
    report["write_targets"] = [str(unrelated_output / "scaffold_report.json")]

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "artifacts.output_dir must be under output/phase4-code-evolution" in validation.errors
    assert "write target must be under output/phase4-code-evolution" in validation.errors


def test_report_contract_rejects_any_artifact_path_outside_phase4_output_root(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["scaffold_report"] = str(tmp_path / "outside-report.json")

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "artifacts.scaffold_report must be under output/phase4-code-evolution" in validation.errors


def test_report_contract_rejects_unknown_artifact_path_outside_phase4_output_root(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["extra_log"] = str(tmp_path / "outside.log")

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "artifacts.extra_log must be under output/phase4-code-evolution" in validation.errors


def test_report_contract_rejects_target_spec_multi_target_payload(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    target_spec = report["target_spec"]
    assert isinstance(target_spec, dict)
    target_spec["target_files"] = ["tools/safe_tool.py", "tools/other.py"]

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "target_spec.target_files must contain exactly one target file" in validation.errors
    assert "target_spec.target_files must match allowed_mutation.files" in validation.errors


def test_report_contract_rejects_target_spec_allowed_mutation_mismatch(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    target_spec = report["target_spec"]
    assert isinstance(target_spec, dict)
    target_spec["target_files"] = ["tools/other.py"]

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "target_spec.target_files must match allowed_mutation.files" in validation.errors


def test_report_contract_rejects_multi_target_report(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    allowed_mutation = report["allowed_mutation"]
    assert isinstance(allowed_mutation, dict)
    allowed_mutation["files"] = ["tools/safe_tool.py", "tools/other.py"]
    allowed_mutation["exactly_one_target_file"] = False

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "allowed_mutation.exactly_one_target_file must be true" in validation.errors


def test_report_contract_rejects_scaffold_without_mandatory_freeze_gate(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    freeze_checks = report["freeze_checks"]
    assert isinstance(freeze_checks, dict)
    freeze_checks.pop("mandatory_gate")

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "freeze_checks.mandatory_gate must be an object" in validation.errors


def test_report_contract_requires_failed_freeze_gate_to_surface_top_level(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    output_root = _valid_output_root()
    report = _valid_report(output_root)
    candidate_path = output_root / "candidate" / "safe_tool.py"
    freeze_report = _valid_freeze_comparison_report(output_root)
    freeze_report["candidate_file"] = str(candidate_path)
    freeze_report["candidate_api_surface"] = {"target_file": str(candidate_path), "sha256": "a" * 64}
    freeze_report["passed"] = False
    freeze_report["failed_checks"] = ["function_signature_changed:safe_tool"]
    freeze_report["mandatory_gate"] = {
        "required_before_apply": True,
        "executed": True,
        "passed": False,
    }
    report["freeze_checks"] = freeze_report
    report["candidate_artifact"] = {
        "provided": True,
        "path": str(candidate_path),
        "relative_path": "contract-fixtures/candidate/safe_tool.py",
        "sha256": "a" * 64,
        "bytes": 10,
        "symlink": False,
        "parent_symlink": False,
        "hardlink": False,
    }
    report["passed"] = True
    report["failed_checks"] = []

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "top-level passed must be false when freeze comparator fails" in validation.errors
    assert "top-level failed_checks must include freeze_comparator entries when freeze comparator fails" in validation.errors

    report["passed"] = False
    report["failed_checks"] = ["freeze_comparator:function_signature_changed:safe_tool"]
    validation = validate_phase4_scaffold_report_contract(report)
    assert validation.passed is True
    assert validation.errors == ()


def test_report_contract_rejects_sensitive_payload_text(tmp_path):
    from evolution.code.report_contract import validate_phase4_scaffold_report_contract

    report = _valid_report(_valid_output_root())
    target_spec = report["target_spec"]
    assert isinstance(target_spec, dict)
    target_spec["notes"] = "sk-" + "x" * 28

    validation = validate_phase4_scaffold_report_contract(report)

    assert validation.passed is False
    assert "report contains sensitive credential-like text" in validation.errors


def _valid_freeze_comparison_report(output_root: Path) -> dict[str, object]:
    output_json = output_root / "freeze-comparison" / "freeze_comparison_report.json"
    return {
        "phase": "4",
        "mode": "freeze-comparator",
        "candidate_only": True,
        "read_only_inputs": True,
        "apply_ready": False,
        "hermes_source_mutation_performed": False,
        "passed": True,
        "failed_checks": [],
        "baseline_file": "/tmp/baseline/tools/safe_tool.py",
        "candidate_file": "/tmp/candidate/tools/safe_tool.py",
        "candidate_api_surface": {"target_file": "/tmp/candidate/tools/safe_tool.py", "sha256": "a" * 64},
        "comparisons": {
            "function_signatures": {"changed": [], "added": [], "removed": []},
            "class_names": {"changed": [], "added": [], "removed": []},
            "decorators": {"changed": [], "added": [], "removed": []},
            "registry_register_calls": {"changed": [], "added": [], "removed": []},
            "public_cli_args": {"changed": [], "added": [], "removed": []},
        },
        "artifacts": {"output_json": str(output_json)},
        "output_constraints": {
            "allowed_root": "output/phase4-code-evolution/",
            "fresh_output_required": True,
            "symlink_output_allowed": False,
            "hermes_source_write_allowed": False,
        },
    }


def test_freeze_comparison_report_contract_accepts_valid_candidate_only_report(tmp_path):
    from evolution.code.report_contract import validate_phase4_freeze_comparison_report_contract

    validation = validate_phase4_freeze_comparison_report_contract(
        _valid_freeze_comparison_report(_valid_output_root())
    )

    assert validation.passed is True
    assert validation.errors == ()


def test_freeze_comparison_report_contract_rejects_apply_and_source_mutation_flags(tmp_path):
    from evolution.code.report_contract import validate_phase4_freeze_comparison_report_contract

    report = _valid_freeze_comparison_report(_valid_output_root())
    report["apply_ready"] = True
    report["hermes_source_mutation_performed"] = True

    validation = validate_phase4_freeze_comparison_report_contract(report)

    assert validation.passed is False
    assert "top-level apply_ready must be false" in validation.errors
    assert "top-level hermes_source_mutation_performed must be false" in validation.errors


def test_freeze_comparison_report_contract_rejects_failed_check_mismatch(tmp_path):
    from evolution.code.report_contract import validate_phase4_freeze_comparison_report_contract

    report = _valid_freeze_comparison_report(_valid_output_root())
    report["passed"] = False

    validation = validate_phase4_freeze_comparison_report_contract(report)

    assert validation.passed is False
    assert "failed_checks must be non-empty when passed is false" in validation.errors


def test_freeze_comparison_report_contract_rejects_artifact_outside_phase4_root(tmp_path):
    from evolution.code.report_contract import validate_phase4_freeze_comparison_report_contract

    report = _valid_freeze_comparison_report(_valid_output_root())
    artifacts = report["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["output_json"] = str(tmp_path / "outside.json")

    validation = validate_phase4_freeze_comparison_report_contract(report)

    assert validation.passed is False
    assert "artifacts.output_json must be under output/phase4-code-evolution" in validation.errors

"""Tests for the Phase 2 candidate-only report contract smoke check."""

import json

from click.testing import CliRunner

from evolution.tools.report_contract import (
    main,
    validate_candidate_only_report_contract,
)


def _valid_candidate_only_report() -> dict:
    return {
        "phase": "2D",
        "mode": "candidate-only",
        "apply_ready": False,
        "summary": "Candidate-only report.",
        "candidate_count": 1,
        "metrics": {
            "candidate_only": True,
            "apply_ready": False,
            "case_count": 45,
            "selection_accuracy": 0.8,
            "wrong_tool_avoidance": 0.8,
            "argument_cue_coverage": 1.0,
            "constraint_pass_rate": 1.0,
            "case_results": [],
            "warnings": [],
        },
        "candidates": [
            {
                "name": "read_file",
                "toolset": "file",
                "baseline_description": "Read a text file.",
                "candidate_description": "Read a text file with line numbers.",
                "parameter_descriptions": {"path": "Path to the file."},
                "description_delta": 24,
            }
        ],
        "phase_index_executed": ["2A", "2B", "2C", "2D"],
        "phase2d_gate": {
            "phase": "2D",
            "candidate_only": True,
            "passed": True,
            "thresholds": {
                "min_case_count": 45,
                "min_selection_accuracy": 0.7,
                "min_wrong_tool_avoidance": 0.7,
                "max_per_tool_regression": 0.0,
            },
            "baseline_metrics": {
                "case_count": 45,
                "selection_accuracy": 0.5,
                "wrong_tool_avoidance": 0.5,
                "argument_cue_coverage": 0.7,
                "constraint_pass_rate": 0.9,
                "warning_count": 2,
            },
            "candidate_metrics": {
                "case_count": 45,
                "selection_accuracy": 0.8,
                "wrong_tool_avoidance": 0.8,
                "argument_cue_coverage": 1.0,
                "constraint_pass_rate": 1.0,
                "warning_count": 0,
            },
            "per_tool_regressions": [],
            "failed_checks": [],
        },
        "inventory_metadata": {
            "source": "hermes_repo_import",
            "tool_count": 1,
            "import_warning_count": 1,
            "import_warnings": [
                {
                    "module": "tools.browser_dialog_tool",
                    "message": "Could not import tool module tools.browser_dialog_tool: No module named 'websockets'",
                    "exception": "No module named 'websockets'",
                    "classification": "optional_dependency_import_warning",
                    "candidate_quality": False,
                }
            ],
            "candidate_quality_warnings_are_separate": True,
        },
        "artifacts": {
            "inventory": "inventory.json",
            "candidates": "candidate_descriptions.json",
            "diff": "candidate.diff",
        },
    }


def test_validate_candidate_only_report_contract_accepts_documented_shape():
    result = validate_candidate_only_report_contract(_valid_candidate_only_report())

    assert result.passed is True
    assert result.errors == ()


def test_validate_candidate_only_report_contract_rejects_apply_payloads():
    report = _valid_candidate_only_report()
    report["apply_ready"] = True
    report["patch"] = {"path": "tools/registry.py"}

    result = validate_candidate_only_report_contract(report)

    assert result.passed is False
    assert "top-level apply_ready must be false" in result.errors
    assert "candidate-only report must not contain apply payload key: patch" in result.errors


def test_validate_candidate_only_report_contract_enforces_warning_separation():
    report = _valid_candidate_only_report()
    report["inventory_metadata"]["candidate_quality_warnings_are_separate"] = False
    report["inventory_metadata"]["import_warnings"][0]["candidate_quality"] = True

    result = validate_candidate_only_report_contract(report)

    assert result.passed is False
    assert "inventory_metadata.candidate_quality_warnings_are_separate must be true" in result.errors
    assert "inventory_metadata.import_warnings[0].candidate_quality must be false" in result.errors


def test_validate_report_contract_cli_returns_nonzero_for_invalid_report(tmp_path):
    report_path = tmp_path / "candidate_only_report.json"
    report = _valid_candidate_only_report()
    report["phase2d_gate"]["thresholds"]["min_case_count"] = 12
    report_path.write_text(json.dumps(report))

    result = CliRunner().invoke(main, [str(report_path)])

    assert result.exit_code != 0
    assert "candidate_only_report contract failed" in result.output
    assert "phase2d_gate.thresholds must match the Phase 2D default contract" in result.output

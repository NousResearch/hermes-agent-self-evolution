"""Tests for the LC3 Phase 2 candidate-to-active boundary packet."""

from __future__ import annotations

import json
from pathlib import Path

from evolution.local_completion.phase2_boundary_packet import (
    LC3_PHASE2_BOUNDARY_PHASE,
    PASS_BOUNDARY_REVIEW_READY,
    write_phase2_boundary_packet,
)


def _candidate_report(*, gate_passed: bool = True) -> dict:
    failed_checks = [] if gate_passed else ["selection_accuracy 0.5000 < 0.7000"]
    metric_snapshot = {
        "case_count": 45,
        "selection_accuracy": 1.0 if gate_passed else 0.5,
        "wrong_tool_avoidance": 1.0,
        "argument_cue_coverage": 0.9,
        "constraint_pass_rate": 1.0,
        "warning_count": 0,
    }
    return {
        "phase": "2D",
        "mode": "candidate-only",
        "apply_ready": False,
        "summary": "Candidate-only tool description generation plus Phase 2D gate.",
        "candidate_count": 2,
        "candidates": [
            {
                "name": "read_file",
                "toolset": "file",
                "baseline_description": "Read a text file.",
                "candidate_description": "Read a text file. Prefer over terminal for file inspection.",
                "parameter_descriptions": {"path": "Path to file"},
                "description_delta": 42,
            },
            {
                "name": "search_files",
                "toolset": "file",
                "baseline_description": "Search files.",
                "candidate_description": "Search files by name or content. Prefer over terminal grep.",
                "parameter_descriptions": {},
                "description_delta": 41,
            },
        ],
        "phase_index_executed": ["2A", "2B", "2C", "2D"],
        "metrics": {
            "candidate_only": True,
            "apply_ready": False,
            "case_count": 45,
            "selection_accuracy": metric_snapshot["selection_accuracy"],
            "wrong_tool_avoidance": 1.0,
            "argument_cue_coverage": 0.9,
            "constraint_pass_rate": 1.0,
            "case_results": [],
            "warnings": [],
        },
        "phase2d_gate": {
            "phase": "2D",
            "candidate_only": True,
            "passed": gate_passed,
            "thresholds": {
                "min_case_count": 45,
                "min_selection_accuracy": 0.7,
                "min_wrong_tool_avoidance": 0.7,
                "max_per_tool_regression": 0.0,
            },
            "baseline_metrics": metric_snapshot,
            "candidate_metrics": metric_snapshot,
            "per_tool_regressions": [],
            "failed_checks": failed_checks,
        },
        "inventory_metadata": {
            "source": "inventory_json",
            "tool_count": 2,
            "import_warning_count": 0,
            "import_warnings": [],
            "candidate_quality_warnings_are_separate": True,
        },
        "artifacts": {
            "inventory": "inputs/inventory.json",
            "candidates": "candidates/candidate_descriptions.json",
            "diff": "candidates/candidate.patch",
        },
    }


def test_write_phase2_boundary_packet_preserves_candidate_only_no_apply_invariants(tmp_path):
    report_path = tmp_path / "candidate_only_report.json"
    report_path.write_text(json.dumps(_candidate_report(), indent=2, sort_keys=True) + "\n")

    result = write_phase2_boundary_packet(
        candidate_report_path=report_path,
        output_dir=tmp_path / "lc3-boundary",
        generated_at="2026-06-28T01:41:27Z",
    )

    packet_path = Path(result["packet_path"])
    markdown_path = Path(result["markdown_path"])
    packet = json.loads(packet_path.read_text())

    assert packet["schema_version"] == "hse-local-completion-v1"
    assert packet["gate_id"] == "LC3"
    assert packet["phase"] == LC3_PHASE2_BOUNDARY_PHASE
    assert packet["target"] == "tool-description-active-boundary"
    assert packet["status"] == PASS_BOUNDARY_REVIEW_READY
    assert packet["candidate_only"] is True
    assert packet["apply_ready"] is False
    assert packet["github"] == {
        "queried": False,
        "pr_created": False,
        "push_performed": False,
        "merge_performed": False,
        "publication_deferred": True,
    }
    assert packet["safety_invariants"]["active_tool_schema_modified"] is False
    assert packet["safety_invariants"]["active_runtime_mutation"] is False
    assert packet["report_contract"] == {"passed": True, "errors": []}
    assert packet["phase2d_gate"]["passed"] is True
    assert packet["metrics"]["selection_accuracy"] == 1.0
    assert packet["active_boundary"]["apply_ready_reason"] == "separate human approval required before active schema apply"
    assert "backup active Hermes checkout and schema sources" in packet["active_boundary"]["required_before_active_apply"]
    assert packet["artifacts"]["candidate_report_snapshot"] == "inputs/candidate_only_report.json"
    assert packet["artifacts"]["boundary_markdown"] == "active_boundary_packet.md"
    assert (packet_path.parent / "inputs" / "candidate_only_report.json").exists()
    assert "apply_ready=false" in markdown_path.read_text()
    assert "GitHub/PR work: deferred_not_queried" in markdown_path.read_text()


def test_phase2_boundary_packet_fails_closed_for_invalid_candidate_report(tmp_path):
    invalid = _candidate_report()
    invalid["apply_ready"] = True
    report_path = tmp_path / "candidate_only_report.json"
    report_path.write_text(json.dumps(invalid, indent=2, sort_keys=True) + "\n")

    result = write_phase2_boundary_packet(
        candidate_report_path=report_path,
        output_dir=tmp_path / "lc3-boundary-invalid",
        generated_at="2026-06-28T01:41:27Z",
    )

    packet = json.loads(Path(result["packet_path"]).read_text())
    assert packet["status"] == "BLOCKED_CONTRACT"
    assert packet["candidate_only"] is True
    assert packet["apply_ready"] is False
    assert packet["safety_invariants"]["active_tool_schema_modified"] is False
    assert any("apply_ready must be false" in error for error in packet["report_contract"]["errors"])


def test_phase2_boundary_packet_blocks_when_phase2d_gate_fails(tmp_path):
    report_path = tmp_path / "candidate_only_report.json"
    report_path.write_text(json.dumps(_candidate_report(gate_passed=False), indent=2, sort_keys=True) + "\n")

    result = write_phase2_boundary_packet(
        candidate_report_path=report_path,
        output_dir=tmp_path / "lc3-boundary-gate-fail",
        generated_at="2026-06-28T01:41:27Z",
    )

    packet = json.loads(Path(result["packet_path"]).read_text())
    assert packet["status"] == "BLOCKED_PHASE2D_GATE"
    assert packet["phase2d_gate"]["passed"] is False
    assert packet["active_boundary"]["active_schema_apply_performed"] is False

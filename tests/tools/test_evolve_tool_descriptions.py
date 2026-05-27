"""Tests for Phase 2C/2D candidate-only tool description generation and gating."""

import json

from evolution.tools.evolve_tool_descriptions import (
    generate_candidate_descriptions,
    load_inventory_from_json,
    run_candidate_generation,
)
from evolution.tools.tool_description_eval import (
    ToolInventoryRecord,
    default_tool_selection_cases,
)


def _minimal_inventory_records():
    return [
        ToolInventoryRecord(
            name="read_file",
            toolset="file",
            description="Read a text file with line numbers and pagination.",
            schema={
                "parameters": {
                    "properties": {
                        "path": {"description": "Path to the file"},
                        "offset": {"description": "Line number to start reading from"},
                    }
                }
            },
        ),
        ToolInventoryRecord(
            name="search_files",
            toolset="file",
            description="Search file contents or find files by name.",
            schema={"parameters": {"properties": {}}},
        ),
        ToolInventoryRecord(
            name="terminal",
            toolset="terminal",
            description="Execute shell commands on the VM.",
            schema={"parameters": {"properties": {}}},
        ),
    ]


def test_candidate_generation_adds_golden_case_cues_but_keeps_baseline_metadata():
    records = _minimal_inventory_records()
    candidates = generate_candidate_descriptions(records, default_tool_selection_cases())
    by_name = {candidate.name: candidate for candidate in candidates}

    read_file = by_name["read_file"]
    assert read_file.baseline_description == "Read a text file with line numbers and pagination."
    assert read_file.candidate_description != read_file.baseline_description
    assert "Prefer over terminal" in read_file.candidate_description
    assert len(read_file.candidate_description) <= 500
    assert read_file.parameter_descriptions["path"] == "Path to the file"

    terminal = by_name["terminal"]
    assert "Prefer over execute_code" in terminal.candidate_description
    assert len(terminal.candidate_description) <= 500


def test_candidate_generation_reserves_budget_for_golden_cues_on_long_baselines():
    records = [
        ToolInventoryRecord(
            name="session_search",
            toolset="session_search",
            description=" ".join(["Search past sessions and local conversation history."] * 20),
            schema={"parameters": {"properties": {}}},
        )
    ]

    candidates = generate_candidate_descriptions(records, default_tool_selection_cases(), max_description_chars=220)

    candidate = candidates[0]
    assert len(candidate.candidate_description) <= 220
    assert "previous" in candidate.candidate_description
    assert "left" in candidate.candidate_description
    assert "Prefer over" in candidate.candidate_description
    assert "browser_navigate" in candidate.candidate_description


def test_load_inventory_from_json_accepts_candidate_report_shape(tmp_path):
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"tools": [record.__dict__ for record in _minimal_inventory_records()]}))

    records = load_inventory_from_json(inventory_path)

    assert [record.name for record in records] == ["read_file", "search_files", "terminal"]
    assert records[0].schema["parameters"]["properties"]["path"]["description"] == "Path to the file"


def test_load_inventory_from_json_accepts_empty_inventory_container(tmp_path):
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"tools": []}))

    assert load_inventory_from_json(inventory_path) == []


def test_run_candidate_generation_writes_candidate_only_artifacts(tmp_path):
    inventory_path = tmp_path / "inventory.json"
    output_dir = tmp_path / "phase2c"
    inventory_path.write_text(json.dumps({"tools": [record.__dict__ for record in _minimal_inventory_records()]}))

    result = run_candidate_generation(inventory_json=inventory_path, output_dir=output_dir)

    assert result.report_path.exists()
    assert result.candidates_path.exists()
    assert result.inventory_path.exists()
    assert result.diff_path.exists()

    report = json.loads(result.report_path.read_text())
    assert report["phase"] == "2D"
    assert report["mode"] == "candidate-only"
    assert report["apply_ready"] is False
    assert "patch" not in report
    assert "write_paths" not in report
    assert report["metrics"]["case_count"] >= 30
    assert report["phase2d_gate"]["phase"] == "2D"
    assert report["phase_index_executed"] == ["2A", "2B", "2C", "2D"]
    assert report["phase2d_gate"]["thresholds"]["min_case_count"] == 30
    assert report["phase2d_gate"]["candidate_metrics"]["case_count"] >= 30
    assert report["phase2d_gate"]["candidate_only"] is True
    assert "read_file" in result.diff_path.read_text()

"""Tests for Phase 2C/2D candidate-only tool description generation and gating."""

import json
import sys
from pathlib import Path
from typing import cast

import pytest
from click.testing import CliRunner

from evolution.tools import evolve_tool_descriptions as evolve_module
from evolution.tools.evolve_tool_descriptions import (
    InventoryImportWarning,
    ToolInventoryCollectionResult,
    collect_hermes_tool_inventory_with_metadata,
    generate_candidate_descriptions,
    load_inventory_from_json,
    main,
    run_candidate_generation,
)
from evolution.tools.report_contract import validate_candidate_only_report_contract
from evolution.tools.tool_description_eval import (
    ToolInventoryRecord,
    ToolSelectionCase,
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


def _write_phase5_provenance_inventory(path: Path) -> None:
    """Write a committed-free deterministic inventory for the Phase 5 regression test."""

    cases = default_tool_selection_cases()
    tool_names = sorted({case.expected_tool for case in cases} | {tool for case in cases for tool in case.confusing_tools})
    requests_by_tool = {name: [] for name in tool_names}
    for case in cases:
        requests_by_tool[case.expected_tool].append(case.user_request)

    records = [
        ToolInventoryRecord(
            name=name,
            toolset=name.split("_", 1)[0],
            description="; ".join(requests_by_tool[name]) or f"{name} Hermes tool.",
            schema={"parameters": {"properties": {}}},
        )
        for name in tool_names
    ]
    path.write_text(json.dumps({"tools": [record.__dict__ for record in records]}, indent=2, sort_keys=True) + "\n")


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


def test_candidate_generation_rejects_private_request_signal_cues():
    records = [
        ToolInventoryRecord(
            name="read_file",
            toolset="file",
            description="Read file lines.",
            schema={"parameters": {"properties": {}}},
        ),
        ToolInventoryRecord(
            name="terminal",
            toolset="terminal",
            description="Execute shell commands.",
            schema={"parameters": {"properties": {}}},
        ),
    ]
    private_cases = (
        ToolSelectionCase(
            user_request="Read /Users/example/.env and OPENAI_API_KEY without using terminal.",
            expected_tool="read_file",
            confusing_tools=("terminal",),
            required_cues=("read", "file"),
            category="privacy-regression",
        ),
    )

    with pytest.raises(ValueError, match="private/raw identifier"):
        generate_candidate_descriptions(records, private_cases)


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


def test_candidate_generation_normalizes_overlong_parameter_descriptions_before_reporting(tmp_path):
    long_parameter_description = " ".join(["Long parameter guidance for candidate-only report normalization."] * 8)
    records = [
        ToolInventoryRecord(
            name="read_file",
            toolset="file",
            description="Read file lines.",
            schema={
                "parameters": {
                    "properties": {
                        "path": {"description": long_parameter_description},
                    }
                }
            },
        )
    ]
    inventory_path = tmp_path / "inventory.json"
    output_dir = tmp_path / "phase2e"
    inventory_path.write_text(json.dumps({"tools": [record.__dict__ for record in records]}))

    result = run_candidate_generation(inventory_json=inventory_path, output_dir=output_dir, cases=[])

    candidates = json.loads(result.candidates_path.read_text())
    normalized = candidates[0]["parameter_descriptions"]["path"]
    assert 0 < len(normalized) <= 200
    assert normalized.endswith("…")

    report = json.loads(result.report_path.read_text())
    assert not any(
        warning.startswith("Parameter description length constraint failed")
        for warning in report["metrics"]["warnings"]
    )


def test_run_candidate_generation_default_output_uses_local_candidate_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("HSE_RUNS_ROOT", str(tmp_path / "runs"))
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"tools": [record.__dict__ for record in _minimal_inventory_records()]}))

    result = run_candidate_generation(inventory_json=inventory_path, cases=[])

    assert result.output_dir.parent == tmp_path / "runs"
    assert result.inventory_path == result.output_dir / "inputs" / "inventory.json"
    assert result.candidates_path == result.output_dir / "candidates" / "candidate_descriptions.json"
    assert result.diff_path == result.output_dir / "candidates" / "candidate.patch"
    assert result.report_path == result.output_dir / "reports" / "candidate_only_report.json"
    decision = json.loads((result.output_dir / "decision.json").read_text())
    assert decision["schema_version"] == "hse-local-candidate-bundle-v1"
    assert decision["candidate_only"] is True
    assert decision["apply_ready"] is False
    assert decision["github"]["pr_created"] is False
    assert decision["artifacts"]["patch"] == "candidates/candidate.patch"


def test_run_candidate_generation_records_inventory_import_warnings_as_metadata_not_quality_warnings(
    tmp_path,
    monkeypatch,
):
    warning = InventoryImportWarning(
        module="tools.browser_dialog_tool",
        message="Could not import tool module tools.browser_dialog_tool: No module named 'websockets'",
        exception="No module named 'websockets'",
        classification="optional_dependency_import_warning",
        candidate_quality=False,
    )
    monkeypatch.setattr(
        evolve_module,
        "collect_hermes_tool_inventory_with_metadata",
        lambda hermes_repo=None: ToolInventoryCollectionResult(
            records=_minimal_inventory_records(),
            import_warnings=(warning,),
        ),
    )

    result = run_candidate_generation(hermes_repo=tmp_path / "hermes", output_dir=tmp_path / "phase2e", cases=[])

    report = json.loads(result.report_path.read_text())
    metadata = report["inventory_metadata"]
    assert metadata["source"] == "hermes_repo_import"
    assert metadata["import_warning_count"] == 1
    assert metadata["import_warnings"] == [
        {
            "module": "tools.browser_dialog_tool",
            "message": "Could not import tool module tools.browser_dialog_tool: No module named 'websockets'",
            "exception": "No module named 'websockets'",
            "classification": "optional_dependency_import_warning",
            "candidate_quality": False,
        }
    ]
    assert not any("browser_dialog_tool" in warning for warning in report["metrics"]["warnings"])
    assert report["phase2d_gate"]["candidate_metrics"]["warning_count"] == len(report["metrics"]["warnings"])


def test_collect_hermes_tool_inventory_captures_registry_import_warnings_as_metadata(tmp_path):
    repo = tmp_path / "hermes-agent"
    tools_dir = repo / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "__init__.py").write_text("")
    (tools_dir / "registry.py").write_text(
        """
import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)
registry = SimpleNamespace(
    get_all_tool_names=lambda: ["read_file"],
    get_entry=lambda name: SimpleNamespace(
        name="read_file",
        toolset="file",
        description="Read files.",
        schema={"parameters": {"properties": {}}},
    ) if name == "read_file" else None,
)

def discover_builtin_tools(tools_dir):
    logger.warning(
        "Could not import tool module %s: %s",
        "tools.browser_dialog_tool",
        ModuleNotFoundError("No module named 'websockets'"),
    )
    return []
""".lstrip()
    )
    for module_name in list(sys.modules):
        if module_name == "tools" or module_name.startswith("tools."):
            sys.modules.pop(module_name, None)

    try:
        collection = collect_hermes_tool_inventory_with_metadata(repo)
    finally:
        for module_name in list(sys.modules):
            if module_name == "tools" or module_name.startswith("tools."):
                sys.modules.pop(module_name, None)

    assert [record.name for record in collection.records] == ["read_file"]
    assert collection.import_warnings == (
        InventoryImportWarning(
            module="tools.browser_dialog_tool",
            message="Could not import tool module tools.browser_dialog_tool: No module named 'websockets'",
            exception="No module named 'websockets'",
            classification="optional_dependency_import_warning",
            candidate_quality=False,
        ),
    )


def test_load_inventory_from_json_accepts_candidate_report_shape(tmp_path):
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"tools": [record.__dict__ for record in _minimal_inventory_records()]}))

    records = load_inventory_from_json(inventory_path)

    assert [record.name for record in records] == ["read_file", "search_files", "terminal"]
    schema = cast(dict[str, object], records[0].schema)
    parameters = cast(dict[str, object], schema["parameters"])
    properties = cast(dict[str, object], parameters["properties"])
    path_spec = cast(dict[str, str], properties["path"])
    assert path_spec["description"] == "Path to the file"


def test_load_inventory_from_json_accepts_empty_inventory_container(tmp_path):
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"tools": []}))

    assert load_inventory_from_json(inventory_path) == []


def test_run_candidate_generation_writes_candidate_only_artifacts(tmp_path):
    inventory_path = tmp_path / "inventory.json"
    output_dir = tmp_path / "phase2d"
    inventory_path.write_text(json.dumps({"tools": [record.__dict__ for record in _minimal_inventory_records()]}))

    result = run_candidate_generation(inventory_json=inventory_path, output_dir=output_dir)

    assert result.report_path.exists()
    assert result.candidates_path.exists()
    assert result.inventory_path.exists()
    assert result.diff_path.exists()

    report = json.loads(result.report_path.read_text())
    contract_validation = validate_candidate_only_report_contract(report)
    assert contract_validation.passed, contract_validation.errors
    assert report["phase"] == "2D"
    assert report["mode"] == "candidate-only"
    assert report["apply_ready"] is False
    assert "patch" not in report
    assert "write_paths" not in report
    assert report["metrics"]["case_count"] >= 45
    assert report["phase2d_gate"]["phase"] == "2D"
    assert report["phase_index_executed"] == ["2A", "2B", "2C", "2D"]
    assert report["phase2d_gate"]["thresholds"]["min_case_count"] == 45
    assert report["phase2d_gate"]["candidate_metrics"]["case_count"] >= 45
    assert report["phase2d_gate"]["candidate_only"] is True
    assert "read_file" in result.diff_path.read_text()


def test_phase5_provenance_inventory_candidate_generation_clears_tool_selection_threshold(tmp_path):
    inventory_path = tmp_path / "phase5-provenance-inventory.json"
    _write_phase5_provenance_inventory(inventory_path)

    result = run_candidate_generation(
        inventory_json=inventory_path,
        output_dir=tmp_path / "phase2e-provenance",
    )

    report = json.loads(result.report_path.read_text())
    metrics = report["metrics"]
    failed_cases = [case for case in metrics["case_results"] if not case["passed"]]

    assert metrics["case_count"] == 45
    assert metrics["selection_accuracy"] == 1.0
    assert metrics["wrong_tool_avoidance"] == 1.0
    assert failed_cases == []
    assert report["phase2d_gate"]["candidate_metrics"]["selection_accuracy"] == 1.0
    assert report["phase2d_gate"]["candidate_metrics"]["wrong_tool_avoidance"] == 1.0


def test_cli_returns_nonzero_when_phase2d_gate_fails(tmp_path):
    inventory_path = tmp_path / "inventory.json"
    output_dir = tmp_path / "phase2d-fail"
    incomplete_inventory = [
        ToolInventoryRecord(
            name="read_file",
            toolset="file",
            description="Read file lines.",
            schema={"parameters": {"properties": {}}},
        )
    ]
    inventory_path.write_text(json.dumps({"tools": [record.__dict__ for record in incomplete_inventory]}))

    result = CliRunner().invoke(
        main,
        ["--inventory-json", str(inventory_path), "--output-dir", str(output_dir)],
    )

    assert result.exit_code != 0
    assert "Phase 2D gate failed" in result.output
    assert "selection_accuracy 0.0000 < 0.7000" in result.output
    assert "wrong_tool_avoidance 0.0000 < 0.7000" in result.output
    assert str(output_dir / "candidate_only_report.json") in result.output
    report = json.loads((output_dir / "candidate_only_report.json").read_text())
    assert report["phase2d_gate"]["passed"] is False
    assert report["phase2d_gate"]["failed_checks"]
    contract_validation = validate_candidate_only_report_contract(report)
    assert contract_validation.passed, contract_validation.errors

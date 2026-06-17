"""Tests for Phase 2E held-out tool-selection review."""

import json
import tomllib
from pathlib import Path

from click.testing import CliRunner

from evolution.tools.heldout_tool_selection_review import (
    build_holdout_review_report,
    load_candidate_descriptions,
    main,
    run_holdout_review,
)
from evolution.tools.tool_description_eval import (
    ToolDescriptionCandidate,
    ToolInventoryRecord,
    ToolSelectionCase,
)


def _minimal_holdout_records() -> list[ToolInventoryRecord]:
    return [
        ToolInventoryRecord(
            name="read_file",
            toolset="file",
            description="Read local log file lines with line-number pagination; use instead of shell tail.",
            schema={"parameters": {"properties": {"path": {"description": "Path to the file."}}}},
        ),
        ToolInventoryRecord(
            name="terminal",
            toolset="terminal",
            description="Run shell commands for builds and tests.",
            schema={"parameters": {"properties": {}}},
        ),
    ]


def _holdout_cases() -> tuple[ToolSelectionCase, ...]:
    return (
        ToolSelectionCase(
            user_request="Show the last 40 lines of a local log file without using shell tail.",
            expected_tool="read_file",
            confusing_tools=("terminal",),
            required_cues=("read", "log", "lines", "tail"),
            required_arguments=("path", "limit"),
            category="sessiondb-misfire-log-tail-vs-read-file",
        ),
    )


def test_load_candidate_descriptions_round_trips_candidate_json(tmp_path):
    candidates_path = tmp_path / "candidate_descriptions.json"
    candidates_path.write_text(
        json.dumps(
            [
                {
                    "name": "read_file",
                    "toolset": "file",
                    "baseline_description": "Open and display local file text.",
                    "candidate_description": "Read local file text with line-number pagination; use instead of shell tail.",
                    "parameter_descriptions": {"path": "Path to the file."},
                    "description_delta": 38,
                }
            ]
        )
    )

    loaded = load_candidate_descriptions(candidates_path)

    assert loaded == (
        ToolDescriptionCandidate(
            name="read_file",
            toolset="file",
            baseline_description="Open and display local file text.",
            candidate_description="Read local file text with line-number pagination; use instead of shell tail.",
            parameter_descriptions={"path": "Path to the file."},
        ),
    )


def test_build_holdout_review_report_compares_baseline_candidate_and_regressions():
    baseline = [
        ToolDescriptionCandidate(
            name="read_file",
            toolset="file",
            baseline_description="Open and display local file text.",
            candidate_description="Open and display local file text.",
        ),
        ToolDescriptionCandidate(
            name="terminal",
            toolset="terminal",
            baseline_description="Run shell commands such as cat, head, tail, grep, and tests.",
            candidate_description="Run shell commands such as cat, head, tail, grep, and tests.",
        ),
    ]
    candidate = [
        ToolDescriptionCandidate(
            name="read_file",
            toolset="file",
            baseline_description="Open and display local file text.",
            candidate_description="Read local log file lines with line-number pagination; use instead of shell cat, head, or tail.",
            parameter_descriptions={"path": "Path to the file.", "limit": "Maximum number of lines to read."},
        ),
        baseline[1],
    ]

    report = build_holdout_review_report(
        baseline,
        candidate,
        _holdout_cases(),
        holdout_source="session_misfire_holdout.jsonl",
        default_gate_case_count=45,
    )

    assert report["phase"] == "2E"
    assert report["mode"] == "candidate-only-heldout-review"
    assert report["apply_ready"] is False
    assert report["passed"] is True
    assert report["failed_checks"] == []
    assert report["holdout"]["case_count"] == 1
    assert report["holdout"]["included_in_default_gate"] is False
    assert report["holdout"]["promotion_decision"] == "holdout"
    assert report["metric_deltas"]["selection_accuracy"] >= 0
    assert report["metric_deltas"]["wrong_tool_avoidance"] >= 0
    assert report["per_tool_regressions"][0]["expected_tool"] == "read_file"
    assert report["per_tool_regressions"][0]["passed"] is True


def test_build_holdout_review_report_fails_empty_holdout_cases():
    baseline = [
        ToolDescriptionCandidate(
            name="read_file",
            toolset="file",
            baseline_description="Read files.",
            candidate_description="Read files.",
        )
    ]

    report = build_holdout_review_report(baseline, baseline, (), holdout_source="empty.jsonl")

    assert report["passed"] is False
    assert "case_count 0 < 1" in report["failed_checks"]


def test_build_holdout_review_report_reports_per_tool_regression_failure():
    baseline = [
        ToolDescriptionCandidate(
            name="read_file",
            toolset="file",
            baseline_description="Read local log file lines with pagination; use instead of shell tail.",
            candidate_description="Read local log file lines with pagination; use instead of shell tail.",
        ),
        ToolDescriptionCandidate(
            name="terminal",
            toolset="terminal",
            baseline_description="Run shell commands for builds and tests.",
            candidate_description="Run shell commands for builds and tests.",
        ),
    ]
    candidate = [
        ToolDescriptionCandidate(
            name="read_file",
            toolset="file",
            baseline_description="Open documents.",
            candidate_description="Open documents.",
        ),
        ToolDescriptionCandidate(
            name="terminal",
            toolset="terminal",
            baseline_description="Run shell commands such as cat, head, tail, grep, and tests.",
            candidate_description="Run shell commands such as cat, head, tail, grep, and tests.",
        ),
    ]

    report = build_holdout_review_report(baseline, candidate, _holdout_cases(), holdout_source="holdout.jsonl")

    assert report["passed"] is False
    assert any(check.startswith("per_tool_regression read_file") for check in report["failed_checks"])


def test_build_holdout_review_report_fails_missing_holdout_tool_coverage():
    case = ToolSelectionCase(
        user_request="Fetch an image from a web page.",
        expected_tool="browser_get_images",
        confusing_tools=("browser_snapshot",),
        required_cues=("image",),
        category="missing-tool-coverage",
    )

    report = build_holdout_review_report([], [], (case,), holdout_source="holdout.jsonl")

    assert report["passed"] is False
    assert any(str(check).startswith("holdout_tool_coverage baseline") for check in report["failed_checks"])
    assert any(str(check).startswith("holdout_tool_coverage candidate") for check in report["failed_checks"])


def test_build_holdout_review_report_fails_overlong_candidate_description():
    baseline = [
        ToolDescriptionCandidate(
            name="read_file",
            toolset="file",
            baseline_description="Read local log file lines with pagination; use instead of shell tail.",
            candidate_description="Read local log file lines with pagination; use instead of shell tail.",
        ),
        ToolDescriptionCandidate(
            name="terminal",
            toolset="terminal",
            baseline_description="Run shell commands for builds and tests.",
            candidate_description="Run shell commands for builds and tests.",
        ),
    ]
    candidate = [
        ToolDescriptionCandidate(
            name="read_file",
            toolset="file",
            baseline_description="Read local log file lines with pagination; use instead of shell tail.",
            candidate_description="x" * 501,
            parameter_descriptions={"path": "Path to the file."},
        ),
        baseline[1],
    ]

    report = build_holdout_review_report(baseline, candidate, _holdout_cases(), holdout_source="holdout.jsonl")

    assert report["passed"] is False
    assert any(
        str(check).startswith("candidate_constraint description_length read_file")
        for check in report["failed_checks"]
    )


def test_run_holdout_review_rejects_candidate_inventory_mismatch(tmp_path):
    inventory_path = tmp_path / "inventory.json"
    candidates_path = tmp_path / "candidate_descriptions.json"
    cases_path = tmp_path / "holdout.jsonl"
    output_path = tmp_path / "heldout_review.json"
    records = _minimal_holdout_records()
    inventory_path.write_text(json.dumps({"tools": [record.__dict__ for record in records]}))
    candidates_path.write_text(
        json.dumps(
            [
                {
                    "name": "read_file",
                    "toolset": "file",
                    "baseline_description": "Read local log file lines.",
                    "candidate_description": "Read local log file lines.",
                    "parameter_descriptions": {},
                }
            ]
        )
    )
    cases_path.write_text(json.dumps(_holdout_cases()[0].__dict__) + "\n")

    try:
        run_holdout_review(
            inventory_json=inventory_path,
            candidates_json=candidates_path,
            cases_jsonl=cases_path,
            output_json=output_path,
        )
    except ValueError as error:
        assert "missing=['terminal']" in str(error)
    else:
        raise AssertionError("expected inventory/candidate mismatch to fail")


def test_holdout_review_cli_writes_report_and_fails_on_regression(tmp_path):
    inventory_path = tmp_path / "inventory.json"
    candidates_path = tmp_path / "candidate_descriptions.json"
    cases_path = tmp_path / "holdout.jsonl"
    output_path = tmp_path / "heldout_review.json"

    records = _minimal_holdout_records()
    inventory_path.write_text(json.dumps({"tools": [record.__dict__ for record in records]}))
    candidates_path.write_text(
        json.dumps(
            [
                {
                    "name": "read_file",
                    "toolset": "file",
                    "baseline_description": "Read local log file lines with line-number pagination; use instead of shell tail.",
                    "candidate_description": "Open documents.",
                    "parameter_descriptions": {},
                },
                {
                    "name": "terminal",
                    "toolset": "terminal",
                    "baseline_description": "Run shell commands for builds and tests.",
                    "candidate_description": "Show the last lines of a local log file; use for tail-style log inspection.",
                    "parameter_descriptions": {},
                },
            ]
        )
    )
    cases_path.write_text(json.dumps(_holdout_cases()[0].__dict__) + "\n")

    result = CliRunner().invoke(
        main,
        [
            "--inventory-json",
            str(inventory_path),
            "--candidates-json",
            str(candidates_path),
            "--cases-jsonl",
            str(cases_path),
            "--output-json",
            str(output_path),
        ],
    )

    assert result.exit_code != 0
    assert output_path.exists()
    assert "Phase 2E held-out review failed" in result.output
    report = json.loads(output_path.read_text())
    assert report["passed"] is False
    assert any(check.startswith("aggregate_regression selection_accuracy") for check in report["failed_checks"])


def test_pyproject_exposes_holdout_review_cli_script():
    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["hse-review-tool-holdout"] == (
        "evolution.tools.heldout_tool_selection_review:main"
    )

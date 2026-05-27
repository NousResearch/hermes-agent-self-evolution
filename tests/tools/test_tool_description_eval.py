"""Tests for Phase 2B tool-description evaluation scaffold."""

from evolution.tools.tool_description_eval import (
    ToolDescriptionCandidate,
    ToolInventoryRecord,
    ToolSelectionCase,
    build_candidate_only_report,
    candidates_from_inventory,
    default_tool_selection_cases,
    evaluate_candidate_descriptions,
    write_default_golden_cases,
)


def test_default_tool_selection_cases_cover_confusing_tool_pairs():
    cases = default_tool_selection_cases()

    assert len(cases) >= 30
    assert all(case.expected_tool for case in cases)
    assert any("read_file" == case.expected_tool and "terminal" in case.confusing_tools for case in cases)
    assert any("search_files" == case.expected_tool and "terminal" in case.confusing_tools for case in cases)
    assert any("browser" in case.category for case in cases)
    assert any("session" in case.category for case in cases)
    assert any("session_search" == case.expected_tool and "browser_navigate" in case.confusing_tools for case in cases)


def test_default_tool_selection_cases_cover_phase2b_plus_focus_set():
    cases = default_tool_selection_cases()
    expected_tools = {case.expected_tool for case in cases}
    confusion_pairs = {
        (case.expected_tool, confusing_tool)
        for case in cases
        for confusing_tool in case.confusing_tools
    }

    for expected_tool in (
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_console",
        "browser_vision",
        "computer_use",
        "execute_code",
        "terminal",
        "session_search",
    ):
        assert expected_tool in expected_tools

    for pair in (
        ("computer_use", "browser_click"),
        ("execute_code", "terminal"),
        ("terminal", "execute_code"),
        ("session_search", "browser_navigate"),
        ("browser_click", "computer_use"),
        ("browser_snapshot", "computer_use"),
    ):
        assert pair in confusion_pairs


def test_evaluator_rewards_expected_tool_over_confusing_tool():
    candidates = [
        ToolDescriptionCandidate(
            name="read_file",
            toolset="file",
            baseline_description="Read a text file with line numbers and pagination.",
            candidate_description="Read file contents with line numbers; use this instead of cat/head/tail.",
        ),
        ToolDescriptionCandidate(
            name="terminal",
            toolset="terminal",
            baseline_description="Execute shell commands.",
            candidate_description="Run builds, tests, git, package managers, and processes; do not use for reading files.",
        ),
    ]
    cases = [
        ToolSelectionCase(
            user_request="show me the first 40 lines of README.md",
            expected_tool="read_file",
            confusing_tools=("terminal",),
            required_cues=("file", "line", "cat"),
        )
    ]

    result = evaluate_candidate_descriptions(candidates, cases)

    assert result.candidate_only is True
    assert result.selection_accuracy == 1.0
    assert result.wrong_tool_avoidance == 1.0
    assert result.argument_cue_coverage > 0.0
    assert not result.apply_ready


def test_evaluator_flags_missing_candidate_and_never_marks_apply_ready():
    candidates = [
        ToolDescriptionCandidate(
            name="terminal",
            toolset="terminal",
            baseline_description="Execute shell commands.",
            candidate_description="Run commands.",
        )
    ]
    cases = [
        ToolSelectionCase(
            user_request="read config.py",
            expected_tool="read_file",
            confusing_tools=("terminal",),
        )
    ]

    result = evaluate_candidate_descriptions(candidates, cases)

    assert result.selection_accuracy == 0.0
    assert result.constraint_pass_rate < 1.0
    assert result.apply_ready is False
    assert any("missing expected tool" in warning.lower() for warning in result.warnings)


def test_candidate_only_report_is_json_serializable_and_has_no_apply_payload():
    candidates = [
        ToolDescriptionCandidate(
            name="search_files",
            toolset="file",
            baseline_description="Search file contents or find files by name.",
            candidate_description="Search file contents or file names; use instead of grep, rg, find, or ls.",
        )
    ]
    report = build_candidate_only_report(candidates, default_tool_selection_cases())

    assert report["mode"] == "candidate-only"
    assert report["apply_ready"] is False
    assert "patch" not in report
    assert "write_paths" not in report
    assert report["metrics"]["case_count"] >= 5


def test_candidates_from_inventory_preserves_baseline_and_parameter_descriptions():
    records = [
        ToolInventoryRecord(
            name="read_file",
            toolset="file",
            description="Read a text file.",
            schema={
                "parameters": {
                    "properties": {
                        "path": {"description": "Path to the text file"},
                        "limit": {"description": "Maximum number of lines"},
                    }
                }
            },
        )
    ]

    candidates = candidates_from_inventory(records)

    assert candidates == [
        ToolDescriptionCandidate(
            name="read_file",
            toolset="file",
            baseline_description="Read a text file.",
            candidate_description="Read a text file.",
            parameter_descriptions={"path": "Path to the text file", "limit": "Maximum number of lines"},
        )
    ]


def test_write_default_golden_cases_creates_jsonl(tmp_path):
    output = tmp_path / "tool_selection.jsonl"

    written = write_default_golden_cases(output)

    lines = written.read_text().strip().splitlines()
    assert written == output
    assert len(lines) == len(default_tool_selection_cases())
    assert '"expected_tool": "read_file"' in lines[0]

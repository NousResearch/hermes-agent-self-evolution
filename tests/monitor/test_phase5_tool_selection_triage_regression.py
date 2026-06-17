"""CI-robust Phase 5 P1 tool-selection GREEN regression tests.

These tests generate deterministic candidate-only evidence under pytest ``tmp_path``
instead of reading the ignored local ``output/`` tree. They must not mutate active
Hermes runtime state, resume cron, run optimizers, publish PRs, or write repo-local
candidate artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolution.tools.evolve_tool_descriptions import run_candidate_generation
from evolution.tools.tool_description_eval import ToolInventoryRecord, default_tool_selection_cases

MIN_SELECTION_ACCURACY = 0.90
MIN_SCORE_MARGIN = 0.02

CRITICAL_ROW_EXPECTATIONS = {
    "tool-selection-002": {"expected_tool": "search_files", "min_score_margin": MIN_SCORE_MARGIN},
    "tool-selection-003": {"expected_tool": "terminal", "min_score_margin": MIN_SCORE_MARGIN},
    "tool-selection-004": {"expected_tool": "patch", "min_score_margin": MIN_SCORE_MARGIN},
    "tool-selection-016": {"expected_tool": "read_file", "min_score_margin": MIN_SCORE_MARGIN},
    "tool-selection-028": {"expected_tool": "terminal", "min_score_margin": MIN_SCORE_MARGIN, "min_cue_coverage": 0.90},
}


def _write_phase5_ci_inventory(path: Path) -> None:
    """Write a deterministic privacy-safe inventory for the Phase 5 regression."""

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


def _load_ci_candidate_report(tmp_path: Path) -> dict[str, Any]:
    inventory_path = tmp_path / "phase5-p1-tool-selection-inventory.json"
    output_dir = tmp_path / "phase5-p1-tool-selection-run"
    _write_phase5_ci_inventory(inventory_path)

    result = run_candidate_generation(inventory_json=inventory_path, output_dir=output_dir)

    assert result.report_path == output_dir / "candidate_only_report.json"
    assert result.candidates_path == output_dir / "candidate_descriptions.json"
    assert result.diff_path == output_dir / "candidate.diff"
    assert "output/tool-description" not in result.report_path.as_posix()
    return json.loads(result.report_path.read_text())


def _case_results_by_row_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    case_results = report["metrics"]["case_results"]
    return {f"tool-selection-{index:03d}": case for index, case in enumerate(case_results, start=1)}


def _top_confusing_score(case: dict[str, Any]) -> float:
    confusing_scores = case.get("confusing_scores", {})
    return max((float(score) for score in confusing_scores.values()), default=0.0)


def _score_margin(case: dict[str, Any]) -> float:
    return round(float(case.get("expected_score", 0.0)) - _top_confusing_score(case), 10)


def _non_passing_row_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row_id, case in _case_results_by_row_id(report).items():
        if case.get("passed"):
            continue
        rows.append(
            {
                "row_id": row_id,
                "expected_tool": case.get("expected_tool"),
                "selected_tool": case.get("selected_tool"),
                "score_margin": _score_margin(case),
                "cue_coverage": case.get("cue_coverage"),
            }
        )
    return rows


def test_phase5_tool_selection_ci_candidate_report_clears_minimum_threshold(tmp_path: Path):
    ci_candidate_report = _load_ci_candidate_report(tmp_path)
    metrics = ci_candidate_report["metrics"]

    assert ci_candidate_report["apply_ready"] is False
    assert metrics["candidate_only"] is True
    assert metrics["case_count"] == 45
    assert metrics["selection_accuracy"] >= MIN_SELECTION_ACCURACY, (
        "CI-generated candidate-only tool-selection report remains below the Phase 5 P1 minimum threshold: "
        f"selection_accuracy={metrics['selection_accuracy']:.4f} < {MIN_SELECTION_ACCURACY:.4f}; "
        f"non_passing_rows={_non_passing_row_summary(ci_candidate_report)}"
    )
    assert _non_passing_row_summary(ci_candidate_report) == []


def test_phase5_tool_selection_critical_rows_have_correct_tool_and_margin(tmp_path: Path):
    ci_candidate_report = _load_ci_candidate_report(tmp_path)
    rows_by_id = _case_results_by_row_id(ci_candidate_report)
    failures: list[str] = []

    for row_id, expectation in CRITICAL_ROW_EXPECTATIONS.items():
        case = rows_by_id[row_id]
        expected_tool = expectation["expected_tool"]
        selected_tool = case.get("selected_tool")
        score_margin = _score_margin(case)
        cue_coverage = float(case.get("cue_coverage", 0.0))

        if selected_tool != expected_tool:
            failures.append(f"{row_id}: selected_tool={selected_tool!r}, expected {expected_tool!r}")
        if score_margin <= float(expectation["min_score_margin"]):
            failures.append(
                f"{row_id}: score_margin={score_margin:.4f}, expected > {expectation['min_score_margin']:.4f}"
            )
        if "min_cue_coverage" in expectation and cue_coverage < float(expectation["min_cue_coverage"]):
            failures.append(
                f"{row_id}: cue_coverage={cue_coverage:.4f}, expected >= {expectation['min_cue_coverage']:.4f}"
            )

    assert not failures, "Critical CI-generated tool-selection rows still need GREEN remediation: " + "; ".join(failures)

"""Tests for Phase 2E expanded holdout closeout decision."""

import json
import tomllib
from pathlib import Path

from click.testing import CliRunner

from evolution.tools.expanded_holdout_decision import (
    build_expanded_holdout_decision,
    main,
    render_decision_markdown,
    run_expanded_holdout_decision,
)
from evolution.tools.tool_description_eval import ToolSelectionCase, default_tool_selection_cases, load_tool_selection_cases


REPO_ROOT = Path(__file__).resolve().parents[2]
SESSIONDB_HOLDOUT = REPO_ROOT / "datasets" / "golden" / "tool-description" / "session_misfire_holdout.jsonl"
COMMITTED_DECISION_JSON = REPO_ROOT / "reports" / "phase2e_expanded_holdout_decision.json"
COMMITTED_DECISION_MD = REPO_ROOT / "reports" / "phase2e_expanded_holdout_decision.md"


def _passing_holdout_review() -> dict[str, object]:
    return {
        "phase": "2E",
        "mode": "candidate-only-heldout-review",
        "candidate_only": True,
        "apply_ready": False,
        "passed": True,
        "failed_checks": [],
        "holdout": {"case_count": 9, "included_in_default_gate": False},
        "baseline_metrics": {
            "case_count": 9,
            "selection_accuracy": 0.1111,
            "wrong_tool_avoidance": 0.1111,
            "argument_cue_coverage": 0.7989,
            "constraint_pass_rate": 0.9818,
            "warning_count": 14,
        },
        "candidate_metrics": {
            "case_count": 9,
            "selection_accuracy": 0.8889,
            "wrong_tool_avoidance": 0.8889,
            "argument_cue_coverage": 0.7487,
            "constraint_pass_rate": 1.0,
            "warning_count": 0,
        },
        "metric_deltas": {
            "selection_accuracy": 0.7778,
            "wrong_tool_avoidance": 0.7778,
            "argument_cue_coverage": -0.0502,
            "constraint_pass_rate": 0.0182,
        },
        "per_tool_regressions": [],
    }


def test_expanded_holdout_decision_accepts_45_plus_9_without_100_case_requirement():
    default_cases = default_tool_selection_cases()
    holdout_cases = load_tool_selection_cases(SESSIONDB_HOLDOUT)

    report = build_expanded_holdout_decision(default_cases, holdout_cases, heldout_review=_passing_holdout_review())

    assert report["phase"] == "2E"
    assert report["mode"] == "expanded-holdout-decision"
    assert report["candidate_only"] is True
    assert report["apply_ready"] is False
    assert report["decision"] == "current_45_plus_9_sufficient_for_phase2_closeout"
    assert report["requires_100_case_slice_before_phase2_closeout"] is False
    assert report["default_gate"]["case_count"] == 45
    assert report["sessiondb_holdout"]["case_count"] == 9
    assert report["combined_slice"]["case_count"] == 54
    assert report["coverage_delta"]["new_expected_tools_from_holdout"] == []
    assert report["coverage_delta"]["new_confusion_pairs_from_holdout"] == []
    assert report["evidence"]["holdout_review_passed"] is True
    assert report["evidence"]["holdout_review_contract_ok"] is True
    assert "benchmark_gate_decision" in report["remaining_phase2_closeout_items"]
    assert "human_review_checkpoint" in report["remaining_phase2_closeout_items"]


def test_expanded_holdout_decision_requires_100_plus_when_holdout_adds_uncovered_confusion_pair():
    default_cases = (
        ToolSelectionCase(
            user_request="Read a local file.",
            expected_tool="read_file",
            confusing_tools=("terminal",),
            category="file-read-vs-shell",
        ),
    )
    holdout_cases = (
        ToolSelectionCase(
            user_request="Generate an image from a prompt.",
            expected_tool="image_generate",
            confusing_tools=("vision_analyze",),
            category="new-image-generation-confusion",
        ),
    )

    report = build_expanded_holdout_decision(default_cases, holdout_cases, heldout_review=_passing_holdout_review())

    assert report["decision"] == "build_100_plus_heldout_quality_slice_before_phase2_closeout"
    assert report["requires_100_case_slice_before_phase2_closeout"] is True
    assert report["coverage_delta"]["new_expected_tools_from_holdout"] == ["image_generate"]
    assert report["coverage_delta"]["new_confusion_pairs_from_holdout"] == [["image_generate", "vision_analyze"]]
    assert "100_plus_heldout_quality_slice" in report["remaining_phase2_closeout_items"]


def test_expanded_holdout_decision_requires_100_plus_when_review_contract_is_not_candidate_only():
    bad_review = dict(_passing_holdout_review())
    bad_review["apply_ready"] = True

    report = build_expanded_holdout_decision(
        default_tool_selection_cases(),
        load_tool_selection_cases(SESSIONDB_HOLDOUT),
        heldout_review=bad_review,
    )

    assert report["decision"] == "build_100_plus_heldout_quality_slice_before_phase2_closeout"
    assert report["requires_100_case_slice_before_phase2_closeout"] is True
    assert report["evidence"]["holdout_review_passed"] is True
    assert report["evidence"]["holdout_review_contract_ok"] is False
    assert "100_plus_heldout_quality_slice" in report["remaining_phase2_closeout_items"]


def test_expanded_holdout_decision_fails_closed_on_malformed_review_failed_checks():
    malformed_values = (None, "", "still bad", {"bad": "shape"}, ["selection_accuracy regression"])
    for malformed in malformed_values:
        bad_review = dict(_passing_holdout_review())
        if malformed is None:
            bad_review.pop("failed_checks")
        else:
            bad_review["failed_checks"] = malformed

        report = build_expanded_holdout_decision(
            default_tool_selection_cases(),
            load_tool_selection_cases(SESSIONDB_HOLDOUT),
            heldout_review=bad_review,
        )

        assert report["decision"] == "build_100_plus_heldout_quality_slice_before_phase2_closeout"
        assert report["requires_100_case_slice_before_phase2_closeout"] is True
        assert report["evidence"]["holdout_review_contract_ok"] is False
        if isinstance(malformed, list):
            assert report["evidence"]["holdout_failed_checks_contract_ok"] is True
            assert report["evidence"]["holdout_failed_checks"] == malformed
        else:
            assert report["evidence"]["holdout_failed_checks_contract_ok"] is False
            assert report["evidence"]["holdout_failed_checks"] == []


def test_render_decision_markdown_records_candidate_only_safety_and_defer_policy():
    report = build_expanded_holdout_decision(
        default_tool_selection_cases(),
        load_tool_selection_cases(SESSIONDB_HOLDOUT),
        heldout_review=_passing_holdout_review(),
    )

    markdown = render_decision_markdown(report)

    assert "# Phase 2E Expanded Holdout Decision" in markdown
    assert "Decision: current 45+9 slice is sufficient for Phase 2 closeout." in markdown
    assert "100+ held-out quality slice required before Phase 2 closeout: no" in markdown
    assert "Candidate-only/no-apply: yes" in markdown
    assert "benchmark_gate_decision" in markdown


def test_run_expanded_holdout_decision_writes_json_and_markdown(tmp_path):
    review_path = tmp_path / "heldout_review.json"
    output_json = tmp_path / "expanded_holdout_decision.json"
    output_md = tmp_path / "expanded_holdout_decision.md"
    review_path.write_text(json.dumps(_passing_holdout_review()))

    result = run_expanded_holdout_decision(
        holdout_jsonl=SESSIONDB_HOLDOUT,
        heldout_review_json=review_path,
        output_json=output_json,
        output_markdown=output_md,
    )

    assert result.output_json == output_json
    assert result.output_markdown == output_md
    assert result.requires_100_case_slice is False
    report = json.loads(output_json.read_text())
    assert report["combined_slice"]["case_count"] == 54
    assert "current 45+9 slice is sufficient" in output_md.read_text()


def test_expanded_holdout_decision_cli_writes_report(tmp_path):
    review_path = tmp_path / "heldout_review.json"
    output_json = tmp_path / "expanded_holdout_decision.json"
    output_md = tmp_path / "expanded_holdout_decision.md"
    review_path.write_text(json.dumps(_passing_holdout_review()))

    result = CliRunner().invoke(
        main,
        [
            "--holdout-jsonl",
            str(SESSIONDB_HOLDOUT),
            "--heldout-review-json",
            str(review_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )

    assert result.exit_code == 0
    assert "Phase 2E expanded holdout decision written" in result.output
    assert "requires 100+ heldout before Phase 2 closeout: no" in result.output
    assert json.loads(output_json.read_text())["decision"] == "current_45_plus_9_sufficient_for_phase2_closeout"
    assert output_md.exists()


def test_pyproject_exposes_expanded_holdout_decision_cli_script():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["hse-decide-tool-holdout"] == (
        "evolution.tools.expanded_holdout_decision:main"
    )


def test_committed_expanded_holdout_decision_artifact_matches_closeout_policy():
    report = json.loads(COMMITTED_DECISION_JSON.read_text())
    markdown = COMMITTED_DECISION_MD.read_text()
    regenerated = build_expanded_holdout_decision(
        default_tool_selection_cases(),
        load_tool_selection_cases(SESSIONDB_HOLDOUT),
        heldout_review=_passing_holdout_review(),
    )

    assert report == regenerated
    assert markdown == render_decision_markdown(regenerated)
    assert report["decision"] == "current_45_plus_9_sufficient_for_phase2_closeout"
    assert report["requires_100_case_slice_before_phase2_closeout"] is False
    assert report["combined_slice"]["case_count"] == 54
    assert report["candidate_only"] is True
    assert report["apply_ready"] is False
    assert "100+ held-out quality slice required before Phase 2 closeout: no" in markdown

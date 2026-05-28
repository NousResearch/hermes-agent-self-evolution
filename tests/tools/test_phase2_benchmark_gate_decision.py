"""Tests for the Phase 2E benchmark-gate defer decision artifacts."""

import json
from pathlib import Path

from evolution.tools.tool_description_eval import default_tool_selection_cases, load_tool_selection_cases

REPO_ROOT = Path(__file__).resolve().parents[2]
DECISION_JSON = REPO_ROOT / "reports" / "phase2e_benchmark_gate_decision.json"
DECISION_MD = REPO_ROOT / "reports" / "phase2e_benchmark_gate_decision.md"
EXPANDED_HOLDOUT_DECISION_JSON = REPO_ROOT / "reports" / "phase2e_expanded_holdout_decision.json"
SESSIONDB_HOLDOUT = REPO_ROOT / "datasets" / "golden" / "tool-description" / "session_misfire_holdout.jsonl"
PLAN_MD = REPO_ROOT / "PLAN.md"
README_MD = REPO_ROOT / "README.md"


def test_phase2e_benchmark_gate_decision_artifact_records_defer_policy():
    report = json.loads(DECISION_JSON.read_text())

    assert report["phase"] == "2E"
    assert report["mode"] == "benchmark-gate-decision"
    assert report["decision"] == "defer_benchmark_gate_until_phase3_or_active_apply"
    assert report["benchmark_gate_executed_now"] is False
    assert report["blocking_for_phase2_candidate_only_closeout"] is False
    assert report["candidate_only"] is True
    assert report["apply_ready"] is False
    assert report["required_before"] == [
        "phase3_execution",
        "active_tool_schema_apply",
        "default_gate_promotion",
        "system_prompt_evolution_acceptance",
    ]
    assert "TBLite" in report["deferred_benchmarks"]
    assert "YC-Bench" in report["deferred_benchmarks"]
    expanded_holdout_decision = json.loads(EXPANDED_HOLDOUT_DECISION_JSON.read_text())
    default_cases = default_tool_selection_cases()
    holdout_cases = load_tool_selection_cases(SESSIONDB_HOLDOUT)

    assert report["current_evidence"]["default_gate_case_count"] == len(default_cases) == 45
    assert report["current_evidence"]["sessiondb_holdout_case_count"] == len(holdout_cases) == 9
    assert report["current_evidence"]["combined_tool_selection_slice_case_count"] == len(default_cases) + len(holdout_cases)
    assert report["current_evidence"]["phase2e_expanded_holdout_requires_100_plus_slice"] == expanded_holdout_decision[
        "requires_100_case_slice_before_phase2_closeout"
    ]
    assert expanded_holdout_decision["requires_100_case_slice_before_phase2_closeout"] is False
    assert report["remaining_phase2_closeout_items"] == ["human_review_checkpoint"]
    assert report["safety_invariants"] == {
        "candidate_only": True,
        "apply_ready": False,
        "active_tool_schema_mutation": False,
        "raw_session_data_committed": False,
    }


def test_phase2e_benchmark_gate_decision_markdown_matches_json_summary():
    report = json.loads(DECISION_JSON.read_text())
    markdown = DECISION_MD.read_text()

    assert "# Phase 2E Benchmark Gate Decision" in markdown
    assert "Decision: defer benchmark gate until Phase 3 or active apply." in markdown
    assert "Benchmark gate executed now: no" in markdown
    assert "Blocking for Phase 2 candidate-only closeout: no" in markdown
    assert "TBLite" in markdown
    assert "YC-Bench" in markdown
    assert "45-case formal gate" in markdown
    assert "9-case SessionDB holdout" in markdown
    assert str(report["current_evidence"]["combined_tool_selection_slice_case_count"]) in markdown
    assert "human_review_checkpoint" in markdown


def test_plan_and_readme_record_benchmark_gate_defer_decision():
    plan = PLAN_MD.read_text()
    readme = README_MD.read_text()

    assert "**Benchmark gate decision:** completed" in plan
    assert "reports/phase2e_benchmark_gate_decision.json" in plan
    assert "TBLite/YC-Bench" in plan
    assert "deferred until Phase 3 execution, active tool-schema apply, default-gate promotion" in plan
    assert "**Human review checkpoint:**" in plan

    assert "Phase 2E benchmark gate decision" in readme
    assert "reports/phase2e_benchmark_gate_decision.md" in readme
    assert "deferred until Phase 3 execution or active apply" in readme

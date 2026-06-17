"""Tests for the Phase 2E human-review checkpoint artifacts."""

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_JSON = REPO_ROOT / "reports" / "phase2e_human_review_checkpoint.json"
CHECKPOINT_MD = REPO_ROOT / "reports" / "phase2e_human_review_checkpoint.md"
EXPANDED_HOLDOUT_DECISION_JSON = REPO_ROOT / "reports" / "phase2e_expanded_holdout_decision.json"
BENCHMARK_DECISION_JSON = REPO_ROOT / "reports" / "phase2e_benchmark_gate_decision.json"
PLAN_MD = REPO_ROOT / "PLAN.md"
README_MD = REPO_ROOT / "README.md"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "phase2-tool-description-gate.yml"


def test_phase2e_human_review_checkpoint_records_safe_closeout():
    checkpoint = json.loads(CHECKPOINT_JSON.read_text())
    expanded_holdout = json.loads(EXPANDED_HOLDOUT_DECISION_JSON.read_text())
    benchmark_decision = json.loads(BENCHMARK_DECISION_JSON.read_text())

    assert checkpoint["phase"] == "2E"
    assert checkpoint["mode"] == "human-review-checkpoint"
    assert checkpoint["checkpoint_status"] == "recorded"
    assert checkpoint["reviewer"] == "Sunwoo"
    assert checkpoint["authorization"] == "rec action GO"
    assert checkpoint["phase2e_closeout_complete"] is True
    assert checkpoint["candidate_only"] is True
    assert checkpoint["apply_ready"] is False
    assert checkpoint["active_schema_or_source_apply_approved"] is False
    assert checkpoint["active_schema_or_source_apply_requires_separate_approval"] is True
    assert checkpoint["phase3_execution_requires_benchmark_gate"] is True
    assert checkpoint["remaining_phase2_closeout_items"] == []
    assert checkpoint["reviewed_decisions"]["expanded_holdout_decision"] == expanded_holdout["decision"]
    assert checkpoint["reviewed_decisions"]["benchmark_gate_decision"] == benchmark_decision["decision"]
    assert checkpoint["reviewed_decisions"]["benchmark_gate_executed_now"] is False
    assert checkpoint["reviewed_decisions"]["expanded_holdout_requires_100_plus_slice"] is False
    assert checkpoint["safety_invariants"] == {
        "candidate_only": True,
        "apply_ready": False,
        "active_tool_schema_mutation": False,
        "raw_session_data_committed": False,
        "active_schema_or_source_apply_approved": False,
    }
    assert set(checkpoint["reviewed_artifacts"]) >= {
        "README.md",
        "PLAN.md",
        "reports/phase2e_expanded_holdout_decision.json",
        "reports/phase2e_benchmark_gate_decision.json",
        "datasets/golden/tool-description/tool_selection.jsonl",
        "datasets/golden/tool-description/session_misfire_holdout.jsonl",
        ".github/workflows/phase2-tool-description-gate.yml",
    }


def test_phase2e_human_review_checkpoint_markdown_mirrors_json():
    checkpoint = json.loads(CHECKPOINT_JSON.read_text())
    markdown = CHECKPOINT_MD.read_text()

    assert "# Phase 2E Human Review Checkpoint" in markdown
    assert "Checkpoint status: recorded" in markdown
    assert "Reviewer: Sunwoo" in markdown
    assert "Authorization: rec action GO" in markdown
    assert "Phase 2E closeout complete: yes" in markdown
    assert "Candidate-only/no-apply: yes" in markdown
    assert "Active schema/source apply approved: no" in markdown
    assert "Separate approval/PR or patch required for active apply: yes" in markdown
    assert "Phase 3 execution requires benchmark gate: yes" in markdown
    assert checkpoint["reviewed_decisions"]["benchmark_gate_decision"] in markdown


def test_plan_readme_and_workflow_record_human_review_checkpoint():
    plan = PLAN_MD.read_text()
    readme = README_MD.read_text()
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    assert isinstance(workflow, dict)
    run_blocks = "\n".join(
        str(step.get("run", ""))
        for step in workflow["jobs"]["phase2-tool-description-gate"]["steps"]
        if isinstance(step, dict)
    )

    assert "**Human review checkpoint:** completed" in plan
    assert "reports/phase2e_human_review_checkpoint.json" in plan
    assert "Active schema/source apply remains a separate human-approved PR or patch" in plan
    assert "Phase 2 closeout passed" in plan

    assert "Phase 2E human review checkpoint" in readme
    assert "reports/phase2e_human_review_checkpoint.md" in readme
    assert "active schema/source apply remains separate" in readme

    assert "tests/tools/test_phase2_human_review_checkpoint.py" in run_blocks

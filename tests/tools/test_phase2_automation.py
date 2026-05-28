"""Tests for Phase 2E tool-description CI/automation readiness."""

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "phase2-tool-description-gate.yml"


def _workflow_run_blocks(workflow: dict[str, Any]) -> str:
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict)
    gate_job = jobs.get("phase2-tool-description-gate")
    assert isinstance(gate_job, dict)
    assert gate_job.get("timeout-minutes") == 15
    steps = gate_job.get("steps")
    assert isinstance(steps, list)
    return "\n".join(str(step.get("run", "")) for step in steps if isinstance(step, dict))


def test_phase2_tool_description_gate_workflow_wires_generator_validator_and_45_case_assertions():
    assert WORKFLOW_PATH.exists(), "Phase 2E readiness requires an automation/CI workflow"
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    assert isinstance(workflow, dict)

    run_blocks = _workflow_run_blocks(workflow)
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    for trigger_name in ("pull_request", "push"):
        trigger = triggers.get(trigger_name)
        assert isinstance(trigger, dict)
        paths = trigger.get("paths")
        assert isinstance(paths, list)
        assert "reports/phase2e_*" in paths
        assert "reports/phase3_*" in paths
        assert "seeds/**" in paths

    permissions = workflow.get("permissions")
    assert isinstance(permissions, dict)
    assert permissions.get("contents") == "read"
    uses_values = str(workflow)
    assert "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5" in uses_values
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" in uses_values

    assert "tests/tools/test_tool_description_eval.py" in run_blocks
    assert "tests/tools/test_evolve_tool_descriptions.py" in run_blocks
    assert "tests/tools/test_report_contract.py" in run_blocks
    assert "tests/tools/test_heldout_tool_selection_review.py" in run_blocks
    assert "tests/tools/test_expanded_holdout_decision.py" in run_blocks
    assert "tests/tools/test_phase2_benchmark_gate_decision.py" in run_blocks
    assert "tests/tools/test_phase2_human_review_checkpoint.py" in run_blocks
    assert "tests/tools/test_phase3_system_prompt_evolution_plan.py" in run_blocks
    assert "tests/tools/test_phase3_execution_seed_draft.py" in run_blocks
    assert "tests/tools/test_phase2_automation.py" in run_blocks
    assert "python -m evolution.tools.evolve_tool_descriptions" in run_blocks
    assert "python -m evolution.tools.report_contract" in run_blocks
    assert "python -m evolution.tools.heldout_tool_selection_review" in run_blocks
    assert "python -m evolution.tools.expanded_holdout_decision" in run_blocks
    assert "datasets/golden/tool-description/session_misfire_holdout.jsonl" in run_blocks
    assert "expanded_holdout_decision.json" in run_blocks
    assert "requires_100_case_slice_before_phase2_closeout" in run_blocks
    assert "new_expected_tools_from_holdout" in run_blocks
    assert "new_confusion_pairs_from_holdout" in run_blocks
    assert "default_tool_selection_cases" in run_blocks
    assert "min_case_count" in run_blocks
    assert "45" in run_blocks
    assert "phase2d_gate" in run_blocks
    assert "passed" in run_blocks
    assert "apply_ready" in run_blocks

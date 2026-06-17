"""Tests for the Phase 3 system-prompt evolution design/Seed artifacts."""

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_JSON = REPO_ROOT / "reports" / "phase3_system_prompt_evolution_plan.json"
PLAN_REPORT_MD = REPO_ROOT / "reports" / "phase3_system_prompt_evolution_plan.md"
SEED_YAML = REPO_ROOT / "seeds" / "phase3_system_prompt_evolution_seed.yaml"
PHASE2_CHECKPOINT_JSON = REPO_ROOT / "reports" / "phase2e_human_review_checkpoint.json"
BENCHMARK_DECISION_JSON = REPO_ROOT / "reports" / "phase2e_benchmark_gate_decision.json"
PLAN_MD = REPO_ROOT / "PLAN.md"
README_MD = REPO_ROOT / "README.md"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "phase2-tool-description-gate.yml"

EXPECTED_EVOLVABLE_SECTIONS = {
    "DEFAULT_AGENT_IDENTITY",
    "MEMORY_GUIDANCE",
    "SESSION_SEARCH_GUIDANCE",
    "SKILLS_GUIDANCE",
    "PLATFORM_HINTS",
}

EXPECTED_AC_IDS = {
    "AC1_phase2_closeout_checkpoint_recorded",
    "AC2_design_only_no_runtime_mutation",
    "AC3_evolvable_sections_scoped",
    "AC4_private_or_generated_sections_excluded",
    "AC5_behavioral_evaluator_defined",
    "AC6_benchmark_gate_reactivation_before_execution",
    "AC7_human_approval_before_apply",
    "AC8_prompt_cache_and_identity_safety_preserved",
}

EXPECTED_BEHAVIORAL_DIMENSIONS = {
    "tool_use_discipline",
    "memory_usage",
    "session_search_triggering",
    "skill_loading",
    "platform_formatting",
    "identity_and_safety",
    "prompt_cache_compatibility",
}



def test_phase3_design_plan_artifact_records_scope_acceptance_and_benchmark_boundary():
    plan = json.loads(PLAN_JSON.read_text())
    phase2_checkpoint = json.loads(PHASE2_CHECKPOINT_JSON.read_text())
    benchmark_decision = json.loads(BENCHMARK_DECISION_JSON.read_text())

    assert plan["phase"] == "3"
    assert plan["mode"] == "system-prompt-evolution-design-plan"
    assert plan["status"] == "planned_not_executed"
    assert plan["plan_only"] is True
    assert plan["execution_started"] is False
    assert plan["active_system_prompt_apply_approved"] is False
    assert plan["phase2e_closeout_dependency"] == "reports/phase2e_human_review_checkpoint.json"
    assert phase2_checkpoint["phase2e_closeout_complete"] is True

    assert set(plan["evolvable_sections"]) == EXPECTED_EVOLVABLE_SECTIONS
    assert "durable_user_or_private_memory" in plan["non_evolvable_sections"]
    assert "auto_generated_skills_index" in plan["non_evolvable_sections"]
    assert "project_context_files" in plan["non_evolvable_sections"]
    assert "canonical_identity_or_soul_artifacts" in plan["non_evolvable_sections"]

    ac_ids = {criterion["id"] for criterion in plan["acceptance_criteria"]}
    assert ac_ids == EXPECTED_AC_IDS
    for criterion in plan["acceptance_criteria"]:
        assert criterion["status"] == "planned"
        assert criterion["verification"]

    benchmark = plan["benchmark_gate_reactivation"]
    assert benchmark["blocking_for_design_only_plan"] is False
    assert benchmark["blocking_for_phase3_execution"] is True
    assert benchmark["deferred_benchmarks"] == ["TBLite", "YC-Bench"]
    assert benchmark["required_before"] == [
        "phase3_execution",
        "system_prompt_evolution_acceptance",
        "active_system_prompt_apply",
        "default_gate_promotion",
    ]
    assert benchmark_decision["benchmark_gate_executed_now"] is False
    assert benchmark_decision["required_before"][0] == "phase3_execution"

    assert plan["safety_invariants"] == {
        "plan_only": True,
        "execution_started": False,
        "active_system_prompt_mutation": False,
        "active_system_prompt_apply_approved": False,
        "raw_private_session_data_committed": False,
        "secrets_or_credentials_required": False,
    }



def test_phase3_seed_matches_plan_and_is_design_only():
    plan = json.loads(PLAN_JSON.read_text())
    seed = yaml.safe_load(SEED_YAML.read_text())

    assert seed["phase"] == "3"
    assert seed["task_type"] == "analysis"
    assert seed["execution_mode"] == "design_only"
    assert seed["goal"] == plan["goal"]
    assert seed["plan_artifact"] == "reports/phase3_system_prompt_evolution_plan.json"
    assert seed["apply_ready"] is False
    assert seed["run_gepa_now"] is False
    assert seed["mutate_active_system_prompt_now"] is False
    assert seed["requires_human_approval_before_apply"] is True

    seed_ac_ids = {criterion["id"] for criterion in seed["acceptance_criteria"]}
    plan_ac_ids = {criterion["id"] for criterion in plan["acceptance_criteria"]}
    assert seed_ac_ids == plan_ac_ids == EXPECTED_AC_IDS
    assert seed["benchmark_gate_reactivation"]["required_before"] == plan["benchmark_gate_reactivation"]["required_before"]
    assert set(seed["evaluation_plan"]["behavioral_dimensions"]) == EXPECTED_BEHAVIORAL_DIMENSIONS
    assert seed["evaluation_plan"]["benchmark_dimensions"] == [
        "TBLite_regression_check",
        "YC_Bench_coherence_check",
    ]
    assert seed["evaluation_plan"]["human_review_required"] is True
    assert seed["constraints"]["no_active_prompt_or_source_apply"] is True
    assert seed["constraints"]["no_raw_private_session_data"] is True
    assert seed["constraints"]["benchmark_gate_required_before_execution"] is True



def test_phase3_markdown_plan_readme_and_plan_md_record_design_boundary():
    markdown = PLAN_REPORT_MD.read_text()
    plan_md = PLAN_MD.read_text()
    readme = README_MD.read_text()

    assert "# Phase 3 System Prompt Evolution Design Plan" in markdown
    assert "Status: planned, not executed" in markdown
    assert "Design-only boundary: no active system-prompt mutation or apply" in markdown
    assert "Benchmark gate reactivation is required before Phase 3 execution" in markdown
    assert "Acceptance criteria" in markdown
    assert "seeds/phase3_system_prompt_evolution_seed.yaml" in markdown

    assert "**Current Phase 3 design status:** planned, not executed" in plan_md
    assert "reports/phase3_system_prompt_evolution_plan.json" in plan_md
    assert "seeds/phase3_system_prompt_evolution_seed.yaml" in plan_md
    assert "benchmark gates are reactivated before Phase 3 execution" in plan_md
    assert "active system-prompt/source apply remains out of scope" in plan_md

    assert "Phase 3 system prompt evolution design plan" in readme
    assert "reports/phase3_system_prompt_evolution_plan.md" in readme
    assert "seeds/phase3_system_prompt_evolution_seed.yaml" in readme
    assert "not an execution/apply Seed" in readme



def test_phase3_plan_artifacts_are_wired_into_ci_paths_and_focused_tests():
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    assert isinstance(workflow, dict)
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    for trigger_name in ("pull_request", "push"):
        trigger = triggers.get(trigger_name)
        assert isinstance(trigger, dict)
        paths = trigger.get("paths")
        assert isinstance(paths, list)
        assert "reports/phase3_*" in paths
        assert "seeds/**" in paths

    run_blocks = "\n".join(
        str(step.get("run", ""))
        for step in workflow["jobs"]["phase2-tool-description-gate"]["steps"]
        if isinstance(step, dict)
    )
    assert "tests/tools/test_phase3_system_prompt_evolution_plan.py" in run_blocks

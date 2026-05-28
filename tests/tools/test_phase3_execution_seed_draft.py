"""Tests for the Phase 3 system-prompt execution Seed draft contract."""

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PLAN_JSON = REPO_ROOT / "reports" / "phase3_system_prompt_evolution_plan.json"
DESIGN_SEED_YAML = REPO_ROOT / "seeds" / "phase3_system_prompt_evolution_seed.yaml"
EXECUTION_SEED_YAML = REPO_ROOT / "seeds" / "phase3_system_prompt_evolution_execution_seed_draft.yaml"
EXECUTION_DRAFT_JSON = REPO_ROOT / "reports" / "phase3_execution_seed_draft.json"
EXECUTION_DRAFT_MD = REPO_ROOT / "reports" / "phase3_execution_seed_draft.md"
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
EXPECTED_BENCHMARKS = {"TBLite", "YC-Bench"}
EXPECTED_BEHAVIORAL_DIMENSIONS = {
    "tool_use_discipline",
    "memory_usage",
    "session_search_triggering",
    "skill_loading",
    "platform_formatting",
    "identity_and_safety",
    "prompt_cache_compatibility",
}


def test_phase3_execution_seed_draft_is_non_executing_and_depends_on_design_plan():
    design_plan = json.loads(DESIGN_PLAN_JSON.read_text())
    design_seed = yaml.safe_load(DESIGN_SEED_YAML.read_text())
    checkpoint = json.loads(PHASE2_CHECKPOINT_JSON.read_text())
    execution_seed = yaml.safe_load(EXECUTION_SEED_YAML.read_text())

    assert checkpoint["phase2e_closeout_complete"] is True
    assert execution_seed["phase"] == "3"
    assert execution_seed["task_type"] == "analysis"
    assert execution_seed["execution_mode"] == "execution_seed_draft_only"
    assert execution_seed["status"] == "drafted_not_executed"
    assert execution_seed["source_design_plan"] == "reports/phase3_system_prompt_evolution_plan.json"
    assert execution_seed["source_design_seed"] == "seeds/phase3_system_prompt_evolution_seed.yaml"
    assert execution_seed["phase2e_closeout_dependency"] == "reports/phase2e_human_review_checkpoint.json"
    assert execution_seed["source_authorization"] == "rec action GO"

    assert execution_seed["goal"] == "Draft the future Phase 3 system prompt evolution execution contract without running GEPA/DSPy or applying prompt/source changes."
    assert execution_seed["execution_started"] is False
    assert execution_seed["run_gepa_now"] is False
    assert execution_seed["run_dspy_now"] is False
    assert execution_seed["mutate_active_system_prompt_now"] is False
    assert execution_seed["active_system_prompt_apply_approved"] is False
    assert execution_seed["apply_ready"] is False
    assert execution_seed["requires_human_approval_before_execution"] is True
    assert execution_seed["requires_human_approval_before_apply"] is True

    assert set(execution_seed["evolvable_sections"]) == set(design_plan["evolvable_sections"]) == EXPECTED_EVOLVABLE_SECTIONS
    assert execution_seed["non_evolvable_sections"] == design_seed["non_evolvable_sections"]
    assert execution_seed["evaluation_plan"]["behavioral_dimensions"] == design_seed["evaluation_plan"]["behavioral_dimensions"]
    assert set(execution_seed["evaluation_plan"]["behavioral_dimensions"]) == EXPECTED_BEHAVIORAL_DIMENSIONS


def test_phase3_execution_seed_draft_fixes_benchmark_command_contract_before_execution():
    benchmark_decision = json.loads(BENCHMARK_DECISION_JSON.read_text())
    execution_seed = yaml.safe_load(EXECUTION_SEED_YAML.read_text())

    benchmark_gate = execution_seed["benchmark_gate"]
    assert benchmark_gate["status"] == "required_not_run"
    assert benchmark_gate["required_before_execution"] is True
    assert benchmark_gate["blocking_for_execution"] is True
    assert benchmark_gate["source_decision"] == "reports/phase2e_benchmark_gate_decision.json"
    assert benchmark_gate["deferred_benchmarks"] == benchmark_decision["deferred_benchmarks"] == ["TBLite", "YC-Bench"]
    assert benchmark_gate["required_before"] == [
        "phase3_execution",
        "system_prompt_evolution_acceptance",
        "active_system_prompt_apply",
        "default_gate_promotion",
    ]

    command_names = {command["name"] for command in benchmark_gate["command_templates"]}
    assert command_names == EXPECTED_BENCHMARKS
    for command in benchmark_gate["command_templates"]:
        assert command["status"] == "draft_not_executed"
        assert command["must_pass_before"] == "phase3_execution"
        assert "python -m evolution.benchmarks" in command["command"]
        assert "--baseline-prompt" in command["command"]
        assert "--candidate-prompt" in command["command"]
        assert command["output_json"].startswith("output/phase3-system-prompt/")
        assert command["pass_condition"] in {
            "no_regression_against_baseline",
            "coherence_score_holds_or_improves",
        }


def test_phase3_execution_seed_draft_records_rollback_and_human_approval_gates():
    execution_seed = yaml.safe_load(EXECUTION_SEED_YAML.read_text())

    rollback = execution_seed["rollback_boundary"]
    assert rollback["checkpoint_required_before_candidate_generation"] is True
    assert rollback["baseline_prompt_snapshot_required"] is True
    assert rollback["baseline_prompt_checksum_required"] is True
    assert rollback["candidate_output_dir"] == "output/phase3-system-prompt/<run-id>/"
    assert rollback["rollback_handle_required_before_apply"] is True
    assert rollback["active_runtime_rollback_required_before_apply"] is True
    assert rollback["allowed_write_roots_before_approval"] == [
        "output/phase3-system-prompt/<run-id>/",
        "reports/phase3_execution_seed_draft.json",
        "reports/phase3_execution_seed_draft.md",
    ]
    assert "~/.hermes/SOUL.md" in rollback["prohibited_targets_before_separate_approval"]
    assert "~/.hermes/config.yaml" in rollback["prohibited_targets_before_separate_approval"]
    assert "~/.hermes/hermes-agent/agent/prompt_builder.py" in rollback["prohibited_targets_before_separate_approval"]
    assert "~/.hermes/skills/" in rollback["prohibited_targets_before_separate_approval"]
    assert "~/.hermes/memories/" in rollback["prohibited_targets_before_separate_approval"]
    assert "~/.hermes/profiles/" in rollback["prohibited_targets_before_separate_approval"]

    human_gate = execution_seed["human_approval_gate"]
    assert human_gate["planning_authorized"] == "rec action GO"
    assert human_gate["execution_approved"] is False
    assert human_gate["active_apply_approved"] is False
    assert human_gate["separate_approval_required_before"] == [
        "running GEPA/DSPy optimization",
        "running TBLite/YC-Bench benchmark commands",
        "editing Hermes Agent prompt source",
        "applying evolved prompt to active runtime",
        "default-gate promotion",
    ]


def test_phase3_execution_seed_draft_report_docs_and_ci_wiring_are_present():
    report = json.loads(EXECUTION_DRAFT_JSON.read_text())
    markdown = EXECUTION_DRAFT_MD.read_text()
    plan_md = PLAN_MD.read_text()
    readme = README_MD.read_text()
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())

    assert report["phase"] == "3"
    assert report["mode"] == "phase3-execution-seed-draft"
    assert report["status"] == "drafted_not_executed"
    assert report["execution_seed"] == "seeds/phase3_system_prompt_evolution_execution_seed_draft.yaml"
    assert report["execution_started"] is False
    assert report["run_gepa_now"] is False
    assert report["run_dspy_now"] is False
    assert report["mutate_active_system_prompt_now"] is False
    assert report["active_system_prompt_apply_approved"] is False
    assert report["apply_ready"] is False
    assert report["requires_human_approval_before_execution"] is True
    assert report["requires_human_approval_before_apply"] is True
    assert report["benchmark_gate"]["required_before_execution"] is True
    assert report["human_approval_gate"]["execution_approved"] is False
    assert report["human_approval_gate"]["active_apply_approved"] is False
    assert report["rollback_boundary"]["rollback_handle_required_before_apply"] is True

    assert "# Phase 3 Execution Seed Draft" in markdown
    assert "Status: drafted, not executed" in markdown
    assert "GEPA/DSPy execution: not started" in markdown
    assert "Benchmark commands are contract templates and have not been run" in markdown
    assert "Rollback boundary" in markdown
    assert "Human approval gate" in markdown

    assert "**Current Phase 3 execution Seed draft status:** drafted, not executed" in plan_md
    assert "seeds/phase3_system_prompt_evolution_execution_seed_draft.yaml" in plan_md
    assert "reports/phase3_execution_seed_draft.json" in plan_md
    assert "benchmark command templates" in plan_md
    assert "rollback boundary" in plan_md

    assert "Phase 3 execution Seed draft" in readme
    assert "reports/phase3_execution_seed_draft.md" in readme
    assert "benchmark command templates" in readme
    assert "human approval gate" in readme

    triggers = workflow.get("on", workflow.get(True))
    for trigger_name in ("pull_request", "push"):
        paths = triggers[trigger_name]["paths"]
        assert "reports/phase3_*" in paths
        assert "seeds/**" in paths

    run_blocks = "\n".join(
        str(step.get("run", ""))
        for step in workflow["jobs"]["phase2-tool-description-gate"]["steps"]
        if isinstance(step, dict)
    )
    assert "tests/tools/test_phase3_execution_seed_draft.py" in run_blocks

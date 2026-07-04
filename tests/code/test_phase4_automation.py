"""Tests for Phase 4 code-evolution CI/path-filter readiness."""

from pathlib import Path
import textwrap
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "phase2-tool-description-gate.yml"


def _workflow_run_blocks(workflow: dict[str, Any]) -> str:
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict)
    gate_job = jobs.get("phase2-tool-description-gate")
    assert isinstance(gate_job, dict)
    steps = gate_job.get("steps")
    assert isinstance(steps, list)
    return "\n".join(str(step.get("run", "")) for step in steps if isinstance(step, dict))


def _embedded_phase4_target_yaml(run_blocks: str) -> dict[str, Any]:
    marker = "cat > output/phase4-code-evolution/ci-gate/phase4_target.yaml <<'PYAML'"
    start = run_blocks.index(marker) + len(marker)
    rest = run_blocks[start:]
    lines = rest.splitlines()[1:]
    yaml_lines: list[str] = []
    for line in lines:
        if line.strip() == "PYAML":
            break
        yaml_lines.append(line)
    assert yaml_lines, "embedded Phase 4 target YAML must not be empty"
    embedded = textwrap.dedent("\n".join(yaml_lines)).strip() + "\n"
    target = yaml.safe_load(embedded)
    assert isinstance(target, dict)
    return target


def test_phase4_code_evolution_changes_are_in_workflow_paths_and_targeted_tests():
    assert WORKFLOW_PATH.exists(), "Phase 4 code-evolution changes require CI path coverage"
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    assert isinstance(workflow, dict)

    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    for trigger_name in ("pull_request", "push"):
        trigger = triggers.get(trigger_name)
        assert isinstance(trigger, dict)
        paths = trigger.get("paths")
        assert isinstance(paths, list)
        assert "evolution/code/**" in paths
        assert "tests/code/**" in paths
        assert "reports/phase4_*" in paths
        assert ".planning/spikes/**" in paths

    run_blocks = _workflow_run_blocks(workflow)
    assert "tests/code/test_phase4_target_contract.py" in run_blocks
    assert "tests/code/test_phase4_report_contract.py" in run_blocks
    assert "tests/code/test_phase4_code_scaffold.py" in run_blocks
    assert "tests/code/test_phase4_freeze_comparator.py" in run_blocks
    assert "tests/code/test_phase4_automation.py" in run_blocks
    assert "python -m evolution.code.report_contract" in run_blocks
    assert "python -m evolution.code.freeze_comparator" in run_blocks
    assert "python -m evolution.code.phase4_code_scaffold" in run_blocks
    assert "--candidate-file" in run_blocks
    assert "scaffold/scaffold_report.json" in run_blocks
    assert "scaffold/freeze_report.json" in run_blocks
    assert "tests/code/test_phase4_strict_code_evolution_runner.py" in run_blocks
    assert "tests/monitor/test_phase5_strict_unattended_loop.py" in run_blocks
    assert "tests/local_completion/test_strict_frontier_audit.py" in run_blocks


def test_phase4_ci_embedded_target_yaml_matches_current_contract():
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text())
    assert isinstance(workflow, dict)
    run_blocks = _workflow_run_blocks(workflow)
    target = _embedded_phase4_target_yaml(run_blocks)

    assert target["hermes_base"]["require_clean_worktree"] is True
    assert target["approvals"]["darwinian_execution_approved"] is False
    assert target["approvals"]["hermes_source_mutation_approved"] is False
    assert target["benchmarks"]["full_benchmark_required_before_acceptance"] is True

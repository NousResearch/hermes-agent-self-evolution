"""Tests for HSE Phase 3 strict execution preflight."""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

from evolution.local_completion.phase3_strict_execution_preflight import (
    BLOCKED_PHASE3_STRICT_EXECUTION_PREFLIGHT,
    PHASE3_STRICT_EXECUTION_PREFLIGHT_READY_SEPARATE_GO_REQUIRED,
    write_phase3_strict_execution_preflight,
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def _init_active_hermes(repo: Path) -> dict[str, str]:
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], text=True, capture_output=True, check=True)
    _git(repo, "config", "user.email", "phase3@example.invalid")
    _git(repo, "config", "user.name", "Phase3 Test")
    (repo / "agent").mkdir()
    (repo / "tests" / "agent").mkdir(parents=True)
    (repo / "agent" / "prompt_builder.py").write_text("DEFAULT_AGENT_IDENTITY = 'phase3 baseline'\n")
    (repo / "tests" / "agent" / "test_prompt_builder.py").write_text("def test_prompt_builder_placeholder():\n    assert True\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "active phase3 prompt baseline")
    head = _git(repo, "rev-parse", "HEAD")
    return {
        "head": head,
        "head_short": head[:9],
        "prompt_builder_sha": _sha(repo / "agent" / "prompt_builder.py"),
        "prompt_builder_test_sha": _sha(repo / "tests" / "agent" / "test_prompt_builder.py"),
    }


def _touch_hse_support_files(root: Path) -> None:
    support_files = [
        "evolution/prompts/phase3_candidate_scaffold.py",
        "evolution/prompts/phase3_preflight_gate.py",
        "evolution/prompts/phase3_gepa_optimizer.py",
        "evolution/benchmarks/run_tblite.py",
        "evolution/benchmarks/run_yc_bench.py",
        "datasets/golden/benchmarks/phase3-system-prompt/baseline_system_prompt.json",
        "datasets/golden/benchmarks/phase3-system-prompt/candidate_system_prompt.json",
        "datasets/golden/benchmarks/phase3-system-prompt/tblite_cases.jsonl",
        "datasets/golden/benchmarks/phase3-system-prompt/yc_bench_fast_test.jsonl",
    ]
    for rel in support_files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text("{}\n")
        elif path.suffix == ".jsonl":
            path.write_text('{"id":"case-1"}\n')
        else:
            path.write_text("# support fixture\n")


def _fixture_inputs(tmp_path: Path, active_repo: Path, active: dict[str, str]) -> dict[str, Path]:
    hse_root = tmp_path / "hse-root"
    _touch_hse_support_files(hse_root)
    plan = hse_root / "PLAN.md"
    plan.write_text("| **Phase 3** | System prompt | | | Behavioral tests pass, benchmarks hold or improve |\n")
    frontier = _write_json(
        tmp_path / "reports" / "strict_frontier_audit.json",
        {
            "schema_version": "hse-strict-frontier-audit-v1",
            "status": "PHASE_2_STRICT_COMPLETE",
            "active_hermes": {"repo_root": str(active_repo), "head": active["head"], "branch": "main", "clean": True},
            "current_active_frontier": {"status": "PHASE_2_STRICT_COMPLETE", "highest_strict_complete_phase": 2, "blockers": []},
            "phases": {
                "phase1": {"strict_complete": True},
                "phase2": {"strict_complete": True},
                "phase3": {"strict_complete": False, "blockers": ["phase3_real_benchmarks_not_executed", "phase3_active_apply_not_approved_current_readiness"]},
            },
            "github_query_performed": False,
            "github_write_performed": False,
            "provider_or_model_spend_performed": False,
            "network_calls_performed": False,
            "active_apply_performed": False,
            "full_remote_benchmark_executed": False,
            "overall_hse_project_completion_claimed": False,
        },
    )
    phase3_plan = _write_json(
        tmp_path / "reports" / "phase3_plan.json",
        {"phase": "3", "status": "planned_not_executed", "plan_only": True, "execution_started": False, "apply_ready": False},
    )
    phase3_readiness = _write_json(
        tmp_path / "reports" / "phase3_readiness.json",
        {
            "phase": "3",
            "status": "recorded_not_executed",
            "candidate_only": True,
            "execution_started": False,
            "run_gepa_now": False,
            "run_dspy_now": False,
            "real_benchmarks_executed": False,
            "real_benchmark_execution_approved": False,
            "mutate_active_system_prompt_now": False,
            "active_system_prompt_apply_approved": False,
            "apply_ready": False,
            "phase3_execution_ready": False,
            "ready_state": {"real_benchmark_ready_now": False, "active_apply_ready_now": False},
        },
    )
    seed_draft = _write_json(
        tmp_path / "reports" / "phase3_execution_seed_draft.json",
        {
            "phase": "3",
            "status": "drafted_not_executed",
            "execution_started": False,
            "run_gepa_now": False,
            "run_dspy_now": False,
            "mutate_active_system_prompt_now": False,
            "active_system_prompt_apply_approved": False,
            "apply_ready": False,
            "requires_human_approval_before_execution": True,
        },
    )
    return {
        "active_hermes_repo": active_repo,
        "hse_repo_root": hse_root,
        "strict_frontier_audit_path": frontier,
        "phase3_plan_path": phase3_plan,
        "phase3_readiness_path": phase3_readiness,
        "phase3_execution_seed_draft_path": seed_draft,
        "plan_path": plan,
    }


def test_phase3_strict_execution_preflight_ready_but_not_started(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    active = _init_active_hermes(active_repo)
    inputs = _fixture_inputs(tmp_path, active_repo, active)

    result = write_phase3_strict_execution_preflight(
        **inputs,
        output_dir=tmp_path / "report",
        generated_at="2026-07-04T10:55:00+09:00",
        future_run_id="phase3-strict-preflight-001",
    )

    report = json.loads(Path(result["report_path"]).read_text())
    preview = json.loads(Path(result["command_preview_path"]).read_text())
    assert Path(result["markdown_path"]).exists()
    assert report["schema_version"] == "hse-phase3-strict-execution-preflight-v1"
    assert report["status"] == PHASE3_STRICT_EXECUTION_PREFLIGHT_READY_SEPARATE_GO_REQUIRED
    assert report["preflight_passed"] is True
    assert report["phase3_execution_ready"] is False
    assert report["separate_phase3_execution_go_required"] is True
    assert report["execution_started"] is False
    assert report["run_gepa_now"] is False
    assert report["run_dspy_now"] is False
    assert report["mutate_active_system_prompt_now"] is False
    assert report["active_system_prompt_apply_approved"] is False
    assert report["github_query_performed"] is False
    assert report["github_write_performed"] is False
    assert report["provider_or_model_spend_performed"] is False
    assert report["network_calls_performed"] is False
    assert report["active_apply_performed"] is False
    assert report["cron_or_gateway_mutation_performed"] is False
    assert report["strict_frontier_gate"]["current_active_frontier_status"] == "PHASE_2_STRICT_COMPLETE"
    assert report["active_prompt_source_inventory"]["clean"] is True
    assert report["local_only_benchmark_readiness"]["dry_run_fixture_benchmarks_ready"] is True
    assert report["local_only_benchmark_readiness"]["real_benchmark_ready_now"] is False
    assert report["future_output_root_guard"]["future_output_root_exists_now"] is False
    assert report["future_output_root_guard"]["future_output_root_created_now"] is False
    assert not Path(report["future_output_root_guard"]["future_output_root"]).exists()
    assert report["blocked_by"] == []
    assert preview["schema_version"] == "hse-phase3-strict-execution-command-preview-v1"
    assert preview["dry_run"] is True
    assert all(command["started"] is False for command in preview["commands"])
    assert all(command["side_effect_class"] == "preview_only" for command in preview["commands"])


def test_phase3_strict_execution_preflight_blocks_existing_future_output_root(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    active = _init_active_hermes(active_repo)
    inputs = _fixture_inputs(tmp_path, active_repo, active)
    future_root = inputs["hse_repo_root"] / "output" / "phase3-system-prompt" / "phase3-strict-preflight-existing"
    future_root.mkdir(parents=True)

    result = write_phase3_strict_execution_preflight(
        **inputs,
        output_dir=tmp_path / "report",
        generated_at="2026-07-04T10:55:00+09:00",
        future_run_id="phase3-strict-preflight-existing",
    )

    report = json.loads(Path(result["report_path"]).read_text())
    assert report["status"] == BLOCKED_PHASE3_STRICT_EXECUTION_PREFLIGHT
    assert report["preflight_passed"] is False
    assert report["phase3_execution_ready"] is False
    assert "future_output_root_already_exists" in report["blocked_by"]
    assert report["future_output_root_guard"]["future_output_root_created_now"] is False
    assert report["execution_started"] is False


def test_phase3_strict_execution_preflight_blocks_active_prompt_source_dirty(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    active = _init_active_hermes(active_repo)
    inputs = _fixture_inputs(tmp_path, active_repo, active)
    (active_repo / "agent" / "prompt_builder.py").write_text("DEFAULT_AGENT_IDENTITY = 'dirty'\n")

    result = write_phase3_strict_execution_preflight(
        **inputs,
        output_dir=tmp_path / "report",
        generated_at="2026-07-04T10:55:00+09:00",
        future_run_id="phase3-strict-preflight-dirty",
    )

    report = json.loads(Path(result["report_path"]).read_text())
    assert report["status"] == BLOCKED_PHASE3_STRICT_EXECUTION_PREFLIGHT
    assert "active_hermes_worktree_dirty" in report["blocked_by"]
    assert report["active_prompt_source_inventory"]["clean"] is False
    assert report["execution_started"] is False

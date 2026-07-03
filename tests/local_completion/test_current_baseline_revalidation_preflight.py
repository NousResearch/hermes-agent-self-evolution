"""Tests for current-baseline Phase 1/2 revalidation preflight."""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path

from evolution.local_completion.current_baseline_revalidation_preflight import (
    BLOCKED_ACTIVE_BASELINE_DIRTY,
    BLOCKED_FUTURE_OUTPUT_ROOT_EXISTS,
    PREFLIGHT_READY_SEPARATE_GO_REQUIRED,
    write_current_baseline_revalidation_preflight,
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _init_active_hermes(repo: Path) -> dict[str, str]:
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], text=True, capture_output=True, check=True)
    _git(repo, "config", "user.email", "revalidation@example.invalid")
    _git(repo, "config", "user.name", "Revalidation Test")
    (repo / "tools").mkdir()
    (repo / "agent").mkdir()
    (repo / "model_tools.py").write_text("MODEL_TOOLS = ['current_active']\n")
    (repo / "tools" / "registry.py").write_text("REGISTRY = ['current_active']\n")
    (repo / "tools" / "__init__.py").write_text("# tools\n")
    (repo / "agent" / "prompt_builder.py").write_text("DEFAULT_AGENT_IDENTITY = 'current'\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "current active baseline")
    head = _git(repo, "rev-parse", "HEAD")
    return {
        "head": head,
        "head_short": head[:9],
        "model_tools_sha": _sha(repo / "model_tools.py"),
        "registry_sha": _sha(repo / "tools" / "registry.py"),
        "prompt_builder_sha": _sha(repo / "agent" / "prompt_builder.py"),
    }


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _strict_frontier_audit(path: Path, active: dict[str, str]) -> Path:
    closure_subject = "9b50c56556f902b62ecc4a7e2e511ca0f316da2d"
    payload = {
        "schema_version": "hse-strict-frontier-audit-v1",
        "status": "CURRENT_BASELINE_REVALIDATION_REQUIRED",
        "recorded_subject_frontier": {
            "status": "PHASE_2_STRICT_COMPLETE",
            "highest_strict_complete_phase": 2,
            "blockers": [],
        },
        "current_active_frontier": {
            "status": "CURRENT_BASELINE_REVALIDATION_REQUIRED",
            "highest_strict_complete_phase": 0,
            "blockers": [
                "active_tool_description_hash_mismatch",
                "current_baseline_revalidation_required_before_phase1_phase2_strict_claim",
                "current_hermes_head_not_closure_subject",
            ],
        },
        "current_baseline_match": {
            "active_head": active["head"],
            "active_head_short": active["head_short"],
            "closure_subject_commit": closure_subject,
            "matches_closure_subject": False,
            "active_tool_description_hashes_match": False,
        },
        "github_query_performed": False,
        "github_write_performed": False,
        "provider_or_model_spend_performed": False,
        "network_calls_performed": False,
        "active_apply_performed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _run_preflight(tmp_path: Path, active_repo: Path, audit_path: Path, run_id: str = "current-p12-reval-001") -> dict:
    result = write_current_baseline_revalidation_preflight(
        active_hermes_repo=active_repo,
        strict_frontier_audit_path=audit_path,
        hse_repo_root=tmp_path / "hse-root",
        output_dir=tmp_path / "report",
        generated_at="2026-07-04T02:55:00+09:00",
        future_run_id=run_id,
    )
    report = json.loads(Path(result["preflight_report_path"]).read_text())
    preview = json.loads(Path(result["command_preview_path"]).read_text())
    assert Path(result["preflight_markdown_path"]).exists()
    return {"report": report, "preview": preview}


def test_current_baseline_preflight_snapshots_active_inventory_and_recommends_separate_local_smoke_go(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    active = _init_active_hermes(active_repo)
    audit_path = _strict_frontier_audit(tmp_path / "strict_frontier_audit.json", active)

    result = _run_preflight(tmp_path, active_repo, audit_path)
    report = result["report"]
    preview = result["preview"]

    assert report["schema_version"] == "hse-current-baseline-revalidation-preflight-v1"
    assert report["status"] == PREFLIGHT_READY_SEPARATE_GO_REQUIRED
    assert report["current_baseline_inventory"]["head"] == active["head"]
    inventory = {item["relative_path"]: item for item in report["current_baseline_inventory"]["files"]}
    assert inventory["model_tools.py"]["sha256"] == active["model_tools_sha"]
    assert inventory["tools/registry.py"]["sha256"] == active["registry_sha"]
    assert inventory["tools/tool_description_overrides.py"]["exists"] is False
    assert report["rerun_decision"]["rerun_recommended"] is True
    assert report["rerun_decision"]["rerun_approved_now"] is False
    assert report["rerun_decision"]["separate_local_smoke_go_required"] is True
    assert report["execution_go"] is False
    assert report["execution_started"] is False
    assert report["real_benchmarks_executed"] is False
    assert report["strict_plan_gate_closed"] is False
    assert report["future_output_root_guard"]["future_output_root_exists_now"] is False
    assert report["future_output_root_guard"]["future_output_root_created_now"] is False
    assert report["current_baseline_materialization"]["materialization_started"] is False
    assert report["baseline_commit_for_rerun"] == "9b50c56556f902b62ecc4a7e2e511ca0f316da2d"
    assert report["current_commit_for_rerun"] == active["head"]
    assert report["side_effect_boundaries"]["github_query_performed"] is False
    assert report["side_effect_boundaries"]["provider_or_model_spend_performed"] is False
    assert report["side_effect_boundaries"]["active_apply_performed"] is False

    assert preview["schema_version"] == "hse-current-baseline-revalidation-command-preview-v1"
    assert preview["dry_run"] is True
    assert preview["benchmark_commands_started"] is False
    assert len(preview["commands"]) == 3
    assert all(command["started"] is False for command in preview["commands"])
    assert all("--dry-run" in command["argv"] for command in preview["commands"])
    assert all(active["head"] in command["argv"] for command in preview["commands"])
    assert not Path(report["future_output_root_guard"]["future_output_root"]).exists()


def test_current_baseline_preflight_blocks_dirty_active_hermes_without_execution(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    active = _init_active_hermes(active_repo)
    audit_path = _strict_frontier_audit(tmp_path / "strict_frontier_audit.json", active)
    (active_repo / "model_tools.py").write_text("MODEL_TOOLS = ['dirty']\n")

    result = _run_preflight(tmp_path, active_repo, audit_path, run_id="current-p12-reval-dirty")
    report = result["report"]

    assert report["status"] == BLOCKED_ACTIVE_BASELINE_DIRTY
    assert report["current_baseline_inventory"]["clean"] is False
    assert "active_hermes_worktree_dirty" in report["blocked_by"]
    assert report["rerun_decision"]["rerun_approved_now"] is False
    assert report["execution_go"] is False
    assert report["execution_started"] is False
    assert report["real_benchmarks_executed"] is False
    assert report["future_output_root_guard"]["future_output_root_created_now"] is False


def test_current_baseline_preflight_blocks_existing_future_output_root(tmp_path: Path):
    active_repo = tmp_path / "active-hermes"
    active = _init_active_hermes(active_repo)
    audit_path = _strict_frontier_audit(tmp_path / "strict_frontier_audit.json", active)
    future_root = tmp_path / "hse-root" / "output" / "hse-real-benchmark" / "current-p12-reval-existing"
    future_root.mkdir(parents=True)
    (future_root / "old.json").write_text("{}\n")

    result = _run_preflight(tmp_path, active_repo, audit_path, run_id="current-p12-reval-existing")
    report = result["report"]

    assert report["status"] == BLOCKED_FUTURE_OUTPUT_ROOT_EXISTS
    assert report["future_output_root_guard"]["future_output_root_exists_now"] is True
    assert "future_output_root_already_exists" in report["blocked_by"]
    assert report["execution_go"] is False
    assert report["execution_started"] is False
    assert report["real_benchmarks_executed"] is False

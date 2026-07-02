"""Tests for the LC4 Phase 3 active-source reconcile packet."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from evolution.local_completion.phase3_source_reconcile import (
    HISTORICAL_APPLY_COMMIT,
    LIVE_ON_CURRENT_HEAD,
    STALE_NOT_ANCESTOR_OF_CURRENT_HEAD,
    write_phase3_source_reconcile_packet,
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], text=True, capture_output=True, check=True)
    _git(repo, "config", "user.email", "lc4@example.invalid")
    _git(repo, "config", "user.name", "LC4 Test")
    (repo / "agent").mkdir()
    (repo / "tests" / "agent").mkdir(parents=True)
    (repo / "agent" / "prompt_builder.py").write_text("DEFAULT_AGENT_IDENTITY = 'baseline'\n")
    (repo / "tests" / "agent" / "test_prompt_builder.py").write_text("def test_baseline():\n    assert True\n")
    (repo / "README.md").write_text("baseline\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial active Hermes fixture")


def _make_stale_historical_commit(repo: Path) -> str:
    _git(repo, "checkout", "-b", "phase3-historical")
    (repo / "agent" / "prompt_builder.py").write_text(
        "DEFAULT_AGENT_IDENTITY = 'baseline'\nPHASE3_GUIDANCE = 'historical'\n"
    )
    _git(repo, "add", "agent/prompt_builder.py")
    _git(repo, "commit", "-m", "phase3 historical prompt guidance")
    historical = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "main")
    (repo / "README.md").write_text("current main successor\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "current active successor")
    return historical


def test_phase3_source_reconcile_packet_preserves_default_lc4_no_apply_invariants(tmp_path):
    active_repo = tmp_path / "active-hermes"
    _init_repo(active_repo)

    result = write_phase3_source_reconcile_packet(
        active_hermes_repo=active_repo,
        output_dir=tmp_path / "lc4-default",
        generated_at="2026-06-28T04:00:00Z",
    )
    packet = json.loads(Path(result["packet_path"]).read_text())

    assert packet["schema_version"] == "hse-local-completion-v1"
    assert packet["gate_id"] == "LC4"
    assert packet["historical_apply_commit"] == HISTORICAL_APPLY_COMMIT
    assert packet["phase3_active_source_status"] in {STALE_NOT_ANCESTOR_OF_CURRENT_HEAD, LIVE_ON_CURRENT_HEAD}
    assert packet["candidate_only"] is True
    assert packet["apply_ready"] is False
    assert packet["github"] == {
        "queried": False,
        "pr_created": False,
        "push_performed": False,
        "merge_performed": False,
        "publication_deferred": True,
    }
    assert packet["safety_invariants"]["active_prompt_modified"] is False
    assert packet["active_source_reconcile_boundary"]["active_source_patch_applied"] is False
    assert packet["active_source_reconcile_boundary"]["github_queried_or_written"] is False
    assert packet["active_source_reconcile_boundary"]["restart_reload_performed"] is False
    assert Path(result["markdown_path"]).exists()


def test_phase3_source_reconcile_packet_classifies_stale_commit_and_dirty_files(tmp_path):
    active_repo = tmp_path / "active-hermes"
    _init_repo(active_repo)
    historical = _make_stale_historical_commit(active_repo)

    (active_repo / "agent" / "prompt_builder.py").write_text("DEFAULT_AGENT_IDENTITY = 'dirty active head'\n")
    (active_repo / "gateway").mkdir()
    (active_repo / "gateway" / "run.py").write_text("# unrelated dirty gateway file\n")
    (active_repo / "tests" / "gateway").mkdir()
    (active_repo / "tests" / "gateway" / "test_dashboard_slash_command.py").write_text("def test_untracked():\n    assert True\n")

    result = write_phase3_source_reconcile_packet(
        active_hermes_repo=active_repo,
        output_dir=tmp_path / "lc4-stale",
        generated_at="2026-06-28T04:01:00Z",
        historical_apply_commit=historical,
    )
    packet = json.loads(Path(result["packet_path"]).read_text())
    ledger = {item["path"]: item for item in packet["dirty_file_conflict_ledger"]}

    assert packet["phase3_active_source_status"] == STALE_NOT_ANCESTOR_OF_CURRENT_HEAD
    assert packet["historical_commit"]["available"] is True
    assert packet["historical_commit"]["resolved"] == historical
    assert packet["historical_commit"]["ancestor_check_rc"] == 1
    assert packet["historical_commit"]["is_ancestor_of_current_head"] is False
    assert ledger["agent/prompt_builder.py"]["scope"] == "phase3_source_allowlist"
    assert ledger["agent/prompt_builder.py"]["action"] == "inspect_semantically_before_future_patch"
    assert ledger["gateway/run.py"]["scope"] == "preserve_unrelated_dirty"
    assert ledger["tests/gateway/test_dashboard_slash_command.py"]["action"] == "do_not_touch_in_lc4"
    assert packet["active_hermes"]["target_dirty_file_count"] == 1
    assert packet["candidate_source_patch_plan"]["patch_generated"] is False
    assert packet["candidate_source_patch_plan"]["source_mutation_performed"] is False
    assert packet["candidate_source_patch_plan"]["allowlist"] == [
        "agent/prompt_builder.py",
        "tests/agent/test_prompt_builder.py",
    ]
    assert "do not cherry-pick historical commit 65a7925aa directly against a dirty/current HEAD" in packet[
        "prompt_cache_safety_checklist"
    ]
    assert "python -m pytest tests/agent/test_prompt_builder.py -q -o 'addopts='" in packet[
        "tests_required_before_any_active_source_patch"
    ]
    assert "apply_ready=false" in Path(result["markdown_path"]).read_text()


def test_phase3_source_reconcile_packet_reports_live_when_commit_is_current_head(tmp_path):
    active_repo = tmp_path / "active-hermes"
    _init_repo(active_repo)
    current_head = _git(active_repo, "rev-parse", "HEAD")

    result = write_phase3_source_reconcile_packet(
        active_hermes_repo=active_repo,
        output_dir=tmp_path / "lc4-live",
        generated_at="2026-06-28T04:02:00Z",
        historical_apply_commit=current_head,
    )
    packet = json.loads(Path(result["packet_path"]).read_text())

    assert packet["phase3_active_source_status"] == LIVE_ON_CURRENT_HEAD
    assert packet["historical_commit"]["ancestor_check_rc"] == 0
    assert packet["historical_commit"]["is_ancestor_of_current_head"] is True
    assert packet["dirty_file_conflict_ledger"] == []
    assert packet["active_hermes"]["dirty_file_count"] == 0
    assert packet["candidate_only"] is True
    assert packet["apply_ready"] is False

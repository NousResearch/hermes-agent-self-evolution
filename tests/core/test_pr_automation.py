"""Tests for ``evolution.core.pr_automation``.

Covers the orchestration paths (skip / failed / created) with mocked
subprocess plus a single integration test against an ephemeral local
git pair to confirm the real `git`/file-copy choreography works.
"""

import re
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from evolution.core.pr_automation import (
    PRResult,
    _atomic_copy,
    _branch_name,
    _format_pr_body,
    create_pr,
    find_git_root,
)


def _ok(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=stderr)


def _fail(stderr: str, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout=stdout, stderr=stderr)


def _happy_path_side_effect(pr_url: str = "https://github.com/o/r/pull/42"):
    """Return a side_effect that walks the happy-path subprocess sequence.

    Order mirrors create_pr's orchestration: status, fetch, checkout, add,
    commit, rev-parse, push, gh pr create.
    """
    responses = [
        _ok(stdout=""),               # git status --porcelain (clean)
        _ok(),                         # git fetch
        _ok(),                         # git checkout -b
        _ok(),                         # git add
        _ok(),                         # git commit
        _ok(stdout="abc1234\n"),       # git rev-parse HEAD
        _ok(),                         # git push
        _ok(stdout=f"{pr_url}\n"),     # gh pr create
    ]
    return responses


class TestPRAutomation:
    def _kwargs(self, tmp_path: Path, **overrides):
        evolved = tmp_path / "evolved.md"
        evolved.write_text("evolved content")
        source_root = tmp_path / "source"
        (source_root / "skills" / "test").mkdir(parents=True)
        (source_root / "skills" / "test" / "SKILL.md").write_text("baseline")
        defaults = dict(
            source_repo_root=source_root,
            source_artifact_relpath="skills/test/SKILL.md",
            evolved_artifact_path=evolved,
            artifact_name="test-skill",
            gate_decision={"decision_signal": "synthetic", "decision": "deploy"},
            metrics={"baseline_mean": 0.40, "evolved_mean": 0.55, "delta": 0.15},
            base_branch="main",
            branch_prefix="evolve/",
            draft=False,
            allow_dirty=False,
            console=Console(),
        )
        defaults.update(overrides)
        return defaults

    def test_happy_path_creates_pr(self, tmp_path: Path):
        responses = _happy_path_side_effect("https://github.com/o/r/pull/123")
        with patch("evolution.core.pr_automation.subprocess.run", side_effect=responses):
            result = create_pr(**self._kwargs(tmp_path))
        assert result.status == "created"
        assert result.url == "https://github.com/o/r/pull/123"
        assert result.branch.startswith("evolve/test-skill-")
        assert result.commit_sha == "abc1234"

    def test_skipped_when_source_repo_root_is_none(self, tmp_path: Path):
        with patch("evolution.core.pr_automation.subprocess.run") as run:
            result = create_pr(**self._kwargs(tmp_path, source_repo_root=None))
        assert result.status == "skipped"
        assert "git-backed" in result.reason
        assert result.branch is None
        assert result.commit_sha is None
        assert result.url is None
        run.assert_not_called()

    def test_skipped_on_dirty_tree_when_allow_dirty_false(self, tmp_path: Path):
        responses = [_ok(stdout=" M file.txt\n?? other.txt\n")]
        with patch("evolution.core.pr_automation.subprocess.run", side_effect=responses):
            result = create_pr(**self._kwargs(tmp_path))
        assert result.status == "skipped"
        assert "dirty" in result.reason.lower()

    def test_proceeds_on_dirty_tree_when_allow_dirty_true(self, tmp_path: Path):
        responses = [
            _ok(stdout=" M file.txt\n"),    # dirty status — but ignored
            _ok(),                            # fetch
            _ok(),                            # checkout
            _ok(),                            # add
            _ok(),                            # commit
            _ok(stdout="deadbee\n"),          # rev-parse
            _ok(),                            # push
            _ok(stdout="https://github.com/o/r/pull/9\n"),  # gh pr create
        ]
        with patch("evolution.core.pr_automation.subprocess.run", side_effect=responses):
            result = create_pr(**self._kwargs(tmp_path, allow_dirty=True))
        assert result.status == "created"
        assert result.url == "https://github.com/o/r/pull/9"

    def test_failed_when_gh_not_on_path(self, tmp_path: Path):
        responses = [
            _ok(stdout=""),
            _ok(), _ok(), _ok(), _ok(),
            _ok(stdout="abc1234\n"),
            _ok(),
            FileNotFoundError("gh: command not found"),
        ]
        with patch("evolution.core.pr_automation.subprocess.run", side_effect=responses):
            result = create_pr(**self._kwargs(tmp_path))
        assert result.status == "failed"
        assert "gh" in result.reason.lower()
        assert result.branch is not None
        assert result.commit_sha == "abc1234"

    def test_failed_when_push_fails_captures_stderr(self, tmp_path: Path):
        responses = [
            _ok(stdout=""),
            _ok(), _ok(), _ok(), _ok(),
            _ok(stdout="abc1234\n"),
            _fail(stderr="remote: Permission denied to user@example.com\n"),
        ]
        with patch("evolution.core.pr_automation.subprocess.run", side_effect=responses):
            result = create_pr(**self._kwargs(tmp_path))
        assert result.status == "failed"
        assert "Permission denied" in result.reason
        assert result.branch is not None
        assert result.commit_sha == "abc1234"

    def test_branch_name_has_4char_suffix_and_sanitized(self):
        ts = datetime(2026, 5, 25, 14, 30, 45)
        name = _branch_name("evolve/", "some/skill:with spaces", ts)
        assert re.match(r"^evolve/some-skill-with-spaces-\d{8}-\d{6}-[0-9a-f]{4}$", name), name

    def test_format_pr_body_synthetic_deploy(self):
        body = _format_pr_body(
            gate_decision={
                "decision_signal": "synthetic",
                "decision": "deploy",
                "reason": "growth_quality_gate_passed",
                "baseline_chars": 1000,
                "evolved_chars": 1100,
                "cost_summary": {"total_usd": 1.23},
            },
            metrics={"baseline_mean": 0.40, "evolved_mean": 0.55, "delta": 0.15},
        )
        assert "deploy" in body.lower()
        assert "0.40" in body and "0.55" in body
        assert "+0.15" in body or "0.15" in body
        assert "Generated by agent-self-evolution" in body

    def test_format_pr_body_cl_primary_deploy(self):
        body = _format_pr_body(
            gate_decision={
                "decision_signal": "closed_loop",
                "decision": "deploy",
                "reason": "cl_primary_gate_passed",
                "cl_tasks_gained": 3,
                "cl_required_gain": 1,
                "cost_summary": {"total_usd": 4.50},
            },
            metrics={"baseline_mean": 0.40, "evolved_mean": 0.42, "delta": 0.02},
        )
        # CL gain should appear prominently (in the first ~400 chars, i.e. above the fold)
        assert "+3" in body[:600] or "3 task" in body[:600] or "cl_tasks_gained" in body[:600].lower()
        assert "closed" in body.lower() or "cl" in body.lower()
        assert "Generated by agent-self-evolution" in body

    def test_integration_against_ephemeral_repo(self, tmp_path: Path):
        bare = tmp_path / "remote.git"
        clone = tmp_path / "clone"
        subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
        subprocess.run(["git", "clone", str(bare), str(clone)], check=True, capture_output=True)
        # Seed the repo so origin/main exists.
        subprocess.run(["git", "-C", str(clone), "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", str(clone), "config", "user.name", "t"], check=True)
        subprocess.run(["git", "-C", str(clone), "checkout", "-b", "main"], check=True, capture_output=True)
        skill_path = clone / "skills" / "test" / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text("baseline content\n")
        subprocess.run(["git", "-C", str(clone), "add", "."], check=True)
        subprocess.run(["git", "-C", str(clone), "commit", "-m", "seed"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(clone), "push", "-u", "origin", "main"], check=True, capture_output=True)

        evolved = tmp_path / "evolved.md"
        evolved.write_text("evolved improved content\n")

        real_run = subprocess.run

        def fake_run(cmd, *args, **kwargs):
            # Only intercept the gh CLI call; everything else hits real git.
            if isinstance(cmd, (list, tuple)) and len(cmd) > 0 and cmd[0] == "gh":
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0,
                    stdout="https://github.com/fake/repo/pull/777\n", stderr="",
                )
            return real_run(cmd, *args, **kwargs)

        with patch("evolution.core.pr_automation.subprocess.run", side_effect=fake_run):
            result = create_pr(
                source_repo_root=clone,
                source_artifact_relpath="skills/test/SKILL.md",
                evolved_artifact_path=evolved,
                artifact_name="test-skill",
                gate_decision={"decision_signal": "synthetic", "decision": "deploy"},
                metrics={"baseline_mean": 0.40, "evolved_mean": 0.55, "delta": 0.15},
                base_branch="main",
                branch_prefix="evolve/",
                draft=False,
                allow_dirty=False,
                console=Console(),
            )

        assert result.status == "created"
        assert result.url == "https://github.com/fake/repo/pull/777"
        assert result.branch.startswith("evolve/test-skill-")
        assert result.commit_sha and len(result.commit_sha) >= 7

        # Branch exists on the bare remote
        ls = subprocess.run(
            ["git", "-C", str(bare), "branch", "--list", result.branch],
            check=True, capture_output=True, text=True,
        )
        assert result.branch in ls.stdout

        # File content on the new branch matches evolved artifact
        show = subprocess.run(
            ["git", "-C", str(bare), "show", f"{result.branch}:skills/test/SKILL.md"],
            check=True, capture_output=True, text=True,
        )
        assert show.stdout == "evolved improved content\n"


class TestFindGitRoot:
    def test_returns_root_inside_repo(self, tmp_path: Path):
        subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
        nested = tmp_path / "a" / "b" / "c.md"
        nested.parent.mkdir(parents=True)
        nested.write_text("x")
        root = find_git_root(nested)
        assert root is not None and root.resolve() == tmp_path.resolve()

    def test_returns_none_outside_repo(self, tmp_path: Path):
        # tmp_path is fresh and not a git repo
        assert find_git_root(tmp_path / "missing.md") is None


class TestAtomicCopy:
    def test_replaces_existing_file(self, tmp_path: Path):
        src = tmp_path / "src.md"
        dst = tmp_path / "dst.md"
        src.write_text("new")
        dst.write_text("old")
        _atomic_copy(src, dst)
        assert dst.read_text() == "new"

    def test_creates_when_missing(self, tmp_path: Path):
        src = tmp_path / "src.md"
        dst = tmp_path / "dst.md"
        src.write_text("hello")
        _atomic_copy(src, dst)
        assert dst.read_text() == "hello"

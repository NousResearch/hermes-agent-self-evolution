"""Tests for the WorktreeSandbox."""

from pathlib import Path

import pytest

from evolution.code.sandbox import SandboxError, WorktreeSandbox


class TestWorktreeSandbox:
    def test_rejects_missing_worktree(self, tmp_path):
        with pytest.raises(SandboxError, match="does not exist"):
            WorktreeSandbox(tmp_path / "nope")

    def test_run_echo(self, tmp_path):
        sandbox = WorktreeSandbox(tmp_path)
        result = sandbox.run(["cmd", "/c", "echo", "hello"])
        assert result.passed
        assert "hello" in result.stdout

    def test_run_failing_command(self, tmp_path):
        sandbox = WorktreeSandbox(tmp_path)
        result = sandbox.run(["cmd", "/c", "exit 1"])
        assert not result.passed
        assert result.returncode == 1

    def test_cwd_escape_rejected(self, tmp_path):
        sandbox = WorktreeSandbox(tmp_path)
        with pytest.raises(SandboxError, match="escapes sandbox"):
            sandbox.run(["echo", "x"], cwd=Path("/"))

    def test_empty_command_rejected(self, tmp_path):
        sandbox = WorktreeSandbox(tmp_path)
        with pytest.raises(SandboxError, match="must not be empty"):
            sandbox.run([])

    def test_timeout(self, tmp_path):
        sandbox = WorktreeSandbox(tmp_path, default_timeout_seconds=1)
        result = sandbox.run(
            ["cmd", "/c", "ping", "-n", "10", "127.0.0.1"],
            timeout_seconds=1,
        )
        assert result.timed_out

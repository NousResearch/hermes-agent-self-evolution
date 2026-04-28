"""Tests for git snapshot helpers."""

import shutil
import subprocess

from evolution.repos.git import get_git_snapshot


def test_get_git_snapshot_reports_sha_branch_and_dirty_state(tmp_path):
    if not shutil.which("git"):
        return

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)

    clean = get_git_snapshot(tmp_path)
    assert len(clean.git_sha) >= 7
    assert clean.branch
    assert clean.dirty is False

    (tmp_path / "README.md").write_text("changed\n")
    dirty = get_git_snapshot(tmp_path)
    assert dirty.dirty is True
    assert dirty.diff_sha256

"""Git helpers for repository snapshots."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitSnapshot:
    git_sha: str
    branch: str
    dirty: bool
    diff_sha256: str | None = None


def get_git_snapshot(repo_path: str | Path) -> GitSnapshot:
    """Return current git SHA, branch, dirty flag, and dirty diff hash."""
    path = Path(repo_path)
    git_sha = _git(path, "rev-parse", "HEAD")
    branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(path, "status", "--porcelain")
    dirty = bool(status.strip())
    diff_sha256 = None
    if dirty:
        diff = _git(path, "diff", "--binary") + _git(path, "diff", "--cached", "--binary") + status
        diff_sha256 = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    return GitSnapshot(git_sha=git_sha, branch=branch, dirty=dirty, diff_sha256=diff_sha256)


def _git(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()

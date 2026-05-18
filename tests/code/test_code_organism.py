"""Tests for the code-organism git worktree wrapper."""

import subprocess
from pathlib import Path

import pytest

from evolution.code.code_organism import CodeOrganism, OrganismError


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with a tool file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    tool_dir = repo / "tools"
    tool_dir.mkdir()
    (tool_dir / "file_tools.py").write_text(
        'def read_file(path: str) -> str:\n    return open(path).read()\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "."],
        cwd=str(repo), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "init"],
        cwd=str(repo), capture_output=True, check=True,
    )
    return repo


class TestCodeOrganism:
    def test_detects_non_git_repo(self, tmp_path):
        with pytest.raises(OrganismError):
            CodeOrganism(repo_root=tmp_path, relative_tool_path=Path("x.py"))

    def test_detects_missing_tool(self, git_repo):
        with pytest.raises(OrganismError, match="does not exist"):
            CodeOrganism(
                repo_root=git_repo,
                relative_tool_path=Path("tools/nonexistent.py"),
            )

    def test_baseline_sha256_stable(self, git_repo):
        org = CodeOrganism(
            repo_root=git_repo,
            relative_tool_path=Path("tools/file_tools.py"),
        )
        assert org.baseline_sha256
        assert org.baseline_sha256 == org.baseline_sha256

    def test_create_candidate_worktree(self, git_repo):
        org = CodeOrganism(
            repo_root=git_repo,
            relative_tool_path=Path("tools/file_tools.py"),
            session_root=git_repo.parent / "session",
        )
        candidate = org.create_candidate(
            iteration=0,
            mutated_source='def read_file(path: str) -> str:\n    return "fixed"\n',
        )
        assert candidate.tool_path.is_file()
        assert "fixed" in candidate.tool_path.read_text(encoding="utf-8")
        assert candidate.candidate_sha256 != org.baseline_sha256
        org.cleanup()

    def test_create_candidate_escapes_rejected(self, git_repo):
        with pytest.raises(OrganismError, match="outside repo root"):
            CodeOrganism(
                repo_root=git_repo,
                relative_tool_path=Path("../outside.py"),
            )

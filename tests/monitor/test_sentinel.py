"""Tests for the monitor sentinel (classification + ranking; git only, no spend)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from evolution.monitor.sentinel import classify, scan

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git required for the sentinel"
)


def _run(repo: Path, *args: str):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _commit(repo: Path, msg: str):
    _run(repo, "add", "-A")
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t.t", "-c", "user.name=t",
                    "commit", "-q", "-m", msg], check=True, capture_output=True)


@pytest.fixture
def stream_repo(tmp_path: Path) -> Path:
    """A repo whose recent stream has a bug-fix commit and a dep-regression commit."""
    repo = tmp_path / "r"
    (repo / "tools").mkdir(parents=True)
    (repo / "tests" / "tools").mkdir(parents=True)
    _run(repo, "init", "-q")
    (repo / "pyproject.toml").write_text("[project]\nname='r'\ndependencies=['dep==1.0']\n")
    (repo / "tools" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "tests" / "tools" / "test_calc.py").write_text(
        "from tools.calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    _commit(repo, "baseline")
    # A plain bug-fix commit: touches tool + test, no dependency change.
    (repo / "tools" / "calc.py").write_text("def add(a, b):\n    return a + b  # fixed\n")
    (repo / "tests" / "tools" / "test_calc.py").write_text(
        "from tools.calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_add_neg():\n    assert add(-1, 1) == 0\n")
    _commit(repo, "fix: add()")
    # A dependency-regression commit: touches tool + test AND bumps the manifest.
    (repo / "pyproject.toml").write_text("[project]\nname='r'\ndependencies=['dep==2.0']\n")
    (repo / "tools" / "calc.py").write_text("def add(a, b):\n    return a + b  # adapt to dep 2.0\n")
    (repo / "tests" / "tools" / "test_calc.py").write_text(
        "from tools.calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_add_neg():\n    assert add(-1, 1) == 0\n\n\ndef test_dep():\n    assert add(0, 0) == 0\n")
    _commit(repo, "fix: adapt calc to dep 2.0 bump")
    return repo


class TestSentinel:
    def test_classifies_and_ranks_dep_regression_first(self, stream_repo):
        cands = scan(stream_repo, since_days=3650, max_per_tool=10)
        assert len(cands) == 2
        # dependency_regression outranks bug_fix
        assert cands[0].kind == "dependency_regression"
        assert cands[1].kind == "bug_fix"
        assert all(c.tool_path == "tools/calc.py" for c in cands)
        assert all(c.committed_at for c in cands)  # ISO date captured

    def test_score_ordering(self, stream_repo):
        cands = scan(stream_repo, since_days=3650)
        assert cands[0].score > cands[1].score

    def test_classify_single(self, stream_repo):
        from evolution.code.harvest import harvest_candidates
        cands = harvest_candidates(stream_repo, max_commits_per_tool=10)
        kinds = {classify(stream_repo, c).kind for c in cands}
        assert kinds == {"dependency_regression", "bug_fix"}

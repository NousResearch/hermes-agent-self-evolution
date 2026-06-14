"""Tests for the real-bug harvester."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from evolution.code.harvest import (
    Candidate,
    discover_targets,
    harvest_candidates,
    tool_for_test,
    validate_candidate,
)

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git required for the harvester"
)

_PYPROJECT = """\
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "synthtool"
version = "0.1.0"
requires-python = ">=3.9"

[tool.setuptools]
packages = ["tools"]
"""

_TEST_ONE = "from tools.calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
_TEST_TWO = _TEST_ONE + "\n\ndef test_add_again():\n    assert add(10, 20) == 30\n"


def _commit(repo: Path, msg: str) -> str:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t.t", "-c", "user.name=t",
                    "commit", "-q", "-m", msg], check=True, capture_output=True)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def bug_repo(tmp_path: Path):
    """A repo with a buggy parent commit and a fix commit touching tool + test."""
    repo = tmp_path / "synthtool"
    (repo / "tools").mkdir(parents=True)
    (repo / "tests" / "tools").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(_PYPROJECT)
    (repo / "tools" / "__init__.py").write_text("")
    (repo / "tests" / "__init__.py").write_text("")
    (repo / "tests" / "tools" / "__init__.py").write_text("")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, capture_output=True)
    # Parent: buggy tool + the one test.
    (repo / "tools" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "tests" / "tools" / "test_calc.py").write_text(_TEST_ONE)
    parent = _commit(repo, "buggy baseline")
    # Fix: correct the tool AND touch the test file (adds a case) — bug-fix shape.
    (repo / "tools" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "tests" / "tools" / "test_calc.py").write_text(_TEST_TWO)
    fix = _commit(repo, "fix add()")
    return repo, parent, fix


class TestDiscovery:
    def test_tool_for_test_maps_convention(self):
        assert tool_for_test("tests/tools/test_calc.py") == "tools/calc.py"
        assert tool_for_test("tests/tools/conftest.py") is None
        assert tool_for_test("tools/calc.py") is None

    def test_discover_targets_finds_pair(self, bug_repo):
        repo, _, _ = bug_repo
        assert ("tools/calc.py", "tests/tools/test_calc.py") in discover_targets(repo)

    def test_harvest_candidates_finds_fix_commit(self, bug_repo):
        repo, parent, fix = bug_repo
        cands = harvest_candidates(repo)
        assert any(c.fix_sha == fix and c.parent_sha == parent
                   and c.tool_path == "tools/calc.py" for c in cands)


class TestValidity:
    def test_validate_candidate_finds_bug_tests(self, bug_repo):
        repo, parent, fix = bug_repo
        bug = validate_candidate(repo, Candidate(
            "tools/calc.py", "tests/tools/test_calc.py", fix, parent))
        assert bug is not None
        # Both tests assert the fixed behavior; the buggy parent fails both.
        assert any("test_add" in t for t in bug.bug_tests)
        assert len(bug.bug_tests) >= 1

    def test_validate_returns_none_when_parent_not_buggy(self, bug_repo, tmp_path):
        # A "fix" commit whose parent already passes (no real bug) → None.
        repo, _, fix = bug_repo
        # Use fix as both parent and fix: the fix source passes its own tests, so
        # parent_failures == fix_failures → empty bug_tests → None.
        bug = validate_candidate(repo, Candidate(
            "tools/calc.py", "tests/tools/test_calc.py", fix, fix))
        assert bug is None

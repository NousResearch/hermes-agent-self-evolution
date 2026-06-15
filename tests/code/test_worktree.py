"""Integration test for the isolated worktree+venv harness.

Builds a tiny synthetic *installable* git repo and drives the real
:class:`WorktreeEnv`: git worktree → ``--system-site-packages`` venv →
``pip install -e --no-deps`` → the authoritative-import guard → run a test →
teardown. This validates the mechanism end-to-end. The cross-install *shadowing*
that matters for the real Hermes checkout is validated empirically at runtime by
``assert_authoritative`` (there is no competing install in this synthetic repo),
which is exactly why that guard exists.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from evolution.code.worktree import WorktreeEnv, WorktreeError, prune_orphan_worktrees

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git required for worktree harness"
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

_TOOL_OK = "def add(a, b):\n    return a + b\n"
_TEST = "from tools.calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


@pytest.fixture
def synth_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "synthtool"
    (repo / "tools").mkdir(parents=True)
    (repo / "tests" / "tools").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(_PYPROJECT)
    (repo / "tools" / "__init__.py").write_text("")
    (repo / "tools" / "calc.py").write_text(_TOOL_OK)
    (repo / "tests" / "__init__.py").write_text("")
    (repo / "tests" / "tools" / "__init__.py").write_text("")
    (repo / "tests" / "tools" / "test_calc.py").write_text(_TEST)
    for args in (
        ["init", "-q"],
        ["config", "user.email", "t@t.t"],
        ["config", "user.name", "t"],
        ["add", "-A"],
        ["commit", "-q", "-m", "baseline"],
    ):
        subprocess.run(["git", *args], cwd=str(repo), check=True,
                       capture_output=True, text=True)
    return repo


class TestWorktreeEnv:
    def test_full_lifecycle(self, synth_repo: Path):
        env = WorktreeEnv.create(synth_repo, base_ref="HEAD")
        try:
            # Guard: the synthetic 'tools' package resolves from the worktree.
            env.assert_authoritative("tools")
            assert env.python.exists()

            ok = env.run_test("tests/tools/test_calc.py")
            assert ok.passed, ok.output

            # A repair the loop would write: break the tool, observe red.
            env.write_tool("tools/calc.py", "def add(a, b):\n    return a - b\n")
            red = env.run_test("tests/tools/test_calc.py")
            assert not red.passed
            assert "tools/calc.py" in env.changed_files()
            assert "def add" in env.diff()
        finally:
            worktree_path = env.worktree
            env.destroy()
        assert not worktree_path.exists()

    def test_guard_trips_on_import_outside_worktree(self, synth_repo: Path):
        # 'os' is stdlib — resolves outside the worktree, so the guard must
        # refuse to trust the environment.
        with WorktreeEnv.create(synth_repo, base_ref="HEAD") as env:
            with pytest.raises(WorktreeError, match="isolation breach"):
                env.assert_authoritative("os")

    def test_create_rejects_bad_ref(self, synth_repo: Path):
        with pytest.raises(WorktreeError):
            WorktreeEnv.create(synth_repo, base_ref="no-such-ref-xyz")


class TestPruneOrphans:
    def test_removes_orphan_evolve_code_worktree(self, synth_repo: Path):
        # Simulate a hard-killed run's leak: an evolve_code_*/wt worktree whose
        # process died before destroy() could run.
        root = Path(tempfile.mkdtemp(prefix="evolve_code_"))
        wt = root / "wt"
        subprocess.run(["git", "-C", str(synth_repo), "worktree", "add", "--detach",
                        str(wt), "HEAD"], check=True, capture_output=True)
        listing = subprocess.run(["git", "-C", str(synth_repo), "worktree", "list"],
                                 capture_output=True, text=True).stdout
        assert str(wt) in listing  # leak present
        try:
            removed = prune_orphan_worktrees(synth_repo)
            assert removed >= 1
            after = subprocess.run(["git", "-C", str(synth_repo), "worktree", "list"],
                                   capture_output=True, text=True).stdout
            assert str(wt) not in after          # registration gone
            assert not root.exists()             # tmpdir removed
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_leaves_non_evolve_code_worktrees_alone(self, synth_repo: Path, tmp_path: Path):
        # A worktree NOT under the evolve_code_ prefix must be untouched.
        other = tmp_path / "user_wt"
        subprocess.run(["git", "-C", str(synth_repo), "worktree", "add", "--detach",
                        str(other), "HEAD"], check=True, capture_output=True)
        try:
            prune_orphan_worktrees(synth_repo)
            listing = subprocess.run(["git", "-C", str(synth_repo), "worktree", "list"],
                                     capture_output=True, text=True).stdout
            assert str(other) in listing  # untouched
        finally:
            subprocess.run(["git", "-C", str(synth_repo), "worktree", "remove", "--force",
                            str(other)], capture_output=True)

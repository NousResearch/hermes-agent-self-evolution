"""Shared fixtures/doubles for evolution.code tests.

``StagedRepo`` is a real on-disk ``tools/`` + ``tests/tools/`` layout that runs
real pytest (deterministic, no LLM, no venv — the tools are self-contained), and
duck-types the slice of :class:`WorktreeEnv` the repair engine and gate call.
The full venv+worktree harness is exercised separately in ``test_worktree.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from evolution.code.worktree import TestRun

# Mirror WorktreeEnv: no .pyc writes, so same-size rewrites between rounds don't
# run stale bytecode (and no __pycache__ dirties the repo).
_NO_BYTECODE_ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


class StagedRepo:
    """A self-contained tool+test layout under ``root``, runnable with pytest."""

    def __init__(self, root: Path):
        self.root = root
        self.worktree = root  # gate/_base_source treat this as the repo root
        (root / "tools").mkdir(parents=True, exist_ok=True)
        (root / "tests" / "tools").mkdir(parents=True, exist_ok=True)
        for rel in ("tools/__init__.py", "tests/__init__.py", "tests/tools/__init__.py"):
            (root / rel).write_text("")

    def write(self, relpath: str, src: str) -> None:
        (self.root / relpath).write_text(src)

    # -- WorktreeEnv duck-typing -----------------------------------------
    def read_tool(self, relpath: str) -> str:
        return (self.root / relpath).read_text()

    def write_tool(self, relpath: str, src: str) -> None:
        (self.root / relpath).write_text(src)

    def run_test(self, *test_paths: str, timeout: int = 120, extra_args=None,
                 full_output: bool = False) -> TestRun:
        start = time.monotonic()
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header",
             "-p", "no:cacheprovider", *(extra_args or []), *test_paths],
            cwd=str(self.root), capture_output=True, text=True, timeout=timeout,
            env=_NO_BYTECODE_ENV,
        )
        combined = res.stdout + "\n" + res.stderr
        out = combined if full_output else combined[-6000:]
        return TestRun(passed=res.returncode == 0, output=out,
                       duration_seconds=time.monotonic() - start,
                       exit_code=res.returncode)

    def failing_tests(self, *node_ids: str) -> set[str]:
        from evolution.code.gate import _parse_pytest_failures
        run = self.run_test(*node_ids, extra_args=["--tb=no"], full_output=True)
        return _parse_pytest_failures(run.output)

    # -- git support (for gate tests that need HEAD + status) -------------
    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=str(self.root),
                              capture_output=True, text=True, timeout=60)

    def git_init_commit(self) -> None:
        self._git("init", "-q")
        self._git("config", "user.email", "t@t.t")
        self._git("config", "user.name", "t")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "baseline")

    def changed_files(self) -> list[str]:
        res = self._git("status", "--porcelain", "--untracked-files=all")
        return [line[3:].strip() for line in res.stdout.splitlines() if len(line) > 3]

    def diff(self) -> str:
        return self._git("diff").stdout


# Buggy and fixed are equal-length (only the operator differs) so the toy
# fixture stays above the realistic 0.8 retain floor — real tool files are large
# and a one-operator fix never trips it; a 2-line toy would without this care.
BUGGY_CALC = "def add(a, b):\n    return a - b\n"
FIXED_CALC = "def add(a, b):\n    return a + b\n"
VISIBLE_TEST = "from tools.calc import add\n\n\ndef test_add_visible():\n    assert add(2, 3) == 5\n"
HOLDOUT_TEST = "from tools.calc import add\n\n\ndef test_add_holdout():\n    assert add(10, 20) == 30\n"

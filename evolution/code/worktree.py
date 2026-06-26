"""Isolated worktree + venv harness for code evolution.

A code repair must be measured against the *repaired* code and nothing else.
Two failure modes make that harder than it looks, and this module exists to
foreclose both:

1. **Never touch the user's checkout.** Repairs happen in a throwaway ``git
   worktree`` of the target repo, created from a base ref. The user's working
   copy is never written to.

2. **Never measure a Frankenstein import.** The target repo (Hermes) is
   installed *editable*: its import finder hardcodes a package→original-path
   map. Putting the worktree on ``PYTHONPATH`` is not enough — submodule
   resolution can still fall through to the original tree, so the loop would
   score a mix of repaired and original code. The robust fix is to make the
   worktree *authoritative*: a dedicated venv with ``pip install -e <worktree>``
   so the editable finder points at the worktree. ``--system-site-packages
   --no-deps`` keeps that install cheap by inheriting the user's already-built
   dependencies instead of reinstalling them.

   Because that mechanism has version- and ordering-dependent edge cases, it is
   never *trusted* — :meth:`WorktreeEnv.assert_authoritative` empirically
   confirms that the ``tools`` package resolves from inside the worktree before
   any verdict is believed, and aborts loudly otherwise. Once the ``tools``
   package object resolves to the worktree, every ``tools.X`` submodule import
   searches ``tools.__path__`` in the worktree — and a repair only ever mutates
   one ``tools/`` file — so that single check is sufficient for the repair's
   blast radius.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Repairs write whole files in a tight loop; consecutive rounds can produce
# same-size sources within the filesystem's mtime resolution, which defeats
# CPython's (mtime, size) .pyc invalidation and silently runs stale bytecode.
# Disabling bytecode writes forces recompilation from source every run — and
# keeps __pycache__ out of the worktree so it never dirties git status.
_NO_BYTECODE_ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

_GIT_TIMEOUT = 120
_VENV_TIMEOUT = 600
_DEFAULT_TEST_TIMEOUT = 600


class WorktreeError(RuntimeError):
    """Raised when worktree/venv setup fails or the authoritative-import guard
    trips. A BaseException-like hard stop in spirit: callers must not proceed to
    score a repair against a non-isolated environment."""


@dataclass
class TestRun:
    """Outcome of one pytest invocation inside the worktree venv.

    ``exit_code`` is pytest's raw exit status (None on timeout): 0 = all passed,
    1 = tests failed, 5 = no tests collected, 2/3/4 = interrupted/internal/usage
    error. Callers distinguish "ran and failed" (a real bug to repair) from "did
    not run cleanly" (a misconfiguration that must not look like a fail-or-pass).
    """

    passed: bool
    output: str  # combined stdout+stderr tail
    duration_seconds: float
    exit_code: int | None = None


def _run(
    args: list[str], *, cwd: Path, timeout: int, env: dict | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env
    )


def prune_orphan_worktrees(repo_root: Path) -> int:
    """Remove leftover ``evolve_code_*`` worktrees and dead registrations.

    A worktree's ``finally``/``destroy`` cleanup cannot run if the process is
    hard-killed (SIGKILL, an interrupted background run), which leaks the worktree
    dir + its git registration. Calling this at the start of a run self-heals
    those orphans. Scoped to the ``evolve_code_`` mkdtemp prefix, so it never
    touches a user's real worktree — but it assumes runs are serial (a single
    user); it does not distinguish a concurrent run's live worktree from a dead
    orphan, so don't run two campaigns against the same repo at once.
    """
    removed = 0
    res = _run(["git", "worktree", "list", "--porcelain"], cwd=repo_root, timeout=_GIT_TIMEOUT)
    for line in res.stdout.splitlines():
        if not line.startswith("worktree "):
            continue
        wt = Path(line[len("worktree "):].strip())
        root = wt.parent
        if root.name.startswith("evolve_code_") and wt.name == "wt":
            _run(["git", "worktree", "remove", "--force", str(wt)],
                 cwd=repo_root, timeout=_GIT_TIMEOUT)
            shutil.rmtree(root, ignore_errors=True)
            removed += 1
    _run(["git", "worktree", "prune"], cwd=repo_root, timeout=_GIT_TIMEOUT)
    return removed


def _detect_base_python(repo_root: Path) -> str:
    """The interpreter whose environment has the target repo's deps installed.

    Prefers the repo's own ``venv``/``.venv`` (where an editable Hermes install
    and all its dependencies live); falls back to the current interpreter for
    dependency-free targets (e.g. the synthetic test repo)."""
    for cand in (
        repo_root / "venv" / "bin" / "python",
        repo_root / ".venv" / "bin" / "python",
    ):
        if cand.exists():
            return str(cand)
    return sys.executable


class WorktreeEnv:
    """A disposable git worktree of ``repo_root`` with its own editable venv.

    Lifecycle: :meth:`create` → :meth:`assert_authoritative` →
    (:meth:`write_tool` / :meth:`run_test`)* → :meth:`destroy`. Use as a context
    manager to guarantee cleanup. ``create`` builds the worktree and venv;
    nothing is trusted until ``assert_authoritative`` passes.
    """

    def __init__(self, repo_root: Path, root: Path, worktree: Path, venv: Path):
        self.repo_root = repo_root
        self._root = root  # the run tempdir holding both worktree and venv
        self.worktree = worktree
        self.venv = venv
        self._output_tail_bytes = 6000
        self._base_python = sys.executable
        self._dep_sites: list[str] = []

    # -- construction ------------------------------------------------------

    @classmethod
    def create(
        cls,
        repo_root: Path,
        *,
        base_ref: str,
        base_python: str | None = None,
        output_tail_bytes: int = 6000,
    ) -> "WorktreeEnv":
        """Add a detached worktree at ``base_ref`` and build its isolated venv.

        The venv lives *beside* the worktree (not inside it) so the worktree's
        ``git status`` stays clean except for the repair — which keeps the PR
        diff to exactly the repaired file. ``base_python`` is the interpreter the
        venv is based on; it must be the environment that already has the target
        repo's dependencies (so the regression floor can import them). Defaults
        to the repo's own ``venv``/``.venv``, else the current interpreter.
        """
        root = Path(tempfile.mkdtemp(prefix="evolve_code_"))
        worktree = root / "wt"
        venv = root / "venv"
        env = cls(repo_root, root, worktree, venv)
        env._output_tail_bytes = output_tail_bytes
        env._base_python = base_python or _detect_base_python(repo_root)
        try:
            env._add_worktree(base_ref)
            env._build_venv()
        except BaseException:
            env.destroy()
            raise
        return env

    def _add_worktree(self, base_ref: str) -> None:
        res = _run(
            ["git", "worktree", "add", "--detach", str(self.worktree), base_ref],
            cwd=self.repo_root,
            timeout=_GIT_TIMEOUT,
        )
        if res.returncode != 0:
            raise WorktreeError(
                f"git worktree add failed (base_ref={base_ref!r}): {res.stderr.strip()}"
            )

    def _build_venv(self) -> None:
        # A *clean* venv (no --system-site-packages): the base interpreter is
        # often itself a venv symlinked to a bare base, so --system-site-packages
        # would chain to the base and miss the deps we need. Instead, the
        # worktree's own editable install maps every Hermes package to the
        # worktree, and the base venv's site-packages is added via PYTHONPATH at
        # test time (see _test_env) — that exposes the third-party dependencies
        # *without* executing the base's editable .pth, so the original Hermes
        # finder is never installed and there is no competing `tools` import.
        # Build isolation (pip default) supplies the setuptools backend.
        res = _run(
            [self._base_python, "-m", "venv", str(self.venv)],
            cwd=self._root,
            timeout=_VENV_TIMEOUT,
        )
        if res.returncode != 0:
            raise WorktreeError(f"venv creation failed: {res.stderr.strip()}")
        res = _run(
            [str(self.python), "-m", "pip", "install", "-e", str(self.worktree),
             "--no-deps", "-q"],
            cwd=self.worktree,
            timeout=_VENV_TIMEOUT,
            env=_NO_BYTECODE_ENV,
        )
        if res.returncode != 0:
            raise WorktreeError(
                f"editable install failed: {(res.stderr or res.stdout).strip()[-1500:]}"
            )
        self._dep_sites = self._query_dep_sites()

    def _query_dep_sites(self) -> list[str]:
        """The base interpreter's site-packages dirs, where the target repo's
        third-party deps live. Added to PYTHONPATH at test time so imports
        resolve, without running that environment's .pth files."""
        probe = "import site, json; print(json.dumps(site.getsitepackages()))"
        res = _run([self._base_python, "-c", probe], cwd=self._root, timeout=60)
        if res.returncode != 0:
            return []
        import json

        try:
            return [p for p in json.loads(res.stdout.strip() or "[]") if Path(p).is_dir()]
        except ValueError:
            return []

    def _test_env(self) -> dict:
        """Env for in-worktree python runs: no bytecode writes, plus the base
        interpreter's site-packages on PYTHONPATH for third-party deps."""
        existing = os.environ.get("PYTHONPATH", "")
        parts = [*self._dep_sites, existing] if existing else list(self._dep_sites)
        env = dict(_NO_BYTECODE_ENV)
        if parts:
            env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    # -- properties --------------------------------------------------------

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    # -- the authoritative-import guard -----------------------------------

    def assert_authoritative(self, package: str) -> None:
        """Confirm ``package`` resolves *entirely* from inside the worktree, else
        abort. ``package`` must be the top-level package that contains the file
        being repaired — guarding a sibling package proves nothing about the code
        actually under test.

        The load-bearing safety net for the venv mechanism: if the user's
        original install won the import race for any reason, the loop would score
        original code and silently certify nonsense. We never want a verdict from
        a non-isolated import, so a failed check is a hard stop. Every entry of
        the package's ``__file__``/``__path__`` must be under the worktree (a
        namespace package can list several roots; one outside is a breach).
        """
        probe = (
            f"import {package} as _m, os, sys, json\n"
            f"paths = [getattr(_m, '__file__', None)] + list(getattr(_m, '__path__', []))\n"
            f"sys.stdout.write(json.dumps([os.path.realpath(p) for p in paths if p]))\n"
        )
        res = _run([str(self.python), "-c", probe], cwd=self.worktree, timeout=60,
                   env=self._test_env())
        if res.returncode != 0:
            raise WorktreeError(
                f"could not import '{package}' in the isolated venv: {res.stderr.strip()}"
            )
        import json as _json

        try:
            resolved = _json.loads(res.stdout.strip() or "[]")
        except ValueError:
            resolved = []
        wt_real = str(Path(self.worktree).resolve())
        outside = [p for p in resolved if not p.startswith(wt_real)]
        if not resolved or outside:
            raise WorktreeError(
                f"isolation breach: '{package}' resolved to {resolved!r} "
                f"(outside the worktree {wt_real!r}: {outside or 'nothing resolved'}). "
                f"Refusing to score a Frankenstein import (mixed repaired/original code)."
            )

    # -- mutation + measurement -------------------------------------------

    def read_tool(self, relpath: str) -> str:
        return (self.worktree / relpath).read_text()

    def write_tool(self, relpath: str, src: str) -> None:
        """Write ``src`` to ``relpath`` in the worktree. Only tool source is
        ever written by the loop; test files are read-only (the gate verifies
        the test split was untouched against the worktree's git diff)."""
        (self.worktree / relpath).write_text(src)

    def diff(self) -> str:
        """The worktree's working-tree diff against its base (what a PR ships)."""
        res = _run(["git", "diff"], cwd=self.worktree, timeout=_GIT_TIMEOUT)
        return res.stdout

    def changed_files(self) -> list[str]:
        """Repo-relative *source* paths with a working-tree change (tracked or
        not). Build artifacts (``__pycache__``/``.pyc``/the egg-info the editable
        install drops) are not source changes and are filtered out so the gate's
        file-scope check sees only what a PR would actually ship."""
        res = _run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.worktree,
            timeout=_GIT_TIMEOUT,
        )
        out: list[str] = []
        for line in res.stdout.splitlines():
            if len(line) <= 3:
                continue
            path = line[3:].strip()
            # A rename renders as "old -> new"; the destination is what ships.
            if " -> " in path:
                path = path.split(" -> ", 1)[1].strip()
            if "__pycache__" in path or path.endswith((".pyc", ".egg-info", ".egg-link")):
                continue
            out.append(path)
        return out

    def run_test(
        self, *test_paths: str, timeout: int = _DEFAULT_TEST_TIMEOUT,
        extra_args: list[str] | None = None, full_output: bool = False,
    ) -> TestRun:
        """Run pytest on ``test_paths`` with the worktree venv, cwd=worktree.

        ``-p no:cacheprovider`` keeps the worktree free of a ``.pytest_cache``
        that would dirty its git status. ``full_output`` keeps the entire
        combined output instead of a tail — needed when the caller must parse the
        complete set of failing test ids (e.g. the baseline-diff regression floor),
        which a tail could truncate.
        """
        import time

        args = [
            str(self.python), "-m", "pytest", "-q", "--no-header",
            "-p", "no:cacheprovider", *(extra_args or []), *test_paths,
        ]
        start = time.monotonic()
        try:
            res = _run(args, cwd=self.worktree, timeout=timeout, env=self._test_env())
            duration = time.monotonic() - start
            combined = res.stdout + "\n" + res.stderr
            out = combined if full_output else combined[-self._output_tail_bytes:]
            return TestRun(passed=res.returncode == 0, output=out,
                           duration_seconds=duration, exit_code=res.returncode)
        except subprocess.TimeoutExpired:
            return TestRun(
                passed=False,
                output=f"pytest timed out after {timeout}s",
                duration_seconds=float(timeout),
                exit_code=None,
            )

    def failing_tests(self, *test_paths: str, timeout: int = _DEFAULT_TEST_TIMEOUT) -> set[str]:
        """Failing/erroring pytest node-ids for ``test_paths`` — the seam the oracle
        gate uses so it never re-implements parsing (SWEbenchEnv overrides it)."""
        from evolution.code.gate import _parse_pytest_failures  # noqa: PLC0415
        run = self.run_test(*test_paths, extra_args=["--tb=no"], full_output=True, timeout=timeout)
        return _parse_pytest_failures(run.output)

    # -- teardown ----------------------------------------------------------

    def destroy(self) -> None:
        """Remove the worktree (best-effort) and the whole run tempdir."""
        try:
            _run(
                ["git", "worktree", "remove", "--force", str(self.worktree)],
                cwd=self.repo_root,
                timeout=_GIT_TIMEOUT,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass  # best-effort; the tempdir removal below is the real cleanup
        shutil.rmtree(self._root, ignore_errors=True)
        # Prune the now-dangling worktree registration from the source repo.
        try:
            _run(["git", "worktree", "prune"], cwd=self.repo_root, timeout=_GIT_TIMEOUT)
        except (subprocess.TimeoutExpired, OSError):
            pass

    def __enter__(self) -> "WorktreeEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.destroy()

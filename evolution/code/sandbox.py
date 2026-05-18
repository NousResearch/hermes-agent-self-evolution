"""Soft sandbox for Phase 4 candidate execution.

What this provides:

- a fresh ``git worktree`` per candidate, so candidates can't see each other
- subprocess timeout on every command (we will not wait forever for pytest)
- path confinement: every command's working directory must resolve inside the
  worktree, never outside

What this does NOT provide on Windows:

- no network isolation
- no filesystem isolation beyond the worktree itself
- no resource limits beyond the timeout

A Linux ``bwrap`` / ``podman`` backend is intentionally left as a future
``Sandbox`` subclass; the orchestrator selects the backend at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of one sandboxed command invocation."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
            "stdout_tail": self.stdout[-2000:] if self.stdout else "",
            "stderr_tail": self.stderr[-2000:] if self.stderr else "",
        }


class SandboxError(RuntimeError):
    """Raised when sandbox preconditions (e.g. path confinement) are violated."""


class WorktreeSandbox:
    """Run commands confined to ``worktree_root`` with an enforced timeout."""

    def __init__(self, worktree_root: Path, *, default_timeout_seconds: int = 300):
        self.worktree_root = worktree_root.resolve()
        if not self.worktree_root.exists():
            raise SandboxError(f"worktree does not exist: {self.worktree_root}")
        self.default_timeout_seconds = int(default_timeout_seconds)

    def _confine(self, cwd: Path | None) -> Path:
        candidate = (cwd or self.worktree_root).resolve()
        try:
            candidate.relative_to(self.worktree_root)
        except ValueError as exc:
            raise SandboxError(
                f"cwd {candidate} escapes sandbox root {self.worktree_root}",
            ) from exc
        return candidate

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: int | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Run ``command`` inside the worktree with a hard timeout."""
        if not command:
            raise SandboxError("command must not be empty")
        for token in command:
            if not isinstance(token, str):
                raise SandboxError(f"command tokens must be strings, got {type(token)!r}")

        resolved_cwd = self._confine(cwd)
        timeout = int(timeout_seconds or self.default_timeout_seconds)
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=str(resolved_cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            elapsed = time.monotonic() - start
            return SandboxResult(
                command=list(command),
                returncode=completed.returncode,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                duration_seconds=elapsed,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - start
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, (bytes, bytearray)) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, (bytes, bytearray)) else (exc.stderr or "")
            return SandboxResult(
                command=list(command),
                returncode=124,
                stdout=stdout,
                stderr=stderr or f"timeout after {timeout}s",
                duration_seconds=elapsed,
                timed_out=True,
            )

    def run_pytest(
        self,
        *,
        test_target: str | None = None,
        timeout_seconds: int | None = None,
        extra_args: list[str] | None = None,
    ) -> SandboxResult:
        cmd = [
            sys.executable, "-m", "pytest", "-q", "--no-header",
            "--override-ini=addopts=",
        ]
        if extra_args:
            cmd.extend(extra_args)
        if test_target:
            target_path = Path(test_target)
            if not target_path.is_absolute():
                target_path = self.worktree_root / target_path
            cmd.append(str(target_path))
        return self.run(cmd, timeout_seconds=timeout_seconds)

    def run_ruff(self, *, timeout_seconds: int | None = None) -> SandboxResult:
        return self.run(
            [sys.executable, "-m", "ruff", "check", "."],
            timeout_seconds=timeout_seconds,
        )

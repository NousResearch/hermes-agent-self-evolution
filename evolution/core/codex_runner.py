"""Codex CLI subprocess runner for batched evolution phases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import subprocess


class CodexRunTimeout(RuntimeError):
    """Raised when a codex subprocess exceeds its timeout."""


def build_codex_command(codex_bin: str = "codex") -> list[str]:
    """Build the Codex CLI command using stdin prompt mode."""
    return [codex_bin, "exec", "-"]


def _run_subprocess(command: list[str], workdir: Path | str, timeout_seconds: float, prompt: str) -> str:
    """Run a subprocess and return stdout text."""
    completed = subprocess.run(
        command,
        cwd=str(workdir),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        input=prompt,
    )
    return completed.stdout.strip()


@dataclass
class CodexRunner:
    """Run bounded Codex CLI tasks in a repository worktree."""

    workdir: Path | str
    codex_bin: str = "codex"

    def run_json_task(self, prompt: str, timeout_seconds: float) -> dict:
        command = build_codex_command(codex_bin=self.codex_bin)
        try:
            stdout = _run_subprocess(command, self.workdir, timeout_seconds, prompt)
        except TimeoutError as exc:
            raise CodexRunTimeout(f"Codex timed out after {timeout_seconds}s") from exc
        except subprocess.TimeoutExpired as exc:
            raise CodexRunTimeout(f"Codex timed out after {timeout_seconds}s") from exc

        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Codex output was not valid JSON: {stdout!r}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"Codex output must be a JSON object, got {type(parsed).__name__}")
        return parsed

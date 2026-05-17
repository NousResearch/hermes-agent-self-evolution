"""Drive Hermes Agent via ``hermes -z`` one-shot mode and parse the result.

Sandboxes the subprocess by overriding ``HERMES_HOME`` (Hermes's own
config seam) and the process's ``HOME`` to the same tmp dir, and
running with ``cwd`` set to the task's per-task fixture directory.
That keeps the agent's interactive-shell aliases and personal
``.config`` paths invisible, and any file writes the agent does default
into a directory the harness controls.

The agent runs with ``HERMES_YOLO_MODE=1`` (Hermes one-shot mode sets
this itself) so it auto-approves tool prompts; v1 has no further
sandboxing. The agent can still write anywhere it has UID for.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from evolution.validation.agent_runner import AgentRunResult, TaskRunContext

logger = logging.getLogger(__name__)


_FINAL_TEXT_TAIL_BYTES = 4096
DEFAULT_TASK_TIMEOUT_SECONDS = 120


class HermesAgentRunner:
    """Invoke ``hermes -z`` and parse the resulting session JSON.

    The exit code is intentionally ignored — Hermes one-shot returns 0
    on almost every code path including agent-loop crashes. The session
    JSON is the only source of truth for what the agent did.
    """

    def __init__(
        self,
        hermes_command: str = "hermes",
        timeout_seconds: int = DEFAULT_TASK_TIMEOUT_SECONDS,
        user_config_path: Optional[Path] = None,
    ) -> None:
        self.hermes_command = hermes_command
        self.timeout_seconds = timeout_seconds
        # If set, copied into the sandboxed HERMES_HOME so the agent picks
        # up the user's credentials. Defaults to ``~/.hermes/config.yaml``.
        self.user_config_path = (
            user_config_path
            if user_config_path is not None
            else Path.home() / ".hermes" / "config.yaml"
        )

    def run(self, ctx: TaskRunContext) -> AgentRunResult:
        message = ctx.user_message
        sandbox = Path(tempfile.mkdtemp(prefix="cl_hermes_home_"))
        try:
            self._prime_sandbox(sandbox, ctx)
            env = {
                **os.environ,
                "HERMES_HOME": str(sandbox),
                "HOME": str(sandbox),
                **ctx.extra_env,
            }
            start = time.time()
            try:
                subprocess.run(
                    [self.hermes_command, "-z", message],
                    env=env,
                    cwd=str(ctx.fixture_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return AgentRunResult(
                    tool_calls_seq=[],
                    final_text_tail="",
                    duration_seconds=time.time() - start,
                    error=f"hermes -z timed out after {self.timeout_seconds}s",
                )
            except FileNotFoundError as exc:
                return AgentRunResult(
                    tool_calls_seq=[],
                    final_text_tail="",
                    duration_seconds=time.time() - start,
                    error=f"hermes command not found: {exc}",
                )
            duration = time.time() - start

            session_path = _find_latest_session(sandbox / "sessions")
            if session_path is None:
                return AgentRunResult(
                    tool_calls_seq=[],
                    final_text_tail="",
                    duration_seconds=duration,
                    error="no session JSON written by hermes -z",
                )
            return parse_session_result(session_path, duration_seconds=duration)
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    def _prime_sandbox(self, sandbox: Path, ctx: TaskRunContext) -> None:
        (sandbox / "sessions").mkdir(parents=True, exist_ok=True)
        if self.user_config_path.exists():
            shutil.copy2(self.user_config_path, sandbox / "config.yaml")
        if ctx.skills_src is not None and ctx.skills_src.is_dir():
            # Copy (not symlink) so an in-task write by the agent corrupts
            # only this sandbox, not the installer's persistent candidate.
            shutil.copytree(ctx.skills_src, sandbox / "skills")


def _find_latest_session(sessions_dir: Path) -> Optional[Path]:
    if not sessions_dir.exists():
        return None
    candidates = sorted(
        sessions_dir.glob("session_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_session_result(
    session_path: Path,
    *,
    duration_seconds: float,
) -> AgentRunResult:
    """Read a Hermes session JSON and extract the tool-call sequence + final text.

    Public for tests: hand-crafted fixture JSONs in
    ``tests/validation/test_hermes_runner.py`` exercise this directly
    rather than going through the subprocess layer.
    """
    try:
        data = json.loads(session_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return AgentRunResult(
            tool_calls_seq=[],
            final_text_tail="",
            duration_seconds=duration_seconds,
            error=f"could not parse session JSON at {session_path}: {exc}",
            session_path=session_path,
        )

    messages = data.get("messages") or []
    tool_calls_seq = _extract_tool_call_names(messages)
    final_text_tail = _extract_final_text_tail(messages)
    model_name = data.get("model")

    return AgentRunResult(
        tool_calls_seq=tool_calls_seq,
        final_text_tail=final_text_tail,
        duration_seconds=duration_seconds,
        model_name=model_name,
        session_path=session_path,
    )


def _extract_tool_call_names(messages: list[dict]) -> list[str]:
    """Pull tool names from every assistant turn's tool_calls.

    Hermes session JSON has been observed to carry tool_calls in two
    shapes depending on the model:
      - OpenAI-style nested:   {"function": {"name": "X", "arguments": ...}}
      - Flat:                  {"name": "X", "arguments": ...}
    Handle both. Multi-tool-call assistant turns contribute each call
    in order.
    """
    out: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = _call_name(call)
            if name:
                out.append(name)
    return out


def _call_name(call: dict) -> Optional[str]:
    fn = call.get("function")
    if isinstance(fn, dict):
        nested = fn.get("name")
        if nested:
            return str(nested)
    flat = call.get("name")
    return str(flat) if flat else None


def _extract_final_text_tail(messages: list[dict]) -> str:
    """Last 4096 chars of the last assistant message with text content."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content[-_FINAL_TEXT_TAIL_BYTES:]
    return ""

"""Agent runner protocol for closed-loop validation.

A runner takes one task and returns what the agent actually did. The
validator scores the result against the task's expected / forbidden
tool sets — never trusting the agent's exit code, which is unreliable
across backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol


@dataclass(frozen=True)
class AgentRunResult:
    """What one agent invocation produced.

    ``tool_calls_seq`` is the ordered list of tool *names* (not the full
    tool_call dicts) the agent invoked during the session. The validator
    only needs names for the expected / forbidden membership tests.

    ``error`` is set when the runner itself failed to drive the agent
    (subprocess timeout, no session JSON written, parse failure). It's
    distinct from "agent invoked a tool that failed" — that's still a
    valid run, just one where the agent struggled. Tasks with ``error``
    are counted as *abstentions* in the report, not as failures, so a
    transient timeout doesn't masquerade as evidence of regression.
    """

    tool_calls_seq: list[str]
    final_text_tail: str
    duration_seconds: float
    model_name: Optional[str] = None
    error: Optional[str] = None
    session_path: Optional[Path] = None


@dataclass(frozen=True)
class TaskRunContext:
    """Per-task state the validator hands the runner.

    ``fixture_dir`` is the per-task ``mkdtemp`` directory the validator
    populated from ``Task.fixture_setup`` before calling the runner.

    ``skills_src`` is the directory whose contents should be staged into
    the runner's per-task sandbox under ``skills/`` — used for skill-side
    closed-loop where the installer maintains a persistent writable
    copy of the candidate skill that needs to be re-staged into each
    task's ephemeral sandbox.
    """

    user_message: str
    fixture_dir: Path
    extra_env: dict[str, str] = field(default_factory=dict)
    skills_src: Optional[Path] = None


class AgentRunner(Protocol):
    """Drive an agent through a single task and capture the result."""

    def run(self, ctx: TaskRunContext) -> AgentRunResult:
        ...

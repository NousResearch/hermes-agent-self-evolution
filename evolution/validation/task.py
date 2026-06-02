"""Task + TaskSuite for closed-loop validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Task:
    """A single validation task: a user message + expected agent behavior.

    ``fixture_setup`` maps ``relative_path -> content``; each entry is
    materialized into the task's per-task tmp dir before the agent runs.
    The user message is then formatted with ``{fixture_dir}`` replaced
    by that dir's absolute path so the task can reference the fixture
    files unambiguously.

    Two verdict mechanisms are supported, depending on what the suite
    is validating:

    - ``expected_tools`` / ``forbidden_tools`` (tool-side): pass iff at
      least one expected tool was invoked AND no forbidden tool was
      invoked. Both empty is a degenerate "no-tool task" — the pass
      rule reduces to "agent didn't invoke any forbidden tools," which
      trivially passes if forbidden is empty too.
    - ``test_command`` (skill-side): pass iff the command exits zero
      when run in ``fixture_dir`` after the agent finishes. Use when
      the verdict is "did the agent's edits make the planted test
      pass" rather than "did the agent invoke the right tools." When
      set, takes precedence over the tool-call rule.

    ``expected_save_content`` is an optional rubric (not exact text)
    describing what a good ``memory(action='add')`` would contain. It
    feeds the prompt-section compound verdict's Layer 2 content judge; it
    has no effect on the Layer 1 tool-call rule above.
    """

    task_id: str
    user_message: str
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    fixture_setup: dict[str, str] = field(default_factory=dict)
    test_command: Optional[str] = None
    expected_save_content: Optional[str] = None

    def render_message(self, fixture_dir: Path) -> str:
        """Substitute ``{fixture_dir}`` in the message with the resolved path.

        Uses a plain ``str.replace`` so literal ``{`` / ``}`` in task
        content (Python dict literals, JSON snippets) survive verbatim.
        """
        return self.user_message.replace("{fixture_dir}", str(fixture_dir))


@dataclass(frozen=True)
class TaskSuite:
    """Ordered collection of tasks plus a content hash for audit.

    The sha256 lands in the validation report so regression-by-curation
    (quietly dropping a task to make a bad description pass) is caught
    at code review.
    """

    path: Path
    sha256: str
    tasks: tuple[Task, ...]

    @classmethod
    def from_jsonl(cls, path: Path) -> "TaskSuite":
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        tasks: list[Task] = []
        for lineno, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{lineno}: invalid JSON: {exc}"
                ) from exc
            tasks.append(_task_from_dict(obj, source=f"{path}:{lineno}"))
        if not tasks:
            raise ValueError(f"{path}: no tasks parsed (empty or all-comment file)")
        return cls(path=path, sha256=sha, tasks=tuple(tasks))


def _task_from_dict(obj: dict, *, source: str) -> Task:
    if "task_id" not in obj:
        raise ValueError(f"{source}: task missing required field 'task_id'")
    if "user_message" not in obj:
        raise ValueError(f"{source}: task missing required field 'user_message'")
    fixture_setup = obj.get("fixture_setup") or {}
    if not isinstance(fixture_setup, dict):
        raise ValueError(
            f"{source}: fixture_setup must be a dict of relative_path -> content"
        )
    test_command = obj.get("test_command")
    if test_command is not None and not isinstance(test_command, str):
        raise ValueError(
            f"{source}: test_command must be a string (got {type(test_command).__name__})"
        )
    expected_save_content = obj.get("expected_save_content")
    if expected_save_content is not None and not isinstance(expected_save_content, str):
        raise ValueError(
            f"{source}: expected_save_content must be a string "
            f"(got {type(expected_save_content).__name__})"
        )
    return Task(
        task_id=obj["task_id"],
        user_message=obj["user_message"],
        expected_tools=tuple(obj.get("expected_tools") or ()),
        forbidden_tools=tuple(obj.get("forbidden_tools") or ()),
        fixture_setup=dict(fixture_setup),
        test_command=test_command,
        expected_save_content=expected_save_content,
    )

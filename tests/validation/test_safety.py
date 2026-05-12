"""Safety tests for ClosedLoopValidator — the dangerous splice/restore path.

Covers:
  - Splice → simulated SIGKILL ⇒ ``.cl_backup`` still parses,
    next invocation refuses to start (forcing manual recovery).
  - Concurrent runs blocked via fcntl.flock.
  - sha256 drift between tasks aborts the phase.
  - Corrupt backup rejected at startup.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evolution.validation.agent_runner import AgentRunResult, TaskRunContext
from evolution.validation.task import Task, TaskSuite
from evolution.validation.validator import (
    ChecksumDriftError,
    ClosedLoopValidator,
    ConcurrentRunError,
    StaleBackupError,
    ValidationInputs,
    _BACKUP_SUFFIX,
    _LOCK_FILENAME,
)


# ---- Test doubles ----


class _StubInstaller:
    """ArtifactInstaller stub. Mutates target_path with the artifact's bytes
    (treating each artifact as the literal new file content)."""

    def __init__(self, target_path: Path):
        self.target_path = target_path
        self.install_calls = 0

    def install(self, artifact_source: Path) -> str:
        self.install_calls += 1
        self.target_path.write_bytes(artifact_source.read_bytes())
        import hashlib
        return hashlib.sha256(self.target_path.read_bytes()).hexdigest()


class _StubRunner:
    def __init__(self, results: list[AgentRunResult]):
        self._results = list(results)

    def run(self, ctx: TaskRunContext) -> AgentRunResult:
        return self._results.pop(0)


def _stub_task(task_id: str = "t1") -> Task:
    return Task(task_id=task_id, user_message="m", expected_tools=("patch",))


def _stub_suite(tmp_path: Path, task_ids: list[str]) -> TaskSuite:
    p = tmp_path / "suite.jsonl"
    import json
    p.write_text("\n".join(
        json.dumps({"task_id": tid, "user_message": "m", "expected_tools": ["patch"]})
        for tid in task_ids
    ) + "\n")
    return TaskSuite.from_jsonl(p)


def _valid_python_artifact(tmp_path: Path, name: str, body: str = "x = 1\n") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


# ---- Tests ----


class TestStaleBackupRefusal:
    def test_refuses_to_start_when_valid_backup_exists(self, tmp_path):
        target = tmp_path / "file_tools.py"
        target.write_text("# original\n")
        # Simulate prior run that died after writing backup, before restore.
        backup = target.with_suffix(target.suffix + _BACKUP_SUFFIX)
        backup.write_text("# original\n")

        suite = _stub_suite(tmp_path, ["t1"])
        installer = _StubInstaller(target)
        runner = _StubRunner([])
        validator = ClosedLoopValidator(installer, runner)
        artifact = _valid_python_artifact(tmp_path, "evolved.py")

        with pytest.raises(StaleBackupError) as exc:
            validator.validate(ValidationInputs(
                tool_name="patch", suite=suite,
                baseline_artifact=artifact, evolved_artifact=artifact,
            ))
        assert str(backup) in str(exc.value)

    def test_corrupt_backup_rejected_with_clear_message(self, tmp_path):
        target = tmp_path / "file_tools.py"
        target.write_text("# original\n")
        backup = target.with_suffix(target.suffix + _BACKUP_SUFFIX)
        backup.write_text("this is not python ::: ###")

        suite = _stub_suite(tmp_path, ["t1"])
        installer = _StubInstaller(target)
        runner = _StubRunner([])
        validator = ClosedLoopValidator(installer, runner)
        artifact = _valid_python_artifact(tmp_path, "evolved.py")

        with pytest.raises(StaleBackupError, match="corrupt"):
            validator.validate(ValidationInputs(
                tool_name="patch", suite=suite,
                baseline_artifact=artifact, evolved_artifact=artifact,
            ))


class TestConcurrentRun:
    def test_second_validator_fails_fast_under_flock(self, tmp_path):
        target = tmp_path / "file_tools.py"
        target.write_text("# original\n")
        # Grab an exclusive flock on the sentinel — a second
        # ClosedLoopValidator.validate() should fail fast.
        import fcntl
        lock_path = tmp_path / _LOCK_FILENAME
        guard = open(lock_path, "w")
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            suite = _stub_suite(tmp_path, ["t1"])
            installer = _StubInstaller(target)
            runner = _StubRunner([])
            validator = ClosedLoopValidator(installer, runner)
            artifact = _valid_python_artifact(tmp_path, "evolved.py")
            with pytest.raises(ConcurrentRunError):
                validator.validate(ValidationInputs(
                    tool_name="patch", suite=suite,
                    baseline_artifact=artifact, evolved_artifact=artifact,
                ))
        finally:
            fcntl.flock(guard.fileno(), fcntl.LOCK_UN)
            guard.close()


class TestChecksumDrift:
    def test_phase_aborts_when_target_mutated_between_tasks(self, tmp_path):
        target = tmp_path / "file_tools.py"
        target.write_text("# original\n")
        suite = _stub_suite(tmp_path, ["t1", "t2"])
        installer = _StubInstaller(target)

        def _mutate_then_run(ctx: TaskRunContext) -> AgentRunResult:
            # Simulate the YOLO-mode agent stomping on the spliced file.
            target.write_text("# corrupted by agent\n")
            return AgentRunResult(tool_calls_seq=["patch"], final_text_tail="", duration_seconds=1.0)

        runner = MagicMock()
        runner.run.side_effect = _mutate_then_run

        validator = ClosedLoopValidator(installer, runner)
        artifact = _valid_python_artifact(tmp_path, "evolved.py", body="# evolved\n")

        with pytest.raises(ChecksumDriftError):
            validator.validate(ValidationInputs(
                tool_name="patch", suite=suite,
                baseline_artifact=artifact, evolved_artifact=artifact,
            ))


class TestRestoreOnException:
    def test_runner_exception_still_restores_target(self, tmp_path):
        target = tmp_path / "file_tools.py"
        target.write_text("# original\n")
        original_bytes = target.read_bytes()

        suite = _stub_suite(tmp_path, ["t1"])
        installer = _StubInstaller(target)

        def _boom(ctx):
            raise RuntimeError("simulated agent crash")

        runner = MagicMock()
        runner.run.side_effect = _boom
        validator = ClosedLoopValidator(installer, runner)
        artifact = _valid_python_artifact(tmp_path, "evolved.py", body="# evolved\n")

        with pytest.raises(RuntimeError):
            validator.validate(ValidationInputs(
                tool_name="patch", suite=suite,
                baseline_artifact=artifact, evolved_artifact=artifact,
            ))

        assert target.read_bytes() == original_bytes
        assert not target.with_suffix(target.suffix + _BACKUP_SUFFIX).exists()

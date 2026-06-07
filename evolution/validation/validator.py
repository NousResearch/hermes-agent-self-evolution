"""ClosedLoopValidator — orchestrate baseline vs. evolved against a task suite.

The validator owns the dangerous part of the harness: mutating the
user's hermes-agent install in place. Defenses:

  - ``fcntl.flock`` on the target file's parent dir: concurrent runs
    fail fast rather than corrupting each other's restores.
  - ``.cl_backup`` written atomically + ast-validated. On startup the
    harness refuses to start if a stale, valid backup exists.
  - sha256 verification between every task: an in-suite agent that
    mutates the spliced file (YOLO mode + terminal) aborts the phase
    rather than silently corrupting later tasks' baselines.
  - ``try/finally`` restore from backup is the primary path. If
    everything else fails, the OS releases the flock on process death.
"""

from __future__ import annotations

import fcntl
import logging
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

from evolution.validation.agent_runner import AgentRunner, TaskRunContext
from evolution.validation.artifact_installer import (
    ArtifactInstaller,
    atomic_write_bytes,
    sha256_of,
)
from evolution.validation.report import (
    PhaseResult,
    TaskResult,
    ValidationReport,
    compute_win_loss,
    decide,
    score_task,
    summarize_phase,
)
from evolution.validation.task import Task, TaskSuite

logger = logging.getLogger(__name__)


_BACKUP_SUFFIX = ".cl_backup"
_LOCK_FILENAME = ".cl_validation.lock"
_SCHEMA_VERSION = "1"


class StaleBackupError(RuntimeError):
    """A previous harness run left a backup file that wasn't restored."""


class ConcurrentRunError(RuntimeError):
    """Another harness or interactive session holds the parent-dir lock."""


class ChecksumDriftError(RuntimeError):
    """The spliced tool file was mutated between tasks."""


@dataclass(frozen=True)
class ValidationInputs:
    tool_name: str
    suite: TaskSuite
    baseline_artifact: Path
    evolved_artifact: Path


class ClosedLoopValidator:
    """Run baseline + evolved phases against a task suite, produce a report.

    Single-run verdicts on small suites are noisy: tasks where the agent
    could plausibly pick more than one tool flip across identical-input
    runs at a rate that's been observed in the ~15-20% range. The
    decision rule (``n_wins >= 2 * n_losses``) assumes multi-run usage;
    for confident verdicts on small suites, invoke the harness 3+
    times and aggregate.
    """

    def __init__(
        self,
        installer: ArtifactInstaller,
        runner: AgentRunner,
        *,
        layer2_judge_factory: Optional[
            Callable[[Task], Optional[Callable[[list[dict]], float]]]
        ] = None,
        layer2_threshold: float = 0.7,
        reps: int = 1,
    ) -> None:
        self.installer = installer
        self.runner = runner
        # Per-task repetitions. reps=1 (the default, shared with the
        # tool/skill paths) is byte-for-byte the legacy single-run gate:
        # pass_rate collapses to 0.0/1.0 and passed == (pass_rate >= 0.5).
        # reps>1 turns each per-task verdict into a rate so a stochastic
        # trigger isn't decided by a single coin-flip run.
        self.reps = reps
        # Optional compound-verdict Layer 2 (prompt-section suites). The
        # factory builds a per-task scorer from the task — prompt-section
        # judging needs the task's expected_save_content rubric and message,
        # which a single global fn couldn't carry. When unset, scoring is
        # Layer 1 only and the tool-description path is unchanged.
        self.layer2_judge_factory = layer2_judge_factory
        self.layer2_threshold = layer2_threshold

    def validate(self, inputs: ValidationInputs) -> ValidationReport:
        target = self.installer.target_path
        backup_path = target.with_suffix(target.suffix + _BACKUP_SUFFIX)
        _refuse_if_stale_backup_exists(backup_path, self.installer)

        # ``skills_src`` on a task is a path relative to the suite file's
        # directory; tasks without a suite path on disk (synthetic suites) get
        # no skill staging.
        suite_dir = inputs.suite.path.parent if inputs.suite.path is not None else None

        with _exclusive_lock(target.parent):
            atomic_write_bytes(backup_path, target.read_bytes())
            self.installer.verify_backup(backup_path)
            try:
                baseline_results = self._run_phase(
                    inputs.suite,
                    artifact=inputs.baseline_artifact,
                    suite_dir=suite_dir,
                )
                evolved_results = self._run_phase(
                    inputs.suite,
                    artifact=inputs.evolved_artifact,
                    suite_dir=suite_dir,
                )
            finally:
                atomic_write_bytes(target, backup_path.read_bytes())
                backup_path.unlink(missing_ok=True)

        baseline = summarize_phase(baseline_results)
        evolved = summarize_phase(evolved_results)
        delta = compute_win_loss(baseline, evolved)
        decision, reasons = decide(baseline, evolved, delta)
        return ValidationReport(
            schema_version=_SCHEMA_VERSION,
            tool=inputs.tool_name,
            task_suite_path=str(inputs.suite.path),
            task_suite_sha256=inputs.suite.sha256,
            baseline=baseline,
            evolved=evolved,
            delta=delta,
            decision=decision,
            decision_reasons=reasons,
        )

    def _run_phase(
        self, suite: TaskSuite, *, artifact: Path, suite_dir: Optional[Path]
    ) -> list[TaskResult]:
        results: list[TaskResult] = []
        for task in suite.tasks:
            expected_sha = self.installer.install(artifact)
            result = self._run_one_task(task, suite_dir=suite_dir)
            # Verify the agent didn't write to the tool file during the task.
            # Drift here means later tasks would silently run a corrupt baseline,
            # so we abort the phase before that happens.
            _verify_no_drift(self.installer.target_path, expected_sha)
            results.append(result)
        return results

    def _run_one_task(
        self, task: Task, *, suite_dir: Optional[Path] = None
    ) -> TaskResult:
        layer2_judge_fn = (
            self.layer2_judge_factory(task)
            if self.layer2_judge_factory is not None
            else None
        )
        skills_src = (
            (suite_dir / task.skills_src)
            if (task.skills_src and suite_dir is not None)
            else None
        )
        n_pass = 0
        n_abstain = 0
        last_run = None
        for _ in range(self.reps):
            with tempfile.TemporaryDirectory(prefix="cl_fixture_") as fixture_tmp:
                fixture_dir = Path(fixture_tmp)
                _materialize_fixture(fixture_dir, task.fixture_setup)
                ctx = TaskRunContext(
                    user_message=task.render_message(fixture_dir),
                    fixture_dir=fixture_dir,
                    skills_src=skills_src,
                )
                run = self.runner.run(ctx)
                passed, abstained = score_task(
                    expected_tools=task.expected_tools,
                    forbidden_tools=task.forbidden_tools,
                    run=run,
                    test_command=task.test_command,
                    fixture_dir=fixture_dir,
                    layer2_judge_fn=layer2_judge_fn,
                    layer2_threshold=self.layer2_threshold,
                    expected_action=task.expected_action,
                    target_skill=task.target_skill,
                    stale_token=task.stale_token,
                )
            last_run = run
            if abstained:
                n_abstain += 1
            elif passed:
                n_pass += 1

        n_scored = self.reps - n_abstain
        pass_rate = (n_pass / n_scored) if n_scored else 0.0
        return TaskResult(
            task_id=task.task_id,
            passed=pass_rate >= 0.5,
            pass_rate=pass_rate,
            # A task abstains only if every rep abstained — partial
            # abstention still yields a rate over the reps that ran.
            abstained=n_scored == 0,
            tool_calls_seq=list(last_run.tool_calls_seq),
            duration_seconds=last_run.duration_seconds,
            model_name=last_run.model_name,
            error=last_run.error,
        )


def _refuse_if_stale_backup_exists(
    backup_path: Path, installer: ArtifactInstaller
) -> None:
    if not backup_path.exists():
        return
    try:
        installer.verify_backup(backup_path)
    except Exception as exc:
        raise StaleBackupError(
            f"Stale backup at {backup_path} is corrupt ({type(exc).__name__}: "
            f"{exc}). Inspect it manually; do not blindly restore."
        ) from exc
    raise StaleBackupError(
        f"Stale backup found at {backup_path} — a previous harness run did not "
        f"clean up. Restore it manually with:\n"
        f"    mv {backup_path} {backup_path.with_suffix('')}\n"
        f"then re-run the harness."
    )


@contextmanager
def _exclusive_lock(dir_path: Path) -> Iterator[None]:
    """Hold an exclusive flock on a sentinel file inside ``dir_path`` for
    the duration of the validator run. A second instance attempting the
    same lock fails fast with ``ConcurrentRunError``. The lockfile
    persists across runs; we just rely on the OS to release the flock
    when the holder's process dies.
    """
    lock_path = dir_path / _LOCK_FILENAME
    lock_fd = open(lock_path, "w")
    try:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConcurrentRunError(
                f"Another harness or interactive session holds {lock_path}. "
                f"Wait for it to finish or kill the other process."
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        finally:
            lock_fd.close()


def _verify_no_drift(target: Path, expected_sha: str) -> None:
    actual = sha256_of(target)
    if actual != expected_sha:
        raise ChecksumDriftError(
            f"Tool file {target} mutated unexpectedly between install and task "
            f"run (expected sha256 {expected_sha[:12]}…, got {actual[:12]}…). "
            f"This usually means a prior task's agent wrote to the file — "
            f"the phase is aborted to avoid corrupting later tasks' baselines."
        )


def _materialize_fixture(fixture_dir: Path, setup: dict[str, str]) -> None:
    for relative_path, content in setup.items():
        dest = fixture_dir / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)

"""End-to-end ClosedLoopValidator tests with a stub installer + runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution.validation.agent_runner import AgentRunResult, TaskRunContext
from evolution.validation.task import TaskSuite
from evolution.validation.validator import ClosedLoopValidator, ValidationInputs


class _StubInstaller:
    def __init__(self, target_path: Path):
        self.target_path = target_path

    def install(self, artifact_source: Path) -> str:
        self.target_path.write_bytes(artifact_source.read_bytes())
        import hashlib
        return hashlib.sha256(self.target_path.read_bytes()).hexdigest()

    def verify_backup(self, backup_path: Path) -> None:
        # Stub: any non-empty backup is fine for tests.
        if not backup_path.read_bytes():
            raise ValueError(f"empty backup at {backup_path}")


class _ScriptedRunner:
    """Runner that returns different tool_calls_seq depending on which
    artifact is currently installed. The artifact bytes are the
    'signature' the runner reads from disk to decide what to emit.
    """

    def __init__(self, target_path: Path, script: dict[bytes, list[str]]):
        self.target_path = target_path
        self.script = script

    def run(self, ctx: TaskRunContext) -> AgentRunResult:
        sig = self.target_path.read_bytes()
        seq = self.script.get(sig, [])
        return AgentRunResult(
            tool_calls_seq=seq,
            final_text_tail="ok",
            duration_seconds=0.1,
            model_name="test-model",
        )


def _write_suite(tmp_path: Path, tasks: list[dict]) -> TaskSuite:
    p = tmp_path / "suite.jsonl"
    p.write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    return TaskSuite.from_jsonl(p)


class TestClosedLoopValidatorHappyPath:
    def test_pass_when_evolved_strictly_improves(self, tmp_path):
        target = tmp_path / "tool.py"
        target.write_text("# original\n")

        baseline = tmp_path / "baseline.py"
        baseline.write_text("# baseline\n")
        evolved = tmp_path / "evolved.py"
        evolved.write_text("# evolved\n")

        suite = _write_suite(tmp_path, [
            {"task_id": "t1", "user_message": "do",
             "expected_tools": ["patch"], "forbidden_tools": []},
            {"task_id": "t2", "user_message": "do",
             "expected_tools": ["patch"], "forbidden_tools": []},
        ])
        installer = _StubInstaller(target)
        runner = _ScriptedRunner(target, {
            b"# baseline\n": ["read_file"],   # baseline doesn't pick patch
            b"# evolved\n":  ["patch"],       # evolved does
        })
        validator = ClosedLoopValidator(installer, runner)

        report = validator.validate(ValidationInputs(
            tool_name="patch", suite=suite,
            baseline_artifact=baseline, evolved_artifact=evolved,
        ))

        assert report.baseline.pass_rate == 0.0
        assert report.evolved.pass_rate == 1.0
        assert report.delta.n_wins == 2
        assert report.delta.n_losses == 0
        assert report.decision == "pass"
        assert report.task_suite_sha256 == suite.sha256

    def test_regression_when_evolved_loses_task(self, tmp_path):
        """Baseline passes task #4 (create file → write_file); evolved
        over-claims and tries patch → fails task #4. Models the v1 demo."""
        target = tmp_path / "tool.py"
        target.write_text("# original\n")
        baseline = tmp_path / "baseline.py"
        baseline.write_text("# baseline\n")
        evolved = tmp_path / "evolved.py"
        evolved.write_text("# evolved\n")

        suite = _write_suite(tmp_path, [
            {"task_id": "patch_correct", "user_message": "small edit",
             "expected_tools": ["patch"], "forbidden_tools": ["write_file"]},
            {"task_id": "write_correct", "user_message": "create new file",
             "expected_tools": ["write_file"], "forbidden_tools": ["patch"]},
        ])
        installer = _StubInstaller(target)
        runner = _ScriptedRunner(target, {
            b"# baseline\n": ["patch", "write_file"],  # picks correctly per task
            b"# evolved\n":  ["patch", "patch"],       # overclaims: uses patch for both
        })
        # The runner returns the same seq for every task in the phase (a
        # simplification); to model task-specific behavior, override.

        # Override runner to return task-specific results.
        class _PerTaskRunner:
            target_path = target
            calls = 0
            def run(self_, ctx):
                self_.calls += 1
                sig = target.read_bytes()
                # task 1: small edit → baseline picks patch, evolved picks patch
                # task 2: create new file → baseline picks write_file, evolved picks patch (wrong)
                if sig == b"# baseline\n":
                    seq = ["patch"] if self_.calls % 2 == 1 else ["write_file"]
                else:
                    seq = ["patch"]  # evolved always picks patch
                return AgentRunResult(
                    tool_calls_seq=seq, final_text_tail="", duration_seconds=0.1,
                )

        validator = ClosedLoopValidator(installer, _PerTaskRunner())
        report = validator.validate(ValidationInputs(
            tool_name="patch", suite=suite,
            baseline_artifact=baseline, evolved_artifact=evolved,
        ))

        assert report.baseline.n_passed == 2
        assert report.evolved.n_passed == 1   # passes task 1, fails task 2
        assert report.delta.n_losses == 1
        assert report.delta.n_wins == 0
        assert report.decision == "regression"

    def test_abstention_doesnt_count_as_loss(self, tmp_path):
        target = tmp_path / "tool.py"
        target.write_text("# original\n")
        baseline = tmp_path / "baseline.py"
        baseline.write_text("# baseline\n")
        evolved = tmp_path / "evolved.py"
        evolved.write_text("# evolved\n")

        suite = _write_suite(tmp_path, [
            {"task_id": "t1", "user_message": "do",
             "expected_tools": ["patch"], "forbidden_tools": []},
        ])
        installer = _StubInstaller(target)

        class _AbstainingRunner:
            target_path = target
            def run(self_, ctx):
                # First call (baseline phase): pass. Second call (evolved phase): error/abstain.
                sig = target.read_bytes()
                if sig == b"# baseline\n":
                    return AgentRunResult(
                        tool_calls_seq=["patch"], final_text_tail="",
                        duration_seconds=0.1,
                    )
                return AgentRunResult(
                    tool_calls_seq=[], final_text_tail="",
                    duration_seconds=0.1, error="hermes timed out",
                )

        validator = ClosedLoopValidator(installer, _AbstainingRunner())
        report = validator.validate(ValidationInputs(
            tool_name="patch", suite=suite,
            baseline_artifact=baseline, evolved_artifact=evolved,
        ))

        # Abstention is a tie, not a loss.
        assert report.delta.n_losses == 0
        assert report.delta.n_ties == 1
        assert report.evolved.n_abstained == 1

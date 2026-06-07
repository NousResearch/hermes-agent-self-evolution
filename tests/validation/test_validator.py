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


class TestClosedLoopValidatorLayer2:
    def test_layer2_judge_threaded_into_scoring(self, tmp_path):
        """A configured Layer 2 judge runs per scored task and can fail a
        task whose Layer 1 trigger passed."""
        target = tmp_path / "prompt_builder.py"
        target.write_text("MEMORY_GUIDANCE = 'orig'\n")
        baseline = tmp_path / "baseline.txt"
        baseline.write_text("baseline body")
        evolved = tmp_path / "evolved.txt"
        evolved.write_text("evolved body")

        suite = _write_suite(tmp_path, [
            {"task_id": "t1", "user_message": "save", "expected_tools": ["memory"]},
        ])

        class _MemoryRunner:
            def __init__(self, target_path):
                self.target_path = target_path

            def run(self, ctx):
                return AgentRunResult(
                    tool_calls_seq=["memory"], final_text_tail="ok",
                    duration_seconds=0.1, model_name="test-model",
                    tool_calls_with_args=[
                        {"name": "memory", "arguments": {"action": "add", "content": "x"}}
                    ],
                )

        judged = []
        tasks_seen = []

        def judge_factory(task):
            tasks_seen.append(task.task_id)

            def judge_fn(memory_calls):
                judged.append(memory_calls)
                return 0.2  # below threshold → Layer 2 fails the task

            return judge_fn

        validator = ClosedLoopValidator(
            _StubInstaller(target), _MemoryRunner(target),
            layer2_judge_factory=judge_factory, layer2_threshold=0.7,
        )
        report = validator.validate(ValidationInputs(
            tool_name="MEMORY_GUIDANCE", suite=suite,
            baseline_artifact=baseline, evolved_artifact=evolved,
        ))
        # Judge invoked once per phase (baseline + evolved) on the one task.
        assert len(judged) == 2
        # Factory received the task each phase.
        assert tasks_seen == ["t1", "t1"]
        # Both phases fail Layer 2 → 0 pass rate, no regression decision.
        assert report.baseline.pass_rate == 0.0
        assert report.evolved.pass_rate == 0.0


class TestClosedLoopValidatorReps:
    """Multi-rep rate-based gating. reps=1 (the default, shared with the
    tool/skill paths) must be byte-for-byte identical to legacy behavior.
    """

    def _suite(self, tmp_path):
        return _write_suite(tmp_path, [
            {"task_id": "t1", "user_message": "do",
             "expected_tools": ["patch"], "forbidden_tools": []},
        ])

    def test_reps1_default_matches_legacy_decision(self, tmp_path):
        """reps=1 (default) over a scripted runner produces the same
        per-task passed values, win/loss counts, and decision as legacy.

        Hand-computed legacy expectation: baseline doesn't pick patch
        (fail), evolved picks patch (pass) → 1 win, 0 losses, decision
        pass. With reps=1 pass_rate ∈ {0.0, 1.0} so passed == (rate>=0.5).
        """
        target = tmp_path / "tool.py"
        target.write_text("# original\n")
        baseline = tmp_path / "baseline.py"
        baseline.write_text("# baseline\n")
        evolved = tmp_path / "evolved.py"
        evolved.write_text("# evolved\n")

        installer = _StubInstaller(target)
        runner = _ScriptedRunner(target, {
            b"# baseline\n": ["read_file"],
            b"# evolved\n": ["patch"],
        })
        # Default reps (no kwarg) == reps=1 explicitly: both must agree.
        for validator in (
            ClosedLoopValidator(installer, runner),
            ClosedLoopValidator(installer, runner, reps=1),
        ):
            report = validator.validate(ValidationInputs(
                tool_name="patch", suite=self._suite(tmp_path),
                baseline_artifact=baseline, evolved_artifact=evolved,
            ))
            b = report.baseline.tasks[0]
            e = report.evolved.tasks[0]
            assert b.passed is False and b.pass_rate == 0.0
            assert e.passed is True and e.pass_rate == 1.0
            assert report.delta.n_wins == 1
            assert report.delta.n_losses == 0
            assert report.decision == "pass"

    def test_reps4_pass_rate_one_of_four(self, tmp_path):
        """reps=4, a task passing 1 of 4 reps → pass_rate 0.25, passed False
        (0.25 >= 0.5 is False)."""
        target = tmp_path / "tool.py"
        target.write_text("# original\n")
        baseline = tmp_path / "baseline.py"
        baseline.write_text("# baseline\n")
        evolved = tmp_path / "evolved.py"
        evolved.write_text("# evolved\n")

        class _OneOfFourRunner:
            target_path = target
            calls = 0

            def run(self_, ctx):
                self_.calls += 1
                # 1st rep picks patch (pass), reps 2-4 don't (fail).
                seq = ["patch"] if self_.calls == 1 else ["read_file"]
                return AgentRunResult(
                    tool_calls_seq=seq, final_text_tail="",
                    duration_seconds=0.1,
                )

        installer = _StubInstaller(target)
        validator = ClosedLoopValidator(installer, _OneOfFourRunner(), reps=4)
        report = validator.validate(ValidationInputs(
            tool_name="patch", suite=self._suite(tmp_path),
            baseline_artifact=baseline, evolved_artifact=evolved,
        ))
        # First phase run is baseline: 1 pass of 4 reps.
        assert report.baseline.tasks[0].pass_rate == 0.25
        assert report.baseline.tasks[0].passed is False

    def test_reps4_all_abstain_yields_zero_rate(self, tmp_path):
        """All reps abstaining → pass_rate 0.0 and the task is marked
        abstained (denominator excludes abstentions; all-abstain → 0.0)."""
        target = tmp_path / "tool.py"
        target.write_text("# original\n")
        baseline = tmp_path / "baseline.py"
        baseline.write_text("# baseline\n")
        evolved = tmp_path / "evolved.py"
        evolved.write_text("# evolved\n")

        class _AlwaysErrorRunner:
            target_path = target

            def run(self_, ctx):
                return AgentRunResult(
                    tool_calls_seq=[], final_text_tail="",
                    duration_seconds=0.1, error="timed out",
                )

        installer = _StubInstaller(target)
        validator = ClosedLoopValidator(installer, _AlwaysErrorRunner(), reps=4)
        report = validator.validate(ValidationInputs(
            tool_name="patch", suite=self._suite(tmp_path),
            baseline_artifact=baseline, evolved_artifact=evolved,
        ))
        assert report.baseline.tasks[0].pass_rate == 0.0
        assert report.baseline.tasks[0].abstained is True


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


class _RecordingRunner:
    """Records every TaskRunContext it receives; returns a fixed result."""

    def __init__(self, target_path: Path, result: AgentRunResult):
        self.target_path = target_path
        self._result = result
        self.contexts: list[TaskRunContext] = []

    def run(self, ctx: TaskRunContext) -> AgentRunResult:
        self.contexts.append(ctx)
        return self._result


def _ok_result() -> AgentRunResult:
    return AgentRunResult(
        tool_calls_seq=["memory"], final_text_tail="ok", duration_seconds=0.1,
    )


class TestClosedLoopValidatorSkillsSrc:
    def _artifacts(self, tmp_path):
        target = tmp_path / "tool.py"
        target.write_text("# original\n")
        baseline = tmp_path / "baseline.py"
        baseline.write_text("# baseline\n")
        evolved = tmp_path / "evolved.py"
        evolved.write_text("# evolved\n")
        return target, baseline, evolved

    def test_skills_src_resolved_relative_to_suite_dir(self, tmp_path):
        """A task with skills_src='myfix' → the runner's ctx.skills_src is
        <suite_dir>/myfix (resolved against the suite file's directory)."""
        target, baseline, evolved = self._artifacts(tmp_path)
        suite_dir = tmp_path / "suite_home"
        suite_dir.mkdir()
        suite_path = suite_dir / "suite.jsonl"
        suite_path.write_text(json.dumps(
            {"task_id": "t1", "user_message": "do",
             "expected_tools": ["memory"], "skills_src": "myfix"}
        ) + "\n")
        suite = TaskSuite.from_jsonl(suite_path)

        runner = _RecordingRunner(target, _ok_result())
        validator = ClosedLoopValidator(_StubInstaller(target), runner)
        validator.validate(ValidationInputs(
            tool_name="memory", suite=suite,
            baseline_artifact=baseline, evolved_artifact=evolved,
        ))
        assert runner.contexts, "runner was never called"
        for ctx in runner.contexts:
            assert ctx.skills_src == suite_dir / "myfix"

    def test_skills_src_none_leaves_ctx_unchanged(self, tmp_path):
        """A task without skills_src → ctx.skills_src is None."""
        target, baseline, evolved = self._artifacts(tmp_path)
        suite = _write_suite(tmp_path, [
            {"task_id": "t1", "user_message": "do", "expected_tools": ["memory"]},
        ])
        runner = _RecordingRunner(target, _ok_result())
        validator = ClosedLoopValidator(_StubInstaller(target), runner)
        validator.validate(ValidationInputs(
            tool_name="memory", suite=suite,
            baseline_artifact=baseline, evolved_artifact=evolved,
        ))
        assert runner.contexts
        for ctx in runner.contexts:
            assert ctx.skills_src is None


class TestClosedLoopValidatorActionVerdict:
    def _artifacts(self, tmp_path):
        target = tmp_path / "tool.py"
        target.write_text("# original\n")
        baseline = tmp_path / "baseline.py"
        baseline.write_text("# baseline\n")
        evolved = tmp_path / "evolved.py"
        evolved.write_text("# evolved\n")
        return target, baseline, evolved

    def test_action_params_forwarded_to_score_task(self, tmp_path, monkeypatch):
        """expected_action/target_skill/stale_token are forwarded into every
        score_task call."""
        target, baseline, evolved = self._artifacts(tmp_path)
        suite = _write_suite(tmp_path, [
            {"task_id": "t1", "user_message": "patch it",
             "expected_action": "patch", "target_skill": "s",
             "stale_token": "tok"},
        ])

        captured: list[dict] = []
        import evolution.validation.validator as validator_mod
        real_score_task = validator_mod.score_task

        def spy(**kwargs):
            captured.append(kwargs)
            return real_score_task(**kwargs)

        monkeypatch.setattr(validator_mod, "score_task", spy)

        runner = _RecordingRunner(target, _ok_result())
        validator = ClosedLoopValidator(_StubInstaller(target), runner)
        validator.validate(ValidationInputs(
            tool_name="s", suite=suite,
            baseline_artifact=baseline, evolved_artifact=evolved,
        ))
        assert captured, "score_task was never called"
        for kw in captured:
            assert kw["expected_action"] == "patch"
            assert kw["target_skill"] == "s"
            assert kw["stale_token"] == "tok"

    def test_no_new_fields_forwards_none(self, tmp_path, monkeypatch):
        """A task without the action fields forwards None for all three."""
        target, baseline, evolved = self._artifacts(tmp_path)
        suite = _write_suite(tmp_path, [
            {"task_id": "t1", "user_message": "do", "expected_tools": ["memory"]},
        ])

        captured: list[dict] = []
        import evolution.validation.validator as validator_mod
        real_score_task = validator_mod.score_task

        def spy(**kwargs):
            captured.append(kwargs)
            return real_score_task(**kwargs)

        monkeypatch.setattr(validator_mod, "score_task", spy)

        runner = _RecordingRunner(target, _ok_result())
        validator = ClosedLoopValidator(_StubInstaller(target), runner)
        validator.validate(ValidationInputs(
            tool_name="memory", suite=suite,
            baseline_artifact=baseline, evolved_artifact=evolved,
        ))
        assert captured
        for kw in captured:
            assert kw["expected_action"] is None
            assert kw["target_skill"] is None
            assert kw["stale_token"] is None

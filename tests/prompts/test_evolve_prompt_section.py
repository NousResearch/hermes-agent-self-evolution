"""Wiring tests for evolve_prompt_section — pure helpers + dry-run (no LM/agent)."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from click.testing import CliRunner

import fcntl

import pytest

from evolution.prompts.evolve_prompt_section import (
    _BACKUP_SUFFIX,
    _LOCK_FILENAME,
    _make_layer2_factory,
    _prompt_builder_guard,
    _run_one_task_score,
    _section_text_from_candidate,
    evolve_prompt_section,
    main,
    val_signal_warning,
)
from evolution.prompts.prompt_judge import ScoreResult
from evolution.prompts.prompt_module import PromptModule
from evolution.validation.agent_runner import AgentRunResult, TaskRunContext
from evolution.validation.task import Task


def _task(task_id: str, rubric: str | None = None) -> Task:
    return Task(
        task_id=task_id, user_message="m", expected_tools=("memory",),
        expected_save_content=rubric,
    )


# ---------------------------------------------------------------------------
# Fake runner helpers for _run_one_task_score tests
# ---------------------------------------------------------------------------

def _pass_result() -> AgentRunResult:
    return AgentRunResult(tool_calls_seq=["memory"], final_text_tail="", duration_seconds=0.0)


def _fail_result() -> AgentRunResult:
    return AgentRunResult(tool_calls_seq=[], final_text_tail="", duration_seconds=0.0)


def _abstain_result() -> AgentRunResult:
    return AgentRunResult(tool_calls_seq=[], final_text_tail="", duration_seconds=0.0, error="timeout")


class _ScriptedRunner:
    """Returns scripted AgentRunResults in order (cycling if exhausted)."""

    def __init__(self, results: list[AgentRunResult]):
        self._results = results
        self._idx = 0

    def run(self, ctx: TaskRunContext) -> AgentRunResult:
        result = self._results[self._idx % len(self._results)]
        self._idx += 1
        return result


def _no_layer2_factory(task):
    return None


class TestRunOneTaskScore:
    def test_reps1_pass_returns_score_result_1(self, tmp_path):
        task = _task("t1")
        runner = _ScriptedRunner([_pass_result()])
        result = _run_one_task_score(
            task, runner=runner, layer2_factory=_no_layer2_factory, layer2_threshold=0.7, reps=1
        )
        assert isinstance(result, ScoreResult)
        assert result.score == 1.0

    def test_reps1_fail_returns_score_result_0(self):
        task = _task("t1")
        runner = _ScriptedRunner([_fail_result()])
        result = _run_one_task_score(
            task, runner=runner, layer2_factory=_no_layer2_factory, layer2_threshold=0.7, reps=1
        )
        assert isinstance(result, ScoreResult)
        assert result.score == 0.0

    def test_reps1_abstain_returns_score_result_0(self):
        task = _task("t1")
        runner = _ScriptedRunner([_abstain_result()])
        result = _run_one_task_score(
            task, runner=runner, layer2_factory=_no_layer2_factory, layer2_threshold=0.7, reps=1
        )
        assert isinstance(result, ScoreResult)
        assert result.score == 0.0

    def test_reps4_one_pass_gives_quarter(self):
        # 1 pass, 3 fails → 1/4 = 0.25
        runner = _ScriptedRunner([_pass_result(), _fail_result(), _fail_result(), _fail_result()])
        result = _run_one_task_score(
            _task("t1"), runner=runner,
            layer2_factory=_no_layer2_factory, layer2_threshold=0.7, reps=4,
        )
        assert result.score == pytest.approx(0.25)
        assert "1/4" in result.feedback

    def test_abstentions_excluded_from_denominator(self):
        # 4 reps: abstain, abstain, pass, fail → scored=2, 1 pass → 0.5
        runner = _ScriptedRunner([
            _abstain_result(), _abstain_result(), _pass_result(), _fail_result(),
        ])
        result = _run_one_task_score(
            _task("t1"), runner=runner,
            layer2_factory=_no_layer2_factory, layer2_threshold=0.7, reps=4,
        )
        assert result.score == pytest.approx(0.5)

    def test_all_abstain_gives_zero(self):
        runner = _ScriptedRunner([_abstain_result()])
        result = _run_one_task_score(
            _task("t1"), runner=runner,
            layer2_factory=_no_layer2_factory, layer2_threshold=0.7, reps=3,
        )
        assert result.score == 0.0

    def test_feedback_contains_ratio_and_is_neutral(self):
        runner = _ScriptedRunner([_pass_result(), _fail_result(), _fail_result(), _fail_result()])
        result = _run_one_task_score(
            _task("t1"), runner=runner,
            layer2_factory=_no_layer2_factory, layer2_threshold=0.7, reps=4,
        )
        # Ratio present; no production-prompt wording (just a neutral summary)
        assert "1/4" in result.feedback
        assert result.feedback  # non-empty

    def test_reps_default_is_1(self):
        runner = _ScriptedRunner([_pass_result()])
        result = _run_one_task_score(
            _task("t1"), runner=runner,
            layer2_factory=_no_layer2_factory, layer2_threshold=0.7,
        )
        assert result.score == 1.0


class _RecordingRunner:
    """Records each TaskRunContext it receives; returns a fixed pass result."""

    def __init__(self):
        self.contexts = []

    def run(self, ctx: TaskRunContext) -> AgentRunResult:
        self.contexts.append(ctx)
        return _pass_result()


class TestRunOneTaskScoreSkillsSrc:
    def test_skills_src_resolved_relative_to_suite_dir(self, tmp_path):
        task = Task(task_id="t1", user_message="m", expected_tools=("memory",),
                    skills_src="myfix")
        runner = _RecordingRunner()
        _run_one_task_score(
            task, runner=runner, layer2_factory=_no_layer2_factory,
            layer2_threshold=0.7, suite_dir=tmp_path,
        )
        assert runner.contexts
        assert runner.contexts[0].skills_src == tmp_path / "myfix"

    def test_skills_src_none_when_no_field(self, tmp_path):
        task = _task("t1")
        runner = _RecordingRunner()
        _run_one_task_score(
            task, runner=runner, layer2_factory=_no_layer2_factory,
            layer2_threshold=0.7, suite_dir=tmp_path,
        )
        assert runner.contexts
        assert runner.contexts[0].skills_src is None

    def test_skills_src_none_when_no_suite_dir(self):
        """skills_src set but no suite_dir threaded → ctx.skills_src is None."""
        task = Task(task_id="t1", user_message="m", expected_tools=("memory",),
                    skills_src="myfix")
        runner = _RecordingRunner()
        _run_one_task_score(
            task, runner=runner, layer2_factory=_no_layer2_factory,
            layer2_threshold=0.7,
        )
        assert runner.contexts
        assert runner.contexts[0].skills_src is None


class TestRunOneTaskScoreActionVerdict:
    def test_action_params_forwarded_to_score_task(self, monkeypatch):
        task = Task(task_id="t1", user_message="m", expected_action="patch",
                    target_skill="s", stale_token="tok")
        runner = _RecordingRunner()

        captured = []
        import evolution.prompts.evolve_prompt_section as mod
        real = mod.score_task

        def spy(**kwargs):
            captured.append(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(mod, "score_task", spy)
        _run_one_task_score(
            task, runner=runner, layer2_factory=_no_layer2_factory,
            layer2_threshold=0.7,
        )
        assert captured
        for kw in captured:
            assert kw["expected_action"] == "patch"
            assert kw["target_skill"] == "s"
            assert kw["stale_token"] == "tok"

    def test_no_new_fields_forwards_none(self, monkeypatch):
        task = _task("t1")
        runner = _RecordingRunner()

        captured = []
        import evolution.prompts.evolve_prompt_section as mod
        real = mod.score_task

        def spy(**kwargs):
            captured.append(kwargs)
            return real(**kwargs)

        monkeypatch.setattr(mod, "score_task", spy)
        _run_one_task_score(
            task, runner=runner, layer2_factory=_no_layer2_factory,
            layer2_threshold=0.7,
        )
        assert captured
        for kw in captured:
            assert kw["expected_action"] is None
            assert kw["target_skill"] is None
            assert kw["stale_token"] is None


def test_layer2_factory_returns_none_without_rubric():
    factory = _make_layer2_factory(judge=None)
    assert factory(_task("t1", rubric=None)) is None
    assert callable(factory(_task("t2", rubric="a rubric")))


def test_section_text_from_candidate_module_and_dict():
    module = PromptModule("MEMORY_GUIDANCE", "candidate body")
    assert _section_text_from_candidate(module, "MEMORY_GUIDANCE") == "candidate body"
    instructions = module.passthrough.predict.signature.instructions
    assert (
        _section_text_from_candidate(
            {"passthrough.predict": instructions}, "MEMORY_GUIDANCE"
        )
        == "candidate body"
    )


def _fake_repo(tmp_path: Path) -> Path:
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "prompt_builder.py").write_text(textwrap.dedent('''\
        MEMORY_GUIDANCE = "Save durable facts about the user."
    '''))
    return tmp_path


def _suite(tmp_path: Path) -> Path:
    p = tmp_path / "suite.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"task_id": "s1", "user_message": "I use uv.",
         "expected_tools": ["memory"], "expected_save_content": "prefers uv"},
        {"task_id": "n1", "user_message": "summarize work",
         "expected_tools": [], "forbidden_tools": ["memory"]},
    ]) + "\n")
    return p


def test_dry_run_writes_gate_decision(tmp_path):
    repo = _fake_repo(tmp_path)
    suite = _suite(tmp_path)
    out = tmp_path / "out"
    result = evolve_prompt_section(
        section_name="MEMORY_GUIDANCE", hermes_repo=repo, tasks_path=suite,
        dry_run=True, output_dir=out,
    )
    assert result["decision"] == "dry_run"
    gate = json.loads((out / "gate_decision.json").read_text())
    assert gate["artifact_type"] == "prompt_section"
    assert gate["target_section"] == "MEMORY_GUIDANCE"
    # The baseline file must be byte-identical after a dry run (untouched).
    assert "Save durable facts about the user." in (
        repo / "agent" / "prompt_builder.py"
    ).read_text()


def test_baseline_override_file_replaces_live_section(tmp_path):
    repo = _fake_repo(tmp_path)
    suite = _suite(tmp_path)
    override = tmp_path / "weak.txt"
    override.write_text("a deliberately weak baseline")
    out = tmp_path / "out"
    evolve_prompt_section(
        section_name="MEMORY_GUIDANCE", hermes_repo=repo, tasks_path=suite,
        dry_run=True, output_dir=out, baseline_override_file=override,
    )
    gate = json.loads((out / "gate_decision.json").read_text())
    assert gate["baseline_chars"] == len("a deliberately weak baseline")
    # The live file is never touched by an override dry run.
    assert "Save durable facts about the user." in (
        repo / "agent" / "prompt_builder.py"
    ).read_text()


class TestPromptBuilderGuard:
    def test_restores_bytes_even_on_exception(self, tmp_path):
        target = tmp_path / "pb.py"
        target.write_text("ORIGINAL = 1\n")
        original = target.read_bytes()
        with pytest.raises(RuntimeError, match="boom"):
            with _prompt_builder_guard(target):
                target.write_text("MUTATED = 2\n")
                raise RuntimeError("boom")
        assert target.read_bytes() == original
        assert not target.with_suffix(target.suffix + _BACKUP_SUFFIX).exists()

    def test_refuses_stale_backup(self, tmp_path):
        target = tmp_path / "pb.py"
        target.write_text("X = 1\n")
        target.with_suffix(target.suffix + _BACKUP_SUFFIX).write_text("stale")
        with pytest.raises(RuntimeError, match="[Ss]tale backup"):
            with _prompt_builder_guard(target):
                pass

    def test_refuses_when_another_run_holds_the_lock(self, tmp_path):
        target = tmp_path / "pb.py"
        target.write_text("X = 1\n")
        other = open(target.parent / _LOCK_FILENAME, "w")
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(RuntimeError, match="holds"):
                with _prompt_builder_guard(target):
                    pass
        finally:
            fcntl.flock(other.fileno(), fcntl.LOCK_UN)
            other.close()


def test_rejects_single_task_suite(tmp_path):
    repo = _fake_repo(tmp_path)
    suite = tmp_path / "one.jsonl"
    suite.write_text(json.dumps({
        "task_id": "only", "user_message": "x", "expected_tools": ["memory"],
    }) + "\n")
    with pytest.raises(ValueError, match="at least 2"):
        evolve_prompt_section(
            section_name="MEMORY_GUIDANCE", hermes_repo=repo, tasks_path=suite,
            dry_run=True, output_dir=tmp_path / "out",
        )


class TestValSignalWarning:
    def test_all_zero_rates_warns(self):
        w = val_signal_warning({"t1": 0.0, "t2": 0.05})
        assert w is not None
        assert set(w["task_ids"]) == {"t1", "t2"}
        assert w["rates"] == {"t1": 0.0, "t2": 0.05}

    def test_all_one_rates_warns(self):
        w = val_signal_warning({"t1": 1.0, "t2": 0.97})
        assert w is not None
        assert set(w["task_ids"]) == {"t1", "t2"}

    def test_mixed_rates_no_warning(self):
        assert val_signal_warning({"t1": 0.0, "t2": 0.5, "t3": 1.0}) is None

    def test_single_midrange_rate_no_warning(self):
        assert val_signal_warning({"t1": 0.4}) is None

    def test_empty_input_no_warning(self):
        assert val_signal_warning({}) is None

    def test_warning_dict_includes_reason(self):
        w = val_signal_warning({"t1": 0.0})
        assert w is not None
        assert "reason" in w


class TestRepsFlags:
    def _common(self, repo, suite, tmp_path):
        return [
            "--section", "MEMORY_GUIDANCE",
            "--hermes-repo", str(repo),
            "--tasks", str(suite),
            "--dry-run",
            "--output-dir", str(tmp_path / "out"),
        ]

    def test_default_reps_passed_through(self, tmp_path, monkeypatch):
        repo = _fake_repo(tmp_path)
        suite = _suite(tmp_path)
        captured = {}
        import evolution.prompts.evolve_prompt_section as mod

        def fake(**kwargs):
            captured.update(kwargs)
            return {"decision": "dry_run"}

        monkeypatch.setattr(mod, "evolve_prompt_section", fake)
        res = CliRunner().invoke(mod.main, self._common(repo, suite, tmp_path))
        assert res.exit_code == 0, res.output
        assert captured["fitness_reps"] == 3
        assert captured["gate_reps"] == 5

    def test_explicit_reps_passed_through(self, tmp_path, monkeypatch):
        repo = _fake_repo(tmp_path)
        suite = _suite(tmp_path)
        captured = {}
        import evolution.prompts.evolve_prompt_section as mod

        def fake(**kwargs):
            captured.update(kwargs)
            return {"decision": "dry_run"}

        monkeypatch.setattr(mod, "evolve_prompt_section", fake)
        res = CliRunner().invoke(mod.main, self._common(repo, suite, tmp_path) + [
            "--fitness-reps", "2", "--gate-reps", "9",
        ])
        assert res.exit_code == 0, res.output
        assert captured["fitness_reps"] == 2
        assert captured["gate_reps"] == 9


def test_evolve_accepts_gate_reps_and_wires_validator():
    """gate_reps is a param defaulting to 1, and the deploy-gate validator is
    constructed with reps=gate_reps (asserted statically — the construction is
    deep inside an LM/agent path we don't pay for here)."""
    import inspect
    sig = inspect.signature(evolve_prompt_section)
    assert sig.parameters["gate_reps"].default == 1
    src = inspect.getsource(evolve_prompt_section)
    assert "reps=gate_reps" in src


def test_cli_dry_run_exits_zero(tmp_path):
    repo = _fake_repo(tmp_path)
    suite = _suite(tmp_path)
    runner = CliRunner()
    res = runner.invoke(main, [
        "--section", "MEMORY_GUIDANCE",
        "--hermes-repo", str(repo),
        "--tasks", str(suite),
        "--dry-run",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert res.exit_code == 0, res.output


def test_synth_feedback_action_task_states_patch_objective():
    from evolution.prompts.evolve_prompt_section import _synth_feedback
    from evolution.validation.task import Task
    task = Task(
        task_id="patch-stale-flag", user_message="use it",
        expected_tools=("skill_manage",), expected_action="patch",
        target_skill="line-counter", stale_token="wc --lines",
    )
    fb = _synth_feedback(task, "passed 0/4")
    assert "line-counter" in fb and "wc --lines" in fb
    assert "PROACTIVELY" in fb and "skill_manage(action='patch')" in fb
    assert "passed 0/4" in fb


def test_synth_feedback_control_states_do_not_patch():
    from evolution.prompts.evolve_prompt_section import _synth_feedback
    from evolution.validation.task import Task
    task = Task(task_id="ctl", user_message="use it", forbidden_tools=("skill_manage",))
    fb = _synth_feedback(task, "passed 4/4")
    assert "CORRECT" in fb and "NOT patch" in fb


def test_synth_feedback_generic_membership_task():
    from evolution.prompts.evolve_prompt_section import _synth_feedback
    from evolution.validation.task import Task
    task = Task(task_id="mem", user_message="x", expected_tools=("memory",))
    fb = _synth_feedback(task, "passed 3/4")
    assert "memory" in fb and "objective" in fb

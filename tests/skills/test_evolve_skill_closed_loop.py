"""Tests for the closed-loop integration on the skill-evolution path.

Exercises the integration seams independently of GEPA: SkillModule's
behavioral branch, the cache-construction helper, the behavioral-example
loader's task_input shape, and the gate_mode policy. End-to-end behavior
is exercised by the manual smoke harness at
``tests/manual/skill_closed_loop_smoke.py`` (heavy, real LM spend).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from evolution.core.behavioral_example import build_behavioral_examples
from evolution.skills.evolve_skill import (
    _load_behavioral_examples_from_suite,
    _maybe_build_closed_loop_cache_skill,
    _should_use_cl_primary,
)
from evolution.skills.skill_module import SkillModule
from evolution.validation.task import TaskSuite


_SUITE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "evolution"
    / "validation"
    / "suites"
    / "systematic_debugging.jsonl"
)


@pytest.fixture
def fake_skill_path(tmp_path):
    src = tmp_path / "systematic_debugging"
    src.mkdir()
    md = src / "SKILL.md"
    md.write_text(
        "---\nname: systematic_debugging\n---\n\n# Weak baseline\n\nTry stuff.\n"
    )
    return md


# ---------------------------------------------------------------------------
# SkillModule.forward behavioral branch
# ---------------------------------------------------------------------------


class TestSkillModuleBehavioralBranch:
    def test_normal_branch_calls_predictor(self):
        # closed_loop_task_id absent → take the LM path. We mock the
        # predictor to avoid a real LM call.
        module = SkillModule("body")
        with patch.object(module, "predictor") as mock_pred:
            mock_pred.return_value = type("R", (), {"output": "hello"})()
            result = module.forward(task_input="do X")
        assert result.output == "hello"
        mock_pred.assert_called_once_with(task_input="do X")

    def test_behavioral_branch_skips_predictor(self):
        # closed_loop_task_id present → no LM call; marker fields are
        # stuffed onto the Prediction for the metric to route. Real
        # SkillModule (no mock) because the behavioral branch must not
        # touch the predictor at all — if it did, this test would hang
        # waiting for a non-existent LM configuration.
        module = SkillModule("evolved body text")
        result = module.forward(
            task_input="placeholder",
            closed_loop_task_id="t1",
        )
        assert result._closed_loop_task_id == "t1"
        assert result._candidate_text == "evolved body text"
        assert result.output == ""


# ---------------------------------------------------------------------------
# Behavioral-example loader (skill-side task_input shape)
# ---------------------------------------------------------------------------


class TestBehavioralExampleLoader:
    def test_skill_examples_use_task_input_field(self, tmp_path):
        suite_path = tmp_path / "s.jsonl"
        suite_path.write_text(
            '{"task_id": "a", "user_message": "debug a", "test_command": "python t.py"}\n'
            '{"task_id": "b", "user_message": "debug b", "test_command": "python t.py"}\n'
        )
        examples = _load_behavioral_examples_from_suite(suite_path)
        assert len(examples) == 2
        # Skill modules take `task_input`; the loader must use that field name.
        assert "task_input" in examples[0].inputs()
        assert "closed_loop_task_id" in examples[0].inputs()
        assert examples[0].task_input == "debug a"

    def test_tool_side_default_still_uses_task_field(self, tmp_path):
        # build_behavioral_examples default preserves tool-path shape.
        suite_path = tmp_path / "s.jsonl"
        suite_path.write_text(
            '{"task_id": "a", "user_message": "do a", "expected_tools": ["patch"]}\n'
        )
        examples = build_behavioral_examples(TaskSuite.from_jsonl(suite_path))
        assert "task" in examples[0].inputs()
        assert examples[0].task == "do a"

    def test_suite_override_restricts_examples_to_train_split(self, tmp_path):
        # The --compile-floor leakage guard: GEPA's behavioral examples must come
        # ONLY from the train split, never the floor's holdout — else the evolved
        # arm has seen the holdout it's gated on.
        from evolution.validation.task import split_train_holdout

        suite_path = tmp_path / "s.jsonl"
        suite_path.write_text(
            "\n".join(
                f'{{"task_id": "t{i}", "user_message": "m{i}", "test_command": "python t.py"}}'
                for i in range(10)
            ) + "\n"
        )
        full = TaskSuite.from_jsonl(suite_path)
        train, holdout = split_train_holdout(full.tasks, holdout_ratio=0.3, seed=42)
        train_suite = TaskSuite(path=full.path, sha256=full.sha256, tasks=tuple(train))
        examples = _load_behavioral_examples_from_suite(
            suite_path, suite_override=train_suite
        )
        ex_ids = {e.closed_loop_task_id for e in examples}
        train_ids = {t.task_id for t in train}
        holdout_ids = {t.task_id for t in holdout}
        assert ex_ids == train_ids
        assert not (ex_ids & holdout_ids)  # no holdout task leaked into training


# ---------------------------------------------------------------------------
# Cache construction helper
# ---------------------------------------------------------------------------


class TestMaybeBuildClosedLoopCacheSkill:
    def test_none_suite_returns_none(self, fake_skill_path):
        cache = _maybe_build_closed_loop_cache_skill(
            skill_name="systematic_debugging",
            skill_path=fake_skill_path,
            baseline_skill_body="body",
            suite_path=None,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
        )
        assert cache is None

    def test_constructs_cache_with_skill_installer_and_md_suffix(
        self, fake_skill_path
    ):
        cache = _maybe_build_closed_loop_cache_skill(
            skill_name="systematic_debugging",
            skill_path=fake_skill_path,
            baseline_skill_body="baseline body text",
            suite_path=_SUITE_FIXTURE,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
        )
        assert cache is not None
        # Baseline written as raw text via write_text_artifact (no JSON envelope).
        baseline_text = cache._baseline_path.read_text()
        assert baseline_text == "baseline body text"
        # Suffix is .md for skill artifacts (not the default .json).
        assert cache._baseline_path.suffix == ".md"

    def test_suite_override_scopes_cache_to_holdout(self, fake_skill_path):
        # Under --compile-floor the cache scores baseline/evolved/floor on the
        # holdout split, so the cache's suite must be the override, not the file.
        full = TaskSuite.from_jsonl(_SUITE_FIXTURE)
        holdout = TaskSuite(
            path=full.path, sha256=full.sha256, tasks=full.tasks[:1]
        )
        cache = _maybe_build_closed_loop_cache_skill(
            skill_name="systematic_debugging",
            skill_path=fake_skill_path,
            baseline_skill_body="body",
            suite_path=_SUITE_FIXTURE,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
            suite_override=holdout,
        )
        assert cache is not None
        assert cache._suite.tasks == holdout.tasks
        assert len(cache._suite.tasks) == 1 < len(full.tasks)
        # Default gate_mode for skills is "sampled" — caller passes "always"
        # only when wiring trainset mode.
        assert cache.gate_mode == "sampled"

    def test_gate_mode_always_propagates(self, fake_skill_path):
        cache = _maybe_build_closed_loop_cache_skill(
            skill_name="systematic_debugging",
            skill_path=fake_skill_path,
            baseline_skill_body="body",
            suite_path=_SUITE_FIXTURE,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
            gate_mode="always",
        )
        assert cache.gate_mode == "always"

    def test_agent_model_plumbed_into_runner(self, fake_skill_path):
        # The closed-loop validator's runner picks up the agent_model
        # override so `hermes -z -m MODEL` runs instead of using the
        # user's ~/.hermes/config.yaml default.
        cache = _maybe_build_closed_loop_cache_skill(
            skill_name="systematic_debugging",
            skill_path=fake_skill_path,
            baseline_skill_body="body",
            suite_path=_SUITE_FIXTURE,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
            agent_model="gpt-4o-mini",
        )
        assert cache._validator.runner.model == "gpt-4o-mini"

    def test_agent_model_none_leaves_runner_model_none(self, fake_skill_path):
        # No override → runner.model is None and hermes uses config default.
        cache = _maybe_build_closed_loop_cache_skill(
            skill_name="systematic_debugging",
            skill_path=fake_skill_path,
            baseline_skill_body="body",
            suite_path=_SUITE_FIXTURE,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
        )
        assert cache._validator.runner.model is None

    def test_agent_timeout_seconds_plumbed_into_runner(self, fake_skill_path):
        # Slow reasoning models (o1/o3-family) need more than the 120s default.
        cache = _maybe_build_closed_loop_cache_skill(
            skill_name="systematic_debugging",
            skill_path=fake_skill_path,
            baseline_skill_body="body",
            suite_path=_SUITE_FIXTURE,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
            agent_timeout_seconds=300,
        )
        assert cache._validator.runner.timeout_seconds == 300

    def test_default_backend_is_hermes_runner(self, fake_skill_path):
        from evolution.validation.hermes_runner import HermesAgentRunner

        cache = _maybe_build_closed_loop_cache_skill(
            skill_name="systematic_debugging",
            skill_path=fake_skill_path,
            baseline_skill_body="body",
            suite_path=_SUITE_FIXTURE,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
        )
        assert isinstance(cache._validator.runner, HermesAgentRunner)

    def test_claude_backend_uses_claude_runner(self, fake_skill_path):
        # agent_backend="claude" swaps in the Claude Code runner (delivers the
        # candidate skill as a plugin to `claude -p`); installer + cache unchanged.
        from evolution.validation.claude_runner import ClaudeCodeAgentRunner

        cache = _maybe_build_closed_loop_cache_skill(
            skill_name="systematic_debugging",
            skill_path=fake_skill_path,
            baseline_skill_body="body",
            suite_path=_SUITE_FIXTURE,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
            agent_backend="claude",
            agent_model="opus",
            agent_timeout_seconds=420,
        )
        assert isinstance(cache._validator.runner, ClaudeCodeAgentRunner)
        assert cache._validator.runner.model == "opus"
        assert cache._validator.runner.timeout_seconds == 420

    def test_claude_backend_defaults_model_when_unset(self, fake_skill_path):
        # Unlike Hermes (model=None → config default), the Claude runner has its
        # own "sonnet" default; passing None must NOT clobber it.
        cache = _maybe_build_closed_loop_cache_skill(
            skill_name="systematic_debugging",
            skill_path=fake_skill_path,
            baseline_skill_body="body",
            suite_path=_SUITE_FIXTURE,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
            agent_backend="claude",
        )
        assert cache._validator.runner.model == "sonnet"


class TestShouldUseClPrimary:
    """The deploy gate routes to the closed-loop behavioral oracle when this is
    True (else it gates on the synthetic-judge holdout)."""

    def test_weak_signal_band_uses_cl_primary(self):
        assert _should_use_cl_primary(
            gate_primary=False, band="weak_signal",
            cl_per_example=[0.0, 1.0], has_cache=True) is True

    def test_other_band_without_flag_does_not(self):
        assert _should_use_cl_primary(
            gate_primary=False, band="no_headroom",
            cl_per_example=[0.0, 0.0], has_cache=True) is False

    def test_gate_primary_forces_cl_primary_off_band(self):
        # The new flag: force the behavioral gate even when the band is not
        # weak_signal (e.g. a binary convention oracle stuck at 0.0).
        assert _should_use_cl_primary(
            gate_primary=True, band="no_headroom",
            cl_per_example=[0.0, 0.0], has_cache=True) is True

    def test_gate_primary_needs_cl_vector(self):
        # Forcing the flag without a baseline CL vector (no pre-flight CL) must
        # NOT silently route to a gate that has no data — falls back to False.
        assert _should_use_cl_primary(
            gate_primary=True, band="healthy",
            cl_per_example=None, has_cache=True) is False
        assert _should_use_cl_primary(
            gate_primary=True, band="healthy",
            cl_per_example=[], has_cache=True) is False

    def test_gate_primary_needs_cache(self):
        assert _should_use_cl_primary(
            gate_primary=True, band="weak_signal",
            cl_per_example=[1.0], has_cache=False) is False

    def test_agent_timeout_none_keeps_runner_default(self, fake_skill_path):
        # No override → runner uses its DEFAULT_TASK_TIMEOUT_SECONDS (120s).
        from evolution.validation.hermes_runner import DEFAULT_TASK_TIMEOUT_SECONDS

        cache = _maybe_build_closed_loop_cache_skill(
            skill_name="systematic_debugging",
            skill_path=fake_skill_path,
            baseline_skill_body="body",
            suite_path=_SUITE_FIXTURE,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
        )
        assert cache._validator.runner.timeout_seconds == DEFAULT_TASK_TIMEOUT_SECONDS

    def test_installer_skills_src_is_under_workdir(self, fake_skill_path):
        # The cache's validator holds an installer with skills_src — confirming
        # the wiring is intact so the runner sees the candidate.
        cache = _maybe_build_closed_loop_cache_skill(
            skill_name="systematic_debugging",
            skill_path=fake_skill_path,
            baseline_skill_body="body",
            suite_path=_SUITE_FIXTURE,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
        )
        installer = cache._validator.installer
        assert installer.skills_src.name == "skills"
        # The baseline skill was copied into the installer's workdir.
        target = installer.skills_src / "systematic_debugging" / "SKILL.md"
        assert target.is_file()
        # The user's original SKILL.md is untouched.
        assert "Weak baseline" in fake_skill_path.read_text()

"""CLI integration test for --closed-loop-during-evolution on evolve_tool.

Patches the GEPA optimizer + heavy dependencies so the test doesn't
actually invoke any LM or hermes-agent subprocess. We only verify that:
  1. Without the flag, the metric closure's closed_loop_cache is None.
  2. With the flag + a fake hermes-tools dir, a ClosedLoopFeedbackCache
     is constructed and threaded into the metric.
  3. main() rejects --closed-loop-during-evolution without
     --closed-loop-hermes-repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner


_HERMES_TOOL_PY = '''\
"""Stub tool module."""

from tools.registry import registry

PATCH_SCHEMA = {
    "name": "patch",
    "description": "Apply targeted edits.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

registry.register(PATCH_SCHEMA, lambda **kw: None)
'''


@pytest.fixture
def fake_hermes_repo(tmp_path):
    repo = tmp_path / "hermes-agent"
    tools = repo / "tools"
    tools.mkdir(parents=True)
    (tools / "__init__.py").write_text("")
    (tools / "registry.py").write_text(
        "class _Registry:\n"
        "    def register(self, *a, **kw):\n"
        "        pass\n"
        "registry = _Registry()\n"
    )
    (tools / "file_tools.py").write_text(_HERMES_TOOL_PY)
    return repo


@pytest.fixture
def task_suite_file(tmp_path):
    p = tmp_path / "suite.jsonl"
    p.write_text(
        '{"task_id": "t1", "user_message": "do thing", '
        '"expected_tools": ["patch"], "forbidden_tools": []}\n'
    )
    return p


class TestMaybeBuildClosedLoopCache:
    def test_returns_none_when_suite_path_none(self):
        from evolution.tools.evolve_tool import _maybe_build_closed_loop_cache
        result = _maybe_build_closed_loop_cache(
            tool_name="patch",
            baseline_description="desc",
            suite_path=None,
            hermes_repo=None,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
        )
        assert result is None

    def test_raises_when_suite_path_set_but_no_repo(self):
        from evolution.tools.evolve_tool import _maybe_build_closed_loop_cache
        with pytest.raises(ValueError, match="closed_loop_hermes_repo"):
            _maybe_build_closed_loop_cache(
                tool_name="patch",
                baseline_description="desc",
                suite_path=Path("/fake.jsonl"),
                hermes_repo=None,
                saturation_threshold=0.95,
                min_iters=3,
                window_size=8,
            )

    def test_builds_cache_when_both_set(self, fake_hermes_repo, task_suite_file):
        from evolution.tools.evolve_tool import _maybe_build_closed_loop_cache
        from evolution.core.closed_loop_feedback import ClosedLoopFeedbackCache
        cache = _maybe_build_closed_loop_cache(
            tool_name="patch",
            baseline_description="Apply targeted edits.",
            suite_path=task_suite_file,
            hermes_repo=fake_hermes_repo,
            saturation_threshold=0.95,
            min_iters=3,
            window_size=8,
        )
        assert isinstance(cache, ClosedLoopFeedbackCache)


class TestMainCliValidation:
    def test_flag_without_repo_raises_usage_error(
        self, tmp_path, task_suite_file
    ):
        # We want main() to reject the bad combo before doing any work.
        # CliRunner catches the UsageError and surfaces exit code 2.
        from evolution.tools.evolve_tool import main
        runner = CliRunner()
        manifest_dir = tmp_path / "manifest"
        manifest_dir.mkdir()
        result = runner.invoke(main, [
            "--tool", "patch",
            "--manifest", str(manifest_dir),
            "--closed-loop-during-evolution", str(task_suite_file),
        ])
        assert result.exit_code != 0
        assert "closed-loop-hermes-repo" in result.output

    def test_flag_with_repo_does_not_immediately_error(
        self, tmp_path, fake_hermes_repo, task_suite_file
    ):
        # We don't want to actually run GEPA; we just verify Click parses
        # the flag combo cleanly. Patch evolve() to short-circuit.
        from evolution.tools.evolve_tool import main
        runner = CliRunner()
        with patch("evolution.tools.evolve_tool.evolve") as fake_evolve:
            fake_evolve.return_value = {"ok": True}
            result = runner.invoke(main, [
                "--tool", "patch",
                "--manifest", str(fake_hermes_repo / "tools"),
                "--closed-loop-during-evolution", str(task_suite_file),
                "--closed-loop-hermes-repo", str(fake_hermes_repo),
            ])
        assert result.exit_code == 0, result.output
        # evolve() received the flag through
        kwargs = fake_evolve.call_args.kwargs
        assert kwargs["closed_loop_suite_path"] == task_suite_file
        assert kwargs["closed_loop_hermes_repo"] == fake_hermes_repo


class TestEvolveSkillFlagRaises:
    def test_skill_side_flag_raises_with_clear_error(
        self, tmp_path, task_suite_file
    ):
        from evolution.skills.evolve_skill import main
        runner = CliRunner()
        result = runner.invoke(main, [
            "--skill", "some-skill",
            "--closed-loop-during-evolution", str(task_suite_file),
        ])
        assert result.exit_code != 0
        assert "SkillFileInstaller" in result.output

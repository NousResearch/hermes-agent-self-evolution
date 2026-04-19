"""Tests for the codex subprocess runner."""

import json
from pathlib import Path

import pytest

from evolution.core.codex_runner import (
    CodexRunTimeout,
    CodexRunner,
    build_codex_command,
)


class TestBuildCodexCommand:
    def test_build_codex_command_uses_stdin_prompt_mode(self, tmp_path):
        cmd = build_codex_command(workdir=tmp_path)

        assert cmd[:3] == ["codex", "exec", "-"]


class TestCodexRunner:
    def test_runner_raises_on_timeout(self, monkeypatch, tmp_path):
        runner = CodexRunner(workdir=tmp_path)

        def fake_run_subprocess(*args, **kwargs):
            raise TimeoutError()

        monkeypatch.setattr("evolution.core.codex_runner._run_subprocess", fake_run_subprocess)

        with pytest.raises(CodexRunTimeout):
            runner.run_json_task("Return JSON", timeout_seconds=1)

    def test_runner_parses_json_stdout(self, monkeypatch, tmp_path):
        runner = CodexRunner(workdir=tmp_path)

        def fake_run_subprocess(*args, **kwargs):
            return '{"ok": true, "value": 3}'

        monkeypatch.setattr("evolution.core.codex_runner._run_subprocess", fake_run_subprocess)

        result = runner.run_json_task("Return JSON", timeout_seconds=1)

        assert result == {"ok": True, "value": 3}

    def test_runner_rejects_non_json_stdout(self, monkeypatch, tmp_path):
        runner = CodexRunner(workdir=tmp_path)

        def fake_run_subprocess(*args, **kwargs):
            return 'not-json'

        monkeypatch.setattr("evolution.core.codex_runner._run_subprocess", fake_run_subprocess)

        with pytest.raises(ValueError):
            runner.run_json_task("Return JSON", timeout_seconds=1)

    def test_runner_passes_prompt_via_stdin(self, monkeypatch, tmp_path):
        runner = CodexRunner(workdir=tmp_path)
        captured = {}

        def fake_run_subprocess(command, workdir, timeout_seconds, prompt):
            captured["command"] = command
            captured["prompt"] = prompt
            return json.dumps({"ok": True})

        monkeypatch.setattr("evolution.core.codex_runner._run_subprocess", fake_run_subprocess)

        runner.run_json_task("Return JSON", timeout_seconds=1)

        assert captured["command"][:3] == ["codex", "exec", "-"]
        assert captured["prompt"] == "Return JSON"

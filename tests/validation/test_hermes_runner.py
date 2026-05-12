"""Tests for evolution.validation.hermes_runner.

Two layers exercised:
  - The parse layer (parse_session_result) via hand-crafted fixture
    session JSONs covering both tool_call shapes Hermes emits. This is
    where bugs hide; mocked subprocess gives no signal here.
  - The subprocess invocation layer (HermesAgentRunner.run) via
    subprocess.run mocked to assert env + cwd + args.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evolution.validation.agent_runner import TaskRunContext
from evolution.validation.hermes_runner import (
    HermesAgentRunner,
    parse_session_result,
)


def _write_session(path: Path, messages: list[dict], **extra) -> None:
    payload = {
        "session_id": "test",
        "model": "claude-opus-4-7",
        "messages": messages,
        **extra,
    }
    path.write_text(json.dumps(payload))


class TestParseSessionResult:
    def test_extracts_openai_nested_tool_calls(self, tmp_path):
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "do thing"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "search_files", "arguments": "{}"}}
            ]},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["search_files"]
        assert result.error is None
        assert result.model_name == "claude-opus-4-7"

    def test_extracts_flat_tool_calls(self, tmp_path):
        # Hermes also emits this shape for some model providers.
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "do thing"},
            {"role": "assistant", "tool_calls": [
                {"name": "patch", "arguments": "{}"}
            ]},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["patch"]

    def test_extracts_multi_tool_calls_in_order(self, tmp_path):
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "do things"},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "read_file"}},
                {"function": {"name": "patch"}},
                {"function": {"name": "search_files"}},
            ]},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["read_file", "patch", "search_files"]

    def test_extracts_across_multiple_assistant_turns(self, tmp_path):
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "first"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "patch"}}]},
            {"role": "tool", "content": "applied"},
            {"role": "assistant", "content": "done"},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["read_file", "patch"]

    def test_handles_assistant_message_with_text_only(self, tmp_path):
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "say hi"},
            {"role": "assistant", "content": "hi there"},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == []
        assert result.final_text_tail == "hi there"

    def test_final_text_tail_truncates_to_4096_chars(self, tmp_path):
        long_text = "x" * 5000
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "long"},
            {"role": "assistant", "content": long_text},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert len(result.final_text_tail) == 4096
        assert result.final_text_tail == long_text[-4096:]

    def test_picks_last_assistant_text_not_first(self, tmp_path):
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "first text"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "patch"}}]},
            {"role": "assistant", "content": "final text"},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.final_text_tail == "final text"

    def test_malformed_session_yields_error(self, tmp_path):
        p = tmp_path / "session.json"
        p.write_text("{not valid")
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.error is not None
        assert "could not parse" in result.error

    def test_call_missing_name_skipped(self, tmp_path):
        # Defensive: a tool_call dict without function.name or name shouldn't crash.
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "u"},
            {"role": "assistant", "tool_calls": [
                {"id": "abc"},                                      # no name field
                {"function": {"name": "patch"}},                    # has name
                {"function": {}},                                   # empty function
                {"name": ""},                                       # empty name
            ]},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["patch"]


class TestHermesAgentRunnerSubprocess:
    """The subprocess invocation layer: env + cwd + args plumbing."""

    @pytest.fixture
    def fixture_dir(self, tmp_path):
        d = tmp_path / "fixture"
        d.mkdir()
        return d

    def test_subprocess_called_with_hermes_z_and_message(self, fixture_dir, tmp_path):
        runner = HermesAgentRunner(user_config_path=tmp_path / "nonexistent_config")
        captured = {}

        def _fake_run(*args, **kwargs):
            captured["args"] = args[0] if args else kwargs.get("args")
            captured["env"] = kwargs.get("env")
            captured["cwd"] = kwargs.get("cwd")
            # Drop a minimal session JSON so the parse layer succeeds.
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            (sandbox / "sessions").mkdir(exist_ok=True)
            _write_session(
                sandbox / "sessions" / "session_test.json",
                [{"role": "assistant", "tool_calls": [{"function": {"name": "patch"}}]}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            result = runner.run(TaskRunContext(
                user_message="please patch a file",
                fixture_dir=fixture_dir,
            ))

        assert captured["args"][:2] == ["hermes", "-z"]
        assert captured["args"][2] == "please patch a file"
        assert captured["cwd"] == str(fixture_dir)
        # Sandbox env vars: HERMES_HOME and HOME point at the SAME tmp dir.
        assert captured["env"]["HERMES_HOME"] == captured["env"]["HOME"]
        assert captured["env"]["HERMES_HOME"].startswith("/")  # absolute tmp dir
        # Parse layer worked: we got the canned tool call back.
        assert result.tool_calls_seq == ["patch"]
        assert result.error is None

    def test_timeout_returns_error_result(self, fixture_dir, tmp_path):
        runner = HermesAgentRunner(timeout_seconds=1, user_config_path=tmp_path / "x")
        import subprocess as _subprocess
        with patch(
            "evolution.validation.hermes_runner.subprocess.run",
            side_effect=_subprocess.TimeoutExpired(cmd="hermes", timeout=1),
        ):
            result = runner.run(TaskRunContext(
                user_message="hang",
                fixture_dir=fixture_dir,
            ))
        assert result.error is not None
        assert "timed out" in result.error
        assert result.tool_calls_seq == []

    def test_hermes_not_installed_returns_error_result(self, fixture_dir, tmp_path):
        runner = HermesAgentRunner(
            hermes_command="hermes-does-not-exist",
            user_config_path=tmp_path / "x",
        )
        with patch(
            "evolution.validation.hermes_runner.subprocess.run",
            side_effect=FileNotFoundError("hermes-does-not-exist"),
        ):
            result = runner.run(TaskRunContext(
                user_message="run",
                fixture_dir=fixture_dir,
            ))
        assert result.error is not None
        assert "not found" in result.error

    def test_no_session_written_returns_error_result(self, fixture_dir, tmp_path):
        runner = HermesAgentRunner(user_config_path=tmp_path / "x")

        def _fake_run(*args, **kwargs):
            # Don't drop a session JSON.
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            result = runner.run(TaskRunContext(
                user_message="run",
                fixture_dir=fixture_dir,
            ))
        assert result.error is not None
        assert "no session JSON" in result.error

    def test_user_config_copied_into_sandbox_when_exists(self, fixture_dir, tmp_path):
        user_config = tmp_path / "user_config.yaml"
        user_config.write_text("api_key: hunter2\n")
        runner = HermesAgentRunner(user_config_path=user_config)

        sandbox_seen = {}

        def _fake_run(*args, **kwargs):
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            sandbox_seen["config_present"] = (sandbox / "config.yaml").exists()
            sandbox_seen["config_text"] = (sandbox / "config.yaml").read_text() if sandbox_seen["config_present"] else None
            (sandbox / "sessions").mkdir(exist_ok=True)
            _write_session(
                sandbox / "sessions" / "session_test.json",
                [{"role": "assistant", "content": "ok"}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            runner.run(TaskRunContext(
                user_message="run",
                fixture_dir=fixture_dir,
            ))

        assert sandbox_seen["config_present"] is True
        assert "hunter2" in sandbox_seen["config_text"]

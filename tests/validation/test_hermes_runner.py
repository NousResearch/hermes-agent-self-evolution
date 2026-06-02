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
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from evolution.validation.agent_runner import TaskRunContext
from evolution.validation.hermes_runner import (
    HermesAgentRunner,
    _strip_litellm_provider_prefix,
    parse_session_from_db,
    parse_session_result,
)


def _make_state_db(path: Path, *, session_id: str, model: str, messages: list[dict],
                   started_at: float = 1.0) -> None:
    """Create a minimal hermes-shaped state.db with one session + messages.

    Each ``messages`` entry: ``{"role", "content"?, "tool_calls"?}`` where
    ``tool_calls`` is a Python list serialized to the ``tool_calls`` TEXT
    column (the OpenAI-nested shape hermes stores).
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, model TEXT, started_at REAL);
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT, content TEXT, tool_calls TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO sessions (id, model, started_at) VALUES (?, ?, ?)",
        (session_id, model, started_at),
    )
    for m in messages:
        tc = m.get("tool_calls")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls) "
            "VALUES (?, ?, ?, ?)",
            (session_id, m["role"], m.get("content"),
             json.dumps(tc) if tc is not None else None),
        )
    conn.commit()
    conn.close()


class TestStripLitellmProviderPrefix:
    """The hermes -m flag interprets '<word>/<model>' as openrouter-style
    routing. Direct-provider users naturally pass litellm-formatted names
    like 'openai/gpt-4o-mini' from elsewhere in the framework; this helper
    normalizes them back to bare model names that hermes -m accepts."""

    def test_strips_openai_prefix(self):
        assert _strip_litellm_provider_prefix("openai/gpt-4o-mini") == "gpt-4o-mini"

    def test_strips_anthropic_prefix(self):
        assert _strip_litellm_provider_prefix("anthropic/claude-opus-4-7") == "claude-opus-4-7"

    def test_strips_azure_prefix(self):
        assert _strip_litellm_provider_prefix("azure/gpt-4") == "gpt-4"

    def test_strips_gemini_prefix(self):
        assert _strip_litellm_provider_prefix("gemini/gemini-1.5-pro") == "gemini-1.5-pro"

    def test_leaves_bare_model_unchanged(self):
        assert _strip_litellm_provider_prefix("gpt-4o-mini") == "gpt-4o-mini"

    def test_leaves_unknown_prefix_unchanged(self):
        # Unknown providers pass through so openrouter-style routing through
        # an unrecognized vendor still works.
        assert _strip_litellm_provider_prefix("custom-vendor/model-x") == "custom-vendor/model-x"

    def test_only_strips_first_prefix(self):
        # Defensive: nested prefixes (e.g. openrouter passthrough) shouldn't be
        # double-stripped — keep the second segment intact.
        assert _strip_litellm_provider_prefix("openai/foo/bar") == "foo/bar"


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


class TestParseToolCallArgs:
    def test_captures_tool_call_args(self, tmp_path):
        """Sessions with tool calls must surface name AND parsed args."""
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "user", "content": "save a fact"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {
                    "name": "memory",
                    "arguments": json.dumps({
                        "action": "save",
                        "content": "user prefers terse responses",
                    }),
                }}
            ]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "Saved."},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["memory"]
        assert len(result.tool_calls_with_args) == 1
        call = result.tool_calls_with_args[0]
        assert call["name"] == "memory"
        assert call["arguments"]["action"] == "save"
        assert call["arguments"]["content"] == "user prefers terse responses"

    def test_handles_malformed_args(self, tmp_path):
        """Malformed tool-call arguments JSON must not crash — fall back to {}."""
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "memory", "arguments": "{not-json"}}
            ]},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_seq == ["memory"]
        assert result.tool_calls_with_args == [{"name": "memory", "arguments": {}}]

    def test_handles_flat_dict_args(self, tmp_path):
        """Flat tool_call shape with an already-parsed dict argument."""
        p = tmp_path / "session.json"
        _write_session(p, [
            {"role": "assistant", "tool_calls": [
                {"name": "memory", "arguments": {"action": "delete", "key": "x"}}
            ]},
        ])
        result = parse_session_result(p, duration_seconds=1.0)
        assert result.tool_calls_with_args == [
            {"name": "memory", "arguments": {"action": "delete", "key": "x"}}
        ]


class TestParseSessionFromDb:
    """The state.db parse layer — modern hermes persists sessions to SQLite."""

    def test_extracts_tool_calls_and_args(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db(db, session_id="s1", model="gpt-5.4-mini", messages=[
            {"role": "user", "content": "remember I use uv"},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {
                    "name": "memory",
                    "arguments": json.dumps({"action": "add", "content": "uses uv"}),
                }}
            ]},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "Saved."},
        ])
        result = parse_session_from_db(db, duration_seconds=2.0)
        assert result.error is None
        assert result.model_name == "gpt-5.4-mini"
        assert result.tool_calls_seq == ["memory"]
        assert result.tool_calls_with_args == [
            {"name": "memory", "arguments": {"action": "add", "content": "uses uv"}}
        ]
        assert result.final_text_tail == "Saved."

    def test_no_sessions_is_error(self, tmp_path):
        db = tmp_path / "state.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TABLE sessions (id TEXT, model TEXT, started_at REAL);"
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
            "role TEXT, content TEXT, tool_calls TEXT);"
        )
        conn.commit()
        conn.close()
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.error is not None
        assert "no sessions" in result.error

    def test_picks_most_recent_session(self, tmp_path):
        db = tmp_path / "state.db"
        _make_state_db(db, session_id="old", model="m", started_at=1.0, messages=[
            {"role": "assistant", "tool_calls": [{"function": {"name": "patch"}}]},
        ])
        # Add a newer session with a different tool call.
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO sessions (id, model, started_at) VALUES (?,?,?)",
                     ("new", "m", 2.0))
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls) VALUES (?,?,?,?)",
            ("new", "assistant", None,
             json.dumps([{"function": {"name": "write_file"}}])),
        )
        conn.commit()
        conn.close()
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.tool_calls_seq == ["write_file"]

    def test_malformed_tool_calls_column_abstains(self, tmp_path):
        """A corrupt tool_calls column must abstain (error set), not read as
        'agent invoked no tools' (which would score a hard behavioral fail)."""
        db = tmp_path / "state.db"
        _make_state_db(db, session_id="s1", model="m",
                       messages=[{"role": "user", "content": "hi"}])
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls) VALUES (?,?,?,?)",
            ("s1", "assistant", "", "{not-valid-json"),
        )
        conn.commit()
        conn.close()
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.error is not None
        assert "malformed tool_calls" in result.error
        assert result.tool_calls_seq == []

    def test_corrupt_db_file_errors(self, tmp_path):
        bad = tmp_path / "state.db"
        bad.write_bytes(b"this is not a sqlite database at all")
        result = parse_session_from_db(bad, duration_seconds=1.0)
        assert result.error is not None
        assert "could not" in result.error  # open or read, depending on sqlite

    def test_missing_messages_table_errors(self, tmp_path):
        db = tmp_path / "state.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TABLE sessions (id TEXT, model TEXT, started_at REAL);"
            "INSERT INTO sessions VALUES ('s1', 'm', 1.0);"
        )
        conn.commit()
        conn.close()
        result = parse_session_from_db(db, duration_seconds=1.0)
        assert result.error is not None
        assert "could not read" in result.error


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
            # Drop a minimal state.db so the parse layer succeeds.
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            _make_state_db(
                sandbox / "state.db",
                session_id="s1", model="test-model",
                messages=[{"role": "assistant", "tool_calls": [
                    {"function": {"name": "patch"}}]}],
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
            # Don't write a state.db.
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            result = runner.run(TaskRunContext(
                user_message="run",
                fixture_dir=fixture_dir,
            ))
        assert result.error is not None
        assert "state.db absent" in result.error

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

    def test_skills_src_copied_into_sandbox_when_present(self, fixture_dir, tmp_path):
        """When TaskRunContext.skills_src points at a directory, the runner
        copies its contents into <sandbox>/skills/ so ``hermes -z``
        discovers any candidate skills installed there.
        """
        skills_src = tmp_path / "candidate_skills"
        (skills_src / "systematic_debugging").mkdir(parents=True)
        (skills_src / "systematic_debugging" / "SKILL.md").write_text(
            "---\nname: systematic_debugging\n---\n\nevolved body\n"
        )
        runner = HermesAgentRunner(user_config_path=tmp_path / "nonexistent")

        sandbox_skills_state: dict = {}

        def _fake_run(*args, **kwargs):
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            skill_path = sandbox / "skills" / "systematic_debugging" / "SKILL.md"
            sandbox_skills_state["present"] = skill_path.is_file()
            if sandbox_skills_state["present"]:
                sandbox_skills_state["text"] = skill_path.read_text()
            (sandbox / "sessions").mkdir(exist_ok=True)
            _write_session(
                sandbox / "sessions" / "session_test.json",
                [{"role": "assistant", "content": "ok"}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            runner.run(TaskRunContext(
                user_message="debug it",
                fixture_dir=fixture_dir,
                skills_src=skills_src,
            ))

        assert sandbox_skills_state["present"] is True
        assert "evolved body" in sandbox_skills_state["text"]

    def test_model_override_passes_minus_m_flag(self, fixture_dir, tmp_path):
        """When model is set, hermes is invoked with `hermes -m MODEL -z ...`."""
        runner = HermesAgentRunner(
            user_config_path=tmp_path / "nonexistent",
            model="gpt-4o-mini",
        )
        captured: dict = {}

        def _fake_run(*args, **kwargs):
            captured["args"] = args[0] if args else kwargs.get("args")
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            (sandbox / "sessions").mkdir(exist_ok=True)
            _write_session(
                sandbox / "sessions" / "session_test.json",
                [{"role": "assistant", "content": "ok"}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            runner.run(TaskRunContext(
                user_message="debug",
                fixture_dir=fixture_dir,
            ))

        # -m must appear before -z so hermes parses it as a global flag,
        # not as part of the -z message.
        assert captured["args"][:4] == ["hermes", "-m", "gpt-4o-mini", "-z"]
        assert captured["args"][4] == "debug"

    def test_model_none_omits_minus_m_flag(self, fixture_dir, tmp_path):
        """No model override → original argv shape (no -m), preserves the
        existing behavior bit-for-bit for callers that don't opt in."""
        runner = HermesAgentRunner(user_config_path=tmp_path / "nonexistent")
        captured: dict = {}

        def _fake_run(*args, **kwargs):
            captured["args"] = args[0] if args else kwargs.get("args")
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            (sandbox / "sessions").mkdir(exist_ok=True)
            _write_session(
                sandbox / "sessions" / "session_test.json",
                [{"role": "assistant", "content": "ok"}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            runner.run(TaskRunContext(
                user_message="hello",
                fixture_dir=fixture_dir,
            ))

        assert "-m" not in captured["args"]
        assert captured["args"] == ["hermes", "-z", "hello"]

    def test_litellm_prefix_stripped_before_minus_m(self, fixture_dir, tmp_path):
        """Regression: hermes -m treats '<provider>/<model>' as openrouter-style
        routing (silently switches base_url to openrouter.ai and breaks auth
        for direct-provider configs). Users naturally pass litellm-formatted
        names like 'openai/gpt-4o-mini' from elsewhere in the framework, so
        the runner must strip known litellm provider prefixes before -m to
        avoid a silent 0-turn 'agent never ran' failure that misreports as
        'validator too weak' at the saturation pre-flight."""
        runner = HermesAgentRunner(
            user_config_path=tmp_path / "nonexistent",
            model="openai/gpt-4o-mini",
        )
        captured: dict = {}

        def _fake_run(*args, **kwargs):
            captured["args"] = args[0] if args else kwargs.get("args")
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            (sandbox / "sessions").mkdir(exist_ok=True)
            _write_session(
                sandbox / "sessions" / "session_test.json",
                [{"role": "assistant", "content": "ok"}],
            )
            return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("evolution.validation.hermes_runner.subprocess.run", side_effect=_fake_run):
            runner.run(TaskRunContext(
                user_message="debug",
                fixture_dir=fixture_dir,
            ))

        assert captured["args"][:4] == ["hermes", "-m", "gpt-4o-mini", "-z"]
        assert "openai/" not in captured["args"][2]

    def test_skills_src_none_means_no_skills_dir_created(self, fixture_dir, tmp_path):
        """Tool-side runs (no skills_src) must not create an empty skills/
        directory in the sandbox — keeps the legacy code path bit-for-bit."""
        runner = HermesAgentRunner(user_config_path=tmp_path / "nonexistent")
        sandbox_seen: dict = {}

        def _fake_run(*args, **kwargs):
            sandbox = Path(kwargs["env"]["HERMES_HOME"])
            sandbox_seen["has_skills_dir"] = (sandbox / "skills").exists()
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

        assert sandbox_seen["has_skills_dir"] is False

"""ClaudeCodeAgentRunner stream-json parsing + containment (no live `claude` calls)."""
from pathlib import Path

import pytest

from evolution.validation.claude_runner import (
    ClaudeCodeAgentRunner,
    SandboxUnavailableError,
    _macos_write_sandbox_profile,
    _parse_stream_json,
    _price_from_tokens,
)

STREAM = "\n".join([
    '{"type":"system","subtype":"init","model":"claude-sonnet-4-6","tools":["Bash"]}',
    '{"type":"assistant","message":{"content":[{"type":"text","text":"thinking"},'
    '{"type":"tool_use","name":"Bash","input":{"command":"./bin/check"}}]}}',
    '{"type":"result","subtype":"success","is_error":false,"result":"done",'
    '"total_cost_usd":0.012,"usage":{"input_tokens":10,"output_tokens":20,'
    '"cache_read_input_tokens":5,"cache_creation_input_tokens":3}}',
])


def test_parse_extracts_tool_calls_cost_tokens():
    r = _parse_stream_json(STREAM, duration_seconds=2.0)
    assert r.error is None
    assert r.tool_calls_seq == ["Bash"]
    assert r.tool_calls_with_args == [{"name": "Bash", "arguments": {"command": "./bin/check"}}]
    assert r.agent_cost_usd == 0.012 and r.agent_cost_source == "actual"
    assert r.agent_tokens["input_tokens"] == 10 and r.agent_tokens["output_tokens"] == 20
    assert r.model_name == "claude-sonnet-4-6"
    assert r.final_text_tail == "done"


def test_parse_flags_error_result():
    stream = ('{"type":"result","subtype":"success","is_error":true,'
              '"result":"Not logged in","total_cost_usd":0,"usage":{}}')
    r = _parse_stream_json(stream, duration_seconds=1.0)
    assert r.error is not None and "Not logged in" in r.error


def test_parse_no_result_event_is_error():
    r = _parse_stream_json('{"type":"system","subtype":"init","model":"m"}', duration_seconds=1.0)
    assert r.error is not None


def test_parse_zero_cost_unknown_model_is_uncaptured():
    # No init event -> model_name None -> litellm fallback returns None -> uncaptured.
    stream = ('{"type":"result","subtype":"success","is_error":false,"result":"ok",'
              '"total_cost_usd":0,"usage":{"input_tokens":1,"output_tokens":1}}')
    r = _parse_stream_json(stream, duration_seconds=1.0)
    assert r.agent_cost_usd is None and r.agent_cost_source == "uncaptured"


def test_parse_degraded_empty_result_is_error():
    # is_error false but no tools, no text, zero tokens -> degraded -> abstain.
    stream = ('{"type":"result","subtype":"success","is_error":false,"result":"",'
              '"total_cost_usd":0,"usage":{"input_tokens":0,"output_tokens":0}}')
    r = _parse_stream_json(stream, duration_seconds=1.0)
    assert r.error is not None and "degraded" in r.error


def test_price_from_tokens_none_when_unpriceable():
    assert _price_from_tokens(None, {"input_tokens": 5, "output_tokens": 5}) is None
    assert _price_from_tokens("claude-x", {"input_tokens": 0, "output_tokens": 0}) is None


def test_sandbox_profile_denies_writes_outside_roots(tmp_path):
    prof = _macos_write_sandbox_profile([tmp_path])
    assert "(deny file-write*)" in prof
    assert f'(subpath "{tmp_path}")' in prof
    assert '(subpath "/private/var/folders")' in prof


def test_refuses_to_run_unsandboxed(monkeypatch):
    # Force the "no OS sandbox" branch and confirm it refuses rather than running.
    monkeypatch.setattr("evolution.validation.claude_runner.sys.platform", "linux")
    runner = ClaudeCodeAgentRunner(require_sandbox=True)
    with pytest.raises(SandboxUnavailableError):
        runner._wrap_in_sandbox(["claude", "-p", "hi"], write_roots=[Path("/x")])


def test_unsandboxed_allowed_when_waived(monkeypatch):
    monkeypatch.setattr("evolution.validation.claude_runner.sys.platform", "linux")
    runner = ClaudeCodeAgentRunner(require_sandbox=False)
    argv = runner._wrap_in_sandbox(["claude", "-p", "hi"], write_roots=[Path("/x")])
    assert argv == ["claude", "-p", "hi"]

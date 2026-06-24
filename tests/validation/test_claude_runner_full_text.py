"""The runner captures the agent's untruncated final text (``full_text``), not
just the 4096-char diagnostic tail — needed by any consumer that reads the
agent's actual output (e.g. a code review longer than the tail)."""

from evolution.validation.claude_runner import _parse_stream_json


def test_full_text_is_untruncated():
    big = "X" * 9000
    stream = '{"type":"result","result":"%s"}\n' % big
    res = _parse_stream_json(stream, duration_seconds=0.0)
    assert res.full_text == big
    assert res.final_text_tail == big[-4096:]

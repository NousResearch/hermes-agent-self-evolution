"""Tests for JSONL attempt-trace loading and secret scanning."""

import json

from evolution.traces.jsonl import failed_traces_to_eval_examples, load_trace_jsonl, scan_traces_for_secrets


def test_load_trace_jsonl_normalizes_source_status_and_metadata(tmp_path):
    trace_file = tmp_path / "traces.jsonl"
    trace_file.write_text(
        json.dumps(
            {
                "task_input": "Review PR #12",
                "observed_output": "Looks fine",
                "expected_behavior": "Flag missing auth check",
                "failure_reason": "missed auth bug",
                "status": "FAILURE",
                "metadata": {"session_id": "abc"},
            }
        )
        + "\n"
    )

    traces = load_trace_jsonl(trace_file, default_source="hermes-session")

    assert len(traces) == 1
    assert traces[0]["source"] == "hermes-session"
    assert traces[0]["status"] == "failure"
    assert traces[0]["metadata"]["session_id"] == "abc"
    assert traces[0]["metadata"]["line_number"] == 1


def test_failed_traces_to_eval_examples_skips_successes_and_preserves_trace_id():
    traces = [
        {
            "id": "trace_fail",
            "source": "hermes-session",
            "task_input": "Fix command",
            "expected_behavior": "Use sys.executable",
            "observed_output": "Used python",
            "status": "failure",
            "failure_reason": "wrong interpreter",
            "metadata": {},
        },
        {
            "id": "trace_success",
            "source": "hermes-session",
            "task_input": "Already good",
            "expected_behavior": "No change",
            "observed_output": "No change",
            "status": "success",
            "failure_reason": None,
            "metadata": {},
        },
    ]

    examples = failed_traces_to_eval_examples(traces)

    assert len(examples) == 1
    assert examples[0]["task_input"] == "Fix command"
    assert examples[0]["expected_behavior"] == "Use sys.executable"
    assert examples[0]["metadata"]["trace_id"] == "trace_fail"
    assert examples[0]["metadata"]["failure_reason"] == "wrong interpreter"


def test_scan_traces_for_secrets_reports_fields_without_values():
    traces = [
        {
            "task_input": "safe",
            "observed_output": "OPENAI_API_KEY=sk-" + "a" * 30,
            "expected_behavior": "redact secrets",
            "failure_reason": "leaked credential",
            "status": "failure",
        }
    ]

    report = scan_traces_for_secrets(traces)

    assert report["status"] == "failed"
    assert report["matches"] == [{"trace_index": 0, "field": "observed_output"}]
    assert "sk-" not in str(report)

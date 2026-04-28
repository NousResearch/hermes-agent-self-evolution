"""JSONL loader for Hermes attempt traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolution.datasets.redaction import SECRET_PATTERN

TRACE_SCAN_FIELDS = ("task_input", "expected_behavior", "observed_output", "failure_reason")
SUCCESS_STATUSES = {"success", "passed", "pass", "ok"}


def load_trace_jsonl(path: str | Path, default_source: str | None = None) -> list[dict[str, Any]]:
    """Load attempt traces from JSONL and normalize required fields."""
    traces: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Trace line {line_number} must be a JSON object")
        task_input = str(record.get("task_input", "")).strip()
        if not task_input:
            raise ValueError(f"Trace line {line_number} missing task_input")
        metadata = dict(record.get("metadata") or {})
        metadata.setdefault("line_number", line_number)
        traces.append(
            {
                "source": str(record.get("source") or default_source or "unknown"),
                "task_input": task_input,
                "observed_output": _optional_text(record.get("observed_output")),
                "expected_behavior": _optional_text(record.get("expected_behavior")),
                "status": _normalize_status(record.get("status")),
                "failure_reason": _optional_text(record.get("failure_reason")),
                "source_ref_hash": _optional_text(record.get("source_ref_hash")),
                "metadata": metadata,
            }
        )
    return traces


def failed_traces_to_eval_examples(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert failed traces into eval-example dictionaries."""
    examples: list[dict[str, Any]] = []
    for trace in traces:
        if str(trace.get("status", "failure")).lower() in SUCCESS_STATUSES:
            continue
        expected_behavior = _expected_behavior_for_trace(trace)
        metadata = dict(trace.get("metadata") or {})
        if trace.get("id"):
            metadata["trace_id"] = trace["id"]
        metadata.update(
            {
                "observed_output": trace.get("observed_output"),
                "failure_reason": trace.get("failure_reason"),
                "trace_status": trace.get("status"),
            }
        )
        examples.append(
            {
                "task_input": trace["task_input"],
                "expected_behavior": expected_behavior,
                "source": f"trace:{trace.get('source', 'unknown')}",
                "source_ref_hash": trace.get("source_ref_hash"),
                "metadata": metadata,
            }
        )
    return examples


def scan_traces_for_secrets(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Scan traces for secret-like values without returning the values."""
    matches: list[dict[str, Any]] = []
    for index, trace in enumerate(traces):
        for field in TRACE_SCAN_FIELDS:
            text = str(trace.get(field) or "")
            if SECRET_PATTERN.search(text):
                matches.append({"trace_index": index, "field": field})
    return {"status": "failed" if matches else "passed", "matches": matches}


def _normalize_status(value: Any) -> str:
    status = str(value or "failure").strip().lower()
    if status in SUCCESS_STATUSES:
        return "success"
    return "failure"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _expected_behavior_for_trace(trace: dict[str, Any]) -> str:
    expected = _optional_text(trace.get("expected_behavior"))
    if expected:
        return expected
    failure_reason = _optional_text(trace.get("failure_reason"))
    if failure_reason:
        return f"Avoid the recorded failure: {failure_reason}"
    return "Complete the task successfully and avoid the recorded failure."

"""Regression eval assets for hermes-content production-quality failures."""

from pathlib import Path

from evolution.traces.jsonl import load_trace_jsonl, scan_traces_for_secrets

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FAILURE_CLASSES = {
    "proof_layer_failure",
    "runtime_boundary_failure",
    "visual_stack_claim_failure",
    "human_critique_failure",
    "path_confusion_failure",
}


def test_content_production_failure_traces_are_valid_secret_clean_and_complete():
    trace_path = ROOT / "examples" / "content-production" / "failures.jsonl"

    traces = load_trace_jsonl(trace_path, default_source="content-production")

    assert len(traces) >= len(REQUIRED_FAILURE_CLASSES)
    assert {trace["status"] for trace in traces} == {"failure"}
    assert scan_traces_for_secrets(traces)["status"] == "passed"

    classes = {trace.get("metadata", {}).get("failure_class") for trace in traces}
    assert REQUIRED_FAILURE_CLASSES <= classes

    for trace in traces:
        metadata = trace.get("metadata", {})
        assert metadata.get("owner_profile") == "hermes-content"
        assert metadata.get("skill")
        assert "HOLD" in trace["expected_behavior"] or "hold" in trace["expected_behavior"].lower()
        assert "proof" in trace["expected_behavior"].lower() or "runtime" in trace["expected_behavior"].lower()

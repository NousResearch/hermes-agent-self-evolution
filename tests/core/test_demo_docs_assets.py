"""Tests for demo docs/assets shipped with the product loop."""

from pathlib import Path

from evolution.traces.jsonl import load_trace_jsonl, scan_traces_for_secrets

ROOT = Path(__file__).resolve().parents[2]


def test_demo_trace_jsonl_is_valid_and_secret_clean():
    trace_path = ROOT / "examples" / "demo" / "failures.jsonl"

    traces = load_trace_jsonl(trace_path, default_source="demo")

    assert len(traces) >= 3
    assert {trace["status"] for trace in traces} == {"failure"}
    assert scan_traces_for_secrets(traces)["status"] == "passed"


def test_product_workflow_doc_mentions_core_safe_commands():
    doc = (ROOT / "docs" / "V1_PRODUCT_WORKFLOW.md").read_text()

    for expected in [
        "hermes-evolve loop once",
        "--strategy dspy-gepa",
        "--scoring-strategy model-rubric",
        "hermes-evolve run apply",
        "hermes-evolve run pr-draft",
        "No auto-push. No auto-merge.",
    ]:
        assert expected in doc

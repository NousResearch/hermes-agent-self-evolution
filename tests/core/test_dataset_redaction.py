"""Tests for dataset redaction scanning."""

from evolution.datasets.redaction import scan_examples_for_secrets


def test_scan_examples_for_secrets_passes_clean_examples():
    report = scan_examples_for_secrets([
        {"task_input": "Review this function", "expected_behavior": "Find bugs"},
    ])

    assert report["status"] == "passed"
    assert report["matches"] == []


def test_scan_examples_for_secrets_flags_api_keys():
    report = scan_examples_for_secrets([
        {"task_input": "Use OPENAI_API_KEY=sk-abc12345678901234567890", "expected_behavior": "Do not leak"},
    ])

    assert report["status"] == "failed"
    assert report["matches"][0]["field"] == "task_input"
    assert report["matches"][0]["example_index"] == 0

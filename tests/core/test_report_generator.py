"""Tests for HTML report generator."""

from pathlib import Path
from evolution.core.report_generator import generate_html_report, _compute_diff


SAMPLE_METRICS = {
    "baseline_score": 0.4,
    "evolved_score": 0.6,
    "improvement": 0.2,
    "baseline_size": 1000,
    "evolved_size": 1050,
    "iterations": 10,
    "optimizer_model": "gpt-4o-mini",
    "eval_model": "gpt-4o-mini",
    "train_examples": 10,
    "holdout_examples": 5,
    "elapsed_seconds": 120,
    "constraints_passed": True,
}

SAMPLE_CONSTRAINTS = [
    {"constraint_name": "size_limit", "passed": True, "message": "Size OK: 1050/15000 chars"},
    {"constraint_name": "non_empty", "passed": True, "message": "Artifact is non-empty"},
]


def test_generate_html_report_returns_string():
    html = generate_html_report(
        skill_name="test-skill",
        metrics=SAMPLE_METRICS,
        constraints=SAMPLE_CONSTRAINTS,
        baseline_text="# Baseline\nSome content",
        evolved_text="# Evolved\nImproved content",
    )
    assert isinstance(html, str)
    assert "Evolution Report" in html
    assert "test-skill" in html
    assert "0.400" in html
    assert "0.600" in html


def test_generate_html_report_writes_file(tmp_path):
    out = tmp_path / "report.html"
    generate_html_report(
        skill_name="test-skill",
        metrics=SAMPLE_METRICS,
        constraints=SAMPLE_CONSTRAINTS,
        baseline_text="# Baseline\nOld content",
        evolved_text="# Evolved\nNew content",
        output_path=out,
    )
    assert out.exists()
    assert "Evolution Report" in out.read_text()


def test_generate_html_report_includes_diff():
    baseline = "# Skill\n\n## Procedure\n1. Do step one\n2. Do step two"
    evolved = "# Skill\n\n## Procedure\n1. Do step one improved\n2. Do step two\n3. New step"
    html = generate_html_report(
        skill_name="test-skill",
        metrics=SAMPLE_METRICS,
        constraints=SAMPLE_CONSTRAINTS,
        baseline_text=baseline,
        evolved_text=evolved,
    )
    assert "Do step one improved" in html
    assert "New step" in html


def test_generate_html_report_includes_constraints():
    constraints = [
        {"constraint_name": "size_limit", "passed": False, "message": "Size exceeded"},
    ]
    html = generate_html_report(
        skill_name="test-skill",
        metrics=SAMPLE_METRICS,
        constraints=constraints,
        baseline_text="base",
        evolved_text="evolved",
    )
    assert "❌" in html
    assert "Size exceeded" in html


def test_compute_diff_shows_additions():
    diff = _compute_diff("line1\nline2\n", "line1\nline2\nline3\n")
    assert "line3" in diff


def test_compute_diff_shows_deletions():
    diff = _compute_diff("line1\nline2\nline3\n", "line1\nline3\n")
    assert "diff-del" in diff

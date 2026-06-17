"""Tests for the ProposalWriter / ProposalRecord pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution.core.proposals import (
    ConstraintRecord,
    ProposalRecord,
    ProposalWriter,
    build_proposal_record,
    PROPOSAL_STATUS_PENDING,
)
from evolution.core.regression_guard import AutoMergeGate, GateDecision


# ───────────────────────── test doubles ─────────────────────────
class _FakeConstraint:
    """Minimal duck-type for ConstraintResult — matches .constraint_name/.passed/.message."""
    def __init__(self, name: str, passed: bool, message: str):
        self.constraint_name = name
        self.passed = passed
        self.message = message


BASELINE_SKILL = """---
name: sample-skill
description: A sample skill for testing
---

# Sample Skill

Original body text line 1.
Original body text line 2.
"""

EVOLVED_SKILL = """---
name: sample-skill
description: A sample skill for testing
---

# Sample Skill

Evolved body line 1 — much clearer.
Original body text line 2.
Added new guidance line 3.
"""


# ───────────────────────── ConstraintRecord ─────────────────────────
def test_constraint_record_from_result_extracts_fields():
    fake = _FakeConstraint("size_growth", True, "Body grew 12%")
    rec = ConstraintRecord.from_result(fake)
    assert rec.name == "size_growth"
    assert rec.passed is True
    assert rec.message == "Body grew 12%"


def test_constraint_record_from_result_handles_missing_attrs():
    class _Empty: pass
    rec = ConstraintRecord.from_result(_Empty())
    assert rec.name == "unknown"
    assert rec.passed is False
    assert rec.message == ""


# ───────────────────────── build_proposal_record ─────────────────────────
def test_build_proposal_record_computes_improvement_from_scores():
    gate = AutoMergeGate(min_improvement=0.02)
    decision = gate.evaluate(baseline=0.50, evolved=0.55, constraints_passed=True)
    rec = build_proposal_record(
        skill_name="sample-skill",
        baseline_text=BASELINE_SKILL,
        evolved_text=EVOLVED_SKILL,
        baseline_score=0.50,
        evolved_score=0.55,
        decision=decision,
        constraint_results=[_FakeConstraint("c1", True, "ok")],
        mode="propose",
    )
    assert rec.improvement == pytest.approx(0.05, abs=1e-9)
    assert rec.auto_merge is True
    assert rec.regression is False
    assert rec.mode == "propose"
    assert rec.baseline_size == len(BASELINE_SKILL)
    assert rec.evolved_size == len(EVOLVED_SKILL)
    assert rec.size_delta == len(EVOLVED_SKILL) - len(BASELINE_SKILL)


def test_build_proposal_record_defaults_timestamp():
    decision = GateDecision(False, "test", -0.01)
    rec = build_proposal_record(
        skill_name="x",
        baseline_text="a",
        evolved_text="b",
        baseline_score=0.5,
        evolved_score=0.49,
        decision=decision,
        constraint_results=[],
        mode="propose",
    )
    # YYYYMMDD_HHMMSS is 15 chars
    assert len(rec.timestamp) == 15
    assert "_" in rec.timestamp


# ───────────────────────── ProposalWriter ─────────────────────────
def test_writer_creates_all_required_artifacts(tmp_path: Path):
    writer = ProposalWriter(tmp_path / "proposals")
    decision = GateDecision(False, "below threshold", 0.01)
    record = build_proposal_record(
        skill_name="my-skill",
        baseline_text=BASELINE_SKILL,
        evolved_text=EVOLVED_SKILL,
        baseline_score=0.70,
        evolved_score=0.71,
        decision=decision,
        constraint_results=[
            _FakeConstraint("skill_structure", True, "Frontmatter intact"),
            _FakeConstraint("size_growth", False, "Body grew 40%"),
        ],
        mode="propose",
        timestamp="20260418_060000",
    )
    out_dir = writer.write(record)

    assert out_dir == tmp_path / "proposals" / "my-skill" / "20260418_060000"
    assert out_dir.is_dir()
    for fname in ("baseline_skill.md", "evolved_skill.md", "diff.patch",
                  "decision.json", "constraints.json", "review.md", "STATUS"):
        assert (out_dir / fname).exists(), f"missing {fname}"


def test_writer_status_starts_pending(tmp_path: Path):
    writer = ProposalWriter(tmp_path)
    rec = build_proposal_record(
        skill_name="s",
        baseline_text="a",
        evolved_text="b",
        baseline_score=0.5,
        evolved_score=0.5,
        decision=GateDecision(False, "no change", 0.0),
        constraint_results=[],
        mode="propose",
        timestamp="20260418_060001",
    )
    out = writer.write(rec)
    assert (out / "STATUS").read_text().strip() == PROPOSAL_STATUS_PENDING


def test_writer_baseline_and_evolved_exact_roundtrip(tmp_path: Path):
    writer = ProposalWriter(tmp_path)
    rec = build_proposal_record(
        skill_name="s",
        baseline_text=BASELINE_SKILL,
        evolved_text=EVOLVED_SKILL,
        baseline_score=0.5,
        evolved_score=0.55,
        decision=GateDecision(True, "ok", 0.05),
        constraint_results=[],
        mode="propose",
        timestamp="20260418_060002",
    )
    out = writer.write(rec)
    assert (out / "baseline_skill.md").read_text() == BASELINE_SKILL
    assert (out / "evolved_skill.md").read_text() == EVOLVED_SKILL


def test_writer_decision_json_shape(tmp_path: Path):
    writer = ProposalWriter(tmp_path)
    decision = GateDecision(True, "Improvement +0.050 meets threshold", 0.05)
    rec = build_proposal_record(
        skill_name="s",
        baseline_text="a",
        evolved_text="b",
        baseline_score=0.70,
        evolved_score=0.75,
        decision=decision,
        constraint_results=[],
        mode="auto",
        metadata={"iterations": 10, "eval_model": "test-model"},
        timestamp="20260418_060003",
    )
    out = writer.write(rec)
    payload = json.loads((out / "decision.json").read_text())
    assert payload["skill_name"] == "s"
    assert payload["baseline_score"] == pytest.approx(0.70)
    assert payload["evolved_score"] == pytest.approx(0.75)
    assert payload["improvement"] == pytest.approx(0.05, abs=1e-9)
    assert payload["mode"] == "auto"
    assert payload["auto_merge"] is True
    assert payload["regression"] is False
    assert payload["metadata"]["iterations"] == 10
    assert payload["metadata"]["eval_model"] == "test-model"


def test_writer_constraints_json_captures_results(tmp_path: Path):
    writer = ProposalWriter(tmp_path)
    constraints = [
        _FakeConstraint("skill_structure", True, "frontmatter ok"),
        _FakeConstraint("size_growth", False, "grew 3x"),
        _FakeConstraint("semantic_preservation", True, "name/description preserved"),
    ]
    rec = build_proposal_record(
        skill_name="s",
        baseline_text="a",
        evolved_text="b",
        baseline_score=0.5,
        evolved_score=0.5,
        decision=GateDecision(False, "below", 0.0),
        constraint_results=constraints,
        mode="propose",
        timestamp="20260418_060004",
    )
    out = writer.write(rec)
    payload = json.loads((out / "constraints.json").read_text())
    assert len(payload) == 3
    names = [p["name"] for p in payload]
    assert names == ["skill_structure", "size_growth", "semantic_preservation"]
    assert [p["passed"] for p in payload] == [True, False, True]


def test_writer_diff_patch_contains_changes(tmp_path: Path):
    writer = ProposalWriter(tmp_path)
    rec = build_proposal_record(
        skill_name="s",
        baseline_text=BASELINE_SKILL,
        evolved_text=EVOLVED_SKILL,
        baseline_score=0.5,
        evolved_score=0.6,
        decision=GateDecision(True, "ok", 0.1),
        constraint_results=[],
        mode="propose",
        timestamp="20260418_060005",
    )
    out = writer.write(rec)
    diff = (out / "diff.patch").read_text()
    assert "---" in diff and "+++" in diff
    assert "Evolved body line 1 — much clearer." in diff
    assert "-Original body text line 1." in diff


def test_writer_diff_patch_empty_when_identical(tmp_path: Path):
    writer = ProposalWriter(tmp_path)
    rec = build_proposal_record(
        skill_name="s",
        baseline_text=BASELINE_SKILL,
        evolved_text=BASELINE_SKILL,
        baseline_score=0.5,
        evolved_score=0.5,
        decision=GateDecision(False, "no change", 0.0),
        constraint_results=[],
        mode="propose",
        timestamp="20260418_060006",
    )
    out = writer.write(rec)
    assert (out / "diff.patch").read_text() == ""


def test_writer_review_md_flags_regression(tmp_path: Path):
    writer = ProposalWriter(tmp_path)
    rec = build_proposal_record(
        skill_name="s",
        baseline_text=BASELINE_SKILL,
        evolved_text=EVOLVED_SKILL,
        baseline_score=0.70,
        evolved_score=0.60,
        decision=GateDecision(False, "Regression detected: -0.100", -0.10, regression=True),
        constraint_results=[],
        mode="propose",
        timestamp="20260418_060007",
    )
    out = writer.write(rec)
    review = (out / "review.md").read_text()
    assert "Regression detected" in review
    assert "NOT eligible" in review


def test_writer_review_md_shows_auto_merge_when_eligible(tmp_path: Path):
    writer = ProposalWriter(tmp_path)
    rec = build_proposal_record(
        skill_name="s",
        baseline_text=BASELINE_SKILL,
        evolved_text=EVOLVED_SKILL,
        baseline_score=0.70,
        evolved_score=0.75,
        decision=GateDecision(True, "Improvement +0.050 meets threshold", 0.05),
        constraint_results=[_FakeConstraint("c1", True, "ok")],
        mode="propose",
        timestamp="20260418_060008",
    )
    out = writer.write(rec)
    review = (out / "review.md").read_text()
    assert "eligible for auto-merge" in review
    assert "✅" in review


def test_writer_can_write_multiple_proposals_same_skill(tmp_path: Path):
    writer = ProposalWriter(tmp_path)
    for i, ts in enumerate(["20260418_060010", "20260418_060011"]):
        rec = build_proposal_record(
            skill_name="s",
            baseline_text=BASELINE_SKILL,
            evolved_text=EVOLVED_SKILL,
            baseline_score=0.5,
            evolved_score=0.5 + 0.01 * i,
            decision=GateDecision(False, "below", 0.01 * i),
            constraint_results=[],
            mode="propose",
            timestamp=ts,
        )
        writer.write(rec)
    skill_root = tmp_path / "s"
    assert sorted(p.name for p in skill_root.iterdir()) == ["20260418_060010", "20260418_060011"]


def test_writer_creates_root_dir_if_missing(tmp_path: Path):
    target = tmp_path / "nonexistent" / "nested" / "proposals"
    writer = ProposalWriter(target)
    assert target.is_dir()

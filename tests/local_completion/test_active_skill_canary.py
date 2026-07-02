"""Tests for the LC2 active skill local canary packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evolution.local_completion.active_skill_canary import (
    LC2_ACTIVE_SKILL_CANARY_PHASE,
    scan_skill_text,
    write_active_skill_canary_packet,
)


ACTIVE_SKILL = """---
name: github-code-review
description: Review GitHub pull requests safely.
---

# GitHub Code Review

## External Write Gate

Never post, approve, request changes, merge, or otherwise write externally unless explicitly approved.

## First-use local-only canary

Run a local candidate-only canary before trusting this active skill as a baseline.
"""

CANDIDATE_SKILL = """---
name: github-code-review
description: Review GitHub pull requests safely.
---

# GitHub Code Review

## External Write Gate

Never post, approve, request changes, merge, or otherwise write externally unless explicitly approved.

## Candidate Note

Candidate wording differs for review only.
"""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_scan_skill_text_detects_headings_write_gate_and_assignment_like_risk():
    clean = scan_skill_text(ACTIVE_SKILL)

    assert clean["sha256"] == _sha(ACTIVE_SKILL)
    assert clean["line_count"] == len(ACTIVE_SKILL.splitlines())
    assert clean["headings"] == ["GitHub Code Review", "External Write Gate", "First-use local-only canary"]
    assert clean["external_write_gate_present"] is True
    assert clean["high_risk_assignment_like_hits"] == 0
    assert clean["scan_clean"] is True

    risky = scan_skill_text(ACTIVE_SKILL + '\nDANGEROUS_TOKEN="not-a-real-token-value-1234567890"')
    assert risky["scan_clean"] is False
    assert risky["high_risk_assignment_like_hits"] == 1


def test_write_active_skill_canary_packet_creates_candidate_only_bundle(tmp_path):
    active_path = tmp_path / "active" / "SKILL.md"
    candidate_path = tmp_path / "candidate" / "SKILL.md"
    active_path.parent.mkdir()
    candidate_path.parent.mkdir()
    active_path.write_text(ACTIVE_SKILL)
    candidate_path.write_text(CANDIDATE_SKILL)
    runs_root = tmp_path / "runs"

    result = write_active_skill_canary_packet(
        active_skill_path=active_path,
        candidate_skill_path=candidate_path,
        runs_root=runs_root,
        run_id="pytest-lc2",
        generated_at="2026-06-28T01:29:41Z",
        expected_active_sha=_sha(ACTIVE_SKILL),
    )

    bundle_root = Path(result["bundle_root"])
    decision_path = bundle_root / "decision.json"
    decision = json.loads(decision_path.read_text())

    assert result["decision_path"] == str(decision_path)
    assert decision["schema_version"] == "hse-local-completion-v1"
    assert decision["gate_id"] == "LC2"
    assert decision["phase"] == LC2_ACTIVE_SKILL_CANARY_PHASE
    assert decision["target"] == "github-code-review"
    assert decision["status"] == "PASS_PROVISIONAL_BASELINE_CANARY"
    assert decision["candidate_only"] is True
    assert decision["apply_ready"] is False
    assert decision["github"] == {
        "queried": False,
        "pr_created": False,
        "push_performed": False,
        "merge_performed": False,
        "publication_deferred": True,
    }
    assert decision["safety_invariants"]["active_skill_modified"] is False
    assert decision["safety_invariants"]["active_runtime_mutation"] is False
    assert decision["active_skill"]["sha256"] == _sha(ACTIVE_SKILL)
    assert decision["canary"]["external_write_gate_present"] is True
    assert decision["canary"]["current_hash_decision"] == "ACCEPT_AS_PROVISIONAL_SAFETY_HARDENED_BASELINE_FOR_READ_ONLY_USE_ONLY"
    assert decision["candidate_comparison"]["sha_match"] is False
    assert decision["candidate_comparison"]["status"] == "HASH_DRIFT_REVIEW_REQUIRED"
    assert decision["artifacts"]["active_snapshot"] == "inputs/active_skill.md"
    assert (bundle_root / "inputs" / "active_skill.md").read_text() == ACTIVE_SKILL
    assert (bundle_root / "inputs" / "rebuild_candidate_skill.md").read_text() == CANDIDATE_SKILL
    assert "Candidate wording differs" in (bundle_root / "candidates" / "active_vs_candidate.patch").read_text()
    assert "PASS_PROVISIONAL_BASELINE_CANARY" in (bundle_root / "reports" / "canary_report.md").read_text()
    assert "no active apply" in (bundle_root / "reports" / "rollback.md").read_text()


def test_write_active_skill_canary_packet_blocks_on_missing_gate_or_sha_mismatch(tmp_path):
    active_path = tmp_path / "active.md"
    candidate_path = tmp_path / "candidate.md"
    active_path.write_text(ACTIVE_SKILL.replace("## External Write Gate\n\n", ""))
    candidate_path.write_text(CANDIDATE_SKILL)

    result = write_active_skill_canary_packet(
        active_skill_path=active_path,
        candidate_skill_path=candidate_path,
        runs_root=tmp_path / "runs-missing-gate",
        run_id="pytest-lc2-blocked",
        generated_at="2026-06-28T01:29:41Z",
        expected_active_sha=_sha(active_path.read_text()),
    )
    decision = json.loads(Path(result["decision_path"]).read_text())
    assert decision["status"] == "BLOCKED"
    assert "missing_external_write_gate" in decision["canary"]["blockers"]

    active_path.write_text(ACTIVE_SKILL)
    with pytest.raises(ValueError, match="active skill SHA mismatch"):
        write_active_skill_canary_packet(
            active_skill_path=active_path,
            candidate_skill_path=candidate_path,
            runs_root=tmp_path / "runs-sha-mismatch",
            run_id="pytest-lc2-sha-mismatch",
            generated_at="2026-06-28T01:29:41Z",
            expected_active_sha="0" * 64,
        )

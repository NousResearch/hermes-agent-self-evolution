"""Tests for holdout-skip reporting (t_ac619234).

A skipped holdout (remaining budget < HOLDOUT_SKIP_THRESHOLD_SECONDS) is a
NON-measurement. These tests pin the three invariants from the reporting fix:

1. metrics.json baseline_score / evolved_score / improvement are NULL (N/A
   downstream), never placeholder 0.0 — a fabricated "+0.000 (+0.0%)" must
   never be misread as "no improvement".
2. GEPA's candidate survives the skip: handle_holdout_skip returns it in
   candidate_full so the save step can write evolved_candidate.md alongside
   the baseline-copy evolved_skill.md (later budget-raised re-run can
   diff/reuse it).
3. The deployed artifact stays the baseline text (byte-identical →
   has_diff=0 downstream), mirroring the 8b regression guard.
"""

import json

from evolution.skills.evolve_skill import (
    build_metrics,
    handle_holdout_skip,
)


def _fake_skill(body="body text", frontmatter="---\nname: x\n---\n"):
    return {"body": body, "raw": frontmatter + body}


def _real_holdout_result():
    """A normal (non-skipped) holdout result with real scores."""
    return {
        "baseline_scores": [0.4, 0.5],
        "evolved_scores": [0.6, 0.6],
        "examples_evaluated": 13,
        "examples_total": 13,
        "budget_exceeded": False,
        "cache_hits": 11,
        "calls_made": 2,
        "skipped": False,
    }


def test_skip_preserves_candidate_and_keeps_baseline():
    skill = _fake_skill()
    candidate_body = "evolved by GEPA"
    candidate_full = "---\nname: x\n---\nevolved by GEPA"
    (holdout_result, avg_b, avg_e, imp,
     evolved_body, evolved_full, candidate) = handle_holdout_skip(
        skill, candidate_body, candidate_full, ["ex1", "ex2"], 30.0)

    assert holdout_result["skipped"] is True
    assert holdout_result["examples_total"] == 2
    assert holdout_result["examples_evaluated"] == 0
    # Placeholders, NOT measurements — reporting layer renders N/A.
    assert avg_b == 0.0 and avg_e == 0.0 and imp == 0.0
    # Deployed artifact stays the baseline (no evidence to deploy)...
    assert evolved_body == skill["body"]
    assert evolved_full == skill["raw"]
    # ...but GEPA's candidate survives for a budget-raised re-run.
    assert candidate == candidate_full


def test_metrics_null_when_skipped():
    skill = _fake_skill()
    (holdout_result, avg_b, avg_e, imp,
     _, _, candidate) = handle_holdout_skip(skill, "cand body", "cand full", [], 10.0)
    m = build_metrics(
        "test-skill", "20260802_000000", 3, "opt", "eval",
        avg_b, avg_e, imp,
        len(skill["body"]), len(skill["body"]),
        25, 12, 13, 430.0, True, holdout_result, candidate, 480,
    )
    assert m["holdout_skipped"] is True
    assert m["holdout_candidate_preserved"] is True
    assert m["holdout_complete"] is False
    assert m["baseline_score"] is None
    assert m["evolved_score"] is None
    assert m["improvement"] is None
    # JSON round-trip: null, not 0.0 (what the report layer actually reads).
    blob = json.loads(json.dumps(m))
    assert blob["baseline_score"] is None
    assert blob["evolved_score"] is None
    assert blob["improvement"] is None


def test_metrics_numeric_when_not_skipped():
    """Non-skipped runs keep real float scores (no behavior change)."""
    skill = _fake_skill()
    holdout_result = _real_holdout_result()
    m = build_metrics(
        "test-skill", "20260802_000000", 3, "opt", "eval",
        0.45, 0.6, 0.15, len(skill["body"]), len(skill["body"]) + 3,
        25, 12, 13, 300.0, True, holdout_result, None, 480,
    )
    assert m["holdout_skipped"] is False
    assert m["holdout_complete"] is True
    assert m["holdout_candidate_preserved"] is False
    assert m["baseline_score"] == 0.45
    assert m["evolved_score"] == 0.6
    assert m["improvement"] == 0.15

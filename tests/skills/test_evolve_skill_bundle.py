"""Tests for Phase 1 local skill candidate bundle output."""

from __future__ import annotations

import json

from evolution.core.constraints import ConstraintResult
from evolution.skills.evolve_skill import write_skill_candidate_bundle


BASELINE_SKILL = """---
name: demo-skill
description: Demo skill
---

# Demo Skill

Do the baseline thing.
"""

EVOLVED_SKILL = """---
name: demo-skill
description: Demo skill
---

# Demo Skill

Do the evolved thing with verification.
"""


def test_write_skill_candidate_bundle_uses_local_standard_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("HSE_RUNS_ROOT", str(tmp_path / "runs"))
    constraints = [ConstraintResult(True, "skill_structure", "ok")]
    metrics = {
        "baseline_score": 0.5,
        "evolved_score": 0.7,
        "improvement": 0.2,
        "constraints_passed": True,
        "holdout_examples": 3,
        "holdout_total_examples": 3,
    }

    bundle_root = write_skill_candidate_bundle(
        skill_name="demo-skill",
        baseline_skill=BASELINE_SKILL,
        evolved_skill=EVOLVED_SKILL,
        metrics=metrics,
        constraint_results=constraints,
    )

    assert bundle_root.parent == tmp_path / "runs"
    assert (bundle_root / "inputs" / "target_manifest.json").exists()
    assert (bundle_root / "candidates" / "baseline_skill.md").read_text() == BASELINE_SKILL
    assert (bundle_root / "candidates" / "evolved_skill.md").read_text() == EVOLVED_SKILL
    patch_text = (bundle_root / "candidates" / "candidate.patch").read_text()
    assert "--- baseline_skill.md" in patch_text
    assert "+Do the evolved thing with verification." in patch_text
    assert json.loads((bundle_root / "eval" / "metrics.json").read_text()) == metrics
    constraints_payload = json.loads((bundle_root / "eval" / "constraint_results.json").read_text())
    assert constraints_payload == [{"constraint_name": "skill_structure", "details": None, "message": "ok", "passed": True}]
    assert (bundle_root / "reports" / "report.md").exists()
    assert (bundle_root / "reports" / "rollback.md").exists()

    decision = json.loads((bundle_root / "decision.json").read_text())
    assert decision["status"] == "PASS_CANDIDATE_ONLY"
    assert decision["candidate_only"] is True
    assert decision["apply_ready"] is False
    assert decision["github"]["pr_created"] is False
    assert decision["safety_invariants"]["active_skill_modified"] is False
    assert decision["artifacts"]["patch"] == "candidates/candidate.patch"


def test_write_skill_candidate_bundle_marks_no_diff_as_no_go(tmp_path, monkeypatch):
    monkeypatch.setenv("HSE_RUNS_ROOT", str(tmp_path / "runs"))

    bundle_root = write_skill_candidate_bundle(
        skill_name="demo-skill",
        baseline_skill=BASELINE_SKILL,
        evolved_skill=BASELINE_SKILL,
        metrics={"improvement": 0.5, "constraints_passed": True},
        constraint_results=[ConstraintResult(True, "skill_structure", "ok")],
    )

    decision = json.loads((bundle_root / "decision.json").read_text())
    assert decision["status"] == "NO_DIFF_NO_GO"
    assert (bundle_root / "candidates" / "candidate.patch").read_text() == "No candidate skill changes.\n"


def test_write_skill_candidate_bundle_redacts_private_seed_skill_path_from_machine_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("HSE_RUNS_ROOT", str(tmp_path / "runs"))
    private_seed_path = "/Users/example/private/SKILL.md"

    bundle_root = write_skill_candidate_bundle(
        skill_name="demo-skill",
        baseline_skill=BASELINE_SKILL,
        evolved_skill=EVOLVED_SKILL,
        metrics={
            "improvement": 0.2,
            "constraints_passed": True,
            "seed_skill_path": private_seed_path,
        },
        constraint_results=[ConstraintResult(True, "skill_structure", "ok")],
    )

    decision = json.loads((bundle_root / "decision.json").read_text())
    eval_metrics = json.loads((bundle_root / "eval" / "metrics.json").read_text())
    assert decision["metrics"]["seed_skill_path"] == "[redacted-seed-skill-path]"
    assert eval_metrics["seed_skill_path"] == "[redacted-seed-skill-path]"
    assert private_seed_path not in json.dumps(decision)
    assert private_seed_path not in json.dumps(eval_metrics)

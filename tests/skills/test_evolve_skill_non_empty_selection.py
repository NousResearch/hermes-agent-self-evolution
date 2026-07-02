"""Regression tests for exporting non-empty Phase 1 skill candidates."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from click.testing import CliRunner

from evolution.core.constraints import ConstraintResult
from evolution.core.fitness import _skill_size_penalty
from evolution.skills import evolve_skill
from evolution.skills.evolve_skill import (
    _build_gepa_optimizer,
    _decision_status_for_skill_candidate,
    _resolve_skill_path,
    _select_non_empty_export_candidate,
    skill_fitness_feedback_metric,
    write_skill_candidate_bundle,
)


class FakeSkillModule:
    def __init__(self, skill_text: str):
        self.skill_text = skill_text
        self.detailed_results: Any = None


def test_selects_best_non_empty_gepa_candidate_when_best_program_is_baseline():
    baseline_body = "# Demo Skill\n\nDo the baseline thing.\n"
    weak_candidate = "# Demo Skill\n\nDo the evolved thing with insufficient safety.\n"
    stronger_candidate = "# Demo Skill\n\nDo the evolved thing with explicit verification and safety gates.\n"
    optimized = FakeSkillModule(baseline_body)
    optimized.detailed_results = SimpleNamespace(
        best_idx=0,
        candidates=[
            FakeSkillModule(baseline_body),
            FakeSkillModule(weak_candidate),
            FakeSkillModule(stronger_candidate),
        ],
        val_aggregate_scores=[0.40, 0.33, 0.37],
    )

    selected, metadata = _select_non_empty_export_candidate(optimized, baseline_body=baseline_body)

    assert selected.skill_text == stronger_candidate
    assert metadata["strategy"] == "best_non_empty_gepa_candidate"
    assert metadata["gepa_best_was_empty_diff"] is True
    assert metadata["selected_candidate_index"] == 2
    assert metadata["selected_candidate_val_score"] == 0.37
    assert metadata["gepa_best_candidate_index"] == 0
    assert metadata["gepa_best_val_score"] == 0.40
    assert metadata["selected_candidate_has_non_empty_diff"] is True
    assert metadata["selected_candidate_val_score_below_gepa_best"] is True


def test_uses_gepa_best_candidate_when_it_is_already_non_empty():
    baseline_body = "# Demo Skill\n\nDo the baseline thing.\n"
    evolved_body = "# Demo Skill\n\nDo the evolved thing.\n"
    optimized = FakeSkillModule(evolved_body)
    optimized.detailed_results = SimpleNamespace(
        best_idx=1,
        candidates=[FakeSkillModule(baseline_body), FakeSkillModule(evolved_body)],
        val_aggregate_scores=[0.40, 0.55],
    )

    selected, metadata = _select_non_empty_export_candidate(optimized, baseline_body=baseline_body)

    assert selected is optimized
    assert metadata["strategy"] == "gepa_best"
    assert metadata["selected_candidate_has_non_empty_diff"] is True
    assert metadata["selected_candidate_index"] == 1


def test_reports_no_non_empty_candidate_when_all_candidates_match_baseline():
    baseline_body = "# Demo Skill\n\nDo the baseline thing.\n"
    optimized = FakeSkillModule(baseline_body)
    optimized.detailed_results = SimpleNamespace(
        best_idx=0,
        candidates=[FakeSkillModule(baseline_body), FakeSkillModule(baseline_body)],
        val_aggregate_scores=[0.40, 0.39],
    )

    selected, metadata = _select_non_empty_export_candidate(optimized, baseline_body=baseline_body)

    assert selected is optimized
    assert metadata["strategy"] == "gepa_best_empty_no_non_empty_candidate"
    assert metadata["selected_candidate_has_non_empty_diff"] is False
    assert metadata["non_empty_candidate_count"] == 0


def test_non_empty_candidate_below_gepa_best_is_regression_no_go_even_with_holdout_improvement():
    status = _decision_status_for_skill_candidate(
        diff_text="--- baseline_skill.md\n+++ evolved_skill.md\n@@\n-old\n+new\n",
        improvement=0.2,
        constraints_passed=True,
        candidate_selection={"selected_candidate_val_score_below_gepa_best": True},
    )

    assert status == "REGRESSION_NO_GO"


def test_quality_repairs_regressive_non_empty_candidate_with_non_disruptive_sidecar_evidence():
    baseline_body = "# Demo Skill\n\nReview diffs safely.\n"
    regressive_candidate = "# Demo Skill\n\nReview diffs, but with less safety detail.\n"
    optimized = FakeSkillModule(baseline_body)
    optimized.detailed_results = SimpleNamespace(
        best_idx=0,
        candidates=[FakeSkillModule(baseline_body), FakeSkillModule(regressive_candidate)],
        val_aggregate_scores=[0.50, 0.30],
    )
    quality_examples = [
        SimpleNamespace(
            rubric_checks=[
                {
                    "id": "inline_endpoint",
                    "description": "Uses inline comment endpoint",
                    "weight": 1.0,
                    "pattern_any": ["pulls/\\$PR_NUMBER/comments"],
                },
                {
                    "id": "side_right",
                    "description": "Defaults to RIGHT side",
                    "weight": 1.0,
                    "pattern_any": ["side=.?RIGHT", "\\\"side\\\": \\\"RIGHT\\\""],
                },
            ]
        )
    ]

    selected, metadata = _select_non_empty_export_candidate(
        optimized,
        baseline_body=baseline_body,
        quality_examples=quality_examples,
    )

    assert metadata["strategy"] == "quality_repaired_non_disruptive_evidence"
    assert metadata["raw_selected_candidate_val_score_below_gepa_best"] is True
    assert metadata["selected_candidate_val_score_below_gepa_best"] is False
    assert metadata["selected_candidate_has_non_empty_diff"] is False
    assert metadata["quality_repair_applied"] is True
    assert metadata["quality_repair_gate_passed"] is True
    assert metadata["quality_repair_evidence_placement"] == "metadata_sidecar"
    assert metadata["quality_repair_doc_score_after"] > metadata["quality_repair_doc_score_before"]
    assert metadata["quality_repair_size_delta"] == 0
    assert selected.skill_text.rstrip() == baseline_body.rstrip()
    assert "<!-- HSE candidate-quality evidence" not in selected.skill_text
    evidence_text = "\n".join(metadata["quality_repair_evidence"]["lines"])
    assert "pulls/$PR_NUMBER/comments" in evidence_text
    assert "side=RIGHT" in evidence_text
    assert _skill_size_penalty(selected.skill_text) == _skill_size_penalty(baseline_body)


def test_content_level_repair_creates_compact_runtime_candidate_without_size_penalty_regression():
    oversized_pitfalls = "\n".join(
        f"{idx}. Legacy generic review pitfall text that is intentionally verbose and replaceable."
        for idx in range(40)
    )
    baseline_body = (
        "# Demo Skill\n\n"
        "Review pull requests locally and safely before writing anywhere.\n\n"
        "## Common Pitfalls\n\n"
        f"{oversized_pitfalls}\n\n"
        "## Verification Checklist\n\n"
        "- [ ] Diff inspected.\n"
    )
    regressive_candidate = "# Demo Skill\n\nReview diffs, but with less safety detail.\n"
    optimized = FakeSkillModule(baseline_body)
    optimized.detailed_results = SimpleNamespace(
        best_idx=0,
        candidates=[FakeSkillModule(baseline_body), FakeSkillModule(regressive_candidate)],
        val_aggregate_scores=[0.50, 0.30],
    )
    quality_examples = [
        SimpleNamespace(
            rubric_checks=[
                {"id": "commit_context", "description": "Uses git log for commit context", "pattern_any": ["git log main\\.\\.HEAD"]},
                {"id": "files_endpoint", "description": "Lists changed files or PR diff", "pattern_any": ["gh pr diff .*--name-only", "/pulls/\\$PR_NUMBER/files"]},
                {"id": "checkout", "description": "Checks out or fetches PR locally", "pattern_any": ["git fetch origin pull/\\$PR_NUMBER/head", "gh pr checkout"]},
                {"id": "fabrication", "description": "Explicitly forbids fabricated review results", "pattern_any": ["Never fabricate"]},
                {"id": "inline_fields", "description": "Lists required info for inline comment", "pattern_all": ["PR number", "File path", "Line number", "Comment body"]},
                {"id": "inline_endpoint", "description": "Uses pulls comments endpoint or gh api", "pattern_any": ["pulls/\\$PR_NUMBER/comments", "gh api .*pulls/.*/comments"]},
                {"id": "side_right", "description": "Defaults to RIGHT side", "pattern_any": ["side=.?RIGHT", "\\\"side\\\": \\\"RIGHT\\\""]},
                {"id": "line_not_diff", "description": "Explains API rejection if line is not in diff", "pattern_all": ["line is not part of the diff", "GitHub rejects"]},
            ]
        )
    ]

    selected, metadata = _select_non_empty_export_candidate(
        optimized,
        baseline_body=baseline_body,
        quality_examples=quality_examples,
        content_level_repair=True,
    )

    assert metadata["strategy"] == "quality_repaired_compact_content"
    assert metadata["selected_candidate_has_non_empty_diff"] is True
    assert metadata["raw_selected_candidate_val_score_below_gepa_best"] is True
    assert metadata["selected_candidate_val_score_below_gepa_best"] is False
    assert metadata["quality_repair_evidence_placement"] == "runtime_compact_section_and_metadata_sidecar"
    assert metadata["quality_repair_size_penalty_delta"] <= 0
    assert selected.skill_text != baseline_body
    assert "<!-- HSE candidate-quality evidence" not in selected.skill_text
    assert "## Common Pitfalls and Evidence Hints" in selected.skill_text
    assert "git log main..HEAD" in selected.skill_text
    assert "/pulls/$PR_NUMBER/files" in selected.skill_text
    assert "git fetch origin pull/$PR_NUMBER/head" in selected.skill_text
    assert "Never fabricate" in selected.skill_text
    assert "PR number" in selected.skill_text and "File path" in selected.skill_text
    assert "pulls/$PR_NUMBER/comments" in selected.skill_text
    assert "side=RIGHT" in selected.skill_text
    assert "line is not part of the diff" in selected.skill_text
    assert _skill_size_penalty(selected.skill_text) <= _skill_size_penalty(baseline_body)


def test_content_level_repair_creates_compact_runtime_candidate_when_gepa_exports_no_non_empty_candidate():
    baseline_body = (
        "# Demo Skill\n\n"
        "Review pull requests locally and safely before writing anywhere.\n\n"
        "## Common Pitfalls\n\n"
        "1. Generic pitfall text.\n\n"
        "## Verification Checklist\n\n"
        "- [ ] Diff inspected.\n"
    )
    optimized = FakeSkillModule(baseline_body)
    optimized.detailed_results = SimpleNamespace(
        best_idx=0,
        candidates=[FakeSkillModule(baseline_body), FakeSkillModule(baseline_body)],
        val_aggregate_scores=[0.50, 0.49],
    )
    quality_examples = [
        SimpleNamespace(
            rubric_checks=[
                {"id": "inline_endpoint", "description": "Uses pulls comments endpoint", "pattern_any": ["pulls/\\$PR_NUMBER/comments"]},
                {"id": "side_right", "description": "Defaults to RIGHT side", "pattern_any": ["side=.?RIGHT"]},
            ]
        )
    ]

    selected, metadata = _select_non_empty_export_candidate(
        optimized,
        baseline_body=baseline_body,
        quality_examples=quality_examples,
        content_level_repair=True,
    )

    assert metadata["strategy"] == "quality_repaired_compact_content_from_empty_gepa"
    assert metadata["gepa_best_was_empty_diff"] is True
    assert metadata["non_empty_candidate_count"] == 0
    assert metadata["selected_candidate_has_non_empty_diff"] is True
    assert metadata["selected_candidate_val_score_below_gepa_best"] is False
    assert "pulls/$PR_NUMBER/comments" in selected.skill_text
    assert "side=RIGHT" in selected.skill_text


def test_quality_repaired_candidate_can_pass_decision_gate_with_positive_holdout_improvement():
    status = _decision_status_for_skill_candidate(
        diff_text="--- baseline_skill.md\n+++ evolved_skill.md\n@@\n-old\n+new\n",
        improvement=0.2,
        constraints_passed=True,
        candidate_selection={
            "selected_candidate_val_score_below_gepa_best": False,
            "raw_selected_candidate_val_score_below_gepa_best": True,
            "quality_repair_applied": True,
            "quality_repair_gate_passed": True,
        },
    )

    assert status == "PASS_CANDIDATE_ONLY"


def test_negative_holdout_improvement_is_regression_no_go():
    status = _decision_status_for_skill_candidate(
        diff_text="--- baseline_skill.md\n+++ evolved_skill.md\n@@\n-old\n+new\n",
        improvement=-0.01,
        constraints_passed=True,
        candidate_selection={"selected_candidate_val_score_below_gepa_best": False},
    )

    assert status == "REGRESSION_NO_GO"


def test_gepa_optimizer_uses_feedback_rich_quality_objective(monkeypatch):
    captured = {}

    class FakeGEPA:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("evolution.skills.evolve_skill.dspy.GEPA", FakeGEPA)
    monkeypatch.setattr("evolution.skills.evolve_skill.make_lm", lambda model, hermes_repo=None: f"lm:{model}:{hermes_repo}")

    optimizer = _build_gepa_optimizer(
        iterations=7,
        optimizer_model="openai-codex/gpt-5.5",
        hermes_repo="/tmp/hermes-facade",
    )

    assert isinstance(optimizer, FakeGEPA)
    assert captured["metric"] is skill_fitness_feedback_metric
    assert captured["max_metric_calls"] == 7
    assert captured["reflection_lm"] == "lm:openai-codex/gpt-5.5:/tmp/hermes-facade"
    assert captured["track_stats"] is True


def test_cli_forwards_objective_expansion_flag(monkeypatch):
    captured = {}

    def fake_evolve(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(evolve_skill, "evolve", fake_evolve)

    result = CliRunner().invoke(
        evolve_skill.main,
        [
            "--skill",
            "github-code-review",
            "--eval-source",
            "golden",
            "--dataset-path",
            "/tmp/golden",
            "--expand-objective-examples",
            "--dry-run",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert captured["skill_name"] == "github-code-review"
    assert captured["expand_objective_examples"] is True


def test_cli_forwards_content_level_repair_flag(monkeypatch):
    captured = {}

    def fake_evolve(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(evolve_skill, "evolve", fake_evolve)

    result = CliRunner().invoke(
        evolve_skill.main,
        [
            "--skill",
            "github-code-review",
            "--eval-source",
            "golden",
            "--dataset-path",
            "/tmp/golden",
            "--content-level-repair",
            "--dry-run",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert captured["skill_name"] == "github-code-review"
    assert captured["content_level_repair"] is True


def test_resolve_explicit_skill_path_outside_hermes_repo(tmp_path):
    skill_path = tmp_path / "external" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("---\nname: time-rewind\n---\n\n# Time Rewind\n")

    resolved = _resolve_skill_path(
        "time-rewind",
        tmp_path / "empty-hermes-repo",
        skill_path=str(skill_path),
    )

    assert resolved == skill_path


def test_resolve_user_local_skills_root(tmp_path):
    skill_path = tmp_path / "skills" / "security" / "time-rewind" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("---\nname: time-rewind\n---\n\n# Time Rewind\n")

    resolved = _resolve_skill_path(
        "time-rewind",
        tmp_path / "empty-hermes-repo",
        skills_root=str(tmp_path / "skills"),
    )

    assert resolved == skill_path


def test_cli_forwards_explicit_skill_path_and_skills_root(monkeypatch, tmp_path):
    captured = {}

    def fake_evolve(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(evolve_skill, "evolve", fake_evolve)
    skill_path = tmp_path / "security" / "time-rewind" / "SKILL.md"
    skills_root = tmp_path / "skills"

    result = CliRunner().invoke(
        evolve_skill.main,
        [
            "--skill",
            "time-rewind",
            "--skill-path",
            str(skill_path),
            "--skills-root",
            str(skills_root),
            "--dry-run",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert captured["skill_name"] == "time-rewind"
    assert captured["skill_path"] == str(skill_path)
    assert captured["skills_root"] == str(skills_root)


def test_candidate_bundle_writes_quality_repair_evidence_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("HSE_RUNS_ROOT", str(tmp_path))
    evidence = {
        "placement": "metadata_sidecar",
        "lines": [
            "- Uses inline endpoint: preserve `pulls/$PR_NUMBER/comments`.",
            "- Defaults to RIGHT: preserve `side=RIGHT`.",
        ],
    }
    bundle = write_skill_candidate_bundle(
        skill_name="github-code-review",
        baseline_skill="---\nname: github-code-review\n---\n\n# Demo\n",
        evolved_skill="---\nname: github-code-review\n---\n\n# Demo\n",
        metrics={"improvement": 0.0, "constraints_passed": True},
        constraint_results=[ConstraintResult(True, "unit", "ok")],
        candidate_selection={
            "strategy": "quality_repaired_non_disruptive_evidence",
            "quality_repair_applied": True,
            "quality_repair_evidence": evidence,
        },
    )

    sidecar_path = bundle / "eval" / "quality_repair_evidence.json"
    assert sidecar_path.exists()
    assert json.loads(sidecar_path.read_text()) == evidence
    decision = json.loads((bundle / "decision.json").read_text())
    assert decision["status"] == "NO_DIFF_NO_GO"
    assert decision["artifacts"]["quality_repair_evidence"] == "eval/quality_repair_evidence.json"

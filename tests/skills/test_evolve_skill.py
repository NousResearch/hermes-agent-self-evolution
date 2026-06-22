"""Integration tests for the evolve_skill pipeline.

These tests exercise the orchestration logic in `evolve()` without making
any real LLM API calls — DSPy components are mocked. They focus on the
control flow: dry-run short-circuit, skill discovery, pre-flight model
check, and the synthetic happy path with everything mocked out.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evolution.skills.evolve_skill import evolve


SKILL_TEXT = """---
name: demo-skill
description: A demo skill used for testing the evolution pipeline.
---

# Demo Skill

1. Read the task.
2. Do the thing.
3. Verify the result.
"""


@pytest.fixture
def hermes_repo(tmp_path):
    """A minimal fake hermes-agent repo containing one skill."""
    skill_dir = tmp_path / "skills" / "testing" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_TEXT, encoding="utf-8")
    return tmp_path


# ── Dry run ───────────────────────────────────────────────────────────────


def test_dry_run_returns_none_without_api(hermes_repo):
    # dry_run must short-circuit BEFORE any LLM/preflight call.
    result = evolve(
        skill_name="demo-skill",
        hermes_repo=str(hermes_repo),
        dry_run=True,
    )
    assert result is None


def test_dry_run_does_not_call_preflight(hermes_repo):
    with patch("evolution.skills.evolve_skill.dspy.LM") as mock_lm:
        evolve(skill_name="demo-skill", hermes_repo=str(hermes_repo), dry_run=True)
    # No model should be instantiated in dry-run mode
    mock_lm.assert_not_called()


# ── Skill discovery ───────────────────────────────────────────────────────


def test_missing_skill_exits(hermes_repo):
    with patch("evolution.skills.evolve_skill.dspy.LM"):
        with pytest.raises(SystemExit):
            evolve(
                skill_name="does-not-exist",
                hermes_repo=str(hermes_repo),
                dry_run=True,
            )


# ── Pre-flight model check ────────────────────────────────────────────────


def test_preflight_failure_exits(hermes_repo):
    # A model that raises on call should abort with SystemExit.
    def boom(*args, **kwargs):
        raise RuntimeError("model unreachable")

    mock_lm = MagicMock()
    mock_lm.side_effect = boom  # calling test_lm("respond with OK") raises

    with patch("evolution.skills.evolve_skill.dspy.LM", return_value=mock_lm):
        with pytest.raises(SystemExit):
            evolve(
                skill_name="demo-skill",
                hermes_repo=str(hermes_repo),
                dry_run=False,
                eval_source="synthetic",
            )


# ── End-to-end synthetic path (fully mocked) ──────────────────────────────


def _run_mocked_evolution(hermes_repo, tmp_path, monkeypatch, evolved_body):
    """Run evolve() through GEPA with all LLM I/O mocked.

    The dataset builder and GEPA optimizer are mocked so no API calls happen.
    GEPA returns an optimized module whose `skill_text` is `evolved_body`.
    Returns the evolve() return value.
    """
    monkeypatch.chdir(tmp_path)  # so ./datasets and ./output land in tmp

    from evolution.core.dataset_builder import (
        SyntheticDatasetBuilder,
        EvalDataset,
        EvalExample,
    )

    fake_dataset = EvalDataset(
        train=[EvalExample("task a", "rubric a"), EvalExample("task b", "rubric b")],
        val=[EvalExample("task v", "rubric v")],
        holdout=[EvalExample("task h", "rubric h")],
    )

    preflight_lm = MagicMock(return_value="OK")
    optimized = MagicMock()
    optimized.skill_text = evolved_body
    # Holdout eval calls optimized_module(task_input=...) -> .output
    optimized.return_value = MagicMock(output="evolved answer for the task")

    with patch("evolution.skills.evolve_skill.dspy.LM", return_value=preflight_lm), \
         patch("evolution.skills.evolve_skill.dspy.configure"), \
         patch("evolution.skills.evolve_skill.dspy.context"), \
         patch.object(SyntheticDatasetBuilder, "generate", return_value=fake_dataset), \
         patch("evolution.skills.evolve_skill.dspy.GEPA") as mock_gepa, \
         patch("evolution.skills.evolve_skill.SkillModule") as mock_skillmod:

        mock_gepa.return_value.compile.return_value = optimized
        # Baseline module also returns a parseable output object
        mock_skillmod.return_value.return_value = MagicMock(output="baseline answer")

        return evolve(
            skill_name="demo-skill",
            hermes_repo=str(hermes_repo),
            iterations=1,
            eval_source="synthetic",
            dry_run=False,
            create_pr=False,
            benchmark_gate=False,
        )


def test_e2e_semantic_drift_saves_failed_variant(hermes_repo, tmp_path, monkeypatch):
    """An evolved body that drifts from baseline must fail the semantic gate
    and be written to output/<skill>/evolved_FAILED.md without deploying."""
    result = _run_mocked_evolution(
        hermes_repo, tmp_path, monkeypatch,
        evolved_body="Totally unrelated content about cooking pasta.",
    )
    assert result is None  # bailed at constraint gate
    failed = tmp_path / "output" / "demo-skill" / "evolved_FAILED.md"
    assert failed.exists()


def test_e2e_preserved_skill_runs_to_completion(hermes_repo, tmp_path, monkeypatch):
    """An evolved body close to the baseline passes the gates and the run
    completes, writing metrics.json and the evolved skill to output/."""
    # Keep wording and length close to baseline so BOTH the semantic check
    # (Jaccard ≥ 0.7) and the growth limit (≤ +20%) pass.
    evolved = (
        "# Demo Skill\n\n"
        "1. Read the task.\n"
        "2. Do the thing.\n"
        "3. Verify the result now.\n"
    )
    _run_mocked_evolution(hermes_repo, tmp_path, monkeypatch, evolved_body=evolved)

    runs = list((tmp_path / "output" / "demo-skill").glob("*/metrics.json"))
    assert runs, "expected a metrics.json from a completed run"
    import json
    metrics = json.loads(runs[0].read_text())
    assert metrics["skill_name"] == "demo-skill"
    assert metrics["constraints_passed"] is True

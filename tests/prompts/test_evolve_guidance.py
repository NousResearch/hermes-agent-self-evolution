"""Integration tests for the Phase 3 orchestrator (all LLM I/O mocked)."""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evolution.prompts.evolve_guidance import evolve_guidance


def _fake_repo(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "prompt_builder.py").write_text(textwrap.dedent('''
        MEMORY_GUIDANCE = "Keep memory compact."
        TASK_COMPLETION_GUIDANCE = "Finish the job."
        SKILLS_GUIDANCE = "Save skills."
        TOOL_USE_ENFORCEMENT_GUIDANCE = "Use tools."
        OPENAI_MODEL_EXECUTION_GUIDANCE = "Be precise."
        GOOGLE_MODEL_OPERATIONAL_GUIDANCE = "Be concise."
        DEFAULT_AGENT_IDENTITY = "You are Hermes."
    '''), encoding="utf-8")
    return tmp_path


def test_dry_run_returns_none_without_api(tmp_path):
    repo = _fake_repo(tmp_path)
    result = evolve_guidance(hermes_repo=str(repo), dry_run=True)
    assert result is None


def test_dry_run_does_not_instantiate_lm(tmp_path):
    repo = _fake_repo(tmp_path)
    with patch("evolution.prompts.evolve_guidance.dspy.LM") as mock_lm:
        evolve_guidance(hermes_repo=str(repo), dry_run=True)
    mock_lm.assert_not_called()


def test_missing_repo_exits(tmp_path):
    with patch("evolution.prompts.evolve_guidance.dspy.LM"):
        with pytest.raises(SystemExit):
            evolve_guidance(hermes_repo=str(tmp_path / "nope"), dry_run=True)


def test_e2e_writes_output(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    from evolution.prompts.guidance_behaviors import BehavioralProbe, ProbeDataset
    fake_ds = ProbeDataset(
        train=[BehavioralProbe("Save dark mode", "use memory", "memory")],
        val=[BehavioralProbe("Run tests", "use terminal", "tool_use")],
        holdout=[BehavioralProbe("Write script", "produce code", "task_completion")],
    )

    preflight_lm = MagicMock(return_value="OK")
    optimized = MagicMock()
    optimized.guidance = {
        "MEMORY_GUIDANCE": "Keep memory compact.",
        "TASK_COMPLETION_GUIDANCE": "Finish the job.",
        "SKILLS_GUIDANCE": "Save skills.",
        "TOOL_USE_ENFORCEMENT_GUIDANCE": "Use tools.",
        "OPENAI_MODEL_EXECUTION_GUIDANCE": "Be precise.",
        "GOOGLE_MODEL_OPERATIONAL_GUIDANCE": "Be concise.",
    }
    optimized.return_value = MagicMock(score=0.85)

    with patch("evolution.prompts.evolve_guidance.dspy.LM", return_value=preflight_lm), \
         patch("evolution.prompts.evolve_guidance.dspy.configure"), \
         patch("evolution.prompts.evolve_guidance.dspy.context"), \
         patch("evolution.prompts.evolve_guidance.dspy.GEPA") as mock_gepa, \
         patch("evolution.prompts.evolve_guidance.GuidanceJudgeModule") as mock_mod:
        mock_gepa.return_value.compile.return_value = optimized
        mock_mod.return_value.return_value = MagicMock(score=0.85)
        mock_mod.return_value.guidance = optimized.guidance

        evolve_guidance(hermes_repo=str(repo), iterations=1, dry_run=False, write_back=False)

    runs = list((tmp_path / "output" / "guidance").glob("*/metrics.json"))
    assert runs, "expected metrics.json from a completed run"
    import json
    metrics = json.loads(runs[0].read_text(encoding="utf-8"))
    assert "baseline_behavioral_score" in metrics
    assert "evolved_behavioral_score" in metrics

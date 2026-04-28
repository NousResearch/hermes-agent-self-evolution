"""Tests for DB-backed DSPy/GEPA run execution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from evolution.orchestrator.executor import execute_skill_run
from tests.core.test_run_execution import _seed_skill_run
from tests.core.test_dspy_gepa_optimizer import _FakeDSpy, _FakeSkillModule


def test_dspy_gepa_strategy_persists_optimizer_metadata_and_excludes_holdout(tmp_path, monkeypatch):
    root, _repo_path, store, run, _target, _dataset, _examples = _seed_skill_run(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value")
    fake_dspy = _FakeDSpy()

    result = execute_skill_run(
        store=store,
        root=root,
        run_id=run["id"],
        strategy="dspy-gepa",
        provider="deepseek",
        optimizer_model="deepseek-v4-pro",
        eval_model="deepseek-v4-flash",
        max_tokens=1024,
        temperature=0.1,
        timeout=45.0,
        extra_body={"thinking": {"type": "disabled"}},
        dspy_model_prefix="openai",
        dspy_module=fake_dspy,
        skill_module_factory=_FakeSkillModule,
    )

    assert result["run"]["status"] == "completed"
    candidates = store.list_candidates(run["id"])
    evolved = next(candidate for candidate in candidates if candidate["role"] == "evolved")
    metadata = evolved["metadata_json"]
    assert metadata["source"] == "dspy_gepa"
    assert metadata["strategy"] == "dspy-gepa"
    assert metadata["provider"] == "deepseek"
    assert metadata["optimizer_model"] == "deepseek-v4-pro"
    assert metadata["eval_model"] == "deepseek-v4-flash"
    assert metadata["max_full_evals"] == 2
    assert metadata["train_examples"] == 1
    assert metadata["val_examples"] == 1
    assert metadata["holdout_examples_used_for_generation"] == 0
    assert "secret-value" not in str(metadata)

    evolved_text = Path(store.get_artifact(evolved["artifact_id"])["storage_uri"]).read_text()
    assert "GEPA optimized body" in evolved_text
    assert "train-only calibration" in evolved_text
    assert "validation-only rubric" in evolved_text
    assert "holdout-only forbidden leak" not in evolved_text

    evaluations = store.list_evaluations(run["id"])
    assert len(evaluations) == 6
    assert {evaluation["details_json"]["strategy"] for evaluation in evaluations} == {"dspy-gepa"}

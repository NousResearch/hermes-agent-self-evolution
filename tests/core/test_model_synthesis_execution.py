"""Tests for model-backed optimizer execution strategies."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from evolution.orchestrator.executor import execute_skill_run
from tests.core.test_run_execution import _seed_skill_run


class _FakeCompletions:
    def __init__(self, calls):
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "# Test Skill\n\n"
                            "Follow the existing procedure.\n\n"
                            "## Model Synthesis\n"
                            "- mention train-only calibration\n"
                            "- mention validation-only rubric\n"
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )


class _FakeClient:
    def __init__(self, calls):
        self.chat = SimpleNamespace(completions=_FakeCompletions(calls))


def test_model_synthesis_strategy_uses_optimizer_model_and_persists_metadata(tmp_path, monkeypatch):
    root, _repo_path, store, run, _target, _dataset, _examples = _seed_skill_run(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value")
    calls = []

    def factory(*, api_key, base_url, timeout):
        assert api_key == "secret-value"
        assert base_url == "https://api.deepseek.com"
        assert timeout == 30.0
        return _FakeClient(calls)

    result = execute_skill_run(
        store=store,
        root=root,
        run_id=run["id"],
        strategy="model-synthesis",
        optimizer_model="deepseek-v4-pro",
        provider="deepseek",
        max_tokens=512,
        temperature=0.0,
        timeout=30.0,
        extra_body={"thinking": {"type": "disabled"}},
        client_factory=factory,
    )

    assert result["run"]["status"] == "completed"
    assert calls[0]["model"] == "deepseek-v4-pro"
    assert calls[0]["max_tokens"] == 512
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "holdout-only forbidden leak" not in calls[0]["messages"][0]["content"]

    candidates = store.list_candidates(run["id"])
    evolved = next(candidate for candidate in candidates if candidate["role"] == "evolved")
    metadata = evolved["metadata_json"]
    assert metadata["source"] == "model_synthesis"
    assert metadata["strategy"] == "model-synthesis"
    assert metadata["provider"] == "deepseek"
    assert metadata["optimizer_model"] == "deepseek-v4-pro"
    assert metadata["extra_body"] == {"thinking": {"type": "disabled"}}
    assert metadata["holdout_examples_used_for_generation"] == 0
    assert "secret-value" not in str(metadata)

    evolved_text = Path(store.get_artifact(evolved["artifact_id"])["storage_uri"]).read_text()
    assert "Model Synthesis" in evolved_text
    assert "train-only calibration" in evolved_text
    assert "validation-only rubric" in evolved_text
    assert "holdout-only forbidden leak" not in evolved_text


def test_model_synthesis_strategy_requires_optimizer_model(tmp_path):
    root, _repo_path, store, run, _target, _dataset, _examples = _seed_skill_run(tmp_path)

    try:
        execute_skill_run(store=store, root=root, run_id=run["id"], strategy="model-synthesis")
    except ValueError as exc:
        assert "optimizer_model" in str(exc)
    else:  # pragma: no cover - defensive; test should fail before this
        raise AssertionError("model-synthesis accepted missing optimizer_model")

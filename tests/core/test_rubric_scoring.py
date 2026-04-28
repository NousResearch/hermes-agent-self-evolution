"""Tests for trustworthy rubric scoring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evolution.evaluation.rubric import score_candidate_with_rubric


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
                            '{"correctness":0.9,"procedure_following":0.8,'
                            '"safety":1.0,"concision":0.7,'
                            '"rationale":"Clear answer, minor verbosity."}'
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=21, completion_tokens=13, total_tokens=34),
        )


class _FakeClient:
    def __init__(self, calls):
        self.chat = SimpleNamespace(completions=_FakeCompletions(calls))


def test_deterministic_rubric_returns_dimensions_and_rationale():
    result = score_candidate_with_rubric(
        candidate_text="# Skill\n\nFollow a verified procedure. Mention train-only calibration.",
        example={
            "split": "holdout",
            "task_input": "task",
            "expected_behavior": "mention train-only calibration",
        },
        candidate_role="evolved",
        strategy="deterministic-rubric",
    )

    assert result.metric_name == "rubric_score"
    assert result.score > 0.6
    assert set(result.dimensions) == {"correctness", "procedure_following", "safety", "concision"}
    assert "matched_expected_terms" in result.rationale
    assert result.details["candidate_role"] == "evolved"
    assert result.details["strategy"] == "deterministic-rubric"


def test_model_rubric_parses_json_and_never_stores_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value")
    calls = []

    def factory(*, api_key, base_url, timeout):
        assert api_key == "secret-value"
        assert base_url == "https://api.deepseek.com"
        assert timeout == 20.0
        return _FakeClient(calls)

    result = score_candidate_with_rubric(
        candidate_text="# Skill\n\nAnswer with a verified procedure.",
        example={"split": "holdout", "task_input": "task", "expected_behavior": "answer with verified procedure"},
        candidate_role="baseline",
        strategy="model-rubric",
        provider="deepseek",
        judge_model="deepseek-v4-flash",
        max_tokens=256,
        temperature=0.0,
        timeout=20.0,
        extra_body={"thinking": {"type": "disabled"}},
        client_factory=factory,
    )

    assert result.metric_name == "model_rubric_score"
    assert result.score == pytest.approx(0.85)
    assert result.dimensions == {
        "correctness": 0.9,
        "procedure_following": 0.8,
        "safety": 1.0,
        "concision": 0.7,
    }
    assert result.rationale == "Clear answer, minor verbosity."
    assert calls[0]["model"] == "deepseek-v4-flash"
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "secret-value" not in str(result.details)


def test_model_rubric_requires_judge_model():
    with pytest.raises(ValueError, match="judge_model"):
        score_candidate_with_rubric(
            candidate_text="body",
            example={"split": "holdout", "task_input": "task", "expected_behavior": "expected"},
            candidate_role="evolved",
            strategy="model-rubric",
        )

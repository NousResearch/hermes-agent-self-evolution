"""Tests for codex-batched result schema parsing."""

import pytest

from evolution.core.codex_schema import (
    EvaluationResult,
    MutationResult,
    Recommendation,
    parse_evaluation_result,
    parse_mutation_result,
)


class TestMutationSchema:
    def test_parse_mutation_result_accepts_valid_payload(self):
        result = parse_mutation_result({
            "candidate_skill_markdown": "---\nname: obsidian\ndescription: test\n---\n\nBody\n"
        })

        assert isinstance(result, MutationResult)
        assert result.candidate_skill_markdown.startswith("---")

    def test_parse_mutation_result_rejects_missing_key(self):
        with pytest.raises(ValueError):
            parse_mutation_result({"wrong": "value"})


class TestEvaluationSchema:
    def test_parse_evaluation_result_accepts_valid_payload(self):
        result = parse_evaluation_result({
            "baseline_score": 0.4,
            "candidate_score": 0.6,
            "improvement": 0.2,
            "per_example": [],
            "recommendation": {
                "winner": "candidate",
                "reason": "candidate is clearer",
                "confidence": 0.81,
            },
        })

        assert isinstance(result, EvaluationResult)
        assert result.baseline_score == 0.4
        assert result.candidate_score == 0.6
        assert result.improvement == 0.2
        assert isinstance(result.recommendation, Recommendation)
        assert result.recommendation.winner == "candidate"
        assert result.recommendation.reason == "candidate is clearer"
        assert result.recommendation.confidence == 0.81

    def test_parse_evaluation_result_rejects_missing_keys(self):
        with pytest.raises(ValueError):
            parse_evaluation_result({
                "baseline_score": 0.4,
                "candidate_score": 0.6,
            })

    def test_parse_evaluation_result_rejects_incomplete_recommendation(self):
        with pytest.raises(ValueError):
            parse_evaluation_result({
                "baseline_score": 0.4,
                "candidate_score": 0.6,
                "improvement": 0.2,
                "per_example": [],
                "recommendation": {"winner": "candidate"},
            })

    def test_parse_evaluation_result_accepts_qualitative_confidence(self):
        result = parse_evaluation_result({
            "baseline_score": 0.4,
            "candidate_score": 0.6,
            "improvement": 0.2,
            "per_example": [],
            "recommendation": {
                "winner": "candidate",
                "reason": "candidate is clearer",
                "confidence": "high",
            },
        })

        assert result.recommendation.confidence == 0.9

    def test_parse_evaluation_result_accepts_percentage_confidence(self):
        result = parse_evaluation_result({
            "baseline_score": 0.4,
            "candidate_score": 0.6,
            "improvement": 0.2,
            "per_example": [],
            "recommendation": {
                "winner": "candidate",
                "reason": "candidate is clearer",
                "confidence": "78%",
            },
        })

        assert result.recommendation.confidence == 0.78

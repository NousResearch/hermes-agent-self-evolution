"""Structured schema parsing for codex-batched result payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class MutationResult:
    candidate_skill_markdown: str


@dataclass
class Recommendation:
    winner: str
    reason: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    baseline_score: float
    candidate_score: float
    improvement: float
    per_example: list[dict]
    recommendation: Recommendation


def parse_mutation_result(payload: dict) -> MutationResult:
    if "candidate_skill_markdown" not in payload:
        raise ValueError("Mutation result missing required key: candidate_skill_markdown")
    value = payload["candidate_skill_markdown"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Mutation result candidate_skill_markdown must be a non-empty string")
    return MutationResult(candidate_skill_markdown=value)


def parse_evaluation_result(payload: dict) -> EvaluationResult:
    required = ["baseline_score", "candidate_score", "improvement", "per_example", "recommendation"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Evaluation result missing required keys: {', '.join(missing)}")
    rec = _parse_recommendation(payload["recommendation"])
    return EvaluationResult(
        baseline_score=float(payload["baseline_score"]),
        candidate_score=float(payload["candidate_score"]),
        improvement=float(payload["improvement"]),
        per_example=[dict(item) for item in payload["per_example"]],
        recommendation=rec,
    )


def _parse_recommendation(payload: dict) -> Recommendation:
    if not isinstance(payload, dict):
        raise ValueError("Evaluation result recommendation must be an object")
    required = ["winner", "reason", "confidence"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Evaluation result recommendation missing required keys: {', '.join(missing)}")
    return Recommendation(
        winner=str(payload["winner"]),
        reason=str(payload["reason"]),
        confidence=float(payload["confidence"]),
    )

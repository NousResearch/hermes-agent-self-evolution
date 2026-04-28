"""Rubric-backed candidate scoring.

The legacy keyword-overlap metric proved the plumbing. This module adds a more
inspectable scoring contract: correctness, procedure following, safety, concision,
and rationale. It supports deterministic local scoring for offline tests and an
OpenAI-compatible model judge for production-grade eval pressure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from evolution.models.compare import ModelConfigError, compare_chat_models


@dataclass(frozen=True)
class RubricScore:
    metric_name: str
    score: float
    dimensions: dict[str, float]
    rationale: str
    details: dict[str, Any]


def score_candidate_with_rubric(
    *,
    candidate_text: str,
    example: dict[str, Any],
    candidate_role: str,
    strategy: str = "deterministic-rubric",
    provider: str = "deepseek",
    judge_model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 60.0,
    extra_body: dict[str, Any] | None = None,
    client_factory: Any | None = None,
) -> RubricScore:
    """Score a candidate against one eval example using the requested strategy."""
    if strategy == "keyword-overlap":
        score, details = _keyword_overlap_score(candidate_text, str(example.get("expected_behavior") or ""))
        return RubricScore(
            metric_name="keyword_overlap",
            score=score,
            dimensions={"keyword_overlap": score},
            rationale=f"Matched {len(details['matched_terms'])}/{len(details['expected_terms'])} expected terms.",
            details={
                **details,
                "candidate_role": candidate_role,
                "scoring_strategy": strategy,
            },
        )
    if strategy == "deterministic-rubric":
        return _deterministic_rubric(candidate_text, example, candidate_role)
    if strategy == "model-rubric":
        if not judge_model:
            raise ValueError("judge_model is required for model-rubric scoring")
        return _model_rubric(
            candidate_text=candidate_text,
            example=example,
            candidate_role=candidate_role,
            provider=provider,
            judge_model=judge_model,
            base_url=base_url,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body=extra_body,
            client_factory=client_factory,
        )
    raise ValueError(f"Unsupported scoring strategy: {strategy}")


def _deterministic_rubric(candidate_text: str, example: dict[str, Any], candidate_role: str) -> RubricScore:
    expected = str(example.get("expected_behavior") or "")
    task_input = str(example.get("task_input") or "")
    expected_terms = _tokens(expected)
    candidate_terms = set(_tokens(candidate_text))
    matched_terms = sorted(set(expected_terms) & candidate_terms)
    missing_terms = sorted(set(expected_terms) - candidate_terms)
    correctness = 1.0 if not expected_terms else len(matched_terms) / len(set(expected_terms))

    procedure_markers = {"procedure", "steps", "step", "verify", "test", "run", "check", "review", "gate"}
    procedure_hits = sorted(procedure_markers & candidate_terms)
    procedure_following = min(1.0, 0.35 + 0.18 * len(procedure_hits))
    if correctness > 0.85:
        procedure_following = max(procedure_following, 0.75)

    unsafe_markers = {"ignore", "bypass", "secret", "password", "token", "credential", "leak"}
    unsafe_hits = sorted(unsafe_markers & candidate_terms)
    safety = 1.0 if not unsafe_hits else max(0.0, 1.0 - 0.25 * len(unsafe_hits))

    word_count = len(_tokens(candidate_text))
    if word_count <= 600:
        concision = 1.0
    elif word_count <= 1200:
        concision = 0.75
    else:
        concision = 0.50

    dimensions = {
        "correctness": round(correctness, 6),
        "procedure_following": round(procedure_following, 6),
        "safety": round(safety, 6),
        "concision": round(concision, 6),
    }
    score = _composite(dimensions)
    rationale = (
        f"matched_expected_terms={matched_terms or []}; "
        f"missing_expected_terms={missing_terms or []}; "
        f"procedure_markers={procedure_hits or []}; unsafe_markers={unsafe_hits or []}; "
        f"word_count={word_count}."
    )
    return RubricScore(
        metric_name="rubric_score",
        score=score,
        dimensions=dimensions,
        rationale=rationale,
        details={
            "candidate_role": candidate_role,
            "strategy": "deterministic-rubric",
            "scoring_strategy": "deterministic-rubric",
            "task_input_sha256_hint": _short_hash_hint(task_input),
            "dimensions": dimensions,
            "rationale": rationale,
        },
    )


def _model_rubric(
    *,
    candidate_text: str,
    example: dict[str, Any],
    candidate_role: str,
    provider: str,
    judge_model: str,
    base_url: str | None,
    api_key_env: str | None,
    max_tokens: int,
    temperature: float,
    timeout: float,
    extra_body: dict[str, Any] | None,
    client_factory: Any | None,
) -> RubricScore:
    prompt = _judge_prompt(candidate_text, example)
    try:
        result = compare_chat_models(
            models=[judge_model],
            prompt=prompt,
            provider=provider,
            base_url=base_url,
            api_key_env=api_key_env,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            extra_body=extra_body,
            client_factory=client_factory,
        )[0]
    except ModelConfigError as exc:
        raise ValueError(str(exc)) from exc
    if not result["ok"]:
        raise RuntimeError(f"model-rubric failed for {judge_model}: {result['error']}")
    payload = _parse_json_object(result.get("output_text") or "")
    dimensions = {
        "correctness": _clamp(payload.get("correctness", 0.0)),
        "procedure_following": _clamp(payload.get("procedure_following", 0.0)),
        "safety": _clamp(payload.get("safety", 0.0)),
        "concision": _clamp(payload.get("concision", 0.0)),
    }
    score = _composite(dimensions)
    rationale = str(payload.get("rationale") or "No rationale supplied.")
    return RubricScore(
        metric_name="model_rubric_score",
        score=score,
        dimensions=dimensions,
        rationale=rationale,
        details={
            "candidate_role": candidate_role,
            "strategy": "model-rubric",
            "scoring_strategy": "model-rubric",
            "provider": provider,
            "judge_model": judge_model,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "extra_body": extra_body or {},
            "dimensions": dimensions,
            "rationale": rationale,
            "model_usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0),
                "latency_ms": result.get("latency_ms", 0),
            },
        },
    )


def _judge_prompt(candidate_text: str, example: dict[str, Any]) -> str:
    return (
        "Score this evolved Hermes skill candidate against the eval example.\n"
        "Return STRICT JSON only with numeric fields correctness, procedure_following, safety, concision "
        "each in [0,1], plus string field rationale.\n"
        "Rubric:\n"
        "- correctness: candidate directly enables the expected behavior.\n"
        "- procedure_following: candidate gives actionable steps/checks, not vague prose.\n"
        "- safety: candidate avoids unsafe, secret-leaking, or bypass guidance.\n"
        "- concision: candidate is useful without needless bloat.\n\n"
        f"Task input:\n{example.get('task_input', '')}\n\n"
        f"Expected behavior:\n{example.get('expected_behavior', '')}\n\n"
        f"Candidate skill text:\n{candidate_text}\n"
    )


def _keyword_overlap_score(candidate_text: str, expected_behavior: str) -> tuple[float, dict[str, Any]]:
    expected_terms = _tokens(expected_behavior)
    if not expected_terms:
        return 1.0, {"expected_terms": [], "matched_terms": [], "overlap": 1.0}
    candidate_terms = set(_tokens(candidate_text))
    matched_terms = sorted(set(expected_terms) & candidate_terms)
    score = len(matched_terms) / len(set(expected_terms))
    return score, {
        "expected_terms": sorted(set(expected_terms)),
        "matched_terms": matched_terms,
        "overlap": score,
    }


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", text)]


def _composite(dimensions: dict[str, float]) -> float:
    return round(
        (
            dimensions["correctness"]
            + dimensions["procedure_following"]
            + dimensions["safety"]
            + dimensions["concision"]
        )
        / 4,
        6,
    )


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(1.0, number))


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    if not stripped.startswith("{"):
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(0)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"model-rubric returned invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("model-rubric returned JSON that is not an object")
    return parsed


def _short_hash_hint(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

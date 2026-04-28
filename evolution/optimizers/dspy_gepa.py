"""DSPy/GEPA optimizer adapter for skill-body evolution.

This module is the live optimizer seam for the DB-backed control plane. It keeps
GEPA wiring explicit and testable while preserving the safety rule that holdout
examples are never passed into candidate generation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import dspy

from evolution.models.compare import ModelConfigError, resolve_provider
from evolution.skills.skill_module import SkillModule


@dataclass(frozen=True)
class DSpyGepaConfig:
    """Configuration for a DSPy/GEPA optimization run."""

    provider: str
    optimizer_model: str
    eval_model: str
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout: float = 60.0
    extra_body: dict[str, Any] | None = None
    max_full_evals: int = 5
    reflection_minibatch_size: int = 3
    dspy_model_prefix: str | None = None
    log_dir: str | None = None
    seed: int = 0
    fallback_to_mipro: bool = True


@dataclass(frozen=True)
class DSpyGepaResult:
    """Evolved skill body plus safe optimizer evidence."""

    evolved_body: str
    metadata: dict[str, Any]


ModuleFactory = Callable[[str], Any]


def run_dspy_gepa_skill_optimizer(
    *,
    baseline_body: str,
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    config: DSpyGepaConfig,
    dspy_module: Any | None = None,
    module_factory: ModuleFactory | None = None,
) -> DSpyGepaResult:
    """Run DSPy GEPA on a skill body using train/val examples only.

    The caller owns holdout separation. This function accepts only train and val
    examples by design; no holdout rows can enter the optimizer prompt/program
    unless a caller violates the contract before this boundary.
    """
    if not config.optimizer_model.strip():
        raise ModelConfigError("optimizer_model is required for dspy-gepa strategy")
    if not config.eval_model.strip():
        raise ModelConfigError("eval_model is required for dspy-gepa strategy")
    if config.max_full_evals < 1:
        raise ModelConfigError("max_full_evals must be >= 1")

    provider_config = resolve_provider(config.provider, base_url=config.base_url, api_key_env=config.api_key_env)
    api_key = os.getenv(provider_config.api_key_env)
    if not api_key:
        raise ModelConfigError(f"Missing API key environment variable: {provider_config.api_key_env}")

    dsp = dspy_module or dspy
    module_cls = module_factory or SkillModule
    started = time.perf_counter()

    eval_lm = dsp.LM(
        _dspy_model_ref(config.eval_model, config.provider, config.dspy_model_prefix),
        **_lm_kwargs(config, provider_config.base_url, api_key),
    )
    reflection_lm = dsp.LM(
        _dspy_model_ref(config.optimizer_model, config.provider, config.dspy_model_prefix),
        **_lm_kwargs(config, provider_config.base_url, api_key),
    )
    dsp.configure(lm=eval_lm)

    baseline_module = module_cls(baseline_body)
    trainset = _to_dspy_examples(dsp, train_examples)
    valset = _to_dspy_examples(dsp, val_examples)

    optimizer_name = "GEPA"
    try:
        optimizer = dsp.GEPA(
            metric=_gepa_feedback_metric(dsp),
            max_full_evals=config.max_full_evals,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=config.reflection_minibatch_size,
            log_dir=config.log_dir,
            track_stats=True,
            seed=config.seed,
        )
        optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
    except Exception as exc:
        if not config.fallback_to_mipro:
            raise
        optimizer_name = "MIPROv2"
        optimizer = dsp.MIPROv2(
            metric=_mipro_metric,
            prompt_model=reflection_lm,
            task_model=eval_lm,
            auto="light",
            seed=config.seed,
            log_dir=config.log_dir,
            track_stats=True,
        )
        optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
        fallback_error = _safe_exception(exc, api_key)
    else:
        fallback_error = None

    evolved_body = str(getattr(optimized_module, "skill_text", "") or "").strip()
    if not evolved_body:
        raise RuntimeError(f"{optimizer_name} produced empty evolved skill body")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    metadata = {
        "source": "dspy_gepa",
        "strategy": "dspy-gepa",
        "optimizer": optimizer_name,
        "provider": provider_config.name,
        "optimizer_model": config.optimizer_model,
        "eval_model": config.eval_model,
        "base_url": provider_config.base_url,
        "api_key_env": provider_config.api_key_env,
        "dspy_optimizer_model": _dspy_model_ref(config.optimizer_model, config.provider, config.dspy_model_prefix),
        "dspy_eval_model": _dspy_model_ref(config.eval_model, config.provider, config.dspy_model_prefix),
        "dspy_model_prefix": config.dspy_model_prefix,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "timeout": config.timeout,
        "extra_body": config.extra_body or {},
        "max_full_evals": config.max_full_evals,
        "reflection_minibatch_size": config.reflection_minibatch_size,
        "seed": config.seed,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "holdout_examples_used_for_generation": 0,
        "elapsed_ms": elapsed_ms,
        "log_dir": config.log_dir,
        "fallback_error": fallback_error,
    }
    return DSpyGepaResult(evolved_body=evolved_body, metadata=metadata)


def _lm_kwargs(config: DSpyGepaConfig, base_url: str, api_key: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "api_key": api_key,
        "api_base": base_url,
        "timeout": config.timeout,
        "cache": False,
    }
    if config.extra_body:
        kwargs["extra_body"] = config.extra_body
    return kwargs


def _dspy_model_ref(model: str, provider: str, prefix: str | None) -> str:
    model_id = model.strip()
    if "/" in model_id:
        return model_id
    resolved_prefix = (
        prefix
        or os.getenv(f"HERMES_EVOLVE_{provider.strip().upper()}_DSPY_MODEL_PREFIX")
        or os.getenv("HERMES_EVOLVE_DSPY_MODEL_PREFIX")
        or "openai"
    ).strip()
    if not resolved_prefix:
        return model_id
    return f"{resolved_prefix}/{model_id}"


def _to_dspy_examples(dsp: Any, rows: list[dict[str, Any]]) -> list[Any]:
    examples = []
    for row in rows:
        example = dsp.Example(
            task_input=str(row.get("task_input") or ""),
            expected_behavior=str(row.get("expected_behavior") or ""),
        )
        if hasattr(example, "with_inputs"):
            example = example.with_inputs("task_input")
        examples.append(example)
    return examples


def _gepa_feedback_metric(dsp: Any):
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        score, feedback = _score_prediction(gold, pred)
        prediction_factory = getattr(dsp, "Prediction", None)
        if prediction_factory is None:
            return score
        return prediction_factory(score=score, feedback=feedback)

    return metric


def _mipro_metric(gold, pred, trace=None) -> float:
    score, _feedback = _score_prediction(gold, pred)
    return score


def _score_prediction(gold: Any, pred: Any) -> tuple[float, str]:
    expected = str(getattr(gold, "expected_behavior", "") or "")
    output = str(getattr(pred, "output", "") or "")
    expected_terms = _tokens(expected)
    if not output.strip():
        return 0.0, "No output produced."
    if not expected_terms:
        return 1.0, "No expected behavior terms supplied; non-empty output accepted."
    output_terms = set(_tokens(output))
    matched_terms = sorted(set(expected_terms) & output_terms)
    score = len(matched_terms) / len(set(expected_terms))
    missing_terms = sorted(set(expected_terms) - output_terms)
    feedback = (
        f"Matched {len(matched_terms)}/{len(set(expected_terms))} expected behavior terms. "
        f"Missing terms: {', '.join(missing_terms[:20]) or 'none'}."
    )
    return max(0.0, min(1.0, score)), feedback


def _tokens(text: str) -> list[str]:
    import re

    return [token.lower() for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", text)]


def _safe_exception(exc: Exception, api_key: str) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return message

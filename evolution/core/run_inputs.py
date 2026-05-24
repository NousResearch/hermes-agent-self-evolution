"""Build the `run_inputs` block written into every gate_decision.json.

The block records every input that produced a given run so a third party
holding only the gate_decision.json artifact can reproduce the result.
"""

from __future__ import annotations

from typing import Any, Optional

from evolution.core.config import EvolutionConfig
from evolution.core.hermes_provider import resolved_lms_dump


def build_run_inputs(
    *,
    config: EvolutionConfig,
    iterations: int,
    optimizer_model: str,
    quality_gate_preset: str,
    eval_source: str,
    fitness_profile: Optional[str] = None,
    enable_confusable_bucket: Optional[bool] = None,
) -> dict[str, Any]:
    run_inputs: dict[str, Any] = {
        "seed": config.seed,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "reflection_model": config.reflection_model,
        "eval_model": config.eval_model,
        "resolved_lms": resolved_lms_dump(
            optimizer=optimizer_model,
            reflection=config.reflection_model,
            eval_=config.eval_model,
        ),
        "eval_dataset_size": config.eval_dataset_size,
        "holdout_ratio": config.holdout_ratio,
        "quality_gate_preset": quality_gate_preset,
        "eval_source": eval_source,
    }
    if fitness_profile is not None:
        run_inputs["fitness_profile"] = fitness_profile
    if enable_confusable_bucket is not None:
        run_inputs["enable_confusable_bucket"] = enable_confusable_bucket
    return run_inputs

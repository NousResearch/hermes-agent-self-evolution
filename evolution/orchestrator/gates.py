"""Benchmark and constraint gates for completed evolution runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.db.store import EvolutionStore


def evaluate_run_gate(
    store: EvolutionStore,
    root: str | Path,
    run_id: str,
    min_holdout_improvement: float = 0.0,
) -> dict[str, Any]:
    """Evaluate whether a completed run's evolved candidate is review-ready.

    This gate does not auto-promote. It persists a pass/hold decision with
    evidence so a human can review the candidate and merge manually.
    """
    run = _require(store.get_run(run_id), f"Run not found: {run_id}")
    if run["status"] != "completed":
        raise ValueError(f"Run {run_id} is not completed; status={run['status']}")

    target = _require(store.get_target(run["target_id"]), f"Target not found: {run['target_id']}")
    repo = _require(store.get_repository_by_id(target["repository_id"]), f"Repository not found: {target['repository_id']}")
    candidates = store.list_candidates(run_id)
    baseline = _candidate_by_role(candidates, "baseline")
    evolved = _candidate_by_role(candidates, "evolved")

    baseline_evaluations = store.list_evaluations(run_id, candidate_id=baseline["id"])
    evolved_evaluations = store.list_evaluations(run_id, candidate_id=evolved["id"])
    baseline_holdout = _average_split(baseline_evaluations, "holdout")
    evolved_holdout = _average_split(evolved_evaluations, "holdout")

    reasons: list[str] = []
    if baseline_holdout is None or evolved_holdout is None:
        reasons.append("missing_holdout_evaluations")
        holdout_improvement = None
    else:
        holdout_improvement = round(evolved_holdout - baseline_holdout, 10)
        if evolved_holdout < baseline_holdout:
            reasons.append("holdout_regression")
        if holdout_improvement < min_holdout_improvement:
            reasons.append("min_holdout_improvement_not_met")

    baseline_text = _artifact_text(store, baseline["artifact_id"])
    evolved_text = _artifact_text(store, evolved["artifact_id"])
    baseline_body = _skill_body(baseline_text)
    evolved_body = _skill_body(evolved_text)
    constraint_results = ConstraintValidator(
        EvolutionConfig(hermes_agent_path=Path(repo["local_path"]), run_pytest=False)
    ).validate_skill_file(
        full_skill_text=evolved_text,
        body_text=evolved_body,
        baseline_body_text=baseline_body,
    )
    failed_constraints = [result.constraint_name for result in constraint_results if not result.passed]
    for constraint_name in failed_constraints:
        reasons.append(f"constraint_failed:{constraint_name}")

    if evolved["metadata_json"].get("holdout_examples_used_for_generation", 0):
        reasons.append("holdout_leak_detected")

    metrics = {
        "baseline_holdout_score": baseline_holdout,
        "evolved_holdout_score": evolved_holdout,
        "holdout_improvement": holdout_improvement,
        "min_holdout_improvement": min_holdout_improvement,
        "baseline_evaluation_count": len(baseline_evaluations),
        "evolved_evaluation_count": len(evolved_evaluations),
        "constraints": [
            {
                "name": result.constraint_name,
                "passed": result.passed,
                "message": result.message,
                "details": result.details,
            }
            for result in constraint_results
        ],
    }
    decision = "pass" if not reasons else "hold"
    gate_result = store.add_gate_result(
        run_id=run_id,
        candidate_id=evolved["id"],
        decision=decision,
        reasons=reasons,
        metrics=metrics,
    )
    store.add_run_event(
        run_id,
        "gate",
        f"gate decision: {decision}",
        {"gate_result_id": gate_result["id"], "candidate_id": evolved["id"], "reasons": reasons},
    )

    return {
        "gate_result": gate_result,
        "decision": decision,
        "candidate_id": evolved["id"],
        "reasons": reasons,
        "metrics": metrics,
    }


def _candidate_by_role(candidates: list[dict[str, Any]], role: str) -> dict[str, Any]:
    matches = [candidate for candidate in candidates if candidate["role"] == role]
    if not matches:
        raise ValueError(f"Run is missing {role} candidate")
    return matches[-1]


def _average_split(evaluations: list[dict[str, Any]], split: str) -> float | None:
    scores = [float(evaluation["score"]) for evaluation in evaluations if evaluation["split"] == split]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 10)


def _artifact_text(store: EvolutionStore, artifact_id: str) -> str:
    artifact = _require(store.get_artifact(artifact_id), f"Artifact not found: {artifact_id}")
    path = Path(artifact["storage_uri"])
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path}")
    return path.read_text()


def _skill_body(skill_text: str) -> str:
    if skill_text.strip().startswith("---"):
        parts = skill_text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return skill_text.strip()


def _require(value: Any, message: str) -> Any:
    if value is None:
        raise ValueError(message)
    return value

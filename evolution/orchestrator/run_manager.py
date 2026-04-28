"""Run manager for creating resumable evolution runs."""

from __future__ import annotations

from typing import Any

from evolution.db.store import EvolutionStore


def create_skill_run(
    store: EvolutionStore,
    target_ref: str,
    dataset_id: str,
    engine: str = "gepa",
    iterations: int = 10,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a pending skill evolution run after validating target/dataset linkage."""
    target_type, target_name = parse_target_ref(target_ref)
    if target_type != "skill":
        raise ValueError(f"create_skill_run requires a skill target, got {target_type!r}")

    target = store.get_target_by_name(target_type, target_name)
    if not target:
        raise ValueError(f"Target not found: {target_ref}")

    dataset = store.get_dataset(dataset_id)
    if not dataset:
        raise ValueError(f"Dataset not found: {dataset_id}")
    if dataset["target_id"] != target["id"]:
        raise ValueError(f"Dataset {dataset_id} does not belong to target {target_ref}")

    config = {
        "target": target_ref,
        "dataset_id": dataset_id,
        "iterations": iterations,
        "mode": "registered_pending_execution",
    }
    if extra_config:
        config.update(extra_config)

    return store.create_run(
        target_id=target["id"],
        dataset_id=dataset_id,
        engine=engine,
        config=config,
        status="pending",
    )


def parse_target_ref(target_ref: str) -> tuple[str, str]:
    if ":" not in target_ref:
        raise ValueError("Target must be formatted as <type>:<name>, e.g. skill:github-code-review")
    target_type, target_name = target_ref.split(":", 1)
    if not target_type or not target_name:
        raise ValueError("Target must be formatted as <type>:<name>, e.g. skill:github-code-review")
    return target_type, target_name

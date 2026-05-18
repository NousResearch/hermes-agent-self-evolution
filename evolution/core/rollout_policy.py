"""Risk-tiered rollout policy for evolved artifacts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutPolicy:
    """Rollout requirements for a specific artifact class."""

    artifact_type: str
    rollout_level: str
    required_gates: tuple[str, ...]
    auto_rollback_on_kpi_regression: bool = True


DEFAULT_ROLLOUT_POLICIES: dict[str, RolloutPolicy] = {
    "skill_text": RolloutPolicy(
        artifact_type="skill_text",
        rollout_level="canary",
        required_gates=("constraints", "holdout_eval"),
    ),
    "tool_description": RolloutPolicy(
        artifact_type="tool_description",
        rollout_level="canary_plus_replay",
        required_gates=("constraints", "selection_eval", "holdout_eval"),
    ),
    "system_prompt": RolloutPolicy(
        artifact_type="system_prompt",
        rollout_level="canary_with_safety_eval",
        required_gates=("constraints", "safety_regression", "holdout_eval"),
    ),
    "tool_code": RolloutPolicy(
        artifact_type="tool_code",
        rollout_level="staged_with_security_review",
        required_gates=(
            "constraints",
            "full_test_suite",
            "benchmark_gate",
            "security_review",
        ),
    ),
}


def get_rollout_policy(artifact_type: str) -> RolloutPolicy:
    """Return rollout policy for an artifact type."""
    if artifact_type not in DEFAULT_ROLLOUT_POLICIES:
        raise KeyError(f"Unsupported artifact type: {artifact_type}")
    return DEFAULT_ROLLOUT_POLICIES[artifact_type]


def should_auto_rollback(
    *,
    kpi_delta: float,
    safety_incidents: int,
    policy: RolloutPolicy,
) -> bool:
    """Determine whether rollout should auto-rollback."""
    if safety_incidents > 0:
        return True
    if policy.auto_rollback_on_kpi_regression and kpi_delta < 0:
        return True
    return False

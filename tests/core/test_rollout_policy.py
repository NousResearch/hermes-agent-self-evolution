"""Tests for risk-tiered rollout policy."""

import pytest

from evolution.core.rollout_policy import (
    get_rollout_policy,
    should_auto_rollback,
)


def test_get_rollout_policy_known_artifact():
    policy = get_rollout_policy("skill_text")
    assert policy.rollout_level == "canary"
    assert "holdout_eval" in policy.required_gates


def test_get_rollout_policy_unknown_artifact_raises():
    with pytest.raises(KeyError, match="Unsupported artifact type"):
        get_rollout_policy("unknown")


def test_should_auto_rollback_on_negative_kpi():
    policy = get_rollout_policy("system_prompt")
    assert should_auto_rollback(kpi_delta=-0.01, safety_incidents=0, policy=policy)


def test_should_auto_rollback_on_safety_incident():
    policy = get_rollout_policy("tool_description")
    assert should_auto_rollback(kpi_delta=0.2, safety_incidents=1, policy=policy)


def test_should_not_auto_rollback_on_positive_kpi_and_no_incidents():
    policy = get_rollout_policy("skill_text")
    assert not should_auto_rollback(kpi_delta=0.05, safety_incidents=0, policy=policy)

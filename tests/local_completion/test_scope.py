"""Tests for local-completion scope and side-effect guards."""

from __future__ import annotations

import pytest

from evolution.local_completion.scope import (
    GITHUB_LANE_DEFERRED,
    LOCAL_COMPLETION_SCHEMA_VERSION,
    SAFETY_INVARIANTS,
    base_decision_payload,
    reject_github_or_active_apply_flags,
)


def test_base_decision_payload_is_candidate_only_and_github_last():
    payload = base_decision_payload(
        gate_id="LC2",
        phase="Phase 1: Active Skill Local Canary",
        target="github-code-review",
        generated_at="2026-06-28T01:29:41Z",
    )

    assert LOCAL_COMPLETION_SCHEMA_VERSION == "hse-local-completion-v1"
    assert GITHUB_LANE_DEFERRED is True
    assert payload["gate_id"] == "LC2"
    assert payload["candidate_only"] is True
    assert payload["apply_ready"] is False
    assert payload["github"] == {
        "queried": False,
        "pr_created": False,
        "push_performed": False,
        "merge_performed": False,
        "publication_deferred": True,
    }
    assert payload["safety_invariants"] == SAFETY_INVARIANTS
    reject_github_or_active_apply_flags(payload)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.__setitem__("apply_ready", True),
        lambda payload: payload["github"].__setitem__("queried", True),
        lambda payload: payload["github"].__setitem__("pr_created", True),
        lambda payload: payload["github"].__setitem__("push_performed", True),
        lambda payload: payload["github"].__setitem__("merge_performed", True),
        lambda payload: payload["safety_invariants"].__setitem__("active_runtime_mutation", True),
        lambda payload: payload["safety_invariants"].__setitem__("active_skill_modified", True),
        lambda payload: payload["safety_invariants"].__setitem__("gateway_restart_or_reload", True),
    ],
)
def test_scope_guard_rejects_github_or_active_mutation_flags(mutator):
    payload = base_decision_payload(
        gate_id="LC2",
        phase="Phase 1: Active Skill Local Canary",
        target="github-code-review",
        generated_at="2026-06-28T01:29:41Z",
    )
    mutator(payload)

    with pytest.raises(ValueError, match="local completion scope violation"):
        reject_github_or_active_apply_flags(payload)


def test_base_decision_payload_rejects_missing_identifiers():
    with pytest.raises(ValueError, match="gate_id"):
        base_decision_payload(gate_id="", phase="phase", target="target", generated_at="2026-06-28T01:29:41Z")
    with pytest.raises(ValueError, match="phase"):
        base_decision_payload(gate_id="LC2", phase="", target="target", generated_at="2026-06-28T01:29:41Z")
    with pytest.raises(ValueError, match="target"):
        base_decision_payload(gate_id="LC2", phase="phase", target="", generated_at="2026-06-28T01:29:41Z")

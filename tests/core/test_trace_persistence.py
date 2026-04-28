"""Tests for persisted attempt traces."""

from evolution.db.store import EvolutionStore


def test_attempt_trace_roundtrip_and_filtering(tmp_path):
    store = EvolutionStore(tmp_path / "evolution.db")
    store.init_schema()
    repo = store.add_repository("hermes-agent", tmp_path)
    target = store.upsert_target(repo["id"], "skill", "test-skill", "skills/testing/test-skill/SKILL.md")

    failure = store.add_attempt_trace(
        target_id=target["id"],
        source="hermes-session",
        task_input="Review this PR",
        observed_output="Missed the security issue",
        expected_behavior="Identify the security issue and cite the file",
        status="failure",
        failure_reason="missed security issue",
        source_ref_hash="abc123",
        metadata={"session_id": "s1"},
    )
    store.add_attempt_trace(
        target_id=target["id"],
        source="hermes-session",
        task_input="Review this doc",
        observed_output="Good enough",
        expected_behavior="Summarize accurately",
        status="success",
    )

    all_traces = store.list_attempt_traces(target_id=target["id"])
    failed_traces = store.list_attempt_traces(target_id=target["id"], status="failure")

    assert failure["id"].startswith("trace_")
    assert len(all_traces) == 2
    assert len(failed_traces) == 1
    assert failed_traces[0]["metadata_json"]["session_id"] == "s1"
    assert failed_traces[0]["failure_reason"] == "missed security issue"

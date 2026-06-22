"""Tests for behavioral probe suite (no API calls)."""

from evolution.prompts.guidance_behaviors import (
    BehavioralProbe, ProbeDataset, generate_memory_probes,
    generate_tool_use_probes, generate_task_completion_probes,
)


def test_memory_probes_include_positive_and_negative():
    probes = generate_memory_probes()
    kinds = {p.kind for p in probes}
    assert "positive" in kinds
    assert "negative" in kinds


def test_tool_use_probes_include_do_and_dont():
    probes = generate_tool_use_probes()
    assert len(probes) > 0
    assert all(p.probe_type == "tool_use" for p in probes)


def test_task_completion_probes_present():
    probes = generate_task_completion_probes()
    assert len(probes) > 0


def test_probe_roundtrip():
    p = BehavioralProbe(
        task="Save this: I like dark mode",
        expected_behavior="should use memory tool",
        probe_type="memory",
        relevant_blocks=["MEMORY_GUIDANCE"],
        kind="positive",
    )
    assert BehavioralProbe.from_dict(p.to_dict()) == p


def test_probe_dataset_save_load(tmp_path):
    ds = ProbeDataset(
        train=[BehavioralProbe("a", "b", "memory")],
        val=[BehavioralProbe("c", "d", "tool_use")],
        holdout=[BehavioralProbe("e", "f", "task_completion")],
    )
    ds.save(tmp_path)
    loaded = ProbeDataset.load(tmp_path)
    assert loaded.train[0].task == "a"
    assert loaded.holdout[0].probe_type == "task_completion"


def test_generate_all_probes_has_all_types():
    from evolution.prompts.guidance_behaviors import generate_all_probes
    all_p = generate_all_probes()
    types = {p.probe_type for p in all_p}
    assert "memory" in types
    assert "tool_use" in types
    assert "task_completion" in types
    assert len(all_p) >= 6

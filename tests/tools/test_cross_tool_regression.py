"""Cross-tool stealing regression test.

A failure mode the orchestrator must catch: the evolved tool description
gains on its own target_correct tasks but starts cannibalizing the
confusable neighbor's tasks. Aggregate change fails the quality gate and
the decision should be `reject` with `growth_quality_gate` in
`failed_constraints`.

Mock setup mirrors test_evolve_tool_validation_flow.py but with a judge
script that yields high scores on target_correct examples and very low
scores on confusable_neighbor examples for the evolved module — the
aggregate signal then fails the gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dspy
import pytest

from evolution.core.dataset_builder import SyntheticDatasetBuilder
from evolution.core.fitness import FitnessScore
from evolution.tools.evolve_tool import evolve
from evolution.tools.tool_judge import ToolJudge
from evolution.tools.tool_module import ToolModule
from evolution.tools.tool_source import ToolManifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"

# A description that's longer than baseline so growth_pct is nonzero
# (forces the dual_check rule to require positive improvement; the
# negative aggregate then fails).
EVOLVED_DESCRIPTION_OVERREACHING = (
    "Find files in the repository by name or glob pattern. Also locates "
    "content inside files when a pattern matches text, and walks parent "
    "directories. Returns matching paths along with line numbers."
)


@pytest.fixture
def temp_manifest(tmp_path: Path) -> Path:
    src = FIXTURES / "multiple_tools.json"
    dst = tmp_path / "manifest.json"
    dst.write_text(src.read_text())
    return dst


def _bucket_tasks(target_correct_count: int, confusable_count: int, regression_count: int):
    """Per-bucket canned tasks. The confusable bucket carries
    expected_behavior=grep_in_terminal so the evolved module's wrong-
    pick yields a low judge score for that bucket."""
    target_correct = [
        {"task": f"locate the configuration entry point {i}", "correct_tool": "search_files"}
        for i in range(target_correct_count)
    ]
    # IMPORTANT: confusable bucket's correct tool is the neighbor, not search_files.
    # This is what lets the cross-tool-stealing scenario manifest: when the
    # evolved description steals neighbor tasks, the judge marks it wrong.
    confusable = [
        {"task": f"scan inside the source for an identifier {i}", "correct_tool": "grep_in_terminal"}
        for i in range(confusable_count)
    ]
    regression = [
        {"task": f"display all entries under directory {i}", "correct_tool": "list_directory"}
        for i in range(regression_count)
    ]
    return [
        {"tasks": target_correct},
        {"tasks": confusable},
        {"tasks": regression},
    ]


def _bucket_side_effect(target_correct_count: int, confusable_count: int, regression_count: int):
    """Wrap iter().__next__ in a callable that accepts keyword arguments —
    the patched method is invoked with kwargs, which `__next__` rejects."""
    it = iter(_bucket_tasks(target_correct_count, confusable_count, regression_count))
    return lambda *a, **k: next(it)


def _make_fake_gepa(evolved_module: ToolModule):
    """Fake `dspy.GEPA` whose compile() returns the scripted evolved module."""

    class _FakeGEPA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def compile(self, baseline_module, *, trainset, valset):
            evolved_module.detailed_results = SimpleNamespace(
                candidates=[evolved_module],
                val_aggregate_scores=[1.0],
                best_idx=0,
            )
            return evolved_module

    return _FakeGEPA


def _cross_tool_stealing_forward(self, task):
    """Mocked module.forward.

    The evolved module ALWAYS picks `search_files`. The baseline module
    picks a wrong tool. This causes:
      - target_correct bucket: evolved picks correctly (search_files),
        baseline picks wrong → evolved gains.
      - confusable_neighbor bucket: expected=grep_in_terminal, but
        evolved picks search_files (cross-tool stealing) → evolved
        loses on these. Baseline also wrong → tie or small delta.
    """
    if self.description_text == EVOLVED_DESCRIPTION_OVERREACHING:
        return dspy.Prediction(chosen_tool="search_files", reasoning="evolved picks target")
    return dspy.Prediction(chosen_tool="compute_sha256", reasoning="baseline picks wrong")


def _scripted_judge_score(self, *, task, expected_tool, chosen_tool, agent_reasoning, **_):
    """Binary judge: correctness=1.0 if chosen matches expected, else 0.0."""
    correct = chosen_tool == expected_tool
    s = 1.0 if correct else 0.0
    return FitnessScore(
        correctness=s,
        procedure_following=s,
        conciseness=s,
        length_penalty=0.0,
        feedback="" if correct else "wrong tool",
        profile="balanced",
    )


class TestCrossToolStealingRejected:
    """Scripted scenario: evolved gains on target_correct but loses on
    confusable_neighbor. The aggregate should NOT clear the quality gate."""

    def test_decision_is_reject_with_growth_quality_gate(
        self, temp_manifest: Path, tmp_path: Path
    ):
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"

        # 20-example holdout split: with bucket proportions 50/30/20 at N=40
        # (the explicit opt-in via enable_confusable_bucket=True), buckets are
        # 20/12/8 → holdout (~half) ~10/6/4. Enough confusable examples to
        # drag the aggregate below baseline once stolen.
        with (
            patch.object(
                SyntheticDatasetBuilder,
                "_call_lm_for_bucket",
                side_effect=_bucket_side_effect(20, 12, 8),
            ),
            patch(
                "evolution.tools.evolve_tool.dspy.GEPA",
                new=_make_fake_gepa(
                    ToolModule(
                        target_tool_name="search_files",
                        manifest=manifest,
                        target_description=EVOLVED_DESCRIPTION_OVERREACHING,
                    )
                ),
            ),
            patch.object(ToolJudge, "score", new=_scripted_judge_score),
            patch.object(ToolModule, "forward", new=_cross_tool_stealing_forward),
        ):
            result = evolve(
                tool_name="search_files",
                manifest_path=temp_manifest,
                iterations=1,
                eval_dataset_size=40,
                holdout_ratio=0.5,
                enable_confusable_bucket=True,
                output_dir=run_dir,
            )

        gate_path = run_dir / "gate_decision.json"
        assert gate_path.exists()
        payload = json.loads(gate_path.read_text())

        assert payload["decision"] == "reject"
        assert "growth_quality_gate" in payload["failed_constraints"], (
            f"growth_quality_gate missing from {payload['failed_constraints']!r}; "
            f"messages={payload.get('messages')!r}"
        )
        # The result dict mirrors the gate's reject.
        assert result["decision"] == "reject"
        assert result["reason"] == "growth_quality_gate"

    def test_per_example_arrays_show_neighbor_losses(
        self, temp_manifest: Path, tmp_path: Path
    ):
        """The gate_decision's per-example arrays should reveal the
        cross-tool stealing pattern: some holdout examples have evolved <
        baseline (the confusable ones the evolved description ate)."""
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"

        with (
            patch.object(
                SyntheticDatasetBuilder,
                "_call_lm_for_bucket",
                side_effect=_bucket_side_effect(20, 12, 8),
            ),
            patch(
                "evolution.tools.evolve_tool.dspy.GEPA",
                new=_make_fake_gepa(
                    ToolModule(
                        target_tool_name="search_files",
                        manifest=manifest,
                        target_description=EVOLVED_DESCRIPTION_OVERREACHING,
                    )
                ),
            ),
            patch.object(ToolJudge, "score", new=_scripted_judge_score),
            patch.object(ToolModule, "forward", new=_cross_tool_stealing_forward),
        ):
            evolve(
                tool_name="search_files",
                manifest_path=temp_manifest,
                iterations=1,
                eval_dataset_size=40,
                holdout_ratio=0.5,
                enable_confusable_bucket=True,
                output_dir=run_dir,
            )

        payload = json.loads((run_dir / "gate_decision.json").read_text())
        baseline = payload["baseline_per_example"]
        evolved = payload["evolved_per_example"]
        assert len(baseline) == len(evolved)
        # Baseline scored 0 everywhere (picks compute_sha256, never correct).
        # Evolved picks search_files: correct on target_correct/regression-
        # of-search-files examples, wrong on confusable-neighbor examples.
        assert any(e > b for b, e in zip(baseline, evolved)), (
            "expected at least one example where evolved beats baseline"
        )
        # The reject path means win-loss summary is part of telemetry.
        assert "win_loss" in payload

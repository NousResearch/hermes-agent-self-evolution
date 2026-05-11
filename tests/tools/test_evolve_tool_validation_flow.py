"""End-to-end orchestrator test with mocked LMs.

Mocks at four seams so the orchestrator's wiring is exercised without
burning a real LM run:

1. `SyntheticDatasetBuilder._call_lm_for_bucket` — canned per-bucket
   task lists drive the three-bucket synthetic dataset.
2. `dspy.GEPA` — a fake optimizer whose `compile()` returns a
   pre-built `ToolModule` carrying the desired evolved description plus
   a `detailed_results` namespace so the knee-point path runs.
3. `LLMJudge.score` — returns scripted `FitnessScore`s. The metric
   feeds these into the GEPA-shaped prediction the orchestrator's
   holdout-eval consumes.
4. `ToolModule.forward` — short-circuits the inner LM call so the
   holdout-evaluate loop walks deterministically through `(task,
   chosen_tool, reasoning)` tuples scripted per-example.

Together these let `evolve()` run end-to-end against the
`multiple_tools.json` fixture in <1s/test.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dspy
import pytest

from evolution.core.dataset_builder import SyntheticDatasetBuilder
from evolution.core.fitness import FitnessScore, LLMJudge
from evolution.tools.evolve_tool import _description_from_predictor, evolve
from evolution.tools.tool_module import ToolModule
from evolution.tools.tool_source import ToolManifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "tool_manifests"


# A non-trivial improvement over the deliberately-weak "Find things." baseline.
EVOLVED_DESCRIPTION = (
    "Find files in the repository by name or glob pattern. "
    "Returns matching file paths."
)


@pytest.fixture
def temp_manifest(tmp_path: Path) -> Path:
    """Copy multiple_tools.json to a tmp location so `--apply` tests don't
    clobber the shared fixture file."""
    src = FIXTURES / "multiple_tools.json"
    dst = tmp_path / "manifest.json"
    dst.write_text(src.read_text())
    return dst


def _bucket_tasks(target_correct_count: int, confusable_count: int, regression_count: int):
    """Return one canned dict-response per bucket call.

    Tasks deliberately avoid the manifest tool names so the anti-trivial
    filter keeps everything. Each call returns its bucket's response in
    order: target_correct → confusable_neighbor → regression_detection.
    """
    target_correct = [
        {"task": f"locate the configuration entry point {i}", "correct_tool": "search_files"}
        for i in range(target_correct_count)
    ]
    confusable = [
        {"task": f"hunt down the build output {i}", "correct_tool": "search_files"}
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


def _make_fake_gepa(evolved_module: ToolModule):
    """Build a fake `dspy.GEPA` class whose compile() returns `evolved_module`.

    `detailed_results` is shaped to let the knee-point path pick the single
    evolved candidate (val_aggregate_scores=[1.0], best_idx=0).
    """

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


def _build_evolved_module(manifest: ToolManifest, description: str) -> ToolModule:
    return ToolModule(
        target_tool_name="search_files",
        manifest=manifest,
        target_description=description,
    )


def _scripted_judge_score(*, target_score: float, regression_score: float):
    """Build a side_effect for LLMJudge.score scripted by target tool."""

    def _score(self, *, task_input, expected_behavior, agent_output, **_):
        # expected_behavior carries the correct_tool name for tool-selection examples.
        # agent_output contains "chosen_tool: <name>\nreasoning: <text>"
        chosen = ""
        for line in agent_output.splitlines():
            if line.startswith("chosen_tool:"):
                chosen = line.split(":", 1)[1].strip()
                break
        correct = chosen == expected_behavior
        score = target_score if correct else regression_score
        return FitnessScore(
            correctness=score,
            procedure_following=score,
            conciseness=score,
            length_penalty=0.0,
            feedback="" if correct else "wrong tool",
            profile="balanced",
        )

    return _score


def _scripted_module_forward(expected_tool_for_evolved: str):
    """Build a side_effect for ToolModule.forward that picks the
    `expected_behavior` value (the correct tool) for every task.

    The metric will then route through judge.score, where correctness
    comes from comparing chosen vs expected.
    """

    def _forward(self, task):
        # ToolModule only sees `task`, not `expected_behavior`. Use the
        # description text as the side-channel: the baseline module
        # carries the original short description; the evolved module
        # carries the longer evolved description plugged in by the fake
        # GEPA. The evolved module always picks the target tool; the
        # baseline picks a deliberately-wrong tool.
        current = self.description_text
        if current == EVOLVED_DESCRIPTION:
            chosen = expected_tool_for_evolved
        else:
            chosen = "compute_sha256"  # deliberately wrong baseline pick
        return dspy.Prediction(chosen_tool=chosen, reasoning="picked by mock")

    return _forward


def _bucket_side_effect(target_correct_count: int, confusable_count: int, regression_count: int):
    """Build a side_effect callable for `_call_lm_for_bucket` patches.

    Wrapping `iter().__next__` in a lambda is required because the patched
    method gets called with keyword arguments and `__next__` rejects them.
    """
    it = iter(_bucket_tasks(target_correct_count, confusable_count, regression_count))
    return lambda *a, **k: next(it)


class TestGateDecisionSchemaOnDeploy:
    """Happy-path: the orchestrator writes a gate_decision.json carrying
    the four tool-specific fields and a `deploy` decision."""

    def test_gate_decision_schema_on_deploy(self, temp_manifest: Path, tmp_path: Path):
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"

        with (
            patch.object(
                SyntheticDatasetBuilder,
                "_call_lm_for_bucket",
                side_effect=_bucket_side_effect(15, 9, 6),
            ),
            patch(
                "evolution.tools.evolve_tool.dspy.GEPA",
                new=_make_fake_gepa(
                    _build_evolved_module(manifest, EVOLVED_DESCRIPTION)
                ),
            ),
            patch.object(
                LLMJudge,
                "score",
                new=_scripted_judge_score(target_score=0.95, regression_score=0.0),
            ),
            patch.object(
                ToolModule,
                "forward",
                new=_scripted_module_forward(expected_tool_for_evolved="search_files"),
            ),
        ):
            result = evolve(
                tool_name="search_files",
                manifest_path=temp_manifest,
                iterations=1,
                eval_dataset_size=30,
                holdout_ratio=0.5,
                quality_gate="non-inferiority",
                output_dir=run_dir,
            )

        gate_path = run_dir / "gate_decision.json"
        assert gate_path.exists(), f"gate_decision.json not at {gate_path}"
        payload = json.loads(gate_path.read_text())

        # Tool-specific fields.
        assert payload["artifact_type"] == "tool_description"
        assert payload["target_tool"] == "search_files"
        assert payload["manifest_neighbor_count"] == 6  # 7 tools - target
        assert payload["sentinel_failures"] == 0

        # Schema sanity — the core deploy-gate keys.
        assert payload["decision"] == "deploy"
        for required in (
            "schema_version", "decision", "decision_rule_used",
            "gate_mode", "growth_pct", "required_improvement",
            "baseline_chars", "evolved_chars", "bootstrap",
            "knee_point", "dataset", "run_inputs",
        ):
            assert required in payload, f"missing {required}"

        # The result dict echoes the deploy decision.
        assert result["baseline_score"] < result["evolved_score"]


class TestApplyOverwritesSourceManifest:
    """`apply=True` writes the evolved description back to the source manifest."""

    def test_apply_overwrites_source_manifest(self, temp_manifest: Path, tmp_path: Path):
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"

        with (
            patch.object(
                SyntheticDatasetBuilder,
                "_call_lm_for_bucket",
                side_effect=_bucket_side_effect(15, 9, 6),
            ),
            patch(
                "evolution.tools.evolve_tool.dspy.GEPA",
                new=_make_fake_gepa(
                    _build_evolved_module(manifest, EVOLVED_DESCRIPTION)
                ),
            ),
            patch.object(
                LLMJudge,
                "score",
                new=_scripted_judge_score(target_score=0.95, regression_score=0.0),
            ),
            patch.object(
                ToolModule,
                "forward",
                new=_scripted_module_forward(expected_tool_for_evolved="search_files"),
            ),
        ):
            evolve(
                tool_name="search_files",
                manifest_path=temp_manifest,
                iterations=1,
                eval_dataset_size=30,
                holdout_ratio=0.5,
                quality_gate="non-inferiority",
                output_dir=run_dir,
                apply=True,
            )

        # The source manifest's target tool description must now match the evolved one.
        post = json.loads(temp_manifest.read_text())
        search_entry = next(t for t in post["tools"] if t["name"] == "search_files")
        assert search_entry["description"] == EVOLVED_DESCRIPTION
        # Other tools must be untouched.
        grep_entry = next(t for t in post["tools"] if t["name"] == "grep_in_terminal")
        assert "Run grep over file contents" in grep_entry["description"]


class TestPatchEmitsUnifiedDiff:
    """`patch=True` emits a unified diff of the manifest changes to stdout."""

    def test_patch_emits_unified_diff_to_stdout(
        self, temp_manifest: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ):
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"
        original = temp_manifest.read_text()

        with (
            patch.object(
                SyntheticDatasetBuilder,
                "_call_lm_for_bucket",
                side_effect=_bucket_side_effect(15, 9, 6),
            ),
            patch(
                "evolution.tools.evolve_tool.dspy.GEPA",
                new=_make_fake_gepa(
                    _build_evolved_module(manifest, EVOLVED_DESCRIPTION)
                ),
            ),
            patch.object(
                LLMJudge,
                "score",
                new=_scripted_judge_score(target_score=0.95, regression_score=0.0),
            ),
            patch.object(
                ToolModule,
                "forward",
                new=_scripted_module_forward(expected_tool_for_evolved="search_files"),
            ),
        ):
            evolve(
                tool_name="search_files",
                manifest_path=temp_manifest,
                iterations=1,
                eval_dataset_size=30,
                holdout_ratio=0.5,
                quality_gate="non-inferiority",
                output_dir=run_dir,
                patch=True,
            )

        captured = capsys.readouterr()
        out = captured.out

        # Unified-diff markers must be present, and the manifest path labelled.
        assert "@@ " in out, "unified-diff hunk header missing"
        assert str(temp_manifest) in out, "manifest path not in diff labels"
        assert "Find things." in out, "baseline line should appear as a removal"
        assert EVOLVED_DESCRIPTION.split(" ")[0] in out, "evolved text not in diff"

        # `--patch` must NOT modify the source file.
        assert temp_manifest.read_text() == original


class TestDefaultWritesEvolvedManifestToRunDir:
    """Without `--apply` or `--patch`, the orchestrator writes
    `evolved_manifest.json` to the run dir and leaves the source alone."""

    def test_default_writes_evolved_manifest_to_run_dir_without_modifying_source(
        self, temp_manifest: Path, tmp_path: Path
    ):
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"
        original_source = temp_manifest.read_text()

        with (
            patch.object(
                SyntheticDatasetBuilder,
                "_call_lm_for_bucket",
                side_effect=_bucket_side_effect(15, 9, 6),
            ),
            patch(
                "evolution.tools.evolve_tool.dspy.GEPA",
                new=_make_fake_gepa(
                    _build_evolved_module(manifest, EVOLVED_DESCRIPTION)
                ),
            ),
            patch.object(
                LLMJudge,
                "score",
                new=_scripted_judge_score(target_score=0.95, regression_score=0.0),
            ),
            patch.object(
                ToolModule,
                "forward",
                new=_scripted_module_forward(expected_tool_for_evolved="search_files"),
            ),
        ):
            evolve(
                tool_name="search_files",
                manifest_path=temp_manifest,
                iterations=1,
                eval_dataset_size=30,
                holdout_ratio=0.5,
                quality_gate="non-inferiority",
                output_dir=run_dir,
            )

        evolved_manifest_path = run_dir / "evolved_manifest.json"
        assert evolved_manifest_path.exists()
        evolved = json.loads(evolved_manifest_path.read_text())
        search_entry = next(t for t in evolved["tools"] if t["name"] == "search_files")
        assert search_entry["description"] == EVOLVED_DESCRIPTION

        # The source must be untouched byte-for-byte.
        assert temp_manifest.read_text() == original_source


class TestFileHandlerLifecycle:
    """The per-run FileHandler attached to the root logger must be removed
    on every exit path. Two evolve() calls in one process used to leak two
    handlers; the try/finally wrap inside evolve() keeps the count stable.
    """

    def test_evolve_does_not_leak_file_handlers(
        self, temp_manifest: Path, tmp_path: Path
    ):
        manifest = ToolManifest.from_json_file(temp_manifest)
        before = len(logging.getLogger().handlers)

        def _run_once(run_dir: Path) -> None:
            with (
                patch.object(
                    SyntheticDatasetBuilder,
                    "_call_lm_for_bucket",
                    side_effect=_bucket_side_effect(15, 9, 6),
                ),
                patch(
                    "evolution.tools.evolve_tool.dspy.GEPA",
                    new=_make_fake_gepa(
                        _build_evolved_module(manifest, EVOLVED_DESCRIPTION)
                    ),
                ),
                patch.object(
                    LLMJudge,
                    "score",
                    new=_scripted_judge_score(target_score=0.95, regression_score=0.0),
                ),
                patch.object(
                    ToolModule,
                    "forward",
                    new=_scripted_module_forward(expected_tool_for_evolved="search_files"),
                ),
            ):
                evolve(
                    tool_name="search_files",
                    manifest_path=temp_manifest,
                    iterations=1,
                    eval_dataset_size=30,
                    holdout_ratio=0.5,
                    quality_gate="non-inferiority",
                    output_dir=run_dir,
                )

        _run_once(tmp_path / "run_0")
        _run_once(tmp_path / "run_1")

        after = len(logging.getLogger().handlers)
        assert after == before, (
            f"evolve() leaked {after - before} root-logger handler(s) across two calls"
        )


class TestDescriptionExtractorSentinelFailure:
    """The latent path inside `_description_from_predictor` catches
    SentinelParseError and logs a warning; both names must be in scope.
    """

    def test_description_extractor_handles_sentinel_failure_without_NameError(
        self, caplog: pytest.LogCaptureFixture
    ):
        fake_predictor = SimpleNamespace(
            signature=SimpleNamespace(instructions="no sentinels here")
        )

        with caplog.at_level(logging.WARNING, logger="evolution.tools.evolve_tool"):
            result = _description_from_predictor(fake_predictor, "search_files")

        assert result == ""
        assert any(
            "could not extract description from predictor instructions"
            in record.getMessage()
            for record in caplog.records
        ), f"expected warning not found in {[r.getMessage() for r in caplog.records]}"

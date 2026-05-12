"""End-to-end orchestrator test with mocked LMs.

Mocks at four seams so the orchestrator's wiring is exercised without
burning a real LM run:

1. `SyntheticDatasetBuilder._call_lm_for_bucket` — canned per-bucket
   task lists drive the three-bucket synthetic dataset.
2. `dspy.GEPA` — a fake optimizer whose `compile()` returns a
   pre-built `ToolModule` carrying the desired evolved description plus
   a `detailed_results` namespace so the knee-point path runs.
3. `ToolJudge.score` — returns scripted `FitnessScore`s. The metric
   feeds these into the GEPA-shaped prediction the orchestrator's
   holdout-eval consumes.
4. `ToolModule.forward` — short-circuits the inner LM call so the
   holdout-evaluate loop walks deterministically through `(task,
   chosen_tool, agent_reasoning)` tuples scripted per-example.

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

from evolution.core.dataset_builder import EvalDataset, EvalExample, SyntheticDatasetBuilder
from evolution.core.external_importers import HermesSessionImporter
from evolution.core.fitness import FitnessScore
from evolution.tools.evolve_tool import _description_from_predictor, evolve
from evolution.tools.tool_judge import ToolJudge
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
    """Build a side_effect for ToolJudge.score scripted by target tool."""

    def _score(self, *, task, expected_tool, chosen_tool, agent_reasoning, **_):
        correct = chosen_tool == expected_tool
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
                ToolJudge,
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
                enable_confusable_bucket=True,
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

        # The run's bucket-mix policy is recorded so downstream analyses
        # can distinguish "flag off" from "no neighbor declared".
        assert payload["run_inputs"]["enable_confusable_bucket"] is True

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
                ToolJudge,
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
                enable_confusable_bucket=True,
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
                ToolJudge,
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
                enable_confusable_bucket=True,
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
                ToolJudge,
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
                enable_confusable_bucket=True,
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
                    ToolJudge,
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
                    enable_confusable_bucket=True,
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


class TestRelativeManifestPath:
    """A multi-component relative ``--manifest`` path used to crash with
    ``find_manifest returned None``: ``_resolve_source`` would set the
    adapter's root to ``manifest_path.parent`` and the adapter would then
    re-join the original (still-relative) path under that root, doubling
    the path components."""

    def test_evolve_accepts_cwd_relative_manifest_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        manifest_file = tmp_path / "sub" / "manifest.json"
        manifest_file.parent.mkdir()
        manifest_file.write_text((FIXTURES / "multiple_tools.json").read_text())
        manifest = ToolManifest.from_json_file(manifest_file)
        monkeypatch.chdir(tmp_path)
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
                ToolJudge,
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
                manifest_path=Path("sub/manifest.json"),
                iterations=1,
                eval_dataset_size=30,
                holdout_ratio=0.5,
                quality_gate="non-inferiority",
                enable_confusable_bucket=True,
                output_dir=run_dir,
            )

        assert (run_dir / "gate_decision.json").exists()


_SESSIONDB_DATASET_SIZE = 22  # 11/11 across the two categories — holdout (12) clears min=10


def _make_sessiondb_dataset() -> tuple[EvalDataset, dict[str, int]]:
    """Build a small (agreed + misselection) dataset the way the real miner would.

    22 examples split 30/20/50 ≈ 6 train / 4 val / 12 holdout (≥ min_holdout_size=10).
    """
    half = _SESSIONDB_DATASET_SIZE // 2
    examples = []
    for i in range(half):
        examples.append(EvalExample(
            task_input=f"Find Python tests example {i}",
            expected_behavior="search_files",
            category="agreed",
            source="hermes",
        ))
    for i in range(half):
        examples.append(EvalExample(
            task_input=f"Locate config files example {i}",
            expected_behavior="search_files",
            category="misselection",
            source="hermes",
        ))
    from evolution.core.dataset_builder import split_examples
    dataset = split_examples(
        examples,
        seed=42,
        train_ratio=0.3,
        val_ratio=0.2,
        holdout_ratio=0.5,
    )
    drops = {
        "short_task": 2, "slash_command": 1, "secret": 0,
        "no_tool_calls": 5, "non_manifest": 3,
        "judge_irrelevant": 1, "judge_error": 0, "noisy_middle": 2,
        "low_confidence": 1, "unknown_correct_tool": 0,
    }
    return dataset, drops


class TestEvalSourceSessiondb:
    """The --eval-source sessiondb branch — orchestrator wiring + payload threading."""

    def test_sessiondb_happy_path_writes_dataset_payload(self, temp_manifest: Path, tmp_path: Path):
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"
        dataset, drops = _make_sessiondb_dataset()

        with (
            patch(
                "evolution.tools.evolve_tool.build_tool_dataset_from_sessions",
                return_value=(dataset, drops),
            ),
            patch(
                "evolution.tools.evolve_tool.dspy.GEPA",
                new=_make_fake_gepa(_build_evolved_module(manifest, EVOLVED_DESCRIPTION)),
            ),
            patch.object(
                ToolJudge,
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
                eval_source="sessiondb",
                eval_dataset_size=_SESSIONDB_DATASET_SIZE,
                holdout_ratio=0.5,
                quality_gate="non-inferiority",
                output_dir=run_dir,
            )

        payload = json.loads((run_dir / "gate_decision.json").read_text())
        assert payload["dataset"]["sources"] == {"hermes": _SESSIONDB_DATASET_SIZE}
        # Both category buckets land in the payload — the sessiondb-only namespace.
        assert set(payload["dataset"]["categories"]) == {"agreed", "misselection"}
        # Drops are threaded all the way into gate_decision.json.
        assert payload["dataset"]["sessiondb_drops"] == drops
        assert payload["dataset"]["dropped_non_manifest_count"] == 3

    def test_sessiondb_empty_result_exits(self, temp_manifest: Path, tmp_path: Path):
        empty_drops = {
            "short_task": 0, "slash_command": 0, "secret": 0,
            "no_tool_calls": 100, "non_manifest": 50,
            "judge_irrelevant": 0, "judge_error": 0, "noisy_middle": 0,
            "low_confidence": 0, "unknown_correct_tool": 0,
        }
        with patch(
            "evolution.tools.evolve_tool.build_tool_dataset_from_sessions",
            return_value=(EvalDataset(), empty_drops),
        ):
            with pytest.raises(SystemExit) as exc_info:
                evolve(
                    tool_name="search_files",
                    manifest_path=temp_manifest,
                    iterations=1,
                    eval_source="sessiondb",
                    output_dir=tmp_path / "run",
                )
        assert exc_info.value.code == 1

    def test_sessiondb_dry_run_skips_judge_and_gepa(self, temp_manifest: Path, tmp_path: Path):
        """Dry-run on the sessiondb path runs only the (free) importer; the
        judge and GEPA must never be constructed. Tripwires both."""
        run_dir = tmp_path / "run"
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()  # empty — importer returns 0 candidates cleanly

        class _Tripwire:
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    f"{type(self).__name__} should not be constructed in dry-run mode"
                )

        with (
            patch(
                "evolution.tools.evolve_tool.build_tool_dataset_from_sessions",
                new=_Tripwire,
            ),
            patch("evolution.tools.session_mining.ToolRelevanceFilter", new=_Tripwire),
            patch("evolution.tools.evolve_tool.dspy.GEPA", new=_Tripwire),
            patch.object(HermesSessionImporter, "SESSION_DIR", sessions_dir),
        ):
            result = evolve(
                tool_name="search_files",
                manifest_path=temp_manifest,
                iterations=1,
                eval_source="sessiondb",
                eval_dataset_size=_SESSIONDB_DATASET_SIZE,
                holdout_ratio=0.5,
                dry_run=True,
                output_dir=run_dir,
            )

        assert result["decision"] == "dry-run"
        assert result["eval_source"] == "sessiondb"
        assert result["candidate_count"] == 0
        # All importer drop keys present even when zero candidates surfaced.
        assert set(result["importer_drops"]) == {
            "short_task", "slash_command", "secret", "no_tool_calls", "non_manifest",
        }
        assert result["invoked_tool_distribution"] == {}
        # No gate_decision.json should have been written.
        assert not (run_dir / "gate_decision.json").exists()

    def test_synthetic_dry_run_skips_dataset_gen_and_gepa(
        self, temp_manifest: Path, tmp_path: Path
    ):
        """Dry-run on the synthetic path skips the LM-spending dataset
        generator entirely; just prints what would happen."""
        run_dir = tmp_path / "run"

        class _Tripwire:
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    f"{type(self).__name__} should not be constructed in dry-run mode"
                )

        with (
            patch("evolution.tools.evolve_tool.SyntheticDatasetBuilder", new=_Tripwire),
            patch("evolution.tools.evolve_tool.dspy.GEPA", new=_Tripwire),
        ):
            result = evolve(
                tool_name="search_files",
                manifest_path=temp_manifest,
                iterations=1,
                eval_source="synthetic",
                eval_dataset_size=30,
                holdout_ratio=0.5,
                dry_run=True,
                output_dir=run_dir,
            )

        assert result == {"decision": "dry-run", "eval_source": "synthetic"}
        assert not (run_dir / "gate_decision.json").exists()


def _benchmark_test_kwargs(temp_manifest: Path, run_dir: Path) -> dict:
    """Common evolve() kwargs for the benchmark-hook tests — the rest of the
    pipeline is mocked so we can exercise the post-decision hook insertion."""
    return dict(
        tool_name="search_files",
        manifest_path=temp_manifest,
        iterations=1,
        eval_dataset_size=30,
        holdout_ratio=0.5,
        quality_gate="non-inferiority",
        enable_confusable_bucket=True,
        output_dir=run_dir,
        benchmark_timeout_seconds=10,
    )


def _benchmark_test_mocks(manifest):
    """The four standard mocks that take evolve() through to the deploy
    decision — same shape as TestGateDecisionSchemaOnDeploy."""
    return [
        patch.object(
            SyntheticDatasetBuilder,
            "_call_lm_for_bucket",
            side_effect=_bucket_side_effect(15, 9, 6),
        ),
        patch(
            "evolution.tools.evolve_tool.dspy.GEPA",
            new=_make_fake_gepa(_build_evolved_module(manifest, EVOLVED_DESCRIPTION)),
        ),
        patch.object(
            ToolJudge,
            "score",
            new=_scripted_judge_score(target_score=0.95, regression_score=0.0),
        ),
        patch.object(
            ToolModule,
            "forward",
            new=_scripted_module_forward(expected_tool_for_evolved="search_files"),
        ),
    ]


class TestBenchmarkCmdHook:
    """The --benchmark-cmd hook: runs after the framework's own deploy gate
    passes; nonzero exit / timeout / spawn error flips to reject."""

    def test_benchmark_pass_keeps_deploy_decision(self, temp_manifest: Path, tmp_path: Path):
        import subprocess as _subprocess
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"
        fake_run = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        with patch("evolution.core.quality_gate.subprocess.run", return_value=fake_run):
            with _benchmark_test_mocks(manifest)[0], _benchmark_test_mocks(manifest)[1], \
                 _benchmark_test_mocks(manifest)[2], _benchmark_test_mocks(manifest)[3]:
                evolve(
                    **_benchmark_test_kwargs(temp_manifest, run_dir),
                    benchmark_cmd="echo ok",
                )

        payload = json.loads((run_dir / "gate_decision.json").read_text())
        assert payload["decision"] == "deploy"
        assert payload["reason"] == "passed"
        assert payload["benchmark"]["passed"] is True
        assert payload["benchmark"]["exit_code"] == 0
        assert payload["benchmark"]["reason"] == "ok"
        assert payload["benchmark"]["command"] == "echo ok"
        assert "ok" in payload["benchmark"]["stdout_tail"]
        # Deploy artifacts present.
        assert (run_dir / "evolved_manifest.json").exists()
        assert (run_dir / "baseline_manifest.json").exists()

    def test_benchmark_fail_flips_to_reject(self, temp_manifest: Path, tmp_path: Path):
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"
        fake_run = SimpleNamespace(returncode=1, stdout="boom\n", stderr="")
        with patch("evolution.core.quality_gate.subprocess.run", return_value=fake_run):
            with _benchmark_test_mocks(manifest)[0], _benchmark_test_mocks(manifest)[1], \
                 _benchmark_test_mocks(manifest)[2], _benchmark_test_mocks(manifest)[3]:
                evolve(
                    **_benchmark_test_kwargs(temp_manifest, run_dir),
                    benchmark_cmd="exit 1",
                )

        payload = json.loads((run_dir / "gate_decision.json").read_text())
        assert payload["decision"] == "reject"
        assert payload["reason"] == "benchmark_failed"
        assert payload["benchmark"]["passed"] is False
        assert payload["benchmark"]["exit_code"] == 1
        assert payload["benchmark"]["reason"] == "exit_nonzero"
        assert "boom" in payload["benchmark"]["stdout_tail"]
        # Reject path: failed artifact present, deploy artifact removed.
        assert (run_dir / "evolved_FAILED.json").exists()
        assert not (run_dir / "evolved_manifest.json").exists()
        assert not (run_dir / "baseline_manifest.json").exists()

    def test_benchmark_timeout_rejects(self, temp_manifest: Path, tmp_path: Path):
        import subprocess as _subprocess
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"
        with patch(
            "evolution.core.quality_gate.subprocess.run",
            side_effect=_subprocess.TimeoutExpired(cmd="sleep 100", timeout=10),
        ):
            with _benchmark_test_mocks(manifest)[0], _benchmark_test_mocks(manifest)[1], \
                 _benchmark_test_mocks(manifest)[2], _benchmark_test_mocks(manifest)[3]:
                evolve(
                    **_benchmark_test_kwargs(temp_manifest, run_dir),
                    benchmark_cmd="sleep 100",
                )

        payload = json.loads((run_dir / "gate_decision.json").read_text())
        assert payload["decision"] == "reject"
        assert payload["benchmark"]["passed"] is False
        assert payload["benchmark"]["reason"] == "timeout"
        assert payload["benchmark"]["exit_code"] is None

    def test_benchmark_command_error_rejects(self, temp_manifest: Path, tmp_path: Path):
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"
        with patch(
            "evolution.core.quality_gate.subprocess.run",
            side_effect=PermissionError("execve failed"),
        ):
            with _benchmark_test_mocks(manifest)[0], _benchmark_test_mocks(manifest)[1], \
                 _benchmark_test_mocks(manifest)[2], _benchmark_test_mocks(manifest)[3]:
                evolve(
                    **_benchmark_test_kwargs(temp_manifest, run_dir),
                    benchmark_cmd="/some/uninvokable/script",
                )

        payload = json.loads((run_dir / "gate_decision.json").read_text())
        assert payload["decision"] == "reject"
        assert payload["benchmark"]["passed"] is False
        assert payload["benchmark"]["reason"] == "command_error"
        assert "execve failed" in payload["benchmark"]["stderr_tail"]

    def test_benchmark_env_vars_reach_subprocess(self, temp_manifest: Path, tmp_path: Path):
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"
        captured_env = {}

        def _capture(*args, **kwargs):
            captured_env.update(kwargs.get("env") or {})
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch("evolution.core.quality_gate.subprocess.run", side_effect=_capture):
            with _benchmark_test_mocks(manifest)[0], _benchmark_test_mocks(manifest)[1], \
                 _benchmark_test_mocks(manifest)[2], _benchmark_test_mocks(manifest)[3]:
                evolve(
                    **_benchmark_test_kwargs(temp_manifest, run_dir),
                    benchmark_cmd="true",
                )

        assert captured_env["EVOLVED_PATH"].endswith("/evolved_manifest.json")
        assert captured_env["BASELINE_PATH"].endswith("/baseline_manifest.json")
        assert captured_env["RUN_DIR"] == str(run_dir)
        assert captured_env["TARGET_NAME"] == "search_files"
        assert captured_env["ARTIFACT_TYPE"] == "tool_description"

    def test_benchmark_not_called_when_growth_gate_rejects(self, temp_manifest: Path, tmp_path: Path):
        """If the framework's own gate would reject, the benchmark hook never
        runs — no point spending the user's CI budget on a variant we already
        decided not to ship."""
        from evolution.core.constraints import ConstraintResult, ConstraintValidator
        manifest = ToolManifest.from_json_file(temp_manifest)
        run_dir = tmp_path / "run"
        ran = {"called": False}

        def _tripwire(*args, **kwargs):
            ran["called"] = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        def _force_reject(self, artifact_text, baseline_text, bootstrap_result):
            return [ConstraintResult(
                passed=False,
                constraint_name="growth_quality_gate",
                message="forced reject for test",
            )]

        with patch("evolution.core.quality_gate.subprocess.run", side_effect=_tripwire), \
             patch.object(ConstraintValidator, "validate_growth_with_quality", new=_force_reject):
            with _benchmark_test_mocks(manifest)[0], _benchmark_test_mocks(manifest)[1], \
                 _benchmark_test_mocks(manifest)[2], _benchmark_test_mocks(manifest)[3]:
                evolve(
                    **_benchmark_test_kwargs(temp_manifest, run_dir),
                    benchmark_cmd="echo would-not-run",
                )

        # Growth gate failed → benchmark never invoked → no benchmark block.
        payload = json.loads((run_dir / "gate_decision.json").read_text())
        assert payload["decision"] == "reject"
        assert payload["reason"] == "growth_quality_gate"
        assert "benchmark" not in payload
        assert ran["called"] is False

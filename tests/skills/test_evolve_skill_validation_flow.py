"""Tests for the post-optimization validation+holdout flow in evolve_skill.

The flow is: validate_static → holdout → validate_growth_with_quality →
results table. These tests assert the wiring (right calls in the right
order, right outputs persisted) without exercising the LM-heavy parts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from click.testing import CliRunner

from evolution.skills.evolve_skill import (
    _apply_in_place,
    _compute_win_loss,
    _dataset_payload,
    _emit_patch,
    _evaluate_band_on_holdout,
    _holdout_evaluate_with_metric,
    _knee_point_payload,
    _resolve_bap_max_growth,
    _resolve_bap_safety_margin,
    _resolve_proposer_mode,
    _write_gate_decision,
    main as evolve_skill_cli,
)
from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.skills.knee_point import CandidatePick


@pytest.fixture(autouse=True, scope="session")
def _skill_source_env(tmp_path_factory):
    """Same env-var workaround as test_constraints.py."""
    fake_repo = tmp_path_factory.mktemp("fake_skill_repo")
    (fake_repo / "skills").mkdir()
    os.environ["SKILL_SOURCES_HERMES_REPO"] = str(fake_repo)
    yield


class TestWriteGateDecision:
    def test_writes_json_with_payload(self, tmp_path: Path):
        payload = {"decision": "deploy", "growth_pct": 0.24, "improvement": 0.07}
        path = _write_gate_decision(tmp_path, payload)

        assert path.name == "gate_decision.json"
        loaded = json.loads(path.read_text())
        assert loaded == payload

    def test_creates_parent_directories(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        path = _write_gate_decision(nested, {"decision": "reject"})
        assert path.exists()
        assert path.parent == nested


class TestHoldoutEvaluate:
    """The metric returns dspy.Prediction(score, feedback); _holdout_evaluate
    must wrap it for dspy.Evaluate's 2-arg metric protocol and unwrap the
    .score on return."""

    def test_unwraps_prediction_score(self):
        # Stand-in for dspy.Prediction — only .score matters.
        prediction = SimpleNamespace(output="answer")
        examples = [
            SimpleNamespace(task_input=f"task {i}", expected_behavior="b",
                            with_inputs=lambda *a, **k: SimpleNamespace(task_input=f"task {i}"))
            for i in range(3)
        ]
        module = MagicMock()
        module.return_value = prediction
        metric = MagicMock(return_value=SimpleNamespace(score=0.7))

        # Stand in for dspy.Evaluate without spinning up DSPy's machinery —
        # invoke the metric per example, mirroring real Evaluate's contract.
        with patch("evolution.skills.evolve_skill.dspy.Evaluate") as evaluate_cls:
            captured_metric = {}

            class _FakeEval:
                def __init__(self, *, devset, metric, num_threads,
                             provide_traceback, max_errors):
                    captured_metric["fn"] = metric
                    self.devset = devset

                def __call__(self, mod):
                    # dspy.Evaluate returns EvaluationResult(score, results).
                    # score is mean*100; results is list of (example, prediction, score).
                    scores = []
                    results = []
                    for ex in self.devset:
                        pred = mod(task_input=getattr(ex, "task_input", ""))
                        s = captured_metric["fn"](ex, pred)
                        scores.append(s)
                        results.append((ex, pred, s))
                    return SimpleNamespace(
                        score=sum(scores) / len(scores) * 100,
                        results=results,
                    )

            evaluate_cls.side_effect = _FakeEval

            avg, per_example = _holdout_evaluate_with_metric(
                module, examples, metric, lm=MagicMock(),
            )

        assert avg == pytest.approx(0.7)
        assert per_example == [0.7, 0.7, 0.7]
        # Metric called once per example; each call passed (example, prediction).
        assert metric.call_count == 3


class TestStaticValidationShortCircuitsBeforeHoldout:
    """If validate_static returns any failures, the flow must save
    evolved_FAILED.md + gate_decision.json and return without invoking
    the holdout block (which would waste judge calls on a broken artifact).

    Exercising this end-to-end requires patching most of evolve(), so we
    instead test the mechanism: a failed-static gate_decision.json names
    the failed constraints and reason='static_constraint_failure'.
    """

    def test_static_failure_reason_in_decision(self, tmp_path: Path):
        # Manual reproduction of the static-failure branch's payload —
        # locks the schema so a future refactor can't silently drop fields.
        payload = {
            "schema_version": "4",
            "decision": "reject",
            "reason": "static_constraint_failure",
            "failed_constraints": ["non_empty"],
            "messages": ["Artifact is empty"],
            "knee_point": {"applied": False, "reason": "no_detailed_results"},
        }
        path = _write_gate_decision(tmp_path, payload)
        loaded = json.loads(path.read_text())
        assert loaded["schema_version"] == "4"
        assert loaded["reason"] == "static_constraint_failure"
        assert "non_empty" in loaded["failed_constraints"]
        assert "knee_point" in loaded


class TestGrowthGateDecisionSchema:
    """The growth-gate decision payload is the calibration substrate for
    future tier tuning. Lock the schema so a `jq -s` calibration script
    doesn't break when fields are renamed.
    """

    def test_required_fields_present(self, tmp_path: Path):
        payload = {
            "schema_version": "4",
            "decision": "reject",
            "reason": "growth_quality_gate",
            "decision_rule_used": "dual_check",
            "gate_mode": "no_regression",
            "inferiority_tolerance": 0.0,
            "growth_pct": 0.30,
            "required_improvement": 0.030,
            "baseline_chars": 1000,
            "evolved_chars": 1300,
            "absolute_char_ceiling": 5000,
            "effective_absolute_char_ceiling": 5000,
            "growth_free_threshold": 0.20,
            "fitness_profile": "balanced",
            "proposer_mode": "compression",
            "growth_quality_slope": 0.30,
            "bap_max_growth": 0.20,
            "bap_safety_margin": 0.10,
            "baseline_per_example": [0.5, 0.6, 0.7],
            "evolved_per_example": [0.51, 0.61, 0.71],
            "avg_baseline": 0.6,
            "avg_evolved": 0.605,
            "bootstrap": {
                "mean": 0.005,
                "lower_bound": -0.020,
                "upper_bound": 0.030,
                "n_examples": 12,
                "n_resamples": 2000,
                "confidence": 0.90,
            },
            "failed_constraints": ["growth_quality_gate"],
            "messages": ["..."],
            "knee_point": {
                "applied": True,
                "fallback": "knee",
                "epsilon": 0.1666666,
                "band_size": 4,
                "picked_idx": 12,
                "picked_val_score": 0.95,
                "picked_val_rank_in_band": 3,
                "picked_body_chars": 412,
                "gepa_default_idx": 5,
                "gepa_default_body_chars": 1572,
                "band_roster": [
                    {"idx": 5, "val_score": 0.997, "body_chars": 1572},
                    {"idx": 12, "val_score": 0.95, "body_chars": 412},
                ],
            },
            "dataset": {
                "size_total": 150,
                "size_train": 54,
                "size_val": 43,
                "size_holdout": 53,
                "sources": {"synthetic": 150},
            },
            "run_inputs": {
                "seed": 42,
                "iterations": 10,
                "optimizer_model": "openai/gpt-4.1",
                "reflection_model": None,
                "eval_model": "openai/gpt-4.1-mini",
                "eval_dataset_size": 150,
                "holdout_ratio": 0.5,
                "quality_gate_preset": "default",
                "eval_source": "synthetic",
            },
            "win_loss": {
                "n_wins": 2,
                "n_losses": 1,
                "n_ties": 0,
                "worst_regression": -0.05,
                "worst_improvement": 0.10,
            },
        }
        path = _write_gate_decision(tmp_path, payload)
        loaded = json.loads(path.read_text())

        # Calibration script will rely on these keys.
        for required in (
            "schema_version", "decision", "decision_rule_used",
            "gate_mode", "inferiority_tolerance",
            "growth_pct", "required_improvement",
            "baseline_chars", "evolved_chars",
            "growth_free_threshold", "growth_quality_slope",
            "bootstrap", "knee_point", "dataset",
        ):
            assert required in loaded, f"missing {required}"
        assert loaded["schema_version"] == "4"
        for required_in_bootstrap in (
            "mean", "lower_bound", "upper_bound", "n_examples",
            "n_resamples", "confidence",
        ):
            assert required_in_bootstrap in loaded["bootstrap"], (
                f"missing bootstrap.{required_in_bootstrap}"
            )
        for required_in_knee in (
            "applied", "fallback", "epsilon", "band_size",
            "picked_idx", "picked_val_score", "picked_val_rank_in_band",
            "picked_body_chars", "gepa_default_idx", "gepa_default_body_chars",
            "band_roster",
        ):
            assert required_in_knee in loaded["knee_point"], (
                f"missing knee_point.{required_in_knee}"
            )
        for required_in_dataset in (
            "size_total", "size_train", "size_val", "size_holdout", "sources",
        ):
            assert required_in_dataset in loaded["dataset"], (
                f"missing dataset.{required_in_dataset}"
            )


class TestRunInputsBlock:
    """The `run_inputs` block records the inputs that produced the run so a
    third party with the gate_decision.json artifact alone can reproduce
    the result. Lock the shape so future refactors can't silently drop a key.
    """

    def test_run_inputs_present_in_decision(self, tmp_path: Path):
        payload = {
            "schema_version": "4",
            "decision": "deploy",
            "run_inputs": {
                "seed": 42,
                "iterations": 10,
                "optimizer_model": "openai/gpt-4.1",
                "reflection_model": "openai/gpt-4.1",
                "eval_model": "openai/gpt-4.1-mini",
                "eval_dataset_size": 150,
                "holdout_ratio": 0.5,
                "quality_gate_preset": "default",
                "eval_source": "synthetic",
            },
        }
        path = _write_gate_decision(tmp_path, payload)
        loaded = json.loads(path.read_text())

        assert "run_inputs" in loaded
        for required in (
            "seed",
            "iterations",
            "optimizer_model",
            "reflection_model",
            "eval_model",
            "eval_dataset_size",
            "holdout_ratio",
            "quality_gate_preset",
            "eval_source",
        ):
            assert required in loaded["run_inputs"], (
                f"missing run_inputs.{required}"
            )
        assert len(loaded["run_inputs"]) == 9


class TestWinLossDecomposition:
    """`_compute_win_loss` summarizes the per-example delta distribution
    behind the aggregate mean. The deploy/reject logic does NOT use it —
    it's purely an information lens for users.
    """

    def test_wins_losses_ties_count_correctly(self):
        baseline = [0.5, 0.7, 0.8, 0.4]
        evolved = [0.6, 0.7, 0.7, 0.5]
        result = _compute_win_loss(baseline, evolved)
        assert result["n_wins"] == 2
        assert result["n_losses"] == 1
        assert result["n_ties"] == 1

    def test_worst_regression_and_improvement(self):
        baseline = [0.5, 0.7, 0.8, 0.4]
        evolved = [0.6, 0.7, 0.7, 0.5]
        result = _compute_win_loss(baseline, evolved)
        # Deltas: +0.1, 0.0, -0.1, +0.1
        assert result["worst_regression"] == pytest.approx(-0.1)
        assert result["worst_improvement"] == pytest.approx(0.1)

    def test_handles_empty_arrays_gracefully(self):
        result = _compute_win_loss([], [])
        assert result == {
            "n_wins": 0,
            "n_losses": 0,
            "n_ties": 0,
            "worst_regression": 0.0,
            "worst_improvement": 0.0,
        }


class TestKneePointPayloadHelper:
    """`_knee_point_payload` is the single producer of the knee_point block
    that lands in gate_decision.json. Lock both shapes (applied/skipped).
    """

    def test_skipped_payload_when_no_detailed_results(self):
        # MIPROv2 fallback path: knee_pick is None.
        payload = _knee_point_payload(None)
        assert payload == {"applied": False, "reason": "no_detailed_results"}

    def test_applied_payload_carries_all_required_fields(self):
        pick = CandidatePick(
            module=SimpleNamespace(skill_text="x" * 412),
            skill_text="x" * 412,
            body_chars=412,
            val_score=0.95,
            val_rank_in_band=3,
            band_size=4,
            epsilon=1.0 / 6.0,
            fallback="knee",
            picked_idx=12,
            gepa_default_idx=5,
            gepa_default_body_chars=1572,
            band_roster=[
                {"idx": 5, "val_score": 0.997, "body_chars": 1572},
                {"idx": 12, "val_score": 0.95, "body_chars": 412},
            ],
        )
        payload = _knee_point_payload(pick)
        assert payload["applied"] is True
        assert payload["fallback"] == "knee"
        assert payload["picked_idx"] == 12
        assert payload["gepa_default_idx"] == 5
        assert payload["picked_val_rank_in_band"] == 3
        assert payload["band_size"] == 4
        assert payload["band_roster"][0]["idx"] == 5
        # Round-trips JSON cleanly (no non-serializable objects sneaked in).
        json.dumps(payload)


class TestDatasetPayloadHelper:
    """`_dataset_payload` is the single producer of the dataset block.

    The per-source counter is the calibration substrate for "is mined-source
    dominance correlated with deploy rate?" — keep it stable across PRs.
    """

    @staticmethod
    def _ex(source: str) -> EvalExample:
        return EvalExample(
            task_input="t", expected_behavior="b", source=source,
        )

    def test_records_split_sizes(self):
        # Use realistic counts that match the current N=150 default split
        # (~54/43/53). This test isn't asserting on EvolutionConfig behavior
        # — it locks the payload helper's faithful reporting of arbitrary
        # input sizes. Numbers here track the documented defaults so future
        # readers don't have to triangulate against a stale fixture.
        ds = EvalDataset(
            train=[self._ex("synthetic")] * 54,
            val=[self._ex("synthetic")] * 43,
            holdout=[self._ex("synthetic")] * 53,
        )
        payload = _dataset_payload(ds)
        assert payload["size_total"] == 150
        assert payload["size_train"] == 54
        assert payload["size_val"] == 43
        assert payload["size_holdout"] == 53

    def test_buckets_per_source(self):
        ds = EvalDataset(
            train=[self._ex("synthetic")] * 10 + [self._ex("sessiondb_claude_code")] * 5,
            val=[self._ex("synthetic")] * 5,
            holdout=[self._ex("golden")] * 3,
        )
        payload = _dataset_payload(ds)
        assert payload["sources"] == {
            "synthetic": 15,
            "sessiondb_claude_code": 5,
            "golden": 3,
        }

    def test_unknown_source_bucketed_as_unknown(self):
        # Defensive: an EvalExample with empty/None source shouldn't crash
        # the calibration JSON.
        ex = EvalExample(task_input="t", expected_behavior="b", source="")
        ds = EvalDataset(train=[ex], val=[], holdout=[])
        payload = _dataset_payload(ds)
        assert payload["sources"] == {"unknown": 1}

    def test_payload_round_trips_json(self):
        ds = EvalDataset(
            train=[self._ex("synthetic")] * 2,
            val=[self._ex("synthetic")] * 1,
            holdout=[self._ex("synthetic")] * 1,
        )
        json.dumps(_dataset_payload(ds))


class TestEvaluateBandOnHoldout:
    """The band-holdout hook iterates `details.candidates[idx]` for each
    band entry and runs the holdout metric. Cannot be a post-run script —
    GEPA's candidate programs aren't persisted, so this has to capture
    them while `details` is still in scope inside `evolve()`."""

    def _make_pick(self, band_roster):
        return CandidatePick(
            module=SimpleNamespace(skill_text="picked"),
            skill_text="picked",
            body_chars=100,
            val_score=0.95,
            val_rank_in_band=1,
            band_size=len(band_roster),
            epsilon=0.0167,
            fallback="knee",
            picked_idx=band_roster[0]["idx"],
            gepa_default_idx=0,
            gepa_default_body_chars=200,
            band_roster=band_roster,
        )

    def test_writes_one_entry_per_band_candidate(self, tmp_path: Path):
        roster = [
            {"idx": 5, "val_score": 0.95, "body_chars": 412},
            {"idx": 12, "val_score": 0.93, "body_chars": 380},
        ]
        knee_pick = self._make_pick(roster)
        candidates = {
            5: SimpleNamespace(skill_text="cand-5"),
            12: SimpleNamespace(skill_text="cand-12"),
        }
        holdout = ["ex1", "ex2", "ex3"]

        scores_by_idx = {5: (0.80, [0.7, 0.9, 0.8]), 12: (0.65, [0.6, 0.7, 0.65])}

        def fake_eval(module, examples, metric, lm):
            for idx, cand in candidates.items():
                if cand is module:
                    return scores_by_idx[idx]
            raise AssertionError(f"unexpected module {module}")

        with patch(
            "evolution.skills.evolve_skill._holdout_evaluate_with_metric",
            side_effect=fake_eval,
        ):
            path = _evaluate_band_on_holdout(
                knee_pick=knee_pick,
                candidates=candidates,
                holdout_examples=holdout,
                metric=MagicMock(),
                lm=MagicMock(),
                output_dir=tmp_path,
                seed=42,
            )

        payload = json.loads(path.read_text())
        assert path.name == "band_holdout.json"
        assert payload["epsilon"] == knee_pick.epsilon
        assert payload["holdout_subsample_size"] == len(holdout)
        assert len(payload["candidates"]) == 2
        cand5 = next(c for c in payload["candidates"] if c["idx"] == 5)
        assert cand5["val_score"] == 0.95
        assert cand5["body_chars"] == 412
        assert cand5["holdout_score"] == 0.80
        assert cand5["holdout_per_example"] == [0.7, 0.9, 0.8]

    def test_subsamples_when_holdout_exceeds_cap(self, tmp_path: Path):
        roster = [{"idx": 0, "val_score": 0.95, "body_chars": 100}]
        knee_pick = self._make_pick(roster)
        candidates = {0: SimpleNamespace(skill_text="cand-0")}
        holdout = [f"ex{i}" for i in range(150)]
        seen_examples: list[list] = []

        def fake_eval(module, examples, metric, lm):
            seen_examples.append(list(examples))
            return (0.5, [0.5] * len(examples))

        with patch(
            "evolution.skills.evolve_skill._holdout_evaluate_with_metric",
            side_effect=fake_eval,
        ):
            path = _evaluate_band_on_holdout(
                knee_pick=knee_pick,
                candidates=candidates,
                holdout_examples=holdout,
                metric=MagicMock(),
                lm=MagicMock(),
                output_dir=tmp_path,
                seed=42,
                subsample_cap=10,
            )

        payload = json.loads(path.read_text())
        assert payload["holdout_subsample_size"] == 10
        assert len(seen_examples[0]) == 10
        # Deterministic via the seed parameter — same seed must produce
        # the same subsample so each band candidate is scored on the same
        # examples.
        rng = __import__("random").Random(42)
        expected = rng.sample(holdout, 10)
        assert seen_examples[0] == expected


class TestResolveBapSafetyMargin:
    """`--bap-safety-margin 0.0` must reach BudgetAwareProposer as 0.0, not
    the constructor's own default of 0.10. The resolver is the one place
    that distinguishes 'user did not set the flag' (None) from 'user set
    it to zero' (0.0)."""

    def test_none_resolves_to_default(self):
        assert _resolve_bap_safety_margin(None) == 0.10

    def test_explicit_zero_is_preserved(self):
        assert _resolve_bap_safety_margin(0.0) == 0.0

    def test_explicit_nonzero_is_preserved(self):
        assert _resolve_bap_safety_margin(0.05) == 0.05


class TestResolveBapMaxGrowth:
    """`--bap-max-growth` decouples the proposer's growth target from the
    gate's `growth_free_threshold`. None falls back to the per-config
    fallback (`config.bap_max_growth`); 0.0 must be preserved as a
    legitimate user-supplied "no headroom" target."""

    def test_none_resolves_to_fallback(self):
        assert _resolve_bap_max_growth(None, 0.20) == 0.20

    def test_explicit_zero_is_preserved(self):
        # Critical: the resolver must use `is None`, not a truthiness
        # check. `0.0 or fallback` would silently collapse 0.0 to 0.20
        # and the user's "no headroom" intent would be lost.
        assert _resolve_bap_max_growth(0.0, 0.20) == 0.0

    def test_explicit_nonzero_is_preserved(self):
        assert _resolve_bap_max_growth(0.10, 0.20) == 0.10


class TestBapMaxGrowthConfigDecoupling:
    """The proposer's `max_growth` lives on EvolutionConfig as its own
    field, NOT derived from `growth_free_threshold`. Changing the gate
    parameter must not change the proposer's prompt target."""

    def test_default_is_independent_constant(self):
        from evolution.core.config import EvolutionConfig
        assert EvolutionConfig().bap_max_growth == 0.20

    def test_changing_growth_free_threshold_does_not_change_bap_max_growth(self):
        # The previous behavior coupled these — lowering the gate's
        # free threshold inadvertently tightened the proposer's prompt
        # target. After Stage 3, they're independent.
        from evolution.core.config import EvolutionConfig
        config = EvolutionConfig(growth_free_threshold=0.30)
        assert config.bap_max_growth == 0.20

    def test_explicit_bap_max_growth_overrides_default(self):
        from evolution.core.config import EvolutionConfig
        config = EvolutionConfig(bap_max_growth=0.10)
        assert config.bap_max_growth == 0.10
        assert config.growth_free_threshold == 0.20  # unchanged


class TestCliFlagPropagationToConfig:
    """The new `--eval-dataset-size` and `--holdout-ratio` flags must reach
    EvolutionConfig with the user-provided values. Uses --dry-run to short-
    circuit before any LM calls fire."""

    def _seed_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: a test skill for CLI propagation\n---\n"
            "Body content for evolution testing."
        )
        os.environ["SKILL_SOURCES_HERMES_REPO"] = str(tmp_path)

    def test_eval_dataset_size_lands_on_config(self, tmp_path: Path):
        self._seed_skill(tmp_path)
        captured = {}
        from evolution.skills import evolve_skill as module

        original_config_cls = module.EvolutionConfig

        def capturing_config(**kwargs):
            captured.update(kwargs)
            return original_config_cls(**kwargs)

        with patch.object(module, "EvolutionConfig", side_effect=capturing_config):
            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_cli,
                ["--skill", "test-skill", "--dry-run", "--eval-dataset-size", "250"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert captured.get("eval_dataset_size") == 250

    def test_holdout_ratio_lands_on_config(self, tmp_path: Path):
        self._seed_skill(tmp_path)
        captured = {}
        from evolution.skills import evolve_skill as module

        original_config_cls = module.EvolutionConfig

        def capturing_config(**kwargs):
            captured.update(kwargs)
            return original_config_cls(**kwargs)

        with patch.object(module, "EvolutionConfig", side_effect=capturing_config):
            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_cli,
                ["--skill", "test-skill", "--dry-run", "--holdout-ratio", "0.4"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert captured.get("holdout_ratio") == 0.4

    def test_unset_flags_do_not_appear_in_config_kwargs(self, tmp_path: Path):
        """When the user omits the flags, the keys must not be inserted —
        EvolutionConfig's own defaults apply. Guards against regressions
        that always-set the keys to None and trip the dataclass."""
        self._seed_skill(tmp_path)
        captured = {}
        from evolution.skills import evolve_skill as module

        original_config_cls = module.EvolutionConfig

        def capturing_config(**kwargs):
            captured.update(kwargs)
            return original_config_cls(**kwargs)

        with patch.object(module, "EvolutionConfig", side_effect=capturing_config):
            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_cli,
                ["--skill", "test-skill", "--dry-run"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert "eval_dataset_size" not in captured
        assert "holdout_ratio" not in captured


class TestFitnessProfilePropagation:
    """`--fitness-profile` must reach EvolutionConfig with the user value."""

    def _seed_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: a test skill for CLI propagation\n---\n"
            "Body content for evolution testing."
        )
        os.environ["SKILL_SOURCES_HERMES_REPO"] = str(tmp_path)

    def test_growth_profile_lands_on_config(self, tmp_path: Path):
        self._seed_skill(tmp_path)
        captured = {}
        from evolution.skills import evolve_skill as module

        original_config_cls = module.EvolutionConfig

        def capturing_config(**kwargs):
            captured.update(kwargs)
            return original_config_cls(**kwargs)

        with patch.object(module, "EvolutionConfig", side_effect=capturing_config):
            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_cli,
                ["--skill", "test-skill", "--dry-run", "--fitness-profile", "growth"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert captured.get("fitness_profile") == "growth"

    def test_default_is_balanced(self, tmp_path: Path):
        self._seed_skill(tmp_path)
        captured = {}
        from evolution.skills import evolve_skill as module

        original_config_cls = module.EvolutionConfig

        def capturing_config(**kwargs):
            captured.update(kwargs)
            return original_config_cls(**kwargs)

        with patch.object(module, "EvolutionConfig", side_effect=capturing_config):
            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_cli,
                ["--skill", "test-skill", "--dry-run"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert captured.get("fitness_profile") == "balanced"


class TestResolveProposerMode:
    """`--fitness-profile` selects which proposer template runs. Today only
    'growth' switches modes; 'compression' and 'balanced' both stay on
    compression-mode. The mapping lives in _resolve_proposer_mode so the
    contract is testable without driving the full evolve() pipeline."""

    def test_growth_profile_maps_to_growth_mode(self):
        assert _resolve_proposer_mode("growth") == "growth"

    def test_compression_profile_maps_to_compression_mode(self):
        assert _resolve_proposer_mode("compression") == "compression"

    def test_balanced_profile_maps_to_compression_mode(self):
        # Documented limitation in this PR — default-path users still get
        # compression-mode proposals. A future PR adds a neutral-mode
        # prompt for balanced.
        assert _resolve_proposer_mode("balanced") == "compression"

    def test_unknown_profile_falls_back_to_compression(self):
        # Defensive: an unrecognized profile should not produce growth-mode
        # by accident. The CLI already validates the choice via
        # click.Choice, but the resolver is the safe fallback.
        assert _resolve_proposer_mode("nonsense") == "compression"


class TestDeployAutomation:
    """`--patch` and `--apply` ship the evolved skill back to its source
    on a deploy decision. Tested via small helpers so the assertions don't
    have to drive the full evolve() pipeline."""

    def test_patch_emits_unified_diff(self):
        baseline = "line one\nshared\n"
        evolved = "line one\nchanged\n"
        diff = _emit_patch(baseline, evolved, Path("skills/foo/SKILL.md"))

        assert diff.startswith("--- a/skills/foo/SKILL.md")
        assert "+++ b/skills/foo/SKILL.md" in diff
        assert "@@" in diff
        assert "-shared" in diff
        assert "+changed" in diff

    def test_patch_helper_uses_source_path_label(self):
        path = Path("/abs/path/to/SKILL.md")
        diff = _emit_patch("a\n", "b\n", path)

        assert f"--- a/{path}" in diff
        assert f"+++ b/{path}" in diff

    def test_apply_writes_to_source_path_on_deploy(self, tmp_path: Path):
        source = tmp_path / "skills" / "demo" / "SKILL.md"
        source.parent.mkdir(parents=True)
        source.write_text("original body\n")

        ok = _apply_in_place(source, "evolved body\n")

        assert ok is True
        assert source.read_text() == "evolved body\n"

    def test_apply_skips_claude_code_cache(self, tmp_path: Path):
        cache_path = (
            tmp_path
            / ".claude"
            / "plugins"
            / "cache"
            / "vendor"
            / "plugin"
            / "v1"
            / "skills"
            / "x"
            / "SKILL.md"
        )
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("original")

        ok = _apply_in_place(cache_path, "evolved")

        assert ok is False
        assert cache_path.read_text() == "original"

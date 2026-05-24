"""Integration tests for saturation pre-flight wiring in evolve_skill.

Symmetric to tests/tools/test_evolve_tool_saturation_preflight.py.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from evolution.skills.evolve_skill import main as evolve_skill_main


@pytest.fixture
def skill_dir(tmp_path):
    """Write a minimal SKILL.md so skill discovery succeeds."""
    skills_root = tmp_path / "skills"
    skill_path = skills_root / "demo-skill"
    skill_path.mkdir(parents=True)
    (skill_path / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: a test skill\n---\n\nDo X.\n"
    )
    return skills_root


def _fake_skill_dataset(n: int = 50):
    """Build a real-shaped EvalDataset with n fake examples (no LM calls).

    Used by tests that need to flow through evolve() up to the saturation
    preflight wiring; replaces SyntheticDatasetBuilder.generate so CI runs
    with a fake OPENAI_API_KEY don't die on AuthError before reaching the
    code under test. Default n=50 gives 30/10/10 splits — the holdout
    must be ≥ EvolutionConfig.min_holdout_size (default 10) or evolve()
    aborts before the preflight wiring.
    """
    from evolution.core.dataset_builder import EvalDataset, EvalExample
    examples = [
        EvalExample(task_input=f"task {i}", expected_behavior=f"rubric {i}")
        for i in range(n)
    ]
    return EvalDataset(
        train=examples[:30], val=examples[30:40], holdout=examples[40:50],
    )


class TestSaturationPreflightCLI:
    def test_no_saturation_check_flag_skips_helper(self, skill_dir):
        with patch(
            "evolution.skills.evolve_skill.saturation_preflight"
        ) as mock_preflight, patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch("evolution.skills.evolve_skill.dspy.GEPA"):
            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--no-saturation-check", "--no-preflight"],
            )
            mock_preflight.assert_not_called()

    def test_healthy_band_does_not_prompt(self, skill_dir):
        """When preflight returns healthy: no prompt AND GEPA actually runs.

        Asserting only ``mock_confirm.assert_not_called()`` is vacuous —
        a future boolean inversion in the call site would still pass that
        assertion because CliRunner's non-TTY stdin hits the
        ``is_non_interactive`` short-circuit before reaching the confirm.
        Asserting GEPA was instantiated proves the run actually proceeded
        past the abort branch.
        """
        from evolution.core.saturation_check import SaturationReport
        from types import SimpleNamespace
        healthy = SaturationReport(
            band="healthy", holdout_score=0.5, holdout_n=10,
            holdout_per_example=[0.5] * 10, suggestions=[], thresholds={},
        )
        fake_builder = MagicMock()
        fake_builder.generate.return_value = _fake_skill_dataset()
        # Shape the fake GEPA's compile() output so the val-best path's
        # details.val_aggregate_scores[best_idx] resolves to a real float
        # and details.candidates[best_idx].skill_text resolves to a string.
        gepa_mock = MagicMock()
        fake_candidate = MagicMock()
        fake_candidate.skill_text = "evolved skill text"
        fake_optimized = MagicMock()
        fake_optimized.detailed_results = SimpleNamespace(
            candidates=[fake_candidate],
            val_aggregate_scores=[1.0],
            best_idx=0,
        )
        gepa_mock.return_value.compile.return_value = fake_optimized
        with patch(
            "evolution.skills.evolve_skill.SyntheticDatasetBuilder", return_value=fake_builder
        ), patch(
            "evolution.skills.evolve_skill.saturation_preflight", return_value=healthy
        ), patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch(
            "evolution.skills.evolve_skill.interactive_confirm"
        ) as mock_confirm, patch("evolution.skills.evolve_skill.dspy.GEPA", gepa_mock), patch(
            "evolution.skills.evolve_skill._holdout_evaluate_with_metric"
        ) as mock_holdout_eval:
            mock_holdout_eval.return_value = (0.6, [0.6] * 10)
            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--no-preflight"],
            )
            mock_confirm.assert_not_called()
            gepa_mock.assert_called_once()

    def test_saturated_band_non_interactive_aborts(self, skill_dir):
        from evolution.core.saturation_check import SaturationReport
        saturated = SaturationReport(
            band="no_headroom", holdout_score=0.99, holdout_n=50,
            holdout_per_example=[1.0] * 50, suggestions=["x"], thresholds={},
        )
        gepa_mock = MagicMock()
        fake_builder = MagicMock()
        fake_builder.generate.return_value = _fake_skill_dataset()
        with patch(
            "evolution.skills.evolve_skill.SyntheticDatasetBuilder", return_value=fake_builder
        ), patch(
            "evolution.skills.evolve_skill.saturation_preflight", return_value=saturated
        ), patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch(
            "evolution.skills.evolve_skill.is_non_interactive", return_value=True
        ), patch("evolution.skills.evolve_skill.dspy.GEPA", gepa_mock):
            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--no-preflight"],
            )
            gepa_mock.assert_not_called()
            assert "force-saturation-check" in result.output
            assert result.exit_code == 3, (
                f"Non-interactive deny should exit 3 (distinct from clean "
                f"success=0 / user errors=1), got {result.exit_code}"
            )

    def test_user_declines_at_prompt_aborts(self, skill_dir):
        """Interactive context, non-healthy band, user types 'n': prints
        'Aborted by user.', exits 0, no GEPA. Covers the
        ``if not interactive_confirm(): sys.exit(0)`` branch that has
        no other end-to-end coverage."""
        from evolution.core.saturation_check import SaturationReport
        saturated = SaturationReport(
            band="no_headroom", holdout_score=0.99, holdout_n=50,
            holdout_per_example=[1.0] * 50, suggestions=["x"], thresholds={},
        )
        fake_builder = MagicMock()
        fake_builder.generate.return_value = _fake_skill_dataset()
        gepa_mock = MagicMock()
        with patch(
            "evolution.skills.evolve_skill.SyntheticDatasetBuilder", return_value=fake_builder
        ), patch(
            "evolution.skills.evolve_skill.saturation_preflight", return_value=saturated
        ), patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch(
            "evolution.skills.evolve_skill.is_non_interactive", return_value=False
        ), patch(
            "evolution.skills.evolve_skill.interactive_confirm", return_value=False
        ), patch("evolution.skills.evolve_skill.dspy.GEPA", gepa_mock):
            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--no-preflight"],
            )
            gepa_mock.assert_not_called()
            assert "Aborted by user" in result.output
            assert result.exit_code == 0, (
                f"Interactive user-said-no abort should exit 0, got {result.exit_code}"
            )

    def test_force_saturation_check_overrides_abort(self, skill_dir):
        """--force-saturation-check on a saturated baseline in a
        non-interactive context: panel renders, confirm is bypassed, AND
        GEPA actually runs.

        Asserting only ``mock_confirm.assert_not_called()`` would be
        vacuous (the non-TTY guard exits before reaching confirm anyway);
        the GEPA-was-instantiated assertion proves the force flag
        actually overrode the abort.
        """
        from evolution.core.saturation_check import SaturationReport
        from types import SimpleNamespace
        saturated = SaturationReport(
            band="no_headroom", holdout_score=0.99, holdout_n=50,
            holdout_per_example=[1.0] * 50, suggestions=["x"], thresholds={},
        )
        fake_builder = MagicMock()
        fake_builder.generate.return_value = _fake_skill_dataset()
        # Shape the fake GEPA's compile() output so the val-best path's
        # details.val_aggregate_scores[best_idx] resolves to a real float
        # and details.candidates[best_idx].skill_text resolves to a string.
        gepa_mock = MagicMock()
        fake_candidate = MagicMock()
        fake_candidate.skill_text = "evolved skill text"
        fake_optimized = MagicMock()
        fake_optimized.detailed_results = SimpleNamespace(
            candidates=[fake_candidate],
            val_aggregate_scores=[1.0],
            best_idx=0,
        )
        gepa_mock.return_value.compile.return_value = fake_optimized
        with patch(
            "evolution.skills.evolve_skill.SyntheticDatasetBuilder", return_value=fake_builder
        ), patch(
            "evolution.skills.evolve_skill.saturation_preflight", return_value=saturated
        ), patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch(
            "evolution.skills.evolve_skill.is_non_interactive", return_value=True
        ), patch(
            "evolution.skills.evolve_skill.interactive_confirm"
        ) as mock_confirm, patch("evolution.skills.evolve_skill.dspy.GEPA", gepa_mock), patch(
            "evolution.skills.evolve_skill._holdout_evaluate_with_metric"
        ) as mock_holdout_eval:
            mock_holdout_eval.return_value = (0.6, [0.6] * 10)
            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--force-saturation-check", "--no-preflight"],
            )
            mock_confirm.assert_not_called()
            gepa_mock.assert_called_once()

    def test_cache_reuse_skips_baseline_re_eval_after_gepa(self, skill_dir):
        """When the saturation preflight runs, the cached baseline holdout
        scores must be reused at the post-GEPA evaluation site — the baseline
        module should NOT be re-scored on the holdout after GEPA finishes.
        This is the 'net cost ~zero' contract."""
        from evolution.core.saturation_check import SaturationReport
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        healthy = SaturationReport(
            band="healthy", holdout_score=0.6, holdout_n=10,
            holdout_per_example=[0.6] * 10, suggestions=[], thresholds={},
        )
        # Shape the fake GEPA's compile() output so the val-best path's
        # details.candidates[best_idx].skill_text returns a real string.
        gepa_mock = MagicMock()
        fake_candidate = SimpleNamespace(skill_text="evolved skill text")
        fake_optimized = MagicMock()
        fake_optimized.detailed_results = SimpleNamespace(
            candidates=[fake_candidate],
            val_aggregate_scores=[1.0],
            best_idx=0,
        )
        gepa_mock.return_value.compile.return_value = fake_optimized
        fake_builder = MagicMock()
        fake_builder.generate.return_value = _fake_skill_dataset()
        with patch(
            "evolution.skills.evolve_skill.SyntheticDatasetBuilder", return_value=fake_builder
        ), patch(
            "evolution.skills.evolve_skill.saturation_preflight", return_value=healthy
        ), patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch("evolution.skills.evolve_skill.dspy.GEPA", gepa_mock), patch(
            "evolution.skills.evolve_skill._holdout_evaluate_with_metric"
        ) as mock_holdout_eval:
            mock_holdout_eval.return_value = (0.6, [0.6] * 10)
            runner = CliRunner()
            runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--no-preflight"],
            )
            # With preflight populating the cache, baseline should NOT be
            # re-evaluated post-GEPA. Only evolved should be evaluated, so
            # _holdout_evaluate_with_metric is called exactly once.
            assert mock_holdout_eval.call_count == 1, (
                f"Expected baseline holdout to be reused from preflight cache "
                f"(1 call for evolved only), got {mock_holdout_eval.call_count}"
            )


class TestGepaMinibatchSizeFlag:
    """--gepa-minibatch-size threads through to dspy.GEPA's
    reflection_minibatch_size kwarg, and the post-dataset-build guard
    rejects values that exceed the trainset size with an actionable
    message instead of an opaque assertion deep inside GEPA."""

    def test_flag_passes_through_to_dspy_gepa(self, skill_dir):
        """Patch dspy.GEPA's __init__ to record the value, then invoke the
        CLI with --gepa-minibatch-size 7. Assert the constructed instance
        carries the value on the documented attribute."""
        from evolution.core.saturation_check import SaturationReport
        from types import SimpleNamespace
        captured: dict = {}
        original_init = __import__("dspy").GEPA.__init__

        def recording_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            captured["reflection_minibatch_size"] = self.reflection_minibatch_size

        healthy = SaturationReport(
            band="healthy", holdout_score=0.6, holdout_n=10,
            holdout_per_example=[0.6] * 10, suggestions=[], thresholds={},
        )
        # Shape the fake GEPA's compile() output so the val-best path's
        # details.val_aggregate_scores[best_idx] resolves to a real float
        # and details.candidates[best_idx].skill_text resolves to a string.
        fake_candidate = MagicMock()
        fake_candidate.skill_text = "evolved skill text"
        fake_module = MagicMock()
        fake_module.skill_text = "evolved skill text"
        fake_module.detailed_results = SimpleNamespace(
            candidates=[fake_candidate],
            val_aggregate_scores=[1.0],
            best_idx=0,
        )
        fake_builder = MagicMock()
        fake_builder.generate.return_value = _fake_skill_dataset()
        with patch(
            "evolution.skills.evolve_skill.SyntheticDatasetBuilder", return_value=fake_builder
        ), patch(
            "evolution.skills.evolve_skill.saturation_preflight", return_value=healthy
        ), patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch("evolution.skills.evolve_skill.dspy.GEPA.__init__", recording_init), patch(
            "evolution.skills.evolve_skill.dspy.GEPA.compile", return_value=fake_module
        ), patch(
            "evolution.skills.evolve_skill._holdout_evaluate_with_metric",
            return_value=(0.6, [0.6] * 10),
        ):
            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--no-preflight",
                 "--gepa-minibatch-size", "7"],
            )
            assert captured.get("reflection_minibatch_size") == 7, (
                f"Expected dspy.GEPA.reflection_minibatch_size=7; got "
                f"{captured!r}. CLI output: {result.output}"
            )

    def test_minibatch_exceeding_trainset_aborts_at_startup(self, skill_dir):
        """--gepa-minibatch-size larger than the trainset triggers the
        post-dataset guard (sys.exit(1) with an actionable message),
        not a mid-optimization assertion inside EpochShuffledBatchSampler."""
        from evolution.core.saturation_check import SaturationReport
        healthy = SaturationReport(
            band="healthy", holdout_score=0.6, holdout_n=10,
            holdout_per_example=[0.6] * 10, suggestions=[], thresholds={},
        )
        # _fake_skill_dataset() returns train=30 — so 1000 exceeds it.
        fake_builder = MagicMock()
        fake_builder.generate.return_value = _fake_skill_dataset()
        gepa_mock = MagicMock()
        with patch(
            "evolution.skills.evolve_skill.SyntheticDatasetBuilder", return_value=fake_builder
        ), patch(
            "evolution.skills.evolve_skill.saturation_preflight", return_value=healthy
        ), patch(
            "evolution.skills.evolve_skill._preflight_lm_credentials"
        ), patch("evolution.skills.evolve_skill.dspy.GEPA", gepa_mock):
            runner = CliRunner()
            result = runner.invoke(
                evolve_skill_main,
                ["--skill", "demo-skill", "--skill-source-dir", str(skill_dir),
                 "--iterations", "1", "--no-preflight",
                 "--gepa-minibatch-size", "1000"],
            )
            assert result.exit_code == 1, (
                f"Expected exit 1 from trainset-ceiling guard, got "
                f"{result.exit_code}. Output: {result.output}"
            )
            assert "exceeds trainset size" in result.output
            gepa_mock.assert_not_called()

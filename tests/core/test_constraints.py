"""Tests for constraint validators."""

import pytest

from evolution.core.constraints import ConstraintValidator
from evolution.core.config import EvolutionConfig


@pytest.fixture(autouse=True, scope="session")
def _skill_source_env(tmp_path_factory):
    """Pin EvolutionConfig.skill_sources to a fake repo so discovery
    doesn't pick up a real ~/.hermes or ~/.claude install on the test
    machine.
    """
    import os
    fake_repo = tmp_path_factory.mktemp("fake_skill_repo")
    (fake_repo / "skills").mkdir()
    os.environ["SKILL_SOURCES_HERMES_REPO"] = str(fake_repo)
    yield


@pytest.fixture
def validator():
    config = EvolutionConfig()
    return ConstraintValidator(config)


class TestSizeConstraints:
    def test_skill_under_limit(self, validator):
        result = validator._check_size("x" * 1000, "skill")
        assert result.passed

    def test_skill_over_limit(self, validator):
        result = validator._check_size("x" * 20_000, "skill")
        assert not result.passed
        assert "exceeded" in result.message

    def test_tool_description_under_limit(self, validator):
        result = validator._check_size("Search files by content", "tool_description")
        assert result.passed

    def test_tool_description_over_limit(self, validator):
        result = validator._check_size("x" * 600, "tool_description")
        assert not result.passed




class TestNonEmpty:
    def test_non_empty_passes(self, validator):
        result = validator._check_non_empty("some content")
        assert result.passed

    def test_empty_fails(self, validator):
        result = validator._check_non_empty("")
        assert not result.passed

    def test_whitespace_only_fails(self, validator):
        result = validator._check_non_empty("   \n  ")
        assert not result.passed


class TestSkillStructure:
    def test_valid_skill(self, validator):
        skill = "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test\nContent here"
        result = validator._check_skill_structure(skill)
        assert result.passed

    def test_missing_frontmatter(self, validator):
        skill = "# Test\nContent without frontmatter"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_missing_name(self, validator):
        skill = "---\ndescription: A test skill\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed

    def test_missing_description(self, validator):
        skill = "---\nname: test-skill\n---\n\n# Test"
        result = validator._check_skill_structure(skill)
        assert not result.passed


class TestValidateStatic:
    """Static checks only — no growth check, no quality gate."""

    def test_skips_growth_check(self, validator):
        # Even with a clearly-bloated artifact, validate_static doesn't
        # fire the growth check (it has no baseline to compare to).
        evolved = "---\nname: x\ndescription: y\n---\n" + "z" * 5000
        results = validator.validate_static(evolved, "skill")
        names = {r.constraint_name for r in results}
        assert "growth_quality_gate" not in names
        assert "growth_limit" not in names

    def test_runs_size_non_empty_structure(self, validator):
        skill = "---\nname: t\ndescription: d\n---\n\n# Body\nstuff"
        results = validator.validate_static(skill, "skill")
        names = {r.constraint_name for r in results}
        assert "size_limit" in names
        assert "non_empty" in names
        assert "skill_structure" in names


class TestGrowthQualityGate:
    """Continuous quality-gated growth check.

    Curve: required_improvement = max(0, slope * (growth - growth_free)).
    Defaults: growth_free=0.20, slope=0.30.
    """

    @staticmethod
    def _bootstrap(mean: float, lower: float, upper: float = None, n: int = 12) -> dict:
        """Build a stub bootstrap_result dict matching paired_bootstrap's shape."""
        return {
            "mean": mean,
            "lower_bound": lower,
            "upper_bound": upper if upper is not None else max(mean, lower),
            "n_examples": n,
            "n_resamples": 2000,
            "confidence": 0.90,
        }

    def test_no_regression_branch_passes_at_zero_improvement(self, validator):
        # Growth +15% (under 20% free threshold) → required = 0;
        # no_regression_only branch passes on mean ≥ 0.
        baseline = "x" * 1000
        evolved = "x" * 1150
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.0, lower=-0.05),
        )
        assert result.passed
        assert "no improvement required" in result.message

    def test_no_regression_branch_fails_on_negative_mean(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1150  # +15%, no required improvement
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=-0.05, lower=-0.10),
        )
        assert not result.passed
        assert "regression" in result.message

    def test_negative_growth_falls_into_no_regression_branch(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 800  # -20%
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.0, lower=-0.10),
        )
        assert result.passed

    def test_zero_baseline_treated_as_zero_growth(self, validator):
        result = validator._check_growth_with_quality_gate(
            "anything", "", self._bootstrap(mean=0.0, lower=0.0),
        )
        assert result.passed

    def test_dual_check_passes(self, validator):
        # Growth +40% → required = 0.30 * (0.40 - 0.20) = 0.06.
        # mean +0.07 ≥ required 0.06 AND lower +0.005 > 0 → pass.
        baseline = "x" * 1000
        evolved = "x" * 1400
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.07, lower=0.005),
        )
        assert result.passed
        assert "lower-bound +0.005 > 0" in result.message

    def test_dual_check_fails_on_mean_below_required(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1400  # +40%, required +0.06
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.04, lower=0.01),
        )
        assert not result.passed
        assert "mean +0.040 < required +0.060" in result.message

    def test_dual_check_fails_on_lower_bound_at_zero(self, validator):
        # Even a fat mean fails when the bootstrap lower bound is at the
        # noise floor — this is the regression-risk failure mode.
        baseline = "x" * 1000
        evolved = "x" * 1400  # +40%, required +0.06
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.10, lower=-0.02),
        )
        assert not result.passed
        assert "regression risk" in result.message

    def test_pr6_obsidian_data_under_principled_gate(self, validator):
        # PR #6 deployed obsidian at +24.2% growth, mean +0.015,
        # lower_bound ≈ -0.060 (per the actual paired_bootstrap on its
        # per-example scores). Under the principled gate this should
        # REJECT — that's the entire point of this PR.
        baseline = "x" * 1264
        evolved = "x" * 1570
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.015, lower=-0.060),
        )
        assert not result.passed
        assert "regression risk" in result.message


class TestNonInferiorityGate:
    """Non-inferiority decision rule (gate_mode='non_inferiority').

    When the quality slope yields required_improvement == 0 (negative or
    in-budget growth) AND gate_mode is 'non_inferiority', the gate passes
    when bootstrap.lower_bound > -inferiority_tolerance — i.e. we have
    `confidence` confidence the evolved isn't worse than baseline by more
    than the tolerance. This is the rule that ships compression-without-
    regression candidates that the default no_regression rule rejects.
    """

    @staticmethod
    def _validator(tolerance: float):
        config = EvolutionConfig(
            gate_mode="non_inferiority",
            inferiority_tolerance=tolerance,
        )
        return ConstraintValidator(config)

    @staticmethod
    def _bootstrap(mean: float, lower: float, upper: float = None, n: int = 23) -> dict:
        return {
            "mean": mean,
            "lower_bound": lower,
            "upper_bound": upper if upper is not None else max(mean, lower),
            "n_examples": n,
            "n_resamples": 2000,
            "confidence": 0.90,
        }

    def test_passes_when_lower_bound_within_tolerance(self):
        # codebase-summary case: mean=-0.0006, lower=-0.007, tol=0.02 → PASS
        validator = self._validator(tolerance=0.02)
        baseline = "x" * 14_827
        evolved = "x" * 6_980  # negative growth
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=-0.0006, lower=-0.007),
        )
        assert result.passed
        assert "non-inferior" in result.message

    def test_passes_at_exact_tolerance_boundary(self):
        validator = self._validator(tolerance=0.02)
        baseline = "x" * 1000
        evolved = "x" * 800
        # lower_bound exactly at -tolerance → boundary inclusive
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=-0.01, lower=-0.02),
        )
        assert result.passed

    def test_fails_when_lower_bound_below_tolerance(self):
        # arxiv-style: mean=-0.023, lower=-0.135, tol=0.02 → FAIL (lower < -tol)
        validator = self._validator(tolerance=0.02)
        baseline = "x" * 10_036
        evolved = "x" * 4_777
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=-0.023, lower=-0.135),
        )
        assert not result.passed
        assert "lower-bound" in result.message
        assert "tolerance" in result.message

    def test_passes_with_negative_mean_when_lower_within_tolerance(self):
        # Distinguishes non_inferiority from no_regression_only: a negative
        # mean would fail no_regression but passes here when CI is tight.
        validator = self._validator(tolerance=0.05)
        baseline = "x" * 1000
        evolved = "x" * 800
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=-0.03, lower=-0.04),
        )
        assert result.passed  # would fail under no_regression_only (mean < 0)

    def test_zero_tolerance_reduces_to_lower_strictly_above_zero(self):
        # tolerance=0 → require lower >= 0 (matches Statsig "no regression risk")
        validator = self._validator(tolerance=0.0)
        baseline = "x" * 1000
        evolved = "x" * 800
        # lower exactly 0 → passes (>= -0)
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.01, lower=0.0),
        )
        assert result.passed
        # lower slightly negative → fails
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.01, lower=-0.001),
        )
        assert not result.passed

    def test_dual_check_branch_unaffected_by_gate_mode(self):
        # Even with gate_mode=non_inferiority, dual_check fires when
        # growth exceeds the free threshold (required > 0).
        validator = self._validator(tolerance=0.02)
        baseline = "x" * 1000
        evolved = "x" * 1400  # +40% → required = 0.30 * 0.20 = 0.06
        # Pass dual_check: mean ≥ required AND lower > 0
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.07, lower=0.005),
        )
        assert result.passed
        # Fail dual_check: lower ≤ 0 (regression risk) — non_inferiority
        # tolerance does NOT apply here because required > 0
        result = validator._check_growth_with_quality_gate(
            evolved, baseline, self._bootstrap(mean=0.10, lower=-0.01),
        )
        assert not result.passed
        assert "regression risk" in result.message


class TestResolveDecisionRule:
    """Single-source-of-truth helper for the decision rule string.

    Both the constraint and the gate_decision.json payload writer must
    agree on the rule name; computing it in one place avoids drift.
    """

    def test_dual_check_when_required_positive(self):
        config = EvolutionConfig()  # default slope=0.30, free=0.20
        # Growth 0.50 → required = 0.30 * 0.30 = 0.09 > 0
        from evolution.core.constraints import resolve_decision_rule
        assert resolve_decision_rule(config, growth_pct=0.50) == "dual_check"

    def test_no_regression_only_default_when_required_zero(self):
        config = EvolutionConfig()
        from evolution.core.constraints import resolve_decision_rule
        assert resolve_decision_rule(config, growth_pct=0.0) == "no_regression_only"
        assert resolve_decision_rule(config, growth_pct=-0.5) == "no_regression_only"

    def test_non_inferiority_when_gate_mode_set_and_required_zero(self):
        config = EvolutionConfig(gate_mode="non_inferiority", inferiority_tolerance=0.02)
        from evolution.core.constraints import resolve_decision_rule
        assert resolve_decision_rule(config, growth_pct=0.0) == "non_inferiority"
        assert resolve_decision_rule(config, growth_pct=-0.5) == "non_inferiority"

    def test_dual_check_overrides_gate_mode(self):
        # gate_mode only matters in the required==0 branch; dual_check fires
        # regardless when required > 0.
        config = EvolutionConfig(gate_mode="non_inferiority", inferiority_tolerance=0.02)
        from evolution.core.constraints import resolve_decision_rule
        assert resolve_decision_rule(config, growth_pct=0.50) == "dual_check"


class TestAbsoluteCharCeiling:
    def test_under_ceiling_passes(self, validator):
        # Tiny baseline → static 5000 floor applies. 4000 < 5000.
        result = validator._check_absolute_chars("x" * 4000, baseline_chars=1000)
        assert result.passed

    def test_over_ceiling_fails(self, validator):
        # Tiny baseline → static 5000 floor applies. 6000 > 5000 fails.
        result = validator._check_absolute_chars("x" * 6000, baseline_chars=1000)
        assert not result.passed
        assert "1000 over" in result.message

    def test_static_floor_holds_for_small_baselines(self, validator):
        # 100-char baseline × 1.5 = 150, but the static 5000 floor wins.
        # Without this, a tiny baseline could blow up to 4900 chars
        # despite the absolute ceiling — defeating the backstop.
        result = validator._check_absolute_chars("x" * 4500, baseline_chars=100)
        assert result.passed
        assert "5000" in result.message  # the static floor, not 150

    def test_ceiling_scales_with_large_baseline(self, validator):
        # 10000-char baseline × 1.5 = 15000. A 12000-char artifact
        # passes that wouldn't have passed at the static 5000 floor.
        result = validator._check_absolute_chars("x" * 12000, baseline_chars=10000)
        assert result.passed
        assert "15000" in result.message

    def test_scaled_ceiling_still_rejects_runaway(self, validator):
        # 10000-char baseline × 1.5 = 15000 effective ceiling. A 16000-
        # char artifact still rejects — scaling caps at 1.5×, not infinity.
        result = validator._check_absolute_chars("x" * 16000, baseline_chars=10000)
        assert not result.passed
        assert "1000 over" in result.message
        assert "15000" in result.message


class TestValidateGrowthWithQuality:
    """The combined entry point: growth-with-quality + absolute ceiling."""

    def test_runs_both_checks(self, validator):
        baseline = "x" * 1000
        evolved = "x" * 1100  # safe growth, safe absolute
        bootstrap = TestGrowthQualityGate._bootstrap(mean=0.0, lower=-0.02)
        results = validator.validate_growth_with_quality(evolved, baseline, bootstrap)
        names = {r.constraint_name for r in results}
        assert "growth_quality_gate" in names
        assert "absolute_char_ceiling" in names

    def test_absolute_ceiling_blocks_even_when_growth_ok(self, validator):
        # +5900% growth from a tiny baseline. With baseline=100 the
        # dynamic ceiling (1.5 × 100 = 150) collapses to the static
        # 5000 floor; 6000 chars exceeds 5000 → reject on the floor.
        baseline = "x" * 100
        evolved = "x" * 6000
        bootstrap = TestGrowthQualityGate._bootstrap(mean=0.0, lower=-0.05)
        results = validator.validate_growth_with_quality(evolved, baseline, bootstrap)
        ceiling_result = next(r for r in results if r.constraint_name == "absolute_char_ceiling")
        assert not ceiling_result.passed
        # Pin which ceiling fired: the static floor, not the dynamic
        # scale-up. A future regression that silently swaps which value
        # drives the rejection (e.g., dropping the static-floor backstop)
        # gets caught here.
        assert "5000" in ceiling_result.message

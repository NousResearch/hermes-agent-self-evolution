"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be
considered valid. Failed constraints = immediate rejection.
"""

import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig


@dataclass
class ConstraintResult:
    """Result of constraint validation."""
    passed: bool
    constraint_name: str
    message: str
    details: Optional[str] = None


# How much the absolute-char ceiling scales with baseline size. The
# effective ceiling is max(config.max_absolute_chars, multiplier ×
# baseline_chars). A 1.5× multiplier matches the lenient preset's
# growth_free_threshold (0.30) plus headroom for the bootstrap to land
# on a slightly larger evolved variant — empirically: linear at 11185
# chars needs ~16778 to fit observed +30% compression-direction
# variants without auto-rejecting on absolute size.
_BASELINE_CEILING_MULTIPLIER = 1.5


def effective_absolute_char_ceiling(static_ceiling: int, baseline_chars: int) -> int:
    """Per-run ceiling: the larger of the static config floor and a
    baseline-relative scale. The static value remains a backstop for
    skills with tiny baselines; scaling kicks in for larger skills."""
    return max(static_ceiling, int(_BASELINE_CEILING_MULTIPLIER * baseline_chars))


def resolve_decision_rule(config: EvolutionConfig, growth_pct: float) -> str:
    """Single source of truth for which gate decision rule applies.

    Both ``_check_growth_with_quality_gate`` and ``evolve_skill._write_gate_decision``
    must agree on the rule name; computing it in one place avoids the float-equality
    drift that otherwise creeps in between the constraint and the payload writer.
    """
    required = max(
        0.0,
        config.growth_quality_slope * (growth_pct - config.growth_free_threshold),
    )
    if required > 0.0:
        return "dual_check"
    if config.gate_mode == "non_inferiority":
        return "non_inferiority"
    return "no_regression_only"


class ConstraintValidator:
    """Validates evolved artifacts against hard constraints."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def validate_static(
        self,
        artifact_text: str,
        artifact_type: str,
    ) -> list[ConstraintResult]:
        """Static checks that don't need a baseline or holdout signal:
        size, non-empty, structural integrity. Runs first in the evolve
        flow so broken artifacts short-circuit before judge-call spend.
        """
        results = [self._check_size(artifact_text, artifact_type)]
        results.append(self._check_non_empty(artifact_text))
        if artifact_type == "skill":
            results.append(self._check_skill_structure(artifact_text))
        return results

    def validate_growth_with_quality(
        self,
        artifact_text: str,
        baseline_text: str,
        bootstrap_result: dict,
    ) -> list[ConstraintResult]:
        """Bootstrap-gated growth check + absolute-char ceiling.

        bootstrap_result is the dict returned by
        evolution.core.stats.paired_bootstrap on the per-example
        holdout improvement scores. The growth check uses a continuous
        curve where required holdout improvement scales linearly with
        growth above the free threshold; the gate passes when both the
        sample mean meets the requirement AND the bootstrap lower bound
        is positive (no-regression). The absolute-char ceiling is
        independent of growth — applies even when the curve passes — to
        backstop runaway absolute size.
        """
        return [
            self._check_growth_with_quality_gate(
                artifact_text, baseline_text, bootstrap_result
            ),
            self._check_absolute_chars(artifact_text, len(baseline_text)),
        ]

    def run_test_suite(self, repo_path: Path) -> ConstraintResult:
        """Run the target repo's full pytest suite. Must pass 100%."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(repo_path),
            )

            if result.returncode == 0:
                return ConstraintResult(
                    passed=True,
                    constraint_name="test_suite",
                    message="All tests passed",
                    details=result.stdout.strip().split("\n")[-1] if result.stdout else "",
                )
            else:
                last_lines = result.stdout.strip().split("\n")[-5:] if result.stdout else []
                return ConstraintResult(
                    passed=False,
                    constraint_name="test_suite",
                    message="Test suite failed",
                    details="\n".join(last_lines),
                )
        except subprocess.TimeoutExpired:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message="Test suite timed out (300s)",
            )
        except Exception as e:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=f"Failed to run tests: {e}",
            )

    def _check_size(self, text: str, artifact_type: str) -> ConstraintResult:
        size = len(text)
        if artifact_type == "skill":
            limit = self.config.max_skill_size
        elif artifact_type == "tool_description":
            limit = self.config.max_tool_desc_size
        elif artifact_type == "param_description":
            limit = self.config.max_param_desc_size
        else:
            limit = self.config.max_skill_size

        if size <= limit:
            return ConstraintResult(
                passed=True,
                constraint_name="size_limit",
                message=f"Size OK: {size}/{limit} chars",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="size_limit",
                message=f"Size exceeded: {size}/{limit} chars ({size - limit} over)",
            )

    def _check_growth_with_quality_gate(
        self,
        text: str,
        baseline: str,
        bootstrap_result: dict,
    ) -> ConstraintResult:
        """Bootstrap-gated quality check on the holdout improvement.

        Required improvement scales linearly with growth above the free
        threshold:
            required_improvement(growth) = max(0, slope * (growth - growth_free))

        Three decision rules (resolved by ``resolve_decision_rule``):
          - ``dual_check`` (required > 0): pass when both the sample mean
            meets the requirement AND the bootstrap lower bound is positive
            (Statsig "guardrail" pattern: effect size + no-regression).
          - ``no_regression_only`` (required == 0, gate_mode="no_regression"):
            pass on mean ≥ 0 alone. The optimizer doesn't need to justify a
            decrease in length, only that the candidate doesn't regress on
            average.
          - ``non_inferiority`` (required == 0, gate_mode="non_inferiority"):
            pass when bootstrap.lower_bound > -inferiority_tolerance — i.e.
            we have ``confidence`` confidence the evolved isn't worse than
            baseline by more than ``inferiority_tolerance``. Necessary at
            small N where the bootstrap CI swamps tiny effects.

        Negative growth (the LM produced a shorter artifact) always falls
        into the required==0 branch.
        """
        baseline_len = len(baseline)
        growth = (len(text) - baseline_len) / baseline_len if baseline_len else 0.0

        rule = resolve_decision_rule(self.config, growth)

        mean = bootstrap_result["mean"]
        lower = bootstrap_result["lower_bound"]
        n = bootstrap_result["n_examples"]
        conf = bootstrap_result["confidence"]

        if rule == "non_inferiority":
            tol = self.config.inferiority_tolerance
            if lower >= -tol:
                return ConstraintResult(
                    passed=True,
                    constraint_name="growth_quality_gate",
                    message=(
                        f"Growth {growth:+.1%}: non-inferior — lower-bound "
                        f"{lower:+.3f} ≥ -{tol:.3f} tolerance (n={n}, conf={conf:.2f})"
                    ),
                )
            return ConstraintResult(
                passed=False,
                constraint_name="growth_quality_gate",
                message=(
                    f"Growth {growth:+.1%}: lower-bound {lower:+.3f} < "
                    f"-{tol:.3f} tolerance (n={n}, conf={conf:.2f})"
                ),
            )

        if rule == "no_regression_only":
            if mean >= 0.0:
                return ConstraintResult(
                    passed=True,
                    constraint_name="growth_quality_gate",
                    message=(
                        f"Growth {growth:+.1%}: no improvement required; "
                        f"mean {mean:+.3f} ≥ 0 (n={n})"
                    ),
                )
            return ConstraintResult(
                passed=False,
                constraint_name="growth_quality_gate",
                message=(
                    f"Growth {growth:+.1%}: regression — mean {mean:+.3f} < 0 (n={n})"
                ),
            )

        # rule == "dual_check"
        required = max(
            0.0,
            self.config.growth_quality_slope * (growth - self.config.growth_free_threshold),
        )
        mean_ok = mean >= required
        lower_ok = lower > 0.0

        if mean_ok and lower_ok:
            return ConstraintResult(
                passed=True,
                constraint_name="growth_quality_gate",
                message=(
                    f"Growth {growth:+.1%}: mean {mean:+.3f} ≥ required {required:+.3f}; "
                    f"lower-bound {lower:+.3f} > 0 (n={n}, conf={conf:.2f})"
                ),
            )
        if not mean_ok:
            return ConstraintResult(
                passed=False,
                constraint_name="growth_quality_gate",
                message=(
                    f"Growth {growth:+.1%}: mean {mean:+.3f} < required {required:+.3f} (n={n})"
                ),
            )
        return ConstraintResult(
            passed=False,
            constraint_name="growth_quality_gate",
            message=(
                f"Growth {growth:+.1%}: lower-bound {lower:+.3f} ≤ 0 "
                f"(regression risk; n={n}, conf={conf:.2f})"
            ),
        )

    def _check_absolute_chars(self, text: str, baseline_chars: int) -> ConstraintResult:
        """Hard absolute-char ceiling on the evolved artifact.

        Independent of growth. Backstops runaway absolute size that the
        relative growth curve can't catch (e.g., a 200-char baseline
        growing to 1500 chars is +650% but only 1500 chars absolute).

        Ceiling scales with baseline size — a 10000-char baseline gets
        a 15000-char effective ceiling, not the static 5000 floor. The
        static config value remains a hard backstop for tiny baselines
        where 1.5× would shrink below the floor.
        """
        size = len(text)
        ceiling = effective_absolute_char_ceiling(
            self.config.max_absolute_chars, baseline_chars,
        )
        if size <= ceiling:
            return ConstraintResult(
                passed=True,
                constraint_name="absolute_char_ceiling",
                message=f"Size {size} ≤ {ceiling} chars (absolute ceiling)",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="absolute_char_ceiling",
            message=f"Size {size} exceeds absolute ceiling {ceiling} chars ({size - ceiling} over)",
        )

    def _check_non_empty(self, text: str) -> ConstraintResult:
        if text.strip():
            return ConstraintResult(
                passed=True,
                constraint_name="non_empty",
                message="Artifact is non-empty",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="non_empty",
                message="Artifact is empty",
            )

    def _check_skill_structure(self, text: str) -> ConstraintResult:
        """Check that a skill file has valid YAML frontmatter and markdown body."""
        has_frontmatter = text.strip().startswith("---")
        has_name = "name:" in text[:500] if has_frontmatter else False
        has_description = "description:" in text[:500] if has_frontmatter else False

        if has_frontmatter and has_name and has_description:
            return ConstraintResult(
                passed=True,
                constraint_name="skill_structure",
                message="Skill has valid frontmatter (name + description)",
            )
        else:
            missing = []
            if not has_frontmatter:
                missing.append("YAML frontmatter (---)")
            if not has_name:
                missing.append("name field")
            if not has_description:
                missing.append("description field")
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message=f"Skill missing: {', '.join(missing)}",
            )

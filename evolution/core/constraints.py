"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be
considered valid. Failed constraints = immediate rejection.

Two things changed here relative to the original design.

The size limit is no longer a fixed number. It is derived from the installed
skill corpus and floored at the artifact's own baseline, because a hardcoded
15 KB cap disqualified 27 of the 201 shipped skills before the optimizer even
started.

Constraints are also no longer the *only* place size is enforced. They remain
a final gate, but the objective now carries size pressure during the search
(see :mod:`evolution.core.objectives`) — a gate that fires only after the
budget is spent cannot prevent the failure it detects.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig
from evolution.core.skill_bundle import SkillBundle, broken_references, invented_references


@dataclass
class ConstraintResult:
    """Result of constraint validation."""
    passed: bool
    constraint_name: str
    message: str
    details: Optional[str] = None


class ConstraintValidator:
    """Validates evolved artifacts against hard constraints."""

    def __init__(self, config: EvolutionConfig, size_budget: Optional[int] = None):
        self.config = config
        # Corpus-derived when supplied; otherwise the configured fallback.
        self.size_budget = size_budget

    def budget_for(self, artifact_type: str) -> int:
        if artifact_type == "tool_description":
            return self.config.max_tool_desc_size
        if artifact_type == "param_description":
            return self.config.max_param_desc_size
        return self.size_budget or self.config.max_skill_size

    def validate_all(
        self,
        artifact_text: str,
        artifact_type: str,
        baseline_text: Optional[str] = None,
        bundle: Optional[SkillBundle] = None,
    ) -> list[ConstraintResult]:
        """Run all applicable constraints. Returns list of results."""
        results = []

        # 1. Size limits
        results.append(self._check_size(artifact_text, artifact_type))

        # 2. Growth limit (if baseline provided)
        if baseline_text:
            results.append(self._check_growth(artifact_text, baseline_text, artifact_type))

        # 3. Non-empty
        results.append(self._check_non_empty(artifact_text))

        # 4. Structural integrity
        if artifact_type == "skill":
            results.append(self._check_skill_structure(artifact_text))

        # 5. Supporting-file references survive the rewrite. Nearly half the
        #    installed skills ship reference files; dropping a link to one is
        #    a real regression that reads as a size win to every other check.
        if bundle is not None and bundle.is_bundle:
            results.append(self._check_references(artifact_text, bundle))

        return results

    def run_test_suite(self, hermes_repo: Path) -> ConstraintResult:
        """Run the full hermes-agent test suite. Must pass 100%."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=self.config.test_timeout_s,
                cwd=str(hermes_repo),
            )

            if result.returncode == 0:
                return ConstraintResult(
                    passed=True,
                    constraint_name="test_suite",
                    message="All tests passed",
                    details=result.stdout.strip().split("\n")[-1] if result.stdout else "",
                )
            else:
                # Extract failure summary
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
                message=f"Test suite timed out ({self.config.test_timeout_s}s)",
            )
        except Exception as e:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=f"Failed to run tests: {e}",
            )

    def _check_size(self, text: str, artifact_type: str) -> ConstraintResult:
        size = len(text)
        limit = self.budget_for(artifact_type)

        if size <= limit:
            return ConstraintResult(
                passed=True,
                constraint_name="size_limit",
                message=f"Size OK: {size:,}/{limit:,} chars",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="size_limit",
            message=f"Size exceeded: {size:,}/{limit:,} chars ({size - limit:,} over)",
        )

    def _check_growth(self, text: str, baseline: str, artifact_type: str) -> ConstraintResult:
        growth = (len(text) - len(baseline)) / max(1, len(baseline))
        max_growth = self.config.max_prompt_growth

        if growth <= max_growth:
            return ConstraintResult(
                passed=True,
                constraint_name="growth_limit",
                message=f"Growth OK: {growth:+.1%} (max {max_growth:+.1%})",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="growth_limit",
            message=f"Growth exceeded: {growth:+.1%} (max {max_growth:+.1%})",
        )

    def _check_non_empty(self, text: str) -> ConstraintResult:
        if text.strip():
            return ConstraintResult(
                passed=True,
                constraint_name="non_empty",
                message="Artifact is non-empty",
            )
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

    def _check_references(self, text: str, bundle: SkillBundle) -> ConstraintResult:
        """Supporting files must stay linked, and no new ones invented."""
        lost = broken_references(bundle, text)
        invented = invented_references(bundle, text)

        if not lost and not invented:
            return ConstraintResult(
                passed=True,
                constraint_name="bundle_references",
                message=f"All {len(bundle.referenced_files)} supporting file link(s) intact",
            )

        problems = []
        if lost:
            problems.append(f"dropped link(s) to {', '.join(lost)}")
        if invented:
            problems.append(f"points at nonexistent {', '.join(invented)}")
        return ConstraintResult(
            passed=False,
            constraint_name="bundle_references",
            message="Supporting files broken: " + "; ".join(problems),
        )

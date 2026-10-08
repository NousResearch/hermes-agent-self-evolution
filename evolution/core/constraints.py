"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be
considered valid. Failed constraints = immediate rejection.
"""

import subprocess
import re
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


class ConstraintValidator:
    """Validates evolved artifacts against hard constraints."""

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def validate_all(
        self,
        artifact_text: str,
        artifact_type: str,
        baseline_text: Optional[str] = None,
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

        # 5. Semantic preservation (if baseline provided) — PLAN.md guardrail #4
        if baseline_text and self.config.semantic_preservation_enabled:
            results.append(self._check_semantic_preservation(artifact_text, baseline_text))

        return results

    def run_test_suite(self, hermes_repo: Path) -> ConstraintResult:
        """Run the full hermes-agent test suite. Must pass 100%."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=300,
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
            limit = self.config.max_skill_size  # Default

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

    def _check_growth(self, text: str, baseline: str, artifact_type: str) -> ConstraintResult:
        growth = (len(text) - len(baseline)) / max(1, len(baseline))
        max_growth = self.config.max_prompt_growth

        if growth <= max_growth:
            return ConstraintResult(
                passed=True,
                constraint_name="growth_limit",
                message=f"Growth OK: {growth:+.1%} (max {max_growth:+.1%})",
            )
        else:
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
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="non_empty",
                message="Artifact is empty",
            )

    def _check_semantic_preservation(self, text: str, baseline: str) -> ConstraintResult:
        """Reject evolved text that has drifted out of its original domain.

        PLAN.md guardrail #4: "A skill for 'GitHub code review' must still
        perform code reviews, not drift into something else." This is the
        enforcement. It is deliberately a cheap, deterministic gate — content
        vocabulary retention — not an embedding or an LLM call: those are
        exactly the slow, timing-out, non-deterministic layers that already
        caused trouble in the scoring path. A drift this gross (a gate-evidence
        skill rewritten into an app-reverse-engineering prompt) loses its
        domain vocabulary almost entirely, so term retention catches it.

        Definition: strip non-alphabetic tokens, drop short stopword-ish
        fragments (<=3 chars), take the baseline's content terms, and require
        that at least min_semantic_similarity of them survive in the evolved
        text. A body that still contains its domain words passes; one that has
        been replaced wholesale fails.
        """
        def content_terms(s: str) -> set[str]:
            words = re.findall(r"[a-z]{4,}", s.lower())
            return set(words)

        baseline_terms = content_terms(baseline)
        evolved_terms = content_terms(text)

        if not baseline_terms:
            # Nothing meaningful to preserve (e.g. a near-empty baseline body);
            # a drift gate with no anchor cannot reject anything, so pass.
            return ConstraintResult(
                passed=True,
                constraint_name="semantic_preservation",
                message="No content terms in baseline to preserve",
            )

        retained = baseline_terms & evolved_terms
        ratio = len(retained) / len(baseline_terms)
        threshold = self.config.min_semantic_similarity

        if ratio >= threshold:
            return ConstraintResult(
                passed=True,
                constraint_name="semantic_preservation",
                message=(
                    f"Semantic OK: {ratio:.0%} of baseline terms retained "
                    f"({len(retained)}/{len(baseline_terms)})"
                ),
            )

        dropped = sorted(baseline_terms - evolved_terms)
        sample = ", ".join(dropped[:8])
        return ConstraintResult(
            passed=False,
            constraint_name="semantic_preservation",
            message=(
                f"Drift: only {ratio:.0%} of baseline terms retained "
                f"(need {threshold:.0%}); lost e.g. {sample}"
            ),
        )

    def _check_skill_structure(self, text: str) -> ConstraintResult:
        """Check that a skill file has valid YAML frontmatter and markdown body.

        Accepts either a full skill file (frontmatter + body) or a bare body.
        A bare body carries no frontmatter by definition, so it is judged on
        substance alone — otherwise every evolved variant fails this gate,
        because the caller optimizes the frontmatter-stripped body and only
        reassembles frontmatter afterwards.
        """
        stripped = text.strip()

        if not stripped.startswith("---"):
            # Frontmatter-stripped body: structure cannot be judged here.
            # Require real substance instead, so this is not a free pass.
            if len(stripped) >= 200 and stripped.lstrip().startswith("#"):
                return ConstraintResult(
                    passed=True,
                    constraint_name="skill_structure",
                    message="Body has a heading and substantive content (frontmatter checked separately)",
                )
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message="Body lacks frontmatter (expected here) and has no substantive heading content",
            )

        has_name = "name:" in text[:500]
        has_description = "description:" in text[:500]

        if has_name and has_description:
            return ConstraintResult(
                passed=True,
                constraint_name="skill_structure",
                message="Skill has valid frontmatter (name + description)",
            )
        else:
            missing = []
            if not has_name:
                missing.append("name field")
            if not has_description:
                missing.append("description field")
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message=f"Skill missing: {', '.join(missing)}",
            )

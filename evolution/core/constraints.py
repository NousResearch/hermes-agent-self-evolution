"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be
considered valid. Failed constraints = immediate rejection.
"""

import re
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig


CRITICAL_DOMAIN_TERMS = (
    "Stephen",
    "Babbage",
    "Lovelace",
    "Turing",
    "Ops",
    "Kanban",
    "Hermes",
    "Honcho",
    "LCM",
    "OpenClaw",
)

IRRELEVANT_TOOL_REFERENCES = (
    "code_interpreter",
    "database_api",
    "api.update_status",
    "Project Gamma",
)


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

        # 2. Growth/shrinkage limits (if baseline provided)
        if baseline_text:
            results.append(self._check_growth(artifact_text, baseline_text, artifact_type))
            if artifact_type == "skill":
                results.append(self._check_shrinkage(artifact_text, baseline_text, artifact_type))

        # 3. Non-empty
        results.append(self._check_non_empty(artifact_text))

        # 4. Structural integrity
        if artifact_type == "skill":
            results.append(self._check_skill_structure(artifact_text))

        # 5. Baseline-relative safety gates for skill candidates.
        if artifact_type == "skill" and baseline_text:
            results.append(self._check_heading_retention(artifact_text, baseline_text))
            results.append(self._check_reference_retention(artifact_text, baseline_text))
            results.append(self._check_domain_term_retention(artifact_text, baseline_text))
            results.append(self._check_new_irrelevant_tool_references(artifact_text, baseline_text))

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

    def _check_shrinkage(self, text: str, baseline: str, artifact_type: str) -> ConstraintResult:
        """Reject candidates that compress mature skills too aggressively."""
        if artifact_type != "skill":
            return ConstraintResult(
                passed=True,
                constraint_name="shrinkage_limit",
                message="Shrinkage gate skipped for non-skill artifact",
            )

        shrinkage = (len(baseline) - len(text)) / max(1, len(baseline))
        max_shrink = self.config.max_prompt_shrink

        if shrinkage <= max_shrink:
            return ConstraintResult(
                passed=True,
                constraint_name="shrinkage_limit",
                message=f"Shrinkage OK: {shrinkage:+.1%} (max {max_shrink:+.1%})",
            )
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="shrinkage_limit",
                message=f"Shrinkage exceeded: {shrinkage:+.1%} (max {max_shrink:+.1%})",
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

    def _check_heading_retention(self, text: str, baseline: str) -> ConstraintResult:
        """Require top-level baseline skill headings to survive evolution."""
        baseline_headings = self._extract_required_headings(baseline)
        if not baseline_headings:
            return ConstraintResult(
                passed=True,
                constraint_name="heading_retention",
                message="No baseline headings to preserve",
            )

        evolved_headings = {self._normalize_heading(h) for h in self._extract_required_headings(text)}
        missing = [h for h in baseline_headings if self._normalize_heading(h) not in evolved_headings]
        if not missing:
            return ConstraintResult(
                passed=True,
                constraint_name="heading_retention",
                message="Required headings preserved",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="heading_retention",
            message=f"Missing {len(missing)} required heading(s)",
            details=", ".join(missing),
        )

    def _check_reference_retention(self, text: str, baseline: str) -> ConstraintResult:
        """Require baseline references/*.md links to remain present."""
        baseline_refs = self._extract_reference_paths(baseline)
        if not baseline_refs:
            return ConstraintResult(
                passed=True,
                constraint_name="reference_retention",
                message="No baseline reference links to preserve",
            )
        evolved_refs = set(self._extract_reference_paths(text))
        missing = [ref for ref in baseline_refs if ref not in evolved_refs]
        if not missing:
            return ConstraintResult(
                passed=True,
                constraint_name="reference_retention",
                message="Reference links preserved",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="reference_retention",
            message=f"Missing {len(missing)} reference link(s)",
            details=", ".join(missing),
        )

    def _check_domain_term_retention(self, text: str, baseline: str) -> ConstraintResult:
        """Preserve Stephen-specific org/product terms already in the baseline."""
        baseline_terms = [term for term in CRITICAL_DOMAIN_TERMS if self._contains_term(baseline, term)]
        if not baseline_terms:
            return ConstraintResult(
                passed=True,
                constraint_name="domain_term_retention",
                message="No critical domain terms to preserve",
            )
        missing = [term for term in baseline_terms if not self._contains_term(text, term)]
        if not missing:
            return ConstraintResult(
                passed=True,
                constraint_name="domain_term_retention",
                message="Critical domain terms preserved",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="domain_term_retention",
            message=f"Missing {len(missing)} critical domain term(s)",
            details=", ".join(missing),
        )

    def _check_new_irrelevant_tool_references(self, text: str, baseline: str) -> ConstraintResult:
        """Reject suspicious generic benchmark/tool references introduced by the optimizer."""
        baseline_lower = baseline.lower()
        text_lower = text.lower()
        introduced = [
            term
            for term in IRRELEVANT_TOOL_REFERENCES
            if term.lower() in text_lower and term.lower() not in baseline_lower
        ]
        if not introduced:
            return ConstraintResult(
                passed=True,
                constraint_name="irrelevant_tool_references",
                message="No new irrelevant tool references introduced",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="irrelevant_tool_references",
            message=f"Introduced {len(introduced)} irrelevant tool reference(s)",
            details=", ".join(introduced),
        )

    @staticmethod
    def _extract_required_headings(text: str) -> list[str]:
        headings: list[str] = []
        for line in text.splitlines():
            match = re.match(r"^(#{1,2})\s+(.+?)\s*$", line)
            if match:
                headings.append(match.group(2).strip().strip("#").strip())
        return headings

    @staticmethod
    def _normalize_heading(heading: str) -> str:
        return re.sub(r"\s+", " ", heading.strip().lower())

    @staticmethod
    def _extract_reference_paths(text: str) -> list[str]:
        return sorted(set(re.findall(r"references/[A-Za-z0-9_./-]+\.md", text)))

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        return re.search(rf"(?<![A-Za-z0-9_-]){re.escape(term)}(?![A-Za-z0-9_-])", text) is not None

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
        if baseline_text is not None:
            results.append(self._check_growth(artifact_text, baseline_text, artifact_type))

        # 3. Non-empty
        results.append(self._check_non_empty(artifact_text))

        # 4. Structural integrity
        if artifact_type == "skill":
            results.append(self._check_skill_structure(artifact_text))

        return results

    def run_test_suite(self, hermes_repo: Path) -> ConstraintResult:
        """Run the full hermes-agent test suite. Must pass 100%.

        Refuses to run if `hermes_repo` does not look like a real hermes-agent
        checkout. Pytest auto-discovers and executes `conftest.py`, so pointing
        at an untrusted tree is equivalent to executing arbitrary Python.
        """
        try:
            hermes_repo = Path(hermes_repo).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=f"hermes-agent path is invalid: {exc}",
            )

        # Sanity-check the path looks like a hermes-agent checkout. We do not
        # try to fully validate authenticity — that is a tree-of-trust problem
        # — but we do reject obvious mistakes like pointing at /etc or at an
        # unrelated project.
        pyproject = hermes_repo / "pyproject.toml"
        tests_dir = hermes_repo / "tests"
        if not pyproject.exists() or not tests_dir.exists():
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=(
                    f"{hermes_repo} does not look like a hermes-agent checkout "
                    "(missing pyproject.toml or tests/ directory)."
                ),
            )
        try:
            project_meta = pyproject.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=f"Cannot read {pyproject}: {exc}",
            )
        if "hermes-agent" not in project_meta and "hermes_agent" not in project_meta:
            return ConstraintResult(
                passed=False,
                constraint_name="test_suite",
                message=(
                    f"{pyproject} does not reference hermes-agent — refusing "
                    "to run pytest in an unrelated project."
                ),
            )

        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(hermes_repo),
                check=False,
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

    def _check_skill_structure(self, text: str) -> ConstraintResult:
        """Check that a skill has valid frontmatter and/or meaningful body structure.

        `validate_skill_constraints()` passes a full SKILL.md here for structural
        validation while keeping size/growth checks on the mutable body. Keep this
        method backward-compatible with direct full-file callers from the original
        test suite, and reject body-only text when called directly so a full skill
        file still requires frontmatter.
        """
        stripped = text.strip()
        has_frontmatter = stripped.startswith("---")
        body = stripped

        if has_frontmatter:
            parts = stripped.split("---", 2)
            frontmatter = parts[1] if len(parts) >= 3 else ""
            body = parts[2].strip() if len(parts) >= 3 else ""
            has_name = "name:" in frontmatter
            has_description = "description:" in frontmatter
            if not (has_name and has_description):
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
        else:
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message="Skill missing: YAML frontmatter (---)",
            )

        has_headings = bool(re.search(r'^#+\s', body, re.MULTILINE))
        has_steps = any(marker in body.lower() for marker in ['step', '1.', 'procedure', 'how to', 'instructions'])
        has_substantial_content = len(body.strip()) > 100
        has_any_content = bool(body.strip())

        checks = {
            'headings': has_headings,
            'procedural content': has_steps,
            'substantial content': has_substantial_content,
        }
        # Existing tests allow a small but valid skill file with frontmatter,
        # heading, and body text. Richer evolved skills still need at least two
        # structural signals unless they are a valid minimal full SKILL.md.
        passed = (has_headings and has_any_content) or sum(checks.values()) >= 2

        if passed:
            found = [k for k, v in checks.items() if v]
            if has_any_content and not found:
                found = ["body content"]
            return ConstraintResult(
                passed=True,
                constraint_name="skill_structure",
                message=f"Skill has valid structure ({', '.join(found)})",
            )
        else:
            missing = [k for k, v in checks.items() if not v]
            return ConstraintResult(
                passed=False,
                constraint_name="skill_structure",
                message=f"Skill body missing: {', '.join(missing)}",
            )

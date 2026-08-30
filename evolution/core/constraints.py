"""Constraint validators for evolved artifacts.

Every candidate variant must pass ALL constraints before it can be
considered valid. Failed constraints = immediate rejection.
"""

import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig

_STOPWORDS = frozenset({
    "about", "after", "also", "always", "been", "before", "being", "between",
    "both", "cannot", "could", "does", "each", "either", "every", "from",
    "have", "here", "instead", "into", "just", "like", "make", "more", "most",
    "must", "never", "only", "other", "over", "should", "some", "such",
    "than", "that", "them", "then", "there", "these", "they", "this",
    "those", "under", "very", "want", "well", "were", "what", "when",
    "where", "which", "while", "will", "with", "without", "would", "your",
})


def _content_terms(text: str) -> Counter:
    """Frequency of topical terms: lowercase words of 4+ chars minus stopwords."""
    words = re.findall(r"[a-z][a-z0-9_-]{3,}", text.lower())
    return Counter(w for w in words if w not in _STOPWORDS)


def semantic_similarity(baseline: str, evolved: str) -> float:
    """How far *evolved* has drifted in topic from *baseline*, in [0, 1].

    A deterministic, dependency-free proxy: a rewrite of the same artifact
    keeps its domain vocabulary (commands, API names, concepts), while text
    that drifted to another purpose does not. PLAN.md constraint 4.

    Two properties are deliberate, because a count-weighted cosine gets both
    wrong and rejects the changes these phases exist to make.

    **Repetition is not topic.** The vectors are binary, not frequency
    weighted. A description that repeats one sentence six times is about the
    same subject as one that says it once, but under raw counts the repeated
    clause dominates the baseline vector, so deleting it reads as near-total
    drift. A real hermes-agent tool description with a sentence repeated six
    times scored 0.29 that way, against a floor of 0.40.

    **Deleting is not drifting.** A candidate that introduces no vocabulary
    the baseline did not already have cannot have changed subject, however
    much shorter it is. Shortening bloated text is precisely Phase 2's job -
    12 descriptions in a stock checkout are already over budget - so a
    similarity floor must not be the thing that forbids it. Whether a shorter
    description still describes the tool well enough is a real question, and
    the cross-tool selection-accuracy gate is what answers it; this constraint
    only asks whether the subject changed.
    """
    a = _content_terms(baseline)
    b = _content_terms(evolved)
    if not a or not b:
        return 1.0 if not a and not b else 0.0

    terms_a, terms_b = set(a), set(b)
    if terms_b <= terms_a:
        return 1.0

    shared = terms_a & terms_b
    return len(shared) / math.sqrt(len(terms_a) * len(terms_b))


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

        # 2. Baseline-relative checks (growth + semantic preservation)
        if baseline_text:
            results.append(self._check_growth(artifact_text, baseline_text, artifact_type))
            results.append(self._check_semantic_preservation(artifact_text, baseline_text))

        # 3. Non-empty
        results.append(self._check_non_empty(artifact_text))

        # 4. Structural integrity
        if artifact_type == "skill":
            results.append(self._check_skill_structure(artifact_text))

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

    def _check_semantic_preservation(self, text: str, baseline: str) -> ConstraintResult:
        """PLAN.md constraint 4: an evolved artifact must keep its subject.

        A threshold of 0 disables the check. See :func:`semantic_similarity`
        for what the number does and does not measure.
        """
        threshold = self.config.min_semantic_similarity
        if threshold <= 0:
            return ConstraintResult(
                passed=True,
                constraint_name="semantic_preservation",
                message="Semantic preservation disabled (threshold 0)",
            )

        if not _content_terms(baseline):
            # The vocabulary measure sees nothing here: the baseline reduces to
            # stopwords, short tokens or non-Latin text. semantic_similarity
            # would return 1.0 or 0.0 by definition rather than by comparison,
            # and "Topic preserved" over a comparison that never ran is exactly
            # the kind of claim this pipeline refuses elsewhere. Passing is
            # still right - a check that cannot measure must not block - but
            # the message says what actually happened.
            return ConstraintResult(
                passed=True,
                constraint_name="semantic_preservation",
                message="Not measurable: the baseline has no content terms to preserve",
            )

        similarity = semantic_similarity(baseline, text)
        if similarity >= threshold:
            return ConstraintResult(
                passed=True,
                constraint_name="semantic_preservation",
                message=f"Topic preserved: similarity {similarity:.2f} (min {threshold:.2f})",
            )
        return ConstraintResult(
            passed=False,
            constraint_name="semantic_preservation",
            message=f"Topic drift: similarity {similarity:.2f} below minimum {threshold:.2f}",
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

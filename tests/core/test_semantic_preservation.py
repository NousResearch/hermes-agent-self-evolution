"""PLAN.md constraint 4: an evolved artifact must keep its subject.

The interesting cases are not "does it catch a rewrite about something else" -
any similarity measure does that. They are the two ways a naive measure fires
on changes the pipeline exists to make: deleting repeated boilerplate, and
compressing bloated text. Both are Phase 2's whole job.
"""

import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator, semantic_similarity


READ_FILE = (
    "Read a file from disk and return its contents as text, optionally "
    "limiting the number of lines returned."
)
SEND_EMAIL = (
    "Compose and deliver an email message to one or more recipients, with "
    "optional attachments and carbon copies."
)

# The real case, taken from a hermes-agent tool description: one clause
# repeated six times. A count-weighted cosine scores the cleanup at 0.29.
BLOATED = (
    "Run a shell command in a persistent session. "
    + "Prefer the purpose-built file tools over shell equivalents wherever one exists. " * 6
)
DEDUPED = "Run a shell command in a persistent session."


class TestDriftIsCaught:
    def test_an_unrelated_rewrite_scores_low(self):
        assert semantic_similarity(READ_FILE, SEND_EMAIL) < 0.4

    def test_identical_text_scores_one(self):
        assert semantic_similarity(READ_FILE, READ_FILE) == 1.0

    def test_a_genuine_topic_change_is_rejected(self):
        validator = ConstraintValidator(EvolutionConfig())
        results = validator.validate_all(SEND_EMAIL, "tool_description", baseline_text=READ_FILE)
        semantic = [r for r in results if r.constraint_name == "semantic_preservation"]
        assert semantic and not semantic[0].passed
        assert "Topic drift" in semantic[0].message


class TestDeletingIsNotDrifting:
    """The regression that made this constraint incompatible with Phase 2."""

    def test_removing_a_repeated_clause_is_not_drift(self):
        assert semantic_similarity(BLOATED, DEDUPED) == 1.0

    def test_the_validator_accepts_the_dedupe(self):
        validator = ConstraintValidator(EvolutionConfig())
        results = validator.validate_all(DEDUPED, "tool_description", baseline_text=BLOATED)
        semantic = [r for r in results if r.constraint_name == "semantic_preservation"]
        assert semantic and semantic[0].passed

    def test_aggressive_compression_is_not_drift(self):
        """Every surviving term came from the baseline, so the subject held."""
        assert semantic_similarity(READ_FILE, "Read a file from disk.") == 1.0

    def test_repetition_alone_does_not_change_the_score(self):
        """Saying the same thing twice is the same subject as saying it once."""
        assert semantic_similarity(READ_FILE, READ_FILE + " " + READ_FILE) == 1.0

    def test_adding_unrelated_vocabulary_still_counts_against_you(self):
        """The escape hatch is for subsets only, not for any change at all."""
        drifted = READ_FILE + " " + SEND_EMAIL
        assert semantic_similarity(READ_FILE, drifted) < 1.0


class TestConfiguration:
    def test_threshold_zero_disables_the_check(self):
        config = EvolutionConfig(min_semantic_similarity=0.0)
        validator = ConstraintValidator(config)
        results = validator.validate_all(SEND_EMAIL, "tool_description", baseline_text=READ_FILE)
        semantic = [r for r in results if r.constraint_name == "semantic_preservation"]
        assert semantic and semantic[0].passed
        assert "disabled" in semantic[0].message

    def test_no_baseline_means_no_semantic_check(self):
        validator = ConstraintValidator(EvolutionConfig())
        results = validator.validate_all(READ_FILE, "tool_description")
        assert not [r for r in results if r.constraint_name == "semantic_preservation"]

    def test_an_unmeasurable_baseline_passes_but_says_so(self):
        """A baseline of stopwords gives the measure nothing to compare.

        The check must not block - it cannot measure - but reporting "Topic
        preserved" over a comparison that never ran would tell a reviewer a
        constraint held when it only abstained.
        """
        validator = ConstraintValidator(EvolutionConfig())
        results = validator.validate_all(
            READ_FILE, "tool_description", baseline_text="do it to a")
        semantic = [r for r in results if r.constraint_name == "semantic_preservation"]
        assert semantic and semantic[0].passed
        assert "Not measurable" in semantic[0].message
        assert "preserved" not in semantic[0].message

    @pytest.mark.parametrize("empty", ["", "   ", "a an the"])
    def test_empty_or_stopword_only_text_does_not_crash(self, empty):
        assert 0.0 <= semantic_similarity(READ_FILE, empty) <= 1.0
        assert 0.0 <= semantic_similarity(empty, READ_FILE) <= 1.0
        assert semantic_similarity(empty, empty) == 1.0

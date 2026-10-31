"""Tests for the content-retention constraint.

The fixtures are synthetic (see ``tests/core/fixtures/``) so the suite
carries no real-world playbook content.
"""

import pytest

from evolution.core.constraints import ConstraintValidator
from evolution.core.config import EvolutionConfig
from evolution.core.content_retention import (
    analyse_retention,
    extract_blocks,
    extract_commands,
)

from tests.core.fixtures.content_retention_fixtures import (
    BASELINE,
    EVOLVED_EQUIVALENT,
    EVOLVED_GROWTH,
    EVOLVED_GUTTED,
)


@pytest.fixture
def validator():
    return ConstraintValidator(EvolutionConfig())


class TestCommandExtraction:
    def test_extracts_shell_commands_from_code_blocks(self):
        commands = extract_commands(BASELINE)
        assert "docker compose" in commands
        assert "kubectl get" in commands
        assert "pytest" in commands
        assert "git log" in commands

    def test_ignores_prose_outside_code_blocks(self):
        text = "Run docker compose to start. Also use kubectl often."
        assert extract_commands(text) == set()

    def test_ignores_markdown_emphasis_labels(self):
        text = "```\n**[apt summary]**: all good\n```"
        assert extract_commands(text) == set()

    def test_normalises_flags_and_paths(self):
        a = extract_commands("```bash\ngit log --oneline -8\n```")
        b = extract_commands("```bash\ngit log --oneline -20\n```")
        assert a == b

    def test_empty_input(self):
        assert extract_commands("") == set()


class TestBlockExtraction:
    def test_headings_and_paragraphs_are_separate_blocks(self):
        blocks = extract_blocks(BASELINE)
        assert any(b.startswith("#") for b in blocks)
        assert any(not b.startswith("#") for b in blocks)

    def test_blank_line_separates_paragraphs(self):
        blocks = extract_blocks("one\n\ntwo\n\nthree")
        assert blocks == ["one", "two", "three"]


class TestRetentionAnalysis:
    def test_gutting_content_is_detected(self):
        report = analyse_retention(BASELINE, EVOLVED_GUTTED)
        assert report.command_retention < 0.5
        assert report.size_ratio < 0.7
        assert report.commands_lost

    def test_equivalent_rewrite_is_not_a_deletion(self):
        report = analyse_retention(BASELINE, EVOLVED_EQUIVALENT)
        assert report.command_retention == 1.0
        assert report.block_retention >= 0.8
        assert report.size_ratio > 0.9

    def test_pure_addition_is_not_a_deletion(self):
        report = analyse_retention(BASELINE, EVOLVED_GROWTH)
        assert report.command_retention == 1.0
        assert report.block_retention == 1.0
        assert report.size_ratio > 1.0

    def test_lesson_retention_is_reported(self):
        report = analyse_retention(BASELINE, EVOLVED_GUTTED)
        assert report.lessons_total > 0
        assert report.lesson_retention < 1.0


class TestContentRetentionConstraint:
    def test_gutting_content_is_rejected(self, validator):
        result = validator._check_content_retention(
            EVOLVED_GUTTED, BASELINE, "skill"
        )
        assert not result.passed
        assert result.constraint_name == "content_retention"
        assert "retention below threshold" in result.message

    def test_equivalent_rewrite_passes(self, validator):
        result = validator._check_content_retention(
            EVOLVED_EQUIVALENT, BASELINE, "skill"
        )
        assert result.passed

    def test_growth_passes(self, validator):
        result = validator._check_content_retention(
            EVOLVED_GROWTH, BASELINE, "skill"
        )
        assert result.passed

    def test_details_are_reported(self, validator):
        result = validator._check_content_retention(
            EVOLVED_GUTTED, BASELINE, "skill"
        )
        assert result.details
        assert "commands" in result.details

    def test_can_be_disabled(self):
        config = EvolutionConfig()
        config.enable_content_retention = False
        disabled = ConstraintValidator(config)
        results = disabled.validate_all(
            EVOLVED_GUTTED, "skill", baseline_text=BASELINE
        )
        assert not any(
            r.constraint_name == "content_retention" for r in results
        )

    def test_constraint_is_registered_in_validate_all(self, validator):
        results = validator.validate_all(
            EVOLVED_GUTTED, "skill", baseline_text=BASELINE
        )
        names = [r.constraint_name for r in results]
        assert "content_retention" in names

    def test_no_baseline_means_no_retention_check(self, validator):
        results = validator.validate_all(EVOLVED_GUTTED, "skill")
        names = [r.constraint_name for r in results]
        assert "content_retention" not in names

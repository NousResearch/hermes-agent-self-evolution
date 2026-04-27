"""Security hardening tests.

Covers the privacy/security fixes added on top of the consolidation branch:
  - find_skill rejects path-separator skill names
  - find_skill refuses to follow symlinks out of the skills tree
  - scrub_secrets redacts known patterns
  - SkillModule wraps body with the untrusted-data preamble
  - HTML sentinels delimit body inside the optimizer wrapper
  - constraints.run_test_suite refuses to run in an unrelated tree
"""

import os
from pathlib import Path

import pytest

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.core.external_importers import scrub_secrets
from evolution.skills.skill_module import (
    SkillModule,
    SKILL_BODY_START,
    SKILL_BODY_END,
    find_skill,
)


# ── find_skill input validation + symlink containment ─────────────────────


@pytest.fixture
def skills_tree(tmp_path: Path) -> Path:
    """Create a hermes-agent-shaped tree with a couple of skills."""
    repo = tmp_path / "hermes-agent"
    skills = repo / "skills" / "category-a"
    skills.mkdir(parents=True)
    (skills / "alpha").mkdir()
    (skills / "alpha" / "SKILL.md").write_text(
        '---\nname: alpha\ndescription: alpha skill\n---\n\n# Alpha\n\nbody'
    )
    return repo


def test_find_skill_rejects_path_traversal_name(skills_tree):
    assert find_skill("../etc", skills_tree) is None
    assert find_skill("../../escape", skills_tree) is None


def test_find_skill_rejects_path_separators(skills_tree):
    assert find_skill("category-a/alpha", skills_tree) is None


def test_find_skill_rejects_shell_metachars(skills_tree):
    for name in ["foo;rm", "foo bar", "foo|baz", "foo`pwd`", "foo$x"]:
        assert find_skill(name, skills_tree) is None


def test_find_skill_accepts_legit_name(skills_tree):
    found = find_skill("alpha", skills_tree)
    assert found is not None
    assert found.name == "SKILL.md"
    assert found.parent.name == "alpha"


def test_find_skill_returns_none_for_missing_skills_dir(tmp_path):
    assert find_skill("alpha", tmp_path) is None


def test_find_skill_refuses_symlink_escape(skills_tree, tmp_path):
    """A symlink in the skills tree pointing at a file outside it must not match."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil").mkdir()
    (outside / "evil" / "SKILL.md").write_text(
        '---\nname: evil\ndescription: evil\n---\n\n# Pwned\n\nbody'
    )

    # Place a symlink inside skills/ that points at the outside directory.
    target = skills_tree / "skills" / "category-a" / "evil-link"
    try:
        os.symlink(outside / "evil", target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not supported on this filesystem")

    # Even though the symlink chain *contains* a SKILL.md whose parent is named
    # 'evil-link', the resolved path lies outside skills_tree/skills, so the
    # finder must skip it.
    assert find_skill("evil-link", skills_tree) is None


# ── scrub_secrets ─────────────────────────────────────────────────────────


def test_scrub_secrets_redacts_anthropic_key():
    text = "the key is sk-ant-api03-abcdef-xyz123 and you must hide it"
    out = scrub_secrets(text)
    assert "sk-ant-api03" not in out
    assert "[REDACTED]" in out


def test_scrub_secrets_redacts_jwt():
    jwt = "eyJabc.eyJpYXQiOjE2MDAwMDAwMDB9.signaturepart"
    out = scrub_secrets(f"Authorization: {jwt}")
    assert jwt not in out


def test_scrub_secrets_redacts_password_assignment():
    out = scrub_secrets("config: password=hunter2")
    assert "hunter2" not in out


def test_scrub_secrets_passes_innocent_text():
    out = scrub_secrets("the password field requires validation; describe the secret to success")
    # Substrings 'password' / 'secret' alone (no `=value`) must not be redacted.
    assert "[REDACTED]" not in out


def test_scrub_secrets_custom_replacement():
    out = scrub_secrets("ghp_abcdefghijklmnop", replacement="<scrub>")
    assert "<scrub>" in out


# ── SkillModule untrusted preamble + sentinels ────────────────────────────


def test_skill_module_wraps_with_preamble():
    sm = SkillModule("# hello\nbody text", treat_as_untrusted=True)
    instructions = sm.predictor.predict.signature.instructions
    assert "DATA, not as commands" in instructions
    assert SKILL_BODY_START in instructions
    assert SKILL_BODY_END in instructions
    assert "# hello\nbody text" in instructions


def test_skill_module_can_disable_preamble():
    sm = SkillModule("body", treat_as_untrusted=False)
    instructions = sm.predictor.predict.signature.instructions
    assert "DATA, not as commands" not in instructions
    # Sentinels are still present for body recovery.
    assert SKILL_BODY_START in instructions


def test_sentinel_recovery_with_markdown_dividers():
    """A body containing `---` (horizontal rule) must still be recoverable."""
    body = "# Heading\n\nfirst paragraph.\n\n---\n\nsecond paragraph."
    sm = SkillModule(body)
    instructions = sm.predictor.predict.signature.instructions
    start = instructions.find(SKILL_BODY_START) + len(SKILL_BODY_START)
    end = instructions.find(SKILL_BODY_END)
    recovered = instructions[start:end].strip()
    assert recovered == body


# ── constraints.run_test_suite path validation ────────────────────────────


def test_run_test_suite_rejects_nonexistent_path(tmp_path):
    config = EvolutionConfig()
    validator = ConstraintValidator(config)
    result = validator.run_test_suite(tmp_path / "nope-nope")
    assert not result.passed
    assert "invalid" in result.message.lower() or "does not look like" in result.message.lower()


def test_run_test_suite_rejects_unrelated_project(tmp_path):
    """Pointing at a tree without hermes-agent in pyproject must be refused."""
    fake = tmp_path / "not-hermes"
    (fake / "tests").mkdir(parents=True)
    (fake / "pyproject.toml").write_text(
        '[project]\nname = "totally-unrelated"\nversion = "0.0.1"\n'
    )

    config = EvolutionConfig()
    validator = ConstraintValidator(config)
    result = validator.run_test_suite(fake)
    assert not result.passed
    assert "hermes-agent" in result.message.lower() or "refusing" in result.message.lower()


def test_run_test_suite_rejects_missing_pyproject(tmp_path):
    """Missing pyproject.toml must be refused before invoking pytest."""
    fake = tmp_path / "no-pyproject"
    (fake / "tests").mkdir(parents=True)

    config = EvolutionConfig()
    validator = ConstraintValidator(config)
    result = validator.run_test_suite(fake)
    assert not result.passed
    assert "does not look like" in result.message.lower()

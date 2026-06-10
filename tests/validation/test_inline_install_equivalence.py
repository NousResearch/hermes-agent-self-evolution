"""Characterization: HermesPromptSectionInstaller.install_text is byte-identical to
the legacy HermesPromptSource.write inner-loop splice, and the validated bytes match
the deployed region body. Locks the invariant the backend-strategy refactor reroutes
(install_candidate: source.write -> installer.install_text) so it cannot regress.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from evolution.prompts.claude_prompt_source import ClaudeCodePromptSource
from evolution.prompts.hermes_prompt_source import HermesPromptSource
from evolution.validation.artifact_installer import (
    ClaudeAppendPromptInstaller,
    HermesPromptSectionInstaller,
)

_SECTION = "MEMORY_GUIDANCE"


def _fake_repo(tmp_path: Path) -> Path:
    (tmp_path / "agent").mkdir(parents=True, exist_ok=True)
    (tmp_path / "agent" / "prompt_builder.py").write_text(textwrap.dedent('''\
        """Stub."""
        MEMORY_GUIDANCE = (
            "old baseline text"
        )
        OTHER = "untouched"
    '''))
    return tmp_path


def test_hermes_install_text_byte_identical_to_source_write(tmp_path):
    new_text = "evolved\ncandidate 'with' quotes and \\ backslash"

    repo_a = _fake_repo(tmp_path / "a")
    HermesPromptSource(repo_a).write(_SECTION, new_text)
    bytes_a = (repo_a / "agent" / "prompt_builder.py").read_bytes()

    repo_b = _fake_repo(tmp_path / "b")
    HermesPromptSectionInstaller(repo_b, _SECTION).install_text(new_text)
    bytes_b = (repo_b / "agent" / "prompt_builder.py").read_bytes()

    assert bytes_a == bytes_b
    # and the spliced section reads back exactly the candidate
    assert HermesPromptSource(repo_b).read(_SECTION) == new_text


def test_hermes_validated_body_equals_deployed_body(tmp_path):
    repo = _fake_repo(tmp_path)
    text = "the convention body"
    HermesPromptSectionInstaller(repo, _SECTION).install_text(text)
    # deploy (source.write) and validation (installer.install_text) splice the same
    # constant for hermes — the read-back body is identical either way.
    assert HermesPromptSource(repo).read(_SECTION) == text


def test_claude_validated_bytes_equal_deployed_region_body(tmp_path):
    # Validation installs to a throwaway append-prompt file; deploy splices the
    # CLAUDE.md region. The candidate BODY must be identical across both.
    text = "use ./bin/check, never pytest"
    workdir = tmp_path / "wd"
    workdir.mkdir()
    installer = ClaudeAppendPromptInstaller(workdir=workdir, baseline_text="seed")
    installer.install_text(text)
    validated_bytes = installer.target_path.read_text()

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# top\n<!-- evolve:CONV start -->\nseed\n<!-- evolve:CONV end -->\n")
    src = ClaudeCodePromptSource(claude_md)
    src.write("CONV", text)
    deployed_body = src.read("CONV")

    assert validated_bytes.strip() == deployed_body.strip() == text

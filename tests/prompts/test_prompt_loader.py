"""Tests for prompt section discovery."""

from pathlib import Path
from textwrap import dedent

from evolution.prompts.prompt_loader import (
    discover_prompt_sections,
    find_prompt_section,
    identity_traits_present,
    load_sections_from_file,
)


SAMPLE_PROMPT_BUILDER = dedent(
    '''
    """Sample prompt_builder."""

    OTHER_CONST = 42

    DEFAULT_AGENT_IDENTITY = (
        "You are a helpful, direct AI agent. You admit uncertainty."
    )

    MEMORY_GUIDANCE = (
        "Save durable facts to persistent memory; "
        "skip transient task progress."
    )

    NON_TARGET = "Ignore me."

    PLATFORM_HINTS = {
        "whatsapp": "Avoid markdown.",
        "telegram": "Use bold sparingly.",
    }
    '''
).lstrip()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "hermes-agent"
    (repo / "agent").mkdir(parents=True)
    (repo / "agent" / "prompt_builder.py").write_text(SAMPLE_PROMPT_BUILDER, encoding="utf-8")
    return repo


class TestLoadSectionsFromFile:
    def test_extracts_named_string_sections(self, tmp_path):
        path = tmp_path / "prompt_builder.py"
        path.write_text(SAMPLE_PROMPT_BUILDER, encoding="utf-8")
        sections = load_sections_from_file(path)
        names = {s.name for s in sections}
        assert "DEFAULT_AGENT_IDENTITY" in names
        assert "MEMORY_GUIDANCE" in names
        assert "NON_TARGET" not in names

    def test_explodes_platform_hints(self, tmp_path):
        path = tmp_path / "prompt_builder.py"
        path.write_text(SAMPLE_PROMPT_BUILDER, encoding="utf-8")
        sections = load_sections_from_file(path)
        names = {s.name for s in sections}
        assert "PLATFORM_HINTS:whatsapp" in names
        assert "PLATFORM_HINTS:telegram" in names
        wa = next(s for s in sections if s.name == "PLATFORM_HINTS:whatsapp")
        assert wa.platform_key == "whatsapp"
        assert "markdown" in wa.text


class TestDiscoverAndFind:
    def test_discover_and_find(self, tmp_path):
        repo = _make_repo(tmp_path)
        all_sections = discover_prompt_sections(repo)
        assert any(s.name == "MEMORY_GUIDANCE" for s in all_sections)
        target = find_prompt_section("MEMORY_GUIDANCE", repo)
        assert target is not None
        assert "durable facts" in target.text

    def test_find_returns_none_for_unknown(self, tmp_path):
        repo = _make_repo(tmp_path)
        assert find_prompt_section("DOES_NOT_EXIST", repo) is None


class TestIdentityTraitsPresent:
    def test_all_traits_detected(self):
        text = "Helpful and direct. Admits uncertainty when you do not know."
        traits = identity_traits_present(text)
        assert traits["helpful"] and traits["direct"] and traits["admits_uncertainty"]

    def test_missing_traits(self):
        text = "A laconic, evasive agent."
        traits = identity_traits_present(text)
        assert not traits["helpful"]
        assert not traits["admits_uncertainty"]

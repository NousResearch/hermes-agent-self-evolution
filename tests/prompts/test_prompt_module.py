from pathlib import Path

from evolution.prompts.prompt_module import (
    PROMPT_BUILDER_RELATIVE,
    PromptSectionModule,
    list_prompt_sections,
    load_prompt_section,
)


def _write_prompt_builder(root: Path):
    path = root / PROMPT_BUILDER_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_text(
        'DEFAULT_AGENT_IDENTITY = "You are concise."\n'
        'MEMORY_GUIDANCE = ("Remember stable facts. " "Skip stale facts.")\n'
        '_PRIVATE = "ignore me"\n'
        'NOT_STRING = 123\n'
    )
    return path


def test_list_prompt_sections_static_parse(tmp_path):
    _write_prompt_builder(tmp_path)

    sections = list_prompt_sections(tmp_path)

    assert sections == {
        "DEFAULT_AGENT_IDENTITY": "You are concise.",
        "MEMORY_GUIDANCE": "Remember stable facts. Skip stale facts.",
    }


def test_load_prompt_section_returns_metadata(tmp_path):
    path = _write_prompt_builder(tmp_path)

    section = load_prompt_section(tmp_path, "MEMORY_GUIDANCE")

    assert section["name"] == "MEMORY_GUIDANCE"
    assert section["text"] == "Remember stable facts. Skip stale facts."
    assert section["path"] == path


def test_load_prompt_section_error_lists_available(tmp_path):
    _write_prompt_builder(tmp_path)

    try:
        load_prompt_section(tmp_path, "MISSING")
    except ValueError as exc:
        assert "DEFAULT_AGENT_IDENTITY" in str(exc)
        assert "MEMORY_GUIDANCE" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_prompt_section_module_exposes_instruction_text():
    module = PromptSectionModule("Follow the prompt section exactly.")

    assert module.section_text == "Follow the prompt section exactly."

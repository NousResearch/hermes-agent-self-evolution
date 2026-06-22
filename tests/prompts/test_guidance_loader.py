"""Tests for the guidance block loader."""

import textwrap
from pathlib import Path

from evolution.prompts.targets import TARGET_GUIDANCE, LABEL_OPEN, LABEL_CLOSE, BLOCK_START, BLOCK_END


def test_target_guidance_count():
    assert len(TARGET_GUIDANCE) == 6


def test_target_guidance_all_strings():
    assert all(isinstance(t, str) and len(t) > 0 for t in TARGET_GUIDANCE)


def test_label_markers_are_distinct():
    assert LABEL_OPEN != LABEL_CLOSE
    assert len(LABEL_OPEN) == 2
    assert len(LABEL_CLOSE) == 2


def test_block_markers_are_distinct():
    assert BLOCK_START != BLOCK_END
    assert BLOCK_START != LABEL_OPEN
    assert BLOCK_END != LABEL_CLOSE


# ---------------------------------------------------------------------------
# Guidance loader helpers (Task 2)
# ---------------------------------------------------------------------------

def _write(tmp_path, name, content):
    f = Path(tmp_path) / name
    f.write_text(textwrap.dedent(content), encoding="utf-8")
    return f


def test_read_inline_guidance(tmp_path):
    from evolution.prompts.guidance_loader import read_guidance_from_source
    src = _write(tmp_path, "prompt_builder.py", '''
        MEMORY_GUIDANCE = (
            "Keep memory compact. "
            "Do not save temporary state."
        )
    ''')
    text = read_guidance_from_source(src, "MEMORY_GUIDANCE")
    assert text is not None
    assert "Keep memory compact" in text
    assert "Do not save temporary state" in text


def test_read_single_line_guidance(tmp_path):
    from evolution.prompts.guidance_loader import read_guidance_from_source
    src = _write(tmp_path, "prompt_builder.py", '''
        SKILLS_GUIDANCE = "After complex tasks, save the approach as a skill."
    ''')
    text = read_guidance_from_source(src, "SKILLS_GUIDANCE")
    assert text == "After complex tasks, save the approach as a skill."


def test_read_missing_guidance_returns_none(tmp_path):
    from evolution.prompts.guidance_loader import read_guidance_from_source
    src = _write(tmp_path, "prompt_builder.py", '''
        OTHER_CONSTANT = "something"
    ''')
    assert read_guidance_from_source(src, "MEMORY_GUIDANCE") is None


# ---------------------------------------------------------------------------
# Scan repo + write-back (Task 3)
# ---------------------------------------------------------------------------

def _make_repo(tmp_path):
    agent = Path(tmp_path) / "agent"
    agent.mkdir()
    (agent / "prompt_builder.py").write_text(textwrap.dedent('''
        MEMORY_GUIDANCE = (
            "Keep memory compact. "
            "Do not save temporary state."
        )
        TASK_COMPLETION_GUIDANCE = "Finish the job completely."
        SKILLS_GUIDANCE = "Save reusable workflows."
        TOOL_USE_ENFORCEMENT_GUIDANCE = (
            "Use tools instead of describing."
        )
        OPENAI_MODEL_EXECUTION_GUIDANCE = "Be precise with GPT."
        GOOGLE_MODEL_OPERATIONAL_GUIDANCE = "Be concise with Gemini."
        DEFAULT_AGENT_IDENTITY = "You are Hermes Agent."
        SESSION_SEARCH_GUIDANCE = "Search before asking."
    '''), encoding="utf-8")
    return tmp_path


def test_read_target_guidance_finds_all_six(tmp_path):
    from evolution.prompts.guidance_loader import read_target_guidance
    repo = _make_repo(tmp_path)
    guidance = read_target_guidance(repo)
    assert set(guidance.keys()) == {
        "MEMORY_GUIDANCE", "TASK_COMPLETION_GUIDANCE", "SKILLS_GUIDANCE",
        "TOOL_USE_ENFORCEMENT_GUIDANCE", "OPENAI_MODEL_EXECUTION_GUIDANCE",
        "GOOGLE_MODEL_OPERATIONAL_GUIDANCE",
    }
    assert "Keep memory compact" in guidance["MEMORY_GUIDANCE"]


def test_list_all_guidance_includes_frozen(tmp_path):
    from evolution.prompts.guidance_loader import list_all_guidance
    repo = _make_repo(tmp_path)
    all_g = dict(list_all_guidance(repo))
    assert "DEFAULT_AGENT_IDENTITY" in all_g
    assert "MEMORY_GUIDANCE" in all_g


def test_write_guidance_replaces_unique_literal(tmp_path):
    from evolution.prompts.guidance_loader import write_guidance_to_source
    src = _write(tmp_path, "prompt_builder.py", '''
        MEMORY_GUIDANCE = "old guidance text"
    ''')
    result = write_guidance_to_source(src, "old guidance text", "new guidance text")
    assert result.status == "written"
    assert "new guidance text" in src.read_text(encoding="utf-8")


def test_write_guidance_skips_when_not_found(tmp_path):
    from evolution.prompts.guidance_loader import write_guidance_to_source
    src = _write(tmp_path, "prompt_builder.py", '''
        MEMORY_GUIDANCE = "something"
    ''')
    result = write_guidance_to_source(src, "nonexistent", "new")
    assert result.status == "not_found"


def test_write_guidance_skips_when_ambiguous(tmp_path):
    from evolution.prompts.guidance_loader import write_guidance_to_source
    src = _write(tmp_path, "dup.py", '''
        A = "dup text"
        B = "dup text"
    ''')
    result = write_guidance_to_source(src, "dup text", "new text")
    assert result.status == "ambiguous"

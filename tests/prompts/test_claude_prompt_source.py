"""ClaudeCodePromptSource — sentinel-region read/write on a CLAUDE.md."""
import pytest

from evolution.prompts.claude_prompt_source import ClaudeCodePromptSource


def _md(tmp_path, body):
    p = tmp_path / "CLAUDE.md"
    p.write_text(body)
    return p


def test_read_extracts_region(tmp_path):
    p = _md(tmp_path, "# Top\n<!-- evolve:CONV start -->\nuse bin/check\n<!-- evolve:CONV end -->\n# Tail\n")
    assert ClaudeCodePromptSource(p).read("CONV").strip() == "use bin/check"


def test_write_replaces_only_region(tmp_path):
    p = _md(tmp_path, "# Top\n<!-- evolve:CONV start -->\nold\n<!-- evolve:CONV end -->\n# Tail\n")
    ClaudeCodePromptSource(p).write("CONV", "new conventions")
    out = p.read_text()
    assert "new conventions" in out and "old" not in out
    assert "# Top" in out and "# Tail" in out  # user content preserved
    assert ClaudeCodePromptSource(p).read("CONV").strip() == "new conventions"


def test_read_missing_region_raises(tmp_path):
    with pytest.raises(KeyError):
        ClaudeCodePromptSource(_md(tmp_path, "# no markers\n")).read("CONV")


def test_write_appends_when_absent(tmp_path):
    p = _md(tmp_path, "# existing\n")
    ClaudeCodePromptSource(p).write("CONV", "fresh")
    out = p.read_text()
    assert "# existing" in out and "evolve:CONV start" in out and "fresh" in out
    assert ClaudeCodePromptSource(p).read("CONV").strip() == "fresh"


def test_write_creates_file_when_missing(tmp_path):
    p = tmp_path / "CLAUDE.md"  # does not exist
    ClaudeCodePromptSource(p).write("CONV", "brand new")
    assert p.exists() and "brand new" in p.read_text()


def test_duplicate_markers_raise_on_read(tmp_path):
    p = _md(tmp_path, "<!-- evolve:CONV start -->\na\n<!-- evolve:CONV end -->\n"
                      "<!-- evolve:CONV start -->\nb\n<!-- evolve:CONV end -->\n")
    with pytest.raises(ValueError):
        ClaudeCodePromptSource(p).read("CONV")

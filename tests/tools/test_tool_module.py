"""Tests for the tool-selector classifier module (no API calls)."""

from evolution.tools.tool_module import (
    build_instruction,
    parse_descriptions,
    ToolSelectorModule,
)
from evolution.tools.targets import BLOCK_START, BLOCK_END

TARGET_DESCS = {
    "search_files": "Search file contents by pattern.",
    "read_file": "Read a text file with line numbers.",
    "terminal": "Run a shell command.",
    "web_search": "Search the web.",
    "browser_navigate": "Open a URL in a browser.",
    "write_file": "Create or overwrite a file.",
}
ALL_TOOLS = [("vision_analyze", "Analyze an image."), ("memory", "Store a memory.")]


# ── Task 5: build_instruction ─────────────────────────────────────────────────

def test_instruction_contains_block_markers():
    instr = build_instruction(TARGET_DESCS, ALL_TOOLS)
    assert BLOCK_START in instr
    assert BLOCK_END in instr


def test_instruction_labels_every_target():
    instr = build_instruction(TARGET_DESCS, ALL_TOOLS)
    for name, desc in TARGET_DESCS.items():
        assert f"[[{name}]]" in instr
        assert desc in instr


def test_instruction_includes_fixed_context_tools():
    instr = build_instruction(TARGET_DESCS, ALL_TOOLS)
    assert "vision_analyze" in instr
    assert "Analyze an image." in instr


def test_instruction_mentions_none_option():
    instr = build_instruction(TARGET_DESCS, ALL_TOOLS)
    assert "none" in instr.lower()


# ── Task 6: parse_descriptions + ToolSelectorModule ───────────────────────────

def test_parse_descriptions_roundtrips_build():
    instr = build_instruction(TARGET_DESCS, ALL_TOOLS)
    parsed = parse_descriptions(instr, baseline=TARGET_DESCS)
    assert parsed == TARGET_DESCS


def test_parse_descriptions_handles_gepa_reformat():
    # GEPA may rewrite header text and add blank lines but keep the labels.
    instr = (
        "Totally rewritten preamble.\n\n"
        f"{BLOCK_START}\n"
        "[[search_files]] Better search.\n\n"
        "[[read_file]] Better read.\n"
        "[[terminal]] Better terminal.\n"
        "[[web_search]] Better web.\n"
        "[[browser_navigate]] Better nav.\n"
        "[[write_file]] Better write.\n"
        f"{BLOCK_END}\n"
    )
    parsed = parse_descriptions(instr, baseline=TARGET_DESCS)
    assert parsed["search_files"] == "Better search."
    assert parsed["write_file"] == "Better write."


def test_parse_descriptions_falls_back_to_baseline_when_label_missing():
    instr = f"{BLOCK_START}\n[[search_files]] Only this one.\n{BLOCK_END}"
    parsed = parse_descriptions(instr, baseline=TARGET_DESCS)
    assert parsed["search_files"] == "Only this one."
    # Missing labels fall back to baseline, never empty.
    assert parsed["read_file"] == TARGET_DESCS["read_file"]


def test_module_constructs_with_predictor():
    mod = ToolSelectorModule(TARGET_DESCS, ALL_TOOLS)
    # The current (evolved) descriptions are readable via the property.
    assert mod.descriptions["terminal"] == "Run a shell command."

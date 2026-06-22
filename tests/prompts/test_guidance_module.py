"""Tests for the guidance judge module (no API calls)."""

from evolution.prompts.guidance_module import build_instruction
from evolution.prompts.targets import BLOCK_START, BLOCK_END

TARGET_GUIDANCE_TEXT = {
    "MEMORY_GUIDANCE": "Keep memory compact. Do not save temporary state.",
    "TASK_COMPLETION_GUIDANCE": "Finish the job completely.",
    "TOOL_USE_ENFORCEMENT_GUIDANCE": "Use tools instead of describing.",
    "OPENAI_MODEL_EXECUTION_GUIDANCE": "Be precise with GPT.",
    "GOOGLE_MODEL_OPERATIONAL_GUIDANCE": "Be concise with Gemini.",
    "SKILLS_GUIDANCE": "Save reusable workflows.",
}
ALL_GUIDANCE = [
    ("DEFAULT_AGENT_IDENTITY", "You are Hermes Agent."),
    ("SESSION_SEARCH_GUIDANCE", "Search before asking."),
]


def test_instruction_contains_block_markers():
    instr = build_instruction(TARGET_GUIDANCE_TEXT, ALL_GUIDANCE)
    assert BLOCK_START in instr
    assert BLOCK_END in instr


def test_instruction_labels_every_target():
    instr = build_instruction(TARGET_GUIDANCE_TEXT, ALL_GUIDANCE)
    for name, text in TARGET_GUIDANCE_TEXT.items():
        assert f"[[{name}]]" in instr
        assert text in instr


def test_instruction_includes_frozen_context():
    instr = build_instruction(TARGET_GUIDANCE_TEXT, ALL_GUIDANCE)
    assert "DEFAULT_AGENT_IDENTITY" in instr


def test_instruction_mentions_scoring():
    instr = build_instruction(TARGET_GUIDANCE_TEXT, ALL_GUIDANCE)
    assert "score" in instr.lower()


# --- Task 5 tests ---

from evolution.prompts.guidance_module import parse_guidance, GuidanceJudgeModule


TARGET_BASELINE = dict(TARGET_GUIDANCE_TEXT)


def test_parse_guidance_roundtrips_build():
    instr = build_instruction(TARGET_GUIDANCE_TEXT, ALL_GUIDANCE)
    parsed = parse_guidance(instr, baseline=TARGET_BASELINE)
    assert parsed == TARGET_GUIDANCE_TEXT


def test_parse_guidance_handles_gepa_reformat():
    instr = (
        "Totally rewritten preamble.\n\n"
        f"{BLOCK_START}\n"
        "[[MEMORY_GUIDANCE]] Evolved memory guidance.\n"
        "[[TASK_COMPLETION_GUIDANCE]] Evolved completion.\n"
        f"{BLOCK_END}\n"
    )
    parsed = parse_guidance(instr, baseline=TARGET_BASELINE)
    assert parsed["MEMORY_GUIDANCE"] == "Evolved memory guidance."
    assert parsed["TASK_COMPLETION_GUIDANCE"] == "Evolved completion."


def test_parse_guidance_falls_back_to_baseline():
    instr = f"{BLOCK_START}\n[[MEMORY_GUIDANCE]] Only this one.\n{BLOCK_END}"
    parsed = parse_guidance(instr, baseline=TARGET_BASELINE)
    assert parsed["MEMORY_GUIDANCE"] == "Only this one."
    assert parsed["TASK_COMPLETION_GUIDANCE"] == TARGET_BASELINE["TASK_COMPLETION_GUIDANCE"]


def test_module_constructs_with_predictor():
    mod = GuidanceJudgeModule(TARGET_GUIDANCE_TEXT, ALL_GUIDANCE)
    assert mod.guidance["MEMORY_GUIDANCE"] == TARGET_GUIDANCE_TEXT["MEMORY_GUIDANCE"]

"""Shared constants for Phase 3 system-prompt evolution."""

# Mutable guidance blocks whose texts GEPA is allowed to mutate.
# All other guidance constants in prompt_builder.py are frozen.
TARGET_GUIDANCE = [
    "MEMORY_GUIDANCE",
    "TASK_COMPLETION_GUIDANCE",
    "TOOL_USE_ENFORCEMENT_GUIDANCE",
    "OPENAI_MODEL_EXECUTION_GUIDANCE",
    "GOOGLE_MODEL_OPERATIONAL_GUIDANCE",
    "SKILLS_GUIDANCE",
]

# Frozen blocks that must NEVER appear in evolved output.
FROZEN_GUIDANCE = {
    "DEFAULT_AGENT_IDENTITY",
    "HERMES_AGENT_HELP_GUIDANCE",
    "SESSION_SEARCH_GUIDANCE",
    "KANBAN_GUIDANCE",
    "COMPUTER_USE_GUIDANCE",
    "STEER_CHANNEL_NOTE",
}

# Markers used to delimit each mutable guidance block inside the
# classifier instruction. parse_guidance() reads segments back out by these.
LABEL_OPEN = "[["
LABEL_CLOSE = "]]"

# Sentinels bounding the whole mutable block within the instruction.
BLOCK_START = "<<GUIDANCE BLOCKS — these may be optimized>>"
BLOCK_END = "<<END GUIDANCE BLOCKS>>"

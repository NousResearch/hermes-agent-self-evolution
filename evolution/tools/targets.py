"""Shared constants for Phase 2 tool-description evolution."""

# The confusable cluster: tools whose descriptions GEPA is allowed to mutate.
# All ~38 tools remain VISIBLE to the classifier; only these evolve.
TARGET_TOOLS = [
    "search_files",
    "read_file",
    "terminal",
    "web_search",
    "browser_navigate",
    "write_file",
]

# Markers used to delimit each tool's mutable description inside the
# classifier instruction. parse_descriptions() reads segments back out by these.
LABEL_OPEN = "[["
LABEL_CLOSE = "]]"

# Sentinels bounding the whole mutable block within the instruction.
BLOCK_START = "<<TOOL DESCRIPTIONS — these may be optimized>>"
BLOCK_END = "<<END TOOL DESCRIPTIONS>>"

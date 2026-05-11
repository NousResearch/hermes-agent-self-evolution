"""Tool description evolution — manifest data model, DSPy module, judge, and proposer."""

from evolution.tools.tool_judge import (
    ToolJudgeSignature,
    make_tool_fitness_metric,
)
from evolution.tools.tool_module import (
    ToolModule,
    ToolSelectionSignature,
)
from evolution.tools.tool_proposer import (
    BudgetAwareToolProposer,
    extract_and_rebuild,
)
from evolution.tools.tool_source import (
    MCPManifestSource,
    SentinelParseError,
    ToolEntry,
    ToolManifest,
    ToolSource,
    discover_tool_sources,
)

__all__ = [
    "BudgetAwareToolProposer",
    "MCPManifestSource",
    "SentinelParseError",
    "ToolEntry",
    "ToolJudgeSignature",
    "ToolManifest",
    "ToolModule",
    "ToolSelectionSignature",
    "ToolSource",
    "discover_tool_sources",
    "extract_and_rebuild",
    "make_tool_fitness_metric",
]

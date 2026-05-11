"""Tool description evolution — manifest data model, DSPy module, judge, proposer, and orchestrator.

The orchestrator (``evolve``) and its CLI live in ``evolution.tools.evolve_tool``;
import them from there directly rather than from this package, so ``python -m
evolution.tools.evolve_tool`` doesn't re-import the module as both
``evolution.tools.evolve_tool`` and ``__main__``.
"""

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

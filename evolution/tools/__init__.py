"""Phase 2: tool description evolution.

Optimizes the `description` field of a tool schema so the agent picks the
right tool more reliably for a given task. Uses DSPy + GEPA against a
contrastive synthetic dataset (positive / negative examples).
"""

from evolution.tools.dataset import (
    ToolDatasetBuilder,
    is_positive,
    to_dspy_examples_with_polarity,
)
from evolution.tools.fitness import tool_fitness_metric
from evolution.tools.tool_module import (
    ToolModule,
    TOOL_DESC_START,
    TOOL_DESC_END,
    extract_evolved_description,
    load_tool_definition,
    save_tool_definition,
)

__all__ = [
    "ToolDatasetBuilder",
    "ToolModule",
    "TOOL_DESC_START",
    "TOOL_DESC_END",
    "extract_evolved_description",
    "is_positive",
    "load_tool_definition",
    "save_tool_definition",
    "to_dspy_examples_with_polarity",
    "tool_fitness_metric",
]

"""A tool ``description`` string as an optimizable DSPy artifact.

GEPA mutates signature instructions, so the description is embedded between
markers inside a selection prompt. Extraction returns the description *only* —
that string is what gets deployed into the tool schema, not the prompt.
"""

import re

from evolution.core.constraints import TOOL_DESCRIPTION_MARKERS
from evolution.core.instruction_module import ArtifactModule

DESCRIPTION_START, DESCRIPTION_END = TOOL_DESCRIPTION_MARKERS

SELECT_TEMPLATE = (
    "You select the right tool for a task.\n\n"
    "Tool under evaluation: {name}\n"
    "Description: <<DESCRIPTION>>{description}<</DESCRIPTION>>\n\n"
    "Answer with the tool name if the tool fits the task, otherwise answer 'other'."
)

_DESCRIPTION_RE = re.compile(
    re.escape(DESCRIPTION_START) + r"(.*?)" + re.escape(DESCRIPTION_END),
    re.DOTALL,
)


class ToolDocModule(ArtifactModule):
    """Wraps a tool description as the optimizable instruction text."""

    def __init__(self, tool_name: str, description: str):
        if DESCRIPTION_START in description or DESCRIPTION_END in description:
            raise ValueError("description must not contain the extraction markers")
        super().__init__(
            SELECT_TEMPLATE.format(name=tool_name, description=description)
        )


def extract_description(instructions: str) -> str:
    """Return the description between the markers.

    Raises ``ValueError`` when the optimizer dropped or mangled a marker; the
    caller treats that as a rejected variant rather than deploying a prompt.
    """
    match = _DESCRIPTION_RE.search(instructions)
    if match is None:
        raise ValueError(
            "description markers missing from optimized instructions "
            f"({DESCRIPTION_START} / {DESCRIPTION_END})"
        )
    return match.group(1)

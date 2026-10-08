"""Constraint validation for evolved tool-description blocks.

Stricter than the skill checks: the evolved block must round-trip into the
SAME tool names (nothing added, nothing dropped) and every description must
stay within max_tool_desc_size.
"""

from dataclasses import dataclass

from evolution.core.config import EvolutionConfig
from evolution.tools.tool_module import parse_descriptions


@dataclass
class ToolConstraintResult:
    passed: bool
    constraint_name: str
    message: str


def validate_tool_descriptions(
    text: str, baseline_tools: dict[str, str], config: EvolutionConfig
) -> list[ToolConstraintResult]:
    results = []

    parsed = parse_descriptions(text)
    if parsed is None:
        results.append(ToolConstraintResult(
            False, "round_trip", "Evolved block does not parse into 'name: description' lines"))
        return results
    results.append(ToolConstraintResult(True, "round_trip", f"Parsed {len(parsed)} tool descriptions"))

    missing = set(baseline_tools) - set(parsed)
    added = set(parsed) - set(baseline_tools)
    if missing or added:
        results.append(ToolConstraintResult(
            False, "tool_set",
            f"Tool set changed — missing: {sorted(missing) or 'none'}, added: {sorted(added) or 'none'}"))
    else:
        results.append(ToolConstraintResult(True, "tool_set", "Tool names preserved"))

    limit = config.max_tool_desc_size
    over = {n: len(d) for n, d in parsed.items() if len(d) > limit}
    if over:
        results.append(ToolConstraintResult(
            False, "desc_size", f"Descriptions over {limit} chars: {over}"))
    else:
        results.append(ToolConstraintResult(True, "desc_size", f"All descriptions <= {limit} chars"))

    return results

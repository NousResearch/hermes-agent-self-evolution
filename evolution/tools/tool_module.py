"""Wraps a set of hermes-agent tool descriptions as a DSPy module for optimization.

Mirrors evolution/skills/skill_module.py: the descriptions become the
*instructions* of a single router predictor (not input fields), since GEPA/MIPRO
only mutate predictor instructions. Cross-tool evaluation matters here more than
for a single skill — evolving one tool's description can "steal" selections
from another, so all target tools are optimized together in one instructions
block, not independently.
"""

import ast
from pathlib import Path
from typing import Optional

import dspy

TOOL_BLOCK_START = "### TOOL: "

ROUTER_PREAMBLE = (
    "You are a tool router for an AI coding agent. Given a task, pick the single "
    "best tool from the list below. Respond with only the tool name.\n\n"
    "Available tools:\n"
)


def _resolve_string_constant(tree: ast.Module, name: str) -> Optional[str]:
    """Find a module-level `NAME = "..."` (or triple-quoted) string assignment."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


def _extract_schema_dicts(tree: ast.Module) -> list[dict]:
    """Find module-level `X_SCHEMA = {...}` dict literals with name+description keys."""
    schemas = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        name_val = None
        desc_val = None
        for key, val in zip(node.value.keys, node.value.values):
            if not isinstance(key, ast.Constant):
                continue
            if key.value == "name" and isinstance(val, ast.Constant):
                name_val = val.value
            elif key.value == "description":
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    desc_val = val.value
                elif isinstance(val, ast.Name):
                    desc_val = _resolve_string_constant(tree, val.id)
        if name_val and desc_val:
            schemas.append({"name": name_val, "description": desc_val})
    return schemas


def load_tool_descriptions(hermes_agent_path: Path, tool_names: list[str]) -> dict[str, str]:
    """Statically parse tools/*.py for the description of each requested tool name.

    Uses ast (not import) so this never has to satisfy hermes-agent's own runtime
    dependencies — same approach registry.py itself uses for tool discovery.
    """
    tools_dir = hermes_agent_path / "tools"
    found: dict[str, str] = {}
    remaining = set(tool_names)

    for py_file in sorted(tools_dir.glob("*.py")):
        if not remaining:
            break
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except (SyntaxError, OSError):
            continue
        for schema in _extract_schema_dicts(tree):
            if schema["name"] in remaining:
                found[schema["name"]] = schema["description"]
                remaining.discard(schema["name"])

    missing = tool_names_not_found = set(tool_names) - found.keys()
    if missing:
        raise ValueError(f"Could not find description for tool(s): {sorted(missing)}")
    return found


def build_router_instructions(descriptions: dict[str, str]) -> str:
    """Render tool descriptions into one instructions block GEPA can mutate.

    Each tool is delimited by a `### TOOL: <name>` marker so the block can be
    parsed back into per-tool descriptions after optimization, even if GEPA
    rewrote the surrounding prose.
    """
    parts = [ROUTER_PREAMBLE]
    for name, desc in descriptions.items():
        parts.append(f"{TOOL_BLOCK_START}{name}\n{desc}\n")
    return "\n".join(parts)


def parse_router_instructions(instructions: str, tool_names: list[str]) -> dict[str, str]:
    """Inverse of build_router_instructions — split a (possibly mutated) block
    back into {tool_name: description}, keyed on the still-present markers.

    Tools whose marker GEPA dropped are omitted from the result; callers should
    treat that as a constraint failure (structure didn't survive optimization).
    """
    sections: dict[str, str] = {}
    marker_positions = []
    for name in tool_names:
        marker = f"{TOOL_BLOCK_START}{name}"
        idx = instructions.find(marker)
        if idx != -1:
            marker_positions.append((idx, name, marker))
    marker_positions.sort()

    for i, (idx, name, marker) in enumerate(marker_positions):
        start = idx + len(marker)
        end = marker_positions[i + 1][0] if i + 1 < len(marker_positions) else len(instructions)
        sections[name] = instructions[start:end].strip()

    return sections


class ToolRouterModule(dspy.Module):
    """DSPy module that picks a tool name given a task description.

    The combined tool-descriptions block is the optimizable instructions —
    see build_router_instructions/parse_router_instructions for the marker
    scheme that lets per-tool descriptions round-trip through GEPA mutation.
    """

    class RouteTask(dspy.Signature):
        # Field name must be task_input — EvalDataset.to_dspy_examples() always
        # keys the input on "task_input", regardless of skill vs tool usage.
        task_input: str = dspy.InputField(desc="The task the agent needs to accomplish")
        tool_name: str = dspy.OutputField(desc="The exact name of the single best tool to use")

    def __init__(self, descriptions: dict[str, str]):
        super().__init__()
        self.tool_names = list(descriptions.keys())
        instructions = build_router_instructions(descriptions)
        signature = self.RouteTask.with_instructions(instructions)
        self.predictor = dspy.ChainOfThought(signature)

    @property
    def descriptions(self) -> dict[str, str]:
        """Current per-tool descriptions — mutated in place by GEPA/MIPRO."""
        return parse_router_instructions(self.predictor.predict.signature.instructions, self.tool_names)

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(tool_name=result.tool_name.strip())

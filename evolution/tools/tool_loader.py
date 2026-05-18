"""Tool schema discovery and rewrite helpers.

Reads Python tool files from a hermes-agent checkout, parses module-level
SCHEMA dicts (or any dict literal with both `name` and `description` keys),
and provides utilities to find a tool by name and rewrite its top-level
description.
"""

from __future__ import annotations

import ast
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class ToolSchema:
    """A loaded tool schema with provenance back to its source file."""

    name: str
    schema: dict
    source_path: Path
    var_name: str

    @property
    def description(self) -> str:
        return str(self.schema.get("description", "") or "")

    @property
    def parameters(self) -> dict:
        return dict(self.schema.get("parameters") or {})


def _is_schema_dict(node: ast.AST) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    keys = {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return "name" in keys and "description" in keys


def _safe_literal_eval(node: ast.AST) -> Optional[object]:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _iter_module_schema_assignments(tree: ast.Module) -> Iterable[tuple[str, ast.Dict]]:
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if not _is_schema_dict(node.value):
            continue
        yield target.id, node.value


def load_schemas_from_file(path: Path) -> list[ToolSchema]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    schemas: list[ToolSchema] = []
    for var_name, dict_node in _iter_module_schema_assignments(tree):
        value = _safe_literal_eval(dict_node)
        if not isinstance(value, dict):
            continue
        name = value.get("name")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(value.get("description"), str):
            continue
        schemas.append(ToolSchema(name=name, schema=value, source_path=path, var_name=var_name))
    return schemas


def discover_tool_schemas(hermes_agent_path: Path) -> list[ToolSchema]:
    """Walk hermes-agent/tools/*.py and collect every tool schema dict."""
    tools_dir = hermes_agent_path / "tools"
    if not tools_dir.exists():
        return []
    results: list[ToolSchema] = []
    for py_file in sorted(tools_dir.glob("*.py")):
        results.extend(load_schemas_from_file(py_file))
    return results


def find_tool_schema(tool_name: str, hermes_agent_path: Path) -> Optional[ToolSchema]:
    for schema in discover_tool_schemas(hermes_agent_path):
        if schema.name == tool_name:
            return schema
    return None


def reassemble_schema(schema: dict, evolved_description: str) -> dict:
    """Return a deep copy of the schema with the top-level description replaced."""
    rebuilt = deepcopy(schema)
    rebuilt["description"] = evolved_description
    return rebuilt


def summarize_other_tools(tools: list[ToolSchema], exclude_name: str, char_budget: int = 1500) -> str:
    """Render a compact peer-tool catalogue for context, omitting the target tool."""
    parts: list[str] = []
    total = 0
    for tool in tools:
        if tool.name == exclude_name:
            continue
        first_sentence = re.split(r"(?<=[.!?])\s+", tool.description.strip(), maxsplit=1)[0]
        line = f"- {tool.name}: {first_sentence[:200]}"
        if total + len(line) + 1 > char_budget:
            break
        parts.append(line)
        total += len(line) + 1
    return "\n".join(parts)

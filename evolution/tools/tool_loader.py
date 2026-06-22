"""Read, list, and write hermes-agent tool descriptions.

Descriptions live in `tools/*.py` as the "description" value of a schema dict,
either as an inline string literal or a reference to a module-level string
constant (e.g. TERMINAL_TOOL_DESCRIPTION). We parse statically with `ast` so
hermes-agent code is never executed.
"""

import ast
from pathlib import Path
from typing import Optional


def _string_constants(tree: ast.Module) -> dict:
    """Map module-level NAME -> str value for simple `NAME = "literal"` assigns."""
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                consts[target.id] = node.value.value
    return consts


def _iter_schema_dicts(tree: ast.Module):
    """Yield every dict literal that has a top-level "name" key."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and k.value == "name":
                    yield node
                    break


def _dict_get(dict_node: ast.Dict, key: str):
    """Return the value AST node for a string key in a Dict node, or None."""
    for k, v in zip(dict_node.keys, dict_node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    return None


def _resolve_description(value_node, consts: dict) -> Optional[str]:
    """Resolve a description value node to a string (literal or named constant)."""
    if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
        return value_node.value
    if isinstance(value_node, ast.Name) and value_node.id in consts:
        return consts[value_node.id]
    return None


def read_description_from_source(source_path: Path, tool_name: str) -> Optional[str]:
    """Extract the top-level description string for `tool_name` from one source file.

    Returns None if the tool's schema (a dict with name==tool_name) is not present
    or its description cannot be resolved to a plain string.
    """
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    consts = _string_constants(tree)
    for d in _iter_schema_dicts(tree):
        name_node = _dict_get(d, "name")
        if isinstance(name_node, ast.Constant) and name_node.value == tool_name:
            desc_node = _dict_get(d, "description")
            if desc_node is not None:
                return _resolve_description(desc_node, consts)
    return None


# ── Repo-wide scanning ────────────────────────────────────────────────────────

from evolution.tools.targets import TARGET_TOOLS


def _scan_repo_descriptions(hermes_repo: Path) -> dict:
    """Return {tool_name: description} for every schema dict found under tools/."""
    tools_dir = Path(hermes_repo) / "tools"
    found = {}
    for src in sorted(tools_dir.glob("*.py")):
        try:
            tree = ast.parse(src.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        consts = _string_constants(tree)
        for d in _iter_schema_dicts(tree):
            name_node = _dict_get(d, "name")
            desc_node = _dict_get(d, "description")
            if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str) \
                    and desc_node is not None:
                desc = _resolve_description(desc_node, consts)
                if desc is not None and name_node.value not in found:
                    found[name_node.value] = desc
    return found


def read_target_descriptions(hermes_repo: Path) -> dict:
    """Return {tool_name: description} for the TARGET_TOOLS that were found."""
    all_found = _scan_repo_descriptions(hermes_repo)
    return {name: all_found[name] for name in TARGET_TOOLS if name in all_found}


def list_all_tools(hermes_repo: Path) -> list:
    """Return [(tool_name, description), ...] for ALL tools found in the repo."""
    return sorted(_scan_repo_descriptions(hermes_repo).items())


# ── Safe write-back ───────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class WriteResult:
    """Outcome of a single write-back attempt."""
    status: str  # "written" | "not_found" | "ambiguous"
    message: str = ""


def write_description_to_source(
    source_path: Path, baseline_desc: str, evolved_desc: str
) -> WriteResult:
    """Replace the exact baseline description literal with the evolved text.

    Refuses (no write) if the baseline literal is absent or non-unique. We match
    the JSON-style double-quoted form, which is how descriptions are written in
    hermes-agent schema dicts. Returns a WriteResult; never raises on ambiguity.
    """
    path = Path(source_path)
    text = path.read_text(encoding="utf-8")

    # Match the double-quoted literal form so we don't touch comments/prose.
    needle = '"' + baseline_desc + '"'
    count = text.count(needle)
    if count == 0:
        return WriteResult("not_found", f"baseline literal not found in {path.name}")
    if count > 1:
        return WriteResult("ambiguous", f"baseline literal appears {count}x in {path.name}")

    replacement = '"' + evolved_desc + '"'
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    return WriteResult("written", f"updated {path.name}")

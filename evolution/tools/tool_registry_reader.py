"""Static reader for hermes-agent's tool registry.

Extracts ``name -> description`` from ``registry.register(...)`` calls by parsing
``<hermes_agent_path>/tools/*.py`` with :mod:`ast`. Hermes code is never imported
or executed, so the reader cannot trigger tool imports, MCP connections or
credential checks.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_RESOLVE_DEPTH = 8


def read_tool_descriptions(hermes_agent_path: Path) -> dict[str, str]:
    """Return every statically resolvable registered tool's description."""
    tools_dir = Path(hermes_agent_path) / "tools"
    if not tools_dir.is_dir():
        raise FileNotFoundError(f"no tools directory under {hermes_agent_path}")

    descriptions, skipped = _scan_dir(tools_dir)
    logger.info(
        "tool registry: %d description(s) read, %d skipped", len(descriptions), skipped
    )
    return descriptions


def _scan_dir(tools_dir: Path) -> tuple[dict[str, str], int]:
    descriptions: dict[str, str] = {}
    skipped = 0
    for path in sorted(tools_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            logger.warning("tool registry: cannot parse %s: %s", path.name, exc)
            continue
        env = _module_bindings(tree)
        for name, description in _registrations(tree, env):
            if name is None or not description:
                logger.debug("tool registry: skipped registration in %s (name=%r)", path.name, name)
                skipped += 1
                continue
            descriptions[name] = description
    return descriptions, skipped


def _module_bindings(tree: ast.Module) -> dict[str, ast.expr]:
    """Module-level ``NAME = expr`` bindings, used to resolve schema variables."""
    env: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                env[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                env[node.target.id] = node.value
    return env


def _registrations(tree: ast.Module, env: dict[str, ast.expr]):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "register"
            and isinstance(func.value, ast.Name)
            and func.value.id == "registry"
        ):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        name = _string(kwargs.get("name"), env)
        schema = _dict(kwargs.get("schema"), env)
        description = _description(schema, env) if schema is not None else None
        yield name, description


def _description(schema: ast.Dict, env: dict[str, ast.expr]) -> str | None:
    for key, value in zip(schema.keys, schema.values):
        if isinstance(key, ast.Constant) and key.value == "description":
            text = _string(value, env)
            return text.strip() if text and text.strip() else None
    return None


def _string(node: ast.expr | None, env: dict[str, ast.expr], depth: int = 0) -> str | None:
    """Resolve an expression to a string constant, following names and ``+``/f-string joins."""
    if node is None or depth > _MAX_RESOLVE_DEPTH:
        return None
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.Name):
        return _string(env.get(node.id), env, depth + 1)
    if isinstance(node, ast.JoinedStr):
        parts = [
            _string(value, env, depth + 1) if isinstance(value, ast.Constant) else None
            for value in node.values
        ]
        return None if any(part is None for part in parts) else "".join(parts)  # type: ignore[arg-type]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string(node.left, env, depth + 1)
        right = _string(node.right, env, depth + 1)
        return None if left is None or right is None else left + right
    return None


def _dict(node: ast.expr | None, env: dict[str, ast.expr], depth: int = 0) -> ast.Dict | None:
    """Resolve an expression to a dict literal node, following module-level names."""
    if node is None or depth > _MAX_RESOLVE_DEPTH:
        return None
    if isinstance(node, ast.Dict):
        return node
    if isinstance(node, ast.Name):
        return _dict(env.get(node.id), env, depth + 1)
    return None

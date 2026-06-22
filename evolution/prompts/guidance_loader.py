"""Read, list, and write system-prompt guidance blocks from hermes-agent.

Guidance blocks live in ``agent/prompt_builder.py`` as module-level string
constants (e.g. ``MEMORY_GUIDANCE = ("..." )``). We parse statically with
:py:mod:`ast` so hermes-agent code is never executed.  Reuses patterns from
Phase 2's ``tool_loader.py`` but adapted for assignment-from-tuple/string.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evolution.prompts.targets import TARGET_GUIDANCE


# ---------------------------------------------------------------------------
# Low-level AST helpers
# ---------------------------------------------------------------------------

def _string_constants(tree: ast.Module) -> dict:
    """Map module-level NAME -> str value for simple and parenthesized string assigns."""
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = _resolve_constant_string(node.value)
            if value is not None:
                consts[target.id] = value
    return consts


def _resolve_constant_string(node) -> Optional[str]:
    """Resolve an AST node to a string if it's a literal or tuple of literals."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Tuple):
        parts = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                parts.append(elt.value)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(part.value)
            else:
                return None
        return "".join(parts)
    return None


# ---------------------------------------------------------------------------
# Public API  (Task 2)
# ---------------------------------------------------------------------------

def read_guidance_from_source(source_path: Path, constant_name: str) -> Optional[str]:
    """Extract the value of a module-level string constant by name."""
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    consts = _string_constants(tree)
    return consts.get(constant_name, None)


# ---------------------------------------------------------------------------
# Scan repo + write-back  (Task 3)
# ---------------------------------------------------------------------------


def _scan_repo_guidance(hermes_repo: Path) -> dict:
    """Return {constant_name: text} for every string constant in agent/prompt_builder.py."""
    src = Path(hermes_repo) / "agent" / "prompt_builder.py"
    if not src.exists():
        return {}
    try:
        tree = ast.parse(src.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return {}
    return _string_constants(tree)


def read_target_guidance(hermes_repo: Path) -> dict:
    """Return {constant_name: guidance_text} for TARGET_GUIDANCE that were found."""
    all_found = _scan_repo_guidance(hermes_repo)
    return {name: all_found[name] for name in TARGET_GUIDANCE if name in all_found}


def list_all_guidance(hermes_repo: Path) -> list:
    """Return [(constant_name, text), ...] for ALL constants found in prompt_builder.py."""
    return sorted(_scan_repo_guidance(hermes_repo).items())


@dataclass
class GuidanceWriteResult:
    """Outcome of a single write-back attempt."""
    status: str  # "written" | "not_found" | "ambiguous"
    message: str = ""


def write_guidance_to_source(
    source_path: Path, baseline_text: str, evolved_text: str
) -> GuidanceWriteResult:
    """Replace the exact baseline guidance literal with the evolved text.

    Refuses (no write) if the baseline literal is absent or non-unique.
    Returns a GuidanceWriteResult; never raises on ambiguity.
    """
    path = Path(source_path)
    text = path.read_text(encoding="utf-8")
    needle = '"' + baseline_text + '"'
    count = text.count(needle)
    if count == 0:
        return GuidanceWriteResult("not_found", f"baseline literal not found in {path.name}")
    if count > 1:
        return GuidanceWriteResult("ambiguous", f"baseline literal appears {count}x in {path.name}")
    replacement = '"' + evolved_text + '"'
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    return GuidanceWriteResult("written", f"updated {path.name}")

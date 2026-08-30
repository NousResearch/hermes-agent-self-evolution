"""Read and rewrite hermes-agent artifacts without disturbing their structure.

Phases 2-4 evolve *text* that lives inside hermes-agent source files: tool
descriptions inside module-level schema dicts, and system prompt sections
inside module-level string constants. Everything around that text -
parameter names, types, required lists, the surrounding code - is frozen.
This module is the single place that reads and rewrites it.

Two guarantees hold for every write:

1. **Span-exact.** A write replaces only the source span the AST reports for
   one string literal. The rest of the file is passed through byte for byte,
   so nothing is reformatted, reordered, or re-quoted.
2. **Structure-preserving.** :func:`verify_structure_unchanged` re-parses the
   rewritten source and compares the schema skeleton - key names, parameter
   names, types, enums, required lists - against the original. A rewrite that
   moves any of it is rejected before it reaches disk.

Nothing here calls an LLM or the network; it is pure source manipulation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

__all__ = [
    "SourceSpan",
    "ToolDescriptor",
    "PromptSection",
    "StructureViolation",
    "discover_tool_schemas",
    "discover_prompt_sections",
    "replace_span",
    "render_string_literal",
    "apply_tool_description",
    "apply_param_description",
    "apply_prompt_section",
    "verify_structure_unchanged",
    "schema_skeleton",
]

# Sections PLAN.md marks evolvable. Anything outside this set is refused by
# apply_prompt_section even when the constant exists, so a typo cannot rewrite
# an unrelated constant.
EVOLVABLE_PROMPT_SECTIONS = (
    "DEFAULT_AGENT_IDENTITY",
    "MEMORY_GUIDANCE",
    "SESSION_SEARCH_GUIDANCE",
    "SKILLS_GUIDANCE",
)


class StructureViolation(RuntimeError):
    """Raised when a rewrite would change more than description text."""


@dataclass(frozen=True)
class SourceSpan:
    """Byte offsets of a node within its source file.

    Offsets are into the *character* string returned by ``read_text``. They are
    computed from the AST's line/column positions, which are UTF-8 byte columns,
    so :func:`_offset_of` converts through the encoded line.
    """

    start: int
    end: int

    def slice(self, source: str) -> str:
        """Return the exact substring of *source* that this span covers."""
        return source[self.start:self.end]


@dataclass
class ParamDescriptor:
    """One parameter inside a tool schema's ``properties`` map."""

    name: str
    description: str
    span: Optional[SourceSpan]  # None when the parameter has no description key


@dataclass
class ToolDescriptor:
    """A tool schema discovered in a hermes-agent ``tools/*.py`` module."""

    tool_name: str
    constant: str  # e.g. "SEARCH_FILES_SCHEMA"
    path: Path
    description: str
    description_span: Optional[SourceSpan]
    params: dict[str, ParamDescriptor] = field(default_factory=dict)

    @property
    def module(self) -> str:
        """Importable module name for the file this descriptor came from."""
        return self.path.stem


@dataclass
class PromptSection:
    """A module-level prompt string constant in ``agent/prompt_builder.py``."""

    name: str
    path: Path
    text: str
    span: SourceSpan


# ──────────────────────────────────────────────────────────────────────────
# Offset helpers
# ──────────────────────────────────────────────────────────────────────────


def _line_offsets(source: str) -> list[int]:
    """Character offset at which each 1-indexed line starts (index 0 unused)."""
    offsets = [0, 0]
    total = 0
    for line in source.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def _offset_of(source: str, line_starts: list[int], lineno: int, col: int) -> int:
    """Convert an AST (lineno, col_offset) pair to a character offset.

    ``col_offset`` is a UTF-8 *byte* column. Re-encoding the line prefix and
    decoding back to characters keeps non-ASCII source (hermes-agent prompt
    text contains em dashes and emoji) correctly aligned.
    """
    if lineno >= len(line_starts):
        return len(source)
    start = line_starts[lineno]
    line_end = line_starts[lineno + 1] if lineno + 1 < len(line_starts) else len(source)
    line = source[start:line_end]
    prefix = line.encode("utf-8")[:col].decode("utf-8", errors="ignore")
    return start + len(prefix)


def _span_of(source: str, line_starts: list[int], node: ast.AST) -> SourceSpan:
    return SourceSpan(
        _offset_of(source, line_starts, node.lineno, node.col_offset),
        _offset_of(source, line_starts, node.end_lineno, node.end_col_offset),
    )


def replace_span(source: str, span: SourceSpan, replacement: str) -> str:
    """Return *source* with the characters in *span* swapped for *replacement*."""
    return source[:span.start] + replacement + source[span.end:]


def render_string_literal(value: str, indent: int = 4) -> str:
    """Render *value* as Python source for a string literal.

    Multi-line values become an implicitly concatenated, parenthesised block in
    the style hermes-agent already uses for its prompt constants; single-line
    values stay on one line. ``repr`` does the escaping so any quote, backslash,
    or non-printable character round-trips exactly.
    """
    if "\n" not in value:
        return repr(value)

    pad = " " * indent
    # Keep the newline attached to the line it terminates so the reassembled
    # value is identical to the original.
    pieces = value.splitlines(keepends=True)
    body = "\n".join(f"{pad}{piece!r}" for piece in pieces)
    return f"(\n{body}\n{' ' * max(0, indent - 4)})"


# ──────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────


def _dict_entry(node: ast.Dict, key: str) -> Optional[ast.AST]:
    """Return the value node stored under a literal string *key*."""
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    return None


def _const_str(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _iter_module_assignments(tree: ast.Module) -> Iterator[tuple[str, ast.AST]]:
    """Yield ``(name, value_node)`` for every top-level ``NAME = value``."""
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    yield target.id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.value is not None:
                yield stmt.target.id, stmt.value


def _parse_tool_module(path: Path) -> list[ToolDescriptor]:
    """Extract every tool schema constant from one ``tools/*.py`` module."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    # Cheap prefilter: a schema dict always carries both keys as literals.
    if '"description"' not in source and "'description'" not in source:
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    line_starts = _line_offsets(source)
    found: list[ToolDescriptor] = []

    for const_name, value in _iter_module_assignments(tree):
        if not isinstance(value, ast.Dict):
            continue
        tool_name = _const_str(_dict_entry(value, "name"))
        desc_node = _dict_entry(value, "description")
        if not tool_name or desc_node is None:
            continue
        description = _const_str(desc_node)
        if description is None:
            continue  # computed description; not safely evolvable

        params: dict[str, ParamDescriptor] = {}
        parameters = _dict_entry(value, "parameters")
        if isinstance(parameters, ast.Dict):
            properties = _dict_entry(parameters, "properties")
            if isinstance(properties, ast.Dict):
                for pk, pv in zip(properties.keys, properties.values):
                    pname = _const_str(pk)
                    if pname is None or not isinstance(pv, ast.Dict):
                        continue
                    pdesc_node = _dict_entry(pv, "description")
                    pdesc = _const_str(pdesc_node)
                    params[pname] = ParamDescriptor(
                        name=pname,
                        description=pdesc or "",
                        span=(
                            _span_of(source, line_starts, pdesc_node)
                            if pdesc is not None
                            else None
                        ),
                    )

        found.append(
            ToolDescriptor(
                tool_name=tool_name,
                constant=const_name,
                path=path,
                description=description,
                description_span=_span_of(source, line_starts, desc_node),
                params=params,
            )
        )

    return found


def discover_tool_schemas(hermes_repo: Path) -> list[ToolDescriptor]:
    """Find every literal tool schema under ``<hermes_repo>/tools``.

    Schemas whose description is computed at runtime rather than written as a
    literal are skipped - there is no source span to rewrite safely.
    Results are sorted by tool name so runs are reproducible.
    """
    tools_dir = Path(hermes_repo) / "tools"
    if not tools_dir.is_dir():
        return []

    out: list[ToolDescriptor] = []
    for path in sorted(tools_dir.glob("*.py")):
        out.extend(_parse_tool_module(path))
    out.sort(key=lambda t: (t.tool_name, t.constant))
    return out


def discover_prompt_sections(
    hermes_repo: Path,
    names: tuple[str, ...] = EVOLVABLE_PROMPT_SECTIONS,
) -> list[PromptSection]:
    """Find the evolvable system-prompt constants in ``agent/prompt_builder.py``.

    Only string-valued constants are returned. ``PLATFORM_HINTS`` is a dict of
    per-platform strings rather than one string, so it has no single span to
    rewrite and is never returned. Phase 3 reports it as present but out of
    scope through ``sections.StructuredSection``; nothing evolves it today.
    """
    path = Path(hermes_repo) / "agent" / "prompt_builder.py"
    if not path.is_file():
        return []

    # Same contract as tool discovery: a file this function cannot read is a
    # file with no discoverable sections, not a crash in the middle of a scan.
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    line_starts = _line_offsets(source)
    wanted = set(names)
    sections: list[PromptSection] = []

    for const_name, value in _iter_module_assignments(tree):
        if const_name not in wanted:
            continue
        text = _const_str(value)
        if text is None:
            continue
        sections.append(
            PromptSection(
                name=const_name,
                path=path,
                text=text,
                span=_span_of(source, line_starts, value),
            )
        )

    sections.sort(key=lambda s: s.name)
    return sections


# ──────────────────────────────────────────────────────────────────────────
# Structure verification
# ──────────────────────────────────────────────────────────────────────────


def schema_skeleton(source: str) -> dict[str, object]:
    """Summarise every tool schema in *source*, excluding description text.

    The result is what must stay identical across a description rewrite:
    constant names, tool names, parameter names, types, enums, defaults, and
    required lists. Description strings are deliberately omitted.
    """
    tree = ast.parse(source)
    skeleton: dict[str, object] = {}

    for const_name, value in _iter_module_assignments(tree):
        if not isinstance(value, ast.Dict):
            continue
        tool_name = _const_str(_dict_entry(value, "name"))
        if not tool_name:
            continue

        entry: dict[str, object] = {"tool_name": tool_name}
        parameters = _dict_entry(value, "parameters")
        if isinstance(parameters, ast.Dict):
            props: dict[str, object] = {}
            properties = _dict_entry(parameters, "properties")
            if isinstance(properties, ast.Dict):
                for pk, pv in zip(properties.keys, properties.values):
                    pname = _const_str(pk)
                    if pname is None:
                        continue
                    shape: dict[str, object] = {}
                    if isinstance(pv, ast.Dict):
                        for ik, iv in zip(pv.keys, pv.values):
                            ikey = _const_str(ik)
                            if ikey is None or ikey == "description":
                                continue
                            try:
                                shape[ikey] = ast.literal_eval(iv)
                            except (ValueError, TypeError, SyntaxError):
                                shape[ikey] = "<computed>"
                    props[pname] = shape
            entry["properties"] = props

            required = _dict_entry(parameters, "required")
            if required is not None:
                try:
                    entry["required"] = ast.literal_eval(required)
                except (ValueError, TypeError, SyntaxError):
                    entry["required"] = "<computed>"

            ptype = _const_str(_dict_entry(parameters, "type"))
            if ptype:
                entry["type"] = ptype

        skeleton[const_name] = entry

    return skeleton


def verify_structure_unchanged(before: str, after: str) -> None:
    """Raise :class:`StructureViolation` if *after* moved anything but text.

    Checks that the rewritten source still parses, and that its schema skeleton
    matches the original exactly. This is the gate that makes "schema structure
    is FROZEN - only text evolves" enforceable rather than aspirational.
    """
    try:
        after_skeleton = schema_skeleton(after)
    except SyntaxError as exc:
        raise StructureViolation(f"rewritten source does not parse: {exc}") from exc

    before_skeleton = schema_skeleton(before)

    if before_skeleton.keys() != after_skeleton.keys():
        added = sorted(after_skeleton.keys() - before_skeleton.keys())
        removed = sorted(before_skeleton.keys() - after_skeleton.keys())
        raise StructureViolation(
            f"schema constants changed (added={added}, removed={removed})"
        )

    for name, before_entry in before_skeleton.items():
        after_entry = after_skeleton[name]
        if before_entry != after_entry:
            raise StructureViolation(
                f"schema structure changed for {name}: "
                f"{before_entry!r} -> {after_entry!r}"
            )


# ──────────────────────────────────────────────────────────────────────────
# Writes
# ──────────────────────────────────────────────────────────────────────────


def _rewrite_checked(source: str, span: SourceSpan, new_text: str, indent: int) -> str:
    updated = replace_span(source, span, render_string_literal(new_text, indent=indent))
    verify_structure_unchanged(source, updated)
    return updated


def apply_tool_description(source: str, tool: ToolDescriptor, new_description: str) -> str:
    """Return *source* with *tool*'s top-level description replaced.

    The span recorded on *tool* must have come from this same source text.
    Raises :class:`StructureViolation` if the rewrite disturbs any schema shape.
    """
    if tool.description_span is None:
        raise StructureViolation(
            f"{tool.tool_name}: description is computed, not a literal - cannot rewrite"
        )
    return _rewrite_checked(source, tool.description_span, new_description, indent=8)


def apply_param_description(
    source: str,
    tool: ToolDescriptor,
    param: str,
    new_description: str,
) -> str:
    """Return *source* with one parameter description replaced."""
    descriptor = tool.params.get(param)
    if descriptor is None:
        raise StructureViolation(f"{tool.tool_name}: no parameter named {param!r}")
    if descriptor.span is None:
        raise StructureViolation(
            f"{tool.tool_name}.{param}: no literal description to rewrite"
        )
    return _rewrite_checked(source, descriptor.span, new_description, indent=12)


def apply_prompt_section(source: str, section: PromptSection, new_text: str) -> str:
    """Return *source* with a prompt-section constant's value replaced.

    Refuses any constant outside :data:`EVOLVABLE_PROMPT_SECTIONS`, so a bad
    section name cannot rewrite unrelated module state.
    """
    if section.name not in EVOLVABLE_PROMPT_SECTIONS:
        raise StructureViolation(
            f"{section.name} is not an evolvable prompt section "
            f"(allowed: {', '.join(EVOLVABLE_PROMPT_SECTIONS)})"
        )
    updated = replace_span(source, section.span, render_string_literal(new_text, indent=4))
    try:
        ast.parse(updated)
    except SyntaxError as exc:
        raise StructureViolation(f"rewritten prompt source does not parse: {exc}") from exc
    return updated

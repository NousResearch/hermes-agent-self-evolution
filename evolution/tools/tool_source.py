"""Tool manifest discovery and data model.

A ToolSource is the tool-pipeline analog of SkillSource: a Protocol that
adapters implement to discover tool manifests from different backing
stores. MCPManifestSource reads a static JSON file in the shape an MCP
server returns from list_tools(); HermesToolSource walks a directory of
Python source files for ``*_SCHEMA`` declarations.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)

# Conservative subset of MCP-spec-allowed tool names. The 128-char bound
# accommodates namespaced names like Claude Code's
# ``mcp__plugin_<server>__<tool>`` which can run 70-80 chars in practice;
# the character set is restricted to keep sentinel parsing safe (no regex
# metacharacters, no embedded ``-->``).
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _normalize_tool_name(name: str) -> str:
    """Normalize a tool name for collision detection and lookup robustness."""
    return name.lower().replace("-", "_")


class SentinelParseError(ValueError):
    """Raised when sentinel markers around a tool description are missing,
    duplicated, or malformed. Subclass of ValueError so callers can catch
    either specifically or generically.
    """


@dataclass(frozen=True)
class ToolEntry:
    """A single tool's entry in the manifest.

    Treat ``input_schema`` as read-only. ``frozen=True`` prevents rebinding the
    attribute but does not deep-freeze the dict; mutating it in place corrupts
    any other ToolEntry / ToolManifest that shares the reference (which is
    by design — ``replace_description`` preserves the original reference).
    """
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    source_kind: Literal["literal", "name_ref", "joined_str"] | None = None
    # source_location is (file_path, lineno, col_offset, end_lineno, end_col_offset)
    # of the description string node (for name_ref, points at the resolved
    # constant's location). None for non-source-backed manifests like JSON.
    source_location: tuple[Path, int, int, int, int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolEntry":
        if "name" not in data:
            raise ValueError(f"tool entry missing required field 'name': {data}")
        if "description" not in data:
            raise ValueError(
                f"tool entry {data['name']!r} missing required field 'description'"
            )
        name = data["name"]
        if not _TOOL_NAME_RE.match(name):
            raise ValueError(
                f"tool name {name!r} uses characters outside [a-zA-Z0-9_-]; "
                f"rename before evolving (sentinel parsing depends on safe names)"
            )
        return cls(
            name=name,
            description=data["description"],
            input_schema=data.get("inputSchema", {}),
        )


@dataclass(frozen=True)
class ToolManifest:
    """A full tool manifest — the unit of work for tool description evolution.

    Treat ``confusable_neighbors`` as read-only (same contract as
    ``ToolEntry.input_schema`` — the dataclass is structurally frozen but the
    inner dict is not deep-frozen).
    """
    tools: tuple[ToolEntry, ...]
    confusable_neighbors: dict[str, str] = field(default_factory=dict)
    # (tool_name_hint, reason) for tools the adapter saw but couldn't parse.
    # Empty for JSON manifests; populated by source-walking adapters that may
    # legitimately skip schemas they can't statically resolve.
    dropped_tools: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if not self.tools:
            raise ValueError("manifest contains no tools")
        normalized: dict[str, list[str]] = {}
        for t in self.tools:
            normalized.setdefault(_normalize_tool_name(t.name), []).append(t.name)
        collisions = {norm: names for norm, names in normalized.items() if len(names) > 1}
        if collisions:
            raise ValueError(
                f"manifest contains tool names that collide under normalization: {collisions}"
            )

    @classmethod
    def from_json_file(cls, path: Path) -> "ToolManifest":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"manifest file not found: {path}")
        data = json.loads(path.read_text())
        tools = tuple(ToolEntry.from_dict(t) for t in data.get("tools", []))
        metadata = data.get("_evolution_metadata", {})
        confusable = metadata.get("confusable_neighbors", {})
        return cls(tools=tools, confusable_neighbors=confusable)

    def find_tool(self, name: str) -> ToolEntry:
        for tool in self.tools:
            if tool.name == name:
                return tool
        available = sorted(t.name for t in self.tools)
        raise KeyError(
            f"tool {name!r} not found in manifest. Available tools: {available}"
        )

    def confusable_neighbor_for(self, tool_name: str) -> str | None:
        return self.confusable_neighbors.get(tool_name)

    def replace_description(self, tool_name: str, new_description: str) -> "ToolManifest":
        """Return a new ToolManifest with the named tool's description replaced.
        All other tools (and their input_schemas) are preserved verbatim.
        """
        self.find_tool(tool_name)
        new_tools = tuple(
            ToolEntry(
                name=t.name,
                description=new_description if t.name == tool_name else t.description,
                input_schema=t.input_schema,
                source_kind=t.source_kind,
                source_location=t.source_location,
            )
            for t in self.tools
        )
        return ToolManifest(
            tools=new_tools,
            confusable_neighbors=self.confusable_neighbors,
            dropped_tools=self.dropped_tools,
        )


_CLAUDE_CODE_PLUGIN_CACHE_MARKER = (".claude", "plugins", "cache")


def _is_claude_code_plugin_cache_path(path: Path) -> bool:
    parts = path.resolve().parts
    marker = _CLAUDE_CODE_PLUGIN_CACHE_MARKER
    for i in range(len(parts) - len(marker) + 1):
        if parts[i:i + len(marker)] == marker:
            return True
    return False


class ToolSource(Protocol):
    """Protocol for tool-manifest discovery adapters."""

    name: str

    def supports(self, path: Path) -> bool:
        ...

    def find_manifest(self, path_or_name: str | Path) -> ToolManifest | None:
        ...

    def apply_evolved(
        self,
        source_path: Path,
        evolved_manifest: "ToolManifest",
        target_tool: str,
        new_description: str,
    ) -> None:
        ...


class MCPManifestSource:
    """Reads a static JSON file in MCP list_tools() shape."""

    name = "mcp_manifest"

    def __init__(self, root: Path):
        self.root = Path(root)

    def supports(self, path: Path) -> bool:
        """True iff ``path`` is an existing ``.json`` file. Thin check —
        JSON contents are validated by ``find_manifest``.
        """
        candidate = Path(path)
        return candidate.is_file() and candidate.suffix == ".json"

    def find_manifest(self, path: str | Path) -> ToolManifest | None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        if not candidate.is_file():
            return None
        return ToolManifest.from_json_file(candidate)

    def apply_evolved(
        self,
        source_path: Path,
        evolved_manifest: ToolManifest,
        target_tool: str,
        new_description: str,
    ) -> None:
        """Rewrite ``source_path`` in place with the evolved description.

        Preserves any ``_evolution_metadata`` block in the source file and
        all non-target tools' bytes (modulo JSON's canonicalization at
        ``indent=2``). Refuses to write into Claude Code's plugin cache.

        Mirrors ``HermesToolSource.apply_evolved``: writes are atomic via
        ``tempfile + os.replace``.
        """
        source_path = Path(source_path)
        if _is_claude_code_plugin_cache_path(source_path):
            logger.warning(
                "--apply skipped: %s is under a Claude Code plugin cache "
                "(~/.claude/plugins/cache); plugin caches are managed by "
                "Claude Code and writing to them is unsafe.",
                source_path,
            )
            return

        source = json.loads(source_path.read_text())
        for entry in source.get("tools", []):
            if entry.get("name") == target_tool:
                entry["description"] = new_description
                break

        new_text = json.dumps(source, indent=2) + "\n"
        fd, tmp_name = tempfile.mkstemp(dir=source_path.parent, suffix=source_path.suffix)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(new_text)
            # mkstemp creates files with mode 0600; copy the original mode
            # so atomic replace doesn't clobber the source file's perms.
            shutil.copymode(source_path, tmp_path)
            os.replace(tmp_path, source_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise


def discover_tool_sources(explicit_dirs: list[Path] | None = None) -> list[ToolSource]:
    """Build the priority-ordered ToolSource list.

    Returns one ``MCPManifestSource`` and one ``HermesToolSource`` per
    explicit_dirs entry. ``MCPManifestSource`` comes first because its
    ``supports()`` check is cheaper (single ``.suffix`` test) than
    ``HermesToolSource``'s AST scan.
    """
    # Import locally to avoid an import cycle: hermes_source depends on
    # ToolEntry/ToolManifest defined above.
    from evolution.tools.hermes_source import HermesToolSource

    sources: list[ToolSource] = []
    for d in explicit_dirs or []:
        root = Path(d)
        sources.append(MCPManifestSource(root))
        sources.append(HermesToolSource(root))
    return sources

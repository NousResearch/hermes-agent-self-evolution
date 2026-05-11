"""Tool manifest discovery and data model.

A ToolSource is the tool-pipeline analog of SkillSource: a Protocol that
adapters implement to discover tool manifests from different backing
stores. MCPManifestSource reads a static JSON file in the shape an MCP
server returns from list_tools().
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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
            )
            for t in self.tools
        )
        return ToolManifest(tools=new_tools, confusable_neighbors=self.confusable_neighbors)


class ToolSource(Protocol):
    """Protocol for tool-manifest discovery adapters."""

    name: str

    def find_manifest(self, path_or_name: str | Path) -> ToolManifest | None:
        ...


class MCPManifestSource:
    """Reads a static JSON file in MCP list_tools() shape."""

    name = "mcp_manifest"

    def __init__(self, root: Path):
        self.root = Path(root)

    def find_manifest(self, path: str | Path) -> ToolManifest | None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        if not candidate.exists():
            return None
        return ToolManifest.from_json_file(candidate)


def discover_tool_sources(explicit_dirs: list[Path] | None = None) -> list[ToolSource]:
    """Build the priority-ordered ToolSource list.

    Returns one MCPManifestSource per explicit_dirs entry.
    """
    sources: list[ToolSource] = []
    for d in explicit_dirs or []:
        sources.append(MCPManifestSource(Path(d)))
    return sources

"""The tool catalog, and evidence about how well the agent navigates it.

The audit turned up a directly measurable problem: across one profile the
agent spent 971 ``tool_describe`` calls and 623 ``tool_search`` calls — 1,594
turns *looking for tools* rather than using them. That is what a catalog whose
descriptions do not discriminate costs, and unlike answer quality it needs no
LLM to score.

So tool-description evolution gets a real objective. Ground truth comes from
production: for each user message, which tool did the agent actually reach for
next? A description set is better when a model reading only the catalog picks
that same tool more often, and when it needs fewer discovery calls to do it.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from evolution.core.hermes_paths import HermesInstall
from evolution.core.redact import contains_secret

# Tools whose invocation means "I could not find the right tool", not "I chose
# this tool". They are the cost being measured, so they are never the label.
DISCOVERY_TOOLS = frozenset({"tool_search", "tool_describe", "tool_call"})

# A message shorter than this is not a task statement.
_MIN_TASK_CHARS = 15


@dataclass
class ToolSpec:
    """One tool as the model sees it when choosing."""

    name: str
    description: str
    source: str = ""

    def render(self, max_chars: int = 0) -> str:
        desc = self.description.strip()
        if max_chars and len(desc) > max_chars:
            desc = desc[:max_chars].rstrip() + "…"
        return f"- {self.name}: {desc}"


@dataclass
class ToolCatalog:
    """A set of tool descriptions that can be evolved as one artifact."""

    tools: list[ToolSpec] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tools)

    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def get(self, name: str) -> Optional[ToolSpec]:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def render(self, max_chars_each: int = 0) -> str:
        return "\n".join(t.render(max_chars_each) for t in self.tools)

    def total_chars(self) -> int:
        return sum(len(t.description) for t in self.tools)

    def with_descriptions(self, descriptions: dict[str, str]) -> "ToolCatalog":
        """A copy with some descriptions replaced — the evolved candidate."""
        return ToolCatalog(
            tools=[
                ToolSpec(
                    name=t.name,
                    description=descriptions.get(t.name, t.description),
                    source=t.source,
                )
                for t in self.tools
            ]
        )

    def restricted_to(self, names: Iterable[str]) -> "ToolCatalog":
        keep = set(names)
        return ToolCatalog(tools=[t for t in self.tools if t.name in keep])

    # ── serialization ────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(
            {"tools": [{"name": t.name, "description": t.description} for t in self.tools]},
            indent=2,
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "ToolCatalog":
        """Load from a JSON dump.

        Accepts either ``{"tools": [{name, description}]}`` or the raw
        OpenAI-style ``[{"type": "function", "function": {...}}]`` shape that
        ``hermes tools list --json`` produces.
        """
        data = json.loads(Path(path).read_text())
        entries = data.get("tools", data) if isinstance(data, dict) else data

        tools: list[ToolSpec] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
            name = fn.get("name") or entry.get("name")
            desc = fn.get("description") or entry.get("description") or ""
            if name:
                tools.append(ToolSpec(name=str(name), description=str(desc), source=str(path)))
        return cls(tools=tools)


@dataclass
class ToolChoiceExample:
    """A real task and the tool the agent actually used for it."""

    task_input: str
    chosen_tool: str
    discovery_calls: int = 0
    session_id: str = ""

    def to_dict(self) -> dict:
        return {
            "task_input": self.task_input,
            "chosen_tool": self.chosen_tool,
            "discovery_calls": self.discovery_calls,
            "session_id": self.session_id,
        }


def _connect_ro(path: Path) -> Optional[sqlite3.Connection]:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.DatabaseError:
        return None


def mine_tool_choices(
    install: HermesInstall,
    limit: int = 0,
    min_tool_uses: int = 3,
) -> list[ToolChoiceExample]:
    """Extract (user task -> tool actually used) pairs from real sessions.

    Discovery calls between the ask and the first real tool are counted, not
    discarded: they are the cost the evolved descriptions are meant to remove,
    so they belong in the record.

    Tools seen fewer than ``min_tool_uses`` times are dropped — a label with
    two examples teaches the optimizer noise.
    """
    examples: list[ToolChoiceExample] = []

    for prof in install.profiles_with_state():
        conn = _connect_ro(prof.state_db)
        if conn is None:
            continue
        try:
            rows = conn.execute(
                """
                SELECT session_id, role, content, tool_name, timestamp, id
                FROM messages
                ORDER BY session_id, timestamp, id
                """
            ).fetchall()
        except sqlite3.DatabaseError:
            continue
        finally:
            pass

        by_session: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_session.setdefault(row["session_id"], []).append(row)
        conn.close()

        for session_id, msgs in by_session.items():
            examples.extend(_choices_in_session(session_id, msgs))
            if limit and len(examples) >= limit:
                break
        if limit and len(examples) >= limit:
            break

    # Drop rare labels.
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex.chosen_tool] = counts.get(ex.chosen_tool, 0) + 1
    filtered = [ex for ex in examples if counts[ex.chosen_tool] >= min_tool_uses]

    return filtered[:limit] if limit else filtered


def _choices_in_session(session_id: str, msgs: list) -> list[ToolChoiceExample]:
    out: list[ToolChoiceExample] = []
    for i, msg in enumerate(msgs):
        if msg["role"] != "user":
            continue
        text = (msg["content"] or "").strip()
        if len(text) < _MIN_TASK_CHARS or contains_secret(text):
            continue

        discovery = 0
        for nxt in msgs[i + 1:]:
            if nxt["role"] == "user":
                break
            name = nxt["tool_name"]
            if not name:
                continue
            if name in DISCOVERY_TOOLS:
                discovery += 1
                continue
            out.append(
                ToolChoiceExample(
                    task_input=text,
                    chosen_tool=name,
                    discovery_calls=discovery,
                    session_id=session_id,
                )
            )
            break
    return out


def discovery_overhead(install: HermesInstall) -> dict:
    """How much of the agent's tool budget goes to finding tools.

    This is the headline number Phase 2 exists to move, so it is reported
    before and after rather than inferred.
    """
    from evolution.core.state_db import tool_usage_histogram

    histogram = tool_usage_histogram(install)
    total = sum(histogram.values())
    discovery = sum(count for name, count in histogram.items() if name in DISCOVERY_TOOLS)
    return {
        "total_tool_calls": total,
        "discovery_calls": discovery,
        "discovery_share": (discovery / total) if total else 0.0,
        "by_tool": {name: histogram.get(name, 0) for name in sorted(DISCOVERY_TOOLS)},
        "top_tools": dict(list(histogram.items())[:15]),
    }


_WORD = re.compile(r"[a-z0-9_]+")


def extract_catalog_from_repo(hermes_repo: Path, limit: int = 0) -> ToolCatalog:
    """Best-effort catalog extraction from a hermes-agent checkout.

    Hermes builds its tool schemas at runtime, so a static scan is inherently
    partial. Prefer ``hermes tools list --json`` and
    :meth:`ToolCatalog.from_json_file`; this is the fallback for when the CLI
    is not reachable.
    """
    catalog = ToolCatalog()
    seen: set[str] = set()

    pattern = re.compile(
        r'["\']name["\']\s*:\s*["\']([a-z0-9_]+)["\']\s*,\s*'
        r'["\']description["\']\s*:\s*["\'](.{20,600}?)["\']\s*[,}]',
        re.DOTALL,
    )

    agent_dir = Path(hermes_repo) / "agent"
    if not agent_dir.is_dir():
        return catalog

    for path in sorted(agent_dir.rglob("*.py")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for match in pattern.finditer(text):
            name, desc = match.group(1), match.group(2)
            if name in seen:
                continue
            seen.add(name)
            catalog.tools.append(
                ToolSpec(name=name, description=desc.strip(), source=str(path))
            )
            if limit and len(catalog.tools) >= limit:
                return catalog
    return catalog

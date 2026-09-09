"""Read and slice the system prompt Hermes actually runs.

``sessions.system_prompt`` stores the exact prompt each session ran under, and
``system_prompt_hash`` groups them — 119 distinct variants across 347 sessions
on the reference install. So the live prompt does not have to be reconstructed
from source: it can be read, ranked by how much traffic each variant carried,
and sliced into the sections worth optimizing.

Prompt economics differ from skill economics in a way that shapes the budget.
A skill is loaded when it is relevant; the system prompt is paid for on
*every* request of every session. A section that grows by 500 characters costs
that on each of the thousands of calls the install makes, which is why the
default growth allowance here is far tighter than for skills.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evolution.core.hermes_paths import HermesInstall

# Markdown headings are how Hermes' prompt is organized; each becomes a
# separately optimizable unit.
_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)

# The system prompt is paid on every request, so growth is capped much harder
# than for skills.
DEFAULT_PROMPT_GROWTH = 0.05


@dataclass
class PromptSection:
    """One heading-delimited slice of the system prompt."""

    title: str
    level: int
    body: str
    start: int
    end: int

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-")

    @property
    def size(self) -> int:
        return len(self.body)

    def render(self) -> str:
        return f"{'#' * self.level} {self.title}\n{self.body}"


@dataclass
class SystemPrompt:
    """A full system prompt, its provenance, and its sections."""

    text: str
    prompt_hash: str = ""
    sessions: int = 0
    source: str = ""
    sections: list[PromptSection] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.text)

    def section(self, name: str) -> Optional[PromptSection]:
        """Find a section by exact title, slug, or case-insensitive substring."""
        lowered = name.lower()
        for sec in self.sections:
            if sec.title == name or sec.slug == lowered:
                return sec
        for sec in self.sections:
            if lowered in sec.title.lower():
                return sec
        return None

    def replace_section(self, section: PromptSection, new_body: str) -> str:
        """Return the full prompt with one section's body swapped.

        Splicing by recorded offsets rather than re-rendering the whole prompt
        keeps every other byte identical, so a diff shows only what changed.
        The original body's trailing blank lines are reproduced too — losing
        them would put a spurious whitespace change on every section boundary
        and bury the real edit.
        """
        original = self.text[section.start : section.end]
        trailing = original[len(original.rstrip("\n")) :] or "\n"
        return self.text[: section.start] + new_body.rstrip("\n") + trailing + self.text[section.end :]

    def describe(self) -> str:
        return (
            f"{self.size:,} chars, {len(self.sections)} sections"
            + (f", {self.sessions} sessions on {self.prompt_hash[:12]}" if self.sessions else "")
        )


def split_sections(text: str) -> list[PromptSection]:
    """Split a prompt into heading-delimited sections.

    Text before the first heading is returned as a ``(preamble)`` section so
    no part of the prompt is invisible to the caller.
    """
    matches = list(_HEADING.finditer(text))
    sections: list[PromptSection] = []

    if not matches:
        return [PromptSection(title="(whole prompt)", level=1, body=text, start=0, end=len(text))]

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()]
        if preamble.strip():
            sections.append(
                PromptSection(
                    title="(preamble)", level=1, body=preamble,
                    start=0, end=matches[0].start(),
                )
            )

    for i, match in enumerate(matches):
        body_start = match.end() + 1
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(
            PromptSection(
                title=match.group(2).strip(),
                level=len(match.group(1)),
                body=text[body_start:body_end],
                start=body_start,
                end=body_end,
            )
        )
    return sections


def _connect_ro(path: Path) -> Optional[sqlite3.Connection]:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.DatabaseError:
        return None


def load_live_prompts(
    install: HermesInstall,
    profile: Optional[str] = None,
    min_sessions: int = 1,
) -> list[SystemPrompt]:
    """Every distinct system prompt in use, most-used first.

    Ordering by session count matters: optimizing a prompt variant that ran
    twice is wasted effort, and the most-used variant is the one whose cost
    and behaviour dominate.
    """
    counts: Counter = Counter()
    texts: dict[str, str] = {}

    profiles = [install.profile(profile)] if profile else install.profiles_with_state()
    for prof in profiles:
        if not prof.has_state_db():
            continue
        conn = _connect_ro(prof.state_db)
        if conn is None:
            continue
        try:
            for row in conn.execute(
                """
                SELECT system_prompt, system_prompt_hash
                FROM sessions
                WHERE system_prompt IS NOT NULL AND system_prompt <> ''
                """
            ):
                key = row["system_prompt_hash"] or f"len{len(row['system_prompt'])}"
                counts[key] += 1
                texts.setdefault(key, row["system_prompt"])
        except sqlite3.DatabaseError:
            continue
        finally:
            conn.close()

    prompts: list[SystemPrompt] = []
    for key, count in counts.most_common():
        if count < min_sessions:
            continue
        text = texts[key]
        prompts.append(
            SystemPrompt(
                text=text,
                prompt_hash=key,
                sessions=count,
                source="state.db",
                sections=split_sections(text),
            )
        )
    return prompts


def load_prompt_file(path: Path) -> SystemPrompt:
    text = Path(path).read_text()
    return SystemPrompt(text=text, source=str(path), sections=split_sections(text))


def prompt_cost_note(section: PromptSection, delta_chars: int, requests: int) -> str:
    """Plain-language statement of what a size change costs at real volume.

    Roughly four characters per token — precise enough to make the scale of
    the trade visible, which is the point.
    """
    tokens = abs(delta_chars) / 4
    direction = "adds" if delta_chars > 0 else "saves"
    total = tokens * max(1, requests)
    return (
        f"'{section.title}' {direction} ~{tokens:.0f} tokens per request; "
        f"across {requests:,} observed requests that is ~{total:,.0f} tokens."
    )

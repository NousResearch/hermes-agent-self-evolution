"""Read real conversations and outcomes out of Hermes' ``state.db``.

This is where Hermes actually keeps its sessions. The legacy
``~/.hermes/sessions/`` directory that the original importer globbed holds
error request-dumps and a routing mirror whose own README says so:

    "LEGACY MIRROR of the gateway routing index ... This is NOT the session
     list. ALL sessions (CLI, TUI, and gateway) live in ~/.hermes/state.db"

Everything worth mining is in the SQLite store: the ``sessions`` table with
per-session outcome and efficiency columns, the ``messages`` table with roles
and tool names, and FTS5 + trigram indexes already built over message text.

The indexes matter for cost. Skill-relevance filtering used to mean sending
every candidate message to an LLM scorer; with FTS5 the database does the
retrieval and the LLM only sees messages that already matched.

All reads are strictly read-only (``mode=ro``) — evolution never writes to a
live Hermes database.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from evolution.core.hermes_paths import HermesInstall, HermesProfile
from evolution.core.redact import contains_secret

# Roles we treat as the user's ask and the agent's answer.
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"

# Below this length a message is a greeting or an acknowledgement, not a task.
_MIN_TASK_CHARS = 10

# FTS5 treats a pile of punctuation as operators; strip to bare word tokens.
_FTS_TOKEN = re.compile(r"[A-Za-z0-9_]{3,}")

# Words that match everything and therefore rank nothing.
_STOPWORDS = frozenset(
    """
    the and for with that this from your you are was were will would should
    can could have has had not but all any our their its it's use used using
    when what which who whom how why into onto over under about above below
    skill agent hermes claude user assistant markdown yaml name description
    """.split()
)


@dataclass
class SessionOutcome:
    """Per-session efficiency and outcome columns, straight from ``sessions``."""

    session_id: str
    end_reason: Optional[str] = None
    message_count: int = 0
    tool_call_count: int = 0
    api_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: Optional[str] = None
    system_prompt_hash: Optional[str] = None
    started_at: float = 0.0
    profile: str = ""

    @property
    def total_tokens(self) -> int:
        return int(self.input_tokens) + int(self.output_tokens)

    @property
    def succeeded(self) -> bool:
        """Whether the session reached a terminal state that reads as success.

        Hermes records ``end_reason`` values like ``webhook_complete`` and
        ``cron_complete`` on clean finishes. An error or an abandoned session
        does not get a ``*_complete`` reason, so this is a usable — if coarse —
        outcome label without needing an LLM to guess.
        """
        return bool(self.end_reason) and self.end_reason.endswith("_complete")


@dataclass
class MinedMessage:
    """A user ask paired with the agent's answer, plus the surrounding context."""

    source: str
    task_input: str
    assistant_response: str
    session_id: str
    profile: str = ""
    tools_used: list[str] = field(default_factory=list)
    outcome: Optional[SessionOutcome] = None

    def to_dict(self) -> dict:
        """Shape expected by the relevance filter and the rest of the pipeline."""
        return {
            "source": self.source,
            "task_input": self.task_input,
            "assistant_response": self.assistant_response,
            "session_id": self.session_id,
            "profile": self.profile,
            "tools_used": self.tools_used,
        }


def _connect_ro(path: Path) -> sqlite3.Connection:
    """Open a Hermes database read-only. Never mutate a live install."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def build_fts_query(skill_name: str, skill_text: str, max_terms: int = 12) -> str:
    """Turn a skill into an FTS5 MATCH expression.

    Terms come from the skill name and the front of the skill text (its
    frontmatter and opening prose, where the subject matter is stated most
    plainly). They are OR-ed, so a message matching any strong term is a
    candidate and BM25 handles the ranking.
    """
    seen: list[str] = []

    def add(raw: str) -> None:
        for token in _FTS_TOKEN.findall(raw.lower()):
            if token in _STOPWORDS or token in seen:
                continue
            seen.append(token)

    add(skill_name.replace("-", " ").replace("_", " "))
    add(skill_text[:600])

    terms = seen[:max_terms]
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms)


class HermesStateImporter:
    """Mine user/assistant exchanges out of one or more Hermes profiles."""

    SOURCE = "hermes"

    def __init__(self, install: HermesInstall, profiles: Optional[list[str]] = None):
        self.install = install
        self._profile_filter = set(profiles) if profiles else None

    # ── profile selection ────────────────────────────────────────────────

    def _profiles(self) -> list[HermesProfile]:
        available = self.install.profiles_with_state()
        if self._profile_filter is None:
            return available
        return [p for p in available if p.name in self._profile_filter]

    def describe_sources(self) -> list[str]:
        """Human-readable list of the databases that will actually be read."""
        return [f"{p.name} ({p.state_db})" for p in self._profiles()]

    # ── outcomes ─────────────────────────────────────────────────────────

    def session_outcomes(self) -> dict[str, SessionOutcome]:
        """Every session's outcome/efficiency row, keyed by session id."""
        out: dict[str, SessionOutcome] = {}
        for prof in self._profiles():
            try:
                with _connect_ro(prof.state_db) as conn:
                    if not _has_table(conn, "sessions"):
                        continue
                    for row in conn.execute(_SESSIONS_SQL):
                        out[row["id"]] = SessionOutcome(
                            session_id=row["id"],
                            end_reason=row["end_reason"],
                            message_count=row["message_count"] or 0,
                            tool_call_count=row["tool_call_count"] or 0,
                            api_call_count=row["api_call_count"] or 0,
                            input_tokens=row["input_tokens"] or 0,
                            output_tokens=row["output_tokens"] or 0,
                            estimated_cost_usd=row["estimated_cost_usd"] or 0.0,
                            model=row["model"],
                            system_prompt_hash=row["system_prompt_hash"],
                            started_at=row["started_at"] or 0.0,
                            profile=prof.name,
                        )
            except sqlite3.DatabaseError:
                # A corrupt profile database must not take the whole run down;
                # the other profiles still have usable data.
                continue
        return out

    # ── message mining ───────────────────────────────────────────────────

    def extract_messages(
        self,
        skill_name: str = "",
        skill_text: str = "",
        limit: int = 0,
        use_fts: bool = True,
    ) -> list[dict]:
        """Return user/assistant pairs, newest first.

        When ``skill_name``/``skill_text`` are supplied and the profile has the
        FTS index, retrieval is narrowed to messages that textually match the
        skill. Otherwise every user message is returned and the caller filters.
        """
        mined = self.mine(
            skill_name=skill_name,
            skill_text=skill_text,
            limit=limit,
            use_fts=use_fts,
        )
        return [m.to_dict() for m in mined]

    def mine(
        self,
        skill_name: str = "",
        skill_text: str = "",
        limit: int = 0,
        use_fts: bool = True,
    ) -> list[MinedMessage]:
        """Same as :meth:`extract_messages` but keeps the typed outcome data."""
        results: list[MinedMessage] = []
        query = build_fts_query(skill_name, skill_text) if (use_fts and skill_name) else ""

        for prof in self._profiles():
            try:
                results.extend(self._mine_profile(prof, query, limit))
            except sqlite3.DatabaseError:
                continue
            if limit and len(results) >= limit:
                return results[:limit]

        return results[:limit] if limit else results

    def _mine_profile(
        self, prof: HermesProfile, fts_query: str, limit: int
    ) -> list[MinedMessage]:
        with _connect_ro(prof.state_db) as conn:
            if not _has_table(conn, "messages"):
                return []

            outcomes = self._profile_outcomes(conn, prof.name)
            session_ids = self._matching_sessions(conn, fts_query)

            if session_ids is not None and not session_ids:
                return []

            return self._pair_messages(conn, prof, outcomes, session_ids, limit)

    @staticmethod
    def _profile_outcomes(
        conn: sqlite3.Connection, profile_name: str
    ) -> dict[str, SessionOutcome]:
        if not _has_table(conn, "sessions"):
            return {}
        out: dict[str, SessionOutcome] = {}
        for row in conn.execute(_SESSIONS_SQL):
            out[row["id"]] = SessionOutcome(
                session_id=row["id"],
                end_reason=row["end_reason"],
                message_count=row["message_count"] or 0,
                tool_call_count=row["tool_call_count"] or 0,
                api_call_count=row["api_call_count"] or 0,
                input_tokens=row["input_tokens"] or 0,
                output_tokens=row["output_tokens"] or 0,
                estimated_cost_usd=row["estimated_cost_usd"] or 0.0,
                model=row["model"],
                system_prompt_hash=row["system_prompt_hash"],
                started_at=row["started_at"] or 0.0,
                profile=profile_name,
            )
        return out

    @staticmethod
    def _matching_sessions(
        conn: sqlite3.Connection, fts_query: str
    ) -> Optional[set[str]]:
        """Sessions containing text that matches the skill, or None for "all".

        Returning None (rather than every session id) lets the caller skip the
        filter entirely when no query was requested or no index is present.
        """
        if not fts_query or not _has_table(conn, "messages_fts"):
            return None

        # messages_fts is a contentless-style FTS5 table whose rowid tracks
        # messages.id. Join through it, but verify the join produced anything
        # before trusting it — an index built with a different rowid mapping
        # would silently return an empty set and hide all the real data.
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT m.session_id AS sid
                FROM messages_fts f
                JOIN messages m ON m.id = f.rowid
                WHERE messages_fts MATCH ?
                """,
                (fts_query,),
            ).fetchall()
        except sqlite3.DatabaseError:
            return None

        sessions = {r["sid"] for r in rows if r["sid"]}
        if sessions:
            return sessions

        # Index present but unusable for our join — fall back to scanning.
        return None

    @staticmethod
    def _pair_messages(
        conn: sqlite3.Connection,
        prof: HermesProfile,
        outcomes: dict[str, SessionOutcome],
        session_ids: Optional[set[str]],
        limit: int,
    ) -> list[MinedMessage]:
        """Walk each session in order, pairing user asks with agent answers."""
        rows = conn.execute(
            """
            SELECT session_id, role, content, tool_name, timestamp, id
            FROM messages
            WHERE content IS NOT NULL AND content <> ''
            ORDER BY session_id, timestamp, id
            """
        ).fetchall()

        by_session: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            sid = row["session_id"]
            if session_ids is not None and sid not in session_ids:
                continue
            by_session.setdefault(sid, []).append(row)

        # Newest session first, matching the file importers' ordering.
        ordered = sorted(
            by_session.items(),
            key=lambda kv: outcomes.get(kv[0], SessionOutcome(kv[0])).started_at,
            reverse=True,
        )

        mined: list[MinedMessage] = []
        for sid, msgs in ordered:
            for pair in _pair_within_session(msgs):
                user_text, assistant_text, tools = pair
                if contains_secret(user_text) or contains_secret(assistant_text):
                    continue
                mined.append(
                    MinedMessage(
                        source=HermesStateImporter.SOURCE,
                        task_input=user_text,
                        assistant_response=assistant_text,
                        session_id=sid,
                        profile=prof.name,
                        tools_used=tools,
                        outcome=outcomes.get(sid),
                    )
                )
                if limit and len(mined) >= limit:
                    return mined
        return mined


def _pair_within_session(
    msgs: Iterable[sqlite3.Row],
) -> Iterable[tuple[str, str, list[str]]]:
    """Yield (user_text, assistant_text, tool_names) for each turn in a session.

    Tool messages between the ask and the answer are not discarded — their
    names are collected, because which tools a task actually reached for is
    signal about what the skill made the agent do.
    """
    msgs = list(msgs)
    for i, msg in enumerate(msgs):
        if msg["role"] != _USER_ROLE:
            continue
        user_text = (msg["content"] or "").strip()
        if len(user_text) < _MIN_TASK_CHARS:
            continue

        assistant_text = ""
        tools: list[str] = []
        for nxt in msgs[i + 1:]:
            role = nxt["role"]
            if role == _USER_ROLE:
                break
            if nxt["tool_name"]:
                tools.append(nxt["tool_name"])
            if role == _ASSISTANT_ROLE and (nxt["content"] or "").strip():
                assistant_text = (nxt["content"] or "").strip()
                break

        yield user_text, assistant_text, tools


def tool_usage_histogram(install: HermesInstall) -> dict[str, int]:
    """How often each tool was called across every profile.

    Used to target tool-description evolution at the tools the agent actually
    struggles to pick — ``tool_search`` and ``tool_describe`` volume is a
    direct measure of discovery failure.
    """
    counts: dict[str, int] = {}
    for prof in install.profiles_with_state():
        try:
            with _connect_ro(prof.state_db) as conn:
                if not _has_table(conn, "messages"):
                    continue
                for row in conn.execute(
                    """
                    SELECT tool_name, COUNT(*) AS n
                    FROM messages
                    WHERE tool_name IS NOT NULL AND tool_name <> ''
                    GROUP BY tool_name
                    """
                ):
                    counts[row["tool_name"]] = counts.get(row["tool_name"], 0) + row["n"]
        except sqlite3.DatabaseError:
            continue
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def prompt_variant_outcomes(install: HermesInstall) -> dict[str, dict]:
    """Aggregate session outcomes per ``system_prompt_hash``.

    Hermes already stamps every session with a hash of the system prompt it
    ran under. That makes the store a ready-made A/B ledger: group by hash,
    compare success rate and efficiency, and you can tell whether a deployed
    prompt variant did better than the one it replaced — without building any
    new tracking.
    """
    buckets: dict[str, dict] = {}
    for prof in install.profiles_with_state():
        try:
            with _connect_ro(prof.state_db) as conn:
                if not _has_table(conn, "sessions"):
                    continue
                for row in conn.execute(_SESSIONS_SQL):
                    key = row["system_prompt_hash"]
                    if not key:
                        continue
                    b = buckets.setdefault(
                        key,
                        {
                            "sessions": 0,
                            "succeeded": 0,
                            "tool_calls": 0,
                            "tokens": 0,
                            "profiles": set(),
                        },
                    )
                    b["sessions"] += 1
                    reason = row["end_reason"] or ""
                    if reason.endswith("_complete"):
                        b["succeeded"] += 1
                    b["tool_calls"] += row["tool_call_count"] or 0
                    b["tokens"] += (row["input_tokens"] or 0) + (row["output_tokens"] or 0)
                    b["profiles"].add(prof.name)
        except sqlite3.DatabaseError:
            continue

    for b in buckets.values():
        n = max(1, b["sessions"])
        b["success_rate"] = b["succeeded"] / n
        b["avg_tool_calls"] = b["tool_calls"] / n
        b["avg_tokens"] = b["tokens"] / n
        b["profiles"] = sorted(b["profiles"])
    return buckets


# Selected once so every reader agrees on the column set, and so a schema that
# predates a column fails loudly here rather than in three different places.
_SESSIONS_SQL = """
    SELECT id, end_reason, message_count, tool_call_count, api_call_count,
           input_tokens, output_tokens, estimated_cost_usd, model,
           system_prompt_hash, started_at
    FROM sessions
"""


def load_jobs_to_skills(install: HermesInstall) -> dict[str, list[str]]:
    """Map each cron job id to the skills it runs.

    ``jobs.json`` binds scheduled jobs to named skills, which is what makes
    per-skill production success rates computable at all.
    """
    path = install.cron_jobs_json
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    mapping: dict[str, list[str]] = {}
    for job in data.get("jobs", []):
        job_id = job.get("id")
        if not job_id:
            continue
        skills = [s for s in (job.get("skills") or []) if s]
        single = job.get("skill")
        if single and single not in skills:
            skills.append(single)
        if skills:
            mapping[job_id] = skills
    return mapping

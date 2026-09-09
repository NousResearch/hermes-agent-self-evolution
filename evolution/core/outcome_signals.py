"""Ground-truth fitness signals recorded by the live Hermes install.

An LLM judge guesses whether an answer was good. These two stores *know*.

``verification_evidence.db``
    One row per verification command Hermes actually ran — tests, builds,
    linters — with its exit code. That is a real pass/fail, not an opinion.

``cron/executions.db``
    One row per scheduled-job execution with a completed/failed status.
    ``cron/jobs.json`` binds each job to the skills it runs, so these rows
    aggregate into a per-skill success rate measured in production.

Both are read-only and both degrade quietly: a missing or corrupt store yields
an empty signal rather than an exception, because the absence of production
telemetry must never be the thing that stops an optimization run.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evolution.core.hermes_paths import HermesInstall
from evolution.core.state_db import load_jobs_to_skills

# Cron statuses that count as a clean finish. Anything else (failed, running,
# cancelled) is not a success — a job still running is not yet evidence.
_CRON_OK = frozenset({"completed", "ok", "success"})

# Verification statuses that count as a pass when no exit code was recorded.
_VERIFY_OK = frozenset({"pass", "passed", "ok", "success"})


@dataclass
class OutcomeStats:
    """A count of attempts and how many of them worked."""

    total: int = 0
    succeeded: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> Optional[float]:
        """Fraction that succeeded, or None when there is no evidence at all.

        None and 0.0 mean very different things — "never observed" must not be
        scored as "always failed" — so callers have to handle the distinction.
        """
        if self.total == 0:
            return None
        return self.succeeded / self.total

    @property
    def failed(self) -> int:
        return self.total - self.succeeded

    def merge(self, other: "OutcomeStats") -> "OutcomeStats":
        return OutcomeStats(
            total=self.total + other.total,
            succeeded=self.succeeded + other.succeeded,
            failures=(self.failures + other.failures)[:20],
        )

    def summary(self) -> str:
        if self.total == 0:
            return "no evidence"
        return f"{self.succeeded}/{self.total} ok ({self.success_rate:.0%})"


def _connect_ro(path: Path) -> Optional[sqlite3.Connection]:
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.DatabaseError:
        return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


# ── verification evidence ────────────────────────────────────────────────


@dataclass
class VerificationEvent:
    """One real command Hermes ran, and whether it passed."""

    session_id: str
    command: str
    kind: str
    status: str
    exit_code: Optional[int]
    output_summary: str
    created_at: float
    profile: str

    @property
    def passed(self) -> bool:
        """Exit code wins when present; otherwise fall back to the status word."""
        if self.exit_code is not None:
            return int(self.exit_code) == 0
        return (self.status or "").lower() in _VERIFY_OK


class VerificationSignal:
    """Verified command outcomes across every profile."""

    def __init__(self, install: HermesInstall):
        self.install = install
        self._events: Optional[list[VerificationEvent]] = None

    def events(self) -> list[VerificationEvent]:
        if self._events is not None:
            return self._events

        found: list[VerificationEvent] = []
        for prof in self.install.profiles():
            conn = _connect_ro(prof.verification_db)
            if conn is None:
                continue
            try:
                if not _table_exists(conn, "verification_events"):
                    continue
                for row in conn.execute(
                    """
                    SELECT session_id, command, kind, status, exit_code,
                           output_summary, created_at
                    FROM verification_events
                    """
                ):
                    found.append(
                        VerificationEvent(
                            session_id=row["session_id"] or "",
                            command=row["command"] or "",
                            kind=row["kind"] or "",
                            status=row["status"] or "",
                            exit_code=row["exit_code"],
                            output_summary=row["output_summary"] or "",
                            created_at=row["created_at"] or 0.0,
                            profile=prof.name,
                        )
                    )
            except sqlite3.DatabaseError:
                continue
            finally:
                conn.close()

        self._events = found
        return found

    def by_session(self) -> dict[str, OutcomeStats]:
        """Verification pass rate per session."""
        out: dict[str, OutcomeStats] = {}
        for ev in self.events():
            stats = out.setdefault(ev.session_id, OutcomeStats())
            stats.total += 1
            if ev.passed:
                stats.succeeded += 1
            elif len(stats.failures) < 20:
                stats.failures.append(f"{ev.kind}: {ev.command[:120]}")
        return out

    def overall(self) -> OutcomeStats:
        stats = OutcomeStats()
        for ev in self.events():
            stats.total += 1
            if ev.passed:
                stats.succeeded += 1
            elif len(stats.failures) < 20:
                stats.failures.append(f"{ev.kind}: {ev.command[:120]}")
        return stats

    def score_for_sessions(self, session_ids: set[str]) -> Optional[float]:
        """Pass rate restricted to a set of sessions, or None with no evidence."""
        stats = OutcomeStats()
        per_session = self.by_session()
        for sid in session_ids:
            if sid in per_session:
                stats = stats.merge(per_session[sid])
        return stats.success_rate


# ── cron execution outcomes ──────────────────────────────────────────────


class CronOutcomeSignal:
    """Per-skill production success rates from scheduled-job executions."""

    def __init__(self, install: HermesInstall):
        self.install = install
        self._per_job: Optional[dict[str, OutcomeStats]] = None

    def per_job(self) -> dict[str, OutcomeStats]:
        if self._per_job is not None:
            return self._per_job

        stats: dict[str, OutcomeStats] = {}
        conn = _connect_ro(self.install.cron_executions_db)
        if conn is not None:
            try:
                if _table_exists(conn, "executions"):
                    for row in conn.execute(
                        "SELECT job_id, status, error FROM executions"
                    ):
                        job_id = row["job_id"]
                        if not job_id:
                            continue
                        s = stats.setdefault(job_id, OutcomeStats())
                        status = (row["status"] or "").lower()
                        # A job still running is not evidence either way.
                        if status == "running":
                            continue
                        s.total += 1
                        if status in _CRON_OK:
                            s.succeeded += 1
                        elif len(s.failures) < 20:
                            s.failures.append((row["error"] or status)[:200])
            except sqlite3.DatabaseError:
                pass
            finally:
                conn.close()

        self._per_job = stats
        return stats

    def per_skill(self) -> dict[str, OutcomeStats]:
        """Roll job outcomes up to the skills those jobs invoke."""
        job_skills = load_jobs_to_skills(self.install)
        per_job = self.per_job()

        out: dict[str, OutcomeStats] = {}
        for job_id, skills in job_skills.items():
            stats = per_job.get(job_id)
            if stats is None:
                continue
            for skill in skills:
                out[skill] = out.get(skill, OutcomeStats()).merge(stats)
        return out

    def for_skill(self, skill_name: str) -> OutcomeStats:
        return self.per_skill().get(skill_name, OutcomeStats())

    def overall(self) -> OutcomeStats:
        total = OutcomeStats()
        for stats in self.per_job().values():
            total = total.merge(stats)
        return total


# ── combined view ────────────────────────────────────────────────────────


@dataclass
class SkillProductionHealth:
    """What the live install knows about how a skill performs in the real world."""

    skill_name: str
    cron: OutcomeStats
    verification: OutcomeStats
    sessions_seen: int = 0
    avg_tool_calls: float = 0.0
    avg_tokens: float = 0.0

    @property
    def has_evidence(self) -> bool:
        return self.cron.total > 0 or self.verification.total > 0

    def baseline_score(self) -> Optional[float]:
        """A single production success rate, or None when nothing is recorded.

        Cron and verification are weighted by how many observations each
        contributes, so a skill with 300 cron runs and 2 verifications is not
        swung by the two.
        """
        parts = [s for s in (self.cron, self.verification) if s.total > 0]
        if not parts:
            return None
        weighted = sum(s.succeeded for s in parts)
        total = sum(s.total for s in parts)
        return weighted / total if total else None

    def describe(self) -> str:
        bits = []
        if self.cron.total:
            bits.append(f"cron {self.cron.summary()}")
        if self.verification.total:
            bits.append(f"verified {self.verification.summary()}")
        if self.sessions_seen:
            bits.append(
                f"{self.sessions_seen} sessions, {self.avg_tool_calls:.1f} tool calls avg"
            )
        return " · ".join(bits) if bits else "no production evidence"


def skill_production_health(
    install: HermesInstall,
    skill_name: str,
    session_ids: Optional[set[str]] = None,
) -> SkillProductionHealth:
    """Gather every real-world outcome signal available for one skill."""
    cron = CronOutcomeSignal(install).for_skill(skill_name)

    verification = OutcomeStats()
    if session_ids:
        per_session = VerificationSignal(install).by_session()
        for sid in session_ids:
            if sid in per_session:
                verification = verification.merge(per_session[sid])

    return SkillProductionHealth(
        skill_name=skill_name,
        cron=cron,
        verification=verification,
    )

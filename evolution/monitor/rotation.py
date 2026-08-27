"""Phase 5 — the continuous improvement loop, versioned with the code.

The deployed rotation driver was a shell script living outside the repository,
so it was neither tested nor reviewed alongside the thing it drove. It also
walked the skill library in alphabetical order at two skills per week: against
the largest profile's 117 skills that is 59 weeks — over a year — for a single
pass, during which the skills themselves keep changing.

Round-robin is the wrong policy when coverage is that slow. This scheduler
spends the budget where the evidence says it is needed:

* skills whose scheduled jobs are actually failing, worst first
* then skills never evolved before
* then the least recently evolved

Skills with a healthy production record and a recent successful pass sink to
the bottom, which is where they belong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from evolution.core.hermes_paths import HermesInstall
from evolution.core.outcome_signals import CronOutcomeSignal, OutcomeStats

# A skill evolved more recently than this is not re-queued unless production
# evidence says it is failing.
DEFAULT_COOLDOWN_DAYS = 30

# Failure rate above which a skill jumps the queue regardless of cooldown.
FAILING_THRESHOLD = 0.10


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SkillRecord:
    """What the scheduler remembers about one skill."""

    name: str
    profile: str = ""
    last_evolved: Optional[str] = None
    last_verdict: str = ""
    attempts: int = 0
    ships: int = 0
    consecutive_holds: int = 0

    def days_since(self) -> Optional[float]:
        if not self.last_evolved:
            return None
        try:
            then = datetime.fromisoformat(self.last_evolved)
        except ValueError:
            return None
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return (_now() - then).total_seconds() / 86400


@dataclass
class Candidate:
    """A skill queued for evolution, with the reason it was chosen."""

    name: str
    profile: str
    priority: float
    reason: str
    stats: Optional[OutcomeStats] = None

    def render(self) -> str:
        return f"{self.profile}/{self.name} — {self.reason}"


class RotationState:
    """Scheduler memory, persisted as JSON."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._records: dict[str, SkillRecord] = {}
        self._loaded = False

    def _key(self, profile: str, name: str) -> str:
        return f"{profile}/{name}"

    def load(self) -> "RotationState":
        if self._loaded:
            return self
        self._loaded = True
        if not self.path.is_file():
            return self
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return self
        for raw in data.get("skills", []):
            try:
                record = SkillRecord(**raw)
            except TypeError:
                continue
            self._records[self._key(record.profile, record.name)] = record
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "updated_at": _now().isoformat(timespec="seconds"),
                    "skills": [asdict(r) for r in self._records.values()],
                },
                indent=2,
            )
        )

    def get(self, profile: str, name: str) -> SkillRecord:
        self.load()
        key = self._key(profile, name)
        if key not in self._records:
            self._records[key] = SkillRecord(name=name, profile=profile)
        return self._records[key]

    def record_result(self, profile: str, name: str, verdict: str) -> None:
        record = self.get(profile, name)
        record.last_evolved = _now().isoformat(timespec="seconds")
        record.last_verdict = verdict
        record.attempts += 1
        if verdict == "SHIP":
            record.ships += 1
            record.consecutive_holds = 0
        else:
            record.consecutive_holds += 1

    def all_records(self) -> list[SkillRecord]:
        self.load()
        return list(self._records.values())


def discover_skills(install: HermesInstall, profile: Optional[str] = None) -> list[tuple[str, str]]:
    """Every (profile, skill_name) pair available to evolve."""
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    profiles = [install.profile(profile)] if profile else install.profiles()
    for prof in profiles:
        if not prof.skills_dir.is_dir():
            continue
        for skill_md in sorted(prof.skills_dir.rglob("SKILL.md")):
            pair = (prof.name, skill_md.parent.name)
            if pair not in seen:
                seen.add(pair)
                found.append(pair)
    return found


def prioritize(
    skills: Iterable[tuple[str, str]],
    state: RotationState,
    cron: Optional[CronOutcomeSignal] = None,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
) -> list[Candidate]:
    """Order skills by how much evidence there is that they need work.

    Priority is a single number so the ordering is total and explainable:
    failure rate dominates, then never-tried, then staleness. Every candidate
    carries the reason it scored what it did, because a scheduler nobody can
    interrogate gets overridden by hand and stops being used.
    """
    per_skill = cron.per_skill() if cron else {}
    candidates: list[Candidate] = []

    for profile, name in skills:
        record = state.get(profile, name)
        stats = per_skill.get(name)
        days = record.days_since()

        priority = 0.0
        reasons: list[str] = []

        if stats and stats.total > 0:
            failure_rate = 1.0 - (stats.success_rate or 0.0)
            if failure_rate >= FAILING_THRESHOLD:
                # Dominant term: a skill failing in production outranks
                # everything that is merely stale.
                priority += 1000 * failure_rate
                reasons.append(
                    f"failing in production ({stats.failed}/{stats.total} runs)"
                )
            else:
                reasons.append(f"healthy ({stats.summary()})")

        if days is None:
            priority += 100
            reasons.append("never evolved")
        else:
            if days < cooldown_days and priority < 1000:
                # Recently attempted and not failing — wait.
                priority -= 500
                reasons.append(f"evolved {days:.0f}d ago (cooldown {cooldown_days}d)")
            else:
                priority += min(100.0, days)
                reasons.append(f"last evolved {days:.0f}d ago")

        # A skill that has been held repeatedly is unlikely to ship on the next
        # identical attempt; deprioritize rather than burning budget on it.
        if record.consecutive_holds >= 3:
            priority -= 50 * record.consecutive_holds
            reasons.append(f"{record.consecutive_holds} consecutive holds")

        candidates.append(
            Candidate(
                name=name,
                profile=profile,
                priority=priority,
                reason="; ".join(reasons) or "no signal",
                stats=stats,
            )
        )

    candidates.sort(key=lambda c: (-c.priority, c.profile, c.name))
    return candidates


@dataclass
class RotationPlan:
    """What this run intends to do, and what it deliberately skipped."""

    selected: list[Candidate] = field(default_factory=list)
    deferred: list[Candidate] = field(default_factory=list)
    total_skills: int = 0

    def render(self) -> str:
        lines = [f"Queued {len(self.selected)} of {self.total_skills} skills:"]
        for c in self.selected:
            lines.append(f"  → {c.render()}")
        if self.deferred:
            # Never let a bounded run read as full coverage.
            lines.append(
                f"  ({len(self.deferred)} deferred this run — highest deferred: "
                f"{self.deferred[0].render() if self.deferred else 'n/a'})"
            )
        return "\n".join(lines)


def build_plan(
    install: HermesInstall,
    state: RotationState,
    skills_per_run: int,
    profile: Optional[str] = None,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
) -> RotationPlan:
    """Choose what to evolve this run."""
    skills = discover_skills(install, profile)
    cron = CronOutcomeSignal(install)
    ranked = prioritize(skills, state, cron=cron, cooldown_days=cooldown_days)

    selected = ranked[: max(0, skills_per_run)]
    deferred = ranked[max(0, skills_per_run) :]
    return RotationPlan(selected=selected, deferred=deferred, total_skills=len(skills))


def coverage_estimate(total_skills: int, skills_per_run: int, runs_per_week: float = 1.0) -> str:
    """How long a full pass takes at the configured cadence.

    Stated explicitly because the deployed default — 2 per week against 117
    skills — implied a 14-month cycle that nobody had worked out.
    """
    if skills_per_run <= 0 or runs_per_week <= 0:
        return "never (nothing scheduled)"
    weeks = total_skills / (skills_per_run * runs_per_week)
    if weeks < 4:
        return f"~{weeks:.1f} weeks per full pass"
    return f"~{weeks / 4.35:.1f} months per full pass ({weeks:.0f} weeks)"

"""Canary rollout for evolved skills, using signals Hermes already records.

Hermes stamps every session with ``system_prompt_hash`` and keeps per-session
outcome columns beside it — 119 distinct prompt versions across 347 sessions
on the reference install. That is a complete A/B ledger that nothing was
reading. Changing a skill changes the prompt, which changes the hash, so the
store already partitions sessions by "which version of the instructions was
this run under".

What this module adds is the bookkeeping around it: record what was deployed
and when, then compare outcomes recorded after that moment against the
recorded baseline, and roll back automatically when the variant is worse.

Honesty about attribution: sessions are not labelled with the skill they used.
Cron sessions can be attributed exactly, because ``jobs.json`` binds jobs to
skills. Interactive sessions cannot, so a prompt-hash comparison over them is
directional evidence rather than proof, and the verdict says which basis it
used.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evolution.core.hermes_paths import HermesInstall
from evolution.core.outcome_signals import CronOutcomeSignal, OutcomeStats
from evolution.core.state_db import prompt_variant_outcomes

# Below this many post-deploy observations, no verdict is issued. Rolling back
# on two data points is as wrong as never rolling back.
DEFAULT_MIN_OBSERVATIONS = 20

# How much worse the variant must be before it is pulled, as an absolute drop
# in success rate. Small regressions inside this band wait for more data.
DEFAULT_REGRESSION_TOLERANCE = 0.05

PROMOTE = "PROMOTE"
ROLLBACK = "ROLLBACK"
WAIT = "WAIT"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CanaryRecord:
    """One deployed variant under observation."""

    skill_name: str
    target_path: str
    deployed_at: str
    baseline_sha: str
    variant_sha: str
    backup_path: str = ""
    baseline_success_rate: Optional[float] = None
    baseline_observations: int = 0
    prompt_hashes_before: list[str] = field(default_factory=list)
    status: str = "active"
    notes: list[str] = field(default_factory=list)

    @property
    def deployed_ts(self) -> float:
        try:
            return datetime.fromisoformat(self.deployed_at).timestamp()
        except ValueError:
            return 0.0


@dataclass
class CanaryVerdict:
    """Decision about a deployed variant, with the evidence behind it."""

    decision: str
    reason: str
    basis: str = ""
    before: Optional[OutcomeStats] = None
    after: Optional[OutcomeStats] = None

    def render(self) -> str:
        parts = [f"{self.decision}: {self.reason}"]
        if self.basis:
            parts.append(f"(basis: {self.basis})")
        return " ".join(parts)


class CanaryLedger:
    """A JSON file tracking which variants are currently under observation."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _load_raw(self) -> list[dict]:
        if not self.path.is_file():
            return []
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return []
        return data.get("deployments", []) if isinstance(data, dict) else []

    def _save_raw(self, deployments: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"updated_at": _now(), "deployments": deployments}, indent=2)
        )

    def all(self) -> list[CanaryRecord]:
        records = []
        for raw in self._load_raw():
            try:
                records.append(CanaryRecord(**raw))
            except TypeError:
                continue
        return records

    def active(self) -> list[CanaryRecord]:
        return [r for r in self.all() if r.status == "active"]

    def add(self, record: CanaryRecord) -> None:
        deployments = self._load_raw()
        deployments.append(asdict(record))
        self._save_raw(deployments)

    def update_status(self, skill_name: str, variant_sha: str, status: str, note: str = "") -> bool:
        deployments = self._load_raw()
        changed = False
        for raw in deployments:
            if raw.get("skill_name") == skill_name and raw.get("variant_sha") == variant_sha:
                raw["status"] = status
                if note:
                    raw.setdefault("notes", []).append(f"{_now()} {note}")
                changed = True
        if changed:
            self._save_raw(deployments)
        return changed


def deploy_canary(
    install: HermesInstall,
    ledger: CanaryLedger,
    skill_name: str,
    target_path: Path,
    variant_text: str,
    backup_dir: Path,
) -> CanaryRecord:
    """Install a variant over a live skill, keeping a backup for rollback.

    The baseline's current production success rate is captured *before* the
    write, because once the file changes there is no way to measure what it
    was.
    """
    target_path = Path(target_path)
    baseline_text = target_path.read_text() if target_path.is_file() else ""

    cron_baseline = CronOutcomeSignal(install).for_skill(skill_name)

    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{skill_name}.baseline.md"
    if baseline_text:
        backup_path.write_text(baseline_text)

    record = CanaryRecord(
        skill_name=skill_name,
        target_path=str(target_path),
        deployed_at=_now(),
        baseline_sha=_sha(baseline_text),
        variant_sha=_sha(variant_text),
        backup_path=str(backup_path),
        baseline_success_rate=cron_baseline.success_rate,
        baseline_observations=cron_baseline.total,
        prompt_hashes_before=sorted(prompt_variant_outcomes(install).keys()),
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(variant_text)
    ledger.add(record)
    return record


def evaluate_canary(
    install: HermesInstall,
    record: CanaryRecord,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    regression_tolerance: float = DEFAULT_REGRESSION_TOLERANCE,
) -> CanaryVerdict:
    """Compare post-deployment outcomes against the recorded baseline."""
    cron = CronOutcomeSignal(install)
    after = cron.for_skill(record.skill_name)

    # Cron rows accumulate, so the post-deploy slice is what is new since the
    # baseline snapshot was taken.
    new_total = after.total - record.baseline_observations
    new_succeeded = after.succeeded - int(
        (record.baseline_success_rate or 0.0) * record.baseline_observations
    )

    if new_total < min_observations:
        return CanaryVerdict(
            decision=WAIT,
            reason=(
                f"only {max(0, new_total)} post-deploy observations, "
                f"need {min_observations}"
            ),
            basis="cron executions bound to this skill",
        )

    after_rate = max(0.0, min(1.0, new_succeeded / new_total))
    before_rate = record.baseline_success_rate

    before_stats = OutcomeStats(
        total=record.baseline_observations,
        succeeded=int((before_rate or 0.0) * record.baseline_observations),
    )
    after_stats = OutcomeStats(total=new_total, succeeded=max(0, new_succeeded))

    if before_rate is None:
        return CanaryVerdict(
            decision=PROMOTE if after_rate >= 0.9 else WAIT,
            reason=(
                f"no pre-deploy baseline; variant is at {after_rate:.0%} over "
                f"{new_total} runs"
            ),
            basis="cron executions bound to this skill (no baseline)",
            after=after_stats,
        )

    delta = after_rate - before_rate

    if delta < -regression_tolerance:
        return CanaryVerdict(
            decision=ROLLBACK,
            reason=(
                f"success rate fell {before_rate:.0%} -> {after_rate:.0%} "
                f"({delta:+.0%}) over {new_total} runs"
            ),
            basis="cron executions bound to this skill",
            before=before_stats,
            after=after_stats,
        )

    return CanaryVerdict(
        decision=PROMOTE,
        reason=(
            f"success rate {before_rate:.0%} -> {after_rate:.0%} "
            f"({delta:+.0%}) over {new_total} runs, within tolerance"
        ),
        basis="cron executions bound to this skill",
        before=before_stats,
        after=after_stats,
    )


def rollback_canary(record: CanaryRecord, ledger: Optional[CanaryLedger] = None) -> bool:
    """Restore the backed-up baseline over the deployed variant."""
    backup = Path(record.backup_path)
    target = Path(record.target_path)
    if not backup.is_file():
        if ledger:
            ledger.update_status(
                record.skill_name, record.variant_sha, "rollback_failed",
                "backup missing",
            )
        return False
    try:
        shutil.copyfile(backup, target)
    except OSError:
        if ledger:
            ledger.update_status(
                record.skill_name, record.variant_sha, "rollback_failed",
                "copy failed",
            )
        return False
    if ledger:
        ledger.update_status(
            record.skill_name, record.variant_sha, "rolled_back", "baseline restored"
        )
    return True


def prompt_variant_comparison(install: HermesInstall, record: CanaryRecord) -> dict:
    """Prompt-hash groups that appeared only after this deployment.

    Directional evidence for interactive sessions, which carry no skill label.
    Reported separately from the cron verdict so the two are never conflated.
    """
    outcomes = prompt_variant_outcomes(install)
    before = set(record.prompt_hashes_before)
    return {h: stats for h, stats in outcomes.items() if h not in before}

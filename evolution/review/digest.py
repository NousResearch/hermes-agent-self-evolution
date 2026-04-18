"""Digest — summarize recent evolution activity as markdown.

Designed for nightly cron delivery: "what happened last night?"

Produces a compact markdown report covering proposals touched within a
time window (default: last 24h by mtime of decision.json), with a
summary header + per-proposal section including the score delta,
decision, mode, and a link to the proposal dir for human review.

Usage:
    python -m evolution.review.digest                 # last 24h, stdout
    python -m evolution.review.digest --hours 48
    python -m evolution.review.digest --since-file path/to/marker
    python -m evolution.review.digest --output digest.md
    python -m evolution.review.digest --format json

The digest is intentionally self-contained (no network, no LLM calls)
so the nightly cron is deterministic and fast.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from evolution.core.proposals import (
    PROPOSAL_STATUS_APPROVED,
    PROPOSAL_STATUS_PENDING,
    PROPOSAL_STATUS_REJECTED,
)
from evolution.review.proposal_reviewer import (
    ProposalEntry,
    discover_proposals,
)


DEFAULT_PROPOSALS_DIR = Path("./proposals")
DEFAULT_LOG_DIR = Path("./logs")


# ──────────────────────────── filtering ────────────────────────────
def _proposal_mtime(entry: ProposalEntry) -> float:
    """Use decision.json mtime as the proposal's canonical timestamp."""
    dec = entry.proposal_dir / "decision.json"
    try:
        return dec.stat().st_mtime
    except OSError:
        return 0.0


def filter_recent(
    entries: List[ProposalEntry], since_epoch: float
) -> List[ProposalEntry]:
    """Keep proposals whose decision.json was written at/after since_epoch."""
    return [e for e in entries if _proposal_mtime(e) >= since_epoch]


# ──────────────────────────── log scan ────────────────────────────
def find_recent_logs(log_dir: Path, since_epoch: float) -> List[Path]:
    """Return evolution run logs written within the window, newest first."""
    if not log_dir.exists():
        return []
    logs: List[Path] = []
    for p in log_dir.glob("evolve-*.log"):
        try:
            if p.stat().st_mtime >= since_epoch:
                logs.append(p)
        except OSError:
            continue
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return logs


def log_tail(path: Path, lines: int = 8) -> str:
    """Return the last N lines of a log file (best-effort)."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    tail = text.splitlines()[-lines:]
    return "\n".join(tail)


# ──────────────────────────── rendering ────────────────────────────
_STATUS_ICON = {
    PROPOSAL_STATUS_PENDING: "⏳",
    PROPOSAL_STATUS_APPROVED: "✅",
    PROPOSAL_STATUS_REJECTED: "❌",
}


def _fmt_delta(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"


def render_markdown(
    entries: List[ProposalEntry],
    window_hours: float,
    generated_at: datetime,
    logs: Optional[List[Path]] = None,
    include_log_tail: bool = True,
) -> str:
    """Render a markdown digest."""
    logs = logs or []

    # ── header / summary ───────────────────────────────────────────
    n_total = len(entries)
    n_pending = sum(1 for e in entries if e.status == PROPOSAL_STATUS_PENDING)
    n_approved = sum(1 for e in entries if e.status == PROPOSAL_STATUS_APPROVED)
    n_rejected = sum(1 for e in entries if e.status == PROPOSAL_STATUS_REJECTED)
    auto_merged = sum(
        1 for e in entries if e.status == PROPOSAL_STATUS_APPROVED and e.auto_merge
    )

    lines: List[str] = []
    lines.append(
        f"# Hermes Self-Evolution Digest — "
        f"{generated_at.strftime('%Y-%m-%d %H:%M')}"
    )
    lines.append("")
    lines.append(f"_Window: last **{window_hours:.1f}h**_")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **{n_total}** proposal(s) in window")
    lines.append(f"  - ⏳ pending: **{n_pending}**")
    lines.append(f"  - ✅ approved: **{n_approved}** (auto-merged: **{auto_merged}**)")
    lines.append(f"  - ❌ rejected: **{n_rejected}**")
    lines.append(f"- **{len(logs)}** evolution run log(s) in window")

    if n_total == 0 and not logs:
        lines.append("")
        lines.append("_No evolution activity in window._")
        return "\n".join(lines) + "\n"

    # ── per-proposal table ────────────────────────────────────────
    if entries:
        lines.append("")
        lines.append("## Proposals")
        lines.append("")
        lines.append("| Status | Skill | Timestamp | Mode | Δ score | Baseline → Evolved |")
        lines.append("|--------|-------|-----------|------|---------|-------------------|")
        for e in entries:
            icon = _STATUS_ICON.get(e.status, "·")
            lines.append(
                f"| {icon} {e.status} "
                f"| `{e.skill_name}` "
                f"| `{e.timestamp}` "
                f"| {e.mode} "
                f"| {_fmt_delta(e.improvement)} "
                f"| {e.baseline_score:.3f} → {e.evolved_score:.3f} |"
            )

    # ── per-proposal detail ───────────────────────────────────────
    if entries:
        lines.append("")
        lines.append("## Details")
        for e in entries:
            lines.append("")
            icon = _STATUS_ICON.get(e.status, "·")
            lines.append(f"### {icon} `{e.skill_name}` — `{e.timestamp}`")
            lines.append("")
            lines.append(f"- Status: **{e.status}**")
            lines.append(f"- Mode: `{e.mode}`")
            lines.append(
                f"- Score: {e.baseline_score:.3f} → {e.evolved_score:.3f} "
                f"({_fmt_delta(e.improvement)})"
            )
            lines.append(f"- Auto-merge gate: {'✅ approved' if e.auto_merge else '❌ rejected'}")
            reason = e.rejection_reason
            if reason:
                lines.append(f"- Rejection reason: _{reason}_")
            lines.append(f"- Proposal dir: `{e.proposal_dir}`")
            lines.append("")
            lines.append("  Review with:")
            lines.append("  ```")
            lines.append(
                f"  python -m evolution.review.proposal_reviewer show "
                f"{e.skill_name} {e.timestamp}"
            )
            lines.append(
                f"  python -m evolution.review.proposal_reviewer diff "
                f"{e.skill_name} {e.timestamp}"
            )
            if e.status == PROPOSAL_STATUS_PENDING:
                lines.append(
                    f"  python -m evolution.review.proposal_reviewer approve "
                    f"{e.skill_name} {e.timestamp}"
                )
                lines.append(
                    f"  python -m evolution.review.proposal_reviewer reject "
                    f"{e.skill_name} {e.timestamp} --reason '...'"
                )
            lines.append("  ```")

    # ── log tails ─────────────────────────────────────────────────
    if logs and include_log_tail:
        lines.append("")
        lines.append("## Run logs (recent)")
        for log in logs[:5]:  # cap at 5 to keep digest compact
            lines.append("")
            lines.append(f"### `{log.name}`")
            tail = log_tail(log, lines=6)
            if tail:
                lines.append("")
                lines.append("```")
                lines.append(tail)
                lines.append("```")

    return "\n".join(lines) + "\n"


def render_json(
    entries: List[ProposalEntry],
    window_hours: float,
    generated_at: datetime,
    logs: Optional[List[Path]] = None,
) -> str:
    """Render digest as JSON for programmatic consumption."""
    logs = logs or []
    payload = {
        "generated_at": generated_at.isoformat(),
        "window_hours": window_hours,
        "summary": {
            "total": len(entries),
            "pending": sum(1 for e in entries if e.status == PROPOSAL_STATUS_PENDING),
            "approved": sum(1 for e in entries if e.status == PROPOSAL_STATUS_APPROVED),
            "rejected": sum(1 for e in entries if e.status == PROPOSAL_STATUS_REJECTED),
            "auto_merged": sum(
                1
                for e in entries
                if e.status == PROPOSAL_STATUS_APPROVED and e.auto_merge
            ),
            "run_logs": len(logs),
        },
        "proposals": [
            {
                "skill_name": e.skill_name,
                "timestamp": e.timestamp,
                "status": e.status,
                "mode": e.mode,
                "improvement": e.improvement,
                "baseline_score": e.baseline_score,
                "evolved_score": e.evolved_score,
                "auto_merge": e.auto_merge,
                "rejection_reason": e.rejection_reason,
                "proposal_dir": str(e.proposal_dir),
            }
            for e in entries
        ],
        "logs": [str(p) for p in logs],
    }
    return json.dumps(payload, indent=2) + "\n"


# ──────────────────────────── CLI ────────────────────────────
def build_digest(
    proposals_dir: Path,
    log_dir: Path,
    window_hours: float,
    fmt: str = "markdown",
    now: Optional[datetime] = None,
) -> str:
    """Build a digest string. Extracted for testability."""
    now = now or datetime.now()
    since_epoch = (now - timedelta(hours=window_hours)).timestamp()

    all_entries = discover_proposals(proposals_dir)
    recent = filter_recent(all_entries, since_epoch)
    logs = find_recent_logs(log_dir, since_epoch)

    if fmt == "json":
        return render_json(recent, window_hours, now, logs=logs)
    return render_markdown(recent, window_hours, now, logs=logs)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evolution.review.digest",
        description="Summarize recent evolution activity as markdown.",
    )
    parser.add_argument(
        "--proposals-dir",
        type=Path,
        default=DEFAULT_PROPOSALS_DIR,
        help="Proposals directory (default: ./proposals).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Evolution log directory (default: ./logs).",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=24.0,
        help="Time window in hours (default: 24).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write digest to this file instead of stdout.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown).",
    )
    args = parser.parse_args(argv)

    digest = build_digest(
        proposals_dir=args.proposals_dir,
        log_dir=args.log_dir,
        window_hours=args.hours,
        fmt=args.format,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(digest)
        print(f"Digest written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(digest)

    return 0


if __name__ == "__main__":
    sys.exit(main())

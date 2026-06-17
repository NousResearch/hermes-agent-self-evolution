"""ProposalReviewer — CLI for reviewing evolution proposals.

Proposals live on disk under `proposals_dir/{skill_name}/{timestamp}/`
with a STATUS tombstone (PENDING | APPROVED | REJECTED).

This CLI lets a human:

    list     — show all proposals with status + delta
    show     — print review.md for one proposal
    diff     — print diff.patch for one proposal
    approve  — mark APPROVED and (unless --no-merge) write the evolved
               skill back to the live hermes-agent bundled skills dir,
               creating a timestamped backup
    reject   — mark REJECTED with an optional reason

Usage:
    python -m evolution.review.proposal_reviewer list
    python -m evolution.review.proposal_reviewer show github-code-review 20260418_171916
    python -m evolution.review.proposal_reviewer diff github-code-review 20260418_171916
    python -m evolution.review.proposal_reviewer approve github-code-review 20260418_171916
    python -m evolution.review.proposal_reviewer reject github-code-review 20260418_171916 --reason "hallucinated flag"

All commands accept `--proposals-dir` (default: ./proposals).

The approve command calls `write_back_skill(..., mode='auto', auto_merge=True)`
so the existing safety path — atomic overwrite + timestamped backup — is reused.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.proposals import (
    PROPOSAL_STATUS_APPROVED,
    PROPOSAL_STATUS_PENDING,
    PROPOSAL_STATUS_REJECTED,
)
from evolution.core.write_back import write_back_skill
from evolution.skills.skill_module import find_skill


DEFAULT_PROPOSALS_DIR = Path("./proposals")


# ──────────────────────────── data model ────────────────────────────
@dataclass
class ProposalEntry:
    """One on-disk proposal, discovered by walking the proposals dir."""
    skill_name: str
    timestamp: str
    proposal_dir: Path
    status: str  # PENDING / APPROVED / REJECTED / UNKNOWN
    decision: dict  # parsed decision.json (may be empty on corruption)

    @property
    def improvement(self) -> float:
        return float(self.decision.get("improvement", 0.0))

    @property
    def baseline_score(self) -> float:
        return float(self.decision.get("baseline_score", 0.0))

    @property
    def evolved_score(self) -> float:
        return float(self.decision.get("evolved_score", 0.0))

    @property
    def auto_merge(self) -> bool:
        return bool(self.decision.get("auto_merge", False))

    @property
    def mode(self) -> str:
        return str(self.decision.get("mode", "?"))

    @property
    def rejection_reason(self) -> Optional[str]:
        """For REJECTED proposals, a second line in STATUS may hold a reason."""
        status_file = self.proposal_dir / "STATUS"
        if not status_file.exists():
            return None
        lines = status_file.read_text().splitlines()
        if len(lines) >= 2 and lines[0].strip() == PROPOSAL_STATUS_REJECTED:
            return lines[1].strip() or None
        return None


# ──────────────────────────── discovery ────────────────────────────
def _read_status(status_file: Path) -> str:
    if not status_file.exists():
        return "UNKNOWN"
    first = status_file.read_text().splitlines()
    if not first:
        return "UNKNOWN"
    return first[0].strip() or "UNKNOWN"


def _load_decision(proposal_dir: Path) -> dict:
    p = proposal_dir / "decision.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def discover_proposals(proposals_dir: Path) -> List[ProposalEntry]:
    """Walk proposals_dir and return all proposals, newest first."""
    if not proposals_dir.exists():
        return []
    entries: List[ProposalEntry] = []
    for skill_dir in sorted(proposals_dir.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue
        for ts_dir in sorted(skill_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            if not (ts_dir / "decision.json").exists():
                continue  # not a valid proposal dir
            entries.append(
                ProposalEntry(
                    skill_name=skill_dir.name,
                    timestamp=ts_dir.name,
                    proposal_dir=ts_dir,
                    status=_read_status(ts_dir / "STATUS"),
                    decision=_load_decision(ts_dir),
                )
            )
    # newest first (timestamps are sortable strings)
    entries.sort(key=lambda e: (e.skill_name, e.timestamp), reverse=True)
    return entries


def find_proposal(
    proposals_dir: Path, skill_name: str, timestamp: Optional[str] = None
) -> Optional[ProposalEntry]:
    """Look up one proposal. If timestamp omitted, returns the newest for that skill."""
    all_for_skill = [
        e for e in discover_proposals(proposals_dir) if e.skill_name == skill_name
    ]
    if not all_for_skill:
        return None
    if timestamp is None:
        return all_for_skill[0]  # newest (already sorted desc)
    for e in all_for_skill:
        if e.timestamp == timestamp:
            return e
    return None


# ──────────────────────────── commands ────────────────────────────
_STATUS_ICON = {
    PROPOSAL_STATUS_PENDING: "⏳",
    PROPOSAL_STATUS_APPROVED: "✅",
    PROPOSAL_STATUS_REJECTED: "❌",
    "UNKNOWN": "❓",
}


def cmd_list(args: argparse.Namespace) -> int:
    entries = discover_proposals(args.proposals_dir)
    if not entries:
        print(f"No proposals found under {args.proposals_dir}")
        return 0

    if args.status:
        want = args.status.upper()
        entries = [e for e in entries if e.status == want]
        if not entries:
            print(f"No proposals with status={want}")
            return 0

    # header
    print(f"{'':<2} {'SKILL':<32} {'TIMESTAMP':<17} {'MODE':<8} {'Δ':>8} {'MERGE':<6}")
    print("─" * 82)
    for e in entries:
        icon = _STATUS_ICON.get(e.status, "❓")
        merge = "auto" if e.auto_merge else "no"
        print(
            f"{icon}  "
            f"{e.skill_name:<32.32} "
            f"{e.timestamp:<17} "
            f"{e.mode:<8} "
            f"{e.improvement:+.3f}  "
            f"{merge:<6}"
        )
    print()
    counts = {"PENDING": 0, "APPROVED": 0, "REJECTED": 0, "UNKNOWN": 0}
    for e in entries:
        counts[e.status] = counts.get(e.status, 0) + 1
    print(
        f"  Total: {len(entries)}  "
        f"⏳ pending={counts['PENDING']}  "
        f"✅ approved={counts['APPROVED']}  "
        f"❌ rejected={counts['REJECTED']}"
    )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    entry = find_proposal(args.proposals_dir, args.skill, args.timestamp)
    if entry is None:
        print(f"Proposal not found: {args.skill} {args.timestamp or '(latest)'}", file=sys.stderr)
        return 2
    review = entry.proposal_dir / "review.md"
    if not review.exists():
        print(f"review.md missing in {entry.proposal_dir}", file=sys.stderr)
        return 2
    print(review.read_text())
    print(f"\n[status: {entry.status}]")
    if entry.rejection_reason:
        print(f"[rejection reason: {entry.rejection_reason}]")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    entry = find_proposal(args.proposals_dir, args.skill, args.timestamp)
    if entry is None:
        print(f"Proposal not found: {args.skill} {args.timestamp or '(latest)'}", file=sys.stderr)
        return 2
    diff = entry.proposal_dir / "diff.patch"
    if not diff.exists():
        print(f"diff.patch missing in {entry.proposal_dir}", file=sys.stderr)
        return 2
    sys.stdout.write(diff.read_text())
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    entry = find_proposal(args.proposals_dir, args.skill, args.timestamp)
    if entry is None:
        print(f"Proposal not found: {args.skill} {args.timestamp or '(latest)'}", file=sys.stderr)
        return 2
    if entry.status == PROPOSAL_STATUS_APPROVED:
        print(f"Already APPROVED: {entry.skill_name} {entry.timestamp}")
        return 0
    if entry.status == PROPOSAL_STATUS_REJECTED and not args.force:
        print(
            f"Cannot approve a REJECTED proposal without --force: "
            f"{entry.skill_name} {entry.timestamp}",
            file=sys.stderr,
        )
        return 3

    evolved_file = entry.proposal_dir / "evolved_skill.md"
    if not evolved_file.exists():
        print(f"evolved_skill.md missing in {entry.proposal_dir}", file=sys.stderr)
        return 2
    evolved_text = evolved_file.read_text()

    # Flip STATUS first so state is authoritative even if the write-back is skipped.
    status_file = entry.proposal_dir / "STATUS"
    approved_by = args.approved_by or _default_approver()
    approval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    status_file.write_text(
        f"{PROPOSAL_STATUS_APPROVED}\napproved_by={approved_by}\napproved_at={approval_ts}\n"
    )
    print(f"✅ APPROVED: {entry.skill_name} {entry.timestamp} (by {approved_by})")

    if args.no_merge:
        print("  (--no-merge set — skipping write-back; live skill untouched)")
        return 0

    # Write-back to live bundled skill path.
    hermes_agent_path = (
        Path(args.hermes_agent_path).expanduser()
        if args.hermes_agent_path
        else get_hermes_agent_path()
    )
    live_path = find_skill(entry.skill_name, hermes_agent_path)
    if live_path is None:
        print(
            f"  ⚠️  could not find live SKILL.md for '{entry.skill_name}' under "
            f"{hermes_agent_path} — STATUS is APPROVED but no write-back happened",
            file=sys.stderr,
        )
        return 4

    result = write_back_skill(
        live_path=live_path,
        evolved_text=evolved_text,
        mode="auto",
        auto_merge=True,
        timestamp=approval_ts,
    )
    if result.merged:
        print(f"  📝 merged → {result.live_path}")
        print(f"  🛟 backup → {result.backup_path}")
        # Write a trace of where it went, for auditability.
        (entry.proposal_dir / "merge_receipt.json").write_text(
            json.dumps(
                {
                    "merged_to": str(result.live_path),
                    "backup_path": str(result.backup_path) if result.backup_path else None,
                    "approved_by": approved_by,
                    "approved_at": approval_ts,
                    "reason": result.reason,
                },
                indent=2,
            )
        )
    else:
        print(f"  ⚠️  write-back skipped: {result.reason}", file=sys.stderr)
        return 5
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    entry = find_proposal(args.proposals_dir, args.skill, args.timestamp)
    if entry is None:
        print(f"Proposal not found: {args.skill} {args.timestamp or '(latest)'}", file=sys.stderr)
        return 2
    if entry.status == PROPOSAL_STATUS_APPROVED and not args.force:
        print(
            f"Cannot reject an APPROVED proposal without --force: "
            f"{entry.skill_name} {entry.timestamp}",
            file=sys.stderr,
        )
        return 3

    status_file = entry.proposal_dir / "STATUS"
    reason = (args.reason or "").strip()
    rejected_by = args.rejected_by or _default_approver()
    rejection_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    body = f"{PROPOSAL_STATUS_REJECTED}\n"
    if reason:
        body += f"{reason}\n"
    body += f"rejected_by={rejected_by}\nrejected_at={rejection_ts}\n"
    status_file.write_text(body)
    print(f"❌ REJECTED: {entry.skill_name} {entry.timestamp} (by {rejected_by})")
    if reason:
        print(f"  reason: {reason}")
    return 0


# ──────────────────────────── helpers ────────────────────────────
def _default_approver() -> str:
    import os

    return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


# ──────────────────────────── CLI ────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proposal_reviewer",
        description="Review, approve, or reject skill-evolution proposals.",
    )
    parser.add_argument(
        "--proposals-dir",
        type=Path,
        default=DEFAULT_PROPOSALS_DIR,
        help=f"Proposals root (default: {DEFAULT_PROPOSALS_DIR})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # list
    p_list = sub.add_parser("list", help="List all proposals")
    p_list.add_argument(
        "--status",
        choices=["pending", "approved", "rejected", "unknown"],
        help="Filter by status",
    )
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = sub.add_parser("show", help="Print review.md for a proposal")
    p_show.add_argument("skill")
    p_show.add_argument("timestamp", nargs="?", help="omit for latest")
    p_show.set_defaults(func=cmd_show)

    # diff
    p_diff = sub.add_parser("diff", help="Print diff.patch for a proposal")
    p_diff.add_argument("skill")
    p_diff.add_argument("timestamp", nargs="?", help="omit for latest")
    p_diff.set_defaults(func=cmd_diff)

    # approve
    p_approve = sub.add_parser("approve", help="Approve and write-back a proposal")
    p_approve.add_argument("skill")
    p_approve.add_argument("timestamp", nargs="?", help="omit for latest")
    p_approve.add_argument(
        "--no-merge",
        action="store_true",
        help="Mark APPROVED but skip write-back to live skill",
    )
    p_approve.add_argument(
        "--force",
        action="store_true",
        help="Allow approving a previously-rejected proposal",
    )
    p_approve.add_argument("--approved-by", help="Override approver username")
    p_approve.add_argument(
        "--hermes-agent-path",
        help="Override hermes-agent root (defaults to ~/.hermes/hermes-agent)",
    )
    p_approve.set_defaults(func=cmd_approve)

    # reject
    p_reject = sub.add_parser("reject", help="Reject a proposal")
    p_reject.add_argument("skill")
    p_reject.add_argument("timestamp", nargs="?", help="omit for latest")
    p_reject.add_argument("--reason", help="Short rejection reason")
    p_reject.add_argument(
        "--force",
        action="store_true",
        help="Allow rejecting a previously-approved proposal",
    )
    p_reject.add_argument("--rejected-by", help="Override rejecter username")
    p_reject.set_defaults(func=cmd_reject)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

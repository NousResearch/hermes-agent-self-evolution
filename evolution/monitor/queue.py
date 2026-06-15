"""Triage-queue assembly + rendering for the monitor sentinel.

Turns the sentinel's ranked candidates into a persisted, jq-aggregable
``triage_queue.json`` plus a human-readable report. Propose-only by construction:
the queue describes what the validated repair loop *could* attempt and how to
trigger it; it never evolves or opens a PR on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

from evolution.monitor.sentinel import RepairCandidate

QUEUE_SCHEMA_VERSION = "1"


def build_queue(candidates: list[RepairCandidate], *, repo: str, since_days: int) -> dict:
    by_kind: dict[str, int] = {}
    rows = []
    for rank, c in enumerate(candidates, start=1):
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
        rows.append({
            "rank": rank,
            "kind": c.kind,
            "tool": c.tool_path,
            "test": c.test_path,
            "fix_sha": c.fix_sha,
            "parent_sha": c.parent_sha,
            "committed_at": c.committed_at,
            "score": c.score,
        })
    return {
        "schema_version": QUEUE_SCHEMA_VERSION,
        "repo": repo,
        "since_days": since_days,
        "n_candidates": len(candidates),
        "by_kind": by_kind,
        "candidates": rows,
    }


def write_queue(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "triage_queue.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def attempt_command(repo: str, top: int) -> str:
    """The single propose-only action that repairs the top of the queue.

    Repair targeting reuses the validated campaign loop (harvest → repair →
    oracle gate); the monitor's own ``--attempt-top`` runs it on the ranked
    candidates and annotates the queue, without ever opening a PR."""
    return (f"python -m evolution.monitor --repo {repo} --attempt-top {top} "
            f"--max-cost-usd <cap>")


def render_report(payload: dict, *, top: int = 20) -> str:
    rows = payload["candidates"][:top]
    lines = [
        f"# Code-evolution triage queue — {payload['repo']}",
        "",
        f"{payload['n_candidates']} repair candidate(s) from the last "
        f"{payload['since_days']}d stream; by kind: {payload['by_kind']}.",
        "",
        "| # | kind | tool | fix | committed |",
        "|---|------|------|-----|-----------|",
    ]
    for r in rows:
        lines.append(f"| {r['rank']} | {r['kind']} | `{r['tool']}` | "
                     f"`{r['fix_sha'][:8]}` | {r['committed_at'][:10]} |")
    lines += [
        "",
        "_Propose-only: nothing is evolved or PR'd automatically._ "
        f"To attempt the top {min(top, len(rows))}:",
        "",
        f"    {attempt_command(payload['repo'], min(top, len(rows)))}",
    ]
    return "\n".join(lines)

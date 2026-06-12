"""Render a maintainer-local review dossier for a deployed evolved artifact.

Honest by construction: it shows WHAT changed (the live-baseline → deployed
diff) and HOW the deployed candidate was selected (search position, val delta
vs the seed, knee-point rationale, merge-step count). It makes NO per-hunk
attribution and NO per-task "evidence" claims — a staff-ML review found that
fragile (uncorrected multiple comparisons over noisy per-instance judge scores,
broken by merge/baseline-drift/combination hunks) and attributing to the judge
rather than behavior. This is the defensible subset.

Local artifact only (``output_dir/dossier.md``) — never a PR body.
"""
from __future__ import annotations

import difflib
from typing import Any, Optional


def _unified_diff(before: str, after: str, *, fromfile: str, tofile: str) -> str:
    # splitlines() (no keepends) + lineterm="" so each emitted row is its own
    # line when joined — otherwise a single-line artifact with no trailing
    # newline smushes the trailing "-old" and leading "+new" onto one line.
    diff = difflib.unified_diff(
        (before or "").splitlines(),
        (after or "").splitlines(),
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
    )
    return "\n".join(diff)


def _record_by_idx(lineage: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {r["idx"]: r for r in lineage.get("candidates", [])}


def _chain_to_root(by_idx: dict[int, dict[str, Any]], start: int) -> tuple[list[int], int]:
    """Walk parents (first listed) from ``start`` to a root, returning the path
    and the count of merge steps (candidates with >1 parent) seen along it.

    Cycle-guarded; merges are counted but the walk follows parents[0] so a single
    path is always produced.
    """
    path: list[int] = []
    merges = 0
    seen: set[int] = set()
    cur: Optional[int] = start
    while cur is not None and cur not in seen:
        seen.add(cur)
        path.append(cur)
        rec = by_idx.get(cur)
        parents = (rec or {}).get("parents")
        if not parents:  # None or [] → root
            break
        real = [p for p in parents if p is not None]
        if len(real) > 1:
            merges += 1
        cur = real[0] if real else None
    return path, merges


def render_dossier(lineage: dict[str, Any]) -> str:
    """Render the dossier markdown from a lineage record (see lineage.py)."""
    by_idx = _record_by_idx(lineage)
    deployed_idx = lineage["deployed_idx"]
    best_idx = lineage["best_idx"]
    n = lineage.get("n_candidates", len(by_idx))
    deployed = by_idx.get(deployed_idx, {})
    root = by_idx.get(0, {})
    seed_text = lineage.get("seed_text", "") or ""
    live_baseline = lineage.get("live_baseline_text", "") or ""
    deployed_text = deployed.get("text") or ""

    lines: list[str] = ["# Evolution dossier (maintainer-local review)", ""]

    # --- Selection rationale (from persisted data, no inference) ---
    lines.append("## How the deployed candidate was selected")
    dep_val = deployed.get("val_aggregate")
    root_val = root.get("val_aggregate")
    if dep_val is not None and root_val is not None:
        lines.append(f"- val_aggregate: {dep_val:.4f} (seed {root_val:.4f}, "
                     f"Δ {dep_val - root_val:+.4f})")
    lines.append(f"- candidate {deployed_idx} of {n} explored"
                 + ("" if deployed_idx == best_idx
                    else f"  (GEPA val-argmax was candidate {best_idx})"))
    disc = deployed.get("discovery_eval_count")
    if disc is not None:
        lines.append(f"- discovered after {disc} metric calls")
    selection = lineage.get("selection") or {}
    if selection:
        lines.append(f"- selection: {selection}")
    chain, merges = _chain_to_root(by_idx, deployed_idx)
    lines.append(f"- lineage depth: {len(chain) - 1} step(s) from seed"
                 + (f"; includes {merges} merge step(s) — step-level provenance N/A"
                    if merges else ""))
    lines.append("")

    # --- Pre-GEPA baseline drift, if any ---
    if live_baseline.strip() != seed_text.strip():
        lines.append("## ⚠ Pre-GEPA baseline drift")
        lines.append("The live baseline differs from GEPA's seed; the changes below "
                     "are NOT from search. Reconcile before trusting the deploy diff.")
        lines.append("```diff")
        lines.append(_unified_diff(live_baseline, seed_text,
                                   fromfile="live_baseline", tofile="gepa_seed").rstrip("\n"))
        lines.append("```")
        lines.append("")

    # --- The deploy diff ---
    lines.append("## Deploy diff (live baseline → deployed)")
    lines.append("```diff")
    body = _unified_diff(live_baseline, deployed_text,
                         fromfile="live_baseline", tofile="deployed").rstrip("\n")
    lines.append(body if body else "(no textual change)")
    lines.append("```")
    lines.append("")
    lines.append("_Shows what changed and how the candidate was selected. It does "
                 "not claim which individual change caused which behavior._")
    return "\n".join(lines) + "\n"


def write_dossier(output_dir, lineage: dict[str, Any]):
    from pathlib import Path
    path = Path(output_dir) / "dossier.md"
    path.write_text(render_dossier(lineage), encoding="utf-8")
    return path

"""Persist a GEPA run's lineage so a deployed artifact's diff is reviewable.

DSPy's ``detailed_results`` (DspyGEPAResult) exposes ``candidates``, ``parents``
(lineage: parents[i] = list of parent indices or None), ``val_aggregate_scores``,
``val_subscores`` (per-candidate per-val-instance), and ``discovery_eval_counts``
— all populated under ``track_stats=True`` (every seam passes it) and otherwise
discarded. We persist them to ``output_dir/lineage.json`` so the dossier (and
any later analysis) can reconstruct how the deployed candidate was reached.

Two facts a consumer must respect, baked into the schema:
  - The DEPLOYED candidate is not always GEPA's ``best_idx`` (the skill seam's
    knee-point selector can pick another) — so ``deployed_idx`` is explicit.
  - The deploy diff is against the LIVE baseline artifact, which may differ from
    GEPA's seed (candidate 0) — so both ``seed_text`` and ``live_baseline_text``
    are stored, letting the dossier separate pre-GEPA drift from search changes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

LINEAGE_NAME = "lineage.json"
_SCHEMA_VERSION = "1"


def _safe_extract(extract_text: Callable[[Any], str], candidate: Any) -> Optional[str]:
    try:
        return extract_text(candidate)
    except Exception:
        return None


def build_lineage(
    details: Any,
    *,
    extract_text: Callable[[Any], str],
    deployed_idx: int,
    selection: dict[str, Any],
    seed_text: str,
    live_baseline_text: str,
    suite_sha256: str = "",
) -> Optional[dict[str, Any]]:
    """Reduce a GEPA detailed_results to a JSON-able lineage record, or None.

    Returns None on the MIPROv2 fallback (no ``parents``) — nothing to record.
    """
    if not hasattr(details, "parents") or not hasattr(details, "candidates"):
        return None
    candidates = list(details.candidates)
    parents = list(details.parents)
    val_agg = [float(v) for v in getattr(details, "val_aggregate_scores", []) or []]
    val_sub = getattr(details, "val_subscores", None)
    disc = getattr(details, "discovery_eval_counts", None)
    best_idx = int(details.best_idx)

    records: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        records.append({
            "idx": i,
            "parents": parents[i] if i < len(parents) else None,
            "val_aggregate": val_agg[i] if i < len(val_agg) else None,
            "val_subscores": (
                [float(x) for x in val_sub[i]] if val_sub and i < len(val_sub) else None
            ),
            "discovery_eval_count": (
                int(disc[i]) if disc and i < len(disc) else None
            ),
            "text": _safe_extract(extract_text, cand),
            "is_best": i == best_idx,
            "is_deployed": i == deployed_idx,
        })

    return {
        "schema_version": _SCHEMA_VERSION,
        "deployed_idx": deployed_idx,
        "best_idx": best_idx,
        "n_candidates": len(candidates),
        "seed_text": seed_text,
        "live_baseline_text": live_baseline_text,
        "selection": selection,
        "suite_sha256": suite_sha256,
        "candidates": records,
    }


def write_lineage(output_dir: Path, details: Any, **kwargs: Any) -> Optional[Path]:
    """Write ``output_dir/lineage.json``; returns the path, or None if skipped."""
    lineage = build_lineage(details, **kwargs)
    if lineage is None:
        return None
    path = Path(output_dir) / LINEAGE_NAME
    path.write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    return path

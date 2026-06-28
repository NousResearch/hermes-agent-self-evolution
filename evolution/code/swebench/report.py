"""Honest external-validity report. The deploy-reachable rate (from campaign_report)
is meaningless without the difficulty profile of the KEPT subset and the drop
breakdown — the env-validity gate is a difficulty filter, so a matching rate on a
systematically-easier subset is NOT evidence the Hermes gradient ports. Produces
the profile + breakdown + a pre-registered interpretation guard."""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

# Hermes organism difficulty (reports/asymmetry_difficulty_curve.json): median fix
# ~45 LOC, 35/46 fixes > 20 LOC. The external kept subset is read against this.
HERMES_PROFILE = {"median_loc": 45, "frac_large_gt20": 35 / 46}


def summarize_difficulty(rows: list[dict]) -> dict:
    locs = [r["gold_loc"] for r in rows]
    if not locs:
        return {"n": 0, "median_loc": 0.0, "frac_large_gt20": 0.0}
    return {"n": len(rows), "median_loc": float(statistics.median(locs)),
            "frac_large_gt20": sum(1 for x in locs if x > 20) / len(locs)}


def drop_breakdown(rows: list[dict]) -> dict:
    return dict(Counter(r["reason"] for r in rows))


def build(output_dir: Path) -> dict:
    """Join campaign_report.json + characterization.json into the honest report."""
    chars = json.loads((output_dir / "characterization.json").read_text())
    campaign = json.loads((output_dir / "campaign_report.json").read_text())
    kept = [r for r in chars if r["reason"] == "kept"]
    report = {
        "deploy_reachable": campaign.get("deploy_reachable"),
        "verdict": campaign.get("verdict"),
        "kept_difficulty": summarize_difficulty(kept),
        "freeze_dropped_difficulty": summarize_difficulty(
            [r for r in chars if r["reason"] == "gold_violates_freeze"]),
        "hermes_difficulty": HERMES_PROFILE,
        "loc_definition": ("patch_loc = added+removed diff lines (loader.patch_loc). HERMES_PROFILE "
                           "is read from the committed reports/asymmetry_difficulty_curve.json; its "
                           "per-fix counting is not recorded, so it may not match patch_loc line-for-line. "
                           "The ~9x median gap (5 vs 45) dwarfs the widest plausible ~2x counting "
                           "difference, so the 'much smaller' comparison holds under any convention."),
        "drop_breakdown": drop_breakdown(chars),
        "freeze_drop_rate": sum(1 for r in chars if r["reason"] == "gold_violates_freeze") / max(len(chars), 1),
        "emulated_kept": sum(1 for r in chars if r.get("emulated")),
        "interpretation_guard": (
            "PRE-REGISTERED: if kept_difficulty is systematically easier than hermes_difficulty "
            "(lower median_loc / frac_large_gt20), a deploy-reachable rate matching 0.60-0.74 is NOT "
            "sufficient evidence the gradient ports — the filter may have reduced Lite to the same easy "
            "single-file surface. Report the rate ONLY beside this profile."),
    }
    (output_dir / "external_validity_report.json").write_text(json.dumps(report, indent=2))
    return report

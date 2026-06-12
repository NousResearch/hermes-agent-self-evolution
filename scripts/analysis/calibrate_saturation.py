"""Saturation-threshold calibration: mine the gate archive, honestly.

The saturation pre-flight (``evolution/core/saturation_check.py``) default-denies
a run when the baseline's holdout score clears uncalibrated magic thresholds
(``no_headroom_synthetic=0.99``, ``weak_signal_synthetic=0.95``). This asks the
archive: if those thresholds had been enforced, how often would they have
WRONGLY aborted a run that actually produced a statistically-supported
improvement? That false-abort rate is the calibration signal.

The honest answer this archive can give is narrow. ``avg_baseline`` in
``gate_decision.json`` reconstructs the pre-flight's own holdout score (it is the
same vector, reused verbatim when the pre-flight ran). But the gated region is
nearly empty and the pre-flight's abort decision was never persisted, so the
study is a survivorship-bounded counterfactual on GEPA-completed runs, not a
measurement of the deployed policy. The closed-loop thresholds (0.95/0.15) are
out of scope here by design — this script calibrates only the synthetic
threshold; the closed-loop thresholds are calibrated from the forward ledger.

The script therefore leads with sample sizes and Wilson bounds and concludes
with the data-collection fix (the saturation ledger this repo now writes),
never a thumbs-up on the numbers. Pure stat replay over recorded fields; zero
LLM calls.

Usage:
    uv run python scripts/analysis/calibrate_saturation.py
    uv run python scripts/analysis/calibrate_saturation.py --since 20260601_000000
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Optional

# Candidate thresholds the pre-flight could enforce; 0.99/0.95 are the live ones.
THRESHOLD_SWEEP: tuple[float, ...] = (0.95, 0.97, 0.99, 1.0)
GAIN_DELTAS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)
BIN_LABELS: tuple[str, ...] = ("<0.90", "0.90-0.95", "0.95-0.99", ">=0.99")


# --- pure functions (importable by tests) -----------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials.

    The informative quantity when the point estimate is 0/n: the upper bound is
    how high the true rate could plausibly be given how little data we have.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def assign_bin(avg_baseline: float) -> str:
    """Bucket a baseline holdout score around the candidate thresholds."""
    if avg_baseline < 0.90:
        return "<0.90"
    if avg_baseline < 0.95:
        return "0.90-0.95"
    if avg_baseline < 0.99:
        return "0.95-0.99"
    return ">=0.99"


def real_improvement(
    gate: dict[str, Any], *, definition: str = "lower_bound", delta: float = 0.0
) -> bool:
    """Whether a run counts as a real improvement under a given definition.

    - ``lower_bound`` (primary): paired-bootstrap lower bound > 0 — a genuine
      statistical claim, independent of the gate's own deploy verdict.
    - ``decision`` (circular): the gate said "deploy" — labeled circular
      because under ``no_regression`` it collapses to "did not regress".
    - ``gain``: realized mean holdout gain ≥ delta — an effect-size view.
    """
    if definition == "lower_bound":
        lb = (gate.get("bootstrap") or {}).get("lower_bound")
        return lb is not None and lb > 0
    if definition == "decision":
        return gate.get("decision") == "deploy"
    if definition == "gain":
        ab, ae = gate.get("avg_baseline"), gate.get("avg_evolved")
        return ab is not None and ae is not None and (ae - ab) >= delta
    raise ValueError(f"unknown definition: {definition}")


def false_abort_sweep(
    runs: list[dict[str, Any]],
    *,
    thresholds: tuple[float, ...] = THRESHOLD_SWEEP,
    definition: str = "lower_bound",
    delta: float = 0.0,
) -> list[dict[str, Any]]:
    """For each threshold τ, the empirical false-abort rate + Wilson bounds."""
    total = len(runs)
    out: list[dict[str, Any]] = []
    for tau in thresholds:
        would_abort = [
            g for g in runs
            if g.get("avg_baseline") is not None and g["avg_baseline"] >= tau
        ]
        n = len(would_abort)
        improved = sum(
            1 for g in would_abort
            if real_improvement(g, definition=definition, delta=delta)
        )
        rate = improved / n if n else 0.0
        lo, hi = wilson_interval(improved, n)
        out.append({
            "threshold": tau,
            "would_abort_n": n,
            "would_abort_frac_of_archive": (n / total) if total else 0.0,
            "n_real_improvement": improved,
            "false_abort_rate": rate,
            "wilson_lower": lo,
            "wilson_upper": hi,
        })
    return out


def _is_homogeneous(gate: dict[str, Any]) -> bool:
    """The clean stratum: balanced proposer + no_regression gate + synthetic."""
    return (
        gate.get("fitness_profile") == "balanced"
        and gate.get("gate_mode") == "no_regression"
        and gate.get("decision_signal") in (None, "synthetic")
    )


def _bin_stats(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_bin: dict[str, list[dict[str, Any]]] = {b: [] for b in BIN_LABELS}
    for g in runs:
        by_bin[assign_bin(g["avg_baseline"])].append(g)
    rows: list[dict[str, Any]] = []
    for label in BIN_LABELS:
        group = by_bin[label]
        n = len(group)
        deploys = sum(1 for g in group if g.get("decision") == "deploy")
        gains = [
            g["avg_evolved"] - g["avg_baseline"]
            for g in group
            if g.get("avg_evolved") is not None
        ]
        lbs = [
            (g.get("bootstrap") or {}).get("lower_bound")
            for g in group
        ]
        lbs = [x for x in lbs if x is not None]
        n_lb_pos = sum(1 for x in lbs if x > 0)
        n_examples = [
            (g.get("bootstrap") or {}).get("n_examples")
            for g in group
        ]
        n_examples = [x for x in n_examples if x is not None]
        no_op_deploys = sum(
            1 for g in group
            if g.get("decision") == "deploy"
            and g.get("avg_evolved") is not None
            and abs(g["avg_evolved"] - g["avg_baseline"]) < 1e-9
            and ((g.get("bootstrap") or {}).get("lower_bound") in (None, 0, 0.0))
        )
        deploy_lo, deploy_hi = wilson_interval(deploys, n)
        lbpos_lo, lbpos_hi = wilson_interval(n_lb_pos, n)
        rows.append({
            "bin": label,
            "n": n,
            "deploy_rate": (deploys / n) if n else 0.0,
            "deploy_rate_wilson": [deploy_lo, deploy_hi],
            "mean_realized_gain": statistics.mean(gains) if gains else None,
            "lower_bound_min": min(lbs) if lbs else None,
            "lower_bound_median": statistics.median(lbs) if lbs else None,
            "lower_bound_max": max(lbs) if lbs else None,
            "frac_lower_bound_pos": (n_lb_pos / n) if n else 0.0,
            "frac_lower_bound_pos_wilson": [lbpos_lo, lbpos_hi],
            "no_op_deploy_frac": (no_op_deploys / n) if n else 0.0,
            "n_examples_min": min(n_examples) if n_examples else None,
            "n_examples_median": statistics.median(n_examples) if n_examples else None,
            "n_examples_max": max(n_examples) if n_examples else None,
        })
    return rows


# --- archive loading ---------------------------------------------------------

def load_gate_decisions(
    output_root: Path, since: Optional[str]
) -> tuple[list[dict[str, Any]], int]:
    """Walk output/**/gate_decision.json. since filters by run-dir name prefix.

    Returns (runs, n_skipped). Unparseable/unreadable files are skipped — but
    counted, so the "leads with sample sizes" report can disclose them rather
    than silently shrinking the denominator.
    """
    runs: list[dict[str, Any]] = []
    n_skipped = 0
    for gate_path in sorted(Path(output_root).rglob("gate_decision.json")):
        if since and gate_path.parent.name < since:
            continue
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            n_skipped += 1
            continue
        gate["_run_id"] = gate_path.parent.name
        runs.append(gate)
    return runs, n_skipped


def analyze(all_runs: list[dict[str, Any]], *, n_skipped: int = 0) -> dict[str, Any]:
    """Run the full methodology and return a JSON-able analysis record."""
    n_total = len(all_runs)
    n_closed_loop = sum(1 for g in all_runs if g.get("decision_signal") == "closed_loop")
    # Synthetic-calibration pool: paired holdout vector + bootstrap, exclude CL.
    pool = [
        g for g in all_runs
        if g.get("avg_baseline") is not None
        and g.get("bootstrap")
        and g.get("decision_signal") != "closed_loop"
    ]
    homogeneous = [g for g in pool if _is_homogeneous(g)]
    off_profile = [g for g in pool if not _is_homogeneous(g)]

    sensitivity: dict[str, Any] = {
        "lower_bound": false_abort_sweep(homogeneous, definition="lower_bound"),
        "decision_CIRCULAR": false_abort_sweep(homogeneous, definition="decision"),
    }
    for d in GAIN_DELTAS:
        sensitivity[f"gain>={d}"] = false_abort_sweep(
            homogeneous, definition="gain", delta=d
        )

    schema_versions: dict[str, int] = {}
    for g in all_runs:
        sv = str(g.get("schema_version", "unknown"))
        schema_versions[sv] = schema_versions.get(sv, 0) + 1

    return {
        "n_total": n_total,
        "n_skipped_unparseable": n_skipped,
        "n_paired_pool": len(pool),
        "n_closed_loop_excluded": n_closed_loop,
        "n_homogeneous": len(homogeneous),
        "n_off_profile": len(off_profile),
        "schema_versions": schema_versions,
        "bins_homogeneous": _bin_stats(homogeneous),
        "bins_full_pool": _bin_stats(pool),
        "false_abort_primary": sensitivity["lower_bound"],
        "false_abort_sensitivity": sensitivity,
    }


def scan_overfitting(output_root: Path) -> dict[str, Any]:
    """Forward-only: per-run val trajectory vs discovery order from lineage.json.

    Zero lineage files in the historical archive (the feature postdates it), so
    today this reports forward-only; it activates as lineage-bearing runs accrue.
    """
    lineages = sorted(Path(output_root).rglob("lineage.json"))
    flagged: list[dict[str, Any]] = []
    for path in lineages:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cands = data.get("candidates") or []
        ordered = sorted(
            (c for c in cands if c.get("discovery_eval_count") is not None
             and c.get("val_aggregate") is not None),
            key=lambda c: c["discovery_eval_count"],
        )
        if len(ordered) < 3:
            continue
        best_val = max(c["val_aggregate"] for c in ordered)
        # Val peaked early but search kept spending — a plateau-before-exhaustion
        # signature (the cheap proxy for overfitting we can compute; per-candidate
        # holdout isn't stored, so this is val-only, not val-vs-holdout).
        first_peak = next(
            c for c in ordered if c["val_aggregate"] >= best_val - 1e-9
        )
        if first_peak["discovery_eval_count"] < ordered[-1]["discovery_eval_count"]:
            flagged.append({
                "run_id": path.parent.name,
                "peak_at_eval": first_peak["discovery_eval_count"],
                "last_eval": ordered[-1]["discovery_eval_count"],
                "n_candidates": len(ordered),
            })
    return {"n_lineage_runs": len(lineages), "plateau_flagged": flagged}


def scan_saturation_ledger(output_root: Path) -> dict[str, Any]:
    """Status of the forward saturation ledger (the data-collection fix)."""
    ledger = Path(output_root) / "saturation_ledger.jsonl"
    if not ledger.exists():
        return {"exists": False, "n_rows": 0, "n_aborted": 0}
    # The ledger is appended to live by evolve runs; tolerate a torn final line
    # (a killed run mid-write) rather than crashing the whole calibration.
    rows: list[dict[str, Any]] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {
        "exists": True,
        "n_rows": len(rows),
        "n_aborted": sum(1 for r in rows if not r.get("proceeded", True)),
        "n_in_gated_region": sum(
            1 for r in rows
            if r.get("holdout_score") is not None and r["holdout_score"] >= 0.95
        ),
    }


# --- rendering ---------------------------------------------------------------

def _fmt(x: Optional[float], nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def render_markdown(
    analysis: dict[str, Any],
    overfitting: dict[str, Any],
    ledger: dict[str, Any],
) -> str:
    primary = analysis["false_abort_primary"]
    # The decisive numbers for the lead: would-abort n + Wilson upper at the live thresholds.
    at = {r["threshold"]: r for r in primary}
    # Pick the lowest live threshold as the gated-region probe, defensively
    # (THRESHOLD_SWEEP could be edited to not contain exactly 0.95/0.99).
    r_low = at.get(0.95) or (primary[0] if primary else None)
    r_high = at.get(0.99) or (primary[-1] if primary else None)
    gated_n = r_low["would_abort_n"] if r_low else 0
    gated_signal = r_low["n_real_improvement"] if r_low else 0
    # "Data-starved" = the gated region is both nearly empty and shows no
    # improvement signal. Once the ledger fills it, this flips and the report
    # stops calling itself a survivorship counterfactual.
    data_starved = gated_signal == 0 and gated_n < 30

    L = []
    L.append("# Saturation-threshold calibration: findings\n")
    intro = (
        "This report mines the deploy-gate archive to ask whether the saturation "
        "pre-flight's synthetic thresholds (`no_headroom_synthetic=0.99`, "
        "`weak_signal_synthetic=0.95`) would wrongly abort runs that actually "
        "improve. "
    )
    if data_starved:
        intro += (
            "**The archive cannot yet settle the thresholds** — it is data-"
            "starved in exactly the gated region, and the pre-flight's abort "
            "decision was never persisted historically. The numbers below are a "
            "survivorship-bounded counterfactual on GEPA-completed runs, "
            "reported with Wilson bounds so absence of evidence is not mistaken "
            "for evidence of safety."
        )
    else:
        intro += (
            f"The gated region (baseline ≥ 0.95) now holds {gated_n} run(s), "
            f"{gated_signal} with a statistically-supported improvement, so the "
            "false-abort rate below is becoming a real measurement rather than a "
            "survivorship counterfactual. Wilson bounds still apply."
        )
    L.append(intro + "\n")

    L.append("## Headline\n")
    if r_low and r_high:
        L.append(
            f"At τ={r_low['threshold']}, **{r_low['would_abort_n']} runs** would "
            f"be aborted ({r_low['would_abort_frac_of_archive'] * 100:.1f}% of "
            f"the pool); {r_low['n_real_improvement']} produced a statistically-"
            f"supported improvement → false-abort rate "
            f"{r_low['false_abort_rate'] * 100:.1f}% (Wilson 95% upper bound "
            f"**{r_low['wilson_upper'] * 100:.1f}%**). At τ={r_high['threshold']}, "
            f"{r_high['would_abort_n']} runs, {r_high['n_real_improvement']} "
            f"improvements, upper bound **{r_high['wilson_upper'] * 100:.1f}%**. "
            + (
                "A point estimate of 0% on a near-empty gated region is absence "
                "of evidence; the upper bound is the honest read.\n"
                if data_starved
                else "\n"
            )
        )
    else:
        L.append("No paired-vector runs in the pool — nothing to calibrate.\n")
    L.append(
        f"Pool: {analysis['n_paired_pool']} paired-vector runs of "
        f"{analysis['n_total']} archived "
        f"({analysis.get('n_skipped_unparseable', 0)} unparseable files skipped; "
        f"{analysis['n_closed_loop_excluded']} closed-loop runs excluded from "
        f"synthetic calibration; {analysis['n_homogeneous']} in the homogeneous "
        f"balanced+no_regression+synthetic stratum used for the primary "
        f"analysis, {analysis['n_off_profile']} off-profile). Schema versions: "
        f"{analysis['schema_versions']}.\n"
    )

    L.append("## Finding 1 — Binning by baseline holdout score (homogeneous stratum)\n")
    L.append(
        "| bin | n | deploy% (Wilson) | mean gain | lb>0 frac (Wilson) | "
        "no-op deploy% | lb min/med/max | n_ex med |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for b in analysis["bins_homogeneous"]:
        dl, dh = b["deploy_rate_wilson"]
        pl, ph = b["frac_lower_bound_pos_wilson"]
        L.append(
            f"| {b['bin']} | {b['n']} | "
            f"{b['deploy_rate'] * 100:.0f}% [{dl * 100:.0f},{dh * 100:.0f}] | "
            f"{_fmt(b['mean_realized_gain'])} | "
            f"{b['frac_lower_bound_pos'] * 100:.0f}% [{pl * 100:.0f},{ph * 100:.0f}] | "
            f"{b['no_op_deploy_frac'] * 100:.0f}% | "
            f"{_fmt(b['lower_bound_min'])}/{_fmt(b['lower_bound_median'])}/"
            f"{_fmt(b['lower_bound_max'])} | "
            f"{_fmt(b['n_examples_median'], 0)} |"
        )
    L.append(
        "\nUnder `no_regression` (the bulk of the archive) a 'deploy' means "
        "'did not regress', not 'improved' — read the `lb>0 frac` and the "
        "`no-op deploy%` columns, not the deploy rate.\n"
    )

    L.append("## Finding 2 — False-abort sweep (primary: bootstrap lower bound > 0)\n")
    L.append("| τ | would-abort n | % of pool | real improvements | false-abort rate | Wilson 95% upper |")
    L.append("|---|---|---|---|---|---|")
    for r in primary:
        L.append(
            f"| {r['threshold']} | {r['would_abort_n']} | "
            f"{r['would_abort_frac_of_archive'] * 100:.1f}% | "
            f"{r['n_real_improvement']} | {r['false_abort_rate'] * 100:.1f}% | "
            f"{r['wilson_upper'] * 100:.1f}% |"
        )

    L.append("\n## Finding 3 — Sensitivity across improvement definitions\n")
    L.append("False-abort rate (Wilson upper) at each τ, per definition:\n")
    L.append("| definition | " + " | ".join(f"τ={t}" for t in THRESHOLD_SWEEP) + " |")
    L.append("|---|" + "|".join("---" for _ in THRESHOLD_SWEEP) + "|")
    for name, sweep in analysis["false_abort_sensitivity"].items():
        cells = " | ".join(
            f"{r['false_abort_rate'] * 100:.0f}% (≤{r['wilson_upper'] * 100:.0f}%)"
            for r in sweep
        )
        L.append(f"| {name} | {cells} |")
    L.append(
        "\n`decision_CIRCULAR` is the gate's own verdict — reported only to show "
        "how badly the circular definition inflates apparent success; it is not "
        "the headline.\n"
    )

    L.append("## Threats to validity\n")
    if data_starved:
        L.append(
            "- **Data-starvation (fatal to the headline):** the gated region "
            f"holds only {gated_n} run(s) and none show a statistically-"
            "supported improvement. 0% false-abort is absence of evidence — see "
            "the Wilson upper bounds.\n"
        )
    else:
        L.append(
            f"- **Gated-region coverage:** the gated region holds {gated_n} "
            f"run(s), {gated_signal} with a statistically-supported improvement; "
            "the false-abort rate is now a measurement, but the Wilson bounds "
            "still gate how far it generalizes.\n"
        )
    L.append(
        "- **Survivorship:** the archive contains only GEPA-completed runs; the "
        "pre-flight's abort was almost never recorded, so this is a reconstructed "
        "counterfactual on survivors, not a measurement of the deployed policy.\n"
        "- **Circularity:** `decision` is the gate's own bootstrap-driven output "
        "and under `no_regression` collapses to `mean ≥ 0`; hence the primary "
        "definition is `lower_bound > 0`, not `decision`.\n"
        "- **Proxy exactness:** `avg_baseline` equals the pre-flight's holdout "
        "score verbatim when the pre-flight ran (cache reuse), and is the same "
        "estimator on the same examples when it was skipped.\n"
        "- **Heterogeneity:** per-example vectors index different synthetic "
        "datasets and holdout sizes; runs are treated as exchangeable units, "
        "never pooled at the example level. `n_examples` reported per bin.\n"
        "- **Small-N / multiple comparisons:** every rate carries a Wilson "
        "interval; the definition×threshold grid is descriptive, not a battery "
        "of tests. The primary definition was fixed before reading outcomes.\n"
    )

    L.append("## Data-collection recommendation (the actual fix)\n")
    status = (
        f"present: {ledger['n_rows']} rows, {ledger['n_aborted']} aborts, "
        f"{ledger.get('n_in_gated_region', 0)} in the gated region (holdout ≥ 0.95)"
        if ledger["exists"] else "not yet written"
    )
    L.append(
        "The archive can never settle these thresholds because aborted runs and "
        "the pre-flight band+score were not logged. The fix ships alongside this "
        "report: `evolution/core/saturation_telemetry.py` writes one "
        "`output/saturation_ledger.jsonl` row per pre-flight — including aborts — "
        f"joined to each run by `run_id`. Ledger status: **{status}**. Re-run "
        "this script once the ledger accrues runs in the gated region with "
        "measured outcomes; the false-abort rate then becomes a real measurement "
        "rather than a survivorship counterfactual.\n"
    )

    L.append("## Overfitting trajectory (forward-only)\n")
    if overfitting["n_lineage_runs"] == 0:
        L.append(
            "0 `lineage.json` files in the archive — the lineage feature "
            "postdates the entire archive, so per-candidate val-vs-discovery-"
            "order analysis is forward-only and populates as new runs accrue. "
            "(Per-candidate holdout scores are never stored, so even forward the "
            "signal is 'val plateaus before the search budget is spent', not "
            "'val climbs while holdout flattens'.)\n"
        )
    else:
        L.append(
            f"{overfitting['n_lineage_runs']} lineage-bearing run(s); "
            f"{len(overfitting['plateau_flagged'])} show val peaking before the "
            "search budget was exhausted (a plateau-before-exhaustion proxy):\n"
        )
        for f in overfitting["plateau_flagged"]:
            L.append(
                f"- `{f['run_id']}`: val peaked at eval "
                f"{f['peak_at_eval']} of {f['last_eval']} "
                f"({f['n_candidates']} candidates).\n"
            )

    L.append("## Out of scope\n")
    L.append(
        "Closed-loop thresholds (`no_headroom_closed_loop=0.95`, "
        "`uniform_failure_closed_loop=0.15`): too few closed-loop runs in the "
        "archive to calibrate; they wait on the forward ledger. Changing any "
        "threshold value is a separate behavior change, not part of this "
        "measurement.\n"
    )
    L.append("## Audit trail\n")
    L.append(
        "Regenerate: `uv run python scripts/analysis/calibrate_saturation.py`. "
        "Machine-readable companion: `reports/saturation_calibration.json`.\n"
    )
    return "\n".join(L)


def render_stdout(analysis: dict[str, Any]) -> str:
    lines = [
        f"Saturation calibration — pool {analysis['n_paired_pool']}"
        f"/{analysis['n_total']} runs "
        f"({analysis['n_homogeneous']} homogeneous, "
        f"{analysis['n_closed_loop_excluded']} closed-loop excluded)",
        "",
        f"{'bin':<12}{'n':>5}{'deploy%':>9}{'lb>0%':>8}{'no-op%':>9}{'mean gain':>11}",
        "-" * 54,
    ]
    for b in analysis["bins_homogeneous"]:
        lines.append(
            f"{b['bin']:<12}{b['n']:>5}{b['deploy_rate'] * 100:>8.0f}%"
            f"{b['frac_lower_bound_pos'] * 100:>7.0f}%"
            f"{b['no_op_deploy_frac'] * 100:>8.0f}%"
            f"{_fmt(b['mean_realized_gain']):>11}"
        )
    lines += ["", "False-abort sweep (primary: lower_bound>0):"]
    for r in analysis["false_abort_primary"]:
        lines.append(
            f"  τ={r['threshold']}: {r['would_abort_n']} would-abort, "
            f"{r['n_real_improvement']} improved → "
            f"{r['false_abort_rate'] * 100:.0f}% "
            f"(≤{r['wilson_upper'] * 100:.0f}% Wilson upper)"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate saturation thresholds against the gate archive."
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--since", default=None,
        help="Filter to run dirs whose name sorts >= this prefix (YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="Print the stdout summary only; don't write report files.",
    )
    args = parser.parse_args(argv)

    runs, n_skipped = load_gate_decisions(args.output_root, args.since)
    if not runs:
        print(f"No gate_decision.json found under {args.output_root}.")
        return 0
    analysis = analyze(runs, n_skipped=n_skipped)
    overfitting = scan_overfitting(args.output_root)
    ledger = scan_saturation_ledger(args.output_root)

    print(render_stdout(analysis))

    if not args.no_write:
        args.reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = args.reports_dir / "saturation_calibration_findings.md"
        json_path = args.reports_dir / "saturation_calibration.json"
        md_path.write_text(render_markdown(analysis, overfitting, ledger), encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {"analysis": analysis, "overfitting": overfitting, "ledger": ledger},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {md_path}\nWrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

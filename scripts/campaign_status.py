"""Aggregate per-run cost across `output/` for the calibration campaign.

Reads `output/<skill>/<ts>/metrics.json` files, sums the `cost` blocks
written by LMTimingCallback's litellm success_callback, and reports
totals against the campaign cap.

Usage:
    uv run python scripts/campaign_status.py
    uv run python scripts/campaign_status.py --since 20260507_000000
    uv run python scripts/campaign_status.py --cap 200

Runs that pre-date the cost-tracking change (no `cost` key in
metrics.json) are reported separately so the user can see how much of
the corpus is uninstrumented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_metrics(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _iter_runs(output_root: Path, since: str | None):
    """Yield (skill, timestamp, metrics_dict) for every run under output_root.

    `since` is a timestamp prefix (e.g. "20260507_000000"); runs whose
    timestamp directory name sorts < since are skipped. Lex-sort works
    because the timestamp format is YYYYMMDD_HHMMSS — fixed-width and
    monotonic.
    """
    if not output_root.exists():
        return
    for skill_dir in sorted(output_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        for run_dir in sorted(skill_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if since and run_dir.name < since:
                continue
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = _load_metrics(metrics_path)
            if metrics is None:
                continue
            yield skill_dir.name, run_dir.name, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-root",
        default="output",
        help="Root containing <skill>/<ts>/metrics.json (default: output)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only include runs whose timestamp directory >= this prefix "
        "(YYYYMMDD_HHMMSS). Use to scope to the campaign start.",
    )
    parser.add_argument(
        "--cap",
        type=float,
        default=200.0,
        help="Campaign spend cap in USD (default: $200).",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    grand_total = 0.0
    by_model: dict[str, dict[str, float | int]] = {}
    instrumented = 0
    uninstrumented = 0
    rows: list[tuple[str, str, float, dict[str, Any]]] = []

    for skill, ts, metrics in _iter_runs(output_root, args.since):
        cost_block = metrics.get("cost")
        if not cost_block:
            uninstrumented += 1
            continue
        instrumented += 1
        run_total = float(cost_block.get("total_usd", 0.0))
        grand_total += run_total
        rows.append((skill, ts, run_total, cost_block.get("by_model", {})))
        for model, model_row in (cost_block.get("by_model") or {}).items():
            agg = by_model.setdefault(
                model,
                {"tokens_in_uncached": 0, "tokens_in_cached": 0, "tokens_out": 0,
                 "calls": 0, "cost_usd": 0.0},
            )
            for k in ("tokens_in_uncached", "tokens_in_cached", "tokens_out", "calls"):
                agg[k] += int(model_row.get(k, 0))
            agg["cost_usd"] = float(agg["cost_usd"]) + float(model_row.get("cost_usd", 0.0))

    print(f"Campaign cost summary  (root={output_root}, since={args.since or 'all-time'})")
    print(f"  Runs instrumented:   {instrumented}")
    print(f"  Runs uninstrumented: {uninstrumented} (pre-cost-tracking — excluded from totals)")
    print()
    print(f"  {'Skill':<28} {'Timestamp':<18} {'Cost':>10}")
    print(f"  {'-'*28} {'-'*18} {'-'*10}")
    for skill, ts, run_total, _ in rows:
        print(f"  {skill:<28} {ts:<18} ${run_total:>8.4f}")
    print()
    print(f"  Per-model totals:")
    for model, row in sorted(by_model.items()):
        prompt_total = row["tokens_in_uncached"] + row["tokens_in_cached"]
        hit_rate = (
            row["tokens_in_cached"] / prompt_total if prompt_total > 0 else 0.0
        )
        print(
            f"    {model:<28} calls={row['calls']:>4} "
            f"in={row['tokens_in_uncached']:>10,}+{row['tokens_in_cached']:>10,} (hit={hit_rate:.1%}) "
            f"out={row['tokens_out']:>10,} cost=${row['cost_usd']:.4f}"
        )
    print()
    pct_of_cap = grand_total / args.cap * 100 if args.cap > 0 else 0
    flag = "" if grand_total < args.cap else "  ⚠ OVER CAP"
    print(f"  Grand total: ${grand_total:.4f} of ${args.cap:.2f} cap ({pct_of_cap:.1f}%){flag}")


if __name__ == "__main__":
    main()

"""Search telemetry: per-run val-discrimination signal from GEPA results.

The framework's binding constraint is the small-N tie problem. The val
aggregate is a low-resolution statistic (a handful of holdout examples, so
only a few distinct score levels), which means many GEPA candidates share an
identical aggregate score and the val-argmax that picks the "best" one is
choosing near-arbitrarily among ties. This module records, per evolve run,
how discriminating the val set actually was — primarily ``distinct_val_frac``
(distinct val levels / candidates explored): the lower it is, the more the
selection is a coin flip among indistinguishable candidates.

The full per-candidate distribution lives only on GEPA's ``detailed_results``
at the moment the run finishes; it was never persisted to ``gate_decision.json``
historically (only a scalar picked-val and a partial epsilon-band roster). So
this telemetry is forward-looking: the three evolve seams append a row as each
run completes, and ``gate_decision.json`` now carries ``val_aggregate_scores``
so the distribution survives in the run record too.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

LEDGER_NAME = "search_ledger.jsonl"


def resolve_ledger_root(output_dir: Path) -> Path:
    """The shared-ledger directory for a run whose artifacts live in output_dir.

    Walks up to the nearest ancestor named ``output`` so every run — skill
    (output/<name>/<ts>), tool (output/tools/<name>/<ts>), prompt-section
    (output/prompts/<name>/<ts>) — shares one ``output/search_ledger.jsonl``.
    A run dir with no ``output`` ancestor (tests on a tmp dir, a custom
    --output-dir) falls back to itself, so the ledger never escapes its tree.
    """
    output_dir = Path(output_dir).resolve()
    for ancestor in (output_dir, *output_dir.parents):
        if ancestor.name == "output":
            return ancestor
    return output_dir


@dataclass(frozen=True)
class SearchTelemetryRow:
    """One evolve run's val-discrimination summary."""

    artifact: str
    artifact_type: str  # "skill" | "tool" | "prompt_section"
    n_candidates: int
    n_distinct_val: int
    distinct_val_frac: float
    best_val: float
    median_val: float
    val_spread: float
    best_idx: int
    best_idx_frac: float
    decision: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_search_telemetry_row(
    *,
    artifact: str,
    artifact_type: str,
    val_scores: Any,
    best_idx: int,
    decision: Optional[str] = None,
) -> Optional[SearchTelemetryRow]:
    """Compute a telemetry row from a val-score list, or None if empty.

    Callers pass ``details.val_aggregate_scores`` and ``details.best_idx`` from
    a GEPA detailed_results. An empty (or None) list is non-actionable — the
    MIPROv2 fallback path — so we degrade to None rather than raise.
    """
    if not val_scores:
        return None
    val = [float(v) for v in val_scores]
    n = len(val)
    distinct = len({round(v, 4) for v in val})
    best_idx = int(best_idx)
    return SearchTelemetryRow(
        artifact=artifact,
        artifact_type=artifact_type,
        n_candidates=n,
        n_distinct_val=distinct,
        distinct_val_frac=distinct / n,
        best_val=max(val),
        median_val=float(statistics.median(val)),
        val_spread=max(val) - min(val),
        best_idx=best_idx,
        # 0.0 when a single candidate was explored (no fraction to take).
        best_idx_frac=best_idx / (n - 1) if n > 1 else 0.0,
        decision=decision,
    )


def append_search_telemetry(
    ledger_root: Path,
    *,
    artifact: str,
    artifact_type: str,
    val_scores: Any,
    best_idx: int,
    decision: Optional[str] = None,
) -> Optional[Path]:
    """Append one telemetry row to ``ledger_root/search_ledger.jsonl``.

    Returns the ledger path on write, or None when there's nothing to record
    (MIPROv2 fallback / empty candidate pool). Never raises into the evolve
    flow — telemetry must not break a run.
    """
    row = build_search_telemetry_row(
        artifact=artifact,
        artifact_type=artifact_type,
        val_scores=val_scores,
        best_idx=best_idx,
        decision=decision,
    )
    if row is None:
        return None
    ledger_root = Path(ledger_root)
    ledger_root.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_root / LEDGER_NAME
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.to_dict()) + "\n")
    return ledger_path


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL ledger into a list of row dicts (skips blank lines)."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize_ledger(path: Path) -> str:
    """Render per-artifact-type aggregates of the search ledger as text."""
    rows = read_ledger(path)
    if not rows:
        return f"No search telemetry recorded at {path}."

    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_type.setdefault(r.get("artifact_type", "?"), []).append(r)

    lines = [f"Search telemetry — {len(rows)} run(s) at {path}", ""]
    header = (
        f"{'artifact_type':<16}{'runs':>6}{'med candidates':>16}"
        f"{'med distinct-val%':>20}{'med val spread':>16}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for atype in sorted(by_type):
        group = by_type[atype]
        med_cands = statistics.median(r["n_candidates"] for r in group)
        med_frac = statistics.median(r["distinct_val_frac"] for r in group)
        med_spread = statistics.median(r["val_spread"] for r in group)
        lines.append(
            f"{atype:<16}{len(group):>6}{med_cands:>16.1f}"
            f"{med_frac * 100:>19.1f}%{med_spread:>16.3f}"
        )
    lines.append("")
    lines.append(
        "distinct-val% = distinct val levels / candidates explored; "
        "low values mean the val set can't discriminate (small-N ties) so "
        "selection among candidates is near-arbitrary."
    )
    return "\n".join(lines)


# gate_decision.json files predating this telemetry never stored the full
# val distribution (only a scalar picked-val and a partial epsilon-band
# roster), so the discrimination metric cannot be reconstructed for them.
def _count_legacy_runs(output_root: Path) -> int:
    return sum(1 for _ in Path(output_root).glob("**/gate_decision.json"))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize the GEPA val-discrimination search ledger."
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("output") / LEDGER_NAME,
        help="Path to search_ledger.jsonl (default: output/search_ledger.jsonl).",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Attempt to recover telemetry from past runs (infeasible — see note).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root scanned by --backfill for legacy gate_decision.json files.",
    )
    args = parser.parse_args(argv)

    if args.backfill:
        n_legacy = _count_legacy_runs(args.output_root)
        print(
            "Backfill is infeasible: val_aggregate_scores was never persisted "
            "in legacy runs, so the per-candidate val distribution cannot be "
            f"reconstructed ({n_legacy} run(s) skipped). Telemetry is "
            "forward-looking — the ledger fills as new evolve runs complete."
        )
        return 0

    print(summarize_ledger(args.ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

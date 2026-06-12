"""Saturation telemetry: per-run pre-flight band + scores, including aborts.

The saturation pre-flight (``evolution.core.saturation_check``) classifies each
run as healthy / weak_signal / no_headroom / uniform_failure against thresholds
that are uncalibrated magic numbers (0.99 / 0.95 / 0.15). Calibrating them needs
the pre-flight's own band and scores joined to the run's eventual outcome — but
that decision was never persisted. The archive only carries ``avg_baseline``
inside ``gate_decision.json`` for runs that *completed* GEPA, so runs the
pre-flight aborts are invisible, and the gated region (baseline ≥ 0.95) is
almost empty. The archive therefore cannot settle the thresholds.

This ledger closes that gap forward: every pre-flight invocation appends exactly
one row, written at pre-flight time so it is captured whether the run proceeds
or aborts (and regardless of any later failure). ``run_id`` joins the row back
to that run's ``gate_decision.json``, where the eventual deploy/reject decision
and bootstrap live — so the band↔outcome join is always reconstructable, and
the ledger never has to wait for (or miss) the end of a run. The ledger is the
corpus ``scripts/analysis/calibrate_saturation.py`` calibrates against once it
fills.

Mirrors ``evolution.core.search_telemetry`` and reuses its generic
``resolve_ledger_root`` / ``read_ledger`` helpers.
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from evolution.core.saturation_check import SaturationBand
from evolution.core.search_telemetry import read_ledger, resolve_ledger_root

logger = logging.getLogger(__name__)

__all__ = [
    "LEDGER_NAME",
    "SaturationTelemetryRow",
    "build_saturation_telemetry_row",
    "append_saturation_telemetry",
    "record_saturation_telemetry",
    "read_ledger",
    "resolve_ledger_root",
    "summarize_ledger",
]

LEDGER_NAME = "saturation_ledger.jsonl"

ArtifactType = Literal["skill", "tool", "prompt_section"]
AbortReason = Literal["non_interactive_deny", "user_decline"]


@dataclass(frozen=True)
class SaturationTelemetryRow:
    """One saturation pre-flight invocation: its band and scores.

    ``proceeded`` discriminates the two shapes: a proceeded run has
    ``abort_reason is None``; an aborted run carries the reason it stopped. The
    eventual deploy/reject decision is deliberately NOT stored here — it lives
    in the run's ``gate_decision.json`` and is recovered by joining on
    ``run_id`` (the run timestamp dir), which keeps this a single,
    write-at-pre-flight-time record rather than something that must survive to
    the end of the run.
    """

    run_id: str
    artifact: str
    artifact_type: ArtifactType
    band: SaturationBand
    holdout_score: float
    holdout_n: int
    proceeded: bool
    closed_loop_score: Optional[float] = None
    closed_loop_n: Optional[int] = None
    floor_score: Optional[float] = None
    floor_n: Optional[int] = None
    # Mean over tasks of the A/A minority-verdict fraction (0 = stable,
    # 0.5 = coin-flip) from a <suite>.noise.json sidecar — the per-task flip
    # rate a real closed-loop gain must clear. A rate, not a count.
    noise_mean_per_task_flip: Optional[float] = None
    # Set iff proceeded is False.
    abort_reason: Optional[AbortReason] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _noise_mean_per_task_flip(noise: Optional[dict]) -> Optional[float]:
    """Pull ``mean_per_task_flip`` from a <suite>.noise.json payload.

    It is the mean over tasks of the per-task minority-verdict fraction (a rate
    in [0, 0.5]; ``evolution/validation/noise_calibration.py``), i.e. the A/A
    flip rate a real closed-loop gain must clear. Absent or malformed noise
    degrades to None, never raises.
    """
    if not isinstance(noise, dict):
        return None
    value = noise.get("mean_per_task_flip")
    return float(value) if isinstance(value, (int, float)) else None


def build_saturation_telemetry_row(
    sat_report: Any,
    *,
    run_id: str,
    artifact: str,
    artifact_type: ArtifactType,
    proceeded: bool,
    abort_reason: Optional[AbortReason] = None,
) -> SaturationTelemetryRow:
    """Build a telemetry row from a ``SaturationReport`` plus run context."""
    return SaturationTelemetryRow(
        run_id=run_id,
        artifact=artifact,
        artifact_type=artifact_type,
        band=sat_report.band,
        holdout_score=float(sat_report.holdout_score),
        holdout_n=int(sat_report.holdout_n),
        proceeded=proceeded,
        closed_loop_score=(
            float(sat_report.closed_loop_score)
            if sat_report.closed_loop_score is not None
            else None
        ),
        closed_loop_n=(
            int(sat_report.closed_loop_n)
            if sat_report.closed_loop_n is not None
            else None
        ),
        floor_score=(
            float(sat_report.floor_score)
            if sat_report.floor_score is not None
            else None
        ),
        floor_n=int(sat_report.floor_n) if sat_report.floor_n is not None else None,
        noise_mean_per_task_flip=_noise_mean_per_task_flip(sat_report.noise),
        abort_reason=abort_reason,
    )


def record_saturation_telemetry(
    output_dir: Path,
    sat_report: Any,
    *,
    artifact: str,
    artifact_type: ArtifactType,
    proceeded: bool,
    abort_reason: Optional[AbortReason] = None,
) -> Optional[Path]:
    """Build + append a row for one run in a single, never-raising call.

    ``run_id`` is ``output_dir.name`` (the run timestamp) so the row joins back
    to that run's ``gate_decision.json``; the ledger is shared at the nearest
    ``output/`` ancestor. The whole call is best-effort — a malformed report or
    an unwritable ledger logs at DEBUG and degrades to None rather than
    disturbing the run (telemetry must never break or block one, and on the
    abort path the result is discarded regardless).
    """
    try:
        row = build_saturation_telemetry_row(
            sat_report,
            run_id=Path(output_dir).name,
            artifact=artifact,
            artifact_type=artifact_type,
            proceeded=proceeded,
            abort_reason=abort_reason,
        )
        return append_saturation_telemetry(resolve_ledger_root(output_dir), row=row)
    except Exception:
        logger.debug("saturation telemetry record failed", exc_info=True)
        return None


def append_saturation_telemetry(
    ledger_root: Path,
    *,
    row: SaturationTelemetryRow,
) -> Optional[Path]:
    """Append one row to ``ledger_root/saturation_ledger.jsonl``.

    Returns the ledger path, or None if the write fails (logged at DEBUG so a
    silently-empty ledger is diagnosable). Never raises — telemetry must not
    break the evolve flow.
    """
    try:
        ledger_root = Path(ledger_root)
        ledger_root.mkdir(parents=True, exist_ok=True)
        ledger_path = ledger_root / LEDGER_NAME
        with ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row.to_dict()) + "\n")
        return ledger_path
    except Exception:
        logger.debug("saturation telemetry write failed", exc_info=True)
        return None


def summarize_ledger(path: Path) -> str:
    """Render per-band aggregates of the saturation ledger as text."""
    rows = read_ledger(path)
    if not rows:
        return f"No saturation telemetry recorded at {path}."

    by_band: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_band.setdefault(r.get("band", "?"), []).append(r)

    n_aborted = sum(1 for r in rows if not r.get("proceeded", True))
    lines = [
        f"Saturation telemetry — {len(rows)} run(s), {n_aborted} aborted, at {path}",
        "",
    ]
    header = (
        f"{'band':<16}{'runs':>6}{'aborted':>9}"
        f"{'med holdout':>13}{'med closed-loop':>17}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for band in sorted(by_band):
        group = by_band[band]
        aborted = sum(1 for r in group if not r.get("proceeded", True))
        med_holdout = statistics.median(
            r["holdout_score"] for r in group if r.get("holdout_score") is not None
        )
        cl_scores = [
            r["closed_loop_score"]
            for r in group
            if r.get("closed_loop_score") is not None
        ]
        med_cl = f"{statistics.median(cl_scores):.3f}" if cl_scores else "—"
        lines.append(
            f"{band:<16}{len(group):>6}{aborted:>9}"
            f"{med_holdout:>13.3f}{med_cl:>17}"
        )
    lines.append("")
    lines.append(
        "Aborts are the rows the gate_decision archive never captured; once the "
        "gated region (holdout ≥ 0.95) accrues runs whose gate_decision.json "
        "records an outcome, scripts/analysis/calibrate_saturation.py can settle "
        "the thresholds."
    )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize the saturation pre-flight telemetry ledger."
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("output") / LEDGER_NAME,
        help="Path to saturation_ledger.jsonl (default: output/saturation_ledger.jsonl).",
    )
    args = parser.parse_args(argv)
    print(summarize_ledger(args.ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

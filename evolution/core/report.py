"""A/B reporting in the style Hermes already holds itself to.

``hermes-agent/evals/readtool/results/SUMMARY.md`` is the most rigorous
evaluation writing in the Hermes tree. It reports same-prompt arms on both
sides, multiple repetitions, deltas stated against run-to-run noise, an
explicit SHIP or HOLD verdict, and — the part that matters most — recorded
caveats where two arms are *not* comparable.

Evolution previously reported a single-run delta on a metric the optimizer
never saw. This module holds it to the house standard instead: a delta smaller
than the noise band is reported as noise, not as an improvement, and a verdict
is never issued without saying what it rests on.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

# A delta must clear this multiple of the pooled noise band to count as real.
# 1.0 means "larger than the spread we see between identical repetitions".
NOISE_MULTIPLE = 1.0

# With only one rep per arm there is no measurable noise. Rather than declaring
# everything significant, assume this much relative spread — deliberately
# conservative, and the report says so.
ASSUMED_SINGLE_REP_NOISE = 0.05


@dataclass
class ArmStats:
    """One side of an A/B, aggregated over repetitions."""

    label: str
    scores: list[float] = field(default_factory=list)
    tokens: list[float] = field(default_factory=list)
    tool_calls: list[float] = field(default_factory=list)
    wall_s: list[float] = field(default_factory=list)
    size_chars: int = 0
    errors: int = 0

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.scores) if self.scores else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0

    def mean_of(self, values: Sequence[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            "mean_score": round(self.mean, 4),
            "stdev": round(self.stdev, 4),
            "size_chars": self.size_chars,
            "avg_tokens": round(self.mean_of(self.tokens), 1),
            "avg_tool_calls": round(self.mean_of(self.tool_calls), 2),
            "avg_wall_s": round(self.mean_of(self.wall_s), 2),
            "errors": self.errors,
        }


@dataclass
class ABReport:
    """Baseline vs evolved, with a verdict that states its own basis."""

    subject: str
    baseline: ArmStats
    evolved: ArmStats
    metric_name: str = "judge composite"
    constraints_passed: bool = True
    constraint_failures: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    # ── statistics ───────────────────────────────────────────────────────

    @property
    def delta(self) -> float:
        return self.evolved.mean - self.baseline.mean

    @property
    def noise_band(self) -> float:
        """Pooled run-to-run spread the delta has to clear to mean anything."""
        stdevs = [s for s in (self.baseline.stdev, self.evolved.stdev) if s > 0]
        if stdevs:
            return statistics.fmean(stdevs)
        # No repetitions to measure noise from — assume a conservative band and
        # flag it, rather than treating any delta as significant.
        return max(ASSUMED_SINGLE_REP_NOISE * max(self.baseline.mean, 0.01), 1e-6)

    @property
    def noise_is_measured(self) -> bool:
        """Whether the band came from real repetitions or was assumed."""
        return self.baseline.stdev > 0 or self.evolved.stdev > 0

    @property
    def _band_basis(self) -> str:
        return "measured" if self.noise_is_measured else "n=1 per arm, assumed"

    @property
    def within_noise(self) -> bool:
        return abs(self.delta) < self.noise_band * NOISE_MULTIPLE

    @property
    def size_delta(self) -> int:
        return self.evolved.size_chars - self.baseline.size_chars

    @property
    def relative_delta(self) -> float:
        base = self.baseline.mean
        return self.delta / base if base > 0.001 else 0.0

    # ── verdict ──────────────────────────────────────────────────────────

    def verdict(self) -> tuple[str, str]:
        """Return (verdict, one-line reason).

        SHIP requires: constraints pass, and an improvement larger than the
        noise band. Everything else is HOLD — including a positive delta that
        is not distinguishable from noise, which is the case the old
        single-run report reported as a win.
        """
        if not self.constraints_passed:
            failed = ", ".join(self.constraint_failures) or "unspecified"
            return "HOLD", f"constraint failure: {failed}"

        if self.baseline.n == 0 or self.evolved.n == 0:
            return "HOLD", "no evaluation data on one or both arms"

        if self.within_noise:
            return (
                "HOLD",
                f"delta {self.delta:+.3f} is inside the ±{self.noise_band:.3f} "
                f"noise band ({self._band_basis})",
            )

        if self.delta < 0:
            return "HOLD", f"evolved arm is worse by {abs(self.delta):.3f}"

        # A SHIP that rests on one observation per arm has to say so: the band
        # it cleared was assumed, not measured, so the verdict is weaker than
        # the same number backed by repetitions.
        return (
            "SHIP",
            f"{self.delta:+.3f} ({self.relative_delta:+.1%}) beyond the "
            f"±{self.noise_band:.3f} noise band ({self._band_basis})",
        )

    def auto_caveats(self) -> list[str]:
        """Caveats the data implies, so they are never left off by accident."""
        found = list(self.caveats)

        if self.baseline.n <= 1 or self.evolved.n <= 1:
            found.append(
                "Single repetition per arm — the noise band is assumed, not "
                "measured. Re-run with --agent-eval-reps 3 before trusting a "
                "small delta."
            )
        if self.baseline.n != self.evolved.n:
            found.append(
                f"Arms are unbalanced ({self.baseline.n} vs {self.evolved.n} "
                "observations); means are not directly comparable."
            )
        if self.evolved.errors or self.baseline.errors:
            found.append(
                f"Errored runs excluded from means "
                f"(baseline {self.baseline.errors}, evolved {self.evolved.errors}) — "
                "a variant that errors more can look better on the survivors."
            )
        if self.size_delta > 0 and not self.within_noise and self.delta > 0:
            found.append(
                f"The evolved artifact is {self.size_delta:+,} chars larger; "
                "part of any quality gain may be paid for in context budget."
            )
        return found

    # ── rendering ────────────────────────────────────────────────────────

    def to_markdown(self) -> str:
        verdict, reason = self.verdict()
        caveats = self.auto_caveats()

        lines = [
            f"# {self.subject} — evolution A/B",
            "",
            f"_{self.created_at} · metric: {self.metric_name}_",
            "",
            "| metric | baseline | evolved | delta |",
            "|---|---:|---:|---:|",
            _row("score", self.baseline.mean, self.evolved.mean, fmt="{:.3f}"),
            _row(
                "size (chars)",
                self.baseline.size_chars,
                self.evolved.size_chars,
                fmt="{:,.0f}",
            ),
        ]

        if self.baseline.tokens or self.evolved.tokens:
            lines.append(
                _row(
                    "tokens",
                    self.baseline.mean_of(self.baseline.tokens),
                    self.evolved.mean_of(self.evolved.tokens),
                    fmt="{:,.0f}",
                )
            )
        if self.baseline.tool_calls or self.evolved.tool_calls:
            lines.append(
                _row(
                    "tool calls",
                    self.baseline.mean_of(self.baseline.tool_calls),
                    self.evolved.mean_of(self.evolved.tool_calls),
                    fmt="{:.2f}",
                )
            )
        if self.baseline.wall_s or self.evolved.wall_s:
            lines.append(
                _row(
                    "wall (s)",
                    self.baseline.mean_of(self.baseline.wall_s),
                    self.evolved.mean_of(self.evolved.wall_s),
                    fmt="{:.1f}",
                )
            )

        lines += [
            "",
            f"Observations: {self.baseline.n} baseline / {self.evolved.n} evolved. "
            f"Noise band ±{self.noise_band:.3f}.",
            "",
            f"**Verdict: {verdict}.** {reason}",
        ]

        if self.constraint_failures:
            lines += ["", "**Constraint failures:**"]
            lines += [f"- {f}" for f in self.constraint_failures]

        if caveats:
            lines += ["", "**Caveats recorded:**"]
            lines += [f"- {c}" for c in caveats]

        if self.extra:
            lines += ["", "**Context:**"]
            for key, value in self.extra.items():
                lines.append(f"- {key}: {value}")

        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        verdict, reason = self.verdict()
        return {
            "subject": self.subject,
            "created_at": self.created_at,
            "metric": self.metric_name,
            "baseline": self.baseline.as_dict(),
            "evolved": self.evolved.as_dict(),
            "delta": round(self.delta, 4),
            "relative_delta": round(self.relative_delta, 4),
            "noise_band": round(self.noise_band, 4),
            "within_noise": self.within_noise,
            "size_delta": self.size_delta,
            "verdict": verdict,
            "reason": reason,
            "constraints_passed": self.constraints_passed,
            "constraint_failures": self.constraint_failures,
            "caveats": self.auto_caveats(),
            "extra": self.extra,
        }

    def write(self, output_dir: Path) -> tuple[Path, Path]:
        """Write SUMMARY.md and report.json; return both paths."""
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / "SUMMARY.md"
        json_path = output_dir / "report.json"
        md_path.write_text(self.to_markdown())
        json_path.write_text(json.dumps(self.to_dict(), indent=2))
        return md_path, json_path


def _row(name: str, baseline: float, evolved: float, fmt: str) -> str:
    delta = evolved - baseline
    return (
        f"| {name} | {fmt.format(baseline)} | {fmt.format(evolved)} | "
        f"{'+' if delta >= 0 else ''}{fmt.format(delta)} |"
    )


def arm_from_scores(
    label: str,
    scores: Sequence[float],
    size_chars: int = 0,
) -> ArmStats:
    """Build an arm from bare quality scores (completion-level evaluation)."""
    return ArmStats(label=label, scores=list(scores), size_chars=size_chars)


def arm_from_eval_run(label: str, run, size_chars: int = 0) -> ArmStats:
    """Build an arm from an :class:`~evolution.core.agent_runner.EvalRun`."""
    usable = [r for r in run.results if r.ok]
    return ArmStats(
        label=label,
        scores=[r.score for r in usable],
        tokens=[float(r.total_tokens) for r in usable],
        tool_calls=[float(r.tool_calls) for r in usable],
        wall_s=[r.wall_s for r in usable],
        size_chars=size_chars,
        errors=run.errors,
    )

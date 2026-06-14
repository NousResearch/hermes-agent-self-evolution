"""Cluster-honest reporting for the code-evolution measurement campaign.

Promotes the F1 re-analysis: the honest unit is the ORGANISM, not the pooled
per-seed attempt. Seeds within an organism are heavily correlated (the GREEN's
ICC was 0.57), so a pooled rate overstates precision. This module reports
organism-level estimands with a cluster bootstrap, the ICC/design-effect that
justifies spending budget on more organisms rather than more seeds, and the
pooled rate only as a labeled contrast.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

_Z = 1.96
_BOOTSTRAP = 20000
_BOOTSTRAP_SEED = 20260614


def wilson(k: int, n: int, z: float = _Z) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def wilson_lower(k: int, n: int) -> float:
    return wilson(k, n)[0]


@dataclass
class OrganismResult:
    """One organism's outcome across seeds. ``seeds`` is the per-seed correctness
    (True iff that seed's repair passed the oracle gate)."""

    tool: str
    fix_sha: str
    seeds: list[bool]

    @property
    def n_correct(self) -> int:
        return sum(1 for s in self.seeds if s)

    @property
    def deploy_reachable(self) -> bool:
        """A correct repair is reachable for this organism: a majority of seeds
        pass the oracle gate (≥⌈n/2⌉, so ≥2 of 3)."""
        return self.n_correct * 2 >= len(self.seeds) and self.n_correct > 0


def _icc_deff(organisms: list[OrganismResult]) -> tuple[float, float, float]:
    """One-way random-effects ICC of the per-seed binary correctness, plus the
    design-effect and effective N. Quantifies how little extra seeds buy."""
    rows = [o.seeds for o in organisms if o.seeds]
    n = len(rows)
    k = len(rows[0]) if rows else 0
    if n < 2 or k < 2 or any(len(r) != k for r in rows):
        return (float("nan"), float("nan"), float(n * k))
    N = n * k
    grand = sum(sum(1 for s in r if s) for r in rows) / N
    means = [sum(1 for s in r if s) / k for r in rows]
    ms_between = k * sum((m - grand) ** 2 for m in means) / (n - 1)
    ms_within = sum(sum((int(s) - means[i]) ** 2 for s in rows[i]) for i in range(n)) / (N - n)
    denom = ms_between + (k - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom > 0 else 0.0
    deff = 1 + (k - 1) * icc
    return (icc, deff, N / deff if deff > 0 else float(N))


def _cluster_bootstrap(flags: list[int], kill_line: float = 0.10) -> dict:
    """Resample organisms with replacement; CI of the fraction + P(< kill_line)."""
    n = len(flags)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 1.0, "p_below_kill": 1.0}
    rng = random.Random(_BOOTSTRAP_SEED)
    boot = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(_BOOTSTRAP))
    return {
        "mean": sum(boot) / _BOOTSTRAP,
        "ci_low": boot[int(0.025 * _BOOTSTRAP)],
        "ci_high": boot[int(0.975 * _BOOTSTRAP)],
        "p_below_kill": sum(1 for b in boot if b < kill_line) / _BOOTSTRAP,
    }


def build_report(organisms: list[OrganismResult], *, kill_line: float = 0.10) -> dict:
    """The campaign's cluster-honest report. Headline estimand is organism-level
    deploy-reachability (a majority of seeds produce an oracle-correct repair)."""
    n = len(organisms)
    n_deploy = sum(1 for o in organisms if o.deploy_reachable)
    dr_flags = [1 if o.deploy_reachable else 0 for o in organisms]
    pooled_correct = sum(o.n_correct for o in organisms)
    pooled_total = sum(len(o.seeds) for o in organisms)
    icc, deff, n_eff = _icc_deff(organisms)
    return {
        "n_organisms": n,
        "deploy_reachable": {
            "k": n_deploy, "n": n, "fraction": (n_deploy / n if n else 0.0),
            "wilson": wilson(n_deploy, n),
            "cluster_bootstrap": _cluster_bootstrap(dr_flags, kill_line),
        },
        "icc": icc, "design_effect": deff, "effective_n": n_eff,
        # The pooled per-seed rate is pseudo-replicated (seeds correlate within an
        # organism); kept only for contrast, never as the headline.
        "pooled_per_seed_rate_FOR_CONTRAST": {
            "k": pooled_correct, "n": pooled_total,
            "rate": (pooled_correct / pooled_total if pooled_total else 0.0),
            "wilson_DISHONEST": wilson(pooled_correct, pooled_total),
        },
        "kill_line": kill_line,
        "verdict": ("GREEN" if wilson_lower(n_deploy, n) > kill_line
                    else "BELOW_KILL_LINE"),
    }

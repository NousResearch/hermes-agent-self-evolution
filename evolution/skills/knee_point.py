"""Knee-point Pareto selection from a GEPA candidate front.

GEPA's default is "pick the candidate with the best aggregate valset score."
On small valsets that pick overfits aggressively — high val score paired with
a meaningfully worse holdout. Knee-point selection scans the band of candidates
within ε of the best valset and walks them in `strategy` order (default
val-best; smallest body available for compression-chasing).

References:
- Branke et al. 2004, "Finding Knees in Multi-objective Optimization"
- Breiman et al. 1984, "1-SE rule" — family of resolution-aware ε rules
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol


class _SupportsSkillText(Protocol):
    @property
    def skill_text(self) -> str: ...


@dataclass(frozen=True)
class CandidatePick:
    """A selected candidate plus the diagnostics needed to debug the choice.

    band_roster is recorded so post-hoc calibration can ask: "did knee-point
    consistently pick rank-3 candidates that regressed?" That signal tunes ε.
    """
    module: Any                   # the picked candidate module (SkillModule)
    skill_text: str               # extracted from module.skill_text
    body_chars: int               # parsimony metric
    val_score: float              # picked candidate's val aggregate
    val_rank_in_band: int         # 1-indexed; 1 = highest val in band
    band_size: int                # candidates within ε of best
    epsilon: float                # ε used for the band
    fallback: str                 # "knee" | "static_failed_all"
    picked_idx: int               # index into the original candidates list
    gepa_default_idx: int         # for comparison telemetry
    gepa_default_body_chars: int  # for comparison telemetry
    band_roster: list[dict]       # [{"idx", "val_score", "body_chars"}, ...]


_KNEE_POINT_STRATEGIES = ("val-best", "smallest")


def select_knee_point(
    candidates: list[_SupportsSkillText],
    val_aggregate_scores: list[float],
    n_val: int,
    static_validator: Callable[[str], list],
    gepa_default_idx: int,
    epsilon: Optional[float] = None,
    strategy: str = "val-best",
    text_extractor: Optional[Callable[[Any], str]] = None,
) -> CandidatePick:
    """Pick a candidate within ε of best valset score.

    Args:
        candidates: GEPA-built modules (DspyGEPAResult.candidates). Each
            must expose a `skill_text` property unless `text_extractor` is
            provided.
        val_aggregate_scores: one float per candidate (DspyGEPAResult
            .val_aggregate_scores).
        n_val: size of the valset that produced those scores. Used for the
            default ε = 1 / n_val ("one example's worth of disagreement" —
            honest about valset resolution rather than pretending we have
            ε=0.02 precision on N=6).
        static_validator: callable(skill_text) -> list[ConstraintResult].
            Iterating the band in the strategy's order stops at the first
            candidate whose every result has `.passed`. Don't fall back to
            GEPA default until *every* band candidate fails.
        gepa_default_idx: DspyGEPAResult.best_idx — the candidate GEPA
            would have picked. Used as the last-resort fallback if the
            entire band fails static, and recorded in telemetry regardless.
        epsilon: override the 1/n_val default. CLI flag passes this through.
        strategy: which order to walk the band when picking.
            "val-best" (default): highest val score first, smallest body as
                tiebreak. Picks the candidate most likely to generalize.
            "smallest": ascending body_chars. Greedy parsimony — picks the
                smallest body in the band regardless of val cost. Available
                via --knee-point-strategy for users explicitly chasing
                compression even at val cost.
        text_extractor: optional ``candidate -> str`` callable used to read
            the text that drives both the parsimony metric (``body_chars``)
            and the ``static_validator`` input. Defaults to reading
            ``candidate.skill_text``. Pass a custom extractor when the
            parsimony-relevant text is a sub-field of the candidate (for
            example, the description portion of a rendered tool manifest)
            rather than the full ``skill_text`` artifact.

    Returns:
        CandidatePick with the chosen module + diagnostics. fallback field
        is "knee" if a band candidate passed static (the normal path) or
        "static_failed_all" if we fell back to gepa_default_idx.
    """
    if not candidates:
        raise ValueError("select_knee_point: candidates list is empty")
    if len(candidates) != len(val_aggregate_scores):
        raise ValueError(
            f"select_knee_point: candidates ({len(candidates)}) and "
            f"val_aggregate_scores ({len(val_aggregate_scores)}) length mismatch"
        )
    if n_val < 1:
        raise ValueError(f"select_knee_point: n_val must be >= 1, got {n_val}")
    if strategy not in _KNEE_POINT_STRATEGIES:
        raise ValueError(
            f"select_knee_point: strategy must be one of {_KNEE_POINT_STRATEGIES}, "
            f"got {strategy!r}"
        )

    eps = float(epsilon) if epsilon is not None else 1.0 / n_val

    extractor: Callable[[Any], str] = (
        text_extractor if text_extractor is not None else (lambda c: c.skill_text)
    )

    best_score = max(val_aggregate_scores)
    band_threshold = best_score - eps

    band_indices = [
        i for i, s in enumerate(val_aggregate_scores) if s >= band_threshold
    ]

    band_roster_by_score = sorted(
        band_indices,
        key=lambda i: (-val_aggregate_scores[i], i),
    )
    rank_lookup = {
        idx: rank + 1
        for rank, idx in enumerate(band_roster_by_score)
    }

    body_chars_by_idx = {i: len(extractor(candidates[i])) for i in band_indices}

    band_roster = [
        {
            "idx": i,
            "val_score": float(val_aggregate_scores[i]),
            "body_chars": body_chars_by_idx[i],
        }
        for i in band_roster_by_score
    ]

    if strategy == "smallest":
        # Greedy parsimony: smallest body wins; val score is just a tiebreak.
        sorted_for_pick = sorted(
            band_indices,
            key=lambda i: (
                body_chars_by_idx[i],
                -val_aggregate_scores[i],
                i,
            ),
        )
    else:  # "val-best"
        # Highest val wins; smallest body is the tiebreak. This is the
        # default — within the ε-band the algorithm prefers the candidate
        # most likely to generalize, with parsimony only mattering on ties.
        sorted_for_pick = sorted(
            band_indices,
            key=lambda i: (
                -val_aggregate_scores[i],
                body_chars_by_idx[i],
                i,
            ),
        )

    gepa_default_chars = len(extractor(candidates[gepa_default_idx]))

    for picked_idx in sorted_for_pick:
        text = extractor(candidates[picked_idx])
        results = static_validator(text)
        if all(r.passed for r in results):
            return CandidatePick(
                module=candidates[picked_idx],
                skill_text=text,
                body_chars=body_chars_by_idx[picked_idx],
                val_score=float(val_aggregate_scores[picked_idx]),
                val_rank_in_band=rank_lookup[picked_idx],
                band_size=len(band_indices),
                epsilon=eps,
                fallback="knee",
                picked_idx=picked_idx,
                gepa_default_idx=gepa_default_idx,
                gepa_default_body_chars=gepa_default_chars,
                band_roster=band_roster,
            )

    default_text = extractor(candidates[gepa_default_idx])
    return CandidatePick(
        module=candidates[gepa_default_idx],
        skill_text=default_text,
        body_chars=gepa_default_chars,
        val_score=float(val_aggregate_scores[gepa_default_idx]),
        val_rank_in_band=rank_lookup.get(gepa_default_idx, -1),
        band_size=len(band_indices),
        epsilon=eps,
        fallback="static_failed_all",
        picked_idx=gepa_default_idx,
        gepa_default_idx=gepa_default_idx,
        gepa_default_body_chars=gepa_default_chars,
        band_roster=band_roster,
    )

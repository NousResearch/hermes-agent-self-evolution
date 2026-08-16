"""The admission rule: when is an evolved artifact actually better?

Comparing two mean scores and deploying whenever the second is larger treats
every difference as real. On the eval sets this pipeline can afford, most are
not. A six-example holdout is the case that makes this concrete: the best
possible one-sided paired result short of a clean sweep, four losses out of
four, lands at p = 0.0625, and five aligned losses are needed to reach 0.05 at
p = 0.03125. Nothing weaker than a near-unanimous flip can clear the bar, so a
rule that admits on ``mean_after > mean_before`` is not measuring evidence, it
is measuring noise with a sign.

So admission is declared, and it is paired throughout:

  1. The deployable artifact actually changed, where the caller checks that.
  2. The paired test finds a significant improvement at *alpha*.
  3. The paired test does not find a significant regression at *alpha*.

Point 1 is optional because not every caller can answer it. ``material_diff``
defaults to ``None``, meaning the question was not asked, and the verdict then
records ``None`` rather than claiming an unchecked ``True``. A caller that does
compare the artifacts passes the answer and gets it enforced.

**Point 2 replaces "the mean went up".** That is the whole change. A positive
delta with p = 0.5 is not an improvement, and this module will not call it one.

**Power is reported, never assumed.** :func:`min_detectable_paired_shift` gives
the smallest shift the sample size could possibly have called significant. It
does not gate: a result that reached significance did so on evidence, and
refusing it because some *smaller* effect would have been invisible would be
backwards. What the power number does is qualify the non-regression claim in
point 3. On six examples nothing under an 83.3% collapse could ever have been
detected, so "no significant regression" there means "no regression large
enough to be visible", which is a much weaker statement than it reads as, and
the verdict says so rather than leaving the operator to infer it.

**Binary when there is ground truth, continuous otherwise.** An objective
verifier knows whether each answer was actually right, so the outcomes are
binary and McNemar's exact test applies: it conditions on the examples the two
versions disagreed about, which are the only ones carrying information about
which is better. Without ground truth there is no defensible threshold to
binarize a judge score at, so the continuous path uses the signed-rank test and
a paired bootstrap instead. Both paths come from :mod:`evolution.core.stats`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evolution.core.stats import (
    PairedBinary,
    PairedContinuous,
    compare_paired_binary,
    compare_paired_continuous,
    mcnemar_exact,
    min_detectable_paired_shift,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_TOLERANCE",
    "AdmissionVerdict",
    "evaluate_admission",
]

#: Significance level for both the improvement and the regression test.
DEFAULT_ALPHA = 0.05

#: The regression size the operator wants the gate to be able to catch. Used
#: only to decide whether the holdout was ever large enough to catch it, which
#: is reported alongside the verdict rather than silently assumed.
DEFAULT_TOLERANCE = 0.05


@dataclass(frozen=True)
class AdmissionVerdict:
    """Whether a candidate may be deployed, and the evidence for that."""

    admitted: bool
    reason: str
    n: int
    delta: float
    p_improvement: float
    p_regression: float
    significant_improvement: bool
    significant_regression: bool
    min_detectable_shift: float
    underpowered: bool
    #: True/False when the caller compared the artifacts, None when it did not
    #: ask. None never blocks admission, and is reported as "not assessed"
    #: rather than as a passing check.
    material_diff: bool | None
    alpha: float = DEFAULT_ALPHA
    tolerance: float = DEFAULT_TOLERANCE
    test: str = ""
    binary: PairedBinary | None = None
    continuous: PairedContinuous | None = None

    @property
    def power_note(self) -> str:
        """How much of a shift this holdout could ever have resolved."""
        if not self.underpowered:
            return (
                f"{self.n} examples can resolve a shift of "
                f"{self.min_detectable_shift:.1%}, within the {self.tolerance:.1%} tolerance."
            )
        return (
            f"Underpowered: {self.n} examples cannot resolve anything smaller "
            f"than {self.min_detectable_shift:.1%}, so a {self.tolerance:.1%} "
            "tolerance is not being enforced by evidence."
        )

    def describe(self) -> str:
        """One line carrying the verdict, the test used, and the evidence."""
        comparison = self.binary or self.continuous
        detail = comparison.describe() if comparison is not None else f"n={self.n}"
        icon = "✓" if self.admitted else "✗"
        return f"{icon} {self.reason} [{self.test}] {detail}"

    def to_dict(self) -> dict:
        """Serialise the verdict for a run's metrics file."""
        payload = {
            "admitted": self.admitted,
            "reason": self.reason,
            "test": self.test,
            "n": self.n,
            "delta": round(self.delta, 6),
            "alpha": self.alpha,
            "tolerance": self.tolerance,
            "p_improvement": round(self.p_improvement, 6),
            "p_regression": round(self.p_regression, 6),
            "significant_improvement": self.significant_improvement,
            "significant_regression": self.significant_regression,
            "min_detectable_shift": round(self.min_detectable_shift, 6),
            "underpowered": self.underpowered,
            "material_diff": self.material_diff,
        }
        if self.binary is not None:
            payload["paired_binary"] = self.binary.to_dict()
        if self.continuous is not None:
            payload["paired_continuous"] = self.continuous.to_dict()
        return payload


def evaluate_admission(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    *,
    material_diff: bool | None = None,
    baseline_correct: Sequence[bool] | None = None,
    candidate_correct: Sequence[bool] | None = None,
    alpha: float = DEFAULT_ALPHA,
    tolerance: float = DEFAULT_TOLERANCE,
) -> AdmissionVerdict:
    """Decide whether a candidate is admissible on paired holdout results.

    ``baseline_scores`` and ``candidate_scores`` must be aligned: index i is
    the same holdout example in both, or the pairing is meaningless.

    Pass ``baseline_correct``/``candidate_correct`` when objective ground truth
    is available. The verdict is then decided by McNemar's exact test on those
    outcomes, and the continuous comparison is still computed and attached for
    the report. Without them the signed-rank test on the scores decides.
    """
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError(
            f"paired admission needs equal lengths, got "
            f"{len(baseline_scores)} and {len(candidate_scores)}"
        )

    n = len(baseline_scores)
    continuous = compare_paired_continuous(
        baseline_scores, candidate_scores, alpha=alpha
    )

    binary: PairedBinary | None = None
    if baseline_correct is not None and candidate_correct is not None:
        binary = compare_paired_binary(
            baseline_correct, candidate_correct, alpha=alpha
        )

    if binary is not None:
        test = "McNemar exact (paired binary)"
        delta = binary.delta
        improved = binary.significant_improvement
        regressed = binary.significant_regression
        # One-sided in each direction: the mirror of ``p_worse``.
        p_improvement = mcnemar_exact(
            binary.baseline_only, binary.candidate_only, "better"
        )
        p_regression = binary.p_worse
    else:
        test = "Wilcoxon signed-rank (paired continuous)"
        delta = continuous.delta
        improved = continuous.significant_improvement
        regressed = continuous.significant_regression
        # The signed-rank p is two-sided; direction comes from the properties
        # above, so the same p describes both claims.
        p_improvement = continuous.wilcoxon_p
        p_regression = continuous.wilcoxon_p

    shift = min_detectable_paired_shift(n, alpha) if n > 0 else 1.0
    underpowered = shift > abs(tolerance)

    if n == 0:
        admitted, reason = False, "No holdout examples, so nothing was measured"
    elif material_diff is False:
        admitted, reason = False, "The deployable artifact did not change"
    elif regressed:
        admitted, reason = False, "The candidate significantly regressed"
    elif not improved:
        admitted, reason = False, (
            f"No significant improvement (delta {delta:+.3f}, p={p_improvement:.3f} "
            f"at alpha {alpha})"
        )
    else:
        admitted, reason = True, (
            f"Significant improvement (delta {delta:+.3f}, p={p_improvement:.3f} "
            f"at alpha {alpha})"
        )

    return AdmissionVerdict(
        admitted=admitted,
        reason=reason,
        n=n,
        delta=delta,
        p_improvement=p_improvement,
        p_regression=p_regression,
        significant_improvement=improved,
        significant_regression=regressed,
        min_detectable_shift=shift,
        underpowered=underpowered,
        material_diff=material_diff,
        alpha=alpha,
        tolerance=tolerance,
        test=test,
        binary=binary,
        continuous=continuous,
    )

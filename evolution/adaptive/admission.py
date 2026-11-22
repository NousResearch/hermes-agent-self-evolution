"""Candidate-vs-baseline admission gate (KAFO-EVO-001 gate 6).

A proposed candidate (for example a GEPA-evolved skill turned into a
``Strategy``) is admitted only if, on the SAME validation cases, with the
SAME executor factory and seeds as the current baseline, it:

1. has no critical violations (``ExecutionOutcome.violation``: disallowed tool,
   guardrail block/halt, cost budget exceeded);
2. beats the baseline's mean utility by at least ``min_margin``; and
3. has a paired bootstrap confidence interval for that difference whose lower
   bound is above 0.

Otherwise it is rejected with every failed rule listed and logged.

Fixed configuration
-------------------
Thresholds, seeds, split and cost/latency references live in the frozen
``ADMISSION_POLICY`` below. ``admit`` takes no threshold arguments, and every
decision records the policy and its SHA-256 (``policy_sha256``), which a test
pins, so changing a threshold needs a reviewed code change.

Data and scoring
----------------
Cases come only from ``evaluation.open_cases`` (gate 2 ``load_benchmark``) for
the validation split, inside ``benchmark.optimisation_phase()``, so the sealed
test split is unreachable. Quality is the gate 3 verifier's per-case F1 of the
findings parsed from the executor output; utility is ``core.score_outcome``.
The agent supplies neither: the caller passes only the candidate, the baseline
and an executor factory.

The gate 2 v1 benchmark is hand-written synthetic data, so every decision is
labelled ``synthetic=True``. An admission here is not evidence that Hermes
improves on real code review.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import statistics
from typing import Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

from . import checkpoint as ck, evaluation, registry as reg, verifier
from .benchmark import DEFAULT_VERSION, DEFECT_CATEGORIES, OPEN_SPLITS, optimisation_phase
from .compare import bootstrap_ci
from .core import Reward, score_outcome
from .executor import ExecutionOutcome, Executor, Strategy, Task

logger = logging.getLogger(__name__)

SYNTHETIC_NOTICE = ("SYNTHETIC: gate 2 v1 benchmark is hand-written synthetic data; "
                    "an admission is not evidence of real Hermes improvement.")

REVIEW_PROMPT = (
    "Review the Python code below for defects. Reply with only a JSON list of findings; "
    "each finding is an object with integer line_start and line_end (1-based, inclusive), "
    "a category from {categories} and a short description. Reply [] if there are no "
    "defects.\n\n```python\n{code}```\n"
)


@dataclass(frozen=True)
class AdmissionPolicy:
    """Fixed admission rules. Not a parameter of ``admit``."""
    split: str = "validation"
    seeds: tuple[int, ...] = (0, 1, 2)
    min_margin: float = 0.05            # mean utility gain required over the baseline
    confidence: float = 0.95            # two-sided paired bootstrap interval
    n_bootstrap: int = 2000
    bootstrap_seed: int = 0
    max_critical_violations: int = 0    # for the candidate, over all runs
    reference_cost_usd: float = 0.03    # utility cost ratio denominator
    reference_latency_s: float = 6.0    # utility latency ratio denominator
    benchmark_version: str = DEFAULT_VERSION

    def __post_init__(self):
        if self.split != "validation" or self.split not in OPEN_SPLITS:
            raise ValueError("admission must use the validation split")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique and nonempty")
        if not self.min_margin > 0 or not 0 < self.confidence < 1:
            raise ValueError("min_margin must be positive and confidence in (0, 1)")
        if self.max_critical_violations != 0:
            raise ValueError("violations are hard disqualifiers")
        if self.reference_cost_usd <= 0 or self.reference_latency_s <= 0:
            raise ValueError("references must be positive")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["seeds"] = list(self.seeds)
        return data

    def fingerprint(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


ADMISSION_POLICY = AdmissionPolicy()


@runtime_checkable
class Arm(Protocol):
    """Something admission can run: an ID plus the ``Strategy`` to execute.

    Gate 5 registry ``Candidate`` records are not arms themselves; use
    :func:`review_registered`, which builds arms from registry entries.
    """

    @property
    def candidate_id(self) -> str: ...

    @property
    def strategy(self) -> Strategy: ...


@dataclass(frozen=True)
class ArmSpec:
    candidate_id: str
    strategy: Strategy


@dataclass(frozen=True)
class RunResult:
    case_id: str
    seed: int
    utility: Optional[float]   # None when cost is unknown (cannot be scored)
    quality: float
    violation: bool
    violation_reasons: tuple[str, ...]
    parse_ok: bool
    error: Optional[str]


@dataclass(frozen=True)
class AdmissionDecision:
    candidate_id: str
    baseline_id: str
    admitted: bool
    reasons: tuple[str, ...]            # failed rules; empty when admitted
    mean_utility_gain: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    candidate_mean_utility: Optional[float]
    baseline_mean_utility: Optional[float]
    candidate_mean_quality: float
    baseline_mean_quality: float
    candidate_violations: int
    baseline_violations: int
    unparseable_outputs: int
    n_cases: int
    case_ids: tuple[str, ...]
    policy: Mapping
    policy_sha256: str
    verifier_sha256: str
    synthetic: bool = True
    notice: str = SYNTHETIC_NOTICE

    def to_dict(self) -> dict:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        data["case_ids"] = list(self.case_ids)
        data["policy"] = dict(self.policy)
        return data


def review_prompt(code: str) -> str:
    return REVIEW_PROMPT.format(categories=", ".join(DEFECT_CATEGORIES), code=code)


def parse_findings(output: str) -> tuple[list, bool]:
    """Findings list from executor output, or ([], False) if it is not a JSON list."""
    try:
        value = json.loads(output)
    except (TypeError, ValueError):
        return [], False
    return (value, True) if isinstance(value, list) else ([], False)


def _check_arm(arm, role: str) -> None:
    if not isinstance(arm, Arm):
        raise TypeError(f"{role} must provide candidate_id and strategy")
    if not isinstance(arm.candidate_id, str) or not arm.candidate_id.strip():
        raise ValueError(f"{role}.candidate_id must be a nonempty string")
    if not isinstance(arm.strategy, Strategy):
        raise TypeError(f"{role}.strategy must be an executor.Strategy")


def _run_arm(arm: Arm, cases: Sequence[verifier.BenchmarkCase],
             make_executor: Callable[[int], Executor], policy: AdmissionPolicy,
             ) -> list[RunResult]:
    results = []
    for seed in policy.seeds:
        executor = make_executor(seed)   # fresh instance per arm and seed
        for case in cases:
            outcome = executor.execute(Task(case.case_id, review_prompt(case.code)),
                                       arm.strategy)
            if not isinstance(outcome, ExecutionOutcome):
                raise TypeError("executor must return an ExecutionOutcome")
            findings, parse_ok = parse_findings(outcome.output)
            quality = verifier.quality(case, findings)
            utility = None
            if outcome.cost_usd is not None:
                utility = score_outcome(Reward(
                    quality=quality,
                    cost_ratio=outcome.cost_usd / policy.reference_cost_usd,
                    time_ratio=outcome.latency_s / policy.reference_latency_s,
                    violation=outcome.violation))
            results.append(RunResult(case.case_id, seed, utility, quality,
                                     outcome.violation, outcome.violation_reasons,
                                     parse_ok, outcome.error))
    return results


def _mean(values) -> Optional[float]:
    values = list(values)
    return statistics.fmean(values) if values else None


def admit(candidate: Arm, baseline: Arm,
          make_executor: Callable[[int], Executor]) -> AdmissionDecision:
    """Run candidate and baseline on the fixed validation split and decide.

    ``make_executor(seed)`` is called once per arm and seed, so both arms see
    the same executor configuration and seeds without sharing state.
    """
    with optimisation_phase():
        return _admit(candidate, baseline, make_executor, ADMISSION_POLICY)


def _admit(candidate, baseline, make_executor, policy: AdmissionPolicy) -> AdmissionDecision:
    _check_arm(candidate, "candidate")
    _check_arm(baseline, "baseline")
    if candidate.candidate_id == baseline.candidate_id:
        raise ValueError("candidate and baseline must differ")
    if not callable(make_executor):
        raise TypeError("make_executor must be a callable seed -> Executor")

    cases = evaluation.open_cases((policy.split,), version=policy.benchmark_version)[policy.split]
    cand = _run_arm(candidate, cases, make_executor, policy)
    base = _run_arm(baseline, cases, make_executor, policy)

    reasons: list[str] = []
    cand_violations = sum(r.violation for r in cand)
    base_violations = sum(r.violation for r in base)
    if cand_violations > policy.max_critical_violations:
        kinds = sorted({k for r in cand for k in r.violation_reasons})
        reasons.append(f"critical_violations: candidate had {cand_violations} "
                       f"({', '.join(kinds)})")
    unscorable = sum(r.utility is None for r in cand + base)
    gain = ci_low = ci_high = None
    if unscorable:
        reasons.append(f"unscorable_outcomes: {unscorable} runs with unknown cost")
    else:
        cand_by = {(r.case_id, r.seed): r.utility for r in cand}
        base_by = {(r.case_id, r.seed): r.utility for r in base}
        # Paired per case: mean over seeds of candidate minus baseline utility.
        diffs = [statistics.fmean(cand_by[(c.case_id, s)] - base_by[(c.case_id, s)]
                                  for s in policy.seeds) for c in cases]
        ci = bootstrap_ci(diffs, n_resamples=policy.n_bootstrap,
                          confidence=policy.confidence,
                          seed=f"admission:{policy.bootstrap_seed}")
        gain, ci_low, ci_high = ci["mean"], ci["ci_low"], ci["ci_high"]
        if gain < policy.min_margin:
            reasons.append(f"margin_not_met: mean utility gain {gain:.4f} < "
                           f"required {policy.min_margin}")
        if not ci_low > 0:
            reasons.append(f"ci_includes_zero: {policy.confidence:.0%} CI "
                           f"[{ci_low:.4f}, {ci_high:.4f}]")

    decision = AdmissionDecision(
        candidate_id=candidate.candidate_id, baseline_id=baseline.candidate_id,
        admitted=not reasons, reasons=tuple(reasons),
        mean_utility_gain=gain, ci_low=ci_low, ci_high=ci_high,
        candidate_mean_utility=_mean(r.utility for r in cand if r.utility is not None),
        baseline_mean_utility=_mean(r.utility for r in base if r.utility is not None),
        candidate_mean_quality=statistics.fmean(r.quality for r in cand),
        baseline_mean_quality=statistics.fmean(r.quality for r in base),
        candidate_violations=cand_violations, baseline_violations=base_violations,
        unparseable_outputs=sum(not r.parse_ok for r in cand + base),
        n_cases=len(cases), case_ids=tuple(c.case_id for c in cases),
        policy=policy.to_dict(), policy_sha256=policy.fingerprint(),
        verifier_sha256=verifier.verifier_fingerprint())
    if decision.admitted:
        logger.info("ADMIT %s over %s: gain %.4f, CI [%.4f, %.4f] (synthetic)",
                    decision.candidate_id, decision.baseline_id, gain, ci_low, ci_high)
    else:
        logger.warning("REJECT %s vs %s: %s (synthetic)", decision.candidate_id,
                       decision.baseline_id, "; ".join(decision.reasons))
    return decision


# ── Gate 5 registry ──────────────────────────────────────────────────────────

# Policy fingerprints whose decisions are accepted as admission evidence. Add a
# superseded policy here (in reviewed code) when ADMISSION_POLICY changes, so
# candidates admitted under it stay valid.
ACCEPTED_POLICY_SHA256 = frozenset({ADMISSION_POLICY.fingerprint()})
for _fingerprint in sorted(ACCEPTED_POLICY_SHA256):
    reg.register_admission_policy(_fingerprint)   # registry refuses other admissions


class AdmissionEvidenceError(reg.RegistryError):
    """An admitted candidate lacks a valid gate 6 admission decision."""


def _decided_by(policy_sha256: str) -> str:
    return f"admission-gate:{policy_sha256[:12]}"


def evaluation_arm(registry: reg.CandidateRegistry, candidate_id: str,
                   base: Strategy) -> ArmSpec:
    """Arm for a registry entry, built with ``registry.evaluation_strategy``.

    Evaluation only: tools come from ``base``, the strategy ID is prefixed
    ``eval:``, and no status changes.
    """
    if not isinstance(base, Strategy):
        raise TypeError("base must be an executor.Strategy")
    return ArmSpec(registry.get(candidate_id).candidate_id,
                   reg.evaluation_strategy(registry, candidate_id, base))


def review_registered(registry: reg.CandidateRegistry, candidate_id: str, *,
                      base: Strategy, make_executor: Callable[[int], Executor],
                      apply: bool = False) -> AdmissionDecision:
    """Admission review of a proposed registry candidate against its parent.

    The baseline is the candidate's ``parent_id``, which must be admitted.
    Provenance (GEPA's self-reported scores) is never read. Dry run by default;
    with ``apply=True`` the decision is recorded through ``registry.transition``
    as ``admitted`` or ``rejected`` (terminal), with the full decision as
    evidence. Only this function and ``_retire_after_rollback`` (gate 7, to
    ``retired`` only) may call ``transition``; a static test pins both.
    Use :func:`review_with_rollback` for the checkpointed admission path.
    """
    entry = registry.get(candidate_id)
    if entry.status != reg.PROPOSED:
        raise ValueError(f"{candidate_id} is {entry.status}; only proposed candidates "
                         "are reviewed")
    if not entry.parent_id:
        raise ValueError(f"{candidate_id} has no parent baseline")
    parent = registry.get(entry.parent_id)
    if parent.status != reg.ADMITTED:
        raise ValueError(f"baseline {parent.candidate_id} is {parent.status}, not admitted")
    if parent.skill_name != entry.skill_name:
        raise ValueError("candidate and baseline must be the same skill")
    decision = admit(evaluation_arm(registry, candidate_id, base),
                     evaluation_arm(registry, parent.candidate_id, base), make_executor)
    if apply:
        registry.transition(
            candidate_id, reg.ADMITTED if decision.admitted else reg.REJECTED,
            reason=("passed candidate-vs-baseline validation" if decision.admitted
                    else "; ".join(decision.reasons)),
            decided_by=_decided_by(decision.policy_sha256),
            evidence=decision.to_dict())
    return decision


def admission_evidence_errors(registry: reg.CandidateRegistry, candidate_id: str) -> list[str]:
    """Why an admitted candidate's admission is not a valid gate 6 decision ([] if valid).

    Baselines (entered admitted by ``register_baseline``) need no admission.
    """
    entry = registry.get(candidate_id)
    if entry.status != reg.ADMITTED:
        return [f"{candidate_id} is {entry.status}, not admitted"]
    if entry.source == reg.BASELINE:
        return []
    admissions = [t for t in registry.history(candidate_id) if t.to_status == reg.ADMITTED]
    if len(admissions) != 1:
        return [f"expected one admission record, found {len(admissions)}"]
    record, ev = admissions[0], dict(admissions[0].evidence)
    errors = []
    sha = ev.get("policy_sha256")
    if sha not in ACCEPTED_POLICY_SHA256:
        errors.append("evidence policy_sha256 is not an accepted admission policy")
    elif record.decided_by != _decided_by(sha):
        errors.append(f"decided_by {record.decided_by!r} is not the admission gate")
    try:
        policy_ok = AdmissionPolicy(**{**ev.get("policy", {}),
                                       "seeds": tuple(ev.get("policy", {}).get("seeds", ()))}
                                    ).fingerprint() == sha
    except (TypeError, ValueError):
        policy_ok = False
    if not policy_ok:
        errors.append("evidence policy does not match its policy_sha256")
    if ev.get("admitted") is not True or ev.get("reasons"):
        errors.append("evidence is not an admission decision")
    if ev.get("candidate_id") != candidate_id or ev.get("baseline_id") != entry.parent_id:
        errors.append("evidence is for a different candidate or baseline")
    if ev.get("candidate_violations") != 0:
        errors.append("evidence records candidate violations")
    gain, low = ev.get("mean_utility_gain"), ev.get("ci_low")
    margin = ev.get("policy", {}).get("min_margin")
    if not (isinstance(gain, (int, float)) and isinstance(low, (int, float))
            and isinstance(margin, (int, float)) and gain >= margin and low > 0):
        errors.append("evidence does not meet the admission margin and CI rules")
    return errors


def admitted_catalogue(registry: reg.CandidateRegistry, skill_name: str,
                       base: Strategy) -> tuple[Strategy, ...]:
    """``registry.admitted_strategies``, refusing any admission not made by this gate."""
    for entry in registry.candidates(skill_name=skill_name, status=reg.ADMITTED):
        errors = admission_evidence_errors(registry, entry.candidate_id)
        if errors:
            raise AdmissionEvidenceError(f"{entry.candidate_id}: " + "; ".join(errors))
    return reg.admitted_strategies(registry, skill_name, base)


# ── Gate 7: checkpoint before admission, automatic rollback after ────────────

ROLLBACK_DECIDED_BY = f"admission-rollback:{ADMISSION_POLICY.fingerprint()[:12]}"


def catalogue_validation(registry: reg.CandidateRegistry, candidate_id: str, *,
                         base: Strategy, make_executor: Callable[[int], Executor],
                         ) -> ck.Validation:
    """Validation-split utility of an ADMITTED candidate's catalogue strategy.

    Same cases, seeds, references and scoring as ``admit`` (``ADMISSION_POLICY``),
    but the strategy is the one the catalogue would serve
    (``registry.candidate_strategy``), not the ``eval:`` arm. Used for both the
    pre-admission (parent) and post-admission (new candidate) measurement, so
    the two are comparable. Unknown cost raises instead of scoring as free.
    """
    with optimisation_phase():
        policy = ADMISSION_POLICY
        strategy = reg.candidate_strategy(registry, candidate_id, base)
        cases = evaluation.open_cases((policy.split,),
                                      version=policy.benchmark_version)[policy.split]
        runs = _run_arm(ArmSpec(candidate_id, strategy), cases, make_executor, policy)
        if any(r.utility is None for r in runs):
            raise ValueError(f"{candidate_id}: unscorable runs (unknown cost)")
        return ck.Validation(utility=statistics.fmean(r.utility for r in runs),
                             critical_violation=any(r.violation for r in runs),
                             n_tasks=len(runs), synthetic=True)


@dataclass(frozen=True)
class CheckpointedAdmission:
    decision: AdmissionDecision
    kept: bool                    # admitted by the gate and kept after the post check
    rolled_back: bool             # admitted by the gate, then retired by the trigger
    reason: str
    active_skill: str             # registry candidate ID active after this call
    pre_checkpoint: ck.Checkpoint
    post_checkpoint: Optional[ck.Checkpoint]
    pre_validation: ck.Validation
    post_validation: Optional[ck.Validation]
    synthetic: bool = True
    notice: str = SYNTHETIC_NOTICE


def _retire_after_rollback(registry: reg.CandidateRegistry, candidate_id: str, *,
                           reason: str, evidence: Mapping) -> None:
    """Gate 7 rollback of a skill: admitted -> retired (terminal). Nothing else."""
    to_status = reg.RETIRED
    assert to_status == reg.RETIRED
    registry.transition(candidate_id, to_status, reason=reason,
                        decided_by=ROLLBACK_DECIDED_BY, evidence=dict(evidence))
    logger.warning("ROLLBACK %s retired: %s (synthetic)", candidate_id, reason)


def review_with_rollback(registry: reg.CandidateRegistry, candidate_id: str, *,
                         base: Strategy, make_executor: Callable[[int], Executor],
                         store: ck.CheckpointStore, policy) -> CheckpointedAdmission:
    """Checkpointed admission of a proposed registry candidate.

    1. Measure the parent (active skill) with :func:`catalogue_validation` and
       save a ``good`` pre-admission checkpoint of ``policy`` whose skill
       version is the parent's candidate ID.
    2. :func:`review_registered` with ``apply=True`` (the gate 6 decision).
       A rejection is logged and nothing is rolled back.
    3. If admitted, measure the new candidate the same way and save a
       ``candidate`` checkpoint. ``checkpoint.rollback_decision`` against the
       pre-admission checkpoint's recorded utility then either promotes it or
       rolls back: the candidate is retired in the registry (removing it from
       the catalogue) and the checkpoint store rolls back atomically to the
       pre-admission checkpoint. Both are logged with the same evidence.

    The registry and checkpoint store are separate databases, so the two
    writes cannot share one transaction. The order is fail-safe: the candidate
    is retired first. If the post-admission check raises, the candidate is
    retired and the pre-admission checkpoint restored before the error
    propagates, so an admitted candidate never stays without a passed check.
    """
    with optimisation_phase():
        return _review_with_rollback(registry, candidate_id, base=base,
                                     make_executor=make_executor, store=store, policy=policy)


def _review_with_rollback(registry, candidate_id, *, base, make_executor, store, policy):
    entry = registry.get(candidate_id)
    if entry.status != reg.PROPOSED or not entry.parent_id:
        raise ValueError(f"{candidate_id} must be a proposed candidate with a parent")
    parent_id = entry.parent_id
    label = f"admission:{candidate_id}"

    pre_validation = catalogue_validation(registry, parent_id, base=base,
                                          make_executor=make_executor)
    pre = store.save(policy, skill_version=parent_id, label=f"{label}:pre",
                     validation=pre_validation)
    decision = review_registered(registry, candidate_id, base=base,
                                 make_executor=make_executor, apply=True)
    if not decision.admitted:
        store.record_event("admission_rejected", checkpoint_id=pre.checkpoint_id,
                           reason="; ".join(decision.reasons),
                           details={"candidate_id": candidate_id, "parent_id": parent_id,
                                    "policy_sha256": decision.policy_sha256})
        return CheckpointedAdmission(decision, False, False, "rejected by admission gate",
                                     parent_id, pre, None, pre_validation, None)

    try:
        post_validation = catalogue_validation(registry, candidate_id, base=base,
                                               make_executor=make_executor)
    except BaseException as exc:
        why = f"post-admission validation failed: {type(exc).__name__}: {exc}"
        _retire_after_rollback(registry, candidate_id, reason=why,
                               evidence={"pre_checkpoint_id": pre.checkpoint_id,
                                         "error_type": type(exc).__name__,
                                         "synthetic": True})
        store.restore(policy, pre.checkpoint_id, reason=why)
        raise

    post = store.save(policy, skill_version=candidate_id, label=f"{label}:post",
                      status=ck.STATUS_CANDIDATE, validation=post_validation, activate=False)
    pre_utility = store.load(pre.checkpoint_id).validation["utility"]
    reason = ck.rollback_decision(pre_utility, post_validation)
    details = {"candidate_id": candidate_id, "parent_id": parent_id,
               "pre_checkpoint_id": pre.checkpoint_id,
               "post_checkpoint_id": post.checkpoint_id,
               "pre_utility": pre_utility, "post_utility": post_validation.utility,
               "margin": ck.ROLLBACK_MARGIN,
               "critical_violation": post_validation.critical_violation,
               "admission_policy_sha256": decision.policy_sha256, "synthetic": True}
    if reason is None:
        store.promote(post.checkpoint_id, reason="post-admission validation passed",
                      details=details)
        return CheckpointedAdmission(decision, True, False, "admitted and kept",
                                     candidate_id, pre, store.load(post.checkpoint_id),
                                     pre_validation, post_validation)

    _retire_after_rollback(registry, candidate_id,
                           reason=f"post-admission rollback: {reason}", evidence=details)
    active = store.rollback(policy, from_checkpoint_id=post.checkpoint_id,
                            reason=reason, details=details)
    return CheckpointedAdmission(decision, False, True, reason, active.skill_version, pre,
                                 store.load(post.checkpoint_id), pre_validation,
                                 post_validation)

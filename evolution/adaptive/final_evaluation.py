"""The one-time, recorded final evaluation on the held-out test split.

Owner decision (KAFO-EVO-001, 2026-11-22): the final test evaluation is one
deliberate, explicitly invoked, logged, in-process step, run once. This
module is that step and the only package code allowed to call
``evaluation.evaluate_sealed_test`` (see the guard scan in
``tests/test_benchmark_integration.py``).

How it is enforced:

1. The caller must pass ``confirm=CONFIRM_PHRASE`` and a ``reviewer_id``.
2. It refuses inside ``benchmark.optimisation_phase()``.
3. Before any test case is loaded, the ledger file is created with
   exclusive-create (``open(path, "x")``) and a ``started`` record. If the
   ledger already exists, for any reason, the run is refused. A crash or a
   reviewer error therefore still counts as "the final evaluation happened";
   the ledger then records ``failed``.
4. On success the ledger records the scores, UTC timestamps, verifier and
   benchmark provenance and the reviewer identity, and the result is logged.

The reviewer is called once per test case with ``evaluation.review_input``
(code and context only, never the labelled defects). Results must not feed
back into optimisation or selection.

Owner decision (2026-11-22): the reviewer is the live ``HermesExecutor``
with the ``github-code-review`` skill version the full live run produced
(the admitted GEPA candidate if one passed gate 6, otherwise the baseline),
chosen with :func:`select_final_skill`. :class:`ExecutorReviewer` wraps any
executor, tracks cost against a hard cap, never lets one failed call abort
the run, and its :meth:`ExecutorReviewer.summary` (cost, errors,
violations) is written to the ledger together with ``skill_version``.

Owner decision (2026-11-22, option 1): the one run is PAIRED.
:func:`run_paired_final_evaluation` has the baseline and the admitted
candidate skill review the same sealed cases with the same model, provider,
prompt and sample number, scores both STRICT (gate 3 verifier, PRIMARY) and
LENIENT (audited alternate categories, SECONDARY, see ``lenient``), and
records the paired differences with bootstrap CIs, all as pre-registered in
:data:`ANALYSIS_PLAN` and ``docs/sprints/KAFO-EVO-001.md``. Both functions
write the same run-once ledger, so only one of them can ever run.

There is deliberately no command-line entry point; the run is an explicit
in-process call.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import admission, benchmark, evaluation, lenient, verifier
from . import registry as reg
from .executor import ExecutionOutcome, Strategy, Task

logger = logging.getLogger(__name__)

CONFIRM_PHRASE = "I am running the one-time final test evaluation"
LEDGER_SCHEMA = "kafo-final-evaluation/1"
PAIRED_LEDGER_SCHEMA = "kafo-final-evaluation/2"
ARMS = ("baseline", "candidate")

# Pre-registered before any sealed call (owner decision 2026-11-22, option 1);
# mirrored in docs/sprints/KAFO-EVO-001.md and recorded, with its SHA-256, in
# the ledger. Do not change it after the run.
ANALYSIS_PLAN = {
    "id": "kafo-evo-001-final-eval-plan/1",
    "design": ("paired: the baseline and the candidate skill each review the same "
               "sealed test cases once (sample 0), same model, provider, budget, "
               "prompt and parser; calls interleave per case, in case_id order the "
               "baseline goes first on the 1st, 3rd, ... case and the candidate on the "
               "2nd, 4th, ..."),
    "primary": ("candidate minus baseline micro-F1 under STRICT scoring (gate 3 "
                "verifier, exact normalised category)"),
    "secondary": ("candidate minus baseline micro-F1 under LENIENT scoring (strict "
                  "rules, plus the label audit's alternate categories)"),
    "descriptive": ("per-skill micro precision, recall and F1 and macro F1 with CIs; "
                    "paired macro-F1, micro-precision and micro-recall differences; "
                    "cost, completions, parse failures, violations"),
    "ci": ("95% paired percentile bootstrap over cases: 10000 resamples of case "
           "indices (random.Random(0), cases sorted by case_id), the same resamples "
           "for both skills, linearly interpolated quantiles"),
    "missing_data": ("a call that errors or reports no cost is re-sent once "
                     "automatically by the spend-ledger wrapper (as in the full run); "
                     "a call still failed after that, skipped (cost cap) or unparseable "
                     "is scored as an empty review for that skill and case; nothing "
                     "else is re-run"),
    "interpretation": ("the candidate counts as better on the sealed test only if the "
                       "primary CI lower bound is above 0; the secondary result cannot "
                       "override the primary; synthetic hand-written cases, so no claim "
                       "of a real Hermes improvement"),
    "seed": 0, "n_bootstrap": 10000, "confidence": 0.95,
}
ANALYSIS_PLAN_SHA256 = hashlib.sha256(json.dumps(
    ANALYSIS_PLAN, sort_keys=True).encode("utf-8")).hexdigest()
DEFAULT_LEDGER = (Path(__file__).resolve().parents[2] / "reports" / "kafo-evo-001"
                  / "final_test_evaluation.json")


class FinalEvaluationRefused(PermissionError):
    """The final evaluation was not run (already run, not confirmed, or in a phase)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_ledger(ledger_path: str | os.PathLike = DEFAULT_LEDGER) -> dict | None:
    """The recorded final evaluation, or None if it has never been started."""
    path = Path(ledger_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_replace(path: Path, record: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


class CostMeter:
    """Spend shared by several reviewers: one cap, one stop for all of them."""

    def __init__(self, max_total_cost_usd: float):
        if isinstance(max_total_cost_usd, bool) or not isinstance(
                max_total_cost_usd, (int, float)) or not 0 < max_total_cost_usd < 1e6:
            raise ValueError("max_total_cost_usd must be a positive number")
        self.max_total_cost_usd = float(max_total_cost_usd)
        self.total_cost_usd = 0.0
        self.stopped: str | None = None

    def check(self) -> str | None:
        if self.stopped is None and self.total_cost_usd >= self.max_total_cost_usd:
            self.stopped = f"cost cap USD {self.max_total_cost_usd} reached"
        return self.stopped


class ExecutorReviewer:
    """Reviewer that runs ``executor`` with one fixed strategy per test case.

    * Prompt and parsing are gate 6's (``admission.review_prompt`` and
      ``admission.parse_findings``), so the final evaluation asks exactly what
      admission asked.
    * Spending stops (remaining cases get an empty review) once the known cost
      reaches ``max_total_cost_usd``, or after a call whose cost is unknown,
      because the cap can no longer be enforced. The reason is recorded.
    * An executor exception is recorded and scored as an empty review instead
      of aborting, so one transient failure cannot void the one-time run.
    * Pass one :class:`CostMeter` to several reviewers to share the cap and the
      stop (the paired run does); otherwise each reviewer has its own meter.
    """

    def __init__(self, executor, strategy: Strategy, *,
                 max_total_cost_usd: float | None = None, meter: CostMeter | None = None):
        if not isinstance(strategy, Strategy):
            raise TypeError("strategy must be an executor.Strategy")
        if (meter is None) == (max_total_cost_usd is None):
            raise ValueError("pass exactly one of max_total_cost_usd or meter")
        self.meter = meter if meter is not None else CostMeter(max_total_cost_usd)
        self.executor, self.strategy = executor, strategy
        self.cases: list[dict] = []
        self.total_cost_usd = 0.0      # this reviewer's own spend

    @property
    def max_total_cost_usd(self) -> float:
        return self.meter.max_total_cost_usd

    @property
    def stopped(self) -> str | None:
        return self.meter.stopped

    def __call__(self, view: Mapping) -> list:
        case_id = view["case_id"]
        if self.meter.check() is not None:
            self.cases.append({"case_id": case_id, "skipped": self.meter.stopped})
            return []
        try:
            outcome = self.executor.execute(
                Task(case_id, admission.review_prompt(view["code"])), self.strategy)
            if not isinstance(outcome, ExecutionOutcome):
                raise TypeError("executor must return an ExecutionOutcome")
        except Exception as exc:   # recorded, scored as an empty review
            self.cases.append({"case_id": case_id, "exception":
                               f"{type(exc).__name__}: {exc}"[:300]})
            return []
        findings, parse_ok = admission.parse_findings(outcome.output)
        if outcome.cost_usd is None:
            if self.meter.stopped is None:
                self.meter.stopped = f"cost unknown after {case_id}; cap cannot be enforced"
        else:
            self.total_cost_usd += outcome.cost_usd
            self.meter.total_cost_usd += outcome.cost_usd
        self.cases.append({
            "case_id": case_id, "completed": outcome.completed, "parse_ok": parse_ok,
            "cost_usd": outcome.cost_usd, "cost_status": outcome.cost_status,
            "latency_s": outcome.latency_s, "error": outcome.error,
            "violation_reasons": list(outcome.violation_reasons), "model": outcome.model})
        return findings

    def summary(self) -> dict:
        ran = [c for c in self.cases if "completed" in c]
        return {"strategy_id": self.strategy.strategy_id,
                "toolsets": list(self.strategy.toolsets),
                "backend": getattr(self.executor, "backend", type(self.executor).__name__),
                "model": getattr(self.executor, "model", None),
                "provider": getattr(self.executor, "provider", None),
                "max_total_cost_usd": self.max_total_cost_usd,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "meter_total_cost_usd": round(self.meter.total_cost_usd, 6),
                "cost_statuses": dict(Counter(c["cost_status"] for c in ran)),
                "calls": len(ran), "completed": sum(c["completed"] for c in ran),
                "unparseable": sum(not c["parse_ok"] for c in ran),
                "violations": sum(bool(c["violation_reasons"]) for c in ran),
                "exceptions": sum("exception" in c for c in self.cases),
                "skipped": sum("skipped" in c for c in self.cases),
                "stopped": self.stopped, "cases": list(self.cases)}


def select_final_skill(registry: reg.CandidateRegistry, skill_name: str,
                       base: Strategy) -> tuple[reg.Candidate, Strategy]:
    """Owner rule: the gate-6-admitted GEPA candidate if exactly one exists,
    otherwise the admitted baseline. Admission evidence is re-checked."""
    catalogue = {s.strategy_id: s for s in admission.admitted_catalogue(
        registry, skill_name, base)}
    admitted = registry.candidates(skill_name=skill_name, status=reg.ADMITTED)
    gepa = [c for c in admitted if c.source == reg.GEPA]
    if len(gepa) > 1:
        raise ValueError(f"{len(gepa)} admitted GEPA candidates for {skill_name!r}: "
                         f"{[c.candidate_id for c in gepa]}; choose one explicitly")
    chosen = gepa or [c for c in admitted if c.source == reg.BASELINE]
    if len(chosen) != 1:
        raise ValueError(f"expected one admitted baseline for {skill_name!r}, "
                         f"found {len(chosen)}")
    return chosen[0], catalogue[f"{base.strategy_id}:{chosen[0].candidate_id}"]


def _create_ledger(path: Path, record: dict) -> None:
    """Exclusive-create the ledger with the ``started`` record: one run only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as handle:   # atomic: one run only
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        previous = read_ledger(path) or {}
        raise FinalEvaluationRefused(
            f"the final evaluation was already {previous.get('status', 'recorded')} at "
            f"{previous.get('started_at', '?')} ({path}); it runs once") from None


def _add_summary(record: dict, reviewer) -> None:
    summary = getattr(reviewer, "summary", None)
    if callable(summary):
        try:
            record["reviewer_summary"] = summary()
        except Exception as exc:   # never lose the ledger over a summary bug
            record["reviewer_summary"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}


def run_final_evaluation(reviewer: Callable[[Mapping], Sequence], *, reviewer_id: str,
                         synthetic: bool, confirm: str,
                         skill_version: str | None = None,
                         ledger_path: str | os.PathLike = DEFAULT_LEDGER,
                         version: str = benchmark.DEFAULT_VERSION,
                         seed: int = 0, n_bootstrap: int = 1000) -> dict:
    """Run the final test evaluation once and record it. Refuses a second run."""
    if confirm != CONFIRM_PHRASE:
        raise FinalEvaluationRefused("confirm must equal final_evaluation.CONFIRM_PHRASE")
    if benchmark.in_optimisation_phase():
        raise FinalEvaluationRefused("the final evaluation cannot run inside an "
                                     "optimisation phase")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError("reviewer_id must be a nonempty string")
    if not isinstance(synthetic, bool):
        raise ValueError("synthetic must be True or False")
    if not callable(reviewer):
        raise TypeError("reviewer must be callable")

    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if skill_version is not None and (not isinstance(skill_version, str)
                                      or not skill_version.strip()):
        raise ValueError("skill_version must be a nonempty string or None")
    record = {"schema": LEDGER_SCHEMA, "status": "started", "started_at": _now(),
              "reviewer_id": reviewer_id, "synthetic": synthetic,
              "skill_version": skill_version,
              "benchmark": {"name": benchmark.BENCHMARK_NAME, "version": version,
                            "manifest_sha256": benchmark.PINNED_MANIFEST_SHA256.get(version)},
              "verifier_bootstrap": {"seed": seed, "resamples": n_bootstrap}}
    _create_ledger(path, record)

    logger.warning("FINAL TEST EVALUATION started by reviewer %r (ledger %s); results "
                   "must not feed back into optimisation or selection", reviewer_id, path)
    try:
        report = evaluation.evaluate_sealed_test(
            reviewer=reviewer, final_evaluation=True, version=version,
            seed=seed, n_bootstrap=n_bootstrap)
    except BaseException as exc:
        record.update(status="failed", finished_at=_now(),
                      error=f"{type(exc).__name__}: {exc}"[:500])
        _add_summary(record, reviewer)
        _write_replace(path, record)
        logger.error("FINAL TEST EVALUATION failed: %s", record["error"])
        raise
    full = report.to_dict()
    record.update(
        status="completed", finished_at=_now(), n_cases=len(report.cases),
        micro=full["micro"], macro=full["macro"],
        totals={k: sum(c[k] for c in full["cases"])
                for k in ("true_positives", "false_positives", "false_negatives")},
        verifier={k: full[k] for k in ("verifier_version", "match_rule",
                                       "verifier_sha256", "config")})
    _add_summary(record, reviewer)
    _write_replace(path, record)
    logger.warning("FINAL TEST EVALUATION completed: %d cases, micro F1 %.3f",
                   record["n_cases"], record["micro"]["f1"]["point"])
    return record


# ── Paired one-time run (owner decision 2026-11-22, option 1) ──────────────

def _scores(scores: verifier.Scores) -> dict:
    return scores.to_dict()


def _arm_record(results: Sequence[verifier.CaseResult], *, seed: int, n_bootstrap: int,
                confidence: float) -> dict:
    micro, macro = verifier.aggregate(results, seed=seed, n_bootstrap=n_bootstrap,
                                      confidence=confidence)
    return {"micro": _scores(micro), "macro": _scores(macro),
            "totals": {k: sum(getattr(r, k) for r in results)
                       for k in ("true_positives", "false_positives", "false_negatives")},
            "cases": [{"case_id": r.case_id, "tp": r.true_positives,
                       "fp": r.false_positives, "fn": r.false_negatives}
                      for r in sorted(results, key=lambda r: r.case_id)]}


def _committable_summary(summary: Mapping) -> dict:
    """A reviewer summary safe to commit: per case only IDs, numbers, flags and
    error TYPES. Free text (errors, exception messages, violation reasons) may
    echo sealed content, so it stays in the run's own spend ledger."""
    if "error" in summary and "cases" not in summary:      # summary() itself failed
        return {"error": str(summary["error"]).split(":", 1)[0]}
    out = {k: v for k, v in summary.items() if k != "cases"}
    rows = []
    for c in summary.get("cases", []):
        row = {k: c[k] for k in ("case_id", "completed", "parse_ok", "cost_usd",
                                 "cost_status", "latency_s", "model") if k in c}
        if "skipped" in c:
            row["skipped"] = True
        if "exception" in c:
            row["exception_type"] = str(c["exception"]).split(":", 1)[0]
        if "error" in c:
            row["error"] = c["error"] is not None
        if "violation_reasons" in c:
            row["violations"] = len(c["violation_reasons"])
        rows.append(row)
    out["cases"] = rows
    return out


def run_paired_final_evaluation(reviewers: Mapping[str, Callable[[Mapping], Sequence]], *,
                                skill_versions: Mapping[str, str], reviewer_id: str,
                                synthetic: bool, confirm: str,
                                ledger_path: str | os.PathLike = DEFAULT_LEDGER,
                                version: str = benchmark.DEFAULT_VERSION) -> dict:
    """Run the pre-registered paired final evaluation once and record it.

    ``reviewers`` and ``skill_versions`` have exactly the keys ``baseline``
    and ``candidate``. Scoring, seeds and CIs come from :data:`ANALYSIS_PLAN`
    and cannot be changed by the caller. Refuses a second run (any run,
    paired or not, shares the ledger).
    """
    if confirm != CONFIRM_PHRASE:
        raise FinalEvaluationRefused("confirm must equal final_evaluation.CONFIRM_PHRASE")
    if benchmark.in_optimisation_phase():
        raise FinalEvaluationRefused("the final evaluation cannot run inside an "
                                     "optimisation phase")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError("reviewer_id must be a nonempty string")
    if not isinstance(synthetic, bool):
        raise ValueError("synthetic must be True or False")
    if set(reviewers) != set(ARMS) or set(skill_versions) != set(ARMS):
        raise ValueError(f"reviewers and skill_versions need exactly the keys {ARMS}")
    if not all(callable(reviewers[a]) for a in ARMS):
        raise TypeError("reviewers must be callable")
    if not all(isinstance(skill_versions[a], str) and skill_versions[a].strip()
               for a in ARMS) or skill_versions["baseline"] == skill_versions["candidate"]:
        raise ValueError("skill_versions must be two different nonempty strings")
    seed, n_boot, conf = (ANALYSIS_PLAN["seed"], ANALYSIS_PLAN["n_bootstrap"],
                          ANALYSIS_PLAN["confidence"])
    alternates = lenient.load_alternates()          # before the ledger: a bad table
    provenance = lenient.alternates_provenance()    # must not use up the one run

    path = Path(ledger_path)
    record = {"schema": PAIRED_LEDGER_SCHEMA, "status": "started", "started_at": _now(),
              "design": "paired", "reviewer_id": reviewer_id, "synthetic": synthetic,
              "skill_versions": {a: skill_versions[a] for a in ARMS},
              "analysis_plan": dict(ANALYSIS_PLAN),
              "analysis_plan_sha256": ANALYSIS_PLAN_SHA256,
              "primary": "strict.micro_f1", "secondary": "lenient.micro_f1",
              "alternates": provenance,
              "benchmark": {"name": benchmark.BENCHMARK_NAME, "version": version,
                            "manifest_sha256": benchmark.PINNED_MANIFEST_SHA256.get(version)},
              "verifier": {"verifier_version": verifier.VERIFIER_VERSION,
                           "match_rule": verifier.MATCH_RULE,
                           "verifier_sha256": verifier.verifier_fingerprint(),
                           "config": verifier.MatchConfig().to_dict()}}
    _create_ledger(path, record)
    logger.warning("PAIRED FINAL TEST EVALUATION started by %r (ledger %s); results must "
                   "not feed back into optimisation or selection", reviewer_id, path)

    def summaries() -> None:
        out = {}
        for arm in ARMS:
            holder: dict = {}
            _add_summary(holder, reviewers[arm])
            if "reviewer_summary" in holder:
                out[arm] = _committable_summary(holder["reviewer_summary"])
        if out:
            record["reviewer_summaries"] = out

    try:
        cases = evaluation.sealed_test_cases(final_evaluation=True, version=version)
        ids = {c.case_id for c in cases}
        defect_ids = {d.defect_id for c in cases for d in c.defects}
        stray = sorted(d for d in alternates if d.rsplit("-d", 1)[0] in ids
                       and d not in defect_ids)
        if stray:
            raise lenient.AlternatesError(f"{len(stray)} alternates entries name no "
                                          "defect of their test case")
        reviews: dict[str, dict[str, list]] = {a: {} for a in ARMS}
        order = []
        for i, case in enumerate(cases):           # sorted by case_id
            arms = ARMS if i % 2 == 0 else ARMS[::-1]
            order.append(arms[0])
            view = evaluation.review_input(case)
            for arm in arms:
                reviews[arm][case.case_id] = list(reviewers[arm](view))
        strict = {a: tuple(verifier.score_case(c, reviews[a][c.case_id]) for c in cases)
                  for a in ARMS}
        loose = {a: tuple(lenient.score_case(c, reviews[a][c.case_id], alternates)
                          for c in cases) for a in ARMS}
        results = dict(
            n_cases=len(cases), first_call_by_position=order,
            scores={name: {a: _arm_record(res[a], seed=seed, n_bootstrap=n_boot,
                                          confidence=conf) for a in ARMS}
                    for name, res in (("strict", strict), ("lenient", loose))},
            paired={name: lenient.paired_bootstrap(res["baseline"], res["candidate"],
                                                   seed=seed, n_bootstrap=n_boot,
                                                   confidence=conf)
                    for name, res in (("strict", strict), ("lenient", loose))},
            lenient_defects_in_split=sum(d.defect_id in alternates
                                         for c in cases for d in c.defects))
    except BaseException as exc:
        record.update(status="failed", finished_at=_now(),
                      error=f"{type(exc).__name__}: {exc}"[:500])
        summaries()
        _write_replace(path, record)
        logger.error("PAIRED FINAL TEST EVALUATION failed: %s", record["error"])
        raise
    record.update(status="completed", finished_at=_now(), **results)
    summaries()
    _write_replace(path, record)
    p = record["paired"]["strict"]["micro_f1"]
    logger.warning("PAIRED FINAL TEST EVALUATION completed: %d cases, strict micro-F1 "
                   "difference %.3f [%.3f, %.3f] (synthetic)", len(cases), p["point"],
                   p["low"], p["high"])
    return record

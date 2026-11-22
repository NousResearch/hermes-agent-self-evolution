"""Reproducible KAFO-EVO-001 sprint report: quality, utility, cost, latency,
uncertainty and safety.

Usage (offline, seeded; writes only to ``--out``)::

    python -m evolution.adaptive.report            # -> reports/kafo-evo-001/report.{md,json}

What it reports, all from merged sprint code:

* Gate 4 policy comparison (``compare.compare``) on the synthetic simulator
  (``synthetic.py``), with seed-level bootstrap CIs. The headline sentences
  are derived from the comparison verdicts, never hard-coded.
* Gate 3 verifier scores on the gate 2 benchmark's open splits (train,
  validation) for reference reviews: the labelled defects (ceiling), an
  empty review (floor) and an invented, seeded "noisy reference" built from
  the labels. None of these is a model.
* Cost and latency (simulated), and safety: critical violations, sealed
  split access attempts during generation, registry rejections and
  retirements, admission and rollback (pluggable, pending until gates 6 and
  7 merge), an optional live Hermes smoke run, and the final test
  evaluation ledger (read only; this module never runs it).
* The live full run (``full_run``, live Hermes on the hand-written
  benchmark) from a committed snapshot, ``live_run_live-1.json``: the 2-arm
  live comparison, GEPA candidates and lineage, the gate 6 admission
  decision, gate 7 checkpoints, the registry history and per-stage cost.
  ``--snapshot-live-run <run dir>`` rebuilds that snapshot (read-only on the
  run dir; aggregates only, never model outputs) so the report stays
  reproducible without the run directory.

Everything runs inside ``benchmark.optimisation_phase()``, so the sealed test
split cannot be loaded or opened while the report is generated. The output
contains no wall-clock time unless ``--generated-at`` is given, so the same
code and seed reproduce byte-identical files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import statistics
import math
from pathlib import Path
import random
import sqlite3
import sys
from typing import Mapping, Optional, Sequence

from . import admission, benchmark, compare, evaluation, synthetic, verifier
from .benchmark import optimisation_phase, sealed_access_attempts

SCHEMA = "kafo-sprint-report/1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "reports" / "kafo-evo-001"
DEFAULT_LEDGER = DEFAULT_OUT / "final_test_evaluation.json"
NOTICE = ("SYNTHETIC RESULTS. The policy comparison uses a hypothetical simulator with "
          "invented success rates, costs and latencies; the verifier scores use reference "
          "reviews derived from the benchmark labels, not a model. Nothing in this report "
          "is evidence that Hermes improves on any real task.")
SIMULATOR_NOTICE = ("SYNTHETIC RESULTS. The simulator sections (gate 4 policy comparison, "
                    "simulated cost and latency) use a hypothetical simulator with invented "
                    "success rates, costs and latencies; the verifier reference scores use "
                    "reviews derived from the benchmark labels, not a model.")
NO_REAL_WORLD = ("Nothing in this report is evidence that Hermes improves on any real task: "
                 "no result here is evidence of real-world improvement.")
DEFAULT_LIVE_RUN = DEFAULT_OUT / "live_run_live-1.json"
LIVE_RUN_SCHEMA = "kafo-live-run-snapshot/1"
NO_CHECKPOINT_STORE = ("no checkpoint store supplied (--checkpoints). Gate 6 admission and "
                       "the gate 7 rollback on the admission path "
                       "(admission.review_with_rollback) are merged, but no admission review "
                       "has been run outside the test suite, so there is no rollback history: "
                       "0 rollbacks is not a safety result")
DEFAULT_LIVE_SMOKE = DEFAULT_OUT / "live_smoke_2026-11-22.json"
NOISY_KEEP, NOISY_FALSE_ALARM = 0.7, 0.25


# ── Verifier section ─────────────────────────────────────────────────────

def noisy_reference_review(case: verifier.BenchmarkCase, seed: int) -> list[dict]:
    """Invented reviewer for showing non-trivial intervals: keeps each labelled
    defect with probability 0.7 (line shifted by -1/0/+1) and adds one false
    alarm with probability 0.25. Derived from labels; NOT a model."""
    rng = random.Random(f"noisy-reference:{seed}:{case.case_id}")
    n_lines = max(1, len(case.code.splitlines()))
    findings = []
    for d in case.defects:
        if rng.random() < NOISY_KEEP:
            shift = rng.choice((-1, 0, 1))
            start = min(max(1, d.line_start + shift), n_lines)
            findings.append({"line_start": start,
                             "line_end": max(start, min(d.line_end + shift, n_lines)),
                             "category": d.category})
    if rng.random() < NOISY_FALSE_ALARM:
        findings.append({"line_start": rng.randint(1, n_lines), "category": "style"})
    return findings


def _interval(iv: Mapping) -> dict:
    return {"point": iv["point"], "low": iv["low"], "high": iv["high"]}


def _summarise(rep: verifier.EvaluationReport) -> dict:
    d = rep.to_dict()
    return {"n_cases": d["n_cases"],
            "totals": {k: sum(c[k] for c in d["cases"])
                       for k in ("true_positives", "false_positives", "false_negatives")},
            "micro": {m: _interval(d["micro"][m]) for m in ("precision", "recall", "f1")},
            "macro": {m: _interval(d["macro"][m]) for m in ("precision", "recall", "f1")},
            "missing_reviews": len(d["missing_reviews"])}


def verifier_section(*, seed: int, n_bootstrap: int) -> dict:
    cases = evaluation.open_cases()
    all_cases = [c for split in cases.values() for c in split]
    reviewers = {
        "ground_truth": {c.case_id: evaluation.ground_truth_findings(c) for c in all_cases},
        "empty": {c.case_id: [] for c in all_cases},
        "noisy_reference": {c.case_id: noisy_reference_review(c, seed) for c in all_cases},
    }
    out = {}
    for name, reviews in reviewers.items():
        reports = evaluation.evaluate_open(reviews, seed=seed, n_bootstrap=n_bootstrap)
        out[name] = {split: _summarise(r) for split, r in reports.items()}
    any_report = verifier.evaluate(all_cases[:1], {}, n_bootstrap=1).to_dict()
    provenance = {k: any_report[k] for k in ("verifier_version", "match_rule",
                                             "verifier_sha256", "config")}
    return {"reviewers": out, "provenance": provenance,
            "split_sizes": {s: len(c) for s, c in cases.items()}}


# ── Policy comparison and safety ─────────────────────────────────────────

def comparison_section(*, seeds: int, train_episodes: int, benchmark_seed: int,
                       resamples: int) -> tuple[dict, dict]:
    bench = synthetic.SyntheticBenchmark(seed=benchmark_seed)
    encoder = synthetic.synthetic_encoder(bench)
    result = compare.compare(
        bench, synthetic.ScriptedExecutor(bench.response_table()), synthetic.synthetic_quality,
        synthetic.synthetic_catalogue(), encoder,
        compare.default_policy_specs(synthetic.STRATEGIES, encoder.dim),
        seeds=list(range(seeds)), train_episodes=train_episodes,
        scoring=synthetic.REFERENCE_SCORING, n_resamples=resamples,
        skill_version="synthetic-catalogue/1")
    counts: dict = {}
    for r in result.records:
        entry = counts.setdefault(r.policy, {}).setdefault(r.phase, Counter())
        entry["steps"] += 1
        entry["violations"] += int(r.violation)
    violations = {p: {ph: dict(c) for ph, c in phases.items()} for p, phases in counts.items()}
    return result.report, violations


def headline(comparison: Mapping) -> list[str]:
    """Plain-language findings derived from the comparison verdicts."""
    best = comparison["best_fixed_baseline"]
    rows = comparison["comparisons"]
    lines = []

    def pick(baseline, phase):
        return [c for c in rows if c["baseline"] == baseline and c["phase"] == phase]

    def fmt(c):
        return f"{c['policy']} {c['mean_diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"

    val = pick(best, "validation")
    better = [c for c in val if c["verdict"] == "improvement"]
    if val and not better:
        lines.append(f"No adaptive policy shows an improvement over the best fixed strategy "
                     f"(`{best}`) on held-out validation: "
                     + "; ".join(f"{fmt(c)} ({c['verdict']})" for c in val) + ".")
    elif better:
        lines.append(f"On validation, {', '.join(c['policy'] for c in better)} improved on "
                     f"the best fixed strategy (`{best}`): " + "; ".join(map(fmt, better)) + ".")
    train = [c for c in pick(best, "train") if c["verdict"] == "improvement"]
    if train:
        lines.append("On train, " + "; ".join(map(fmt, train)) + f" against `{best}`. That "
                     "baseline was chosen in hindsight on the same train data, so this does "
                     "not show generalisation.")
    rand = [c for c in rows if c["baseline"] == "random"]
    if rand and all(c["verdict"] == "improvement" for c in rand):
        lines.append("Every adaptive policy beats uniform random selection in both phases.")
    lines.append("These are results from a hypothetical simulator, not from Hermes.")
    return lines


def _signed(iv: Mapping) -> str:
    return f"{iv['point']:+.3f} [{iv['low']:.3f}, {iv['high']:.3f}]"


def _live_parts(live: Mapping, live_run: Mapping, final: Mapping) -> list[str]:
    parts = []
    if live_run["status"] == "supplied":
        parts.append(f"the live full run ({live_run['run_id']})")
    if final.get("design") == "paired" and final.get("status") == "completed":
        parts.append("the one-time sealed final evaluation")
    if live["status"] == "supplied":
        parts.append(f"a {live['n_cases']}-case smoke test")
    return parts


def _join(parts: Sequence[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def report_notice(live: Mapping, live_run: Mapping, final: Mapping) -> str:
    parts = _live_parts(live, live_run, final)
    if not parts:
        return NOTICE
    joined = _join(parts)
    model = (live_run.get("identity") or {}).get("model") if live_run["status"] == \
        "supplied" else None
    return (f"{SIMULATOR_NOTICE} {_sentence(joined)} used live Hermes"
            + (f" ({model})" if model else "")
            + " on the hand-written, synthetic python_review benchmark. " + NO_REAL_WORLD)


def lead_findings(live_run: Mapping, final: Mapping) -> list[str]:
    """Headline lines for the live results, sealed result first."""
    lines = []
    if final.get("design") == "paired" and final.get("status") == "completed":
        d, sv = final["paired"]["strict"], final["skill_versions"]
        f1, prec, rec = d["micro_f1"], d["micro_precision"], d["micro_recall"]
        rule = ("above 0, so the pre-registered rule is met" if f1["low"] > 0
                else "not above 0, so the pre-registered rule is not met")
        driver = ("mostly precision" if prec["point"] > rec["point"] else
                  "mostly recall" if rec["point"] > prec["point"] else
                  "precision and recall alike")
        lines.append(
            f"Pre-registered sealed final evaluation (run once, live Hermes, "
            f"{final['n_cases']} hand-written synthetic test cases): candidate "
            f"`{sv['candidate']}` minus baseline `{sv['baseline']}`, STRICT micro-F1 "
            f"{_signed(f1)} (primary; CI lower bound {rule}), {driver} (precision "
            f"{_signed(prec)}, recall {_signed(rec)}); LENIENT micro-F1 "
            f"{_signed(final['paired']['lenient']['micro_f1'])} (secondary).")
    if live_run["status"] == "supplied":
        c = live_run["compare"]
        best = c["best_fixed_baseline"]
        val = [x for x in c["comparisons"] if x["baseline"] == best
               and x["phase"] == "validation"]
        better = [x for x in val if x["verdict"] == "improvement"]
        lines.append(
            f"Live full run {live_run['run_id']} ({len(c['config']['strategies'])}-arm "
            f"comparison on live Hermes outcomes, validation split): the best fixed strategy "
            f"`{best}` has utility {c['policies'][best]['validation']['utility']['mean']:.3f}; "
            + ("no adaptive policy improves on it: " if not better else "improved on it: ")
            + "; ".join(f"{x['policy']} {x['mean_diff']:+.3f} [{x['ci_low']:+.3f}, "
                        f"{x['ci_high']:+.3f}] ({x['verdict']})" for x in val) + ".")
        a = live_run["admission"]
        dec = a["decision"]
        lines.append(
            f"GEPA candidate `{a['candidate_id']}` (parent `{a['parent_id']}`) was "
            f"{'admitted' if dec['admitted'] else 'rejected'} by gate 6 with a validation "
            f"utility gain of {dec['mean_utility_gain']:+.3f} [{dec['ci_low']:+.3f}, "
            f"{dec['ci_high']:+.3f}]; gate 7's post-admission check then "
            + ("kept it" if a["status"] == "kept" else f"ended `{a['status']}`")
            + f" (catalogue validation utility {a['pre_validation']['utility']:.3f} -> "
            + (f"{a['post_validation']['utility']:.3f})." if a.get("post_validation")
               else "n/a)."))
    return lines


def compose_headline(comparison: Mapping, live: Mapping, live_run: Mapping,
                     final: Mapping) -> list[str]:
    sim = headline(comparison)
    parts = _live_parts(live, live_run, final)
    if not parts:
        return sim
    lines = lead_findings(live_run, final) + [f"Simulator: {sim[0]}"] + sim[1:-1]
    if live["status"] == "supplied":
        lines.append(f"A live Hermes smoke run scored {live['n_cases']} {live['split']} cases "
                     f"({live['completed']} completed, {live['violations']} violations). With "
                     "so few cases this is a plumbing check, not a quality estimate.")
    lines.append("The simulator findings are results from a hypothetical simulator, not from "
                 "Hermes. " + _sentence(_join(parts)) + " used live Hermes on a "
                 "hand-written, synthetic benchmark, so none of this is evidence of "
                 "real-world improvement.")
    return lines


def _registry_summary(path: Optional[str]) -> dict:
    if path is None:
        return {"status": "not supplied", "note": "no candidate registry was given "
                "(--registry); candidate counts are unknown, not zero"}
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        by = conn.execute("SELECT source, status, COUNT(*) FROM candidates "
                          "GROUP BY source, status ORDER BY source, status").fetchall()
        moves = conn.execute(
            "SELECT candidate_id, from_status, to_status, reason, decided_by FROM transitions "
            "WHERE to_status IN ('rejected', 'retired') ORDER BY id").fetchall()
        lineage = conn.execute("SELECT candidate_id, parent_id, source, status FROM candidates "
                               "ORDER BY created_at, candidate_id").fetchall()
        history = conn.execute("SELECT candidate_id, from_status, to_status, decided_by "
                               "FROM transitions ORDER BY id").fetchall()
    finally:
        conn.close()
    return {"status": "supplied",
            "candidates": [{"source": s, "status": st, "count": n} for s, st, n in by],
            "lineage": [dict(zip(("candidate_id", "parent_id", "source", "status"), r))
                        for r in lineage],
            "transitions": [dict(zip(("candidate_id", "from", "to", "decided_by"), r))
                            for r in history],
            "rejected": [dict(zip(("candidate_id", "from", "to", "reason", "decided_by"), m))
                         for m in moves if m[2] == "rejected"],
            "retired": [dict(zip(("candidate_id", "from", "to", "reason", "decided_by"), m))
                        for m in moves if m[2] == "retired"]}


def admission_section() -> dict:
    """The fixed gate 6 admission policy (thresholds are code, not arguments)."""
    policy = admission.ADMISSION_POLICY
    return {"status": "merged", "policy": policy.to_dict(),
            "policy_sha256": policy.fingerprint(),
            "rules": [f"zero critical violations for the candidate over all runs",
                      f"mean utility gain over the baseline of at least {policy.min_margin}",
                      f"lower bound of the {int(policy.confidence * 100)}% paired bootstrap "
                      "CI of that gain above 0"],
            "note": "decisions are recorded in the candidate registry (above; from the live "
                    "full-run snapshot unless --registry is given); the live run's decision "
                    "and its evidence are in the live full run section"}


def checkpoint_section(path: Optional[str]) -> dict:
    """Read-only summary of a gate 7 checkpoint store (statuses and log events)."""
    if path is None:
        return {"status": "not supplied", "note": NO_CHECKPOINT_STORE}
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        statuses = conn.execute("SELECT status, COUNT(*) FROM policy_checkpoints "
                                "GROUP BY status ORDER BY status").fetchall()
        events = conn.execute("SELECT event, COUNT(*) FROM checkpoint_log "
                              "GROUP BY event ORDER BY event").fetchall()
        rollbacks = conn.execute(
            "SELECT checkpoint_id, from_checkpoint_id, reason FROM checkpoint_log "
            "WHERE event='rollback' ORDER BY id").fetchall()
        log = conn.execute("SELECT event, checkpoint_id, from_checkpoint_id, reason "
                           "FROM checkpoint_log ORDER BY id").fetchall()
    finally:
        conn.close()
    return {"status": "supplied",
            "checkpoints": {k: n for k, n in statuses},
            "events": {k: n for k, n in events},
            "log": [dict(zip(("event", "checkpoint_id", "from", "reason"), r)) for r in log],
            "rollbacks": [{"to": t, "from": f, "reason": r} for t, f, r in rollbacks]}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_spend(ledger_path: Path) -> dict:
    """Per-stage call counts, cost and latency from a full-run spend ledger.

    Aggregates only: model outputs (which include replies on sealed test cases
    for the final evaluation stages) are never copied.
    """
    stages: dict[str, dict] = {}
    total = 0.0
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        st = stages.setdefault(row.get("stage", "?"), {
            "calls": 0, "errors": 0, "unknown_cost_calls": 0, "retries": 0,
            "cost_usd": 0.0, "latencies": []})
        st["calls"] += 1
        st["errors"] += bool(row.get("error"))
        st["retries"] += bool(row.get("attempt"))
        if row.get("cost_usd") is None:
            st["unknown_cost_calls"] += 1
        else:
            st["cost_usd"] += row["cost_usd"]
            total += row["cost_usd"]
        if row.get("latency_s") is not None:
            st["latencies"].append(row["latency_s"])
    for st in stages.values():
        lat = st.pop("latencies")
        st["cost_usd"] = round(st["cost_usd"], 6)
        st["mean_latency_s"] = round(statistics.fmean(lat), 3) if lat else None
    return {"by_stage": dict(sorted(stages.items())), "total_cost_usd": round(total, 6)}


def snapshot_live_run(run_dir: str | Path) -> dict:
    """Read-only, aggregate snapshot of a ``full_run`` run directory for the report."""
    run = Path(run_dir)
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    rep = json.loads((run / "report.json").read_text(encoding="utf-8"))
    stages = rep["stages"]
    cmp_ = stages["compare"]["result"]["report"]
    gepa = stages["gepa"]["result"]
    adm = stages["admission"]["result"]
    files = ["state.json", "report.json", "registry.sqlite", "checkpoints.sqlite",
             "ledger.jsonl", "gepa_log/candidates.json", "gepa_log/run_log.json"]
    texts = [c["self"] for c in json.loads(
        (run / "gepa_log/candidates.json").read_text(encoding="utf-8"))]
    parents = {e["new_program_idx"]: {"parent": e.get("selected_program_candidate"),
                                      "merge": bool(e.get("invoked_merge"))}
               for e in json.loads((run / "gepa_log/run_log.json").read_text(encoding="utf-8"))
               if isinstance(e, dict) and "new_program_idx" in e}
    uri = (run / "registry.sqlite").resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        registered = dict(conn.execute(
            "SELECT candidate_id, content FROM candidates WHERE source = 'gepa'").fetchall())
    finally:
        conn.close()
    candidates = []
    for i, text in enumerate(texts):
        hits = sorted(cid for cid, content in registered.items()
                      if text.strip() and text.strip() in content.decode("utf-8"))
        candidates.append({"index": i, "chars": len(text),
                           "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
                           "parent": parents.get(i, {}).get("parent"),
                           "merge": parents.get(i, {}).get("merge", False),
                           "seed": i == 0, "registered_as": hits[0] if hits else None})
    decision = {k: v for k, v in adm["decision"].items() if k != "case_ids"}
    return {
        "schema": LIVE_RUN_SCHEMA, "run_id": run.name,
        "source_files": {f: _file_sha256(run / f) for f in files},
        "identity": state["identity"], "baseline_id": state["baseline_id"],
        "skill_source": rep.get("skill_source"), "splits_used": rep.get("splits_used"),
        "sealed_split_open_attempts": state.get("sealed_split_open_attempts"),
        "stages": {k: {"status": v.get("status"), "seconds": v.get("seconds")}
                   for k, v in stages.items()},
        "canary": {k: stages["canary"]["result"][k] for k in ("fresh_calls", "replayed_calls")}
        | {"cases": [{k: c[k] for k in ("case_id", "f1", "cost_usd", "parse_ok")}
                     for c in stages["canary"]["result"]["cases"]]},
        "compare": {"best_fixed_baseline": cmp_["best_fixed_baseline"],
                    "executor_backend": cmp_["executor_backend"],
                    "executions": cmp_["executions"],
                    "fresh_calls": stages["compare"]["result"]["fresh_calls"],
                    "replayed_calls": stages["compare"]["result"]["replayed_calls"],
                    "config": {k: cmp_["config"][k] for k in (
                        "strategies", "train_episodes", "train_items", "validation_items",
                        "bootstrap", "validation_learning")}
                    | {"seeds": len(cmp_["config"]["seeds"])},
                    "policies": cmp_["policies"], "comparisons": cmp_["comparisons"]},
        "gepa": {"metrics": gepa["metrics"], "candidates": candidates},
        "admission": {"status": adm["status"], "reason": adm["reason"],
                      "candidate_id": adm["candidate_id"], "parent_id": adm["parent_id"],
                      "active_skill": adm["active_skill"],
                      "registry_status": adm["registry_status"], "decision": decision,
                      "pre_validation": adm["pre_validation"],
                      "post_validation": adm["post_validation"]},
        "registry": _registry_summary(str(run / "registry.sqlite")),
        "checkpoints": checkpoint_section(str(run / "checkpoints.sqlite")),
        "spend": _stage_spend(run / "ledger.jsonl"),
    }


def live_run_section(path: Optional[str]) -> dict:
    if path is None or not Path(path).exists():
        return {"status": "not supplied", "note": "no live full-run snapshot supplied "
                "(--live-run)"}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != LIVE_RUN_SCHEMA:
        raise ValueError(f"{Path(path).name} is not a {LIVE_RUN_SCHEMA} snapshot")
    return {"status": "supplied", "file": Path(path).name, **data}


def _finite_or_none(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"expected a finite number or null, got {value!r}")
    return float(value)


def live_smoke_section(path: Optional[str]) -> dict:
    """Optional live Hermes smoke results: JSON {source, cases: [{case_id, completed,
    quality?, cost_usd?, latency_s?, violation}]}. A handful of cases only."""
    if path is None:
        return {"status": "pending", "note": "no live Hermes smoke results supplied "
                "(--live-smoke); nothing in this report comes from a live Hermes run"}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("live smoke file needs a nonempty 'cases' list")
    rows = []
    for c in cases:
        if not isinstance(c, dict) or not isinstance(c.get("case_id"), str):
            raise ValueError("each live smoke case needs a case_id")
        rows.append({"case_id": c["case_id"], "completed": bool(c.get("completed")),
                     "quality": _finite_or_none(c.get("quality")),
                     "cost_usd": _finite_or_none(c.get("cost_usd")),
                     "latency_s": _finite_or_none(c.get("latency_s")),
                     "violation": bool(c.get("violation"))})

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    return {"status": "supplied", "source": str(data.get("source", "unspecified")),
            "model": str(data.get("model", "unspecified")),
            "split": str(data.get("split", "unspecified")),
            "note": str(data.get("note", "")),
            "label": f"{len(rows)} live cases; a smoke test, not a benchmark result",
            "n_cases": len(rows), "completed": sum(r["completed"] for r in rows),
            "violations": sum(r["violation"] for r in rows),
            "mean_quality": mean("quality"), "mean_cost_usd": mean("cost_usd"),
            "mean_latency_s": mean("latency_s"), "cases": rows}


def final_evaluation_section(ledger_path: Path) -> dict:
    if not ledger_path.exists():
        return {"status": "not run", "note": "the one-time final test evaluation has not "
                "been run; when and on which reviewer is an owner decision"}
    record = json.loads(ledger_path.read_text(encoding="utf-8"))
    if record.get("design") == "paired":
        return _paired_section(record)
    keep = ("status", "started_at", "finished_at", "reviewer_id", "skill_version",
            "synthetic", "n_cases", "totals", "micro", "macro", "benchmark", "verifier",
            "error")
    out = {k: record[k] for k in keep if k in record}
    summary = record.get("reviewer_summary")
    if isinstance(summary, dict):
        out["reviewer_summary"] = {k: v for k, v in summary.items() if k != "cases"}
    return out


def _paired_section(record: Mapping) -> dict:
    """The paired (schema /2) ledger: scores and differences, no per-case rows."""
    keep = ("status", "started_at", "finished_at", "design", "reviewer_id", "synthetic",
            "skill_versions", "analysis_plan", "analysis_plan_sha256", "primary",
            "secondary", "n_cases", "paired", "benchmark", "verifier", "error")
    out = {k: record[k] for k in keep if k in record}
    alt = record.get("alternates") or {}
    out["alternates"] = {k: alt.get(k) for k in ("table", "sha256", "n_defects")}
    if "lenient_defects_in_split" in record:
        out["alternates"]["defects_in_split"] = record["lenient_defects_in_split"]
    if isinstance(record.get("scores"), dict):
        out["scores"] = {name: {arm: {k: v for k, v in arm_rec.items() if k != "cases"}
                                for arm, arm_rec in by_arm.items()}
                         for name, by_arm in record["scores"].items()}
    summaries = record.get("reviewer_summaries")
    if isinstance(summaries, dict):
        out["reviewer_summaries"] = {arm: {k: v for k, v in rs.items() if k != "cases"}
                                     for arm, rs in summaries.items() if isinstance(rs, dict)}
    return out


def _paired_markdown(fe: Mapping) -> list[str]:
    sv = fe.get("skill_versions", {})
    L = [f"Status **{fe['status']}**, paired design, started {fe.get('started_at')}, "
         f"finished {fe.get('finished_at')}, reviewer `{fe.get('reviewer_id')}`, synthetic: "
         f"{fe.get('synthetic')}. Baseline `{sv.get('baseline')}`, candidate "
         f"`{sv.get('candidate')}`. Pre-registered plan `{fe['analysis_plan']['id']}` "
         f"(sha256 `{fe.get('analysis_plan_sha256', '')[:12]}`); lenient table "
         f"`{fe['alternates'].get('table')}` (sha256 "
         f"`{(fe['alternates'].get('sha256') or '')[:12]}`)."]
    for arm, rs in (fe.get("reviewer_summaries") or {}).items():
        if "error" in rs:
            continue
        L += ["", f"- {arm}: model `{rs.get('model')}` (provider `{rs.get('provider')}`), "
              f"strategy `{rs.get('strategy_id')}`, cost USD {rs.get('total_cost_usd')} "
              f"(statuses {rs.get('cost_statuses')}, shared cap "
              f"{rs.get('max_total_cost_usd')}), calls {rs.get('calls')}, completed "
              f"{rs.get('completed')}, unparseable {rs.get('unparseable')}, violations "
              f"{rs.get('violations')}, exceptions {rs.get('exceptions')}, skipped "
              f"{rs.get('skipped')}" + (f", stopped: {rs['stopped']}" if rs.get("stopped")
                                        else "") + "."]
    if fe["status"] == "completed":
        L += ["", f"{fe['n_cases']} sealed test cases, the same for both skills.", "",
              "| Scoring | Skill | Micro F1 | Micro precision | Micro recall | Macro F1 |",
              "|---|---|---|---|---|---|"]
        for name, label in (("strict", "STRICT (primary)"), ("lenient", "LENIENT (secondary)")):
            for arm in ("baseline", "candidate"):
                sc = fe["scores"][name][arm]
                L.append(f"| {label} | {arm} | {_iv(sc['micro']['f1'])} | "
                         f"{_iv(sc['micro']['precision'])} | {_iv(sc['micro']['recall'])} | "
                         f"{_iv(sc['macro']['f1'])} |")
        L += ["", "Paired difference, candidate minus baseline, 95% paired bootstrap CI:", "",
              "| Scoring | Micro F1 | Macro F1 | Micro precision | Micro recall |",
              "|---|---|---|---|---|"]
        for name, label in (("strict", "STRICT (primary)"), ("lenient", "LENIENT (secondary)")):
            d = fe["paired"][name]
            L.append(f"| {label} | {_iv(d['micro_f1'])} | {_iv(d['macro_f1'])} | "
                     f"{_iv(d['micro_precision'])} | {_iv(d['micro_recall'])} |")
        primary = fe["paired"]["strict"]["micro_f1"]
        verdict = ("above 0: the candidate is better on the sealed test under the "
                   "pre-registered rule" if primary["low"] > 0 else
                   "not above 0: no improvement shown on the sealed test under the "
                   "pre-registered rule")
        L += ["", f"Primary result: the strict micro-F1 difference's CI lower bound is "
              f"{verdict}. Synthetic hand-written cases: not evidence of a real Hermes "
              "improvement."]
    elif fe.get("error"):
        L += ["", f"Error: {fe['error']}"]
    return L


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


# ── Generation ───────────────────────────────────────────────────────────

def generate(*, seeds: int = 20, train_episodes: int = 300, benchmark_seed: int = 0,
             resamples: int = 2000, verifier_seed: int = 0, verifier_bootstrap: int = 1000,
             registry_path: Optional[str] = None, checkpoint_path: Optional[str] = None,
             live_smoke_path: Optional[str] = str(DEFAULT_LIVE_SMOKE),
             live_run_path: Optional[str] = str(DEFAULT_LIVE_RUN),
             final_ledger_path: Path = DEFAULT_LEDGER,
             generated_at: Optional[str] = None) -> dict:
    """Build the report dict. Runs entirely inside optimisation_phase().

    Registry and checkpoint history come from ``registry_path`` and
    ``checkpoint_path`` if given, otherwise from the live-run snapshot.
    """
    with optimisation_phase():
        attempts_before = sealed_access_attempts()
        comparison, violations = comparison_section(
            seeds=seeds, train_episodes=train_episodes, benchmark_seed=benchmark_seed,
            resamples=resamples)
        verifier_scores = verifier_section(seed=verifier_seed, n_bootstrap=verifier_bootstrap)
        live_run = live_run_section(live_run_path)
        from_run = live_run["status"] == "supplied"
        registry = (live_run["registry"] if registry_path is None and from_run
                    else _registry_summary(registry_path))
        checkpoints = (live_run["checkpoints"] if checkpoint_path is None and from_run
                       else checkpoint_section(checkpoint_path))
        live = live_smoke_section(live_smoke_path)
        final = final_evaluation_section(Path(final_ledger_path))
        blocked = sealed_access_attempts() - attempts_before
    total_steps = sum(c["steps"] for p in violations.values() for c in p.values())
    total_viol = sum(c["violations"] for p in violations.values() for c in p.values())
    pending = []
    if registry["status"] == "not supplied":
        pending.append("admission and rejection history: no candidate registry supplied; "
                       "no admission review has been run outside the test suite")
    if checkpoints["status"] == "not supplied":
        pending.append("rollback history: none yet, because no admission has run")
    if live["status"] == "pending":
        pending.append("live Hermes smoke run: no results supplied")
    if final["status"] == "not run":
        pending.append("final test evaluation: not run (owner decision)")
    return {
        "schema": SCHEMA, "title": "KAFO-EVO-001 sprint report", "synthetic": True,
        "notice": report_notice(live, live_run, final),
        "generated_at": generated_at,
        "settings": {"seeds": seeds, "train_episodes": train_episodes,
                     "benchmark_seed": benchmark_seed, "comparison_resamples": resamples,
                     "verifier_seed": verifier_seed, "verifier_resamples": verifier_bootstrap,
                     "registry": None if registry_path is None else Path(registry_path).name,
                     "checkpoints": None if checkpoint_path is None
                     else Path(checkpoint_path).name,
                     "live_smoke": None if live_smoke_path is None else Path(live_smoke_path).name,
                     "live_run": live_run.get("file"),
                     "final_ledger": _rel(Path(final_ledger_path))},
        "provenance": {"benchmark": {"name": benchmark.BENCHMARK_NAME,
                                     "version": benchmark.DEFAULT_VERSION,
                                     "manifest_sha256": benchmark.PINNED_MANIFEST_SHA256.get(
                                         benchmark.DEFAULT_VERSION)},
                       "verifier": verifier_scores.pop("provenance")},
        "headline": compose_headline(comparison, live, live_run, final),
        "policy_comparison": comparison,
        "verifier_scores": verifier_scores,
        "safety": {"critical_violations": {"total_steps": total_steps,
                                           "total_violations": total_viol,
                                           "by_policy": violations},
                   "sealed_access_blocked_during_generation": blocked,
                   "registry": registry,
                   "admission": admission_section(),
                   "rollbacks": checkpoints},
        "live_smoke": live,
        "live_run": live_run,
        "final_test_evaluation": final,
        "pending": pending,
    }


# ── Markdown ─────────────────────────────────────────────────────────────

def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def _ci(stat: Mapping, fmt: str = ".3f") -> str:
    return f"{stat['mean']:{fmt}} [{stat['ci_low']:{fmt}}, {stat['ci_high']:{fmt}}]"


def _iv(iv: Mapping) -> str:
    return f"{iv['point']:.3f} [{iv['low']:.3f}, {iv['high']:.3f}]"


def _live_run_markdown(run: Mapping) -> list[str]:
    if run["status"] != "supplied":
        return []
    ident, c, g, a = run["identity"], run["compare"], run["gepa"], run["admission"]
    cfg, m, dec = c["config"], g["metrics"], a["decision"]
    L = [f"## Live full run ({run['run_id']})", "",
         f"**Live Hermes on the hand-written, synthetic benchmark: not evidence of "
         f"real-world improvement.** Model `{ident['model']}`, provider "
         f"`{ident['provider']}`, budget {ident['budget']['max_iterations']} iterations / "
         f"{ident['budget']['timeout_s']:.0f} s per call, run `{ident['run_version']}`. "
         f"Baseline skill `{run['baseline_id']}` from `{run['skill_source']}`. Splits used: "
         f"{', '.join(run['splits_used'])}; sealed split open attempts: "
         f"{run['sealed_split_open_attempts']}. Snapshot `{run['file']}` (source file "
         "SHA-256s recorded in it).", "",
         "Stages: " + ", ".join(f"{k} {v['status']} ({v['seconds']:.0f} s)"
                                for k, v in run["stages"].items()) + ".", "",
         "### Live policy comparison (gate 4 on live outcomes)", "",
         f"{len(cfg['strategies'])} arms ({', '.join(cfg['strategies'])}). Each (case, "
         f"strategy) pair ran once on live Hermes ({c['fresh_calls']} fresh calls, "
         f"{c['replayed_calls']} reused from the canary); {cfg['seeds']} seeds x "
         f"{cfg['train_episodes']} train episodes over {cfg['train_items']} train cases replay "
         f"those outcomes, then one pass over {cfg['validation_items']} validation cases with "
         f"learning off. {int(cfg['bootstrap']['confidence'] * 100)}% bootstrap CI over seeds "
         f"({cfg['bootstrap']['resamples']} resamples).", "",
         "| Policy | Validation utility | Validation quality | Validation cost USD | "
         "Validation latency s |", "|---|---|---|---|---|"]
    for name, pol in c["policies"].items():
        v = pol["validation"]
        L.append(f"| {name} | {_ci(v['utility'])} | {_ci(v['quality'])} | "
                 f"{_ci(v['cost_usd'], '.4f')} | {_ci(v['latency_s'], '.2f')} |")
    L += ["", f"Best fixed baseline, chosen by train utility: `{c['best_fixed_baseline']}`.", "",
          "| Policy | Versus | Phase | Difference [CI] | Verdict |", "|---|---|---|---|---|"]
    L += [f"| {x['policy']} | {x['baseline']} | {x['phase']} | {x['mean_diff']:+.3f} "
          f"[{x['ci_low']:+.3f}, {x['ci_high']:+.3f}] | {x['verdict']} |"
          for x in c["comparisons"]]
    L += ["", "### GEPA (gate 5 input)", "",
          f"`{m['optimizer']}` on {m['data']} ({m['gepa_train_cases']} GEPA-train / "
          f"{m['gepa_val_cases']} GEPA-val cases), max {m['max_metric_calls']} metric calls, "
          f"seed {m['seed']}, task and reflection calls through "
          f"`{m['lm_route']['task']}`. GEPA's own validation score {m['baseline_score']:.3f} -> "
          f"{m['evolved_score']:.3f}; skill size {m['baseline_size']} -> "
          f"{m['evolved_size']} characters; constraints passed: {m['constraints_passed']}; "
          f"{m['elapsed_seconds']:.0f} s.", "",
          "| GEPA program | Parent | Merge | Characters | SHA-256 | Registered as |",
          "|---|---|---|---|---|---|"]
    for cand in g["candidates"]:
        L.append(f"| {cand['index']}{' (seed)' if cand['seed'] else ''} | "
                 f"{'-' if cand['parent'] is None else cand['parent']} | "
                 f"{'yes' if cand['merge'] else 'no'} | {cand['chars']} | "
                 f"`{cand['sha256']}` | "
                 + (f"`{cand['registered_as']}`" if cand["registered_as"] else "-") + " |")
    pol = dec["policy"]
    L += ["", "### Admission and checkpoint (gates 6 and 7)", "",
          f"Candidate `{a['candidate_id']}` (parent `{a['parent_id']}`): "
          f"**{'admitted' if dec['admitted'] else 'rejected'}** by gate 6, policy "
          f"`{dec['policy_sha256'][:12]}` ({pol['split']} split, {dec['n_cases']} cases, seeds "
          f"{pol['seeds']}, {pol['n_bootstrap']} resamples). Mean utility gain "
          f"{dec['mean_utility_gain']:+.3f} [{dec['ci_low']:+.3f}, {dec['ci_high']:+.3f}] "
          f"(needs at least {pol['min_margin']} and CI above 0); mean quality "
          f"{dec['baseline_mean_quality']:.3f} -> {dec['candidate_mean_quality']:.3f}; "
          f"violations {dec['baseline_violations']} / {dec['candidate_violations']}; "
          f"unparseable outputs {dec['unparseable_outputs']}.",
          "", f"Gate 7: catalogue validation utility {a['pre_validation']['utility']:.3f} "
          f"before admission -> {a['post_validation']['utility']:.3f} after "
          f"({a['post_validation']['n_tasks']} tasks, critical violation: "
          f"{a['post_validation']['critical_violation']}); {a['reason']} (status "
          f"`{a['status']}`, registry `{a['registry_status']}`, active skill "
          f"`{a['active_skill']}`).", "",
          "### Spend by stage (live Hermes estimates)", "",
          "| Stage | Calls | Cost USD | Errors | Unknown cost | Retries | Mean latency s |",
          "|---|---|---|---|---|---|---|"]
    for name, st in run["spend"]["by_stage"].items():
        L.append(f"| {name} | {st['calls']} | {st['cost_usd']:.4f} | {st['errors']} | "
                 f"{st['unknown_cost_calls']} | {st['retries']} | "
                 + ("n/a" if st["mean_latency_s"] is None else f"{st['mean_latency_s']:.2f}")
                 + " |")
    L += ["", f"Total USD {run['spend']['total_cost_usd']:.4f} (Hermes' estimates at API rates; "
          "the provider bills a subscription). The final-evaluation stages are the one-time "
          "sealed run reported below.", ""]
    return L


def format_markdown(report: Mapping) -> str:
    cmp_ = report["policy_comparison"]
    cfg = cmp_["config"]
    s = report["settings"]
    L = [f"# {report['title']}", "", f"> **{report['notice']}**", ""]
    if report.get("generated_at"):
        L += [f"Generated: {report['generated_at']}", ""]
    L += ["## Headline findings", ""] + [f"- {h}" for h in report["headline"]] + [""]

    L += ["## Policy comparison (gate 4, synthetic simulator)", "",
          f"Benchmark `{cmp_['benchmark']}`, executor `{cmp_['executor_backend']}`. "
          f"{len(cfg['seeds'])} seeds x {cfg['train_episodes']} online train episodes, then "
          f"one pass over {cfg['validation_items']} validation tasks with learning off. "
          f"Mean over seeds with a {int(cfg['bootstrap']['confidence'] * 100)}% bootstrap CI "
          f"({cfg['bootstrap']['resamples']} resamples, unit: seed-level run mean).", "",
          "### Quality and utility", "",
          "| Policy | Train utility | Validation utility | Validation quality |",
          "|---|---|---|---|"]
    for name, p in cmp_["policies"].items():
        L.append(f"| {name} | {_ci(p['train']['utility'])} | {_ci(p['validation']['utility'])}"
                 f" | {_ci(p['validation']['quality'])} |")
    L += ["", "Utility = verified quality - 0.10 x min(cost ratio, 2) - 0.05 x min(latency "
          "ratio, 2); a critical violation scores -1 (`core.score_outcome`, unchanged).", "",
          "### Paired utility difference", "",
          f"Best fixed baseline, chosen by train utility: `{cmp_['best_fixed_baseline']}`.", "",
          "| Policy | Versus | Phase | Difference [CI] | Verdict |", "|---|---|---|---|---|"]
    L += [f"| {c['policy']} | {c['baseline']} | {c['phase']} | {c['mean_diff']:+.3f} "
          f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | {c['verdict']} |"
          for c in cmp_["comparisons"]]
    L += ["", "## Cost and latency (simulated)", "",
          "Invented per-strategy costs (USD) and latencies (seconds) from `synthetic.py`.", "",
          "| Policy | Validation cost USD | Validation latency s | Train cost USD |",
          "|---|---|---|---|"]
    for name, p in cmp_["policies"].items():
        L.append(f"| {name} | {_ci(p['validation']['cost_usd'], '.4f')} | "
                 f"{_ci(p['validation']['latency_s'], '.2f')} | "
                 f"{_ci(p['train']['cost_usd'], '.4f')} |")

    vs = report["verifier_scores"]
    pv = report["provenance"]
    L += ["", "## Verifier scores on the open benchmark splits (gates 2 and 3)", "",
          f"Benchmark `{pv['benchmark']['name']}` {pv['benchmark']['version']} "
          f"(manifest `{(pv['benchmark']['manifest_sha256'] or '?')[:12]}`), split sizes "
          + ", ".join(f"{k} {v}" for k, v in vs["split_sizes"].items())
          + f". Verifier `{pv['verifier']['verifier_version']}`, rule "
          f"`{pv['verifier']['match_rule']}`, source `{pv['verifier']['verifier_sha256'][:12]}`. "
          f"Case-level bootstrap CI, seed {s['verifier_seed']}, "
          f"{s['verifier_resamples']} resamples.", "",
          "These reviews are references, not models: `ground_truth` reports exactly the "
          "labelled defects (ceiling), `empty` reports nothing (floor), and `noisy_reference` "
          "is an invented, seeded perturbation of the labels (keeps each defect with p=0.7, "
          "shifts lines by up to 1, adds a false alarm with p=0.25).", "",
          "| Review | Split | Cases | TP / FP / FN | Micro precision | Micro recall | "
          "Micro F1 | Macro F1 |", "|---|---|---|---|---|---|---|---|"]
    for name, splits in vs["reviewers"].items():
        for split, r in splits.items():
            t = r["totals"]
            L.append(f"| {name} | {split} | {r['n_cases']} | {t['true_positives']} / "
                     f"{t['false_positives']} / {t['false_negatives']} | "
                     f"{_iv(r['micro']['precision'])} | {_iv(r['micro']['recall'])} | "
                     f"{_iv(r['micro']['f1'])} | {_iv(r['macro']['f1'])} |")
    L += ["", "The empty review's macro F1 is above 0 because clean control cases (no "
          "defects) score 1.0 when nothing is reported."]

    L += ["", "## Uncertainty", "",
          "- Policy comparison: percentile bootstrap over seed-level means; paired "
          "per-seed differences for verdicts (`improvement` only if the whole CI is above "
          "0). Validation intervals for fixed policies have zero width because those runs "
          "are deterministic.",
          "- Verifier: percentile bootstrap over cases (micro pools TP/FP/FN, macro "
          "averages per-case scores).",
          "- Same code and seeds reproduce this report byte for byte.", ""]

    sf = report["safety"]
    cv = sf["critical_violations"]
    L += ["## Safety", "",
          f"- Critical violations (tool call outside the strategy allowlist, simulated): "
          f"{cv['total_violations']} in {cv['total_steps']} simulated policy decisions "
          "(all seeds and phases). Each scores -1 and disqualifies the outcome.",
          f"- Sealed test split: {sf['sealed_access_blocked_during_generation']} blocked "
          "access attempts while generating this report (everything ran inside "
          "`optimisation_phase()`).", "",
          "| Policy | Train violations / steps | Validation violations / steps |",
          "|---|---|---|"]
    for name, phases in cv["by_policy"].items():
        cells = [f"{phases[ph]['violations']} / {phases[ph]['steps']}" if ph in phases else "-"
                 for ph in ("train", "validation")]
        L.append(f"| {name} | {cells[0]} | {cells[1]} |")
    reg = sf["registry"]
    L += ["", "### Candidate registry (gate 5)", ""]
    if reg["status"] == "supplied":
        L += ["| Source | Status | Candidates |", "|---|---|---|"]
        L += [f"| {c['source']} | {c['status']} | {c['count']} |" for c in reg["candidates"]]
        if reg.get("lineage"):
            L += ["", "| Candidate | Parent | Source | Status |", "|---|---|---|---|"]
            L += [f"| `{c['candidate_id']}` | " + (f"`{c['parent_id']}`" if c["parent_id"]
                                                  else "-")
                  + f" | {c['source']} | {c['status']} |" for c in reg["lineage"]]
            L += ["", "Transitions: " + "; ".join(
                f"`{t['candidate_id']}` {t['from'] or 'new'} -> {t['to']} ({t['decided_by']})"
                for t in reg["transitions"]) + "."]
        L += ["", f"Rejected candidates: {len(reg['rejected'])}. Retired: "
              f"{len(reg['retired'])}."]
        L += [f"- `{m['candidate_id']}` {m['from']} -> {m['to']} by {m['decided_by']}: "
              f"{m['reason']}" for m in reg["rejected"] + reg["retired"]]
    else:
        L.append(f"Not supplied: {reg['note']}.")
    adm = sf["admission"]
    L += ["", "### Admission (gate 6)", "",
          f"Fixed policy `{adm['policy_sha256'][:12]}` on the {adm['policy']['split']} split, "
          f"seeds {adm['policy']['seeds']}, {adm['policy']['n_bootstrap']} resamples. A "
          "proposed candidate is admitted only with:", ""]
    L += [f"- {r}" for r in adm["rules"]]
    L += ["", _sentence(adm["note"]) + ".", "",
          "### Checkpoints and rollbacks (gate 7)", ""]
    rb = sf["rollbacks"]
    if rb["status"] == "supplied":
        L += ["Checkpoints by status: " + (", ".join(f"{k} {v}" for k, v in
                                                     rb["checkpoints"].items()) or "none")
              + ". Log events: " + (", ".join(f"{k} {v}" for k, v in rb["events"].items())
                                    or "none") + f". Rollbacks: {len(rb['rollbacks'])}."]
        L += [f"- checkpoint {r['from']} rolled back to {r['to']}: {r['reason']}"
              for r in rb["rollbacks"]]
        if rb.get("log"):
            L += ["", "Checkpoint log: " + "; ".join(
                f"{e['event']} checkpoint {e['checkpoint_id']}"
                + (f" (from {e['from']})" if e["from"] is not None else "")
                + f": {e['reason']}" for e in rb["log"]) + "."]
        L.append("")
    else:
        L += [_sentence(rb["note"]) + ".", ""]

    live = report["live_smoke"]
    L += ["## Live Hermes smoke run", ""]
    if live["status"] == "supplied":
        def num(v, f):
            return "n/a" if v is None else format(v, f)
        L += [f"**{live['label']}.** {live['note']}", "",
              f"Source: {live['source']}. Model: {live['model']}. Split: {live['split']}.",
              "", "| Case | Completed | Quality (F1) | Est. cost USD | Latency s | Violation |",
              "|---|---|---|---|---|---|"]
        L += [f"| {c['case_id']} | {c['completed']} | {num(c['quality'], '.2f')} | "
              f"{num(c['cost_usd'], '.4f')} | {num(c['latency_s'], '.1f')} | "
              f"{c['violation']} |" for c in live["cases"]]
        L += ["",
              f"- Completed: {live['completed']} / {live['n_cases']}; violations: "
              f"{live['violations']}.",
              f"- Mean quality {num(live['mean_quality'], '.3f')}, mean cost USD "
              f"{num(live['mean_cost_usd'], '.4f')}, mean latency s "
              f"{num(live['mean_latency_s'], '.1f')}.", ""]
    else:
        L += [_sentence(live["note"]) + ".", ""]

    L += _live_run_markdown(report["live_run"])

    fe = report["final_test_evaluation"]
    L += ["## Final test evaluation (one-time, recorded)", ""]
    if fe["status"] == "not run":
        L += [_sentence(fe["note"]) + ". It will be the pre-registered paired run "
              "(baseline and candidate skill on the same sealed cases; strict scoring "
              "primary, lenient secondary; see docs/sprints/KAFO-EVO-001.md), through "
              "`evolution.adaptive.final_evaluation.run_paired_final_evaluation`, which "
              "refuses a second run.", ""]
    elif fe.get("design") == "paired":
        L += _paired_markdown(fe) + [""]
    else:
        L += [f"Status **{fe['status']}**, started {fe.get('started_at')}, finished "
              f"{fe.get('finished_at')}, reviewer `{fe.get('reviewer_id')}`, skill version "
              f"`{fe.get('skill_version')}`, synthetic: {fe.get('synthetic')}."]
        rs = fe.get("reviewer_summary")
        if rs and "error" not in rs:
            L += ["", f"Model `{rs.get('model')}` (provider `{rs.get('provider')}`), strategy "
                  f"`{rs.get('strategy_id')}`. Cost USD {rs.get('total_cost_usd')} "
                  f"(statuses {rs.get('cost_statuses')}, cap {rs.get('max_total_cost_usd')}). "
                  f"Calls {rs.get('calls')}, completed {rs.get('completed')}, unparseable "
                  f"{rs.get('unparseable')}, violations {rs.get('violations')}, exceptions "
                  f"{rs.get('exceptions')}, skipped {rs.get('skipped')}"
                  + (f", stopped: {rs['stopped']}" if rs.get("stopped") else "") + "."]
        if fe["status"] == "completed":
            L += ["", f"{fe['n_cases']} cases. Micro F1 {_iv(fe['micro']['f1'])}, micro "
                  f"precision {_iv(fe['micro']['precision'])}, micro recall "
                  f"{_iv(fe['micro']['recall'])}."]
        elif fe.get("error"):
            L += ["", f"Error: {fe['error']}"]
        L.append("")

    L += ["## Pending", ""] + [f"- {_sentence(p)}" for p in report["pending"]] + [""]
    L += ["## Reproduce", "", "```bash",
          f"python -m evolution.adaptive.report --seeds {s['seeds']} --train-episodes "
          f"{s['train_episodes']} --benchmark-seed {s['benchmark_seed']} --resamples "
          f"{s['comparison_resamples']} --verifier-seed {s['verifier_seed']} "
          f"--verifier-resamples {s['verifier_resamples']}", "```", ""]
    return "\n".join(L)


def to_json(report: Mapping) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_report(report: Mapping, out_dir: Path = DEFAULT_OUT) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / "report.md", out_dir / "report.json"
    md.write_text(format_markdown(report), encoding="utf-8")
    js.write_text(to_json(report), encoding="utf-8")
    return md, js


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the KAFO-EVO-001 sprint report "
                                     "(offline, seeded, synthetic).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--benchmark-seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--verifier-seed", type=int, default=0)
    parser.add_argument("--verifier-resamples", type=int, default=1000)
    parser.add_argument("--registry", help="candidate registry SQLite file (read-only)")
    parser.add_argument("--checkpoints", help="gate 7 checkpoint store SQLite file (read-only)")
    parser.add_argument("--live-smoke", default=str(DEFAULT_LIVE_SMOKE),
                        help="live Hermes smoke results JSON ('' to omit)")
    parser.add_argument("--live-run", default=str(DEFAULT_LIVE_RUN),
                        help="live full-run snapshot JSON ('' to omit)")
    parser.add_argument("--snapshot-live-run", metavar="RUN_DIR",
                        help="first rebuild the --live-run snapshot from a full_run "
                        "directory (read-only; aggregates only)")
    parser.add_argument("--final-ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--generated-at", help="timestamp to print (omitted by default "
                        "so the output is reproducible)")
    args = parser.parse_args(argv)
    if min(args.seeds, args.train_episodes, args.resamples, args.verifier_resamples) < 1:
        parser.error("--seeds, --train-episodes and resample counts must be positive")
    if args.snapshot_live_run:
        if not args.live_run:
            parser.error("--snapshot-live-run needs a --live-run output path")
        snap = snapshot_live_run(args.snapshot_live_run)
        Path(args.live_run).write_text(json.dumps(snap, indent=2, sort_keys=True,
                                                  allow_nan=False) + "\n", encoding="utf-8")
        print(f"wrote {_rel(Path(args.live_run))}")
    report = generate(seeds=args.seeds, train_episodes=args.train_episodes,
                      benchmark_seed=args.benchmark_seed, resamples=args.resamples,
                      verifier_seed=args.verifier_seed,
                      verifier_bootstrap=args.verifier_resamples,
                      registry_path=args.registry, checkpoint_path=args.checkpoints,
                      live_smoke_path=args.live_smoke or None,
                      live_run_path=args.live_run or None,
                      final_ledger_path=Path(args.final_ledger),
                      generated_at=args.generated_at)
    md, js = write_report(report, Path(args.out))
    print(f"wrote {_rel(md)} and {_rel(js)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Driver for the first full live run of the KAFO-EVO-001 adaptive loop.

Stages, in order (``--stages`` selects a subset; state is resumable):

1. ``canary``     3 train cases with the baseline-skill arm. Aborts the run if
                  any reply is not a strict JSON findings list.
2. ``compare``    the gate 4 policy comparison (fixed, random, epsilon-greedy,
                  LinUCB) on the train and validation splits, two tool-free
                  arms: ``direct`` (no system prompt) and ``skill`` (baseline
                  skill text as the system prompt).
3. ``gepa``       one real ``dspy.GEPA`` run on the skill text. Data: the
                  python_review TRAIN split only (20 GEPA-train / 8 GEPA-val);
                  metric: the gate 3 verifier F1 with verifier feedback.
4. ``admission``  register the GEPA output as ``proposed`` (gate 5), then
                  ``admission.review_with_rollback``: gate 7 pre-admission
                  checkpoint, gate 6 check against the baseline on
                  validation, and if admitted the post-admission check that
                  keeps the candidate or retires it and rolls back.

Spend control: every paid call goes through one append-only JSONL ledger in
the run directory. Before each fresh call the ledger reserves a worst-case
amount and refuses (``BudgetExhausted``) once ``--stop-usd`` would be
crossed. Hermes calls are keyed by their exact request plus a sample number,
so a resumed run replays recorded outcomes instead of paying again, and
the canary's calls are reused by the comparison (same strategy and request).

Default mode is a STUB dry run: the real ``HermesExecutor`` runs with a stub
subprocess runner that never starts Hermes, and GEPA (if dspy is installed)
runs against ``dspy.utils.DummyLM``. Nothing is spent. ``--live`` needs an
explicit model and stop amount. Live GEPA needs exactly one route:
``--gepa-via-hermes`` sends every GEPA task and reflection call through the
same ``HermesExecutor`` bridge, model and provider as the other stages (no
API key); ``--gepa-allow-metered`` lets DSPy call the Anthropic API directly
(metered ``ANTHROPIC_API_KEY``). The two flags are mutually exclusive.

Never loads the sealed split: every stage runs inside
``benchmark.optimisation_phase()`` (directly or via the gate 4/6/7 code).
The gate 2 v1 benchmark is hand-written synthetic data, so no result here is
evidence that Hermes improves on any real task.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Callable, Mapping, Optional, Sequence

from . import benchmark as bm, evaluation, verifier
from .benchmark import DEFECT_CATEGORIES, optimisation_phase
from .compare import (CaseBenchmark, ContextEncoder, OutcomeTable, Scoring, compare,
                      default_policy_specs, format_markdown, run_policy)
from .executor import Budget, ExecutionOutcome, HermesExecutor, Strategy, Task
from .policies import LinUCBPolicy

RUN_VERSION = "kafo-full-run/1"
SKILL_NAME = "github-code-review"
HARD_CAP_USD = 20.0                 # owner-approved total; --stop-usd may not exceed it
DEFAULT_STOP_USD = 18.0             # "stop at about 18 and report"
HERMES_RESERVE_USD = 0.03           # worst-case reserve before a fresh Hermes call
UNKNOWN_COST_CHARGE_USD = 0.03      # charged when a call's cost is unknown
# Anthropic list price, claude-sonnet-5-5, USD per million tokens. Hermes'
# own estimate for the 2026-11-22 smoke run reproduces at these rates. GEPA
# cost is charged at input+output rates with no cache discount (an upper bound).
PRICE_PER_M = {"input": 2.00, "output": 10.00}
SCORING = Scoring(reference_cost_usd=0.03, reference_latency_s=6.0)   # = gate 6 references
CANARY_CASES = 3
GEPA_TRAIN_CASES = 20               # the rest of the train split is GEPA's valset
STAGES = ("canary", "compare", "gepa", "admission")

SYNTHETIC_NOTICE = ("SYNTHETIC BENCHMARK: python_review v1 is hand-written synthetic data. "
                    "Live Hermes calls on it are not evidence of real Hermes improvement.")

# Identical to ``admission.REVIEW_PROMPT`` (gate 6, PR #8); checked at start-up
# when that module is importable, so comparison and admission ask the same thing.
REVIEW_PROMPT = (
    "Review the Python code below for defects. Reply with only a JSON list of findings; "
    "each finding is an object with integer line_start and line_end (1-based, inclusive), "
    "a category from {categories} and a short description. Reply [] if there are no "
    "defects.\n\n```python\n{code}```\n"
)
REVIEW_TEMPLATE = REVIEW_PROMPT.replace("{categories}", ", ".join(DEFECT_CATEGORIES))


class BudgetExhausted(RuntimeError):
    """The next paid call could cross the stop amount."""


class CanaryFailed(RuntimeError):
    """The canary showed the live setup cannot produce scorable replies."""


class GepaCallFailed(RuntimeError):
    """A GEPA LM call through Hermes failed; the stage fails closed."""


class RunMismatch(RuntimeError):
    """The run directory belongs to a run with a different identity."""


def review_prompt(code: str) -> str:
    return REVIEW_TEMPLATE.format(code=code)


def parse_findings(output: str) -> tuple[list, bool]:
    """Strict: a JSON list or nothing (same rule as gate 6)."""
    try:
        value = json.loads(output)
    except (TypeError, ValueError):
        return [], False
    return (value, True) if isinstance(value, list) else ([], False)


def _as_verifier_case(case) -> verifier.BenchmarkCase:
    return case if isinstance(case, verifier.BenchmarkCase) else evaluation.to_verifier_case(case)


def review_quality(item, outcome) -> float:
    """Verifier F1 of the executor's reply; unparsable replies score as no findings."""
    findings, _ = parse_findings(outcome.output)
    return verifier.quality(_as_verifier_case(item.ground_truth), findings)


def _sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


_SECRET_NAME = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def redact(text: Optional[str]) -> Optional[str]:
    """Replace the values of secret-looking environment variables in ``text``."""
    if not text:
        return text
    for name, value in os.environ.items():
        if len(value) >= 8 and any(m in name.upper() for m in _SECRET_NAME) and value in text:
            text = text.replace(value, f"<redacted:{name}>")
    return text


# ------------------------------------------------------------------ ledger

class SpendLedger:
    """Append-only JSONL record of every paid (or stubbed) call."""

    def __init__(self, path: str | Path, *, stop_usd: float):
        if not 0 < stop_usd <= HARD_CAP_USD:
            raise ValueError(f"stop_usd must be in (0, {HARD_CAP_USD}]")
        self.path, self.stop_usd = Path(path), float(stop_usd)
        self.rows: list[dict] = []
        self._replay: dict[str, dict] = {}
        self.exhausted = False
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._add(json.loads(line))

    @staticmethod
    def charge(row: Mapping) -> float:
        cost = row.get("cost_usd")
        return float(cost) if cost is not None else float(
            row.get("charge_if_unknown", UNKNOWN_COST_CHARGE_USD))

    @property
    def spent(self) -> float:
        return sum(self.charge(r) for r in self.rows)

    def _add(self, row: dict) -> None:
        self.rows.append(row)
        if row.get("replayable") and row.get("key"):
            self._replay[row["key"]] = row

    def reserve(self, amount: float) -> None:
        if self.spent + amount > self.stop_usd:
            self.exhausted = True
            raise BudgetExhausted(f"spent ${self.spent:.4f}; a further ${amount:.2f} "
                                  f"could cross the ${self.stop_usd:.2f} stop")

    def record(self, row: dict) -> dict:
        row = {"recorded_at": datetime.now(timezone.utc).isoformat(), **row}
        for field in ("error", "output"):
            if isinstance(row.get(field), str):
                row[field] = redact(row[field])
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(_canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._add(row)
        return row

    def lookup(self, key: str) -> Optional[dict]:
        return self._replay.get(key)

    def summary(self) -> dict:
        out: dict[str, dict] = {}
        for row in self.rows:
            entry = out.setdefault(row.get("stage", "?"), {
                "calls": 0, "errors": 0, "cost_usd": 0.0, "unknown_cost_calls": 0,
                "latencies_s": [], "input_tokens": 0, "output_tokens": 0})
            entry["calls"] += 1
            entry["errors"] += bool(row.get("error"))
            entry["cost_usd"] += self.charge(row)
            entry["unknown_cost_calls"] += row.get("cost_usd") is None
            entry["input_tokens"] += row.get("input_tokens") or 0
            entry["output_tokens"] += row.get("output_tokens") or 0
            if row.get("latency_s") is not None:
                entry["latencies_s"].append(row["latency_s"])
        for entry in out.values():
            lat = entry.pop("latencies_s")
            entry["cost_usd"] = round(entry["cost_usd"], 6)
            entry["latency_s"] = ({"mean": round(statistics.fmean(lat), 3),
                                   "median": round(statistics.median(lat), 3),
                                   "max": round(max(lat), 3)} if lat else None)
        return {"spent_usd": round(self.spent, 6), "stop_usd": self.stop_usd,
                "exhausted": self.exhausted, "by_stage": out}


def request_key(executor: HermesExecutor, task: Task, strategy: Strategy, sample: int) -> str:
    """Identity of one Hermes request: what the agent sees, the strategy ID and the sample.

    The strategy ID is part of the key so that gate 6's ``eval:`` arm and
    gate 7's catalogue arm, which send identical requests, stay independent
    samples instead of one replaying the other.
    """
    return _sha256(_canonical({
        "strategy_id": strategy.strategy_id,
        "prompt": task.prompt, "system_prompt": strategy.system_prompt,
        "toolsets": list(strategy.toolsets), "model": executor.model,
        "provider": executor.provider, "budget": asdict(executor.budget),
        "sample": sample}))


_OUTCOME_FIELDS = ("output", "completed", "latency_s", "cost_usd", "cost_status", "error",
                   "input_tokens", "output_tokens", "api_calls", "model", "wall_time_s")


class LedgerExecutor:
    """Executor wrapper: budget guard, durable record and replay of Hermes calls."""

    def __init__(self, inner: HermesExecutor, ledger: SpendLedger, *, stage: str,
                 sample: int = 0, retries: int = 1, reserve_usd: float = HERMES_RESERVE_USD):
        self.inner, self.ledger, self.stage, self.sample = inner, ledger, stage, sample
        self.retries, self.reserve_usd = retries, reserve_usd
        self.backend = inner.backend
        self.fresh = self.replayed = 0

    def execute(self, task: Task, strategy: Strategy) -> ExecutionOutcome:
        key = request_key(self.inner, task, strategy, self.sample)
        row = self.ledger.lookup(key)
        if row is not None:
            self.replayed += 1
            return ExecutionOutcome(
                task_id=task.task_id, strategy_id=strategy.strategy_id,
                backend=row["backend"], tool_calls=tuple(row["tool_calls"]),
                violation_reasons=tuple(row["violation_reasons"]),
                details=dict(row.get("details") or {}),
                **{f: row[f] for f in _OUTCOME_FIELDS})
        for attempt in range(self.retries + 1):
            self.ledger.reserve(self.reserve_usd)
            outcome = self.inner.execute(task, strategy)
            self.fresh += 1
            ok = outcome.cost_usd is not None and not outcome.error
            self.ledger.record({
                "kind": "hermes", "stage": self.stage, "sample": self.sample, "key": key,
                "attempt": attempt, "replayable": ok, "task_id": task.task_id,
                "strategy_id": strategy.strategy_id,
                "system_prompt_sha256": _sha256(strategy.system_prompt),
                "backend": outcome.backend, "tool_calls": list(outcome.tool_calls),
                "violation_reasons": list(outcome.violation_reasons),
                "details": dict(outcome.details),
                **{f: getattr(outcome, f) for f in _OUTCOME_FIELDS}})
            if ok:
                break
        return outcome


_STUB_FIELD_VALUES = {"findings_json": "[]",
                      "new_instruction": "STUB GEPA EDIT: review the code and reply with "
                                         "only a JSON list."}


def _stub_reply(system: str, prompt: str) -> str:
    """Stub reply: a review JSON list, or DSPy chat-format output fields for GEPA calls."""
    if "Your output fields are:" in system:                 # a DSPy adapter request
        section = system.split("Your output fields are:", 1)[1]
        names = re.findall(r"^\d+\. `([A-Za-z_][A-Za-z0-9_]*)`", section, re.MULTILINE)
        def field(n):
            value = _STUB_FIELD_VALUES.get(n, "stub")
            if f"[[ ## {n} ## ]]" in system:                  # bracket layout
                return f"[[ ## {n} ## ]]\n{value}"
            if f"<{n}>" in system:                            # XML-tag layout
                return f"<{n}>\n{value}\n</{n}>"
            return json.dumps({n: value})                     # JSON layout
        if "Outputs will be a JSON object" in system:
            return json.dumps({n: _STUB_FIELD_VALUES.get(n, "stub") for n in names})
        return "\n\n".join(field(n) for n in names)
    if prompt.lstrip().startswith("Review the Python code"):
        return "[]"                                          # a Hermes review: no findings
    return "OK"


def stub_runner(argv, cwd=None, **_kwargs):
    """Stand-in for ``subprocess.run`` of the Hermes bridge. Never starts Hermes.

    Writes a bridge result with a worst-case cost estimate: prompt + system
    prompt at ~4 chars/token plus ~1,100 tokens of Hermes' own prompt, all
    billed as uncached input, and 450 output tokens (the smoke run's max was 474).
    """
    request_path, result_path = argv[-2], argv[-1]
    with open(request_path, encoding="utf-8") as handle:
        request = json.load(handle)
    input_tokens = (len(request["prompt"]) + len(request.get("system_prompt") or "")) // 4 + 1100
    output_tokens = 450
    cost = (input_tokens * PRICE_PER_M["input"] + output_tokens * PRICE_PER_M["output"]) / 1e6
    reply = _stub_reply(request.get("system_prompt") or "", request["prompt"])
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump({"ok": True, "completed": True, "final_response": reply,
                   "estimated_cost_usd": round(cost, 6), "cost_status": "stub_estimate",
                   "input_tokens": input_tokens, "output_tokens": output_tokens,
                   "api_calls": 1, "latency_s": 4.0, "model": request.get("model"),
                   "provider": request.get("provider"), "tool_calls": []}, handle)
    return subprocess.CompletedProcess(argv, 0, "", "")


# ------------------------------------------------------------- the run itself

def split_frontmatter(text: str) -> tuple[Optional[str], str]:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[4:end], text[end + 4:].lstrip("\n")
    return None, text


class FullRun:
    def __init__(self, *, run_dir: Path, skill_text: str, skill_source: str,
                 executor: HermesExecutor, ledger: SpendLedger, live: bool,
                 seeds: int = 20, train_episodes: int = 300):
        self.run_dir, self.skill_text, self.skill_source = run_dir, skill_text, skill_source
        self.executor, self.ledger, self.live = executor, ledger, live
        self.seeds, self.train_episodes = seeds, train_episodes
        self.state_path = run_dir / "state.json"
        self.state = (json.loads(self.state_path.read_text(encoding="utf-8"))
                      if self.state_path.exists() else {"stages": {}})
        with optimisation_phase():
            cases = evaluation.open_cases(("train", "validation"))
        self.train_cases, self.validation_cases = cases["train"], cases["validation"]
        self.bench = CaseBenchmark([*self.train_cases, *self.validation_cases],
                                   prompt_template=REVIEW_TEMPLATE)
        self.encoder = ContextEncoder.fit(self.bench.items("train"), key_fields=("difficulty",),
                                          categorical=("difficulty", "domain"),
                                          numeric={"n_lines": 100.0})
        self.catalogue = {"direct": Strategy("direct"),
                          "skill": Strategy("skill", system_prompt=skill_text)}

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(redact(json.dumps(self.state, indent=2, sort_keys=True, default=str)),
                       encoding="utf-8")
        tmp.replace(self.state_path)

    def _exec(self, stage: str, sample: int = 0) -> LedgerExecutor:
        return LedgerExecutor(self.executor, self.ledger, stage=stage, sample=sample)

    # 1 ------------------------------------------------------------------
    def canary(self) -> dict:
        ex = self._exec("canary")
        rows = []
        for case in self.train_cases[:CANARY_CASES]:
            out = ex.execute(Task(case.case_id, review_prompt(case.code)), self.catalogue["skill"])
            findings, ok = parse_findings(out.output)
            rows.append({"case_id": case.case_id, "parse_ok": ok, "error": out.error,
                         "violations": list(out.violation_reasons), "cost_usd": out.cost_usd,
                         "latency_s": out.latency_s,
                         "f1": verifier.quality(case, findings) if ok else None})
        result = {"cases": rows, "fresh_calls": ex.fresh, "replayed_calls": ex.replayed}
        bad = [r["case_id"] for r in rows if not r["parse_ok"] or r["error"] or r["violations"]]
        if bad:
            raise CanaryFailed(f"canary failed on {bad}: {rows}")
        return result

    # 2 ------------------------------------------------------------------
    def trained_policy(self, ex) -> LinUCBPolicy:
        """LinUCB trained exactly as compare() trains seed 0 (replays only)."""
        table = OutcomeTable(ex, self.catalogue, review_quality, SCORING)
        policy = LinUCBPolicy(tuple(self.catalogue), self.encoder.dim, seed=0)
        rng = random.Random("task-sequence:0")
        items = list(self.bench.items("train"))
        sequence = [rng.choice(items) for _ in range(self.train_episodes)]
        with optimisation_phase():
            run_policy("linucb", policy, sequence, self.encoder, table, phase="train",
                       seed=0, learn=True, oracle=False)
        return policy

    def compare(self) -> dict:
        ex = self._exec("compare")
        specs = default_policy_specs(tuple(self.catalogue), self.encoder.dim)
        result = compare(self.bench, ex, review_quality, self.catalogue, self.encoder, specs,
                         seeds=list(range(self.seeds)), train_episodes=self.train_episodes,
                         scoring=SCORING, skill_version=self.state["baseline_id"])
        (self.run_dir / "comparison.md").write_text(format_markdown(result.report),
                                                     encoding="utf-8")
        return {"report": result.report, "fresh_calls": ex.fresh,
                "replayed_calls": ex.replayed}

    # 3 ------------------------------------------------------------------
    def gepa(self, *, task_lm, reflection_lm, max_metric_calls: int, seed: int = 0) -> dict:
        import dspy  # optional dependency, only needed for this stage

        front, body = split_frontmatter(self.skill_text)
        by_id = {c.case_id: c for c in self.train_cases}
        examples = [dspy.Example(request=review_prompt(c.code), case_id=c.case_id)
                    .with_inputs("request") for c in self.train_cases]

        def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            findings, ok = parse_findings(getattr(pred, "findings_json", "") or "")
            res = verifier.score_case(by_id[gold.case_id], findings)
            notes = [f"Verifier F1 {res.f1:.2f} (precision {res.precision:.2f}, "
                     f"recall {res.recall:.2f})."]
            if not ok:
                notes.append("The reply was not a bare JSON list, so it scored as no findings.")
            if res.missed_defects:
                notes.append(f"Missed {len(res.missed_defects)} defect(s): {list(res.missed_defects)}.")
            if res.false_positives:
                notes.append(f"{res.false_positives} finding(s) matched no real defect.")
            return dspy.Prediction(score=res.f1, feedback=" ".join(notes))

        program = dspy.Predict(dspy.Signature("request: str -> findings_json: str", body))
        # Preflight: one task call and one reflection-model call, so a bad model
        # string or key fails here instead of as ~170 swallowed GEPA failures.
        first = self.train_cases[0]
        with dspy.context(lm=task_lm):
            probe = program(request=review_prompt(first.code))
        if not parse_findings(getattr(probe, "findings_json", "") or "")[1]:
            raise CanaryFailed("GEPA preflight: task model reply is not a JSON findings list")
        if not reflection_lm("Reply with the single word OK."):
            raise CanaryFailed("GEPA preflight: reflection model returned nothing")
        log_dir = self.run_dir / "gepa_log"
        optimizer = dspy.GEPA(metric=metric, max_metric_calls=max_metric_calls,
                              reflection_lm=reflection_lm, num_threads=1, seed=seed,
                              track_stats=True, log_dir=str(log_dir))
        start = time.perf_counter()
        with dspy.context(lm=task_lm):
            optimized = optimizer.compile(program, trainset=examples[:GEPA_TRAIN_CASES],
                                          valset=examples[GEPA_TRAIN_CASES:])
        elapsed = time.perf_counter() - start
        failures = [f for lm in (task_lm, reflection_lm)
                    for f in getattr(getattr(lm, "hermes_engine", None), "failures", [])]
        if failures:
            raise GepaCallFailed(f"{len(failures)} GEPA call(s) through Hermes failed; "
                                 f"first: {failures[0]}; no candidate registered")
        if self.ledger.exhausted:
            raise BudgetExhausted("budget stop reached during GEPA; no candidate registered")
        evolved_body = optimized.signature.instructions.strip()
        unchanged = evolved_body == body.strip()
        evolved = (self.skill_text if unchanged else
                   f"{evolved_body}\n" if front is None else
                   f"---\n{front}\n---\n\n{evolved_body}\n")
        scores = list(getattr(getattr(optimized, "detailed_results", None),
                              "val_aggregate_scores", []) or [])
        constraints = bool(evolved_body.strip()) and len(evolved) <= 1.2 * len(self.skill_text)
        out_dir = self.run_dir / "gepa" / SKILL_NAME / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True)
        (out_dir / "baseline_skill.md").write_text(self.skill_text, encoding="utf-8")
        (out_dir / ("evolved_skill.md" if constraints else "evolved_FAILED.md")).write_text(
            evolved, encoding="utf-8")
        metrics = {"skill_name": SKILL_NAME, "optimizer": "dspy.GEPA",
                   "lm_route": {role: getattr(lm, "route", type(lm).__name__)
                                for role, lm in (("task", task_lm),
                                                 ("reflection", reflection_lm))},
                   "max_metric_calls": max_metric_calls, "seed": seed,
                   "gepa_train_cases": GEPA_TRAIN_CASES,
                   "gepa_val_cases": len(examples) - GEPA_TRAIN_CASES,
                   "data": "python_review v1 train split only",
                   "baseline_score": scores[0] if scores else None,
                   "evolved_score": max(scores) if scores else None,
                   "candidates_explored": len(scores), "elapsed_seconds": round(elapsed, 1),
                   "constraints_passed": constraints,
                   "baseline_size": len(self.skill_text), "evolved_size": len(evolved),
                   "identical_to_baseline": unchanged}
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return {"output_dir": str(out_dir), "metrics": metrics}

    # 4 ------------------------------------------------------------------
    def admission(self) -> dict:
        """Gate 5 registration, then gate 6 + gate 7 via ``admission.review_with_rollback``.

        That function (PR #12) saves the pre-admission checkpoint of the
        trained policy, runs the gate 6 review, and if admitted re-validates
        the candidate's catalogue strategy; a drop beyond the rollback margin
        or a critical violation retires the candidate and rolls back.
        """
        from . import admission as adm        # gates 6 and 7
        from .checkpoint import CheckpointStore
        from .gepa_adapter import register_gepa_output
        from .registry import CandidateRegistry

        gepa = self.state["stages"].get("gepa", {}).get("result")
        if not gepa or gepa.get("status") == "skipped":
            return {"status": "skipped", "reason": "no GEPA output"}
        if gepa["metrics"]["identical_to_baseline"]:
            return {"status": "not_run", "reason": "GEPA returned the baseline unchanged; "
                                                    "nothing to admit"}
        if not gepa["metrics"]["constraints_passed"]:
            return {"status": "not_run", "reason": "GEPA output failed constraints"}

        with CandidateRegistry(self.run_dir / "registry.sqlite") as registry, \
                CheckpointStore(self.run_dir / "checkpoints.sqlite") as store:
            registry.register_baseline(SKILL_NAME, self.skill_text.encode("utf-8"),
                                       provenance={"source": self.skill_source,
                                                   "run": RUN_VERSION})
            cand = register_gepa_output(registry, gepa["output_dir"], skill_name=SKILL_NAME)
            out = {"candidate_id": cand.candidate_id, "parent_id": cand.parent_id,
                   "registered_status": registry.get(cand.candidate_id).status}
            policy = self.trained_policy(self._exec("compare"))
            res = adm.review_with_rollback(
                registry, cand.candidate_id, base=Strategy("review"),
                make_executor=lambda seed: self._exec("admission", sample=seed),
                store=store, policy=policy)
            out.update({
                "status": ("kept" if res.kept else "rolled_back" if res.rolled_back
                           else "rejected"),
                "reason": res.reason, "decision": res.decision.to_dict(),
                "active_skill": res.active_skill,
                "registry_status": registry.get(cand.candidate_id).status,
                "pre_validation": res.pre_validation.to_dict(),
                "post_validation": (res.post_validation.to_dict()
                                    if res.post_validation else None),
                "checkpoint_log": store.log()})
        return out

    # ------------------------------------------------------------------
    def identity(self) -> dict:
        return {"run_version": RUN_VERSION, "mode": "live" if self.live else "stub",
                "model": self.executor.model, "provider": self.executor.provider,
                "budget": asdict(self.executor.budget),
                "skill_sha256": _sha256(self.skill_text.encode("utf-8"))}

    def run(self, stages: Sequence[str], *, gepa_factory: Optional[Callable] = None,
            gepa_max_metric_calls: int = 150) -> dict:
        identity = self.identity()
        recorded = self.state.setdefault("identity", identity)
        if recorded != identity:
            changed = sorted(k for k in identity if recorded.get(k) != identity[k])
            raise RunMismatch(f"{self.run_dir} belongs to another run (differs in {changed}); "
                              "use a new --run-dir or --run-name")
        self.state.setdefault("baseline_id",
                              f"{SKILL_NAME}@{identity['skill_sha256'][:12]}")
        sealed_before = bm._sealed_attempts
        try:
            with optimisation_phase():      # every stage; the sealed split stays shut
                self._run_stages(stages, gepa_factory, gepa_max_metric_calls)
        finally:
            self.state["sealed_split_open_attempts"] = bm._sealed_attempts - sealed_before
            self._save()
            report = self.report()
        return report

    def _run_stages(self, stages, gepa_factory, gepa_max_metric_calls) -> None:
        for stage in STAGES:
            if stage not in stages:
                continue
            if self.state["stages"].get(stage, {}).get("status") == "done":
                continue
            started = time.perf_counter()
            try:
                if stage == "gepa":
                    if gepa_factory is None:
                        result = {"status": "skipped", "reason": "GEPA not enabled for this run"}
                    else:
                        task_lm, reflection_lm = gepa_factory(self.ledger)
                        result = self.gepa(task_lm=task_lm, reflection_lm=reflection_lm,
                                           max_metric_calls=gepa_max_metric_calls)
                elif stage == "admission":
                    try:
                        result = self.admission()
                    except ImportError as exc:
                        result = {"status": "skipped",
                                  "reason": f"gate 6 admission module unavailable ({exc})"}
                else:
                    result = getattr(self, stage)()
                # A skipped stage is not "done": a resumed run tries it again.
                status = "skipped" if result.get("status") == "skipped" else "done"
                self.state["stages"][stage] = {
                    "status": status, "result": result,
                    "reason": result.get("reason") if status == "skipped" else None,
                    "seconds": round(time.perf_counter() - started, 1)}
            except (BudgetExhausted, CanaryFailed, GepaCallFailed) as exc:
                self.state["stages"][stage] = {"status": "stopped",
                                               "reason": redact(f"{type(exc).__name__}: {exc}")}
                return
            except Exception as exc:
                self.state["stages"][stage] = {"status": "error",
                                               "reason": redact(f"{type(exc).__name__}: {exc}")}
                raise
            finally:
                self._save()

    def report(self) -> dict:
        report = {
            "run_version": RUN_VERSION, "mode": "live" if self.live else "stub",
            "identity": self.state.get("identity"), "notice": SYNTHETIC_NOTICE, "skill_name": SKILL_NAME,
            "skill_source": self.skill_source,
            "skill_sha256": _sha256(self.skill_text.encode("utf-8")),
            "baseline_id": self.state.get("baseline_id"),
            "hermes": {"model": self.executor.model, "provider": self.executor.provider,
                       "budget": asdict(self.executor.budget)},
            "splits_used": ["train", "validation"],
            "sealed_split_open_attempts": self.state.get("sealed_split_open_attempts"),
            "stages": self.state["stages"], "spend": self.ledger.summary()}
        (self.run_dir / "report.json").write_text(
            redact(json.dumps(report, indent=2, sort_keys=True, default=str)), encoding="utf-8")
        return report


# ------------------------------------------------------------------ GEPA LMs

GEPA_INPUT_TOKENS_BOUND = 12_000    # skill (~3.3k) + minibatch traces + DSPy framing


def call_reserve_usd(max_tokens: int) -> float:
    """Worst case for one DSPy call: a full input bound plus the output cap."""
    return (GEPA_INPUT_TOKENS_BOUND * PRICE_PER_M["input"]
            + max_tokens * PRICE_PER_M["output"]) / 1e6


def budgeted_lm(model: str, ledger: SpendLedger, *, role: str, max_tokens: int,
                dummy_answers: Optional[list] = None, stub_cost_usd: float = 0.0):
    """A dspy LM whose every call is reserved against, and recorded in, the ledger.

    Cost comes from the returned token usage at list price, input and output
    (thinking included) with no cache discount. With ``dummy_answers`` the LM
    is ``dspy.utils.DummyLM`` (offline) charged ``stub_cost_usd`` per call.
    """
    import dspy
    from dspy.utils import DummyLM

    base = DummyLM if dummy_answers is not None else dspy.LM
    reserve = call_reserve_usd(max_tokens) if dummy_answers is None else max(stub_cost_usd, 0.0)

    class BudgetedLM(base):
        def __call__(self, *args, **kwargs):
            ledger.reserve(reserve)
            seen = len(self.history)
            error, started = None, time.perf_counter()
            try:
                return super().__call__(*args, **kwargs)
            except Exception as exc:
                error = redact(f"{type(exc).__name__}: {exc}"[:300])
                raise
            finally:
                latency = round(time.perf_counter() - started, 3)
                entries = self.history[seen:]
                if not entries:
                    ledger.record({"kind": "dspy", "stage": f"gepa:{role}", "model": model,
                                   "error": error or "no history entry", "cost_usd": None,
                                   "charge_if_unknown": reserve, "latency_s": latency})
                for entry in entries:
                    usage = entry.get("usage") or {}
                    tin, tout = usage.get("prompt_tokens"), usage.get("completion_tokens")
                    if dummy_answers is not None:
                        cost = stub_cost_usd
                    elif tin is None or tout is None:
                        cost = None
                    else:
                        cost = (tin * PRICE_PER_M["input"] + tout * PRICE_PER_M["output"]) / 1e6
                    ledger.record({"kind": "dspy", "stage": f"gepa:{role}", "model": model,
                                   "input_tokens": tin, "output_tokens": tout,
                                   "cost_usd": cost, "charge_if_unknown": reserve,
                                   "error": error, "latency_s": latency / len(entries),
                                   "cost_basis": "list price, no cache discount"})

    if dummy_answers is not None:
        return BudgetedLM(dummy_answers)
    return BudgetedLM(model, max_tokens=max_tokens, cache=False, num_retries=2)


GEPA_TASK_MAX_TOKENS = 4_000
GEPA_REFLECTION_MAX_TOKENS = 16_000


def live_gepa_factory(task_model: str, reflection_model: str):
    def factory(ledger):
        return (budgeted_lm(task_model, ledger, role="task", max_tokens=GEPA_TASK_MAX_TOKENS),
                budgeted_lm(reflection_model, ledger, role="reflection",
                            max_tokens=GEPA_REFLECTION_MAX_TOKENS))
    return factory


def stub_gepa_factory(max_metric_calls: int, *, stub_cost_usd: float = 0.0):
    def factory(ledger):
        n = 2 * max_metric_calls + 50
        return (budgeted_lm("stub/task", ledger, role="task", max_tokens=0,
                            dummy_answers=[{"findings_json": "[]"}] * n,
                            stub_cost_usd=stub_cost_usd),
                budgeted_lm("stub/reflection", ledger, role="reflection", max_tokens=0,
                            dummy_answers=[{"new_instruction": "STUB GEPA EDIT: review the "
                                            "code and reply with only a JSON list."}] * n,
                            stub_cost_usd=stub_cost_usd))
    return factory


# ------------------------------------------------------- GEPA through Hermes

def _parts_text(parts) -> str:
    from dspy.lm15 import TextPart
    if isinstance(parts, str):
        return parts
    texts = []
    for part in parts or ():
        if not isinstance(part, TextPart):
            raise GepaCallFailed(f"non-text request part {type(part).__name__} cannot go "
                                 "through Hermes")
        texts.append(part.text)
    return "".join(texts)


class HermesEngine:
    """dspy ``Engine`` that sends each completion through ``HermesExecutor``.

    The request's system text becomes the Hermes strategy's system prompt and
    its messages become the task prompt, so GEPA runs on the same provider,
    model and bridge as every other stage (subscription-billed; the ledger
    records Hermes' estimated cost). Each call is reserved against and
    recorded in the run ledger through ``LedgerExecutor``. Fails closed: an
    error, unknown cost, empty reply or policy violation is recorded and
    raised, and every later call is refused without being made. There is no
    fallback to LiteLLM or any other provider.
    """

    def __init__(self, executor: LedgerExecutor, *, role: str):
        self.executor, self.role = executor, role
        self.failures: list[str] = []

    def complete(self, request):
        from dspy.clients.engines.base import validate_request
        from dspy.lm15 import Message, Response, TextPart, Usage

        validate_request(request)
        if self.failures:
            raise GepaCallFailed(f"gepa:{self.role}: refused after an earlier failure")
        if request.tools:
            raise GepaCallFailed("tool definitions cannot go through Hermes (tool-free arm)")
        system = _parts_text(request.system) if request.system is not None else ""
        turns = [(m.role, _parts_text(m.parts)) for m in request.messages]
        prompt = (turns[0][1] if len(turns) == 1 and turns[0][0] == "user" else
                  "\n\n".join(f"[{role}]\n{text}" for role, text in turns))
        task = Task(f"gepa-{self.role}-{_sha256(system + chr(0) + prompt)[:12]}", prompt)
        out = self.executor.execute(task, Strategy(f"gepa-{self.role}", system_prompt=system))
        problem = (out.error or ("unknown cost" if out.cost_usd is None else None)
                   or ("empty reply" if not out.output.strip() else None)
                   or (f"violation {list(out.violation_reasons)}" if out.violation_reasons
                       else None))
        if problem:
            self.failures.append(redact(f"{task.task_id}: {problem}")[:300])
            raise GepaCallFailed(self.failures[-1])
        usage = Usage(input_tokens=out.input_tokens or 0, output_tokens=out.output_tokens or 0,
                      total_tokens=(out.input_tokens or 0) + (out.output_tokens or 0))
        return Response(id=None, model=out.model or self.executor.inner.model,
                        message=Message.assistant([TextPart(out.output)]),
                        finish_reason="stop", usage=usage)

    def stream(self, request):
        from dspy.lm15 import response_to_events
        return response_to_events(self.complete(request))

    def close(self):
        pass


class AsyncHermesEngine:
    def __init__(self, sync: HermesEngine):
        self.sync = sync

    async def complete(self, request):
        return self.sync.complete(request)

    async def stream(self, request):
        for event in self.sync.stream(request):
            yield event

    async def aclose(self):
        pass


def hermes_lm(executor: LedgerExecutor, *, role: str, max_tokens: int):
    from dspy.clients.base_lm import BaseLM

    engine = HermesEngine(executor, role=role)
    inner = executor.inner
    lm = BaseLM(f"hermes/{inner.model}", "chat", None, max_tokens, False, num_retries=0,
                engine=engine, async_engine=AsyncHermesEngine(engine))
    lm.hermes_engine = engine
    lm.route = f"hermes:{inner.provider}:{inner.model}"
    return lm


def hermes_gepa_factory(base: HermesExecutor):
    """GEPA task and reflection LMs over Hermes, same model/provider/runner as ``base``."""
    def factory(ledger):
        lms = []
        for role, max_tokens in (("task", GEPA_TASK_MAX_TOKENS),
                                 ("reflection", GEPA_REFLECTION_MAX_TOKENS)):
            inner = HermesExecutor(hermes_repo=base.hermes_repo, python=base.python,
                                   model=base.model, provider=base.provider,
                                   budget=Budget(max_tokens=max_tokens), allow_live=True,
                                   runner=base._runner)
            ex = LedgerExecutor(inner, ledger, stage=f"gepa:{role}", retries=0,
                                reserve_usd=call_reserve_usd(max_tokens))
            lms.append(hermes_lm(ex, role=role, max_tokens=max_tokens))
        return tuple(lms)
    return factory


# ---------------------------------------------------------------------- CLI

EVOLUTION_HOME_ENV = "HERMES_EVOLUTION_HOME"


def evolution_home(env: Optional[Mapping[str, str]] = None) -> Path:
    """Persistent root for run results: $HERMES_EVOLUTION_HOME, else
    ~/.local/share/hermes-evolution. Run dirs live under <home>/runs/."""
    env = os.environ if env is None else env
    raw = env.get(EVOLUTION_HOME_ENV)
    return Path(raw).expanduser() if raw else Path.home() / ".local" / "share" / "hermes-evolution"


def default_run_dir(run_name: str, env: Optional[Mapping[str, str]] = None) -> Path:
    """<evolution home>/runs/<run_name>; run_name may nest, e.g. kafo-evo-001/live-2."""
    name = Path(run_name)
    if not run_name or name.is_absolute() or ".." in name.parts:
        raise ValueError(f"--run-name must be a relative name without '..': {run_name!r}")
    return evolution_home(env) / "runs" / name


def _under_tmpdir(path: Path, env: Optional[Mapping[str, str]] = None) -> bool:
    """True when ``path`` sits in a temporary directory the OS may clear."""
    env = os.environ if env is None else env
    roots = {tempfile.gettempdir(), "/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp"}
    if env.get("TMPDIR"):
        roots.add(env["TMPDIR"])
    target = path.expanduser().resolve()
    for root in roots:
        try:
            target.relative_to(Path(root).resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


def _inside_repo(path: Path) -> bool:
    repo = Path(__file__).resolve().parents[2]
    try:
        path.resolve().relative_to(repo)
        return True
    except ValueError:
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", default=None,
                   help="state, ledger and report (outside the repo); default "
                        f"${EVOLUTION_HOME_ENV}/runs/<--run-name>")
    p.add_argument("--run-name", default=None,
                   help="run dir name under the persistent root, e.g. kafo-evo-001/live-2 "
                        f"(${EVOLUTION_HOME_ENV}, default ~/.local/share/hermes-evolution)")
    p.add_argument("--skill-file", required=True, help="baseline skill text (read only)")
    p.add_argument("--skill-source", default=None, help="provenance note for the skill file")
    p.add_argument("--stages", default=",".join(STAGES))
    p.add_argument("--live", action="store_true", help="call Hermes for real (spends money)")
    p.add_argument("--model", default=None)
    p.add_argument("--provider", default=None)
    p.add_argument("--hermes-repo", default=None)
    p.add_argument("--hermes-python", default=None)
    p.add_argument("--stop-usd", type=float, default=None)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--train-episodes", type=int, default=300)
    p.add_argument("--gepa-allow-metered", action="store_true",
                   help="live GEPA bills ANTHROPIC_API_KEY (metered), not the Hermes provider")
    p.add_argument("--gepa-via-hermes", action="store_true",
                   help="route GEPA task and reflection calls through Hermes (same model "
                        "and provider as the other stages; no metered API key)")
    p.add_argument("--gepa-task-model", default="anthropic/claude-sonnet-5-5")
    p.add_argument("--gepa-reflection-model", default="anthropic/claude-sonnet-5-5")
    p.add_argument("--gepa-max-metric-calls", type=int, default=150)
    args = p.parse_args(argv)

    stages = tuple(s for s in args.stages.split(",") if s)
    unknown = set(stages) - set(STAGES)
    if unknown:
        p.error(f"unknown stages {sorted(unknown)}")
    if args.run_dir and args.run_name:
        p.error("give --run-dir or --run-name, not both")
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
    elif args.run_name:
        try:
            run_dir = default_run_dir(args.run_name)
        except ValueError as exc:
            p.error(str(exc))
    else:
        p.error(f"give --run-name (stored under ${EVOLUTION_HOME_ENV}/runs, default "
                "~/.local/share/hermes-evolution/runs) or an explicit --run-dir")
    if _inside_repo(run_dir):
        p.error("--run-dir must be outside the repository")
    if _under_tmpdir(run_dir):
        print(f"WARNING: run dir {run_dir} is in a temporary directory the OS may clear; "
              f"results and the registry could be lost. Use --run-name to store it under "
              f"{evolution_home() / 'runs'} (${EVOLUTION_HOME_ENV}).", file=sys.stderr)
    if args.live and (not args.model or args.stop_usd is None):
        p.error("--live needs explicit --model and --stop-usd")
    stop = DEFAULT_STOP_USD if args.stop_usd is None else args.stop_usd
    if args.gepa_via_hermes and args.gepa_allow_metered:
        p.error("--gepa-via-hermes and --gepa-allow-metered are mutually exclusive")
    gepa_factory = None
    gepa_via_hermes = False
    if "gepa" in stages:
        if args.gepa_via_hermes:
            gepa_via_hermes = True          # factory built once the executor exists
        elif not args.live:
            try:
                import dspy  # noqa: F401
                gepa_factory = stub_gepa_factory(args.gepa_max_metric_calls)
            except ImportError:
                gepa_factory = None
        elif args.gepa_allow_metered:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                p.error("live GEPA needs ANTHROPIC_API_KEY in the environment")
            gepa_factory = live_gepa_factory(args.gepa_task_model, args.gepa_reflection_model)
        else:
            p.error("live GEPA needs --gepa-via-hermes (subscription) or "
                    "--gepa-allow-metered (metered API key), or leave gepa out of --stages")
    if args.live:
        try:
            from . import admission as adm
            if adm.REVIEW_PROMPT != REVIEW_PROMPT:
                p.error("REVIEW_PROMPT differs from admission.REVIEW_PROMPT")
        except ImportError:
            if "admission" in stages:
                p.error("the admission stage needs gate 6 (evolution.adaptive.admission)")

    run_dir.mkdir(parents=True, exist_ok=True)
    skill_text = Path(args.skill_file).read_text(encoding="utf-8")
    executor = HermesExecutor(hermes_repo=args.hermes_repo, python=args.hermes_python,
                              model=args.model or "stub-model", provider=args.provider,
                              budget=Budget(), allow_live=True,
                              runner=None if args.live else stub_runner)
    if gepa_via_hermes:
        gepa_factory = hermes_gepa_factory(executor)
    ledger = SpendLedger(run_dir / "ledger.jsonl", stop_usd=stop)
    run = FullRun(run_dir=run_dir, skill_text=skill_text,
                  skill_source=args.skill_source or str(args.skill_file), executor=executor,
                  ledger=ledger, live=args.live, seeds=args.seeds,
                  train_episodes=args.train_episodes)
    try:
        report = run.run(stages, gepa_factory=gepa_factory,
                         gepa_max_metric_calls=args.gepa_max_metric_calls)
    except RunMismatch as exc:
        p.error(str(exc))
    spend = report["spend"]
    print(f"mode={report['mode']} spent=${spend['spent_usd']:.4f} stop=${spend['stop_usd']:.2f}")
    for stage, info in report["stages"].items():
        print(f"  {stage}: {info['status']}" + (f" ({info.get('reason')})"
                                                 if info.get("reason") else ""))
    for stage, entry in spend["by_stage"].items():
        print(f"  ledger {stage}: {entry['calls']} calls, ${entry['cost_usd']:.4f}")
    print(f"report: {run_dir / 'report.json'}")
    return 0 if all(i["status"] in ("done", "skipped") for i in report["stages"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())

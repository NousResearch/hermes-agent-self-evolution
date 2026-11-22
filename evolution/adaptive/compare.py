"""Offline policy comparison harness (KAFO-EVO-001 gate 4).

Compares fixed, random, epsilon-greedy and LinUCB policies under identical
conditions and reports utility, quality, cost, latency, violation rate and
regret with seeded bootstrap confidence intervals.

Fairness rules enforced here:

* Every policy gets the same task sequence for a given seed, the same
  executor (so the same per-execution budget) and the same number of episodes.
* Each (task, strategy) pair is executed and scored ONCE per comparison and
  the result is shared by every policy (common random numbers). This also
  gives an exact per-task oracle, so regret = best reward on that task minus
  the reward obtained.
* Policies only see a ``Context`` built from the task's observable metadata
  (``BenchmarkItem.observable``) before deciding. Ground truth goes to the
  quality function only.
* Only the ``train`` and ``validation`` splits are accepted. Anything else,
  including the sealed ``test`` split, is refused.

Pluggable inputs: ``Benchmark`` (``SyntheticBenchmark``, or ``CaseBenchmark``
over gate 2 ``evolution.adaptive.benchmark.load_split`` cases), any executor
with ``execute(task, strategy)`` (``ScriptedExecutor`` here; PR #3's
``FakeExecutor``/``HermesExecutor`` once merged) and a ``quality_fn(item,
outcome) -> [0, 1]`` (gate 3 verifier once merged). Reward logic is
``core.score_outcome``, unchanged.

Results from a fake/scripted executor or a synthetic benchmark are labelled
SYNTHETIC. They are not evidence that Hermes improves on any real task.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import random
import statistics
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from .benchmark import optimisation_phase
from .core import ExperienceStore, Reward, score_outcome
from .policies import (
    Context, EpsilonGreedyPolicy, FixedPolicy, LinUCBPolicy, Policy, RandomPolicy,
)

ALLOWED_SPLITS = ("train", "validation")
SYNTHETIC_NOTICE = ("SYNTHETIC RESULTS: scripted/fake executor and/or synthetic benchmark. "
                    "Not evidence that Hermes improves on any real task.")
METRICS = ("utility", "quality", "cost_usd", "latency_s", "violation_rate", "regret")


class SealedSplitRefused(PermissionError):
    """Raised when anything other than an open split reaches the harness."""


def check_split(split: str) -> str:
    if split not in ALLOWED_SPLITS:
        raise SealedSplitRefused(
            f"split {split!r} refused: policy comparison may only use {ALLOWED_SPLITS}; "
            "the sealed test split is reserved for final evaluation")
    return split


# ------------------------------------------------------------------ benchmark

@dataclass(frozen=True)
class BenchmarkItem:
    """One task. ``observable`` is all a policy may condition on."""
    task_id: str
    prompt: str
    split: str
    observable: Mapping[str, Any]
    ground_truth: Any = None   # for the quality function only

    def __post_init__(self):
        if not self.task_id or not isinstance(self.task_id, str):
            raise ValueError("task_id must be a nonempty string")
        check_split(self.split)
        object.__setattr__(self, "observable", MappingProxyType(dict(self.observable)))


@dataclass(frozen=True)
class HarnessTask:
    """What the executor receives (same fields as PR #3 ``executor.Task``)."""
    task_id: str
    prompt: str


class Benchmark(Protocol):
    name: str
    synthetic: bool

    def items(self, split: str) -> Sequence[BenchmarkItem]: ...


DEFAULT_REVIEW_PROMPT = (
    "Review the following Python code. Report each defect with line_start, "
    "line_end and category.\n\n```python\n{code}\n```\n")


def _case_field(case, name: str):
    ctx = getattr(case, "context", None)
    if isinstance(ctx, Mapping) and name in ctx:
        return ctx[name]
    if ctx is not None and hasattr(ctx, name):
        return getattr(ctx, name)
    return getattr(case, name)


class CaseBenchmark:
    """Benchmark over code-review cases (gate 2 ``load_split`` or gate 3
    ``verifier.load_cases``). Refuses construction if any case is outside
    the open splits, so sealed test cases never enter the comparison process.
    The gate 2 v1 dataset is hand-written synthetic data, hence the default.
    """

    def __init__(self, cases, *, name: str = "python_review", synthetic: bool = True,
                 prompt_template: str = DEFAULT_REVIEW_PROMPT):
        self.name, self.synthetic = name, synthetic
        self._items: dict[str, list[BenchmarkItem]] = {s: [] for s in ALLOWED_SPLITS}
        seen = set()
        for case in cases:
            split = check_split(case.split)
            if case.case_id in seen:
                raise ValueError(f"duplicate case_id {case.case_id!r}")
            seen.add(case.case_id)
            self._items[split].append(BenchmarkItem(
                task_id=case.case_id, prompt=prompt_template.format(code=case.code),
                split=split, ground_truth=case,
                observable={"difficulty": _case_field(case, "difficulty"),
                            "domain": _case_field(case, "domain"),
                            "n_lines": len(case.code.splitlines())}))

    def items(self, split: str) -> Sequence[BenchmarkItem]:
        return tuple(self._items[check_split(split)])


# -------------------------------------------------------------------- context

@dataclass(frozen=True)
class ContextEncoder:
    """Observable metadata -> ``Context``: bias + one-hot categoricals + scaled numerics.

    An unseen category encodes as all zeros for that field. The vocabulary is
    fitted on train-split observables only (never outcomes or labels).
    """
    key_fields: tuple[str, ...]
    categories: Mapping[str, tuple[str, ...]]
    numeric: Mapping[str, float] = field(default_factory=dict)  # field -> divisor

    def __post_init__(self):
        if not self.key_fields:
            raise ValueError("key_fields cannot be empty")
        for name, scale in self.numeric.items():
            if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"scale for {name!r} must be a positive finite number")
        object.__setattr__(self, "categories",
                           MappingProxyType({k: tuple(v) for k, v in self.categories.items()}))
        object.__setattr__(self, "numeric", MappingProxyType(dict(self.numeric)))

    @property
    def dim(self) -> int:
        return 1 + sum(len(v) for v in self.categories.values()) + len(self.numeric)

    def encode(self, observable: Mapping[str, Any]) -> Context:
        key = "/".join(str(observable[f]) for f in self.key_fields)
        features = [1.0]
        for name, values in self.categories.items():
            features.extend(1.0 if observable.get(name) == v else 0.0 for v in values)
        for name, scale in self.numeric.items():
            features.append(float(observable[name]) / scale)
        return Context(key=key, features=tuple(features))

    @classmethod
    def fit(cls, train_items: Sequence[BenchmarkItem], *, key_fields: Sequence[str],
            categorical: Sequence[str], numeric: Optional[Mapping[str, float]] = None
            ) -> "ContextEncoder":
        if any(item.split != "train" for item in train_items):
            raise ValueError("fit the encoder on train-split items only")
        categories = {name: tuple(sorted({str(i.observable[name]) for i in train_items}))
                      for name in categorical}
        return cls(tuple(key_fields), categories, dict(numeric or {}))


# -------------------------------------------------------------------- scoring

@dataclass(frozen=True)
class Scoring:
    """Fixed references for the cost and latency ratios in ``core.score_outcome``."""
    reference_cost_usd: float
    reference_latency_s: float

    def __post_init__(self):
        for name in ("reference_cost_usd", "reference_latency_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class StepScore:
    reward: float
    quality: float
    cost_usd: float
    latency_s: float
    violation: bool


QualityFn = Callable[[BenchmarkItem, Any], float]


class OutcomeTable:
    """Executes and scores each (task, strategy) pair once; shared by all policies."""

    def __init__(self, executor, catalogue: Mapping[str, Any], quality_fn: QualityFn,
                 scoring: Scoring):
        for key, strategy in catalogue.items():
            if getattr(strategy, "strategy_id", key) != key:
                raise ValueError(f"catalogue key {key!r} != strategy_id {strategy.strategy_id!r}")
        self.executor, self.catalogue = executor, MappingProxyType(dict(catalogue))
        self.quality_fn, self.scoring = quality_fn, scoring
        self._cache: dict[tuple[str, str], StepScore] = {}

    @property
    def executions(self) -> int:
        return len(self._cache)

    def get(self, item: BenchmarkItem, strategy_id: str) -> StepScore:
        key = (item.task_id, strategy_id)
        if key not in self._cache:
            outcome = self.executor.execute(HarnessTask(item.task_id, item.prompt),
                                            self.catalogue[strategy_id])
            cost = getattr(outcome, "cost_usd", None)
            if cost is None:
                raise ValueError(f"cost unknown for {key}; refusing to score it as free")
            quality = self.quality_fn(item, outcome)
            violation = bool(getattr(outcome, "violation"))
            reward = score_outcome(Reward(
                quality=quality, cost_ratio=cost / self.scoring.reference_cost_usd,
                time_ratio=outcome.latency_s / self.scoring.reference_latency_s,
                violation=violation))
            self._cache[key] = StepScore(reward, float(quality), float(cost),
                                         float(outcome.latency_s), violation)
        return self._cache[key]

    def oracle(self, item: BenchmarkItem) -> float:
        return max(self.get(item, s).reward for s in self.catalogue)


# ---------------------------------------------------------------------- runs

@dataclass(frozen=True)
class PolicySpec:
    name: str
    kind: str                         # "fixed" | "random" | "adaptive"
    build: Callable[[int], Policy]    # seed -> fresh policy

    def __post_init__(self):
        if self.kind not in ("fixed", "random", "adaptive"):
            raise ValueError("kind must be fixed, random or adaptive")


def default_policy_specs(strategies: Sequence[str], dim: int, *, epsilon: float = 0.15,
                         alpha: float = 1.0, ridge: float = 1.0) -> list[PolicySpec]:
    strategies = tuple(strategies)
    specs = [PolicySpec(f"fixed:{s}", "fixed", lambda seed, s=s: FixedPolicy(strategies, s))
             for s in strategies]
    specs.append(PolicySpec("random", "random", lambda seed: RandomPolicy(strategies, seed)))
    specs.append(PolicySpec("epsilon_greedy", "adaptive",
                            lambda seed: EpsilonGreedyPolicy(strategies, epsilon, seed)))
    specs.append(PolicySpec("linucb", "adaptive",
                            lambda seed: LinUCBPolicy(strategies, dim, alpha, ridge, seed)))
    return specs


@dataclass(frozen=True)
class StepRecord:
    policy: str
    policy_version: str
    phase: str
    seed: int
    step: int
    task_id: str
    context_key: str
    strategy: str
    propensity: float
    reward: float
    quality: float
    cost_usd: float
    latency_s: float
    violation: bool
    oracle_reward: Optional[float]
    regret: Optional[float]


def run_policy(spec_name: str, policy: Policy, sequence: Sequence[BenchmarkItem],
               encoder: ContextEncoder, table: OutcomeTable, *, phase: str, seed: int,
               learn: bool, oracle: bool) -> list[StepRecord]:
    records = []
    for step, item in enumerate(sequence):
        context = encoder.encode(item.observable)        # observable before deciding
        strategy, propensity = policy.select(context)
        if strategy not in table.catalogue:
            raise ValueError(f"policy chose unknown strategy {strategy!r}")
        if not 0 < propensity <= 1:
            raise ValueError(f"invalid propensity {propensity!r}")
        score = table.get(item, strategy)
        if learn:
            policy.update(context, strategy, score.reward)
        best = table.oracle(item) if oracle else None
        records.append(StepRecord(
            spec_name, policy.version, phase, seed, step, item.task_id, context.key,
            strategy, propensity, score.reward, score.quality, score.cost_usd,
            score.latency_s, score.violation, best,
            None if best is None else best - score.reward))
    return records


# ----------------------------------------------------------------- statistics

def _quantile(sorted_values: Sequence[float], q: float) -> float:
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def bootstrap_ci(values: Sequence[float], *, n_resamples: int, confidence: float,
                 seed: str) -> dict:
    """Mean and percentile bootstrap interval of the mean (seeded, resampling values)."""
    values = [float(v) for v in values]
    if not values:
        raise ValueError("no values")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": 1}
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values)))
                   for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    return {"mean": mean, "ci_low": _quantile(means, alpha),
            "ci_high": _quantile(means, 1 - alpha), "n": len(values)}


def verdict(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "improvement"
    if ci_high < 0:
        return "worse"
    return "no improvement detected"


def _seed_metrics(records: Sequence[StepRecord]) -> dict:
    out = {"utility": statistics.fmean(r.reward for r in records),
           "quality": statistics.fmean(r.quality for r in records),
           "cost_usd": statistics.fmean(r.cost_usd for r in records),
           "latency_s": statistics.fmean(r.latency_s for r in records),
           "violation_rate": statistics.fmean(float(r.violation) for r in records)}
    if records and records[0].regret is not None:
        out["regret"] = statistics.fmean(r.regret for r in records)
        out["cumulative_regret"] = sum(r.regret for r in records)
    return out


# ------------------------------------------------------------------- compare

@dataclass
class ComparisonResult:
    report: dict
    records: list[StepRecord]


def compare(benchmark: Benchmark, executor, quality_fn: QualityFn,
            catalogue: Mapping[str, Any], encoder: ContextEncoder,
            specs: Sequence[PolicySpec], *, seeds: Sequence[int], train_episodes: int,
            scoring: Scoring, oracle: bool = True, n_resamples: int = 2000,
            confidence: float = 0.95, bootstrap_seed: int = 0,
            store: Optional[ExperienceStore] = None,
            skill_version: str = "catalogue/unversioned") -> ComparisonResult:
    """Run every policy on identical task sequences per seed; train online, then
    evaluate the trained policy (no further learning) on the validation split.

    Runs inside ``benchmark.optimisation_phase()``: nothing called from here
    (benchmark, executor, quality function) can load or open the sealed split.
    """
    with optimisation_phase():
        return _compare(benchmark, executor, quality_fn, catalogue, encoder, specs,
                        seeds=seeds, train_episodes=train_episodes, scoring=scoring,
                        oracle=oracle, n_resamples=n_resamples, confidence=confidence,
                        bootstrap_seed=bootstrap_seed, store=store,
                        skill_version=skill_version)


def _compare(benchmark: Benchmark, executor, quality_fn: QualityFn,
            catalogue: Mapping[str, Any], encoder: ContextEncoder,
            specs: Sequence[PolicySpec], *, seeds: Sequence[int], train_episodes: int,
            scoring: Scoring, oracle: bool = True, n_resamples: int = 2000,
            confidence: float = 0.95, bootstrap_seed: int = 0,
            store: Optional[ExperienceStore] = None,
            skill_version: str = "catalogue/unversioned") -> ComparisonResult:
    """Run every policy on identical task sequences per seed; train online, then
    evaluate the trained policy (no further learning) on the validation split."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of unique ints")
    if train_episodes < 1:
        raise ValueError("train_episodes must be positive")
    if not 0 < confidence < 1 or n_resamples < 1:
        raise ValueError("invalid bootstrap settings")
    names = [s.name for s in specs]
    if not specs or len(set(names)) != len(names):
        raise ValueError("policy specs must have unique names")
    train_items = list(benchmark.items("train"))
    validation_items = list(benchmark.items("validation"))
    if not train_items:
        raise ValueError("benchmark has no train items")
    table = OutcomeTable(executor, catalogue, quality_fn, scoring)

    records: list[StepRecord] = []
    per_seed: dict[str, dict[str, list[dict]]] = {n: {"train": [], "validation": []}
                                                   for n in names}
    versions: dict[str, str] = {}
    for seed in seeds:
        sequence_rng = random.Random(f"task-sequence:{seed}")
        train_sequence = [sequence_rng.choice(train_items) for _ in range(train_episodes)]
        for spec in specs:
            policy = spec.build(seed)
            if tuple(policy.strategies) != tuple(catalogue):
                raise ValueError(f"{spec.name}: policy strategies differ from catalogue")
            versions[spec.name] = policy.version
            phases = [("train", train_sequence, True)]
            if validation_items:
                phases.append(("validation", validation_items, False))
            for phase, sequence, learn in phases:
                run = run_policy(spec.name, policy, sequence, encoder, table, phase=phase,
                                 seed=seed, learn=learn, oracle=oracle)
                records.extend(run)
                per_seed[spec.name][phase].append(_seed_metrics(run))

    if store is not None:
        for r in records:
            store.record(r.context_key, r.strategy, r.propensity, r.reward,
                         {"policy": r.policy, "phase": r.phase, "seed": r.seed,
                          "step": r.step, "task_id": r.task_id, "quality": r.quality,
                          "cost_usd": r.cost_usd, "latency_s": r.latency_s,
                          "violation": r.violation, "regret": r.regret},
                         policy_version=r.policy_version, skill_version=skill_version)

    def ci(values, *tag):
        return bootstrap_ci(values, n_resamples=n_resamples, confidence=confidence,
                            seed=":".join(map(str, (bootstrap_seed, *tag))))

    policies_report = {}
    for spec in specs:
        entry = {"kind": spec.kind, "version": versions[spec.name]}
        for phase, runs in per_seed[spec.name].items():
            if runs:
                entry[phase] = {m: ci([r[m] for r in runs], spec.name, phase, m)
                                for m in runs[0]}
        policies_report[spec.name] = entry

    fixed = [s.name for s in specs if s.kind == "fixed"]
    best_fixed = max(fixed, key=lambda n: (policies_report[n]["train"]["utility"]["mean"], n)
                     ) if fixed else None
    baselines = [b for b in (best_fixed, *(s.name for s in specs if s.kind == "random")) if b]
    comparisons = []
    for spec in (s for s in specs if s.kind == "adaptive"):
        for baseline in baselines:
            for phase in ("train", "validation"):
                ours, theirs = per_seed[spec.name][phase], per_seed[baseline][phase]
                if not ours:
                    continue
                diffs = [a["utility"] - b["utility"] for a, b in zip(ours, theirs)]
                stats = ci(diffs, "diff", spec.name, baseline, phase)
                comparisons.append({"policy": spec.name, "baseline": baseline, "phase": phase,
                                    "metric": "utility", "mean_diff": stats["mean"],
                                    "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                                    "verdict": verdict(stats["ci_low"], stats["ci_high"])})

    synthetic = bool(getattr(benchmark, "synthetic", True)) \
        or getattr(executor, "backend", None) != "hermes"
    report = {
        "synthetic": synthetic,
        "notice": SYNTHETIC_NOTICE if synthetic else "",
        "benchmark": getattr(benchmark, "name", type(benchmark).__name__),
        "executor_backend": getattr(executor, "backend", type(executor).__name__),
        "config": {"seeds": list(seeds), "train_episodes": train_episodes,
                   "train_items": len(train_items), "validation_items": len(validation_items),
                   "strategies": list(catalogue), "context_dim": encoder.dim,
                   "context_key_fields": list(encoder.key_fields),
                   "scoring": asdict(scoring), "oracle": oracle,
                   "bootstrap": {"resamples": n_resamples, "confidence": confidence,
                                 "seed": bootstrap_seed, "unit": "seed-level run mean"},
                   "validation_learning": False},
        "executions": table.executions,
        "best_fixed_baseline": best_fixed,
        "policies": policies_report,
        "comparisons": comparisons,
    }
    return ComparisonResult(report, records)


def format_markdown(report: dict) -> str:
    lines = []
    if report["synthetic"]:
        lines += [f"**{report['notice']}**", ""]
    cfg = report["config"]
    lines += [f"Benchmark `{report['benchmark']}`, executor `{report['executor_backend']}`, "
              f"{len(cfg['seeds'])} seeds x {cfg['train_episodes']} train episodes, "
              f"{cfg['validation_items']} validation items, "
              f"{int(cfg['bootstrap']['confidence'] * 100)}% bootstrap CI over seeds.", ""]

    def cell(stat):
        return f"{stat['mean']:.3f} [{stat['ci_low']:.3f}, {stat['ci_high']:.3f}]"

    for phase in ("train", "validation"):
        rows = [(n, p[phase]) for n, p in report["policies"].items() if phase in p]
        if not rows:
            continue
        cols = [m for m in METRICS if m in rows[0][1]]
        lines += [f"### {phase}", "", "| policy | " + " | ".join(cols) + " |",
                  "|---" * (len(cols) + 1) + "|"]
        lines += ["| " + name + " | " + " | ".join(cell(stats[m]) for m in cols) + " |"
                  for name, stats in rows]
        lines.append("")
    lines += [f"Best fixed baseline (by train utility): `{report['best_fixed_baseline']}`", "",
              "| policy | vs | phase | utility diff [CI] | verdict |", "|---|---|---|---|---|"]
    lines += [f"| {c['policy']} | {c['baseline']} | {c['phase']} | {c['mean_diff']:+.3f} "
              f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | {c['verdict']} |"
              for c in report["comparisons"]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .synthetic import (
        REFERENCE_SCORING, STRATEGIES, ScriptedExecutor, SyntheticBenchmark,
        synthetic_catalogue, synthetic_encoder, synthetic_quality,
    )
    parser = argparse.ArgumentParser(
        description="Offline synthetic policy comparison (no network, no Hermes).")
    parser.add_argument("--seeds", type=int, default=20, help="number of replicate seeds")
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--benchmark-seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--json", dest="json_path", help="write the JSON report here")
    parser.add_argument("--db", help="also record every decision in an ExperienceStore")
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.train_episodes < 1 or args.resamples < 1:
        parser.error("--seeds, --train-episodes and --resamples must be positive")
    benchmark = SyntheticBenchmark(seed=args.benchmark_seed)
    encoder = synthetic_encoder(benchmark)
    kwargs = dict(seeds=list(range(args.seeds)), train_episodes=args.train_episodes,
                  scoring=REFERENCE_SCORING, n_resamples=args.resamples,
                  skill_version="synthetic-catalogue/1")
    specs = default_policy_specs(STRATEGIES, encoder.dim)
    executor = ScriptedExecutor(benchmark.response_table())
    if args.db:
        with ExperienceStore(args.db) as store:
            result = compare(benchmark, executor, synthetic_quality, synthetic_catalogue(),
                             encoder, specs, store=store, **kwargs)
    else:
        result = compare(benchmark, executor, synthetic_quality, synthetic_catalogue(),
                         encoder, specs, **kwargs)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(result.report, handle, indent=2, allow_nan=False)
    sys.stdout.write(format_markdown(result.report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

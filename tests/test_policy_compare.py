"""Policy and comparison-harness tests (gate 4). Offline; no network or keys.

Three tests plug in modules from other open PRs (executor #3, verifier #2,
gate 2 benchmark). They use ``pytest.importorskip`` and start running once
those modules are merged into the sprint branch.
"""
import json
import math
from types import SimpleNamespace

import pytest

from evolution.adaptive.compare import (
    ALLOWED_SPLITS, BenchmarkItem, CaseBenchmark, ContextEncoder, OutcomeTable,
    PolicySpec, SealedSplitRefused, Scoring, bootstrap_ci, compare, default_policy_specs,
    format_markdown, main, verdict,
)
from evolution.adaptive.core import ExperienceStore, VIOLATION_REWARD
from evolution.adaptive.policies import (
    Context, EpsilonGreedyPolicy, FixedPolicy, LinUCBPolicy, Policy, RandomPolicy,
)
from evolution.adaptive.synthetic import (
    CORRECT, REFERENCE_SCORING, STRATEGIES, ScriptedExecutor, SyntheticBenchmark,
    synthetic_catalogue, synthetic_encoder, synthetic_quality,
)

S = ("a", "b", "c")
CTX = Context("k", (1.0, 0.0))


# --------------------------------------------------------------- policies

def test_context_validation():
    for bad in [("", (1.0,)), ("k", ()), ("k", [1.0]), ("k", (math.nan,)), ("k", (True,))]:
        with pytest.raises(ValueError):
            Context(*bad)


@pytest.mark.parametrize("policy", [
    FixedPolicy(S, "b"), RandomPolicy(S, seed=1), EpsilonGreedyPolicy(S, 0.2, seed=1),
    LinUCBPolicy(S, dim=2, seed=1)])
def test_policies_share_interface_and_validate_updates(policy):
    assert isinstance(policy, Policy) and policy.version and policy.strategies == S
    for _ in range(10):
        action, propensity = policy.select(CTX)
        assert action in S and 0 < propensity <= 1
        policy.update(CTX, action, 0.5)
    for bad in (math.nan, math.inf, True, "1"):
        with pytest.raises(ValueError):
            policy.update(CTX, "a", bad)
    with pytest.raises(ValueError):
        policy.update(CTX, "zzz", 0.1)
    with pytest.raises(TypeError):
        policy.update("k", "a", 0.1)


def test_fixed_and_random_propensities_and_seeding():
    assert FixedPolicy(S, "c").select(CTX) == ("c", 1.0)
    assert FixedPolicy(S, "c").version == "fixed/1:c"
    with pytest.raises(ValueError):
        FixedPolicy(S, "z")
    a, b = RandomPolicy(S, seed=4), RandomPolicy(S, seed=4)
    picks = [a.select(CTX) for _ in range(50)]
    assert picks == [b.select(CTX) for _ in range(50)]
    assert {p for _, p in picks} == {1 / 3} and {x for x, _ in picks} == set(S)


def test_epsilon_greedy_wraps_existing_bandit():
    policy = EpsilonGreedyPolicy(S, epsilon=0.0, seed=0)
    assert policy.select(CTX) == ("a", 1.0)   # initial sweep, deterministic
    for action, reward in (("a", 0.1), ("b", 0.9), ("c", 0.2)):
        policy.update(CTX, action, reward)
    assert policy.select(CTX) == ("b", 1.0)
    assert policy.version == "epsilon-greedy/1:eps=0"


def _inverse_times(policy, action, xs):
    """Rebuild A = ridge*I + sum x x^T and check A @ A_inv == I."""
    d = policy.dim
    a = [[policy.ridge * (i == j) for j in range(d)] for i in range(d)]
    for x in xs:
        for i in range(d):
            for j in range(d):
                a[i][j] += x[i] * x[j]
    inv = policy.a_inv[action]
    return [[sum(a[i][k] * inv[k][j] for k in range(d)) for j in range(d)] for i in range(d)]


def test_linucb_sherman_morrison_matches_exact_inverse():
    policy = LinUCBPolicy(S, dim=3, alpha=0.5, ridge=2.0)
    xs = [(1.0, 0.5, -1.0), (1.0, 2.0, 0.0), (1.0, -0.3, 0.7)]
    for x in xs:
        policy.update(Context("k", x), "a", 0.3)
    product = _inverse_times(policy, "a", xs)
    for i in range(3):
        for j in range(3):
            assert product[i][j] == pytest.approx(float(i == j), abs=1e-9)
    assert policy.counts == {"a": 3, "b": 0, "c": 0}


def test_linucb_ties_then_learns_context_specific_actions():
    policy = LinUCBPolicy(("a", "b"), dim=3, alpha=0.1, seed=2)
    left, right = Context("left", (1.0, 1.0, 0.0)), Context("right", (1.0, 0.0, 1.0))
    assert policy.select(left)[1] == 0.5           # untrained: all tied, uniform
    for _ in range(30):
        policy.update(left, "a", 0.9)
        policy.update(left, "b", 0.1)
        policy.update(right, "a", 0.1)
        policy.update(right, "b", 0.9)
    assert policy.select(left) == ("a", 1.0)
    assert policy.select(right) == ("b", 1.0)
    with pytest.raises(ValueError):
        policy.select(Context("k", (1.0, 0.0)))     # wrong dimension
    before = [row[:] for row in policy.a_inv["a"]]
    with pytest.raises(ValueError):
        policy.update(left, "a", math.nan)
    assert policy.a_inv["a"] == before              # state untouched on bad reward


def test_linucb_parameter_validation():
    for kwargs in ({"dim": 0}, {"dim": 2, "alpha": -1}, {"dim": 2, "ridge": 0},
                   {"dim": 2, "alpha": math.inf}):
        with pytest.raises(ValueError):
            LinUCBPolicy(S, **kwargs)


# ------------------------------------------------------- splits and context

def test_sealed_and_unknown_splits_refused():
    assert ALLOWED_SPLITS == ("train", "validation")
    for split in ("test", "sealed", "holdout", ""):
        with pytest.raises(SealedSplitRefused):
            BenchmarkItem("t", "p", split, {})
        with pytest.raises(SealedSplitRefused):
            SyntheticBenchmark(n_train=2, n_validation=1).items(split)


def _case(case_id, split, *, nested):
    context = {"difficulty": "simple", "domain": "web"}
    base = dict(case_id=case_id, split=split, code="x = 1\ny = 2\n", defects=())
    if nested:   # gate 2 shape: case.context.difficulty
        return SimpleNamespace(context=SimpleNamespace(**context), **base)
    return SimpleNamespace(**context, **base)   # gate 3 verifier shape: case.difficulty


@pytest.mark.parametrize("nested", [True, False])
def test_case_benchmark_adapts_gate2_and_gate3_cases(nested):
    bench = CaseBenchmark([_case("c1", "train", nested=nested),
                           _case("c2", "validation", nested=nested)])
    (item,) = bench.items("train")
    assert item.observable == {"difficulty": "simple", "domain": "web", "n_lines": 2}
    assert "x = 1" in item.prompt and item.ground_truth.case_id == "c1"
    assert [i.task_id for i in bench.items("validation")] == ["c2"]
    with pytest.raises(SealedSplitRefused):
        CaseBenchmark([_case("c1", "train", nested=nested), _case("t1", "test", nested=nested)])
    with pytest.raises(ValueError):
        CaseBenchmark([_case("c1", "train", nested=nested)] * 2)


def test_context_encoder_uses_observables_only():
    bench = SyntheticBenchmark(n_train=30, n_validation=5)
    encoder = synthetic_encoder(bench)
    assert encoder.dim == 1 + 3 + 3 + 1
    item = bench.items("train")[0]
    context = encoder.encode(item.observable)
    assert context.key == f"{item.observable['difficulty']}/{item.observable['domain']}"
    assert context.features[0] == 1.0 and sum(context.features[1:7]) == 2.0
    unseen = encoder.encode({"difficulty": "alien", "domain": "alien", "n_lines": 50})
    assert unseen.features == (1.0,) + (0.0,) * 6 + (0.5,)
    with pytest.raises(ValueError):
        ContextEncoder.fit(bench.items("validation"), key_fields=("difficulty",),
                           categorical=("difficulty",))


def test_policy_sees_only_the_context_before_deciding():
    bench = SyntheticBenchmark(n_train=10, n_validation=0)
    encoder = synthetic_encoder(bench)
    seen = []

    class Spy:
        strategies, version = STRATEGIES, "spy/1"

        def select(self, context):
            seen.append(("select", context))
            return "direct", 1.0

        def update(self, context, strategy, reward):
            seen.append(("update", context))

    compare(bench, ScriptedExecutor(bench.response_table()), synthetic_quality,
            synthetic_catalogue(), encoder, [PolicySpec("spy", "adaptive", lambda s: Spy())],
            seeds=[0], train_episodes=5, scoring=REFERENCE_SCORING, n_resamples=10)
    assert [kind for kind, _ in seen] == ["select", "update"] * 5
    assert all(type(ctx) is Context for _, ctx in seen)
    selects = [ctx for kind, ctx in seen if kind == "select"]
    updates = [ctx for kind, ctx in seen if kind == "update"]
    assert selects == updates


# ----------------------------------------------------------------- harness

def _small(seeds=(0, 1, 2), episodes=60, **kwargs):
    bench = SyntheticBenchmark(seed=3, n_train=40, n_validation=15)
    encoder = synthetic_encoder(bench)
    executor = ScriptedExecutor(bench.response_table())
    result = compare(bench, executor, synthetic_quality, synthetic_catalogue(), encoder,
                     default_policy_specs(STRATEGIES, encoder.dim), seeds=list(seeds),
                     train_episodes=episodes, scoring=REFERENCE_SCORING, n_resamples=200,
                     **kwargs)
    return bench, executor, result


def test_every_policy_gets_the_same_tasks_and_outcomes():
    bench, executor, result = _small()
    by = {}
    for r in result.records:
        by.setdefault((r.seed, r.phase), {}).setdefault(r.policy, []).append(r.task_id)
    for sequences in by.values():
        assert len({tuple(seq) for seq in sequences.values()}) == 1
    # each (task, strategy) executed exactly once, shared by all policies
    assert len(executor.calls) == len(set(executor.calls)) == result.report["executions"]
    table = {}
    for r in result.records:
        table.setdefault((r.task_id, r.strategy), set()).add(r.reward)
    assert all(len(v) == 1 for v in table.values())


def test_records_carry_propensity_version_and_regret():
    _, _, result = _small()
    versions = {r.policy: r.policy_version for r in result.records}
    assert versions["linucb"].startswith("linucb/1") and versions["random"] == "random/1"
    assert versions["fixed:direct"] == "fixed/1:direct"
    for r in result.records:
        assert 0 < r.propensity <= 1
        assert r.regret == pytest.approx(r.oracle_reward - r.reward) and r.regret >= 0
        assert r.reward == VIOLATION_REWARD if r.violation else r.reward > VIOLATION_REWARD
    assert {r.propensity for r in result.records if r.policy == "random"} == {1 / 3}


def test_report_is_deterministic_labelled_and_consistent():
    _, _, first = _small()
    _, _, second = _small()
    assert json.dumps(first.report, sort_keys=True) == json.dumps(second.report, sort_keys=True)
    report = first.report
    assert report["synthetic"] is True and "SYNTHETIC" in report["notice"]
    assert report["best_fixed_baseline"].startswith("fixed:")
    for entry in report["policies"].values():
        for phase in ("train", "validation"):
            for stat in entry[phase].values():
                assert stat["ci_low"] - 1e-12 <= stat["mean"] <= stat["ci_high"] + 1e-12
    pairs = {(c["policy"], c["baseline"], c["phase"]) for c in report["comparisons"]}
    assert ("linucb", "random", "validation") in pairs
    assert ("epsilon_greedy", report["best_fixed_baseline"], "train") in pairs
    for c in report["comparisons"]:
        assert c["verdict"] == verdict(c["ci_low"], c["ci_high"])
    markdown = format_markdown(report)
    assert markdown.startswith("**SYNTHETIC") and "| linucb |" in markdown
    json.dumps(report, allow_nan=False)


def test_without_oracle_regret_is_omitted():
    _, _, result = _small(seeds=(0,), episodes=10, oracle=False)
    assert all(r.regret is None for r in result.records)
    assert "regret" not in result.report["policies"]["linucb"]["train"]


def test_decisions_recorded_in_experience_store(tmp_path):
    with ExperienceStore(tmp_path / "exp.db") as store:
        _, _, result = _small(seeds=(0,), episodes=10, store=store, skill_version="syn/1")
        rows = store.rows()
    assert len(rows) == len(result.records)
    assert {r["policy_version"] for r in rows} >= {"random/1", "fixed/1:direct"}
    assert all(r["skill_version"] == "syn/1" and 0 < r["propensity"] <= 1 for r in rows)
    assert json.loads(rows[0]["feedback_json"])["phase"] == "train"


def test_compare_input_validation():
    bench = SyntheticBenchmark(n_train=5, n_validation=2)
    encoder = synthetic_encoder(bench)
    args = (bench, ScriptedExecutor(bench.response_table()), synthetic_quality,
            synthetic_catalogue(), encoder, default_policy_specs(STRATEGIES, encoder.dim))
    for bad in ({"seeds": []}, {"seeds": [1, 1]}, {"train_episodes": 0},
                {"confidence": 1.0}):
        kwargs = {"seeds": [0], "train_episodes": 5, "scoring": REFERENCE_SCORING, **bad}
        with pytest.raises(ValueError):
            compare(*args, **kwargs)
    wrong = [PolicySpec("x", "fixed", lambda s: FixedPolicy(("a",), "a"))]
    with pytest.raises(ValueError, match="differ from catalogue"):
        compare(*args[:5], wrong, seeds=[0], train_episodes=1, scoring=REFERENCE_SCORING)
    with pytest.raises(ValueError):
        Scoring(0, 1)


def test_unknown_cost_is_never_scored_as_free():
    bench = SyntheticBenchmark(n_train=2, n_validation=0)
    item = bench.items("train")[0]

    class NoCost:
        def execute(self, task, strategy):
            return SimpleNamespace(output=CORRECT, cost_usd=None, latency_s=1.0,
                                   violation=False)
    table = OutcomeTable(NoCost(), synthetic_catalogue(), synthetic_quality, REFERENCE_SCORING)
    with pytest.raises(ValueError, match="cost unknown"):
        table.get(item, "direct")


def test_bootstrap_and_verdict():
    stats = bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_resamples=500, confidence=0.9, seed="s")
    assert stats["mean"] == 2.5 and stats["ci_low"] < 2.5 < stats["ci_high"]
    assert stats == bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_resamples=500, confidence=0.9,
                                 seed="s")
    assert bootstrap_ci([0.7], n_resamples=10, confidence=0.9, seed="s")["ci_low"] == 0.7
    assert verdict(0.01, 0.2) == "improvement"
    assert verdict(-0.2, -0.01) == "worse"
    assert verdict(-0.1, 0.1) == "no improvement detected"


def test_cli_prints_labelled_report(tmp_path, capsys):
    out = tmp_path / "report.json"
    assert main(["--seeds", "2", "--train-episodes", "20", "--resamples", "50",
                 "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("**SYNTHETIC RESULTS")
    assert json.loads(out.read_text())["synthetic"] is True


# ------------------------------------- plug-ins from other open PRs (skipped until merged)

def test_harness_runs_unchanged_with_pr3_fake_executor():
    executor_mod = pytest.importorskip("evolution.adaptive.executor")
    bench = SyntheticBenchmark(seed=3, n_train=20, n_validation=5)
    encoder = synthetic_encoder(bench)
    fake = executor_mod.FakeExecutor({k: executor_mod.FakeResponse(**v)
                                      for k, v in bench.response_table().items()})
    catalogue = {s: executor_mod.Strategy(s) for s in STRATEGIES}
    kwargs = dict(seeds=[0, 1], train_episodes=30, scoring=REFERENCE_SCORING, n_resamples=50)
    specs = default_policy_specs(STRATEGIES, encoder.dim)
    with_fake = compare(bench, fake, synthetic_quality, catalogue, encoder, specs, **kwargs)
    scripted = compare(bench, ScriptedExecutor(bench.response_table()), synthetic_quality,
                       synthetic_catalogue(), encoder, specs, **kwargs)
    assert with_fake.report["policies"] == scripted.report["policies"]
    assert with_fake.report["synthetic"] is True


def test_gate3_verifier_plugs_in_as_quality_fn():
    verifier = pytest.importorskip("evolution.adaptive.verifier")
    raw = {"case_id": "v1", "split": "train", "language": "python",
           "code": "a = 1\nb = a / 0\n", "context": {"difficulty": "simple", "domain": "math"},
           "defects": [{"defect_id": "v1-d1", "category": "logic_error", "line_start": 2,
                        "line_end": 2, "severity": "high", "description": "division by zero"}]}
    case = verifier.parse_case(raw)
    bench = CaseBenchmark([case])
    (item,) = bench.items("train")
    hit = SimpleNamespace(output=json.dumps([{"line_start": 2, "line_end": 2,
                                              "category": "logic_error"}]))

    def quality_fn(item, outcome):
        return verifier.quality(item.ground_truth, json.loads(outcome.output))
    assert quality_fn(item, hit) == 1.0
    assert quality_fn(item, SimpleNamespace(output="[]")) == 0.0


def test_gate2_open_splits_plug_in():
    # Never opens the sealed file; refusal of test-split cases is covered above.
    benchmark = pytest.importorskip("evolution.adaptive.benchmark")
    cases = list(benchmark.load_split("train")) + list(benchmark.load_split("validation"))
    bench = CaseBenchmark(cases)
    assert bench.items("train") and bench.items("validation")
    encoder = ContextEncoder.fit(bench.items("train"), key_fields=("difficulty",),
                                 categorical=("difficulty",), numeric={"n_lines": 100.0})
    assert encoder.encode(bench.items("validation")[0].observable).key in (
        "simple", "moderate", "complex")

"""Gate 7 wired into the real gate 6 admission path and gate 5 registry.

Offline: FakeExecutor only. Results are SYNTHETIC. With a deterministic fake
executor the post-admission check reproduces admission's own result unless
the catalogue run differs, so the regressions below are scripted: the
``eval:`` arm (used by admission) and the catalogue strategy (used after
admission) are given different responses.
"""
import json

import pytest

from evolution.adaptive import admission as adm
from evolution.adaptive import benchmark as bm
from evolution.adaptive import checkpoint as ck
from evolution.adaptive import evaluation as ev
from evolution.adaptive import registry as reg
from evolution.adaptive.executor import FakeExecutor, FakeResponse, Strategy
from evolution.adaptive.policies import Context, EpsilonGreedyPolicy

BASE = Strategy("reviewer")


@pytest.fixture(scope="module")
def validation():
    return bm.load_split("validation")


@pytest.fixture
def registry(tmp_path):
    with reg.CandidateRegistry(tmp_path / "registry.db") as r:
        yield r


@pytest.fixture
def store(tmp_path):
    with ck.CheckpointStore(tmp_path / "knowledge.sqlite") as s:
        yield s


@pytest.fixture
def pair(registry):
    base = registry.register_baseline("code-review", b"baseline skill\n")
    cand = registry.propose("code-review", b"evolved skill\n", parent_id=base.candidate_id)
    return base, cand


@pytest.fixture
def policy():
    p = EpsilonGreedyPolicy(("direct", "structured"), epsilon=0.1, seed=4)
    for i in range(12):
        c = Context(("simple", "complex")[i % 2], (1.0,))
        s, _ = p.select(c)
        p.update(c, s, 0.7 if s == "structured" else 0.3)
    return p


def truth(case):
    return json.dumps(ev.ground_truth_findings(case))


def factory(validation, specs, phases=None):
    """specs: strategy_id -> output str, callable(case) or FakeResponse. Default '[]'."""
    def make_executor(seed):
        responses = {}
        for case in validation:
            for sid, spec in specs.items():
                value = spec(case) if callable(spec) else spec
                if not isinstance(value, FakeResponse):
                    value = FakeResponse(output=value, cost_usd=0.01, latency_s=2.0)
                responses[(case.case_id, sid)] = value
        executor = FakeExecutor(responses, default=FakeResponse(output="[]", cost_usd=0.01,
                                                                  latency_s=2.0))
        if phases is not None:
            original = executor.execute

            def execute(task, strategy):
                phases.append(bm.in_optimisation_phase())
                return original(task, strategy)
            executor.execute = execute
        return executor
    return make_executor


def ids(pair):
    base, cand = pair
    return {"eval_base": f"eval:reviewer:{base.candidate_id}",
            "eval_cand": f"eval:reviewer:{cand.candidate_id}",
            "cat_base": f"reviewer:{base.candidate_id}",
            "cat_cand": f"reviewer:{cand.candidate_id}"}


def run(registry, store, policy, pair, validation, **specs):
    named = ids(pair)
    make = factory(validation, {named[k]: v for k, v in specs.items()})
    return adm.review_with_rollback(registry, pair[1].candidate_id, base=BASE,
                                    make_executor=make, store=store, policy=policy)


def events(store):
    return [e["event"] for e in store.log()]


# ── kept ─────────────────────────────────────────────────────────────────────

def test_admitted_and_kept_when_post_check_holds(registry, store, policy, pair, validation):
    base, cand = pair
    result = run(registry, store, policy, pair, validation, eval_cand=truth, cat_cand=truth)
    assert result.decision.admitted and result.kept and not result.rolled_back
    assert result.active_skill == cand.candidate_id
    assert registry.get(cand.candidate_id).status == reg.ADMITTED
    assert result.pre_checkpoint.skill_version == base.candidate_id
    assert result.post_checkpoint.skill_version == cand.candidate_id
    assert result.post_checkpoint.status == ck.STATUS_GOOD
    assert store.active_id(policy) == result.post_checkpoint.checkpoint_id
    assert events(store) == ["save", "save", "promote"]
    assert result.synthetic and "SYNTHETIC" in result.notice
    assert adm.admission_evidence_errors(registry, cand.candidate_id) == []


# ── rolled back ──────────────────────────────────────────────────────────────

def _assert_rolled_back(registry, store, policy, pair, result, live_print):
    base, cand = pair
    assert result.decision.admitted and result.rolled_back and not result.kept
    assert result.active_skill == base.candidate_id
    # Skill rollback: retired in the registry, with evidence, by the gate 7 trigger.
    assert registry.get(cand.candidate_id).status == reg.RETIRED
    last = registry.history(cand.candidate_id)[-1]
    assert (last.from_status, last.to_status) == (reg.ADMITTED, reg.RETIRED)
    assert last.decided_by == adm.ROLLBACK_DECIDED_BY
    assert last.evidence["pre_checkpoint_id"] == result.pre_checkpoint.checkpoint_id
    assert last.evidence["post_checkpoint_id"] == result.post_checkpoint.checkpoint_id
    assert last.evidence["margin"] == ck.ROLLBACK_MARGIN
    # Catalogue no longer serves it.
    prompts = [s.system_prompt for s in adm.admitted_catalogue(registry, "code-review", BASE)]
    assert prompts == ["baseline skill\n"]
    # Checkpoint rollback: pointer back on the pre-admission checkpoint, logged.
    assert store.active_id(policy) == result.pre_checkpoint.checkpoint_id
    assert store.load(result.post_checkpoint.checkpoint_id).status == ck.STATUS_ROLLED_BACK
    entry = store.log()[-1]
    assert entry["event"] == "rollback"
    assert entry["checkpoint_id"] == result.pre_checkpoint.checkpoint_id
    details = json.loads(entry["details_json"])
    assert details["candidate_id"] == cand.candidate_id
    assert details["pre_utility"] == last.evidence["pre_utility"]
    assert details["to_skill_version"] == base.candidate_id
    assert ck.canonical_json(ck.policy_state(policy)) == live_print


def test_post_admission_regression_rolls_back(registry, store, policy, pair, validation):
    live = ck.canonical_json(ck.policy_state(policy))
    # Admission sees a strong eval arm; the served catalogue strategy is silent,
    # while the parent's catalogue strategy is correct.
    result = run(registry, store, policy, pair, validation,
                 eval_cand=truth, cat_cand="[]", cat_base=truth)
    assert result.post_validation.utility < (result.pre_validation.utility
                                             - ck.ROLLBACK_MARGIN)
    assert not result.post_validation.critical_violation
    assert "below pre-admission" in result.reason
    _assert_rolled_back(registry, store, policy, pair, result, live)


def test_post_admission_critical_violation_rolls_back(registry, store, policy, pair,
                                                      validation):
    live = ck.canonical_json(ck.policy_state(policy))
    unsafe = lambda case: FakeResponse(output=truth(case), cost_usd=0.01, latency_s=2.0,
                                       tool_calls=("terminal",))
    result = run(registry, store, policy, pair, validation, eval_cand=truth, cat_cand=unsafe)
    assert result.post_validation.critical_violation
    assert "critical violation" in result.reason
    _assert_rolled_back(registry, store, policy, pair, result, live)


def test_small_drop_within_margin_is_kept(registry, store, policy, pair, validation):
    # Post-admission utility equal to the parent's: no drop beyond the margin.
    result = run(registry, store, policy, pair, validation, eval_cand=truth, cat_cand="[]")
    assert result.post_validation.utility == pytest.approx(result.pre_validation.utility)
    assert result.kept and registry.get(pair[1].candidate_id).status == reg.ADMITTED


# ── rejected, errors, partial failure ────────────────────────────────────────

def test_rejected_candidate_is_logged_without_rollback(registry, store, policy, pair,
                                                       validation):
    result = run(registry, store, policy, pair, validation)        # silent candidate
    assert not result.decision.admitted and not result.rolled_back
    assert registry.get(pair[1].candidate_id).status == reg.REJECTED
    assert result.active_skill == pair[0].candidate_id
    assert store.active_id(policy) == result.pre_checkpoint.checkpoint_id
    assert events(store) == ["save", "admission_rejected"]
    assert result.post_checkpoint is None


def test_post_check_error_retires_and_restores(registry, store, policy, pair, validation):
    unknown_cost = FakeResponse(output="[]", cost_usd=None, latency_s=2.0)
    with pytest.raises(ValueError, match="unscorable"):
        run(registry, store, policy, pair, validation, eval_cand=truth,
            cat_cand=unknown_cost)
    cand = pair[1].candidate_id
    assert registry.get(cand).status == reg.RETIRED
    assert registry.history(cand)[-1].decided_by == adm.ROLLBACK_DECIDED_BY
    assert events(store)[-1] == "restore"


def test_candidate_stays_retired_if_checkpoint_rollback_fails(registry, store, policy, pair,
                                                              validation, monkeypatch):
    def broken(*args, **kwargs):
        raise ck.CheckpointError("simulated checkpoint failure")
    monkeypatch.setattr(store, "rollback", broken)
    with pytest.raises(ck.CheckpointError):
        run(registry, store, policy, pair, validation, eval_cand=truth, cat_cand="[]",
            cat_base=truth)
    assert registry.get(pair[1].candidate_id).status == reg.RETIRED   # fail-safe order


def test_refuses_non_proposed_candidate(registry, store, policy, pair, validation):
    with pytest.raises(ValueError, match="proposed"):
        adm.review_with_rollback(registry, pair[0].candidate_id, base=BASE,
                                 make_executor=factory(validation, {}), store=store,
                                 policy=policy)
    assert store.log() == []


def test_callers_cannot_forge_store_events(store):
    with pytest.raises(ValueError):
        store.record_event("rollback", reason="forged")


# ── data access ──────────────────────────────────────────────────────────────

def test_runs_inside_optimisation_phase_on_validation_only(registry, store, policy, pair,
                                                           validation, monkeypatch):
    requested, phases = [], []
    real = bm.load_split
    monkeypatch.setattr(bm, "load_split",
                        lambda split, **kw: (requested.append(split), real(split, **kw))[1])
    named = ids(pair)
    make = factory(validation, {named["eval_cand"]: truth, named["cat_cand"]: truth},
                   phases=phases)
    attempts = bm.sealed_access_attempts()
    adm.review_with_rollback(registry, pair[1].candidate_id, base=BASE, make_executor=make,
                             store=store, policy=policy)
    assert requested and set(requested) == {"validation"}
    assert phases and all(phases) and not bm.in_optimisation_phase()
    assert bm.sealed_access_attempts() == attempts


def test_cannot_reach_held_out_split(registry, store, policy, pair, validation):
    def misbehave(case):
        bm.load_split("test", allow_sealed_test=True)
    with pytest.raises(bm.SealedSplitError):
        run(registry, store, policy, pair, validation, cat_base=misbehave)


def test_rollback_path_stays_inside_optimisation_phase(registry, store, policy, pair,
                                                       validation):
    # Same scripted regression as above, so the post check and rollback both run.
    phases = []
    named = ids(pair)
    make = factory(validation, {named["eval_cand"]: truth, named["cat_cand"]: "[]",
                                named["cat_base"]: truth}, phases=phases)
    attempts = bm.sealed_access_attempts()
    result = adm.review_with_rollback(registry, pair[1].candidate_id, base=BASE,
                                      make_executor=make, store=store, policy=policy)
    assert result.rolled_back
    assert phases and all(phases) and not bm.in_optimisation_phase()
    assert bm.sealed_access_attempts() == attempts


def test_sealed_access_during_post_check_retires_and_restores(registry, store, policy, pair,
                                                              validation):
    # Pre-admission check and admission run normally; only the served candidate
    # strategy (used by the post-admission check) tries to load the sealed split.
    named = ids(pair)
    inner = factory(validation, {named["eval_cand"]: truth, named["cat_cand"]: truth})

    def make_executor(seed):
        executor = inner(seed)
        original = executor.execute

        def execute(task, strategy):
            if strategy.strategy_id == named["cat_cand"]:
                bm.load_split("test", allow_sealed_test=True)
            return original(task, strategy)
        executor.execute = execute
        return executor

    live = ck.canonical_json(ck.policy_state(policy))
    with pytest.raises(bm.SealedSplitError):
        adm.review_with_rollback(registry, pair[1].candidate_id, base=BASE,
                                 make_executor=make_executor, store=store, policy=policy)
    cand = pair[1].candidate_id
    assert registry.get(cand).status == reg.RETIRED
    assert registry.history(cand)[-1].decided_by == adm.ROLLBACK_DECIDED_BY
    assert events(store)[-1] == "restore"
    assert ck.canonical_json(ck.policy_state(policy)) == live
    assert not bm.in_optimisation_phase()

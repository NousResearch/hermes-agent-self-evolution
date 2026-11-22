"""Gate 7: policy/skill checkpoints, restore and rollback.

Offline: no Hermes, no LLM, no network. Benchmark results here are SYNTHETIC
(hand-written cases, fake executor).
"""
import copy
import json
import math
import sqlite3

import pytest

from evolution.adaptive import benchmark as bm
from evolution.adaptive import checkpoint as ck
from evolution.adaptive import evaluation as ev
from evolution.adaptive import verifier as v
from evolution.adaptive.compare import ContextEncoder
from evolution.adaptive.core import ContextualBandit, ExperienceStore
from evolution.adaptive.executor import FakeExecutor, FakeResponse, Strategy
from evolution.adaptive.policies import (
    Context, EpsilonGreedyPolicy, FixedPolicy, LinUCBPolicy, RandomPolicy,
)
from evolution.adaptive.synthetic import REFERENCE_SCORING

S = ("direct", "structured", "tool_assisted")
KEYS = ("simple", "moderate", "complex")
DIM = 3


def ctx(i: int) -> Context:
    key = KEYS[i % len(KEYS)]
    return Context(key, (1.0, float(i % 2), (i % 5) / 5))


def build(kind: str, **kw):
    if kind == "epsilon-greedy":
        return EpsilonGreedyPolicy(S, epsilon=kw.get("epsilon", 0.2), seed=3)
    if kind == "contextual-bandit":
        return ContextualBandit(S, epsilon=0.2, seed=3)
    if kind == "linucb":
        return LinUCBPolicy(S, DIM, alpha=0.5, ridge=1.0, seed=3)
    if kind == "fixed":
        return FixedPolicy(S, "structured")
    if kind == "random":
        return RandomPolicy(S, seed=3)
    raise AssertionError(kind)


KINDS = ("epsilon-greedy", "contextual-bandit", "linucb", "fixed", "random")


def step(policy, i: int) -> str:
    c = ctx(i)
    arg = c.key if isinstance(policy, ContextualBandit) else c
    action, _ = policy.select(arg)
    reward = 0.9 if action == "structured" else 0.2 + 0.01 * (i % 7)
    policy.update(arg, action, reward)
    return action


def train(policy, n: int, start: int = 0):
    for i in range(start, start + n):
        step(policy, i)
    return policy


def fingerprint(policy) -> str:
    return ck.canonical_json(ck.policy_state(policy))


def future(policy, n: int = 25) -> list[str]:
    return [step(policy, 1000 + i) for i in range(n)]


@pytest.fixture
def store():
    s = ck.CheckpointStore(":memory:")
    yield s
    s.close()


def table_rows(conn) -> dict:
    return {t: conn.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall()
            for t in ("policy_checkpoints", "checkpoint_log", "checkpoint_active")}


def rewrite(store, checkpoint_id, mutate, *, update_columns=True):
    """Tamper with the payload and recompute a valid checksum."""
    payload = store.conn.execute("SELECT payload_json FROM policy_checkpoints WHERE id=?",
                                 (checkpoint_id,)).fetchone()[0]
    doc = json.loads(payload)
    mutate(doc)
    new = ck.canonical_json(doc)
    sql = "UPDATE policy_checkpoints SET payload_json=?, sha256=?"
    args = [new, ck._sha256(new)]
    if update_columns:
        sql += ", kind=?, skill_version=?, policy_version=?"
        args += [doc["kind"], doc["skill_version"], doc["policy_version"]]
    store.conn.execute(sql + " WHERE id=?", (*args, checkpoint_id))
    store.conn.commit()


# ── snapshot / restore ───────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", KINDS)
def test_restore_is_exact_and_idempotent(store, kind):
    policy = train(build(kind), 40)
    saved_print = fingerprint(policy)
    reference = copy.deepcopy(policy)
    ckpt = store.save(policy, skill_version="skill/1", label="t")
    assert ckpt.kind == kind and ckpt.policy_version == policy.version
    assert len(ckpt.sha256) == 64

    train(policy, 30, start=40)                   # drift away from the checkpoint
    first = store.restore(policy, ckpt.checkpoint_id)
    assert fingerprint(policy) == saved_print
    second = store.restore(policy, ckpt.checkpoint_id)
    assert fingerprint(policy) == saved_print
    assert first.sha256 == second.sha256 == ckpt.sha256
    assert first.skill_version == "skill/1"
    assert ck.policy_state(policy)[1] == ckpt.state
    # Restored RNG state: the next decisions match the policy at save time.
    assert future(policy) == future(reference)
    assert [e["event"] for e in store.log()] == ["save", "restore", "restore"]
    assert store.active_id(policy) == ckpt.checkpoint_id


@pytest.mark.parametrize("kind", ("epsilon-greedy", "linucb"))
def test_restore_into_fresh_policy_after_reopen(tmp_path, kind):
    path = tmp_path / "knowledge.sqlite"
    policy = train(build(kind), 50)
    reference = copy.deepcopy(policy)
    with ck.CheckpointStore(path) as s:
        cid = s.save(policy, skill_version="skill/7", label="t").checkpoint_id
    fresh = build(kind)
    with ck.CheckpointStore(path) as s:
        restored = s.restore(fresh, cid)
    assert restored.skill_version == "skill/7"
    assert fingerprint(fresh) == fingerprint(reference)
    assert future(fresh) == future(reference)


def test_shares_experience_store_without_changing_its_schema(tmp_path):
    path = tmp_path / "knowledge.sqlite"
    with ExperienceStore(path) as exp:
        exp.record("simple", "direct", 1.0, 0.5, {}, policy_version="p", skill_version="s")
        cs = ck.CheckpointStore(exp)
        cid = cs.save(train(build("linucb"), 5), skill_version="s", label="t").checkpoint_id
        exp.record("simple", "direct", 1.0, 0.4, {}, policy_version="p", skill_version="s")
        assert exp.conn.execute("PRAGMA user_version").fetchone()[0] == ExperienceStore.SCHEMA_VERSION
        assert len(exp.rows()) == 2
        cs.close()                                  # does not close the shared connection
        assert cs.load(cid).checkpoint_id == cid
    with ExperienceStore(path) as exp:
        assert len(exp.rows()) == 2
        assert ck.CheckpointStore(exp).load(cid).skill_version == "s"


def test_unsupported_policy_type_is_refused(store):
    class Custom(EpsilonGreedyPolicy):
        pass
    with pytest.raises(TypeError):
        store.save(Custom(S), skill_version="s", label="t")


# ── corruption detection ─────────────────────────────────────────────────────

def _flip_mean(doc):
    doc["state"]["bandit"]["means"][0][2] += 0.5


CORRUPTIONS = {
    "payload_edited": lambda s, cid: s.conn.execute(
        "UPDATE policy_checkpoints SET payload_json=replace(payload_json, '\"counts\"', "
        "'\"Counts\"') WHERE id=?", (cid,)),
    "checksum_edited": lambda s, cid: s.conn.execute(
        "UPDATE policy_checkpoints SET sha256=? WHERE id=?", ("0" * 64, cid)),
    "truncated_json": lambda s, cid: s.conn.execute(
        "UPDATE policy_checkpoints SET payload_json=substr(payload_json, 1, 40), "
        "sha256=? WHERE id=?",
        (ck._sha256(s.conn.execute("SELECT substr(payload_json, 1, 40) FROM "
                                   "policy_checkpoints WHERE id=?", (cid,)).fetchone()[0]),
         cid)),
    "skill_column_swapped": lambda s, cid: s.conn.execute(
        "UPDATE policy_checkpoints SET skill_version='skill/evil' WHERE id=?", (cid,)),
    "unknown_kind": lambda s, cid: rewrite(s, cid, lambda d: d.update(kind="mystery")),
    "rehashed_bad_count": lambda s, cid: rewrite(
        s, cid, lambda d: d["state"]["bandit"]["counts"][0].__setitem__(2, -4)),
    "rehashed_nonnumeric_mean": lambda s, cid: rewrite(
        s, cid, lambda d: d["state"]["bandit"]["means"][0].__setitem__(2, "high")),
    "rehashed_extra_field": lambda s, cid: rewrite(
        s, cid, lambda d: d["state"].update(reward_weights=[0, 0])),
    "rehashed_bad_rng": lambda s, cid: rewrite(
        s, cid, lambda d: d["state"]["bandit"]["rng"].__setitem__(1, [1, 2, 3])),
    "noncanonical_payload": lambda s, cid: s.conn.execute(
        "UPDATE policy_checkpoints SET payload_json=?, sha256=? WHERE id=?",
        (lambda p: (p, ck._sha256(p)))(
            json.dumps(json.loads(s.conn.execute(
                "SELECT payload_json FROM policy_checkpoints WHERE id=?", (cid,)).fetchone()[0]),
                indent=1)) + (cid,)),
}


@pytest.mark.parametrize("corruption", sorted(CORRUPTIONS))
def test_corruption_is_detected_and_changes_nothing(store, corruption):
    policy = train(build("epsilon-greedy"), 40)
    cid = store.save(policy, skill_version="skill/1", label="t").checkpoint_id
    train(policy, 10, start=40)
    live = fingerprint(policy)
    CORRUPTIONS[corruption](store, cid)
    store.conn.commit()
    before = table_rows(store.conn)

    if not corruption.startswith("rehashed_"):      # record-level: load() catches it
        with pytest.raises(ck.CheckpointCorrupted):
            store.load(cid)
    with pytest.raises(ck.CheckpointCorrupted):      # state-level needs the live policy
        store.restore(policy, cid)

    assert fingerprint(policy) == live
    after = table_rows(store.conn)
    assert after["policy_checkpoints"] == before["policy_checkpoints"]
    assert after["checkpoint_active"] == before["checkpoint_active"]
    new_events = [row[1] for row in after["checkpoint_log"][len(before["checkpoint_log"]):]]
    assert new_events == ["corruption_detected"]


@pytest.mark.parametrize("mutate", [
    lambda d: d["state"]["bandit"].update(epsilon=0.9),          # hyperparameter
    lambda d: d["state"]["bandit"].update(strategies=["direct", "structured", "x"]),
    lambda d: d.update(policy_version="epsilon-greedy/1:eps=0.9"),
])
def test_rehashed_config_change_is_refused(store, mutate):
    """A valid checksum is not enough: a checkpoint cannot reconfigure the policy."""
    policy = train(build("epsilon-greedy"), 20)
    cid = store.save(policy, skill_version="s", label="t").checkpoint_id
    live = fingerprint(policy)
    rewrite(store, cid, lambda d: (mutate(d), d["state"].update(version=d["policy_version"])
                                   if "version" in d["state"] else None))
    with pytest.raises((ck.CheckpointMismatch, ck.CheckpointCorrupted)):
        store.restore(policy, cid)
    assert fingerprint(policy) == live
    assert policy.bandit.epsilon == 0.2


def test_restore_into_differently_configured_policy_is_refused(store):
    cid = store.save(train(build("linucb"), 10), skill_version="s", label="t").checkpoint_id
    other = LinUCBPolicy(S, DIM, alpha=2.0, seed=3)
    with pytest.raises(ck.CheckpointMismatch):
        store.restore(other, cid)
    with pytest.raises(ck.CheckpointMismatch):
        store.restore(build("epsilon-greedy"), cid)
    with pytest.raises(ck.CheckpointMismatch):
        store.restore(LinUCBPolicy(("a", "b"), DIM, alpha=0.5, seed=3), cid)
    assert other.counts == {a: 0 for a in S}


# ── rollback trigger ─────────────────────────────────────────────────────────

def val(utility, violation=False):
    return ck.Validation(utility=utility, critical_violation=violation, n_tasks=10)


def test_trigger_boundary_is_strict():
    pre = 0.5
    at_margin = pre - ck.ROLLBACK_MARGIN
    assert ck.rollback_decision(pre, val(at_margin)) is None
    assert ck.rollback_decision(pre, val(math.nextafter(at_margin, -math.inf)))
    assert ck.rollback_decision(pre, val(pre + 0.3)) is None


def test_trigger_critical_violation_overrides_utility():
    reason = ck.rollback_decision(0.1, val(0.9, violation=True))
    assert reason and "critical violation" in reason


@pytest.mark.parametrize("bad", [math.nan, math.inf, True, "0.5"])
def test_trigger_rejects_non_finite_inputs(bad):
    with pytest.raises((ValueError, TypeError)):
        ck.rollback_decision(bad, val(0.5))
    with pytest.raises(ValueError):
        ck.Validation(utility=bad, critical_violation=False)


def test_margin_is_a_fixed_module_constant():
    assert ck.ROLLBACK_MARGIN == 0.02
    import inspect
    params = inspect.signature(ck.admit_with_rollback).parameters
    assert "margin" not in params


# ── rollback: atomic, logged, last good ──────────────────────────────────────

def _pre_and_candidate(store, kind="epsilon-greedy"):
    policy = train(build(kind), 40)
    pre = store.save(policy, skill_version="skill/1", label="pre", validation=val(0.6))
    pre_print = fingerprint(policy)
    train(policy, 30, start=40)
    cand = store.save(policy, skill_version="skill/2", label="cand",
                      status=ck.STATUS_CANDIDATE, validation=val(0.4), activate=False)
    return policy, pre, pre_print, cand


def test_rollback_restores_last_good_and_logs(store):
    policy, pre, pre_print, cand = _pre_and_candidate(store)
    target = store.rollback(policy, from_checkpoint_id=cand.checkpoint_id,
                            reason="utility drop", details={"post_utility": 0.4})
    assert target.checkpoint_id == pre.checkpoint_id and target.skill_version == "skill/1"
    assert fingerprint(policy) == pre_print
    assert store.load(cand.checkpoint_id).status == ck.STATUS_ROLLED_BACK
    assert store.active_id(policy) == pre.checkpoint_id
    entry = store.log()[-1]
    assert entry["event"] == "rollback" and entry["reason"] == "utility drop"
    assert (entry["checkpoint_id"], entry["from_checkpoint_id"]) == (pre.checkpoint_id,
                                                                     cand.checkpoint_id)
    details = json.loads(entry["details_json"])
    assert details["from_skill_version"] == "skill/2"
    assert details["to_skill_version"] == "skill/1"
    assert details["post_utility"] == 0.4

    # A rolled-back checkpoint is never a restore or rollback target again.
    with pytest.raises(ck.CheckpointError):
        store.restore(policy, cand.checkpoint_id)
    assert cand.checkpoint_id not in store.last_good(cand.kind, cand.policy_version)


def test_rollback_is_idempotent_in_effect(store):
    policy, pre, pre_print, cand = _pre_and_candidate(store)
    store.rollback(policy, from_checkpoint_id=cand.checkpoint_id, reason="r")
    store.rollback(policy, from_checkpoint_id=cand.checkpoint_id, reason="r")
    assert fingerprint(policy) == pre_print
    assert store.active_id(policy) == pre.checkpoint_id


@pytest.mark.parametrize("fail_on", ["rollback_log", "active_pointer"])
def test_rollback_is_atomic(store, monkeypatch, fail_on):
    policy, pre, pre_print, cand = _pre_and_candidate(store)
    live = fingerprint(policy)
    before = table_rows(store.conn)

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("simulated failure")

    if fail_on == "rollback_log":
        real = store._log
        monkeypatch.setattr(store, "_log", lambda event, *a, **k:
                            boom() if event == "rollback" else real(event, *a, **k))
    else:
        monkeypatch.setattr(store, "_set_active", boom)
    with pytest.raises(sqlite3.OperationalError):
        store.rollback(policy, from_checkpoint_id=cand.checkpoint_id, reason="r")

    assert fingerprint(policy) == live                 # in-memory unchanged
    assert table_rows(store.conn) == before             # status/pointer/log unchanged
    assert not store.conn.in_transaction


def test_rollback_skips_corrupt_good_checkpoint(store):
    policy = train(build("linucb"), 20)
    oldest = store.save(policy, skill_version="skill/0", label="oldest")
    oldest_print = fingerprint(policy)
    train(policy, 10, start=20)
    newer = store.save(policy, skill_version="skill/1", label="newer")
    train(policy, 10, start=30)
    cand = store.save(policy, skill_version="skill/2", label="cand",
                      status=ck.STATUS_CANDIDATE, activate=False)
    store.conn.execute("UPDATE policy_checkpoints SET sha256='bad' WHERE id=?",
                       (newer.checkpoint_id,))
    store.conn.commit()

    target = store.rollback(policy, from_checkpoint_id=cand.checkpoint_id, reason="r")
    assert target.checkpoint_id == oldest.checkpoint_id
    assert fingerprint(policy) == oldest_print
    status = dict(store.conn.execute("SELECT id, status FROM policy_checkpoints"))
    assert status[newer.checkpoint_id] == ck.STATUS_CORRUPT
    assert "corruption_detected" in [e["event"] for e in store.log()]


def test_rollback_fails_closed_without_good_checkpoint(store):
    policy, pre, _, cand = _pre_and_candidate(store)
    store.conn.execute("UPDATE policy_checkpoints SET payload_json='{' WHERE id=?",
                       (pre.checkpoint_id,))
    store.conn.commit()
    live = fingerprint(policy)
    with pytest.raises(ck.NoGoodCheckpoint):
        store.rollback(policy, from_checkpoint_id=cand.checkpoint_id, reason="r")
    assert fingerprint(policy) == live
    assert store.load(cand.checkpoint_id).status == ck.STATUS_CANDIDATE


# ── admission flow ───────────────────────────────────────────────────────────

class Validator:
    def __init__(self, *results):
        self.results, self.seen = list(results), []

    def __call__(self, policy):
        self.seen.append(policy)
        return self.results.pop(0)


def poison(policy):
    for i in range(30):
        c = ctx(i)
        policy.update(c, "tool_assisted", 1.0)
    return "skill/2"


def test_admission_kept_when_validation_holds(store):
    policy = train(build("epsilon-greedy"), 40)
    validate = Validator(val(0.60), val(0.59))      # within margin
    result = ck.admit_with_rollback(store, policy, skill_version="skill/1",
                                    admit=poison, validate=validate)
    assert result.admitted and result.skill_version == "skill/2"
    assert result.active_checkpoint.status == ck.STATUS_GOOD
    assert store.active_id(policy) == result.candidate_checkpoint.checkpoint_id
    assert all(p is not policy for p in validate.seen)  # validation used copies
    assert [e["event"] for e in store.log()][-1] == "promote"
    assert result.synthetic


@pytest.mark.parametrize("post", [val(0.50), val(0.70, violation=True)])
def test_admission_rolled_back_on_drop_or_violation(store, post):
    policy = train(build("epsilon-greedy"), 40)
    pre_print = fingerprint(policy)
    result = ck.admit_with_rollback(store, policy, skill_version="skill/1",
                                    admit=poison, validate=Validator(val(0.60), post))
    assert not result.admitted
    assert result.skill_version == "skill/1"
    assert fingerprint(policy) == pre_print
    assert store.active_id(policy) == result.pre_checkpoint.checkpoint_id
    assert result.candidate_checkpoint.status == ck.STATUS_ROLLED_BACK
    entry = store.log()[-1]
    details = json.loads(entry["details_json"])
    assert entry["event"] == "rollback"
    assert details["pre_utility"] == 0.60 and details["margin"] == ck.ROLLBACK_MARGIN
    assert details["critical_violation"] == post.critical_violation


def test_admission_error_restores_pre_checkpoint(store):
    policy = train(build("linucb"), 40)
    pre_print = fingerprint(policy)

    def broken_admit(p):
        poison(p)
        raise RuntimeError("candidate build failed")

    with pytest.raises(RuntimeError):
        ck.admit_with_rollback(store, policy, skill_version="skill/1",
                               admit=broken_admit, validate=Validator(val(0.6)))
    assert fingerprint(policy) == pre_print
    assert store.log()[-1]["event"] == "restore"


# ── validation split only, inside the optimisation phase ─────────────────────

ENCODER = ContextEncoder(("difficulty",), {"difficulty": KEYS}, {"n_lines": 100.0})


@pytest.fixture(scope="module")
def validation_cases():
    return bm.load_benchmark(("validation",))["validation"]


def _harness(cases, misbehave=None, seen_phase=None, tool_violation=False):
    responses = {}
    for case in cases:
        for s in S:
            findings = ev.ground_truth_findings(case) if s == "structured" else []
            tools = ("terminal",) if (tool_violation and s == "tool_assisted") else ()
            responses[(case.case_id, s)] = FakeResponse(
                output=json.dumps(findings), cost_usd=0.03, latency_s=6.0, tool_calls=tools)
    vcases = {c.case_id: ev.to_verifier_case(c) for c in cases}

    def quality_fn(item, outcome):
        if seen_phase is not None:
            seen_phase.append(bm.in_optimisation_phase())
        if misbehave:
            misbehave()
        return v.quality(vcases[item.task_id], json.loads(outcome.output))

    catalogue = {s: Strategy(s) for s in S}
    return lambda policy: ck.validation_utility(policy, FakeExecutor(responses), quality_fn,
                                                catalogue, ENCODER, REFERENCE_SCORING)


def greedy_on(action):
    policy = EpsilonGreedyPolicy(S, epsilon=0.0, seed=1)
    for key in KEYS:
        for s in S:
            policy.update(Context(key, (1.0,)), s, 0.9 if s == action else 0.1)
    return policy


def test_validation_utility_uses_validation_split_only(validation_cases, monkeypatch):
    requested = []
    real = bm.load_split
    monkeypatch.setattr(bm, "load_split", lambda split, **kw: (requested.append(split),
                                                              real(split, **kw))[1])
    seen = []
    validate = _harness(validation_cases, seen_phase=seen)
    policy = greedy_on("structured")
    live = fingerprint(policy)
    attempts = bm.sealed_access_attempts()
    good = validate(policy)
    assert requested == ["validation"]
    assert seen and all(seen) and not bm.in_optimisation_phase()
    assert bm.sealed_access_attempts() == attempts
    assert fingerprint(policy) == live                  # deep copy, RNG untouched
    assert good.n_tasks == len(validation_cases) and good.synthetic
    assert not good.critical_violation
    assert good.utility > validate(greedy_on("direct")).utility


def test_validation_utility_cannot_reach_the_held_out_split(validation_cases):
    def misbehave():
        bm.load_split("test", allow_sealed_test=True)
    with pytest.raises(bm.SealedSplitError):
        _harness(validation_cases, misbehave=misbehave)(greedy_on("structured"))


def test_end_to_end_violating_admission_is_rolled_back(store, validation_cases):
    validate = _harness(validation_cases, tool_violation=True)
    policy = greedy_on("structured")
    pre_print = fingerprint(policy)

    def admit(p):                      # candidate learns to prefer the unsafe strategy
        for key in KEYS:
            p.update(Context(key, (1.0,)), "tool_assisted", 1.0)
            p.update(Context(key, (1.0,)), "structured", -1.0)
        return "skill/2"

    result = ck.admit_with_rollback(store, policy, skill_version="skill/1",
                                    admit=admit, validate=validate)
    assert result.post_validation.critical_violation
    assert not result.admitted and result.skill_version == "skill/1"
    assert fingerprint(policy) == pre_print
    assert result.synthetic


def test_policies_sharing_a_store_roll_back_to_their_own_checkpoints(store):
    eg = train(build("epsilon-greedy"), 30)
    eg_pre = store.save(eg, skill_version="skill/eg-1", label="eg pre")
    eg_print = fingerprint(eg)
    lin = train(build("linucb"), 30)
    lin_pre = store.save(lin, skill_version="skill/lin-1", label="lin pre")  # newer
    lin_print = fingerprint(lin)

    train(eg, 20, start=30)
    eg_cand = store.save(eg, skill_version="skill/eg-2", label="eg cand",
                         status=ck.STATUS_CANDIDATE, activate=False)
    target = store.rollback(eg, from_checkpoint_id=eg_cand.checkpoint_id, reason="drop")
    assert target.checkpoint_id == eg_pre.checkpoint_id
    assert fingerprint(eg) == eg_print
    assert store.active_id(eg) == eg_pre.checkpoint_id
    assert store.active_id(lin) == lin_pre.checkpoint_id      # other line untouched

    train(lin, 20, start=30)
    lin_cand = store.save(lin, skill_version="skill/lin-2", label="lin cand",
                          status=ck.STATUS_CANDIDATE, activate=False)
    assert store.rollback(lin, from_checkpoint_id=lin_cand.checkpoint_id,
                          reason="drop").checkpoint_id == lin_pre.checkpoint_id
    assert fingerprint(lin) == lin_print
    assert store.active_id(eg) == eg_pre.checkpoint_id

    with pytest.raises(ck.CheckpointMismatch):              # wrong policy for this line
        store.rollback(eg, from_checkpoint_id=lin_cand.checkpoint_id, reason="r")


def test_policy_without_own_good_checkpoint_fails_closed(store):
    store.save(train(build("linucb"), 10), skill_version="s", label="other policy")
    eg = train(build("epsilon-greedy"), 10)
    cand = store.save(eg, skill_version="s2", label="cand", status=ck.STATUS_CANDIDATE,
                      activate=False)
    live = fingerprint(eg)
    with pytest.raises(ck.NoGoodCheckpoint):
        store.rollback(eg, from_checkpoint_id=cand.checkpoint_id, reason="r")
    assert fingerprint(eg) == live
    assert store.load(cand.checkpoint_id).status == ck.STATUS_CANDIDATE

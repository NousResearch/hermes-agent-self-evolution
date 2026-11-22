"""Offline tests; no external services required."""
import json
import math
import sqlite3

import pytest
from evolution.adaptive.core import (
    RATIO_CAP, VIOLATION_REWARD, ContextualBandit, ExperienceStore, Reward,
    choose_strategy, reward_bounds, score_outcome,
)


def test_reward_and_hard_violation():
    assert score_outcome(Reward(1.0, 0, 0)) == 1.0
    assert score_outcome(Reward(1.0, 0, 0, violation=True)) == -1.0
    with pytest.raises(ValueError):
        score_outcome(Reward(1.2, 0, 0))


def test_policy_learns_context_specific_actions():
    policy = ContextualBandit(("a", "b"), epsilon=0, seed=1)
    for _ in range(15):
        policy.update("easy", "a", 0.9)
        policy.update("easy", "b", 0.2)
        policy.update("hard", "a", 0.1)
        policy.update("hard", "b", 0.8)
    assert policy.select("easy")[0] == "a"
    assert policy.select("hard")[0] == "b"


def test_propensity_and_reproducibility():
    a = ContextualBandit(("a", "b"), epsilon=0.2, seed=3)
    b = ContextualBandit(("a", "b"), epsilon=0.2, seed=3)
    assert [a.select("x") for _ in range(30)] == [b.select("x") for _ in range(30)]
    assert all(0 < p <= 1 for _, p in [a.select("x") for _ in range(20)])


def test_mapek_roundtrip(tmp_path):
    store = ExperienceStore(tmp_path / "experience.db")
    policy = ContextualBandit(("safe",), epsilon=0.1)
    record = choose_strategy(
        policy, "security",
        lambda ctx, action: {"passed": True},
        lambda result: Reward(float(result["passed"]), 0.2, 0.1),
        store, skill_version="skill-test/1",
    )
    assert record["strategy"] == "safe"
    assert len(store.rows()) == 1
    assert policy.counts[("security", "safe")] == 1
    store.close()


def test_unknown_strategy_rejected():
    policy = ContextualBandit(("allowed",))
    with pytest.raises(ValueError):
        policy.update("x", "not-allowed", 1.0)


# ── Reward safety ──────────────────────────────────────────────────────────

def test_default_reward_range_is_documented_bounds():
    assert reward_bounds() == pytest.approx((-0.30, 1.0))
    assert score_outcome(Reward(0.0, RATIO_CAP, RATIO_CAP)) == pytest.approx(-0.30)
    assert score_outcome(Reward(1.0, 0, 0)) == 1.0


@pytest.mark.parametrize("cost,time", [(20, 0), (0, 1e9), (1e300, 1e300), (2.0001, 50)])
def test_extreme_cost_and_latency_penalties_are_capped(cost, time):
    worst_valid = score_outcome(Reward(0.0, cost, time))
    assert worst_valid >= reward_bounds()[0]
    assert worst_valid > VIOLATION_REWARD
    assert score_outcome(Reward(1.0, cost, time)) > VIOLATION_REWARD


def test_violation_strictly_worse_than_every_valid_outcome():
    worst_valid = score_outcome(Reward(0.0, 1e300, 1e300))
    for quality in (0.0, 1.0):
        assert score_outcome(Reward(quality, 0, 0, violation=True)) < worst_valid


def test_weights_that_could_tie_a_violation_are_rejected():
    with pytest.raises(ValueError):
        score_outcome(Reward(1.0, 0, 0), cost_weight=0.5, time_weight=0.0)


@pytest.mark.parametrize("bad", [
    Reward(math.nan, 0, 0), Reward(1.0, math.inf, 0), Reward(1.0, 0, -math.inf),
    Reward(1.0, math.nan, 0), Reward(True, 0, 0), Reward(1.0, -0.1, 0),
    Reward(1.0, 0, 0, violation=1),
])
def test_invalid_outcome_fields_rejected(bad):
    with pytest.raises(ValueError):
        score_outcome(bad)


# ── Exploration ────────────────────────────────────────────────────────────

def test_epsilon_zero_tries_every_action_once_per_context():
    policy = ContextualBandit(("a", "b", "c"), epsilon=0, seed=0)
    for context in ("x", "y"):
        seen = []
        for _ in range(3):
            action, propensity = policy.select(context)
            assert propensity == 1.0
            seen.append(action)
            policy.update(context, action, 0.3 if action == "a" else 0.1)
        assert seen == ["a", "b", "c"]


def test_epsilon_zero_untried_action_not_starved_by_positive_estimate():
    policy = ContextualBandit(("a", "b"), epsilon=0, seed=0)
    policy.update("c", "a", 0.3)          # a has a positive estimate
    assert policy.select("c") == ("b", 1.0)
    policy.update("c", "b", 0.9)
    assert policy.select("c")[0] == "b"   # greedy after the sweep


def test_epsilon_zero_greedy_propensity_after_sweep():
    policy = ContextualBandit(("a", "b"), epsilon=0, seed=0)
    policy.update("c", "a", 0.9)
    policy.update("c", "b", 0.1)
    assert policy.select("c") == ("a", 1.0)


# ── Numerical validation ───────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [
    math.nan, math.inf, -math.inf, True, False, "0.5", None, [0.5], 1 + 0j,
])
def test_invalid_rewards_rejected_before_state_changes(bad):
    policy = ContextualBandit(("a",), epsilon=0)
    policy.update("c", "a", 0.5)
    with pytest.raises(ValueError):
        policy.update("c", "a", bad)
    assert policy.counts == {("c", "a"): 1}
    assert policy.means == {("c", "a"): 0.5}


@pytest.mark.parametrize("bad_reward", [math.nan, math.inf, True])
def test_store_rejects_invalid_rewards(tmp_path, bad_reward):
    with ExperienceStore(tmp_path / "x.db") as store:
        with pytest.raises(ValueError):
            store.record("c", "a", 0.5, bad_reward, {},
                         policy_version="p", skill_version="s")
        assert store.rows() == []


# ── Provenance and migration ───────────────────────────────────────────────

def test_rows_include_feedback_and_provenance(tmp_path):
    with ExperienceStore(tmp_path / "x.db") as store:
        store.record("c", "a", 0.5, 0.7, {"quality": 1.0},
                     policy_version="eg/1", skill_version="skill/3",
                     recorded_at="2026-11-18T00:00:00+00:00")
        store.record("c", "a", 0.5, 0.7, {}, policy_version="eg/1",
                     skill_version="skill/3")
        first, second = store.rows()
    assert json.loads(first["feedback_json"]) == {"quality": 1.0}
    assert first["policy_version"] == "eg/1"
    assert first["skill_version"] == "skill/3"
    assert first["recorded_at"] == "2026-11-18T00:00:00+00:00"
    assert second["recorded_at"]  # auto-stamped UTC timestamp
    assert second["recorded_at"].endswith("+00:00")


def test_provenance_is_required(tmp_path):
    with ExperienceStore(tmp_path / "x.db") as store:
        with pytest.raises(ValueError):
            store.record("c", "a", 0.5, 0.1, {}, policy_version="",
                         skill_version="s")


def test_choose_strategy_records_policy_and_skill_versions(tmp_path):
    policy = ContextualBandit(("only",), version="eg-test/7")
    with ExperienceStore(tmp_path / "x.db") as store:
        choose_strategy(policy, "ctx", lambda c, s: None,
                        lambda r: Reward(1.0, 0, 0), store,
                        skill_version="skill/2")
        (row,) = store.rows()
    assert (row["policy_version"], row["skill_version"]) == ("eg-test/7", "skill/2")


def test_legacy_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.execute("""CREATE TABLE experiences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        context TEXT NOT NULL, strategy TEXT NOT NULL,
        propensity REAL NOT NULL, reward REAL NOT NULL,
        feedback_json TEXT NOT NULL)""")
    legacy.execute("INSERT INTO experiences (context,strategy,propensity,reward,"
                   "feedback_json) VALUES ('old','a',0.5,0.2,'{}')")
    legacy.commit()
    legacy.close()

    with ExperienceStore(path) as store:
        store.record("new", "a", 1.0, 0.4, {}, policy_version="p",
                     skill_version="s")
        old, new = store.rows()
    assert old["context"] == "old" and old["feedback_json"] == "{}"
    assert old["policy_version"] is None and old["recorded_at"] is None
    assert new["policy_version"] == "p"

    with ExperienceStore(path) as store:  # reopening is idempotent
        assert len(store.rows()) == 2
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == 2


# ── Resource handling ──────────────────────────────────────────────────────

def test_store_closed_by_context_manager_on_error(tmp_path):
    with pytest.raises(RuntimeError):
        with ExperienceStore(tmp_path / "x.db") as store:
            raise RuntimeError("episode failed")
    with pytest.raises(sqlite3.ProgrammingError):
        store.conn.execute("SELECT 1")


def test_cli_closes_store_when_episode_raises(monkeypatch):
    from evolution.adaptive import cli
    closed = []
    real_close = ExperienceStore.close
    monkeypatch.setattr(ExperienceStore, "close",
                        lambda self: (closed.append(True), real_close(self)))

    def boom(*args, **kwargs):
        raise RuntimeError("executor crashed")

    monkeypatch.setattr(cli, "choose_strategy", boom)
    monkeypatch.setattr("sys.argv", ["cli", "--episodes", "3"])
    with pytest.raises(RuntimeError):
        cli.main()
    assert closed == [True]

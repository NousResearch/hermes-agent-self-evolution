from evolution.code.audit_report import tally, wilson


def test_wilson_bounds():
    lo, hi = wilson(0, 5)
    assert lo == 0.0 and 0.0 < hi < 0.6
    assert wilson(0, 0) == (0.0, 0.0)


def test_tally_credits_oracle_only_for_oracle_match():
    records = [
        # org A: a gaming hardcode SLIPPED the oracle (deployed); honest deployed too
        {"organism": "tools/a.py@aaa", "seed": 0, "arm": "direct_inject", "bucket": "scored",
         "deploy": True, "guard": "deployed"},
        {"organism": "tools/a.py@aaa", "seed": 0, "arm": "honest_control", "bucket": "scored",
         "deploy": True, "guard": "deployed"},
        # org B: gaming reached the oracle and was CAUGHT there
        {"organism": "tools/b.py@bbb", "seed": 0, "arm": "direct_inject", "bucket": "scored",
         "deploy": False, "guard": "oracle_match"},
        # org C: gaming rejected at FREEZE — never reached the oracle (must NOT count as caught)
        {"organism": "tools/c.py@ccc", "seed": 0, "arm": "direct_inject", "bucket": "scored",
         "deploy": False, "guard": "freeze"},
        {"organism": "tools/c.py@ccc", "seed": 0, "arm": "repair_engine_gaming",
         "bucket": "never_reached_oracle_match", "deploy": None, "guard": None},
        # a skip row with no arm — ignored
        {"organism": "tools/d.py@ddd", "skip": "no_bug_tests"},
    ]
    out = tally(records)
    A, B, C = out["per_organism"]["tools/a.py@aaa"], out["per_organism"]["tools/b.py@bbb"], out["per_organism"]["tools/c.py@ccc"]
    assert A["slipped_preadjudication"] == 1 and A["reached_oracle_match"] == 1 and A["honest_deploys"] == 1
    assert B["caught_at_oracle"] == 1 and B["reached_oracle_match"] == 1 and B["slipped_preadjudication"] == 0
    assert C["reached_oracle_match"] == 0 and C["caught_at_oracle"] == 0          # freeze != reached
    assert C["guard_breakdown"].get("freeze") == 1
    assert C["secondary_never_reached"] == 1
    assert out["organisms_where_oracle_reached"] == ["tools/a.py@aaa", "tools/b.py@bbb"]
    assert out["organisms_where_hole_unreached"] == ["tools/c.py@ccc"]
    # pooled over reached arms: 2 reached (A,B), 1 slipped (A)
    assert out["pooled_reached_arms"]["n_reached"] == 2
    assert out["pooled_reached_arms"]["n_slipped_preadjudication"] == 1

"""Tests for cluster-honest campaign reporting (C4 kill gate)."""

from evolution.code.campaign_report import OrganismResult, build_report, wilson_lower

# The 8 Tier-A real organisms as per-seed fix outcomes (fixed = round not None),
# from spikes/.../_tierA_records.json. Reproducing the F1 numbers from these is
# the C4 regression check.
_TIER_A = [
    [True, True, True, True, True],      # band
    [True, False, True, True, False],    # other (3/5)
    [True, True, True, True, True],      # band
    [True, True, True, True, True],      # band
    [False, False, False, False, False],  # cliff
    [False, True, True, False, True],    # other (3/5)
    [True, False, True, False, True],    # other (3/5)
    [False, False, False, False, False],  # cliff
]


def _organisms():
    return [OrganismResult(f"tools/t{i}.py", f"sha{i}", s) for i, s in enumerate(_TIER_A)]


class TestDeployReachable:
    def test_majority_rule(self):
        assert OrganismResult("t", "s", [True, True, True]).deploy_reachable
        assert OrganismResult("t", "s", [True, True, False]).deploy_reachable  # 2/3
        assert not OrganismResult("t", "s", [True, False, False]).deploy_reachable  # 1/3
        assert not OrganismResult("t", "s", [False, False, False]).deploy_reachable


class TestBuildReport:
    def test_reproduces_tier_a_deploy_reachable(self):
        rep = build_report(_organisms())
        dr = rep["deploy_reachable"]
        assert dr["k"] == 6 and dr["n"] == 8          # 6/8 organisms deploy-reachable
        assert abs(dr["fraction"] - 0.75) < 1e-9
        lo, hi = dr["wilson"]
        assert 0.40 <= lo <= 0.42 and 0.92 <= hi <= 0.94  # ~[0.41, 0.93]
        assert rep["verdict"] == "GREEN"               # Wilson-lower 0.41 > 0.10

    def test_icc_matches_f1(self):
        rep = build_report(_organisms())
        assert 0.45 <= rep["icc"] <= 0.65              # F1 measured ~0.57
        assert rep["design_effect"] > 2.5

    def test_pooled_rate_is_contrast_only(self):
        rep = build_report(_organisms())
        pooled = rep["pooled_per_seed_rate_FOR_CONTRAST"]
        assert pooled["k"] == 24 and pooled["n"] == 40  # the demoted 24/40 = 0.60
        assert abs(pooled["rate"] - 0.60) < 1e-9

    def test_below_kill_line_when_sparse(self):
        # 1/8 deploy-reachable → Wilson-lower well under 0.10.
        orgs = [OrganismResult(f"t{i}", f"s{i}", [True, True, True] if i == 0
                               else [False, False, False]) for i in range(8)]
        rep = build_report(orgs)
        assert rep["verdict"] == "BELOW_KILL_LINE"
        assert wilson_lower(1, 8) < 0.10

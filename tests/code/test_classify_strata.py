from evolution.code.classify_strata import load_strata_worksheet, validate_strata


def test_worksheet_carries_measured_armB_dr():
    rows = load_strata_worksheet()
    assert len(rows) == 12
    assert all(isinstance(r["armB_dr"], bool) for r in rows)
    assert all(r["stratum"] == "" for r in rows)  # human-filled later


def test_validate_strata_falsifiability_guard():
    rows = [
        {"tool": "a", "fix": "1", "stratum": "pure-contract", "contract_source": "docstring L3", "notes": ""},
        {"tool": "b", "fix": "2", "stratum": "pure-contract", "contract_source": "", "notes": ""},          # error: no source
        {"tool": "c", "fix": "3", "stratum": "pure-contract", "contract_source": "x", "notes": "fuzz-only"}, # reclassified
        {"tool": "d", "fix": "4", "stratum": "state", "contract_source": "", "notes": ""},                   # excluded
        {"tool": "e", "fix": "5", "stratum": "pure-input", "contract_source": "", "notes": ""},
    ]
    out = validate_strata(rows)
    assert any("requires a cited contract_source" in e for e in out["errors"])    # row b
    recls = [r for r in out["rows"] if r["tool"] == "c"][0]
    assert recls["stratum"] == "pure-input"                                       # row c reclassified
    assert out["ratio"]["state_excluded"] == 1                                    # row d excluded
    # row b errors (no cited source) and is excluded from the ratio denominator
    # non_state counts: a (pure-contract), c->pure-input, e (pure-input) = 3
    assert out["ratio"]["non_state_total"] == 3
    # only row a has a valid pure-contract claim
    assert out["ratio"]["pure_contract"] == 1

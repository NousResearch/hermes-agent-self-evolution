import json
from evolution.code.swebench.report import summarize_difficulty, drop_breakdown, HERMES_PROFILE, build

_CHARS = [
    {"instance_id": "a", "reason": "kept", "gold_loc": 4, "gold_hunks": 1, "emulated": True},
    {"instance_id": "b", "reason": "kept", "gold_loc": 50, "gold_hunks": 3, "emulated": False},
    {"instance_id": "c", "reason": "gold_violates_freeze", "gold_loc": 80, "gold_hunks": 5},
    {"instance_id": "d", "reason": "bug_not_reproduced", "gold_loc": 0, "gold_hunks": 0},
]

def test_summarize_difficulty_of_kept():
    s = summarize_difficulty([r for r in _CHARS if r["reason"] == "kept"])
    assert s["n"] == 2 and s["median_loc"] == 27.0 and s["frac_large_gt20"] == 0.5
def test_drop_breakdown_counts_by_reason():
    b = drop_breakdown(_CHARS)
    assert b["gold_violates_freeze"] == 1 and b["bug_not_reproduced"] == 1 and b["kept"] == 2
def test_hermes_profile_present():
    assert HERMES_PROFILE["median_loc"] == 45 and HERMES_PROFILE["frac_large_gt20"] > 0.7
def test_build_joins_and_profiles(tmp_path):
    (tmp_path / "characterization.json").write_text(json.dumps(_CHARS))
    (tmp_path / "campaign_report.json").write_text(json.dumps(
        {"deploy_reachable": {"k": 1, "n": 2, "wilson": [0.1, 0.9]}, "verdict": "GREEN"}))
    rep = build(tmp_path)
    assert rep["deploy_reachable"]["n"] == 2
    assert rep["kept_difficulty"]["n"] == 2
    assert rep["freeze_dropped_difficulty"]["n"] == 1 and rep["freeze_dropped_difficulty"]["median_loc"] == 80.0
    assert "interpretation_guard" in rep and "loc_definition" in rep
    assert rep["emulated_kept"] == 1
    assert (tmp_path / "external_validity_report.json").exists()

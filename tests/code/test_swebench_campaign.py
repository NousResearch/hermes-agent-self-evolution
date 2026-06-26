from pathlib import Path
from evolution.code.swebench.loader import SWEInstance
from evolution.code.swebench.campaign import instances_to_candidates, _arm_cost_ledger
from evolution.code.campaign import run_campaign
from evolution.code.campaign_report import OrganismResult

def _inst(i):
    return SWEInstance(instance_id=f"i{i}", repo="pallets/flask", base_commit="b", version="2.0",
        gold_patch="diff --git a/app.py b/app.py\n+++ b/app.py\n", test_patch="", gold_file="app.py",
        fail_to_pass=("t::a",), pass_to_pass=(), raw={})

def test_candidate_has_fix_sha_and_repo_cluster():
    c = instances_to_candidates([_inst(0)])[0]
    assert c.fix_sha == "i0" and c.tool_path == "pallets/flask"

def test_campaign_runs_with_injected_runner(tmp_path):
    cands = instances_to_candidates([_inst(i) for i in range(3)])
    def fake_runner(c):
        return OrganismResult(tool=c.tool_path, fix_sha=c.fix_sha, seeds=[c.fix_sha != "i2"])
    report = run_campaign(tmp_path, output_dir=tmp_path, max_organisms=3, stages=(3,),
                          candidates=cands, organism_runner=fake_runner)
    assert report["deploy_reachable"]["n"] == 3

def test_arm_cost_ledger_sets_ceiling(monkeypatch):
    import evolution.core.lm_timing_callback as lt
    seen = {}
    monkeypatch.setattr(lt.COST_LEDGER, "set_ceiling", lambda x: seen.setdefault("ceiling", x))
    _arm_cost_ledger(5.0)
    assert seen["ceiling"] == 5.0

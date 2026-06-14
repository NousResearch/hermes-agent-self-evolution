"""Tests for the campaign orchestration (no LM, no worktrees).

Uses the ``organism_runner`` / ``candidates`` injection seams to exercise the
ledger, resume, stratification, and futility-stop logic deterministically.
"""

import json
from pathlib import Path

from evolution.code.campaign import Skip, run_campaign
from evolution.code.campaign_report import OrganismResult
from evolution.code.harvest import Candidate, stratify


def _cands(n, tool="tools/t.py"):
    return [Candidate(tool, "tests/tools/test_t.py", f"sha{i:03d}", f"par{i:03d}")
            for i in range(n)]


class TestStratify:
    def test_interleaves_and_caps_per_tool(self):
        cands = ([Candidate(f"tools/a.py", "t", f"a{i}", f"pa{i}") for i in range(5)]
                 + [Candidate(f"tools/b.py", "t", f"b{i}", f"pb{i}") for i in range(5)])
        out = stratify(cands, max_per_tool=2)
        assert len(out) == 4  # 2 per tool
        tools = [c.tool_path for c in out]
        assert tools.count("tools/a.py") == 2 and tools.count("tools/b.py") == 2
        # interleaved, not grouped
        assert tools[0] != tools[1]


class TestCampaignOrchestration:
    def test_ledger_written_and_report_built(self, tmp_path):
        cands = _cands(4)
        # 3 deploy-reachable, 1 not-valid (None).
        canned = {
            "sha000": OrganismResult("tools/t.py", "sha000", [True, True, True]),
            "sha001": OrganismResult("tools/t.py", "sha001", [True, True, False]),
            "sha002": None,
            "sha003": OrganismResult("tools/t.py", "sha003", [True, True, True]),
        }
        rep = run_campaign(
            Path("/unused"), output_dir=tmp_path, max_organisms=10, stages=(),
            candidates=cands, organism_runner=lambda c: canned[c.fix_sha])
        assert rep["n_organisms"] == 3
        assert rep["deploy_reachable"]["k"] == 3
        rows = [json.loads(l) for l in (tmp_path / "campaign_ledger.jsonl").read_text().splitlines()]
        assert any(r.get("status") == "not_valid" and r["fix_sha"] == "sha002" for r in rows)
        assert sum(1 for r in rows if r.get("status") == "organism") == 3

    def test_skip_reasons_recorded(self, tmp_path):
        cands = _cands(3)
        canned = {
            "sha000": OrganismResult("tools/t.py", "sha000", [True, True, True]),
            "sha001": Skip("too_large"),
            "sha002": Skip("not_valid"),
        }
        rep = run_campaign(Path("/u"), output_dir=tmp_path, max_organisms=10, stages=(),
                           candidates=cands, organism_runner=lambda c: canned[c.fix_sha])
        assert rep["n_organisms"] == 1
        rows = [json.loads(l) for l in (tmp_path / "campaign_ledger.jsonl").read_text().splitlines()]
        statuses = {r["fix_sha"]: r["status"] for r in rows}
        assert statuses["sha001"] == "too_large"
        assert statuses["sha002"] == "not_valid"

    def test_resume_skips_done(self, tmp_path):
        cands = _cands(4)
        canned = {c.fix_sha: OrganismResult("tools/t.py", c.fix_sha, [True, True, True])
                  for c in cands}
        seen: list[str] = []

        def runner(c):
            seen.append(c.fix_sha)
            return canned[c.fix_sha]

        run_campaign(Path("/u"), output_dir=tmp_path, max_organisms=10, stages=(),
                     candidates=cands[:2], organism_runner=runner)
        first = list(seen)
        run_campaign(Path("/u"), output_dir=tmp_path, max_organisms=10, stages=(),
                     candidates=cands, organism_runner=runner)
        # second run only processes the 2 new candidates, not the 2 already done.
        assert first == ["sha000", "sha001"]
        assert seen[2:] == ["sha002", "sha003"]

    def test_futility_stop_at_stage(self, tmp_path):
        cands = _cands(20)
        # every organism fails → deploy-reachable 0 → Wilson-lower 0 < 0.10.
        rep = run_campaign(
            Path("/u"), output_dir=tmp_path, max_organisms=20, stages=(4,),
            candidates=cands,
            organism_runner=lambda c: OrganismResult("tools/t.py", c.fix_sha, [False, False, False]))
        assert rep["n_organisms"] == 4  # stopped at the stage-4 futility boundary
        assert rep["verdict"] == "BELOW_KILL_LINE"

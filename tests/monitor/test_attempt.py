"""Tests for the attempt pass (annotation via an injected runner; no LM)."""

from pathlib import Path

from evolution.code.campaign import Skip
from evolution.code.campaign_report import OrganismResult
from evolution.monitor.attempt import attempt_candidates
from evolution.monitor.queue import build_queue
from evolution.monitor.sentinel import RepairCandidate

_C1 = RepairCandidate("tools/a.py", "tests/tools/test_a.py", "sha1", "par1",
                      "dependency_regression", "2026-06-10T00:00:00+00:00")
_C2 = RepairCandidate("tools/b.py", "tests/tools/test_b.py", "sha2", "par2",
                      "bug_fix", "2026-06-12T00:00:00+00:00")


def _payload():
    return build_queue([_C1, _C2], repo="/r", since_days=90)


class TestAttempt:
    def test_annotates_rows_with_verdicts(self):
        payload = _payload()
        canned = {
            "sha1": OrganismResult("tools/a.py", "sha1", [True, True, True]),
            "sha2": Skip("not_valid"),
        }
        attempt_candidates(Path("/r"), [_C1, _C2], payload,
                           organism_runner=lambda repo, c: canned[c.fix_sha],
                           cost_summary=lambda: {"total_usd": 0.0})
        rows = {r["fix_sha"]: r for r in payload["candidates"]}
        assert rows["sha1"]["attempt"] == {
            "status": "attempted", "correct_seeds": 3, "seeds": 3, "deploy_reachable": True}
        assert rows["sha2"]["attempt"]["status"] == "not_valid"
        assert payload["cost_summary"] == {"total_usd": 0.0}

    def test_cost_ceiling_stops_and_marks(self):
        from evolution.core.lm_timing_callback import CostCeilingExceeded
        payload = _payload()

        def runner(repo, c):
            raise CostCeilingExceeded(99.0, 1.0)

        attempt_candidates(Path("/r"), [_C1, _C2], payload,
                           organism_runner=runner, cost_summary=lambda: {})
        rows = {r["fix_sha"]: r for r in payload["candidates"]}
        assert rows["sha1"]["attempt"] == {"status": "cost_ceiling"}
        assert "attempt" not in rows["sha2"]  # broke before reaching the second

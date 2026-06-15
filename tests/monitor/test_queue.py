"""Tests for triage-queue assembly + rendering (pure, no spend)."""

from evolution.monitor.queue import build_queue, render_report, write_queue
from evolution.monitor.sentinel import RepairCandidate

_DEP = RepairCandidate("tools/a.py", "tests/tools/test_a.py", "dep0sha1", "dep0par1",
                       "dependency_regression", "2026-06-10T00:00:00+00:00")
_BUG = RepairCandidate("tools/b.py", "tests/tools/test_b.py", "bug0sha1", "bug0par1",
                       "bug_fix", "2026-06-12T00:00:00+00:00")


class TestBuildQueue:
    def test_payload_shape_and_counts(self):
        q = build_queue([_DEP, _BUG], repo="/r", since_days=90)
        assert q["n_candidates"] == 2
        assert q["by_kind"] == {"dependency_regression": 1, "bug_fix": 1}
        assert [c["rank"] for c in q["candidates"]] == [1, 2]
        assert q["candidates"][0]["kind"] == "dependency_regression"
        assert q["candidates"][0]["fix_sha"] == "dep0sha1"

    def test_write_queue_roundtrips(self, tmp_path):
        import json
        q = build_queue([_DEP], repo="/r", since_days=30)
        path = write_queue(tmp_path, q)
        assert json.loads(path.read_text())["candidates"][0]["tool"] == "tools/a.py"


class TestRenderReport:
    def test_report_is_propose_only_with_command(self):
        q = build_queue([_DEP, _BUG], repo="/r", since_days=90)
        report = render_report(q, top=10)
        assert "Propose-only" in report
        assert "--attempt-top" in report          # the single action it offers
        assert "tools/a.py" in report and "tools/b.py" in report
        assert "dependency_regression" in report

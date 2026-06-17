"""Tests for evolution.review.digest."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from evolution.core.proposals import (
    PROPOSAL_STATUS_APPROVED,
    PROPOSAL_STATUS_PENDING,
    PROPOSAL_STATUS_REJECTED,
    ProposalWriter,
    build_proposal_record,
)
from evolution.review.digest import (
    build_digest,
    filter_recent,
    find_recent_logs,
    log_tail,
    render_json,
    render_markdown,
)
from evolution.review.proposal_reviewer import discover_proposals


# ──────────────────────────── helpers ────────────────────────────
@dataclass
class _FakeDecision:
    auto_merge: bool
    reason: str
    regression: bool = False


@dataclass
class _FakeConstraint:
    constraint_name: str
    passed: bool
    message: str


def _write_proposal(
    proposals_dir: Path,
    skill: str,
    timestamp: str,
    status: str = PROPOSAL_STATUS_PENDING,
    improvement: float = 0.05,
    baseline: float = 0.70,
    evolved: float = 0.75,
    auto_merge: bool = False,
    mode: str = "propose",
    rejection_reason: str | None = None,
    mtime_offset_hours: float = 0.0,
) -> Path:
    """Create a proposal dir on disk with decision.json and STATUS."""
    # improvement is computed as evolved - baseline inside build_proposal_record
    evolved_score = baseline + improvement if improvement is not None else evolved
    record = build_proposal_record(
        skill_name=skill,
        baseline_text="baseline instructions",
        evolved_text="evolved instructions",
        baseline_score=baseline,
        evolved_score=evolved_score,
        decision=_FakeDecision(
            auto_merge=auto_merge,
            reason=rejection_reason or "ok",
        ),
        constraint_results=[
            _FakeConstraint("size", True, "ok"),
            _FakeConstraint("has_frontmatter", True, "ok"),
        ],
        mode=mode,
        metadata={},
        timestamp=timestamp,
    )
    writer = ProposalWriter(proposals_dir)
    proposal_dir = writer.write(record)

    # Overwrite STATUS if different from PENDING
    if status != PROPOSAL_STATUS_PENDING:
        status_path = proposal_dir / "STATUS"
        if rejection_reason:
            status_path.write_text(f"{status}\n{rejection_reason}\n")
        else:
            status_path.write_text(f"{status}\n")

    # Adjust mtime on decision.json to simulate old/new proposals
    if mtime_offset_hours:
        decision_path = proposal_dir / "decision.json"
        target = time.time() - (mtime_offset_hours * 3600)
        os.utime(decision_path, (target, target))

    return proposal_dir


# ──────────────────────────── filtering ────────────────────────────
class TestFilterRecent:
    def test_keeps_recent(self, tmp_path):
        _write_proposal(tmp_path, "skill-a", "20260418_120000")
        entries = discover_proposals(tmp_path)
        since = (datetime.now() - timedelta(hours=24)).timestamp()
        kept = filter_recent(entries, since)
        assert len(kept) == 1

    def test_drops_old(self, tmp_path):
        _write_proposal(tmp_path, "skill-a", "20260418_120000", mtime_offset_hours=48)
        entries = discover_proposals(tmp_path)
        since = (datetime.now() - timedelta(hours=24)).timestamp()
        kept = filter_recent(entries, since)
        assert len(kept) == 0

    def test_mixed(self, tmp_path):
        _write_proposal(tmp_path, "skill-a", "20260418_120000")  # recent
        _write_proposal(
            tmp_path, "skill-b", "20260417_120000", mtime_offset_hours=48
        )  # old
        entries = discover_proposals(tmp_path)
        since = (datetime.now() - timedelta(hours=24)).timestamp()
        kept = filter_recent(entries, since)
        assert len(kept) == 1
        assert kept[0].skill_name == "skill-a"


# ──────────────────────────── log scan ────────────────────────────
class TestLogScan:
    def test_empty_dir_missing(self, tmp_path):
        assert find_recent_logs(tmp_path / "nope", 0.0) == []

    def test_finds_recent_logs(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "evolve-x-20260418.log").write_text("run output\n")
        (log_dir / "evolve-y-20260418.log").write_text("run output\n")
        (log_dir / "not-a-run.txt").write_text("ignored\n")
        since = (datetime.now() - timedelta(hours=24)).timestamp()
        logs = find_recent_logs(log_dir, since)
        assert len(logs) == 2
        assert all(p.name.startswith("evolve-") for p in logs)

    def test_ignores_old_logs(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        old = log_dir / "evolve-old.log"
        old.write_text("old\n")
        target = time.time() - 48 * 3600
        os.utime(old, (target, target))
        since = (datetime.now() - timedelta(hours=24)).timestamp()
        assert find_recent_logs(log_dir, since) == []

    def test_log_tail(self, tmp_path):
        f = tmp_path / "t.log"
        f.write_text("\n".join(f"line {i}" for i in range(20)) + "\n")
        tail = log_tail(f, lines=3)
        assert tail.splitlines() == ["line 17", "line 18", "line 19"]

    def test_log_tail_missing(self, tmp_path):
        assert log_tail(tmp_path / "nope.log") == ""


# ──────────────────────────── rendering ────────────────────────────
class TestRenderMarkdown:
    def test_empty(self, tmp_path):
        out = render_markdown([], window_hours=24, generated_at=datetime(2026, 4, 18, 7, 0))
        assert "Hermes Self-Evolution Digest" in out
        assert "No evolution activity" in out
        assert "**0** proposal(s)" in out

    def test_summary_counts(self, tmp_path):
        _write_proposal(tmp_path, "a", "20260418_120000", status=PROPOSAL_STATUS_PENDING)
        _write_proposal(
            tmp_path, "b", "20260418_130000", status=PROPOSAL_STATUS_APPROVED, auto_merge=True, mode="auto"
        )
        _write_proposal(
            tmp_path,
            "c",
            "20260418_140000",
            status=PROPOSAL_STATUS_REJECTED,
            rejection_reason="gate rejected",
        )
        entries = discover_proposals(tmp_path)
        out = render_markdown(entries, window_hours=24, generated_at=datetime.now())
        assert "pending: **1**" in out
        assert "approved: **1**" in out
        assert "auto-merged: **1**" in out
        assert "rejected: **1**" in out

    def test_table_rows(self, tmp_path):
        _write_proposal(
            tmp_path,
            "github-code-review",
            "20260418_120000",
            improvement=0.05,
            baseline=0.70,
            evolved=0.75,
        )
        entries = discover_proposals(tmp_path)
        out = render_markdown(entries, window_hours=24, generated_at=datetime.now())
        assert "| `github-code-review` |" in out
        assert "+0.050" in out
        assert "0.700 → 0.750" in out

    def test_rejection_reason_surfaced(self, tmp_path):
        _write_proposal(
            tmp_path,
            "skill-r",
            "20260418_120000",
            status=PROPOSAL_STATUS_REJECTED,
            rejection_reason="hallucinated flag",
        )
        entries = discover_proposals(tmp_path)
        out = render_markdown(entries, window_hours=24, generated_at=datetime.now())
        assert "hallucinated flag" in out

    def test_review_commands_present_for_pending(self, tmp_path):
        _write_proposal(tmp_path, "skill-p", "20260418_120000")
        entries = discover_proposals(tmp_path)
        out = render_markdown(entries, window_hours=24, generated_at=datetime.now())
        assert "proposal_reviewer approve skill-p" in out
        assert "proposal_reviewer reject skill-p" in out

    def test_review_commands_absent_for_approved(self, tmp_path):
        _write_proposal(
            tmp_path, "skill-a", "20260418_120000", status=PROPOSAL_STATUS_APPROVED
        )
        entries = discover_proposals(tmp_path)
        out = render_markdown(entries, window_hours=24, generated_at=datetime.now())
        # show/diff still present; approve/reject should not be
        assert "proposal_reviewer show skill-a" in out
        assert "proposal_reviewer approve skill-a" not in out

    def test_log_tails_included(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "evolve-x.log").write_text("line1\nline2\n")
        entries = []
        out = render_markdown(
            entries,
            window_hours=24,
            generated_at=datetime.now(),
            logs=[log_dir / "evolve-x.log"],
        )
        assert "## Run logs" in out
        assert "line1" in out


class TestRenderJSON:
    def test_shape(self, tmp_path):
        _write_proposal(tmp_path, "a", "20260418_120000")
        entries = discover_proposals(tmp_path)
        out = render_json(entries, window_hours=24, generated_at=datetime.now())
        data = json.loads(out)
        assert data["summary"]["total"] == 1
        assert data["proposals"][0]["skill_name"] == "a"
        assert "window_hours" in data
        assert "generated_at" in data

    def test_empty(self):
        out = render_json([], window_hours=12, generated_at=datetime.now())
        data = json.loads(out)
        assert data["summary"]["total"] == 0
        assert data["proposals"] == []


# ──────────────────────────── build_digest ────────────────────────────
class TestBuildDigest:
    def test_end_to_end_markdown(self, tmp_path):
        prop_dir = tmp_path / "proposals"
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        _write_proposal(prop_dir, "skill-x", "20260418_120000")
        (log_dir / "evolve-skill-x.log").write_text("done\n")

        out = build_digest(
            proposals_dir=prop_dir,
            log_dir=log_dir,
            window_hours=24,
            fmt="markdown",
        )
        assert "## Proposals" in out
        assert "skill-x" in out

    def test_end_to_end_json(self, tmp_path):
        prop_dir = tmp_path / "proposals"
        log_dir = tmp_path / "logs"
        _write_proposal(prop_dir, "skill-y", "20260418_120000")

        out = build_digest(
            proposals_dir=prop_dir,
            log_dir=log_dir,
            window_hours=24,
            fmt="json",
        )
        data = json.loads(out)
        assert data["summary"]["total"] == 1
        assert data["proposals"][0]["skill_name"] == "skill-y"

    def test_window_excludes_old(self, tmp_path):
        prop_dir = tmp_path / "proposals"
        log_dir = tmp_path / "logs"
        _write_proposal(
            prop_dir, "skill-old", "20260418_120000", mtime_offset_hours=48
        )
        _write_proposal(prop_dir, "skill-new", "20260418_130000")

        out = build_digest(
            proposals_dir=prop_dir, log_dir=log_dir, window_hours=24, fmt="json"
        )
        data = json.loads(out)
        assert data["summary"]["total"] == 1
        assert data["proposals"][0]["skill_name"] == "skill-new"

    def test_missing_dirs_graceful(self, tmp_path):
        out = build_digest(
            proposals_dir=tmp_path / "nope",
            log_dir=tmp_path / "also_nope",
            window_hours=24,
            fmt="markdown",
        )
        assert "No evolution activity" in out


# ──────────────────────────── CLI ────────────────────────────
class TestCLI:
    def test_cli_stdout(self, tmp_path, capsys):
        from evolution.review.digest import main

        _write_proposal(tmp_path / "proposals", "skill-z", "20260418_120000")

        rc = main(
            [
                "--proposals-dir",
                str(tmp_path / "proposals"),
                "--log-dir",
                str(tmp_path / "logs"),
                "--hours",
                "24",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "skill-z" in captured.out

    def test_cli_output_file(self, tmp_path):
        from evolution.review.digest import main

        _write_proposal(tmp_path / "proposals", "skill-z", "20260418_120000")
        out_file = tmp_path / "digest.md"

        rc = main(
            [
                "--proposals-dir",
                str(tmp_path / "proposals"),
                "--log-dir",
                str(tmp_path / "logs"),
                "--output",
                str(out_file),
            ]
        )
        assert rc == 0
        assert out_file.exists()
        assert "skill-z" in out_file.read_text()

    def test_cli_json_format(self, tmp_path, capsys):
        from evolution.review.digest import main

        _write_proposal(tmp_path / "proposals", "skill-z", "20260418_120000")

        rc = main(
            [
                "--proposals-dir",
                str(tmp_path / "proposals"),
                "--log-dir",
                str(tmp_path / "logs"),
                "--format",
                "json",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["summary"]["total"] == 1

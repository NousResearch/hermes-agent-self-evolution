"""Tests for the ProposalReviewer CLI."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from evolution.core.proposals import (
    PROPOSAL_STATUS_APPROVED,
    PROPOSAL_STATUS_PENDING,
    PROPOSAL_STATUS_REJECTED,
    ProposalWriter,
    build_proposal_record,
)
from evolution.review import proposal_reviewer as pr


# ──────────────────────────── fixtures ────────────────────────────
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
    skill_name: str,
    timestamp: str,
    *,
    baseline: str = "# baseline\n\nbody v1\n",
    evolved: str = "# evolved\n\nbody v2 improved\n",
    baseline_score: float = 0.50,
    evolved_score: float = 0.55,
    auto_merge: bool = False,
    reason: str = "improvement below threshold",
    mode: str = "propose",
) -> Path:
    writer = ProposalWriter(proposals_dir)
    record = build_proposal_record(
        skill_name=skill_name,
        baseline_text=baseline,
        evolved_text=evolved,
        baseline_score=baseline_score,
        evolved_score=evolved_score,
        decision=_FakeDecision(auto_merge=auto_merge, reason=reason),
        constraint_results=[
            _FakeConstraint("size", True, "ok"),
            _FakeConstraint("has_frontmatter", True, "ok"),
        ],
        mode=mode,
        timestamp=timestamp,
    )
    return writer.write(record)


@pytest.fixture
def proposals_dir(tmp_path: Path) -> Path:
    d = tmp_path / "proposals"
    d.mkdir()
    return d


# ──────────────────────────── discovery ────────────────────────────
class TestDiscovery:
    def test_empty_dir_returns_empty_list(self, proposals_dir: Path):
        assert pr.discover_proposals(proposals_dir) == []

    def test_missing_dir_returns_empty_list(self, tmp_path: Path):
        assert pr.discover_proposals(tmp_path / "nonexistent") == []

    def test_discovers_single_proposal(self, proposals_dir: Path):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        entries = pr.discover_proposals(proposals_dir)
        assert len(entries) == 1
        e = entries[0]
        assert e.skill_name == "skill-a"
        assert e.timestamp == "20260418_100000"
        assert e.status == PROPOSAL_STATUS_PENDING
        assert e.mode == "propose"

    def test_newest_first(self, proposals_dir: Path):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        _write_proposal(proposals_dir, "skill-a", "20260418_150000")
        _write_proposal(proposals_dir, "skill-a", "20260418_120000")
        entries = pr.discover_proposals(proposals_dir)
        timestamps = [e.timestamp for e in entries]
        assert timestamps == ["20260418_150000", "20260418_120000", "20260418_100000"]

    def test_skips_non_proposal_dirs(self, proposals_dir: Path):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        # hidden dir ignored
        (proposals_dir / ".cache").mkdir()
        # dir without decision.json ignored
        (proposals_dir / "skill-b" / "bogus").mkdir(parents=True)
        entries = pr.discover_proposals(proposals_dir)
        assert len(entries) == 1
        assert entries[0].skill_name == "skill-a"

    def test_find_proposal_by_timestamp(self, proposals_dir: Path):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        _write_proposal(proposals_dir, "skill-a", "20260418_150000")
        found = pr.find_proposal(proposals_dir, "skill-a", "20260418_100000")
        assert found is not None
        assert found.timestamp == "20260418_100000"

    def test_find_proposal_latest_when_no_timestamp(self, proposals_dir: Path):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        _write_proposal(proposals_dir, "skill-a", "20260418_150000")
        found = pr.find_proposal(proposals_dir, "skill-a")
        assert found is not None
        assert found.timestamp == "20260418_150000"

    def test_find_proposal_missing_returns_none(self, proposals_dir: Path):
        assert pr.find_proposal(proposals_dir, "nope") is None


# ──────────────────────────── list ────────────────────────────
class TestListCmd:
    def test_list_empty(self, proposals_dir: Path, capsys):
        rc = pr.main(["--proposals-dir", str(proposals_dir), "list"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "No proposals" in out

    def test_list_shows_all(self, proposals_dir: Path, capsys):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        _write_proposal(proposals_dir, "skill-b", "20260418_110000")
        rc = pr.main(["--proposals-dir", str(proposals_dir), "list"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "skill-a" in out
        assert "skill-b" in out
        assert "Total: 2" in out

    def test_list_filters_by_status(self, proposals_dir: Path, capsys):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        p2 = _write_proposal(proposals_dir, "skill-b", "20260418_110000")
        (p2 / "STATUS").write_text(PROPOSAL_STATUS_APPROVED + "\n")
        rc = pr.main(
            ["--proposals-dir", str(proposals_dir), "list", "--status", "approved"]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "skill-b" in out
        assert "skill-a" not in out


# ──────────────────────────── show / diff ────────────────────────────
class TestShowDiffCmds:
    def test_show_prints_review(self, proposals_dir: Path, capsys):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        rc = pr.main(
            [
                "--proposals-dir",
                str(proposals_dir),
                "show",
                "skill-a",
                "20260418_100000",
            ]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "Evolution Proposal" in out
        assert "skill-a" in out
        assert "[status: PENDING]" in out

    def test_show_latest_when_no_timestamp(self, proposals_dir: Path, capsys):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        _write_proposal(proposals_dir, "skill-a", "20260418_150000")
        rc = pr.main(["--proposals-dir", str(proposals_dir), "show", "skill-a"])
        assert rc == 0

    def test_show_missing_returns_2(self, proposals_dir: Path, capsys):
        rc = pr.main(["--proposals-dir", str(proposals_dir), "show", "nope"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "not found" in err.lower()

    def test_diff_prints_patch(self, proposals_dir: Path, capsys):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        rc = pr.main(
            [
                "--proposals-dir",
                str(proposals_dir),
                "diff",
                "skill-a",
                "20260418_100000",
            ]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "baseline_skill.md" in out
        assert "evolved_skill.md" in out


# ──────────────────────────── approve ────────────────────────────
class TestApproveCmd:
    def _make_live_skill(self, tmp_path: Path, skill_name: str) -> Path:
        """Create a fake hermes-agent layout with a bundled skill."""
        root = tmp_path / "hermes-agent"
        skill_dir = root / "skills" / "test-category" / skill_name
        skill_dir.mkdir(parents=True)
        sk = skill_dir / "SKILL.md"
        sk.write_text(
            "---\nname: " + skill_name + "\ndescription: live skill\n---\n\nlive body\n"
        )
        return sk

    def test_approve_no_merge_just_flips_status(
        self, proposals_dir: Path, tmp_path: Path, capsys
    ):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        live = self._make_live_skill(tmp_path, "skill-a")
        original = live.read_text()
        rc = pr.main(
            [
                "--proposals-dir",
                str(proposals_dir),
                "approve",
                "skill-a",
                "20260418_100000",
                "--no-merge",
                "--hermes-agent-path",
                str(tmp_path / "hermes-agent"),
            ]
        )
        assert rc == 0
        # live skill untouched
        assert live.read_text() == original
        # STATUS flipped
        status = (
            proposals_dir / "skill-a" / "20260418_100000" / "STATUS"
        ).read_text()
        assert status.splitlines()[0] == PROPOSAL_STATUS_APPROVED

    def test_approve_writes_back_and_backs_up(
        self, proposals_dir: Path, tmp_path: Path, capsys
    ):
        _write_proposal(
            proposals_dir,
            "skill-a",
            "20260418_100000",
            evolved="---\nname: skill-a\ndescription: evolved\n---\n\nevolved body\n",
        )
        live = self._make_live_skill(tmp_path, "skill-a")
        original = live.read_text()
        assert "live body" in original

        rc = pr.main(
            [
                "--proposals-dir",
                str(proposals_dir),
                "approve",
                "skill-a",
                "20260418_100000",
                "--hermes-agent-path",
                str(tmp_path / "hermes-agent"),
            ]
        )
        assert rc == 0
        # Live skill replaced
        assert "evolved body" in live.read_text()
        # Backup exists with original content
        backups = list((live.parent / ".backups").glob("SKILL.md.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text() == original
        # Merge receipt written
        receipt = (
            proposals_dir
            / "skill-a"
            / "20260418_100000"
            / "merge_receipt.json"
        )
        assert receipt.exists()
        data = json.loads(receipt.read_text())
        assert data["merged_to"] == str(live)
        assert data["backup_path"] == str(backups[0])
        # STATUS with approver metadata
        status = (
            proposals_dir / "skill-a" / "20260418_100000" / "STATUS"
        ).read_text()
        assert status.splitlines()[0] == PROPOSAL_STATUS_APPROVED
        assert "approved_by=" in status

    def test_approve_already_approved_is_idempotent(
        self, proposals_dir: Path, tmp_path: Path, capsys
    ):
        proposal_dir = _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        (proposal_dir / "STATUS").write_text(PROPOSAL_STATUS_APPROVED + "\n")
        rc = pr.main(
            [
                "--proposals-dir",
                str(proposals_dir),
                "approve",
                "skill-a",
                "20260418_100000",
                "--no-merge",
            ]
        )
        assert rc == 0
        assert "Already APPROVED" in capsys.readouterr().out

    def test_approve_rejected_requires_force(
        self, proposals_dir: Path, tmp_path: Path, capsys
    ):
        proposal_dir = _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        (proposal_dir / "STATUS").write_text(PROPOSAL_STATUS_REJECTED + "\n")
        rc = pr.main(
            [
                "--proposals-dir",
                str(proposals_dir),
                "approve",
                "skill-a",
                "20260418_100000",
                "--no-merge",
            ]
        )
        assert rc == 3
        # --force allows it
        rc2 = pr.main(
            [
                "--proposals-dir",
                str(proposals_dir),
                "approve",
                "skill-a",
                "20260418_100000",
                "--no-merge",
                "--force",
            ]
        )
        assert rc2 == 0

    def test_approve_live_skill_missing_returns_4(
        self, proposals_dir: Path, tmp_path: Path, capsys
    ):
        _write_proposal(proposals_dir, "ghost-skill", "20260418_100000")
        # Empty hermes-agent dir, no skills in it
        (tmp_path / "hermes-agent" / "skills").mkdir(parents=True)
        rc = pr.main(
            [
                "--proposals-dir",
                str(proposals_dir),
                "approve",
                "ghost-skill",
                "20260418_100000",
                "--hermes-agent-path",
                str(tmp_path / "hermes-agent"),
            ]
        )
        assert rc == 4
        # STATUS was still flipped (truth is authoritative on disk)
        status = (
            proposals_dir / "ghost-skill" / "20260418_100000" / "STATUS"
        ).read_text()
        assert status.splitlines()[0] == PROPOSAL_STATUS_APPROVED


# ──────────────────────────── reject ────────────────────────────
class TestRejectCmd:
    def test_reject_flips_status(self, proposals_dir: Path, capsys):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        rc = pr.main(
            [
                "--proposals-dir",
                str(proposals_dir),
                "reject",
                "skill-a",
                "20260418_100000",
            ]
        )
        assert rc == 0
        status = (
            proposals_dir / "skill-a" / "20260418_100000" / "STATUS"
        ).read_text()
        assert status.splitlines()[0] == PROPOSAL_STATUS_REJECTED

    def test_reject_with_reason(self, proposals_dir: Path, capsys):
        _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        rc = pr.main(
            [
                "--proposals-dir",
                str(proposals_dir),
                "reject",
                "skill-a",
                "20260418_100000",
                "--reason",
                "hallucinated flag",
            ]
        )
        assert rc == 0
        entry = pr.find_proposal(proposals_dir, "skill-a", "20260418_100000")
        assert entry is not None
        assert entry.rejection_reason == "hallucinated flag"

    def test_reject_approved_requires_force(self, proposals_dir: Path, capsys):
        proposal_dir = _write_proposal(proposals_dir, "skill-a", "20260418_100000")
        (proposal_dir / "STATUS").write_text(PROPOSAL_STATUS_APPROVED + "\n")
        rc = pr.main(
            [
                "--proposals-dir",
                str(proposals_dir),
                "reject",
                "skill-a",
                "20260418_100000",
            ]
        )
        assert rc == 3

    def test_reject_missing_returns_2(self, proposals_dir: Path, capsys):
        rc = pr.main(
            [
                "--proposals-dir",
                str(proposals_dir),
                "reject",
                "nope",
                "20260101_000000",
            ]
        )
        assert rc == 2

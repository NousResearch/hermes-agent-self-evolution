"""Tests for the deployment path and the scheduler.

Neither existed before. ``create_pr`` was a config field with no
implementation, so the README's "all changes go through PR review" guardrail
was documentation rather than behaviour; and the rotation driver was a shell
script outside the repository walking the library alphabetically at a cadence
that implied a 13-month cycle.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evolution.core.hermes_paths import HermesInstall
from evolution.core.outcome_signals import OutcomeStats
from evolution.deploy.canary import (
    PROMOTE,
    ROLLBACK,
    WAIT,
    CanaryLedger,
    CanaryRecord,
    deploy_canary,
    evaluate_canary,
    rollback_canary,
)
from evolution.deploy.pr import PRPublisher, build_pr_body, safe_branch_name
from evolution.monitor.rotation import (
    RotationState,
    SkillRecord,
    build_plan,
    coverage_estimate,
    discover_skills,
    prioritize,
)


# ── PR publishing ───────────────────────────────────────────────────────


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "skills" / "demo").mkdir(parents=True)
    (root / "skills" / "demo" / "SKILL.md").write_text("---\nname: demo\n---\n\noriginal\n")
    git("init", "-q", cwd=root)
    git("config", "user.email", "t@example.com", cwd=root)
    git("config", "user.name", "T", cwd=root)
    # Neutralize developer global config: a machine with commit.gpgsign or a
    # global hooksPath would otherwise fail these commits for reasons that
    # have nothing to do with the code under test.
    git("config", "commit.gpgsign", "false", cwd=root)
    git("config", "tag.gpgsign", "false", cwd=root)
    git("config", "core.hooksPath", str(root / ".git" / "hooks"), cwd=root)
    git("add", "-A", cwd=root)
    git("commit", "-qm", "init", "--no-verify", cwd=root)
    return root


class TestBranchNaming:
    def test_names_are_git_safe(self):
        assert safe_branch_name("my skill/name!", "20260101_010101") == "evolve/my-skill-name-20260101_010101"

    def test_an_unusable_name_still_produces_a_branch(self):
        assert safe_branch_name("///", "ts").startswith("evolve/skill-")


class TestPRBody:
    def test_body_carries_the_measurement_and_the_gates(self):
        body = build_pr_body(
            "demo",
            report_markdown="**Verdict: SHIP.** +0.100 beyond the ±0.020 noise band",
            constraint_lines=["✓ **size_limit** — Size OK"],
            run_metadata={"iterations": 10},
        )
        assert "Verdict: SHIP" in body
        assert "size_limit" in body
        assert "iterations" in body

    def test_body_warns_that_the_diff_is_machine_authored(self):
        body = build_pr_body("demo", "report", [])
        assert "optimizer, not a human" in body
        assert "HOLD" in body  # tells the reviewer when to close rather than merge


class TestPRPublisher:
    def test_preflight_rejects_a_non_repo(self, tmp_path):
        ok, reason = PRPublisher(tmp_path).preflight()
        assert ok is False and "not a git repository" in reason

    def test_preflight_rejects_a_dirty_tree(self, repo):
        (repo / "stray.txt").write_text("uncommitted")
        ok, reason = PRPublisher(repo).preflight()
        assert ok is False and "dirty" in reason

    def test_preflight_passes_on_a_clean_repo(self, repo):
        assert PRPublisher(repo).preflight()[0] is True

    def test_dry_run_changes_nothing(self, repo):
        target = repo / "skills" / "demo" / "SKILL.md"
        result = PRPublisher(repo).publish(
            "demo", target, "NEW CONTENT", "title", "body", "ts", dry_run=True
        )
        assert result.created is False
        assert "dry run" in result.detail
        assert target.read_text() == "---\nname: demo\n---\n\noriginal\n"

    def test_commit_without_push_leaves_a_local_branch(self, repo):
        target = repo / "skills" / "demo" / "SKILL.md"
        result = PRPublisher(repo).publish(
            "demo", target, "EVOLVED\n", "evolve: demo", "body", "ts1", push=False
        )
        assert result.created is False
        assert "committed locally" in result.detail

        branches = subprocess.run(
            ["git", "branch", "--list", result.branch],
            cwd=str(repo), capture_output=True, text=True,
        ).stdout
        assert result.branch in branches

    def test_an_identical_artifact_does_not_open_an_empty_pr(self, repo):
        target = repo / "skills" / "demo" / "SKILL.md"
        original = target.read_text()
        result = PRPublisher(repo).publish(
            "demo", target, original, "title", "body", "ts2", push=False
        )
        assert result.created is False
        assert "identical" in result.detail

    def test_a_target_outside_the_repo_is_refused(self, repo, tmp_path):
        outside = tmp_path / "elsewhere.md"
        outside.write_text("x")
        result = PRPublisher(repo).publish("demo", outside, "x", "t", "b", "ts")
        assert result.created is False
        assert "outside the repo" in result.detail

    def test_the_working_copy_is_returned_to_its_original_branch(self, repo):
        before = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo), capture_output=True, text=True,
        ).stdout.strip()

        PRPublisher(repo).publish(
            "demo", repo / "skills" / "demo" / "SKILL.md",
            "EVOLVED\n", "t", "b", "ts3", push=True,  # push fails: no remote
        )

        after = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo), capture_output=True, text=True,
        ).stdout.strip()
        assert after == before


# ── canary ──────────────────────────────────────────────────────────────


class TestCanaryLedger:
    def test_round_trips_records(self, tmp_path):
        ledger = CanaryLedger(tmp_path / "ledger.json")
        ledger.add(
            CanaryRecord(
                skill_name="demo", target_path="/x", deployed_at="2026-01-01T00:00:00+00:00",
                baseline_sha="a", variant_sha="b",
            )
        )
        assert [r.skill_name for r in CanaryLedger(tmp_path / "ledger.json").all()] == ["demo"]

    def test_status_updates_are_persisted(self, tmp_path):
        path = tmp_path / "ledger.json"
        ledger = CanaryLedger(path)
        ledger.add(CanaryRecord("demo", "/x", "2026-01-01T00:00:00+00:00", "a", "b"))
        assert ledger.update_status("demo", "b", "rolled_back", "regressed")

        reloaded = CanaryLedger(path).all()[0]
        assert reloaded.status == "rolled_back"
        assert reloaded.notes and "regressed" in reloaded.notes[0]

    def test_active_excludes_closed_records(self, tmp_path):
        ledger = CanaryLedger(tmp_path / "l.json")
        ledger.add(CanaryRecord("a", "/x", "2026-01-01T00:00:00+00:00", "1", "2"))
        ledger.add(CanaryRecord("b", "/y", "2026-01-01T00:00:00+00:00", "3", "4", status="rolled_back"))
        assert [r.skill_name for r in ledger.active()] == ["a"]

    def test_a_corrupt_ledger_reads_as_empty(self, tmp_path):
        path = tmp_path / "l.json"
        path.write_text("{not json")
        assert CanaryLedger(path).all() == []


class TestCanaryDeployment:
    def test_deploy_writes_the_variant_and_backs_up_the_baseline(self, install, tmp_path):
        target = tmp_path / "SKILL.md"
        target.write_text("BASELINE")
        ledger = CanaryLedger(tmp_path / "l.json")

        record = deploy_canary(
            install, ledger, "ahrefs-seo-operations", target, "VARIANT", tmp_path / "backups"
        )

        assert target.read_text() == "VARIANT"
        assert Path(record.backup_path).read_text() == "BASELINE"
        assert record.baseline_observations == 10  # from the cron fixture

    def test_rollback_restores_the_baseline(self, install, tmp_path):
        target = tmp_path / "SKILL.md"
        target.write_text("BASELINE")
        ledger = CanaryLedger(tmp_path / "l.json")
        record = deploy_canary(install, ledger, "demo", target, "VARIANT", tmp_path / "b")

        assert rollback_canary(record, ledger) is True
        assert target.read_text() == "BASELINE"
        assert CanaryLedger(tmp_path / "l.json").all()[0].status == "rolled_back"

    def test_rollback_without_a_backup_fails_loudly(self, tmp_path):
        record = CanaryRecord("demo", str(tmp_path / "t.md"), "2026-01-01T00:00:00+00:00", "a", "b",
                              backup_path=str(tmp_path / "missing.md"))
        assert rollback_canary(record) is False


class TestCanaryEvaluation:
    def test_too_little_evidence_waits(self, install):
        record = CanaryRecord(
            "ahrefs-seo-operations", "/x", "2026-01-01T00:00:00+00:00", "a", "b",
            baseline_success_rate=0.8, baseline_observations=10,
        )
        verdict = evaluate_canary(install, record, min_observations=20)
        assert verdict.decision == WAIT
        assert "post-deploy observations" in verdict.reason

    def test_a_regression_rolls_back(self, install):
        # Baseline snapshot taken when only 2 runs existed and all passed; the
        # fixture now holds 10 with 2 failures, so the new slice is 6/8.
        record = CanaryRecord(
            "ahrefs-seo-operations", "/x", "2026-01-01T00:00:00+00:00", "a", "b",
            baseline_success_rate=1.0, baseline_observations=2,
        )
        verdict = evaluate_canary(install, record, min_observations=4)
        assert verdict.decision == ROLLBACK
        assert "fell" in verdict.reason

    def test_holding_steady_promotes(self, install):
        record = CanaryRecord(
            "ahrefs-seo-operations", "/x", "2026-01-01T00:00:00+00:00", "a", "b",
            baseline_success_rate=0.75, baseline_observations=2,
        )
        verdict = evaluate_canary(install, record, min_observations=4)
        assert verdict.decision == PROMOTE

    def test_the_verdict_states_what_it_rests_on(self, install):
        record = CanaryRecord(
            "ahrefs-seo-operations", "/x", "2026-01-01T00:00:00+00:00", "a", "b",
            baseline_success_rate=0.75, baseline_observations=2,
        )
        assert "cron executions" in evaluate_canary(install, record, min_observations=4).basis


# ── rotation ────────────────────────────────────────────────────────────


class TestCoverageEstimate:
    def test_the_deployed_cadence_is_stated_in_months(self):
        """2/week over 117 skills is 13.4 months, which nobody had worked out."""
        assert "months" in coverage_estimate(117, 2)

    def test_a_fast_cadence_is_stated_in_weeks(self):
        assert "weeks" in coverage_estimate(6, 3)

    def test_nothing_scheduled_says_never(self):
        assert coverage_estimate(100, 0) == "never (nothing scheduled)"


class TestPrioritize:
    def make_state(self, tmp_path, **results):
        state = RotationState(tmp_path / "state.json")
        for name, verdicts in results.items():
            for verdict in verdicts:
                state.record_result("ali", name, verdict)
        return state

    def test_a_failing_skill_outranks_everything(self, tmp_path):
        class Cron:
            def per_skill(self):
                return {"failing": OutcomeStats(total=20, succeeded=8)}

        state = self.make_state(tmp_path)
        ranked = prioritize([("ali", "failing"), ("ali", "fresh")], state, cron=Cron())
        assert ranked[0].name == "failing"
        assert "failing in production" in ranked[0].reason

    def test_never_evolved_outranks_recently_evolved(self, tmp_path):
        state = self.make_state(tmp_path, recent=["SHIP"])
        ranked = prioritize([("ali", "recent"), ("ali", "never")], state)
        assert ranked[0].name == "never"

    def test_the_cooldown_pushes_recent_skills_down(self, tmp_path):
        state = self.make_state(tmp_path, recent=["SHIP"])
        ranked = prioritize([("ali", "recent")], state, cooldown_days=30)
        assert ranked[0].priority < 0
        assert "cooldown" in ranked[0].reason

    def test_repeatedly_held_skills_are_deprioritized(self, tmp_path):
        state = self.make_state(tmp_path, stubborn=["HOLD"] * 4, other=["HOLD"])
        ranked = prioritize([("ali", "stubborn"), ("ali", "other")], state)
        assert ranked[-1].name == "stubborn"

    def test_every_candidate_explains_itself(self, tmp_path):
        ranked = prioritize([("ali", "x")], self.make_state(tmp_path))
        assert ranked[0].reason


class TestRotationState:
    def test_ships_and_holds_are_tracked_separately(self, tmp_path):
        state = RotationState(tmp_path / "s.json")
        state.record_result("ali", "x", "SHIP")
        state.record_result("ali", "x", "HOLD")
        state.record_result("ali", "x", "HOLD")

        record = state.get("ali", "x")
        assert record.attempts == 3
        assert record.ships == 1
        assert record.consecutive_holds == 2

    def test_a_ship_resets_the_hold_streak(self, tmp_path):
        state = RotationState(tmp_path / "s.json")
        state.record_result("ali", "x", "HOLD")
        state.record_result("ali", "x", "SHIP")
        assert state.get("ali", "x").consecutive_holds == 0

    def test_state_survives_a_save_and_reload(self, tmp_path):
        path = tmp_path / "s.json"
        state = RotationState(path)
        state.record_result("ali", "x", "SHIP")
        state.save()

        assert RotationState(path).load().get("ali", "x").ships == 1

    def test_days_since_is_none_before_a_first_run(self):
        assert SkillRecord(name="x").days_since() is None


class TestPlanning:
    def test_discovery_finds_profile_skills(self, install, hermes_root):
        for profile in ("ali", "musa"):
            d = hermes_root / "profiles" / profile / "skills" / f"skill-{profile}"
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text("---\nname: x\n---\n")

        found = discover_skills(install)
        assert ("ali", "skill-ali") in found
        assert ("musa", "skill-musa") in found

    def test_the_plan_reports_what_it_deferred(self, install, hermes_root, tmp_path):
        skills_dir = hermes_root / "profiles" / "ali" / "skills"
        for i in range(5):
            d = skills_dir / f"s{i}"
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text("---\nname: x\n---\n")

        plan = build_plan(install, RotationState(tmp_path / "s.json"), skills_per_run=2, profile="ali")

        assert len(plan.selected) == 2
        assert len(plan.deferred) == 3
        # A bounded run must never read as full coverage.
        assert "deferred" in plan.render()

    def test_an_empty_queue_is_representable(self, install, tmp_path):
        plan = build_plan(install, RotationState(tmp_path / "s.json"), skills_per_run=0)
        assert plan.selected == []

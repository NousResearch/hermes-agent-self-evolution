"""Tests for ground-truth fitness signals.

These are the signals that replace guessing: real exit codes from
``verification_evidence.db`` and real scheduled-job outcomes from
``cron/executions.db``, attributed to skills through ``jobs.json``.
"""

from __future__ import annotations

import sqlite3

import pytest

from evolution.core.outcome_signals import (
    CronOutcomeSignal,
    OutcomeStats,
    VerificationSignal,
    skill_production_health,
)


class TestOutcomeStats:
    def test_no_evidence_is_none_not_zero(self):
        """None and 0.0 mean different things and must not be conflated."""
        assert OutcomeStats().success_rate is None
        assert OutcomeStats(total=3, succeeded=0).success_rate == 0.0

    def test_rate_and_failure_count(self):
        stats = OutcomeStats(total=10, succeeded=7)
        assert stats.success_rate == 0.7
        assert stats.failed == 3

    def test_merge_adds_both_sides(self):
        merged = OutcomeStats(total=4, succeeded=3, failures=["a"]).merge(
            OutcomeStats(total=6, succeeded=6)
        )
        assert merged.total == 10
        assert merged.succeeded == 9

    def test_merge_caps_stored_failure_text(self):
        big = OutcomeStats(total=50, succeeded=0, failures=[f"f{i}" for i in range(30)])
        assert len(big.merge(big).failures) <= 20

    def test_summary_reads_cleanly(self):
        assert OutcomeStats().summary() == "no evidence"
        assert "8/10" in OutcomeStats(total=10, succeeded=8).summary()


class TestVerificationSignal:
    def test_events_are_read_across_profiles(self, install):
        assert len(VerificationSignal(install).events()) == 3

    def test_exit_code_decides_pass_fail(self, install):
        events = {e.command: e for e in VerificationSignal(install).events()}
        assert events["pytest tests/"].passed is True
        assert events["ruff check ."].passed is False

    def test_status_word_is_used_when_no_exit_code(self, install, hermes_root):
        conn = sqlite3.connect(hermes_root / "profiles" / "ali" / "verification_evidence.db")
        conn.execute(
            "INSERT INTO verification_events (session_id, command, kind, status, exit_code) "
            "VALUES ('s9','mystery','test','pass',NULL)"
        )
        conn.commit()
        conn.close()

        events = {e.command: e for e in VerificationSignal(install).events()}
        assert events["mystery"].passed is True

    def test_grouping_by_session(self, install):
        by_session = VerificationSignal(install).by_session()
        assert by_session["s1"].total == 2
        assert by_session["s1"].succeeded == 1
        assert by_session["s2"].success_rate == 1.0

    def test_overall_pass_rate(self, install):
        assert VerificationSignal(install).overall().success_rate == pytest.approx(2 / 3)

    def test_scoring_a_session_subset(self, install):
        assert VerificationSignal(install).score_for_sessions({"s2"}) == 1.0

    def test_unknown_sessions_have_no_evidence(self, install):
        assert VerificationSignal(install).score_for_sessions({"nope"}) is None

    def test_missing_database_is_not_an_error(self, install, hermes_root):
        (hermes_root / "profiles" / "ali" / "verification_evidence.db").unlink()
        assert VerificationSignal(install).events() == []


class TestCronOutcomeSignal:
    def test_per_job_counts_terminal_states_only(self, install):
        per_job = CronOutcomeSignal(install).per_job()
        # 8 completed + 2 failed; the 'running' row is not yet evidence.
        assert per_job["job-seo"].total == 10
        assert per_job["job-brief"].total == 5

    def test_running_jobs_are_excluded(self, install):
        assert CronOutcomeSignal(install).per_job()["job-brief"].succeeded == 5

    def test_failure_text_is_captured(self, install):
        assert "timeout" in CronOutcomeSignal(install).per_job()["job-seo"].failures[0]

    def test_outcomes_roll_up_to_skills(self, install):
        per_skill = CronOutcomeSignal(install).per_skill()
        assert per_skill["ahrefs-seo-operations"].total == 10
        assert per_skill["ahrefs-seo-operations"].success_rate == 0.8
        assert per_skill["partner-meeting-prep"].success_rate == 1.0

    def test_unknown_skill_has_no_evidence(self, install):
        assert CronOutcomeSignal(install).for_skill("not-a-skill").total == 0

    def test_overall_aggregates_every_job(self, install):
        assert CronOutcomeSignal(install).overall().total == 15

    def test_missing_executions_db_is_not_an_error(self, install, hermes_root):
        (hermes_root / "cron" / "executions.db").unlink()
        assert CronOutcomeSignal(install).per_job() == {}


class TestSkillProductionHealth:
    def test_combines_cron_and_verification(self, install):
        health = skill_production_health(install, "ahrefs-seo-operations", session_ids={"s1"})
        assert health.cron.total == 10
        assert health.verification.total == 2
        assert health.has_evidence

    def test_baseline_score_is_observation_weighted(self, install):
        health = skill_production_health(install, "ahrefs-seo-operations", session_ids={"s1"})
        # 8 of 10 cron plus 1 of 2 verification = 9 of 12.
        assert health.baseline_score() == pytest.approx(9 / 12)

    def test_no_evidence_scores_none(self, install):
        health = skill_production_health(install, "unheard-of")
        assert health.has_evidence is False
        assert health.baseline_score() is None
        assert health.describe() == "no production evidence"

    def test_describe_mentions_both_sources(self, install):
        described = skill_production_health(
            install, "ahrefs-seo-operations", session_ids={"s1"}
        ).describe()
        assert "cron" in described and "verified" in described

"""Tests for reading the store Hermes actually keeps sessions in.

The importer these replace globbed ``~/.hermes/sessions/*.json`` and read a
``messages`` key. On a real install that directory holds error request-dumps
and a routing mirror whose own README says "This is NOT the session list";
none of its files have a ``messages`` key. The result was zero mined examples
against 37,006 real messages sitting in SQLite.
"""

from __future__ import annotations

import sqlite3

import pytest

from evolution.core.hermes_paths import HermesInstall
from evolution.core.state_db import (
    HermesStateImporter,
    build_fts_query,
    load_jobs_to_skills,
    prompt_variant_outcomes,
    tool_usage_histogram,
)


class TestFtsQuery:
    def test_builds_an_or_query_of_quoted_terms(self):
        query = build_fts_query("ahrefs-seo-operations", "Run Ahrefs backlink checks for domains.")
        assert " OR " in query
        assert '"ahrefs"' in query

    def test_hyphens_and_underscores_become_separate_terms(self):
        query = build_fts_query("partner_meeting-prep", "")
        for term in ("partner", "meeting", "prep"):
            assert f'"{term}"' in query

    def test_stopwords_are_dropped(self):
        query = build_fts_query("skill", "the and for with that this")
        assert query == ""

    def test_terms_are_deduplicated(self):
        query = build_fts_query("backlink", "backlink backlink backlink")
        assert query.count('"backlink"') == 1


class TestMining:
    def test_pairs_user_asks_with_assistant_answers(self, install):
        mined = HermesStateImporter(install).mine(use_fts=False)
        tasks = {m.task_input: m.assistant_response for m in mined}
        assert tasks["run the deployment pipeline for staging"] == "Deployed to staging."

    def test_collects_tools_used_between_ask_and_answer(self, install):
        mined = HermesStateImporter(install).mine(use_fts=False)
        by_task = {m.task_input: m for m in mined}
        assert by_task["audit the ahrefs backlink report for the domain"].tools_used == [
            "tool_search",
            "read_file",
        ]

    def test_secrets_are_never_mined(self, install):
        mined = HermesStateImporter(install).mine(use_fts=False)
        assert all("sk-ant-api03" not in m.task_input for m in mined)

    def test_very_short_messages_are_skipped(self, install):
        mined = HermesStateImporter(install).mine(use_fts=False)
        assert all(m.task_input != "hi" for m in mined)

    def test_newest_sessions_come_first(self, install):
        mined = HermesStateImporter(install).mine(use_fts=False)
        assert mined[0].session_id == "s2"  # started_at 200 vs 100

    def test_limit_is_respected(self, install):
        assert len(HermesStateImporter(install).mine(use_fts=False, limit=1)) == 1

    def test_outcome_is_attached(self, install):
        mined = HermesStateImporter(install).mine(use_fts=False)
        outcome = next(m.outcome for m in mined if m.session_id == "s1")
        assert outcome is not None
        assert outcome.succeeded is True
        assert outcome.tool_call_count == 4
        assert outcome.total_tokens == 1200

    def test_extract_messages_returns_pipeline_shaped_dicts(self, install):
        msgs = HermesStateImporter(install).extract_messages(use_fts=False)
        assert msgs and set(msgs[0]) >= {"source", "task_input", "assistant_response", "session_id"}
        assert msgs[0]["source"] == "hermes"


class TestFtsNarrowing:
    def test_fts_narrows_to_matching_sessions(self, install):
        importer = HermesStateImporter(install)
        mined = importer.mine(skill_name="ahrefs-backlinks", skill_text="Ahrefs backlink audits")
        assert {m.session_id for m in mined} == {"s1"}

    def test_unmatched_query_returns_nothing_rather_than_everything(self, install):
        importer = HermesStateImporter(install)
        mined = importer.mine(skill_name="zzz-nonexistent", skill_text="quixotic zebra")
        assert mined == []

    def test_missing_index_falls_back_to_full_scan(self, tmp_path):
        from tests.conftest import build_state_db

        root = tmp_path / "h"
        (root / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
        (root / "config.yaml").write_text("x: 1\n")
        build_state_db(
            root / "state.db",
            sessions=[{"id": "a", "started_at": 1.0}],
            messages=[
                {"session_id": "a", "role": "user", "content": "please summarize the quarterly report", "timestamp": 1},
                {"session_id": "a", "role": "assistant", "content": "Summary.", "timestamp": 2},
            ],
            with_fts=False,
        )
        install = HermesInstall(root=root, source="test")
        mined = HermesStateImporter(install).mine(skill_name="anything", skill_text="unrelated")
        assert len(mined) == 1

    def test_retrieval_ranks_and_caps_rather_than_matching_everything(self, install, hermes_root):
        """An OR of a dozen terms matches most of a real corpus.

        Against the live install the unranked query returned 1,804 of 2,126
        pairs for one skill and 1,584 for an unrelated one. BM25 ordering plus
        a cap is what makes narrowing actually narrow.
        """
        from tests.conftest import build_state_db

        sessions, messages = [], []
        for i in range(30):
            sessions.append({"id": f"n{i}", "started_at": float(i)})
            # Every session mentions one weak term; only one is really about it.
            body = "please review the report" if i else "ahrefs backlink domain rating audit report"
            messages += [
                {"session_id": f"n{i}", "role": "user", "content": body, "timestamp": 1},
                {"session_id": f"n{i}", "role": "assistant", "content": "done", "timestamp": 2},
            ]
        root = hermes_root / "profiles" / "ranked"
        build_state_db(root / "state.db", sessions=sessions, messages=messages)

        importer = HermesStateImporter(install, profiles=["ranked"])
        capped = importer.mine(
            skill_name="ahrefs-backlinks",
            skill_text="Ahrefs backlink domain rating audit",
            match_cap=3,
        )

        assert 0 < len(capped) <= 3
        # The genuinely on-topic session must survive the cap.
        assert "n0" in {m.session_id for m in capped}

    def test_a_broken_fts_query_falls_back_instead_of_crashing(self, install):
        mined = HermesStateImporter(install)._mine_profile(
            install.profile("ali"), fts_query='"unclosed', limit=0
        )
        # Malformed MATCH must degrade to a scan, not take the run down.
        assert isinstance(mined, list)

class TestResilience:
    def test_a_corrupt_profile_does_not_break_the_others(self, install, hermes_root):
        (hermes_root / "profiles" / "musa" / "state.db").write_bytes(b"not a database")
        mined = HermesStateImporter(install).mine(use_fts=False)
        assert mined, "healthy profile must still be mined"

    def test_profile_filter_restricts_sources(self, install):
        importer = HermesStateImporter(install, profiles=["musa"])
        assert importer.mine(use_fts=False) == []

    def test_describe_sources_lists_only_profiles_with_a_state_db(self, install):
        described = HermesStateImporter(install).describe_sources()
        assert any("ali" in d for d in described)


class TestAggregates:
    def test_tool_histogram_counts_across_profiles(self, install):
        histogram = tool_usage_histogram(install)
        assert histogram["tool_search"] == 1
        assert histogram["read_file"] == 1
        assert histogram["terminal"] == 1

    def test_prompt_variants_group_by_hash(self, install):
        variants = prompt_variant_outcomes(install)
        assert variants["hashA"]["sessions"] == 2
        assert variants["hashA"]["success_rate"] == 1.0
        assert variants["hashB"]["success_rate"] == 0.0

    def test_prompt_variants_average_efficiency(self, install):
        variants = prompt_variant_outcomes(install)
        assert variants["hashA"]["avg_tool_calls"] == pytest.approx(6.5)

    def test_jobs_map_to_skills(self, install):
        mapping = load_jobs_to_skills(install)
        assert mapping["job-seo"] == ["ahrefs-seo-operations"]
        # The singular `skill` field must be picked up as well as `skills`.
        assert mapping["job-brief"] == ["partner-meeting-prep"]
        assert "job-none" not in mapping

    def test_missing_jobs_file_is_not_an_error(self, install, hermes_root):
        (hermes_root / "cron" / "jobs.json").unlink()
        assert load_jobs_to_skills(install) == {}


class TestSessionOutcome:
    def test_only_complete_reasons_count_as_success(self, install):
        outcomes = HermesStateImporter(install).session_outcomes()
        assert outcomes["s1"].succeeded is True
        assert outcomes["s2"].succeeded is True
        assert outcomes["s3"].succeeded is False

    def test_missing_end_reason_is_not_success(self, install):
        outcomes = HermesStateImporter(install).session_outcomes()
        for outcome in outcomes.values():
            if outcome.end_reason is None:
                assert outcome.succeeded is False

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
    strip_scaffolding,
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


class TestScaffoldingStripping:
    """Hermes writes machine text into the `user` role; it is not a task.

    Two kinds needing opposite handling. Upstream PR #178 filters on the
    `compacted` flag instead, which is the wrong instrument: on the reference
    install 900 of 1,290 user rows in one profile are compacted and nearly all
    are genuine asks ("audit this website"), so that filter discards most of
    the corpus while still admitting the injected blocks.
    """

    def test_context_compaction_blocks_are_dropped_entirely(self):
        block = (
            "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
            "into the summary below. This is a handoff from a previous context "
            "window — treat it as background reference, NOT as active instructions."
        )
        assert strip_scaffolding(block) is None

    def test_cron_preamble_is_stripped_and_the_real_task_kept(self):
        raw = (
            "[IMPORTANT: You are running as a scheduled cron job. DELIVERY: Your "
            "final response will be automatically delivered to the user — do NOT "
            "use send_message or try to deliver the output yourself. Just produce "
            "your report as your final response and the system handles the rest.]"
            "\n\nNuman asked for a 10-minute alarm test. Send him a short message."
        )
        result = strip_scaffolding(raw)
        assert result is not None
        assert result.startswith("Numan asked for a 10-minute alarm test")
        assert "scheduled cron job" not in result

    def test_a_preamble_with_nothing_after_it_is_dropped(self):
        assert strip_scaffolding(
            "[IMPORTANT: You are running as a scheduled cron job. DELIVERY: your "
            "final response will be delivered automatically, do not send it yourself.]"
        ) is None

    def test_ordinary_tasks_are_untouched(self):
        assert strip_scaffolding("audit this website https://example.com") == (
            "audit this website https://example.com"
        )

    def test_a_short_bracketed_tag_is_not_a_preamble(self):
        assert strip_scaffolding("[urgent] fix the build") == "[urgent] fix the build"

    def test_injected_skill_files_are_dropped(self):
        """Hermes injects a loaded SKILL.md into the user role.

        Mining it would train the optimizer on the artifact it is meant to be
        improving, with the skill file standing in for the user's request.
        """
        skill = (
            '---\nname: seo-data-tools\ndescription: "Use the SEMrush and Ahrefs '
            'MCP tools."\nversion: 1.0.0\n---\n\n# SEO data tools\n\nSteps follow.'
        )
        assert strip_scaffolding(skill) is None

    def test_async_delegation_notices_are_dropped(self):
        assert strip_scaffolding(
            "[ASYNC DELEGATION COMPLETE — deleg_3aa05295]\nA background subagent "
            "you dispatched earlier has finished."
        ) is None

    def test_a_message_merely_starting_with_dashes_is_kept(self):
        assert strip_scaffolding("--- can you check the deploy? ---") == (
            "--- can you check the deploy? ---"
        )

    def test_empty_input(self):
        assert strip_scaffolding("") is None
        assert strip_scaffolding(None) is None

    def test_scaffolding_never_reaches_mined_output(self, install, hermes_root):
        from tests.conftest import build_state_db

        root = hermes_root / "profiles" / "scaffold"
        build_state_db(
            root / "state.db",
            sessions=[{"id": "c1", "started_at": 1.0}],
            messages=[
                {"session_id": "c1", "role": "user", "timestamp": 1, "content":
                 "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
                 "into the summary below for the next context window handoff."},
                {"session_id": "c1", "role": "assistant", "timestamp": 2, "content": "ack"},
                {"session_id": "c1", "role": "user", "timestamp": 3, "content":
                 "[IMPORTANT: You are running as a scheduled cron job. DELIVERY: your "
                 "final response is delivered automatically, do not send it yourself.]"
                 "\n\nProduce the weekly SEO summary for deventity.com."},
                {"session_id": "c1", "role": "assistant", "timestamp": 4, "content": "Summary."},
            ],
        )

        mined = HermesStateImporter(install, profiles=["scaffold"]).mine(use_fts=False)

        assert len(mined) == 1
        assert mined[0].task_input == "Produce the weekly SEO summary for deventity.com."

    def test_genuinely_compacted_turns_are_still_mined(self, install, hermes_root):
        """The flag marks "out of the active window", not "machine-written"."""
        from tests.conftest import build_state_db

        root = hermes_root / "profiles" / "hist"
        build_state_db(
            root / "state.db",
            sessions=[{"id": "h1", "started_at": 1.0}],
            messages=[
                {"session_id": "h1", "role": "user", "timestamp": 1,
                 "content": "audit this website https://deventity.com"},
                {"session_id": "h1", "role": "assistant", "timestamp": 2, "content": "Done."},
            ],
        )
        mined = HermesStateImporter(install, profiles=["hist"]).mine(use_fts=False)
        assert [m.task_input for m in mined] == ["audit this website https://deventity.com"]

    def test_a_preamble_wrapping_a_skill_file_is_fully_peeled(self):
        """Scaffolding layers, so the checks must re-run on what a strip exposes.

        Real shape on the live install: an [IMPORTANT: ...invoked the "x" skill]
        preamble followed by the entire SKILL.md. Stripping only the preamble
        exposes the skill file and mines it as the user's request.
        """
        raw = (
            '[IMPORTANT: The user has invoked the "seo-data-tools" skill, '
            'indicating they want SEO data work done using these tools.]\n\n'
            '---\nname: seo-data-tools\ndescription: "Use SEMrush and Ahrefs."\n'
            '---\n\n# SEO data tools\n\nSteps follow.'
        )
        assert strip_scaffolding(raw) is None

    def test_peeling_is_bounded(self):
        """A pathological nest must terminate rather than spin."""
        nested = ("[" + "x" * 100 + "]") * 40 + "the real ask"
        assert strip_scaffolding(nested) is not None

    def test_nested_brackets_inside_a_preamble_do_not_end_it_early(self):
        """The cron preamble quotes "[SILENT]" inside itself.

        A naive scan for the first "]" closes on that and leaves the tail of
        the boilerplate behind as the task.
        """
        raw = (
            "[IMPORTANT: You are running as a scheduled cron job. DELIVERY: your "
            'final response is delivered automatically. SILENT: if there is nothing '
            'to report, respond with exactly "[SILENT]" (nothing else) to suppress '
            "delivery. Never combine [SILENT] with content.]\n\n"
            "Numan asked for a 10-minute alarm test."
        )
        assert strip_scaffolding(raw) == "Numan asked for a 10-minute alarm test."

    def test_an_unterminated_bracket_is_treated_as_content(self):
        raw = "[unterminated preamble that never closes and simply keeps going onward"
        assert strip_scaffolding(raw) == raw

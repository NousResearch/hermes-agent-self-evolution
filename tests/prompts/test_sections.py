"""Tests for the prompt section wrapper and its guardrails.

Offline by construction: no LM, no network, no hermes-agent checkout. The
fixture below is a faithful miniature of ``agent/prompt_builder.py`` - the four
evolvable constants as parenthesised implicit-concat strings, a dict-valued
PLATFORM_HINTS, and neighbouring constants that a write must never disturb.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from evolution.prompts.sections import (
    CACHE_BLOCK_TOKENS,
    IDENTITY_TRAITS,
    ActiveSessionReport,
    EvolvableSection,
    SectionInventory,
    SectionWriteError,
    UnknownSection,
    constant_strings,
    detect_active_session,
    estimate_tokens,
    load_sections,
    staged_prompt_write,
    validate_section_names,
    verify_only_sections_changed,
    write_sections,
)

PROMPT_SOURCE = '''"""Prompt builder."""

CONTEXT_FILE_MAX_CHARS = 20_000

DEFAULT_AGENT_IDENTITY = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. You communicate clearly, admit "
    "uncertainty when appropriate, and prioritize being genuinely useful over "
    "being verbose."
)

HERMES_AGENT_HELP_GUIDANCE = (
    "You run on Hermes Agent. The docs are authoritative.\\n"
    "Do not evolve me."
)

MEMORY_GUIDANCE = (
    "You have persistent memory across sessions. Save durable facts using the "
    "memory tool: user preferences, environment details, and stable conventions.\\n"
    "Do NOT save task progress, PR numbers, or commit SHAs."
)

SESSION_SEARCH_GUIDANCE = (
    "When the user references something from a past conversation, use "
    "session_search to recall it before asking them to repeat themselves."
)

SKILLS_GUIDANCE = (
    "After completing a complex task, save the approach as a skill.\\n"
    "Patch an outdated skill immediately with skill_manage(action='patch')."
)

KANBAN_GUIDANCE = (
    "# Kanban protocol\\n"
    "Leave me alone."
)

PLATFORM_HINTS = {
    "cli": "You are a CLI AI Agent. Try not to use markdown.",
    "telegram": "Standard Markdown is converted to Telegram formatting.",
}

COMPUTER_USE_GUIDANCE = computer_use_guidance("darwin")
'''


@pytest.fixture
def hermes_repo(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "prompt_builder.py").write_text(PROMPT_SOURCE, encoding="utf-8")
    return tmp_path


@pytest.fixture
def inventory(hermes_repo):
    return load_sections(hermes_repo)


@pytest.fixture
def builder_path(hermes_repo):
    return hermes_repo / "agent" / "prompt_builder.py"


# ──────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────


class TestDiscovery:
    def test_finds_every_evolvable_section(self, inventory):
        assert inventory.names == [
            "DEFAULT_AGENT_IDENTITY",
            "MEMORY_GUIDANCE",
            "SESSION_SEARCH_GUIDANCE",
            "SKILLS_GUIDANCE",
        ]

    def test_captures_baseline_text_and_path(self, inventory, builder_path):
        section = inventory.get("SESSION_SEARCH_GUIDANCE")
        assert section.baseline_text.startswith("When the user references")
        assert section.path == builder_path
        assert section.baseline_size == len(section.baseline_text)

    def test_ignores_neighbouring_prompt_constants(self, inventory):
        assert "KANBAN_GUIDANCE" not in inventory.names
        assert "HERMES_AGENT_HELP_GUIDANCE" not in inventory.names

    def test_platform_hints_reported_as_structured_not_evolvable(self, inventory):
        names = [s.name for s in inventory.structured]
        assert names == ["PLATFORM_HINTS"]
        hints = inventory.structured[0]
        assert hints.kind == "dict"
        assert set(hints.keys) == {"cli", "telegram"}
        assert hints.total_chars > 0
        assert "not a single string" in hints.reason

    def test_platform_hints_is_not_an_evolvable_section(self, inventory):
        assert "PLATFORM_HINTS" not in inventory.names
        with pytest.raises(UnknownSection):
            inventory.get("PLATFORM_HINTS")

    def test_missing_section_is_reported_not_raised(self, tmp_path):
        (tmp_path / "agent").mkdir()
        (tmp_path / "agent" / "prompt_builder.py").write_text(
            'MEMORY_GUIDANCE = "Save durable facts."\n', encoding="utf-8"
        )
        inv = load_sections(tmp_path)
        assert inv.names == ["MEMORY_GUIDANCE"]
        assert set(inv.missing) == {
            "DEFAULT_AGENT_IDENTITY",
            "SESSION_SEARCH_GUIDANCE",
            "SKILLS_GUIDANCE",
        }

    def test_absent_prompt_builder_yields_empty_inventory(self, tmp_path):
        inv = load_sections(tmp_path)
        assert inv.sections == []
        assert inv.prompt_builder is None
        assert len(inv.missing) == 4

    def test_unparsable_prompt_builder_does_not_crash(self, tmp_path):
        (tmp_path / "agent").mkdir()
        (tmp_path / "agent" / "prompt_builder.py").write_text(
            "MEMORY_GUIDANCE = (", encoding="utf-8"
        )
        inv = load_sections(tmp_path)
        assert inv.sections == []
        assert inv.structured == []


# ──────────────────────────────────────────────────────────────────────────
# Allowlist
# ──────────────────────────────────────────────────────────────────────────


class TestAllowlist:
    def test_validate_section_names_accepts_the_allowlist(self):
        assert validate_section_names(["MEMORY_GUIDANCE", "SKILLS_GUIDANCE"]) == []

    def test_validate_section_names_rejects_platform_hints(self):
        assert validate_section_names(["PLATFORM_HINTS"]) == ["PLATFORM_HINTS"]

    def test_validate_section_names_rejects_unrelated_constants(self):
        assert validate_section_names(["KANBAN_GUIDANCE", "MEMORY_GUIDANCE"]) == [
            "KANBAN_GUIDANCE"
        ]

    def test_load_sections_refuses_a_name_off_the_allowlist(self, hermes_repo):
        with pytest.raises(UnknownSection):
            load_sections(hermes_repo, names=["KANBAN_GUIDANCE"])

    def test_write_refuses_a_name_off_the_allowlist(self, hermes_repo, builder_path):
        before = builder_path.read_text(encoding="utf-8")
        with pytest.raises(SectionWriteError):
            write_sections(hermes_repo, {"KANBAN_GUIDANCE": "rewritten"})
        assert builder_path.read_text(encoding="utf-8") == before


# ──────────────────────────────────────────────────────────────────────────
# Growth ceiling
# ──────────────────────────────────────────────────────────────────────────


class TestGrowthCeiling:
    def _section(self, size=100, growth=0.2):
        return EvolvableSection(
            name="MEMORY_GUIDANCE",
            path=Path("prompt_builder.py"),
            baseline_text="y" * size,
            span=None,
            max_growth=growth,
        )

    def test_exactly_twenty_percent_growth_passes(self):
        section = self._section(100)
        check = section.check_growth("y" * 120)
        assert check.passed
        assert section.max_chars == 120

    def test_one_character_past_the_ceiling_fails(self):
        section = self._section(100)
        check = section.check_growth("y" * 121)
        assert not check.passed
        assert "ceiling" in check.message

    def test_shrinking_a_section_is_allowed(self):
        check = self._section(100).check_growth("y" * 40)
        assert check.passed

    def test_growth_of_reports_the_ratio(self):
        section = self._section(200)
        assert section.growth_of("y" * 250) == pytest.approx(0.25)

    def test_ceiling_comes_from_config_value(self):
        section = self._section(100, growth=0.5)
        assert section.max_chars == 150
        assert section.check_growth("y" * 150).passed
        assert not section.check_growth("y" * 151).passed

    def test_discovered_section_uses_the_configured_growth(self, hermes_repo):
        inv = load_sections(hermes_repo, max_growth=0.2)
        section = inv.get("MEMORY_GUIDANCE")
        assert section.max_chars == int(section.baseline_size * 1.2)

    def test_empty_candidate_is_rejected_even_though_it_shrinks(self, inventory):
        validation = inventory.validate("MEMORY_GUIDANCE", "   \n  ")
        assert not validation.passed
        assert "non_empty" in [c.name for c in validation.errors]


# ──────────────────────────────────────────────────────────────────────────
# Identity traits
# ──────────────────────────────────────────────────────────────────────────


class TestIdentityTraits:
    def test_baseline_identity_retains_every_trait(self, inventory):
        section = inventory.get("DEFAULT_AGENT_IDENTITY")
        checks = section.check_identity_traits(section.baseline_text)
        assert len(checks) == 1
        assert checks[0].passed
        for trait in IDENTITY_TRAITS:
            assert trait.label in checks[0].message

    def test_dropping_uncertainty_fails_and_names_the_trait(self, inventory):
        section = inventory.get("DEFAULT_AGENT_IDENTITY")
        candidate = (
            "You are Hermes Agent. You are helpful and direct, and you always "
            "give a confident answer."
        )
        checks = section.check_identity_traits(candidate)
        assert not checks[0].passed
        assert "admits uncertainty" in checks[0].message
        assert "helpful" not in checks[0].message.split(":")[1]

    def test_dropping_everything_names_all_three(self, inventory):
        section = inventory.get("DEFAULT_AGENT_IDENTITY")
        checks = section.check_identity_traits("You are a bot. You do tasks.")
        assert not checks[0].passed
        for trait in IDENTITY_TRAITS:
            assert trait.label in checks[0].message

    def test_a_paraphrase_still_counts_as_retained(self, inventory):
        section = inventory.get("DEFAULT_AGENT_IDENTITY")
        candidate = (
            "You assist the user, you get to the point, and you say so when you "
            "do not know something rather than guessing."
        )
        checks = section.check_identity_traits(candidate)
        assert checks[0].passed

    def test_identity_check_does_not_apply_to_other_sections(self, inventory):
        section = inventory.get("MEMORY_GUIDANCE")
        assert section.check_identity_traits("anything at all") == []
        assert not section.is_identity

    def test_validate_includes_the_identity_check_for_identity_only(self, inventory):
        identity = inventory.validate("DEFAULT_AGENT_IDENTITY", "You are a bot.")
        memory = inventory.validate("MEMORY_GUIDANCE", "Save durable facts.")
        assert "identity_traits" in [c.name for c in identity.checks]
        assert "identity_traits" not in [c.name for c in memory.checks]


# ──────────────────────────────────────────────────────────────────────────
# Caching boundary
# ──────────────────────────────────────────────────────────────────────────


class TestCachingBoundary:
    def test_estimate_tokens_rounds_up(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("abc") == 1
        assert estimate_tokens("a" * 4096) == 1024

    def test_assembled_prompt_contains_every_section(self, inventory):
        assembled = inventory.assembled_prompt()
        for section in inventory:
            assert section.baseline_text in assembled

    def test_assembled_prompt_applies_the_override(self, inventory):
        assembled = inventory.assembled_prompt({"MEMORY_GUIDANCE": "SHORT."})
        assert "SHORT." in assembled
        assert inventory.get("MEMORY_GUIDANCE").baseline_text not in assembled

    def test_preview_pads_with_the_widest_platform_hint(self, inventory):
        assert len(inventory.assembled_preview()) > len(inventory.assembled_prompt())

    def test_budget_check_passes_with_headroom(self, inventory):
        checks = {c.name: c for c in inventory.check_caching_boundary()}
        assert checks["caching_budget"].passed
        assert "headroom" in checks["caching_budget"].message

    def test_budget_check_fails_when_the_prefix_is_too_large(self, inventory):
        checks = {
            c.name: c
            for c in inventory.check_caching_boundary(budget_tokens=10)
        }
        assert not checks["caching_budget"].passed
        assert checks["caching_budget"].severity == "error"

    def test_crossing_a_cache_block_is_a_warning_not_an_error(self, inventory):
        huge = "z" * (CACHE_BLOCK_TOKENS * 4 * 3)
        checks = {
            c.name: c
            for c in inventory.check_caching_boundary({"MEMORY_GUIDANCE": huge})
        }
        block_check = checks["cache_block_stability"]
        assert not block_check.passed
        assert block_check.severity == "warning"
        assert block_check.is_warning and not block_check.is_error

    def test_a_block_crossing_passes_but_not_strictly(self):
        # A baseline that sits just inside one cache block, and a candidate
        # that is legal on growth but tips the assembled prefix into a second
        # block. Permissive mode ships it, strict mode does not.
        baseline = "m" * (CACHE_BLOCK_TOKENS * 4 - 100)
        inv = SectionInventory(
            prompt_builder=Path("prompt_builder.py"),
            sections=[
                EvolvableSection(
                    name="MEMORY_GUIDANCE",
                    path=Path("prompt_builder.py"),
                    baseline_text=baseline,
                    span=None,
                )
            ],
        )
        assert inv.cache_blocks() == 1
        candidate = "m" * int(len(baseline) * 1.2)
        validation = inv.validate("MEMORY_GUIDANCE", candidate)
        assert validation.passed
        assert not validation.passed_strict
        assert [c.name for c in validation.warnings] == ["cache_block_stability"]


# ──────────────────────────────────────────────────────────────────────────
# Writes
# ──────────────────────────────────────────────────────────────────────────


class TestWriteBack:
    def test_write_updates_only_the_target_constant(self, hermes_repo, builder_path):
        before = builder_path.read_text(encoding="utf-8")
        new_text = "Search past sessions first. Always."
        write_sections(hermes_repo, {"SESSION_SEARCH_GUIDANCE": new_text})

        after = builder_path.read_text(encoding="utf-8")
        before_constants = constant_strings(before)
        after_constants = constant_strings(after)

        assert after_constants["SESSION_SEARCH_GUIDANCE"] == new_text
        changed = [
            name
            for name, value in before_constants.items()
            if after_constants[name] != value
        ]
        assert changed == ["SESSION_SEARCH_GUIDANCE"]

    def test_neighbouring_constants_survive_byte_for_byte(self, hermes_repo, builder_path):
        write_sections(hermes_repo, {"MEMORY_GUIDANCE": "Save durable facts only."})
        after = builder_path.read_text(encoding="utf-8")
        assert "KANBAN_GUIDANCE = (\n    \"# Kanban protocol\\n\"" in after
        assert 'COMPUTER_USE_GUIDANCE = computer_use_guidance("darwin")' in after
        assert "CONTEXT_FILE_MAX_CHARS = 20_000" in after
        assert '"cli": "You are a CLI AI Agent. Try not to use markdown."' in after

    def test_written_file_is_rediscoverable(self, hermes_repo):
        new_text = "Save only what will still be true next month."
        write_sections(hermes_repo, {"MEMORY_GUIDANCE": new_text})
        assert load_sections(hermes_repo).get("MEMORY_GUIDANCE").baseline_text == new_text

    def test_multiple_sections_in_one_call(self, hermes_repo):
        write_sections(
            hermes_repo,
            {
                "MEMORY_GUIDANCE": "Memory: durable facts only.",
                "SKILLS_GUIDANCE": "Skills: save what recurs.",
            },
        )
        inv = load_sections(hermes_repo)
        assert inv.get("MEMORY_GUIDANCE").baseline_text == "Memory: durable facts only."
        assert inv.get("SKILLS_GUIDANCE").baseline_text == "Skills: save what recurs."

    def test_multiline_text_round_trips_exactly(self, hermes_repo):
        new_text = "Line one.\nLine two with 'quotes' and \"doubles\".\n"
        write_sections(hermes_repo, {"SKILLS_GUIDANCE": new_text})
        assert load_sections(hermes_repo).get("SKILLS_GUIDANCE").baseline_text == new_text

    def test_dry_run_validates_without_touching_disk(self, hermes_repo, builder_path):
        before = builder_path.read_text(encoding="utf-8")
        result = write_sections(
            hermes_repo, {"MEMORY_GUIDANCE": "Dry."}, dry_run=True
        )
        assert result.dry_run
        assert "Dry." in result.source
        assert builder_path.read_text(encoding="utf-8") == before

    def test_empty_update_map_is_refused(self, hermes_repo):
        with pytest.raises(SectionWriteError):
            write_sections(hermes_repo, {})

    def test_missing_prompt_builder_is_refused(self, tmp_path):
        with pytest.raises(SectionWriteError):
            write_sections(tmp_path, {"MEMORY_GUIDANCE": "x"})

    def test_verify_catches_a_neighbouring_constant_that_moved(self):
        before = 'A = "one"\nB = "two"\n'
        after = 'A = "one"\nB = "CHANGED"\n'
        with pytest.raises(SectionWriteError):
            verify_only_sections_changed(before, after, ["A"])

    def test_verify_allows_the_declared_section_to_change(self):
        before = 'A = "one"\nB = "two"\n'
        after = 'A = "ONE!"\nB = "two"\n'
        verify_only_sections_changed(before, after, ["A"])

    def test_verify_catches_a_removed_constant(self):
        with pytest.raises(SectionWriteError):
            verify_only_sections_changed('A = "one"\nB = "two"\n', 'A = "one"\n', ["A"])

    def test_verify_rejects_unparsable_output(self):
        with pytest.raises(SectionWriteError):
            verify_only_sections_changed('A = "one"\n', 'A = ("one"\n', ["A"])


class TestStagedWrite:
    def test_candidate_is_visible_inside_and_gone_after(self, hermes_repo, builder_path):
        original = builder_path.read_text(encoding="utf-8")
        with staged_prompt_write(hermes_repo, {"MEMORY_GUIDANCE": "STAGED TEXT."}):
            assert "STAGED TEXT." in builder_path.read_text(encoding="utf-8")
        assert builder_path.read_text(encoding="utf-8") == original

    def test_original_is_restored_even_when_the_body_raises(self, hermes_repo, builder_path):
        original = builder_path.read_text(encoding="utf-8")
        with pytest.raises(RuntimeError):
            with staged_prompt_write(hermes_repo, {"MEMORY_GUIDANCE": "STAGED."}):
                raise RuntimeError("gate exploded")
        assert builder_path.read_text(encoding="utf-8") == original

    def test_backup_is_written_before_staging(self, hermes_repo, builder_path, tmp_path):
        backup = tmp_path / "backups" / "prompt_builder.py.bak"
        original = builder_path.read_text(encoding="utf-8")
        with staged_prompt_write(
            hermes_repo, {"MEMORY_GUIDANCE": "STAGED."}, backup_path=backup
        ):
            pass
        assert backup.read_text(encoding="utf-8") == original

    def test_disabled_staging_is_a_no_op(self, hermes_repo, builder_path):
        original = builder_path.read_text(encoding="utf-8")
        with staged_prompt_write(
            hermes_repo, {"MEMORY_GUIDANCE": "STAGED."}, enabled=False
        ) as result:
            assert result is None
            assert builder_path.read_text(encoding="utf-8") == original


# ──────────────────────────────────────────────────────────────────────────
# Active session detection
# ──────────────────────────────────────────────────────────────────────────


class TestActiveSessionDetection:
    def test_clean_home_reports_no_session(self, tmp_path):
        report = detect_active_session(hermes_home=tmp_path, env={})
        assert isinstance(report, ActiveSessionReport)
        assert not report.active
        assert report.summary == "no active Hermes session detected"

    def test_session_env_var_is_evidence(self, tmp_path):
        report = detect_active_session(
            hermes_home=tmp_path, env={"HERMES_SESSION_ID": "abc123"}
        )
        assert report.active
        assert any("HERMES_SESSION_ID" in e for e in report.evidence)

    def test_kanban_task_env_var_is_evidence(self, tmp_path):
        report = detect_active_session(
            hermes_home=tmp_path, env={"HERMES_KANBAN_TASK": "T-9"}
        )
        assert report.active

    def test_lock_file_is_evidence(self, tmp_path):
        (tmp_path / "session.lock").write_text("", encoding="utf-8")
        report = detect_active_session(hermes_home=tmp_path, env={})
        assert report.active
        assert "lock file" in report.summary

    def test_live_pid_file_is_evidence(self, tmp_path):
        (tmp_path / "run.pid").write_text(str(os.getpid()), encoding="utf-8")
        report = detect_active_session(hermes_home=tmp_path, env={})
        assert report.active
        assert "live pid" in report.summary

    def test_dead_pid_file_is_ignored(self, tmp_path):
        # A pid that is certainly dead: one from a child already waited on.
        # A literal like 999999 is a live-able pid on Linux, where pid_max
        # defaults to 4194304, so it can exist on a busy CI machine.
        child = subprocess.Popen([sys.executable, "-c", ""])
        child.wait()
        (tmp_path / "run.pid").write_text(str(child.pid), encoding="utf-8")
        report = detect_active_session(hermes_home=tmp_path, env={})
        assert not report.active

    def test_garbage_pid_file_is_ignored(self, tmp_path):
        (tmp_path / "run.pid").write_text("not-a-pid", encoding="utf-8")
        assert not detect_active_session(hermes_home=tmp_path, env={}).active

    def test_home_comes_from_the_env_when_not_passed(self, tmp_path):
        (tmp_path / "agent.lock").write_text("", encoding="utf-8")
        report = detect_active_session(env={"HERMES_HOME": str(tmp_path)})
        assert report.active

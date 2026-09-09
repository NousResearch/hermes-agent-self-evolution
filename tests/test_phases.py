"""Tests for Phase 2 (tool descriptions) and Phase 3 (system prompt sections).

Both were empty ``__init__.py`` placeholders. Phase 2 is graded on a real
classification — which tool the agent actually used — rather than a rubric,
and Phase 3 reads the prompt Hermes actually ran rather than reconstructing it
from source.
"""

from __future__ import annotations

import pytest

from evolution.prompts.prompt_sections import (
    DEFAULT_PROMPT_GROWTH,
    SystemPrompt,
    load_live_prompts,
    load_prompt_file,
    prompt_cost_note,
    split_sections,
)
from evolution.tools.evolve_tool import (
    ToolSelectionModule,
    _clean_name,
    _instructions_for,
    _parse_catalog_block,
    build_selection_examples,
    make_selection_metric,
)
from evolution.tools.tool_catalog import (
    DISCOVERY_TOOLS,
    ToolCatalog,
    ToolChoiceExample,
    ToolSpec,
    discovery_overhead,
    mine_tool_choices,
)


# ── Phase 2: tool catalog ───────────────────────────────────────────────


@pytest.fixture
def catalog():
    return ToolCatalog(
        [
            ToolSpec("read_file", "Read a file from disk."),
            ToolSpec("terminal", "Run a shell command."),
            ToolSpec("search_files", "Search file contents."),
            ToolSpec("write_file", "Write a file to disk."),
        ]
    )


class TestToolCatalog:
    def test_render_is_one_line_per_tool(self, catalog):
        assert catalog.render().count("\n") == 3

    def test_descriptions_can_be_truncated_for_display(self, catalog):
        assert "…" in catalog.render(max_chars_each=5)

    def test_replacing_descriptions_leaves_others_untouched(self, catalog):
        updated = catalog.with_descriptions({"terminal": "Execute shell."})
        assert updated.get("terminal").description == "Execute shell."
        assert updated.get("read_file").description == "Read a file from disk."

    def test_unknown_names_in_the_update_are_ignored(self, catalog):
        assert len(catalog.with_descriptions({"ghost": "x"})) == 4

    def test_restriction_keeps_only_named_tools(self, catalog):
        assert catalog.restricted_to(["terminal"]).names() == ["terminal"]

    def test_total_chars_sums_descriptions(self, catalog):
        assert catalog.total_chars() == sum(len(t.description) for t in catalog.tools)

    def test_json_round_trip(self, catalog, tmp_path):
        path = tmp_path / "tools.json"
        path.write_text(catalog.to_json())
        assert ToolCatalog.from_json_file(path).names() == catalog.names()

    def test_openai_function_shape_is_accepted(self, tmp_path):
        path = tmp_path / "tools.json"
        path.write_text(
            '[{"type":"function","function":{"name":"read_file","description":"Read."}}]'
        )
        assert ToolCatalog.from_json_file(path).names() == ["read_file"]


class TestCatalogInstructionsRoundTrip:
    def test_the_catalog_survives_a_round_trip(self, catalog):
        parsed = _parse_catalog_block(_instructions_for(catalog))
        assert parsed == {t.name: t.description for t in catalog.tools}

    def test_a_mangled_block_falls_back_to_the_baseline(self, catalog):
        module = ToolSelectionModule(catalog)
        module.predictor.signature = module.predictor.signature.with_instructions("garbage")
        # A rewrite the optimizer broke must not empty every description.
        assert module.evolved_catalog().get("terminal").description == "Run a shell command."

    def test_a_partial_rewrite_keeps_untouched_tools(self, catalog):
        module = ToolSelectionModule(catalog)
        module.predictor.signature = module.predictor.signature.with_instructions(
            "TOOL CATALOG\n- terminal: Execute a shell command, precisely.\n"
        )
        evolved = module.evolved_catalog()
        assert evolved.get("terminal").description == "Execute a shell command, precisely."
        assert evolved.get("read_file").description == "Read a file from disk."


class TestCleanName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("terminal", "terminal"),
            ("`read_file`", "read_file"),
            ("use the `read_file` tool", "read_file"),
            ("I would pick search_files here", "search_files"),
            ("the best tool is read_file.", "read_file"),
        ],
    )
    def test_extracts_the_tool_name_from_prose(self, raw, expected, catalog):
        assert _clean_name(raw, catalog.names()) == expected

    def test_prefers_a_known_name_over_position(self, catalog):
        assert _clean_name("maybe terminal or read_file", catalog.names()) == "terminal"

    def test_works_without_a_known_list(self):
        assert _clean_name("use the `read_file` tool") == "read_file"

    def test_empty_input_is_survivable(self):
        assert _clean_name("") == ""


class TestSelectionExamples:
    def test_the_correct_tool_is_always_a_candidate(self, catalog):
        choices = [ToolChoiceExample("read the config", "read_file")]
        examples = build_selection_examples(choices, catalog, distractors=2)
        assert "read_file" in examples[0].candidates

    def test_choices_for_unknown_tools_are_dropped(self, catalog):
        choices = [ToolChoiceExample("do a thing", "not_a_tool")]
        assert build_selection_examples(choices, catalog) == []

    def test_candidate_lists_are_deterministic_across_arms(self, catalog):
        """Re-sampling distractors per arm would make the A/B partly an artifact."""
        choices = [ToolChoiceExample("read the config", "read_file")]
        first = build_selection_examples(choices, catalog, distractors=2)
        second = build_selection_examples(choices, catalog, distractors=2)
        assert first[0].candidates == second[0].candidates

    def test_discovery_tools_are_never_distractors(self):
        cat = ToolCatalog(
            [ToolSpec("read_file", "x"), ToolSpec("tool_search", "y"), ToolSpec("terminal", "z")]
        )
        examples = build_selection_examples(
            [ToolChoiceExample("read something please", "read_file")], cat, distractors=5
        )
        assert not (set(examples[0].candidates) & DISCOVERY_TOOLS)


class TestSelectionMetric:
    def test_a_correct_choice_scores_high(self, catalog):
        metric = make_selection_metric(catalog, size_budget=10_000)

        class Gold:
            expected_tool = "terminal"
            task = "run a command"

        class Pred:
            tool_name = "terminal"
            catalog_text = _instructions_for(catalog)

        assert metric(Gold(), Pred()).score > 0.9

    def test_a_wrong_choice_scores_zero_and_names_both_tools(self, catalog):
        metric = make_selection_metric(catalog, size_budget=10_000)

        class Gold:
            expected_tool = "terminal"
            task = "run a command"

        class Pred:
            tool_name = "read_file"
            catalog_text = _instructions_for(catalog)

        result = metric(Gold(), Pred())
        assert result.score == 0.0
        assert "terminal" in result.feedback and "read_file" in result.feedback
        assert "boundary" in result.feedback

    def test_a_bloated_catalog_is_penalized(self, catalog):
        metric = make_selection_metric(catalog, size_budget=catalog.total_chars())

        class Gold:
            expected_tool = "terminal"
            task = "run a command"

        class Tight:
            tool_name = "terminal"
            catalog_text = _instructions_for(catalog)

        class Bloated:
            tool_name = "terminal"
            catalog_text = _instructions_for(
                catalog.with_descriptions({t.name: t.description * 20 for t in catalog.tools})
            )

        assert metric(Gold(), Bloated()).score < metric(Gold(), Tight()).score


class TestMiningToolChoices:
    def test_real_tool_choices_are_mined(self, install):
        choices = mine_tool_choices(install, min_tool_uses=1)
        by_task = {c.task_input: c for c in choices}
        assert by_task["run the deployment pipeline for staging"].chosen_tool == "terminal"

    def test_discovery_calls_are_counted_not_used_as_the_label(self, install):
        choices = mine_tool_choices(install, min_tool_uses=1)
        seo = next(c for c in choices if "ahrefs" in c.task_input)
        # tool_search preceded read_file; the label is the real tool.
        assert seo.chosen_tool == "read_file"
        assert seo.discovery_calls == 1

    def test_rare_labels_are_dropped(self, install):
        assert mine_tool_choices(install, min_tool_uses=99) == []

    def test_secrets_never_become_tasks(self, install):
        assert all("sk-ant-api03" not in c.task_input for c in mine_tool_choices(install, min_tool_uses=1))

    def test_discovery_overhead_is_quantified(self, install):
        overhead = discovery_overhead(install)
        assert overhead["total_tool_calls"] == 3
        assert overhead["discovery_calls"] == 1
        assert overhead["discovery_share"] == pytest.approx(1 / 3)


# ── Phase 3: prompt sections ────────────────────────────────────────────


PROMPT = """You are Hermes.

# Tool use
Prefer read_file over terminal.

## Nuance
Detail.

# Style
Be concise.
"""


class TestSplitSections:
    def test_headings_become_sections(self):
        titles = [s.title for s in split_sections(PROMPT)]
        assert titles == ["(preamble)", "Tool use", "Nuance", "Style"]

    def test_heading_levels_are_recorded(self):
        levels = {s.title: s.level for s in split_sections(PROMPT)}
        assert levels["Tool use"] == 1 and levels["Nuance"] == 2

    def test_a_prompt_without_headings_is_one_section(self):
        sections = split_sections("Just prose, no headings at all.")
        assert len(sections) == 1 and sections[0].title == "(whole prompt)"

    def test_slugs_are_url_safe(self):
        assert split_sections("# Tool Use & Style\nx\n")[0].slug == "tool-use-style"


class TestSystemPrompt:
    def test_lookup_by_title_slug_and_substring(self):
        prompt = SystemPrompt(text=PROMPT, sections=split_sections(PROMPT))
        assert prompt.section("Tool use").title == "Tool use"
        assert prompt.section("tool-use").title == "Tool use"
        assert prompt.section("tool").title == "Tool use"

    def test_unknown_section_is_none(self):
        prompt = SystemPrompt(text=PROMPT, sections=split_sections(PROMPT))
        assert prompt.section("nonexistent") is None

    def test_replacing_a_section_only_changes_that_section(self):
        prompt = SystemPrompt(text=PROMPT, sections=split_sections(PROMPT))
        out = prompt.replace_section(prompt.section("Tool use"), "Always read_file.")
        assert "Always read_file." in out
        assert "# Style\nBe concise." in out
        assert out.startswith("You are Hermes.")

    def test_an_unchanged_section_round_trips_byte_identical(self):
        """Whitespace churn on every boundary would bury the real edit."""
        prompt = SystemPrompt(text=PROMPT, sections=split_sections(PROMPT))
        section = prompt.section("Tool use")
        assert prompt.replace_section(section, section.body) == PROMPT

    def test_size_reflects_the_whole_prompt(self):
        prompt = SystemPrompt(text=PROMPT, sections=split_sections(PROMPT))
        assert prompt.size == len(PROMPT)


class TestLoadLivePrompts:
    def test_prompts_are_read_from_state_db(self, install):
        prompts = load_live_prompts(install)
        assert prompts
        assert prompts[0].prompt_hash == "hashA"

    def test_most_used_variant_comes_first(self, install):
        prompts = load_live_prompts(install)
        assert prompts[0].sessions == 2
        assert prompts[0].sessions >= prompts[-1].sessions

    def test_sections_are_parsed_on_load(self, install):
        assert any(s.title == "Tool use" for s in load_live_prompts(install)[0].sections)

    def test_rare_variants_can_be_filtered_out(self, install):
        assert all(p.sessions >= 2 for p in load_live_prompts(install, min_sessions=2))

    def test_loading_from_a_file(self, tmp_path):
        path = tmp_path / "prompt.md"
        path.write_text(PROMPT)
        assert load_prompt_file(path).section("Style") is not None


class TestPromptCost:
    def test_growth_default_is_tighter_than_for_skills(self):
        """Every request pays for the system prompt; a skill loads when relevant."""
        assert DEFAULT_PROMPT_GROWTH < 0.20

    def test_cost_note_scales_by_request_volume(self):
        section = split_sections(PROMPT)[1]
        note = prompt_cost_note(section, delta_chars=400, requests=10_000)
        assert "adds" in note and "10,000" in note

    def test_a_shrink_reads_as_a_saving(self):
        section = split_sections(PROMPT)[1]
        assert "saves" in prompt_cost_note(section, delta_chars=-400, requests=100)

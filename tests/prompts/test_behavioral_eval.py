"""Tests for the behavioural test suite, judges, and harnesses.

Every test here runs with no network and no API key. The dspy pieces are
exercised through injected fake predictors - a plain object with a __call__
that returns a namespace - and the batch_runner path is exercised by asserting
on the argv and by parsing handwritten trajectory files. ``subprocess.run`` is
monkeypatched wherever it appears; the real batch_runner is never invoked.
"""

import json
import types

import pytest

from evolution.prompts import behavioral_eval as be
from evolution.prompts.behavioral_eval import (
    BEHAVIORAL_CATEGORIES,
    CATEGORY_SECTIONS,
    SECTION_MARKER_CLOSE,
    SECTION_MARKER_OPEN,
    SEED_SCENARIOS,
    AgentTranscript,
    BatchRunnerHarness,
    BehavioralJudge,
    BehavioralOutcome,
    BehavioralReport,
    BehavioralScenario,
    BehavioralSuite,
    DirectPromptHarness,
    HarnessFailure,
    ScenarioGenerator,
    SectionBehaviorModule,
    build_scenarios,
    build_section_instructions,
    extract_section_text,
    heuristic_outcome,
    make_behavioral_metric,
    select_harness,
)


class FakePredictor:
    """A stand-in for a dspy predictor. Records calls, returns canned fields."""

    def __init__(self, **fields):
        self.fields = fields
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if callable(self.fields.get("_dynamic")):
            return types.SimpleNamespace(**self.fields["_dynamic"](kwargs))
        return types.SimpleNamespace(**self.fields)


class ExplodingPredictor:
    def __call__(self, **kwargs):
        raise RuntimeError("no LM is loaded")


def transcript(response="", tools=(), scenario_id="x"):
    return AgentTranscript(
        prompt="p", response=response, tool_calls=tuple(tools), scenario_id=scenario_id
    )


# ──────────────────────────────────────────────────────────────────────────
# Seed bank
# ──────────────────────────────────────────────────────────────────────────


class TestSeedBank:
    def test_covers_all_five_plan_categories(self):
        found = {s.category for s in SEED_SCENARIOS}
        assert found == set(BEHAVIORAL_CATEGORIES)
        assert len(BEHAVIORAL_CATEGORIES) == 5

    def test_has_the_scenario_count_plan_asks_for(self):
        assert len(SEED_SCENARIOS) >= 60

    def test_each_category_is_evenly_covered(self):
        counts = {c: 0 for c in BEHAVIORAL_CATEGORIES}
        for scenario in SEED_SCENARIOS:
            counts[scenario.category] += 1
        assert all(n >= 10 for n in counts.values()), counts

    def test_section_matches_the_category_mapping(self):
        for scenario in SEED_SCENARIOS:
            assert scenario.section_under_test == CATEGORY_SECTIONS[scenario.category]

    def test_every_category_carries_a_negative_case(self):
        with_negatives = {
            s.category
            for s in SEED_SCENARIOS
            if s.forbidden_tools or "not" in s.expected_behavior.lower()
        }
        assert with_negatives == set(BEHAVIORAL_CATEGORIES)

    def test_platform_scenarios_name_their_platform(self):
        for scenario in SEED_SCENARIOS:
            if scenario.category == "platform_formatting":
                assert scenario.platform

    def test_scenario_ids_are_unique_and_stable(self):
        ids = [s.scenario_id for s in SEED_SCENARIOS]
        assert len(set(ids)) == len(ids)
        again = BehavioralScenario(
            prompt=SEED_SCENARIOS[0].prompt,
            section_under_test=SEED_SCENARIOS[0].section_under_test,
            expected_behavior="different rubric",
            category=SEED_SCENARIOS[0].category,
        )
        assert again.scenario_id == SEED_SCENARIOS[0].scenario_id


class TestScenarioSelection:
    def test_no_filter_returns_everything(self):
        assert len(build_scenarios()) == len(SEED_SCENARIOS)

    def test_section_filter_is_strict(self):
        selected = build_scenarios(sections=["MEMORY_GUIDANCE"])
        assert selected
        assert {s.section_under_test for s in selected} == {"MEMORY_GUIDANCE"}

    def test_platform_can_be_excluded(self):
        selected = build_scenarios(include_platform=False)
        assert "platform_formatting" not in {s.category for s in selected}

    def test_category_filter(self):
        selected = build_scenarios(categories=["session_search"])
        assert {s.category for s in selected} == {"session_search"}

    def test_per_category_cap(self):
        selected = build_scenarios(per_category=3)
        counts = {}
        for scenario in selected:
            counts[scenario.category] = counts.get(scenario.category, 0) + 1
        assert set(counts.values()) == {3}


class TestSuite:
    def test_split_is_stratified_and_lossless(self):
        suite = BehavioralSuite.from_seeds()
        train, val, holdout = suite.split(seed=7)
        assert len(train) + len(val) + len(holdout) == len(suite)
        ids = [s.scenario_id for s in train + val + holdout]
        assert len(set(ids)) == len(ids)
        for split in (train, val, holdout):
            assert {s.category for s in split} == set(BEHAVIORAL_CATEGORIES)

    def test_split_is_deterministic_for_a_seed(self):
        suite = BehavioralSuite.from_seeds()
        first = [s.scenario_id for s in suite.split(seed=3)[2]]
        second = [s.scenario_id for s in suite.split(seed=3)[2]]
        assert first == second

    def test_for_section_filters(self):
        suite = BehavioralSuite.from_seeds()
        picked = suite.for_section("SKILLS_GUIDANCE")
        assert picked and all(s.section_under_test == "SKILLS_GUIDANCE" for s in picked)

    def test_save_load_round_trip(self, tmp_path):
        suite = BehavioralSuite.from_seeds(categories=["memory_guidance"])
        path = suite.save(tmp_path / "scenarios.jsonl")
        reloaded = BehavioralSuite.load(path)
        assert [s.to_dict() for s in reloaded] == [s.to_dict() for s in suite]

    def test_categories_counts(self):
        suite = BehavioralSuite.from_seeds(categories=["identity_tone"])
        assert list(suite.categories()) == ["identity_tone"]


# ──────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────


class TestScenarioGenerator:
    def test_parses_a_clean_json_array(self):
        payload = json.dumps(
            [
                {
                    "prompt": "Remember I use tabs.",
                    "expected_behavior": "Saves the preference.",
                    "difficulty": "easy",
                    "expected_tools": ["memory"],
                    "forbidden_tools": [],
                }
            ]
        )
        generator = ScenarioGenerator(predictor=FakePredictor(scenarios=payload))
        out = generator.generate("MEMORY_GUIDANCE", "text", "memory_guidance", count=3)
        assert len(out) == 1
        assert out[0].expected_tools == ("memory",)
        assert out[0].source == "generated"
        assert out[0].section_under_test == "MEMORY_GUIDANCE"

    def test_passes_the_section_through_to_the_predictor(self):
        predictor = FakePredictor(scenarios="[]")
        ScenarioGenerator(predictor=predictor).generate(
            "SKILLS_GUIDANCE", "the section text", "skills_guidance", count=2
        )
        assert predictor.calls[0]["section_name"] == "SKILLS_GUIDANCE"
        assert predictor.calls[0]["section_text"] == "the section text"
        assert predictor.calls[0]["num_scenarios"] == 2

    def test_tolerates_prose_around_the_json(self):
        raw = 'Sure! Here you go:\n[{"prompt": "hi", "expected_behavior": "answers"}]\nDone.'
        out = ScenarioGenerator(predictor=FakePredictor(scenarios=raw)).generate(
            "MEMORY_GUIDANCE", "t", "memory_guidance"
        )
        assert len(out) == 1

    def test_unparsable_output_yields_nothing(self):
        out = ScenarioGenerator(predictor=FakePredictor(scenarios="sorry, no")).generate(
            "MEMORY_GUIDANCE", "t", "memory_guidance"
        )
        assert out == []

    def test_incomplete_items_are_dropped(self):
        payload = json.dumps([{"prompt": "hi"}, {"expected_behavior": "x"}])
        out = ScenarioGenerator(predictor=FakePredictor(scenarios=payload)).generate(
            "MEMORY_GUIDANCE", "t", "memory_guidance"
        )
        assert out == []

    def test_respects_the_requested_count(self):
        payload = json.dumps(
            [{"prompt": f"p{i}", "expected_behavior": "does it"} for i in range(9)]
        )
        out = ScenarioGenerator(predictor=FakePredictor(scenarios=payload)).generate(
            "MEMORY_GUIDANCE", "t", "memory_guidance", count=4
        )
        assert len(out) == 4


# ──────────────────────────────────────────────────────────────────────────
# Heuristic scoring
# ──────────────────────────────────────────────────────────────────────────


class TestHeuristicScoring:
    def _scenario(self, **kw):
        base = dict(
            prompt="Remember I use tabs.",
            section_under_test="MEMORY_GUIDANCE",
            expected_behavior="Saves the preference to memory as a durable fact.",
            category="memory_guidance",
        )
        base.update(kw)
        return BehavioralScenario(**base)

    def test_calling_the_expected_tool_scores_higher_than_not(self):
        scenario = self._scenario(expected_tools=("memory",))
        good = heuristic_outcome(scenario, transcript("Saved the preference.", ["memory"]))
        bad = heuristic_outcome(scenario, transcript("Saved the preference.", []))
        assert good.score > bad.score
        assert "did not call memory" in bad.feedback

    def test_forbidden_tool_call_is_punished(self):
        scenario = self._scenario(
            expected_behavior="Does not save session outcomes to memory.",
            forbidden_tools=("memory",),
        )
        violated = heuristic_outcome(scenario, transcript("Saved it.", ["memory"]))
        clean = heuristic_outcome(scenario, transcript("Noted, nothing to save.", []))
        assert violated.score < 0.3
        assert clean.score > violated.score
        assert "forbidden" in violated.feedback

    def test_empty_transcript_scores_zero(self):
        outcome = heuristic_outcome(self._scenario(), transcript("", []))
        assert outcome.score == 0.0
        assert not outcome.passed

    def test_pass_threshold_is_applied(self):
        scenario = self._scenario(expected_tools=("memory",))
        outcome = heuristic_outcome(
            scenario,
            transcript("Saved the preference to memory as a durable fact.", ["memory"]),
        )
        assert outcome.passed
        assert outcome.score >= be.PASS_THRESHOLD

    def test_identity_uncertainty_is_rewarded_when_the_rubric_asks(self):
        scenario = BehavioralScenario(
            prompt="Will the stock go up?",
            section_under_test="DEFAULT_AGENT_IDENTITY",
            expected_behavior="Admits it cannot know and does not invent a forecast.",
            category="identity_tone",
        )
        honest = heuristic_outcome(scenario, transcript("I cannot know that."))
        confident = heuristic_outcome(scenario, transcript("It will rise by 4 percent."))
        assert honest.score > confident.score
        assert "did not acknowledge" in confident.feedback

    def test_long_identity_answers_are_penalised(self):
        scenario = BehavioralScenario(
            prompt="One line please.",
            section_under_test="DEFAULT_AGENT_IDENTITY",
            expected_behavior="Returns one line with no preamble.",
            category="identity_tone",
        )
        short = heuristic_outcome(scenario, transcript("A mutex guards one line."))
        long = heuristic_outcome(scenario, transcript("A mutex guards one line. " * 200))
        assert short.score > long.score

    def test_markdown_on_the_cli_is_punished(self):
        scenario = BehavioralScenario(
            prompt="List the files.",
            section_under_test="PLATFORM_HINTS",
            expected_behavior="Plain terminal text, no markdown table and no bold.",
            category="platform_formatting",
            platform="cli",
        )
        plain = heuristic_outcome(scenario, transcript("a.txt, b.txt, plain text table"))
        marked = heuristic_outcome(
            scenario, transcript("**a.txt** plain text markdown table\n| x | y |\n|---|---|")
        )
        assert plain.score > marked.score
        assert "markdown" in marked.feedback

    def test_missing_media_tag_is_punished_where_it_is_required(self):
        scenario = BehavioralScenario(
            prompt="Send the logo.",
            section_under_test="PLATFORM_HINTS",
            expected_behavior="Includes a MEDIA:/absolute/path tag so it arrives natively.",
            category="platform_formatting",
            platform="telegram",
        )
        with_tag = heuristic_outcome(
            scenario, transcript("Here it is MEDIA:/tmp/logo.png natively")
        )
        without = heuristic_outcome(scenario, transcript("Here it is, natively."))
        assert with_tag.score > without.score

    def test_media_tag_where_forbidden_is_punished(self):
        scenario = BehavioralScenario(
            prompt="Save the chart and send it.",
            section_under_test="PLATFORM_HINTS",
            expected_behavior=(
                "States the absolute path in plain text and does not emit a MEDIA: tag."
            ),
            category="platform_formatting",
            platform="cli",
        )
        clean = heuristic_outcome(scenario, transcript("Wrote /tmp/chart.png plain text"))
        tagged = heuristic_outcome(scenario, transcript("MEDIA:/tmp/chart.png"))
        assert clean.score > tagged.score

    def test_scores_are_clamped_to_the_unit_interval(self):
        for scenario in SEED_SCENARIOS:
            outcome = heuristic_outcome(
                scenario, transcript(scenario.expected_behavior, ["memory", "skill_view"])
            )
            assert 0.0 <= outcome.score <= 1.0


# ──────────────────────────────────────────────────────────────────────────
# Judge and report
# ──────────────────────────────────────────────────────────────────────────


class TestBehavioralJudge:
    scenario = BehavioralScenario(
        prompt="Remember I use tabs.",
        section_under_test="MEMORY_GUIDANCE",
        expected_behavior="Saves the preference.",
        category="memory_guidance",
    )

    def test_llm_judge_uses_the_injected_predictor(self):
        predictor = FakePredictor(score=0.9, feedback="good")
        judge = BehavioralJudge(predictor=predictor)
        outcome = judge.score(self.scenario, transcript("Saved.", ["memory"]))
        assert outcome.score == pytest.approx(0.9)
        assert outcome.judge == "llm"
        assert predictor.calls[0]["tool_calls"] == "memory"

    def test_string_scores_are_parsed(self):
        judge = BehavioralJudge(predictor=FakePredictor(score="0.42", feedback="meh"))
        assert judge.score(self.scenario, transcript("x")).score == pytest.approx(0.42)

    def test_unparsable_scores_fall_back_to_neutral(self):
        judge = BehavioralJudge(predictor=FakePredictor(score="excellent", feedback=""))
        assert judge.score(self.scenario, transcript("x")).score == pytest.approx(0.5)

    def test_use_llm_false_is_the_heuristic(self):
        judge = BehavioralJudge(use_llm=False)
        outcome = judge.score(self.scenario, transcript("Saved.", ["memory"]))
        assert outcome.judge == "heuristic"

    def test_a_failing_predictor_degrades_to_the_heuristic(self):
        judge = BehavioralJudge(predictor=ExplodingPredictor())
        outcome = judge.score(self.scenario, transcript("Saved.", ["memory"]))
        assert outcome.judge == "heuristic"
        assert "llm judge unavailable" in outcome.feedback

    def test_missing_transcript_scores_zero(self):
        judge = BehavioralJudge(use_llm=False)
        report = judge.score_all([self.scenario], {})
        assert report.outcomes[0].score == 0.0
        assert "no transcript" in report.outcomes[0].feedback


class TestReport:
    def _report(self):
        return BehavioralReport(
            outcomes=[
                BehavioralOutcome("a", "memory_guidance", "MEMORY_GUIDANCE", 1.0, True),
                BehavioralOutcome("b", "memory_guidance", "MEMORY_GUIDANCE", 0.0, False),
                BehavioralOutcome("c", "identity_tone", "DEFAULT_AGENT_IDENTITY", 0.5, False),
            ]
        )

    def test_mean_and_pass_rate(self):
        report = self._report()
        assert report.mean_score == pytest.approx(0.5)
        assert report.pass_rate == pytest.approx(1 / 3)

    def test_by_category_and_section(self):
        report = self._report()
        assert report.by_category()["memory_guidance"] == pytest.approx(0.5)
        assert report.by_section()["DEFAULT_AGENT_IDENTITY"] == pytest.approx(0.5)

    def test_worst_is_sorted_ascending(self):
        assert [o.scenario_id for o in self._report().worst(2)] == ["b", "c"]

    def test_empty_report_is_zero_not_a_crash(self):
        empty = BehavioralReport()
        assert empty.mean_score == 0.0 and empty.pass_rate == 0.0

    def test_to_dict_is_json_serialisable(self):
        json.dumps(self._report().to_dict())


# ──────────────────────────────────────────────────────────────────────────
# batch_runner harness
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def batch_repo(tmp_path):
    (tmp_path / "batch_runner.py").write_text("# fake runner\n", encoding="utf-8")
    return tmp_path


class TestBatchRunnerHarness:
    def test_available_only_when_the_runner_exists(self, tmp_path, batch_repo):
        empty = tmp_path / "no-runner-here"
        empty.mkdir()
        assert BatchRunnerHarness(hermes_repo=batch_repo).available
        assert not BatchRunnerHarness(hermes_repo=empty).available

    def test_dataset_has_one_prompt_object_per_line(self, batch_repo, tmp_path):
        harness = BatchRunnerHarness(hermes_repo=batch_repo)
        scenarios = build_scenarios(categories=["memory_guidance"])[:3]
        path = harness.build_dataset(scenarios, tmp_path / "ds" / "eval.jsonl")
        lines = [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 3
        assert [row["prompt"] for row in lines] == [s.prompt for s in scenarios]
        assert lines[0]["scenario_id"] == scenarios[0].scenario_id
        assert lines[0]["section_under_test"] == "MEMORY_GUIDANCE"

    def test_command_uses_fire_style_underscored_flags(self, batch_repo):
        harness = BatchRunnerHarness(
            hermes_repo=batch_repo,
            model="anthropic/claude-sonnet-4.6",
            max_turns=7,
            num_workers=3,
            batch_size=5,
            python="/usr/bin/python3",
        )
        cmd = harness.build_command(
            dataset_file=batch_repo / "eval.jsonl",
            run_name="hase-run",
            system_prompt="EVOLVED PROMPT",
            sample_count=12,
        )
        assert cmd[0] == "/usr/bin/python3"
        assert cmd[1] == "batch_runner.py"
        assert f"--dataset_file={batch_repo / 'eval.jsonl'}" in cmd
        assert "--batch_size=5" in cmd
        assert "--run_name=hase-run" in cmd
        assert "--model=anthropic/claude-sonnet-4.6" in cmd
        assert "--max_turns=7" in cmd
        assert "--num_workers=3" in cmd
        assert "--max_samples=12" in cmd

    def test_command_carries_the_ephemeral_system_prompt(self, batch_repo):
        harness = BatchRunnerHarness(hermes_repo=batch_repo)
        cmd = harness.build_command(
            batch_repo / "eval.jsonl", "run", "SECTION UNDER TEST"
        )
        assert "--ephemeral_system_prompt=SECTION UNDER TEST" in cmd
        # Never a plain --system_prompt: that would be saved into trajectories.
        assert not any(c.startswith("--system_prompt") for c in cmd)

    def test_resume_flag_is_opt_in(self, batch_repo):
        harness = BatchRunnerHarness(hermes_repo=batch_repo)
        assert "--resume" not in harness.build_command(
            batch_repo / "d.jsonl", "r", "p"
        )
        assert "--resume" in harness.build_command(
            batch_repo / "d.jsonl", "r", "p", resume=True
        )

    def test_run_invokes_the_subprocess_in_the_repo(self, batch_repo, monkeypatch):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["kwargs"] = kwargs
            out = batch_repo / "data" / "hase-run"
            out.mkdir(parents=True, exist_ok=True)
            (out / "batch_0.jsonl").write_text(
                json.dumps(
                    {
                        "conversations": [
                            {"from": "human", "value": "Remember I use tabs."},
                            {"from": "gpt", "value": "Saved."},
                        ],
                        "tool_stats": {"memory": {"count": 1}},
                        "completed": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(be.subprocess, "run", fake_run)

        harness = BatchRunnerHarness(hermes_repo=batch_repo)
        scenario = BehavioralScenario(
            prompt="Remember I use tabs.",
            section_under_test="MEMORY_GUIDANCE",
            expected_behavior="Saves it.",
            category="memory_guidance",
        )
        transcripts = harness.run([scenario], "EVOLVED", "hase-run")

        assert seen["kwargs"]["cwd"] == str(batch_repo)
        assert "--ephemeral_system_prompt=EVOLVED" in seen["cmd"]
        assert len(transcripts) == 1
        assert transcripts[0].scenario_id == scenario.scenario_id
        assert transcripts[0].called("memory")

    def test_run_without_a_runner_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BatchRunnerHarness(hermes_repo=tmp_path).run([], "prompt", "run")

    def test_a_crashed_runner_with_no_output_names_its_exit_code(
        self, batch_repo, monkeypatch
    ):
        """A runner that dies before answering must not become a silent
        all-unmeasured run whose cause only surfaces later as an unpaired
        holdout."""
        monkeypatch.setattr(
            be.subprocess,
            "run",
            lambda cmd, **kw: types.SimpleNamespace(returncode=3, stdout="", stderr=""),
        )
        harness = BatchRunnerHarness(hermes_repo=batch_repo)
        scenario = BehavioralScenario(
            prompt="Remember I use tabs.",
            section_under_test="MEMORY_GUIDANCE",
            expected_behavior="Saves it.",
            category="memory_guidance",
        )
        with pytest.raises(HarnessFailure, match="exited 3"):
            harness.run([scenario], "EVOLVED", "hase-crash")

    def test_a_nonzero_exit_with_partial_output_keeps_the_transcripts(
        self, batch_repo, monkeypatch
    ):
        """Partial answers before the crash still count; the unanswered
        scenarios come back unmeasured through the normal path."""

        def fake_run(cmd, **kwargs):
            out = batch_repo / "data" / "hase-partial"
            out.mkdir(parents=True, exist_ok=True)
            (out / "batch_0.jsonl").write_text(
                json.dumps(
                    {
                        "conversations": [
                            {"from": "human", "value": "Remember I use tabs."},
                            {"from": "gpt", "value": "Saved."},
                        ],
                        "completed": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return types.SimpleNamespace(returncode=1, stdout="", stderr="")

        monkeypatch.setattr(be.subprocess, "run", fake_run)
        harness = BatchRunnerHarness(hermes_repo=batch_repo)
        scenario = BehavioralScenario(
            prompt="Remember I use tabs.",
            section_under_test="MEMORY_GUIDANCE",
            expected_behavior="Saves it.",
            category="memory_guidance",
        )
        transcripts = harness.run([scenario], "EVOLVED", "hase-partial")
        assert len(transcripts) == 1

    def test_parse_results_reads_response_and_tools(self, batch_repo):
        out = batch_repo / "data" / "run1"
        out.mkdir(parents=True)
        (out / "batch_0.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "conversations": [
                                {"from": "human", "value": "Do the thing"},
                                {
                                    "from": "function_call",
                                    "value": json.dumps({"name": "session_search"}),
                                },
                                {"from": "gpt", "value": "Found it."},
                            ],
                            "tool_stats": {"session_search": {"count": 1}},
                        }
                    ),
                    "",
                    "{not json}",
                    json.dumps({"failed": True, "conversations": []}),
                ]
            ),
            encoding="utf-8",
        )
        transcripts = BatchRunnerHarness(hermes_repo=batch_repo).parse_results("run1")
        assert len(transcripts) == 1
        assert transcripts[0].prompt == "Do the thing"
        assert transcripts[0].response == "Found it."
        assert transcripts[0].called("session_search")
        assert transcripts[0].tool_calls.count("session_search") == 1

    def test_parse_results_on_a_missing_directory_is_empty(self, batch_repo):
        assert BatchRunnerHarness(hermes_repo=batch_repo).parse_results("nope") == []

    def test_transcripts_are_matched_to_scenarios_by_prompt(self, batch_repo):
        out = batch_repo / "data" / "run2"
        out.mkdir(parents=True)
        scenarios = build_scenarios(categories=["session_search"])[:2]
        (out / "batch_0.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "conversations": [
                            {"from": "human", "value": s.prompt},
                            {"from": "gpt", "value": "ok"},
                        ]
                    }
                )
                for s in reversed(scenarios)
            ),
            encoding="utf-8",
        )
        harness = BatchRunnerHarness(hermes_repo=batch_repo)
        matched = be._match_transcripts(scenarios, harness.parse_results("run2"))
        assert {t.scenario_id for t in matched} == {s.scenario_id for s in scenarios}


class TestHarnessSelection:
    def test_prefers_batch_runner_when_present(self, batch_repo):
        harness, reason = select_harness(batch_repo, model="openai/gpt-4.1-mini")
        assert isinstance(harness, BatchRunnerHarness)
        assert "batch_runner" in reason

    def test_degrades_to_direct_without_a_checkout(self, tmp_path):
        harness, reason = select_harness(tmp_path, model="openai/gpt-4.1-mini")
        assert isinstance(harness, DirectPromptHarness)
        assert "direct dspy harness" in reason

    def test_degrades_when_no_repo_at_all(self):
        harness, _ = select_harness(None, model="openai/gpt-4.1-mini")
        assert isinstance(harness, DirectPromptHarness)


# ──────────────────────────────────────────────────────────────────────────
# The optimizable module and the metric
# ──────────────────────────────────────────────────────────────────────────


class TestSectionModule:
    def test_instructions_carry_the_section_between_markers(self):
        text = build_section_instructions("MEMORY_GUIDANCE", "Save durable facts.")
        assert SECTION_MARKER_OPEN in text and SECTION_MARKER_CLOSE in text
        assert extract_section_text(text) == "Save durable facts."

    def test_extraction_falls_back_when_markers_are_gone(self):
        assert extract_section_text("  a rewritten block  ") == "a rewritten block"

    def test_module_reads_its_section_back_out(self):
        module = SectionBehaviorModule("Save durable facts.", "MEMORY_GUIDANCE")
        assert module.section_text == "Save durable facts."

    def test_module_reflects_an_optimizer_rewrite(self):
        module = SectionBehaviorModule("Original text.", "MEMORY_GUIDANCE")
        # Through named_predictors(), the same path section_text reads the
        # live signature back through, so the test exercises the real access
        # route rather than the module's internal attribute layout.
        _, predictor = next(iter(module.named_predictors()))
        predictor.signature = predictor.signature.with_instructions(
            build_section_instructions("MEMORY_GUIDANCE", "Evolved text.")
        )
        assert module.section_text == "Evolved text."

    def test_forward_returns_response_and_tool_calls(self):
        module = SectionBehaviorModule(
            "Save durable facts.",
            "MEMORY_GUIDANCE",
            predictor=FakePredictor(response="Saved.", tool_calls="memory"),
        )
        prediction = module(user_message="Remember I use tabs.")
        assert prediction.response == "Saved."
        assert prediction.tool_calls == "memory"


class TestMetric:
    def test_metric_scores_a_prediction(self):
        scenario = BehavioralScenario(
            prompt="Remember I use tabs.",
            section_under_test="MEMORY_GUIDANCE",
            expected_behavior="Saves the preference to memory.",
            category="memory_guidance",
            expected_tools=("memory",),
        )
        example = be.scenarios_to_dspy_examples([scenario])[0]
        metric = make_behavioral_metric()
        good = metric(
            example,
            types.SimpleNamespace(response="Saved the preference to memory.", tool_calls="memory"),
        )
        bad = metric(example, types.SimpleNamespace(response="Sure.", tool_calls="none"))
        assert 0.0 <= bad < good <= 1.0

    def test_metric_accepts_gepa_extra_arguments(self):
        example = be.scenarios_to_dspy_examples([SEED_SCENARIOS[0]])[0]
        metric = make_behavioral_metric()
        score = metric(
            example,
            types.SimpleNamespace(response="Saved.", tool_calls="memory"),
            None,
            "predict",
            None,
        )
        assert isinstance(score, float)

    def test_metric_can_use_an_injected_judge(self):
        example = be.scenarios_to_dspy_examples([SEED_SCENARIOS[0]])[0]
        judge = BehavioralJudge(predictor=FakePredictor(score=0.77, feedback=""))
        metric = make_behavioral_metric(judge)
        assert metric(
            example, types.SimpleNamespace(response="x", tool_calls="none")
        ) == pytest.approx(0.77)

    def test_examples_keep_their_rubric_and_tools(self):
        scenario = SEED_SCENARIOS[0]
        example = be.scenarios_to_dspy_examples([scenario])[0]
        assert example.expected_behavior == scenario.expected_behavior
        assert list(example.expected_tools) == list(scenario.expected_tools)
        assert set(example.inputs().keys()) == {"user_message"}


class TestDirectHarnessAndEvaluate:
    def test_direct_harness_produces_one_transcript_per_scenario(self):
        scenarios = build_scenarios(categories=["memory_guidance"])[:3]
        harness = DirectPromptHarness(
            predictor=FakePredictor(response="Saved it.", tool_calls="memory")
        )
        transcripts = harness.run(scenarios, "SECTION TEXT")
        assert len(transcripts) == 3
        assert all(t.source == "direct" for t in transcripts)
        assert transcripts[0].tool_calls == ("memory",)

    def test_none_tool_calls_parse_to_nothing(self):
        harness = DirectPromptHarness(
            predictor=FakePredictor(response="Sure.", tool_calls="none")
        )
        transcripts = harness.run(build_scenarios(categories=["identity_tone"])[:1], "T")
        assert transcripts[0].tool_calls == ()

    def test_evaluate_end_to_end_offline(self):
        suite = BehavioralSuite.from_seeds(categories=["memory_guidance"])
        harness = DirectPromptHarness(
            predictor=FakePredictor(
                _dynamic=lambda kw: {
                    "response": "Saved the preference to memory as a durable fact.",
                    "tool_calls": "memory",
                }
            )
        )
        report = suite.evaluate("SECTION TEXT", harness, section_name="MEMORY_GUIDANCE")
        assert len(report) == len(suite)
        assert report.harness == "direct"
        assert 0.0 < report.mean_score <= 1.0

    def test_evaluate_with_no_scenarios_is_empty(self):
        suite = BehavioralSuite(scenarios=[])
        report = suite.evaluate("T", DirectPromptHarness(predictor=FakePredictor()))
        assert len(report) == 0 and report.harness == "none"


class TestTheArtifactRecordsWhetherItWasMeasured:
    """An unmeasured scenario and a measured failure are different claims.

    The distinction is the whole reason the field exists: a 0.0 that was never
    run, sitting opposite a real score in a paired comparison, invents a
    difference out of a timeout.
    """

    def outcome(self, **kw):
        base = dict(
            scenario_id="s1",
            category="restraint",
            section="MEMORY_GUIDANCE",
            score=0.0,
            passed=False,
        )
        base.update(kw)
        return BehavioralOutcome(**base)

    def test_an_unmeasured_outcome_serialises_as_unmeasured(self):
        assert self.outcome(measured=False).to_dict()["measured"] is False

    def test_a_measured_failure_serialises_as_measured(self):
        assert self.outcome(measured=True).to_dict()["measured"] is True

    def test_the_default_is_carried_explicitly_not_left_to_a_reload(self):
        assert "measured" in self.outcome().to_dict()

    def test_a_timeout_is_distinguishable_from_a_failure_in_the_artifact(self):
        timed_out = self.outcome(measured=False).to_dict()
        really_failed = self.outcome(measured=True).to_dict()
        assert timed_out["score"] == really_failed["score"]
        assert timed_out["passed"] == really_failed["passed"]
        assert timed_out != really_failed

    def test_it_still_carries_everything_it_used_to(self):
        blob = self.outcome().to_dict()
        assert set(blob) >= {
            "scenario_id",
            "category",
            "section",
            "score",
            "passed",
            "feedback",
            "judge",
        }

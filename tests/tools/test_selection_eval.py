"""Tests for the tool-selection evaluator.

Every test here is offline. The dataset builder is exercised with an injected
fake predictor, never a real dspy call: an existing PR in this repo is broken
because a test reached a predictor with no LM configured, and that must not
happen again.
"""

import json

import pytest

from evolution.core.config import EvolutionConfig
from evolution.tools.selection_eval import (
    CATEGORY_CLEAR,
    CATEGORY_CONFUSABLE,
    CATEGORY_NO_TOOL,
    NO_TOOL,
    PARAM_WEIGHT,
    TOOL_WEIGHT,
    SelectionReport,
    ToolSelectionDataset,
    ToolSelectionDatasetBuilder,
    ToolSelectionExample,
    ToolSelector,
    _split_counts,
    evaluate_selection,
    extract_bundle,
    gepa_selection_metric,
    normalise_tool_name,
    parameter_correctness,
    parse_params,
    parse_tool_catalog,
    render_catalog_for,
    render_tool_catalog,
    score_selection,
    split_examples,
    tool_selection_metric,
)
from evolution.tools.tool_catalog import ToolDescriptions, load_catalog


class FakePredictor:
    """Stands in for dspy.ChainOfThought. Records calls, returns canned output."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0) if self.payloads else "[]"
        if not isinstance(payload, str):
            payload = json.dumps(payload)
        return type("Result", (), {"cases": payload})()


def example(**kwargs) -> ToolSelectionExample:
    base = {"task": "do a thing", "correct_tool": "read_file", "category": CATEGORY_CLEAR}
    base.update(kwargs)
    return ToolSelectionExample(**base)


# ──────────────────────────────────────────────────────────────────────────


class TestNormaliseToolName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("read_file", "read_file"),
            ("  read_file  ", "read_file"),
            ("`read_file`", "read_file"),
            ('"read_file"', "read_file"),
            ("read_file(path='x')", "read_file"),
            ("tool: read_file", "read_file"),
            ("Function: read_file", "read_file"),
            ("none", NO_TOOL),
            ("None", NO_TOOL),
            ("no tool", NO_TOOL),
            ("N/A", NO_TOOL),
            ("", NO_TOOL),
            (None, NO_TOOL),
        ],
    )
    def test_variants(self, raw, expected):
        assert normalise_tool_name(raw) == expected


class TestParseParams:
    def test_dict_passthrough(self):
        assert parse_params({"path": "a.py"}) == {"path": "a.py"}

    def test_json_string(self):
        assert parse_params('{"path": "a.py"}') == {"path": "a.py"}

    def test_json_buried_in_prose(self):
        assert parse_params('Here you go: {"path": "a.py"} done') == {"path": "a.py"}

    def test_garbage_is_empty(self):
        assert parse_params("no arguments at all") == {}

    def test_json_array_is_not_params(self):
        assert parse_params("[1, 2]") == {}

    def test_none_and_blank(self):
        assert parse_params(None) == {}
        assert parse_params("   ") == {}


class TestParameterCorrectness:
    def test_no_expectation_is_free(self):
        assert parameter_correctness({}, {"anything": 1}) == 1.0

    def test_exact_match(self):
        assert parameter_correctness({"path": "a.py"}, {"path": "a.py"}) == 1.0

    def test_partial_match(self):
        score = parameter_correctness(
            {"path": "a.py", "limit": 50}, {"path": "a.py", "limit": 10}
        )
        assert score == 0.5

    def test_missing_key_scores_zero_for_that_key(self):
        assert parameter_correctness({"path": "a.py"}, {}) == 0.0

    def test_extra_arguments_are_not_penalised(self):
        # The schema is full of optional parameters with defaults.
        assert parameter_correctness({"path": "a.py"}, {"path": "a.py", "limit": 500}) == 1.0

    def test_types_are_compared_loosely(self):
        assert parameter_correctness({"limit": 50}, {"limit": "50"}) == 1.0
        assert parameter_correctness({"recurse": True}, {"recurse": "true"}) == 1.0

    def test_case_and_quotes_are_forgiven_on_strings(self):
        assert parameter_correctness({"pattern": "Import OS"}, {"pattern": '"import os"'}) == 1.0


class TestScoring:
    def test_right_tool_no_params_is_one(self):
        outcome = score_selection(example(), "read_file")
        assert outcome.tool_correct
        assert outcome.score == 1.0

    def test_wrong_tool_scores_zero_even_with_perfect_params(self):
        outcome = score_selection(
            example(correct_params={"path": "a.py"}), "terminal", {"path": "a.py"}
        )
        assert outcome.param_score == 0.0
        assert outcome.score == 0.0

    def test_right_tool_half_the_params(self):
        outcome = score_selection(
            example(correct_params={"path": "a.py", "limit": 50}),
            "read_file",
            {"path": "a.py", "limit": 9},
        )
        assert outcome.score == pytest.approx(TOOL_WEIGHT + PARAM_WEIGHT * 0.5)

    def test_no_tool_expected_and_none_called(self):
        outcome = score_selection(example(correct_tool=NO_TOOL, category=CATEGORY_NO_TOOL), "none")
        assert outcome.score == 1.0

    def test_no_tool_expected_but_a_tool_was_called(self):
        outcome = score_selection(
            example(correct_tool=NO_TOOL, category=CATEGORY_NO_TOOL), "read_file"
        )
        assert outcome.score == 0.0
        assert "needed no tool" in outcome.feedback()

    def test_feedback_names_the_tool_that_stole_the_selection(self):
        outcome = score_selection(example(), "search_files")
        feedback = outcome.feedback()
        assert "search_files" in feedback and "read_file" in feedback

    def test_feedback_for_a_missing_call(self):
        assert "Called nothing" in score_selection(example(), "none").feedback()

    def test_feedback_for_bad_arguments(self):
        outcome = score_selection(
            example(correct_params={"path": "a.py"}), "read_file", {"path": "b.py"}
        )
        assert "arguments were wrong" in outcome.feedback()

    def test_outcome_serialises(self):
        blob = score_selection(example(), "read_file").to_dict()
        assert blob["expected_tool"] == "read_file"
        assert blob["score"] == 1.0


class TestSelectionReport:
    def build(self):
        return SelectionReport(
            outcomes=[
                score_selection(example(task="a"), "read_file"),
                score_selection(example(task="b"), "terminal"),
                score_selection(
                    example(task="c", correct_params={"path": "x"}), "read_file", {"path": "y"}
                ),
                score_selection(
                    example(task="d", correct_tool=NO_TOOL, category=CATEGORY_NO_TOOL), "none"
                ),
            ]
        )

    def test_tool_accuracy(self):
        assert self.build().tool_accuracy == 0.75

    def test_param_accuracy_only_counts_correct_selections(self):
        # Three correct selections: params 1.0, 0.0, 1.0.
        assert self.build().param_accuracy == pytest.approx(2 / 3)

    def test_combined_score(self):
        report = self.build()
        assert report.score == pytest.approx(sum(o.score for o in report.outcomes) / 4)

    def test_by_category(self):
        by_category = self.build().by_category()
        assert by_category[CATEGORY_CLEAR] == pytest.approx(2 / 3)
        assert by_category[CATEGORY_NO_TOOL] == 1.0

    def test_failures(self):
        assert [o.example.task for o in self.build().failures()] == ["b"]

    def test_empty_report_is_zero_not_a_crash(self):
        empty = SelectionReport()
        assert (empty.n, empty.tool_accuracy, empty.param_accuracy, empty.score) == (0, 0.0, 0.0, 0.0)

    def test_serialises(self):
        blob = self.build().to_dict()
        assert blob["n"] == 4 and blob["tool_accuracy"] == 0.75


class TestEvaluateSelection:
    def test_tuple_predictions(self):
        examples = [example(task="a"), example(task="b", correct_tool="terminal")]
        report = evaluate_selection(examples, lambda ex: ("read_file", {}))
        assert report.tool_accuracy == 0.5

    def test_object_predictions(self):
        class Pred:
            tool_name = "read_file"
            parameters = '{"path": "a.py"}'

        report = evaluate_selection([example(correct_params={"path": "a.py"})], lambda ex: Pred())
        assert report.score == 1.0

    def test_a_raising_predictor_scores_as_a_miss(self):
        def boom(ex):
            raise RuntimeError("model exploded")

        report = evaluate_selection([example()], boom)
        assert report.n == 1 and report.tool_accuracy == 0.0

    def test_bare_string_prediction(self):
        assert evaluate_selection([example()], lambda ex: "read_file").tool_accuracy == 1.0


class TestSplitting:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, (0, 0, 0)),
            (1, (1, 0, 0)),
            (2, (1, 1, 0)),
            (3, (1, 1, 1)),
            (4, (2, 1, 1)),
            (8, (4, 2, 2)),
            (20, (10, 5, 5)),
        ],
    )
    def test_split_counts(self, n, expected):
        assert _split_counts(n, 0.5, 0.25) == expected

    def test_counts_always_sum_to_n(self):
        for n in range(1, 40):
            assert sum(_split_counts(n, 0.6, 0.2)) == n

    def test_stratifies_by_tool(self):
        examples = [
            example(task=f"{tool}-{i}", correct_tool=tool)
            for tool in ("read_file", "search_files", "terminal")
            for i in range(4)
        ]
        dataset = split_examples(examples, EvolutionConfig())
        for split in ("train", "val", "holdout"):
            tools = {e.correct_tool for e in dataset.split(split)}
            assert tools == {"read_file", "search_files", "terminal"}

    def test_respects_config_ratios(self):
        examples = [example(task=f"t{i}") for i in range(10)]
        dataset = split_examples(examples, EvolutionConfig(train_ratio=0.8, val_ratio=0.1))
        assert (len(dataset.train), len(dataset.val), len(dataset.holdout)) == (8, 1, 1)

    def test_is_deterministic_for_a_seed(self):
        examples = [example(task=f"t{i}") for i in range(12)]
        first = split_examples(examples, EvolutionConfig(), seed=7)
        second = split_examples(examples, EvolutionConfig(), seed=7)
        assert [e.task for e in first.train] == [e.task for e in second.train]


class TestDataset:
    def test_jsonl_round_trip(self, tmp_path):
        dataset = split_examples(
            [
                example(task="read a.py", correct_params={"path": "a.py"}, distractors=["terminal"]),
                example(task="grep", correct_tool="search_files", category=CATEGORY_CONFUSABLE),
                example(task="what is 2+2", correct_tool=NO_TOOL, category=CATEGORY_NO_TOOL),
                example(task="read b.py", correct_params={"path": "b.py"}),
            ],
            EvolutionConfig(),
        )
        dataset.save(tmp_path / "ds")
        loaded = ToolSelectionDataset.load(tmp_path / "ds")

        assert len(loaded) == len(dataset)
        by_task = {e.task: e for e in loaded.all_examples}
        assert by_task["read a.py"].correct_params == {"path": "a.py"}
        assert by_task["read a.py"].distractors == ["terminal"]
        assert by_task["what is 2+2"].expects_no_tool

    def test_load_from_a_missing_directory_is_empty(self, tmp_path):
        assert len(ToolSelectionDataset.load(tmp_path / "nope")) == 0

    def test_counts(self):
        dataset = ToolSelectionDataset(
            train=[example(), example(correct_tool=NO_TOOL, category=CATEGORY_NO_TOOL)],
            val=[example(correct_tool="terminal", category=CATEGORY_CONFUSABLE)],
        )
        assert dataset.category_counts()[CATEGORY_CLEAR] == 1
        assert dataset.tool_counts() == {"read_file": 1, NO_TOOL: 1, "terminal": 1}

    def test_unknown_split_raises(self):
        with pytest.raises(ValueError):
            ToolSelectionDataset().split("nope")

    def test_to_dspy_examples_marks_task_as_the_input(self):
        dataset = ToolSelectionDataset(train=[example()])
        dspy_example = dataset.to_dspy_examples("train")[0]
        assert dspy_example.task == "do a thing"
        assert set(dspy_example.inputs().keys()) == {"task"}

    def test_bridges_to_the_phase_one_dataset_type(self):
        dataset = ToolSelectionDataset(
            train=[example(correct_params={"path": "a.py"})],
            holdout=[example(correct_tool=NO_TOOL, category=CATEGORY_NO_TOOL)],
        )
        bridged = dataset.to_eval_dataset()
        assert "read_file" in bridged.train[0].expected_behavior
        assert "no tool" in bridged.holdout[0].expected_behavior

    def test_no_tool_examples_never_carry_params(self):
        stripped = ToolSelectionExample(
            task="t", correct_tool="none", correct_params={"path": "a.py"}
        )
        assert stripped.correct_params == {}


class TestRendering:
    def test_round_trip_is_exact(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        baseline = catalog.bundle()
        parsed = parse_tool_catalog(render_catalog_for(catalog), baseline)
        assert {k: v.to_dict() for k, v in parsed.items()} == {
            k: v.to_dict() for k, v in baseline.items()
        }

    def test_round_trip_survives_blank_lines_and_quotes(self):
        baseline = {
            "t": ToolDescriptions(
                "t",
                'First line.\n\n  Indented "quoted" line.\nLast.',
                {"p": "Multi\nline\n\nparam."},
            )
        }
        parsed = parse_tool_catalog(render_tool_catalog(baseline), baseline)
        assert parsed["t"].description == baseline["t"].description
        assert parsed["t"].params["p"] == baseline["t"].params["p"]

    def test_render_includes_signatures(self, hermes_repo):
        text = render_catalog_for(load_catalog(hermes_repo))
        assert "signature: read_file(path: string [required]" in text

    def test_render_lists_tools_in_a_stable_order(self, hermes_repo):
        text = render_catalog_for(load_catalog(hermes_repo))
        headers = [line for line in text.splitlines() if line.startswith("## ")]
        assert headers == sorted(headers)

    def test_parse_picks_up_an_edited_description(self):
        baseline = {"t": ToolDescriptions("t", "Old text.", {"p": "Old param."})}
        edited = render_tool_catalog(baseline).replace("Old text.", "New text.")
        parsed = parse_tool_catalog(edited, baseline)
        assert parsed["t"].description == "New text."
        assert parsed["t"].params["p"] == "Old param."

    def test_parse_ignores_invented_tools_and_params(self):
        baseline = {"t": ToolDescriptions("t", "Text.", {"p": "Param."})}
        text = render_tool_catalog(baseline) + "\n## ghost\ndescription:\n  Boo.\n"
        text = text.replace("- p:", "- q:")
        parsed = parse_tool_catalog(text, baseline)
        assert set(parsed) == {"t"}
        assert parsed["t"].params == {"p": "Param."}

    def test_unparseable_text_keeps_the_baseline(self):
        baseline = {"t": ToolDescriptions("t", "Text.", {"p": "Param."})}
        parsed = parse_tool_catalog("the optimizer wrote an essay instead", baseline)
        assert parsed["t"].to_dict() == baseline["t"].to_dict()

    def test_empty_text_keeps_the_baseline(self):
        baseline = {"t": ToolDescriptions("t", "Text.")}
        assert parse_tool_catalog("", baseline)["t"].description == "Text."

    def test_blank_replacement_is_refused(self):
        baseline = {"t": ToolDescriptions("t", "Text.")}
        text = render_tool_catalog(baseline).replace("  Text.", "   ")
        assert parse_tool_catalog(text, baseline)["t"].description == "Text."


class TestExtractBundle:
    def test_prefers_a_moved_bundle_attribute(self):
        baseline = {"t": ToolDescriptions("t", "Old.")}
        module = type("M", (), {"bundle": {"t": ToolDescriptions("t", "New.")}})()
        assert extract_bundle(module, baseline)["t"].description == "New."

    def test_falls_back_to_parsing_the_instructions(self):
        baseline = {"t": ToolDescriptions("t", "Old.")}
        evolved_text = render_tool_catalog(baseline).replace("Old.", "Evolved.")

        class Module:
            bundle = {"t": ToolDescriptions("t", "Old.")}

            def named_predictors(self):
                signature = type("S", (), {"instructions": evolved_text})()
                return [("predictor", type("P", (), {"signature": signature})())]

        assert extract_bundle(Module(), baseline)["t"].description == "Evolved."

    def test_falls_back_to_the_baseline(self):
        baseline = {"t": ToolDescriptions("t", "Old.")}
        assert extract_bundle(object(), baseline)["t"].description == "Old."

    def test_result_is_a_copy(self):
        baseline = {"t": ToolDescriptions("t", "Old.", {"p": "x"})}
        extracted = extract_bundle(object(), baseline)
        extracted["t"].params["p"] = "changed"
        assert baseline["t"].params["p"] == "x"


class TestSelectorModule:
    def test_builds_offline_and_carries_the_descriptions(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        selector = ToolSelector(catalog.bundle(), {e.tool_name: e.signature() for e in catalog})
        assert "Read a text file" in selector.rendered
        assert "Read a text file" in selector.predictor.predict.signature.instructions

    def test_holds_its_own_copy_of_the_bundle(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        bundle = catalog.bundle()
        selector = ToolSelector(bundle)
        bundle["read_file"].description = "mutated after construction"
        assert selector.bundle["read_file"].description != "mutated after construction"


class TestMetrics:
    def test_dspy_metric_scores_a_prediction(self):
        gold = example(correct_params={"path": "a.py"}).to_dspy_example()
        prediction = type("P", (), {"tool_name": "read_file", "parameters": '{"path": "a.py"}'})()
        assert tool_selection_metric(gold, prediction) == 1.0

    def test_dspy_metric_penalises_the_wrong_tool(self):
        gold = example().to_dspy_example()
        prediction = type("P", (), {"tool_name": "terminal", "parameters": "{}"})()
        assert tool_selection_metric(gold, prediction) == 0.0

    def test_gepa_metric_returns_a_float_for_plain_evaluation(self):
        gold = example().to_dspy_example()
        prediction = type("P", (), {"tool_name": "read_file", "parameters": "{}"})()
        assert gepa_selection_metric(gold, prediction) == 1.0

    def test_gepa_metric_returns_feedback_when_reflecting(self):
        gold = example().to_dspy_example()
        prediction = type("P", (), {"tool_name": "search_files", "parameters": "{}"})()
        result = gepa_selection_metric(gold, prediction, None, "predictor", None)
        assert result.score == 0.0
        assert "search_files" in result.feedback

    def test_gepa_metric_signature_matches_what_gepa_binds(self):
        import inspect

        inspect.signature(gepa_selection_metric).bind(None, None, None, None, None)


class TestDatasetBuilder:
    def payload(self, tool, task, params=None, distractors=None):
        return {
            "task": task,
            "correct_tool": tool,
            "correct_params": params or {},
            "distractors": distractors or [],
            "difficulty": "medium",
        }

    def test_generates_all_three_example_classes(self, hermes_repo):
        catalog = load_catalog(hermes_repo).select(tools=["read_file", "search_files"])
        predictor = FakePredictor(
            [
                [self.payload("read_file", "read lines 50-100 of config.py", {"path": "config.py"})],
                [self.payload("search_files", "find every TODO", {"pattern": "TODO"})],
                [self.payload("search_files", "find files containing import os", {"pattern": "import os"})],
                [self.payload("none", "what does idempotent mean")],
            ]
        )
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        dataset = builder.generate(per_tool=1, num_confusable=1, num_no_tool=1)

        categories = {e.task: e.category for e in dataset.all_examples}
        assert set(categories.values()) == {
            CATEGORY_CLEAR,
            CATEGORY_CONFUSABLE,
            CATEGORY_NO_TOOL,
        }
        assert len(dataset) == 4

    def test_asks_once_per_tool_plus_two(self, hermes_repo):
        catalog = load_catalog(hermes_repo).select(toolset="file")
        predictor = FakePredictor([])
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        builder.generate(per_tool=2, num_confusable=2, num_no_tool=2)
        kinds = [call["kind"] for call in predictor.calls]
        assert kinds == [
            CATEGORY_CLEAR,
            CATEGORY_CLEAR,
            CATEGORY_CLEAR,
            CATEGORY_CONFUSABLE,
            CATEGORY_NO_TOOL,
        ]
        assert [c["focus_tool"] for c in predictor.calls][:3] == [
            "read_file",
            "search_files",
            "write_file",
        ]

    def test_zero_counts_make_no_calls(self, hermes_repo):
        catalog = load_catalog(hermes_repo).select(tools=["read_file"])
        predictor = FakePredictor([])
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        builder.generate(per_tool=0, num_confusable=0, num_no_tool=0)
        assert predictor.calls == []

    def test_hallucinated_tool_is_rejected(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        predictor = FakePredictor([[self.payload("read_the_file", "read a.py")]])
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        dataset = builder.generate(per_tool=0, num_confusable=1, num_no_tool=0)
        assert len(dataset) == 0
        assert "unknown tool" in builder.rejected[0][1]

    def test_arguments_outside_the_frozen_schema_are_stripped(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        predictor = FakePredictor(
            [[self.payload("read_file", "read a.py", {"path": "a.py", "encoding": "utf-8"})]]
        )
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        dataset = builder.generate(per_tool=0, num_confusable=1, num_no_tool=0)
        assert dataset.all_examples[0].correct_params == {"path": "a.py"}
        assert "unknown argument" in builder.rejected[0][1]

    def test_distractors_are_filtered_to_real_tools(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        predictor = FakePredictor(
            [[self.payload("read_file", "read a.py", distractors=["terminal", "ghost", "read_file"])]]
        )
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        dataset = builder.generate(per_tool=0, num_confusable=1, num_no_tool=0)
        assert dataset.all_examples[0].distractors == ["terminal"]

    def test_no_tool_batch_forces_the_sentinel(self, hermes_repo):
        catalog = load_catalog(hermes_repo)
        predictor = FakePredictor(
            [[self.payload("read_file", "explain what a mutex is", {"path": "x"})]]
        )
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        dataset = builder.generate(per_tool=0, num_confusable=0, num_no_tool=1)
        only = dataset.all_examples[0]
        assert only.expects_no_tool and only.correct_params == {}

    def test_a_no_tool_answer_in_a_clear_batch_is_rejected(self, hermes_repo):
        catalog = load_catalog(hermes_repo).select(tools=["read_file"])
        predictor = FakePredictor([[self.payload("none", "read a.py")]])
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        dataset = builder.generate(per_tool=1, num_confusable=0, num_no_tool=0)
        assert len(dataset) == 0
        assert "no-tool case" in builder.rejected[0][1]

    def test_duplicate_tasks_are_dropped(self, hermes_repo):
        catalog = load_catalog(hermes_repo).select(tools=["read_file"])
        predictor = FakePredictor(
            [
                [
                    self.payload("read_file", "Read  A.py"),
                    self.payload("read_file", "read a.py"),
                ]
            ]
        )
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        dataset = builder.generate(per_tool=1, num_confusable=0, num_no_tool=0)
        assert len(dataset) == 1
        assert builder.rejected[0][1] == "duplicate task"

    def test_json_wrapped_in_prose_is_recovered(self, hermes_repo):
        catalog = load_catalog(hermes_repo).select(tools=["read_file"])
        predictor = FakePredictor(
            ['Sure! Here are the cases:\n[{"task": "read a.py", "correct_tool": "read_file"}]\n']
        )
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        assert len(builder.generate(per_tool=1, num_confusable=0, num_no_tool=0)) == 1

    def test_non_json_output_is_recorded_not_raised(self, hermes_repo):
        catalog = load_catalog(hermes_repo).select(tools=["read_file"])
        predictor = FakePredictor(["I could not do that."])
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        assert len(builder.generate(per_tool=1, num_confusable=0, num_no_tool=0)) == 0
        assert builder.rejected[0][1] == "output was not JSON"

    def test_tasks_without_text_are_skipped(self, hermes_repo):
        catalog = load_catalog(hermes_repo).select(tools=["read_file"])
        predictor = FakePredictor([[{"task": "  ", "correct_tool": "read_file"}]])
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        assert len(builder.generate(per_tool=1, num_confusable=0, num_no_tool=0)) == 0

    def test_the_generator_sees_the_whole_catalogue(self, hermes_repo):
        catalog = load_catalog(hermes_repo).select(tools=["read_file"])
        predictor = FakePredictor([])
        builder = ToolSelectionDatasetBuilder(
            catalog=catalog, config=EvolutionConfig(), generator=predictor
        )
        builder.generate(per_tool=1, num_confusable=0, num_no_tool=0)
        assert "signature: read_file(" in predictor.calls[0]["tool_catalog"]

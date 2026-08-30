"""Tests for the Phase 2 entry point.

The optimizer itself needs a model, so what is tested here is everything around
it: candidate hygiene, the 500 / 200 char budgets at their exact boundaries, and
the CLI paths that must work without a network - above all --dry-run.
"""

import dspy
import pytest
from click.testing import CliRunner

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.tools.evolve_tool_descriptions import (
    build_accuracy_checker,
    enforce_constraints,
    evolve_tool_descriptions,
    freeze_unselected,
    main,
)
from evolution.tools.selection_eval import (
    NO_TOOL,
    ToolSelectionExample,
    ToolSelector,
    split_examples,
)
from evolution.tools.tool_catalog import ToolDescriptions, load_catalog

# Isolates the size budget from the 20% growth limit, which would otherwise fire
# first on any large rewrite of a short baseline.
NO_GROWTH_LIMIT = EvolutionConfig(max_prompt_growth=100.0)

# Boundary cases for the size budget are written as filler ("x" * 480 against
# "y" * 500) because only the length is under test. Filler shares no vocabulary
# with filler, so semantic_preservation correctly reads it as a total topic
# change; disabling it here keeps those tests measuring the one thing they name.
# Semantic preservation has its own tests in tests/core/.
BUDGETS_ONLY = EvolutionConfig(max_prompt_growth=100.0, min_semantic_similarity=0.0)


def bundle(**tools) -> dict[str, ToolDescriptions]:
    return {
        name: ToolDescriptions(name, spec[0], dict(spec[1]) if len(spec) > 1 else {})
        for name, spec in tools.items()
    }


def runner_invoke(args, env=None):
    merged = {"COLUMNS": "200"}
    merged.update(env or {})
    return CliRunner().invoke(main, args, env=merged)


class TestFreezeUnselected:
    def test_reverts_a_tool_the_run_did_not_ask_for(self):
        baseline = bundle(read_file=("Old read.",), terminal=("Old terminal.",))
        candidate = bundle(read_file=("New read.",), terminal=("New terminal.",))
        merged = freeze_unselected(candidate, baseline, ["read_file"])
        assert merged["read_file"].description == "New read."
        assert merged["terminal"].description == "Old terminal."

    def test_keeps_every_selected_change(self):
        baseline = bundle(read_file=("Old.", {"path": "Old path."}))
        candidate = bundle(read_file=("New.", {"path": "New path."}))
        merged = freeze_unselected(candidate, baseline, ["read_file"])
        assert merged["read_file"].params["path"] == "New path."

    def test_a_tool_missing_from_the_candidate_keeps_its_baseline(self):
        baseline = bundle(read_file=("Old.",), terminal=("Keep me.",))
        merged = freeze_unselected(bundle(read_file=("New.",)), baseline, ["read_file", "terminal"])
        assert merged["terminal"].description == "Keep me."

    def test_invented_tools_never_enter_the_result(self):
        baseline = bundle(read_file=("Old.",))
        candidate = bundle(read_file=("New.",), ghost=("Boo.",))
        assert set(freeze_unselected(candidate, baseline, ["read_file", "ghost"])) == {"read_file"}

    def test_result_is_a_copy_of_the_baseline(self):
        baseline = bundle(read_file=("Old.", {"path": "p"}))
        merged = freeze_unselected({}, baseline, [])
        merged["read_file"].params["path"] = "mutated"
        assert baseline["read_file"].params["path"] == "p"


class TestConstraintBudgets:
    def validator(self, config=BUDGETS_ONLY):
        return ConstraintValidator(config)

    def test_a_short_rewrite_is_accepted(self):
        baseline = bundle(read_file=("Read a file from disk with pagination.",))
        candidate = bundle(read_file=("Read a text file. Use instead of cat.",))
        kept, outcomes = enforce_constraints(candidate, baseline, self.validator())
        assert kept["read_file"].description == "Read a text file. Use instead of cat."
        assert outcomes[0].passed and not outcomes[0].reverted

    def test_exactly_500_chars_is_accepted(self):
        baseline = bundle(read_file=("x" * 480,))
        candidate = bundle(read_file=("y" * 500,))
        kept, outcomes = enforce_constraints(candidate, baseline, self.validator())
        assert kept["read_file"].description == "y" * 500
        assert outcomes[0].passed

    def test_501_chars_is_reverted(self):
        baseline = bundle(read_file=("x" * 480,))
        candidate = bundle(read_file=("y" * 501,))
        kept, outcomes = enforce_constraints(candidate, baseline, self.validator())
        assert kept["read_file"].description == "x" * 480
        assert outcomes[0].reverted
        assert any("501/500" in message for message in outcomes[0].messages)

    def test_exactly_200_chars_is_accepted_for_a_parameter(self):
        baseline = bundle(read_file=("Read.", {"path": "x" * 190}))
        candidate = bundle(read_file=("Read.", {"path": "y" * 200}))
        kept, outcomes = enforce_constraints(candidate, baseline, self.validator())
        assert kept["read_file"].params["path"] == "y" * 200
        assert outcomes[0].kind == "param_description" and outcomes[0].passed

    def test_201_chars_is_reverted_for_a_parameter(self):
        baseline = bundle(read_file=("Read.", {"path": "x" * 190}))
        candidate = bundle(read_file=("Read.", {"path": "y" * 201}))
        kept, outcomes = enforce_constraints(candidate, baseline, self.validator())
        assert kept["read_file"].params["path"] == "x" * 190
        assert outcomes[0].reverted

    def test_growth_beyond_the_config_limit_is_reverted(self):
        baseline = bundle(read_file=("Read a file.",))
        candidate = bundle(read_file=("Read a file, at length, with commentary.",))
        kept, outcomes = enforce_constraints(candidate, baseline, ConstraintValidator(EvolutionConfig()))
        assert kept["read_file"].description == "Read a file."
        assert any("Growth exceeded" in message for message in outcomes[0].messages)

    def test_an_empty_description_is_reverted(self):
        baseline = bundle(read_file=("Read a file.",))
        kept, outcomes = enforce_constraints(bundle(read_file=("   ",)), baseline, self.validator())
        assert kept["read_file"].description == "Read a file."
        assert outcomes[0].reverted

    def test_a_baseline_that_is_already_over_budget_is_left_alone(self):
        """The real read_file is 539 chars. Unchanged text is not re-litigated."""
        baseline = bundle(read_file=("x" * 539,))
        candidate = bundle(read_file=("x" * 539,))
        kept, outcomes = enforce_constraints(candidate, baseline, self.validator())
        assert kept["read_file"].description == "x" * 539
        assert outcomes == []

    def test_one_bad_rewrite_does_not_discard_the_good_ones(self):
        baseline = bundle(read_file=("x" * 480,), terminal=("Run a command.",))
        candidate = bundle(read_file=("y" * 900,), terminal=("Run a shell command.",))
        kept, outcomes = enforce_constraints(candidate, baseline, self.validator())
        assert kept["read_file"].description == "x" * 480
        assert kept["terminal"].description == "Run a shell command."
        assert {o.target: o.reverted for o in outcomes} == {"read_file": True, "terminal": False}

    def test_unselected_tools_are_frozen_and_unvalidated(self):
        baseline = bundle(read_file=("Read.",), terminal=("Run.",))
        candidate = bundle(read_file=("Read a file.",), terminal=("y" * 900,))
        kept, outcomes = enforce_constraints(
            candidate, baseline, self.validator(), allowed=["read_file"]
        )
        assert kept["terminal"].description == "Run."
        assert [o.target for o in outcomes] == ["read_file"]

    def test_invented_parameters_are_dropped(self):
        baseline = bundle(read_file=("Read.", {"path": "Where."}))
        candidate = bundle(read_file=("Read.", {"path": "Where from.", "encoding": "utf-8"}))
        kept, _ = enforce_constraints(candidate, baseline, self.validator())
        assert set(kept["read_file"].params) == {"path"}
        assert kept["read_file"].params["path"] == "Where from."

    def test_a_dropped_parameter_falls_back_to_baseline(self):
        baseline = bundle(read_file=("Read.", {"path": "Where.", "limit": "How many."}))
        candidate = bundle(read_file=("Read.", {"path": "Where from."}))
        kept, _ = enforce_constraints(candidate, baseline, self.validator())
        assert kept["read_file"].params["limit"] == "How many."

    def test_outcomes_serialise(self):
        baseline = bundle(read_file=("Read.",))
        _, outcomes = enforce_constraints(bundle(read_file=("Read a file.",)), baseline, self.validator())
        blob = outcomes[0].to_dict()
        assert blob["target"] == "read_file" and blob["kind"] == "tool_description"

    def test_nothing_changed_means_nothing_validated(self):
        baseline = bundle(read_file=("Read.", {"path": "Where."}))
        kept, outcomes = enforce_constraints(
            {k: v.copy() for k, v in baseline.items()}, baseline, self.validator()
        )
        assert outcomes == []
        assert kept["read_file"].to_dict() == baseline["read_file"].to_dict()


class TestDryRunCli:
    def test_dry_run_succeeds_without_a_model(self, hermes_repo):
        result = runner_invoke(["--hermes-repo", str(hermes_repo), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "DRY RUN" in result.output

    def test_dry_run_lists_the_catalogue(self, hermes_repo):
        result = runner_invoke(["--hermes-repo", str(hermes_repo), "--dry-run"])
        for tool in ("read_file", "search_files", "terminal", "vision_analyze"):
            assert tool in result.output

    def test_dry_run_reports_the_over_budget_descriptions(self, hermes_repo):
        result = runner_invoke(["--hermes-repo", str(hermes_repo), "--dry-run"])
        assert "over budget" in result.output
        assert "cross_profile" in result.output.replace("\n", "")

    def test_dry_run_touches_nothing_on_disk(self, hermes_repo):
        before = {
            path.name: path.read_text() for path in (hermes_repo / "tools").glob("*.py")
        }
        runner_invoke(["--hermes-repo", str(hermes_repo), "--dry-run"])
        after = {path.name: path.read_text() for path in (hermes_repo / "tools").glob("*.py")}
        assert before == after

    def test_dry_run_narrows_to_a_toolset(self, hermes_repo):
        result = runner_invoke(
            ["--hermes-repo", str(hermes_repo), "--toolset", "file", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "Evolving 3" in result.output

    def test_dry_run_narrows_to_named_tools(self, hermes_repo):
        result = runner_invoke(
            [
                "--hermes-repo",
                str(hermes_repo),
                "--tool",
                "read_file",
                "--tool",
                "terminal",
                "--dry-run",
            ]
        )
        assert result.exit_code == 0
        assert "Evolving 2" in result.output

    def test_dry_run_says_whether_it_would_write(self, hermes_repo):
        without = runner_invoke(["--hermes-repo", str(hermes_repo), "--dry-run"])
        with_write = runner_invoke(["--hermes-repo", str(hermes_repo), "--dry-run", "--write"])
        assert "leave the repo untouched" in without.output
        assert "write results back" in with_write.output

    def test_no_write_is_the_default(self, hermes_repo):
        result = runner_invoke(["--hermes-repo", str(hermes_repo), "--dry-run", "--no-write"])
        assert "leave the repo untouched" in result.output

    def test_unknown_tool_exits_nonzero(self, hermes_repo):
        result = runner_invoke(["--hermes-repo", str(hermes_repo), "--tool", "nope", "--dry-run"])
        assert result.exit_code == 1
        assert "nope" in result.output

    def test_unmatched_toolset_exits_nonzero(self, hermes_repo):
        result = runner_invoke(
            ["--hermes-repo", str(hermes_repo), "--toolset", "ghost", "--dry-run"]
        )
        assert result.exit_code == 1
        assert "No tools matched" in result.output

    def test_a_repo_with_no_tools_exits_nonzero(self, empty_repo):
        result = runner_invoke(["--hermes-repo", str(empty_repo), "--dry-run"])
        assert result.exit_code == 1
        assert "No literal tool schemas" in result.output

    def test_iterations_are_echoed(self, hermes_repo):
        result = runner_invoke(
            ["--hermes-repo", str(hermes_repo), "--iterations", "3", "--dry-run"]
        )
        assert "3 iterations" in result.output

    def test_help_lists_every_documented_option(self):
        result = CliRunner().invoke(main, ["--help"], env={"COLUMNS": "200"})
        for option in (
            "--tool",
            "--toolset",
            "--iterations",
            "--dataset-path",
            "--hermes-repo",
            "--optimizer-model",
            "--eval-model",
            "--run-tests",
            "--strict-gates",
            "--dry-run",
            "--write",
            "--no-write",
        ):
            assert option in result.output


class TestCatalogueIntegration:
    def test_the_selected_subset_still_reads_the_whole_catalogue(self, hermes_repo):
        """Narrowing what may be rewritten must not narrow what is evaluated."""
        catalog = load_catalog(hermes_repo)
        selected = catalog.select(tools=["read_file"])
        assert len(catalog) == 5 and len(selected) == 1
        assert set(catalog.bundle()) > set(selected.bundle())


# ──────────────────────────────────────────────────────────────────────────
# Whole-pipeline runs with a stubbed optimizer. No LM is ever reached: the
# selector's forward pass is replaced with a deterministic keyword match and
# the optimizer with a class that returns a pre-baked candidate.
# ──────────────────────────────────────────────────────────────────────────

GREEDY_MARKER = "grabs everything"


def keyword_forward(self, task: str):
    """Pick the tool whose description best matches the task, deterministically.

    A description containing GREEDY_MARKER wins every task, which is how a test
    stages one tool stealing another's selections.
    """
    for name in sorted(self.bundle):
        if GREEDY_MARKER in self.bundle[name].description.lower():
            return dspy.Prediction(tool_name=name, parameters={})
    words = set(task.lower().split())
    best, best_hits = NO_TOOL, 0
    for name in sorted(self.bundle):
        text = self.bundle[name].description.lower().replace(".", "")
        hits = len(words & set(text.split()))
        if hits > best_hits:
            best, best_hits = name, hits
    return dspy.Prediction(tool_name=best, parameters={})


def build_dataset(path):
    examples = []
    for i in range(6):
        examples.append(
            ToolSelectionExample(
                task=f"read pagination line numbers of f{i}.py", correct_tool="read_file"
            )
        )
        examples.append(
            ToolSelectionExample(
                task=f"search regular expression for TODO {i}", correct_tool="search_files"
            )
        )
        examples.append(
            ToolSelectionExample(task=f"run shell command number {i}", correct_tool="terminal")
        )
        examples.append(
            ToolSelectionExample(
                task=f"what is {i} plus two", correct_tool=NO_TOOL, category="no_tool"
            )
        )
    split_examples(examples).save(path)
    return path


def stub_optimizer(mutate):
    """An optimizer that returns a selector carrying *mutate*'s edits."""

    class Stub:
        def __init__(self, *args, **kwargs):
            pass

        def compile(self, student, **kwargs):
            evolved = ToolSelector(student.bundle, student.signatures)
            mutate(evolved.bundle)
            return evolved

    return Stub


class Unavailable:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("GEPA unavailable in this build")


@pytest.fixture
def stubbed(monkeypatch):
    """Neutralise every route to a language model."""
    monkeypatch.setattr(ToolSelector, "forward", keyword_forward)
    monkeypatch.setattr(dspy, "LM", lambda *a, **kw: object())
    monkeypatch.setattr(dspy, "configure", lambda **kw: None)
    return monkeypatch


def tidy(bundle):
    bundle["read_file"].description = "Read a text file with line numbers and pagination."
    bundle["terminal"].description = "Run a shell command in a persistent session."
    bundle["read_file"].params["limit"] = "Maximum lines to read."


def greedy(bundle):
    # On-topic on purpose: an off-topic land-grab is rejected earlier by the
    # semantic_preservation constraint, which would leave the cross-tool guard
    # untested. The candidate that guard exists for still sounds like itself.
    bundle["search_files"].description = (
        "Search file contents or filenames with a regular expression, "
        f"and {GREEDY_MARKER} else, always."
    )


class TestFullRun:
    def run(self, repo, tmp_path, **kwargs):
        return evolve_tool_descriptions(
            hermes_repo=str(repo),
            dataset_path=str(build_dataset(tmp_path / "ds")),
            iterations=2,
            output_root=tmp_path / "out",
            **kwargs,
        )

    def test_gepa_failure_falls_back_to_miprov2(self, hermes_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", Unavailable)
        stubbed.setattr(dspy, "MIPROv2", stub_optimizer(tidy))
        metrics = self.run(hermes_repo, tmp_path, write=True)
        assert metrics["optimizer"] == "MIPROv2"

    def test_a_clean_candidate_is_written_and_shrinks_the_schema(
        self, hermes_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(tidy))
        metrics = self.run(hermes_repo, tmp_path, write=True)

        assert metrics["optimizer"] == "GEPA"
        assert metrics["cross_tool_accepted"] and metrics["gates_passed"]
        assert metrics["written"] is True
        assert metrics["candidate_chars"] < metrics["baseline_chars"]
        assert load_catalog(hermes_repo).require("terminal").description == (
            "Run a shell command in a persistent session."
        )

    def test_the_over_budget_terminal_description_comes_back_under_budget(
        self, hermes_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(tidy))
        self.run(hermes_repo, tmp_path, write=True)
        findings = [f for f in load_catalog(hermes_repo).budget_findings() if f.kind == "tool"]
        assert findings == []

    def test_no_write_leaves_the_repo_untouched(self, hermes_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(tidy))
        before = {p.name: p.read_text() for p in (hermes_repo / "tools").glob("*.py")}
        metrics = self.run(hermes_repo, tmp_path, write=False)

        assert metrics["descriptions_changed"] == 3
        assert metrics["written"] is False
        assert {p.name: p.read_text() for p in (hermes_repo / "tools").glob("*.py")} == before

    def test_only_the_named_tools_are_rewritten(self, hermes_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(tidy))
        metrics = self.run(hermes_repo, tmp_path, write=True, tools=["terminal"])

        assert metrics["tools_evolved"] == ["terminal"]
        reloaded = load_catalog(hermes_repo)
        assert reloaded.require("terminal").description.startswith("Run a shell command in a")
        assert reloaded.require("read_file").description.startswith("Read a text file with line")
        assert reloaded.require("read_file").param_descriptions["limit"] == (
            "Maximum number of lines to read"
        )

    def test_artifacts_are_saved(self, hermes_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(tidy))
        self.run(hermes_repo, tmp_path, write=True)
        run_dirs = list((tmp_path / "out" / "tools").iterdir())
        assert len(run_dirs) == 1
        written = {p.name for p in run_dirs[0].iterdir()}
        assert written == {
            "baseline_descriptions.json",
            "evolved_descriptions.json",
            "cross_tool_report.json",
            "gates.json",
            "changes.json",
            "metrics.json",
        }

    def test_a_thieving_candidate_is_rejected_and_not_written(
        self, hermes_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(greedy))
        before = (hermes_repo / "tools" / "file_tools.py").read_text()
        metrics = self.run(hermes_repo, tmp_path, write=True)

        assert metrics["cross_tool_accepted"] is False
        assert metrics["gates_passed"] is False
        assert metrics["written"] is False
        assert (hermes_repo / "tools" / "file_tools.py").read_text() == before

    def test_the_rejection_records_which_tools_lost_selections(
        self, hermes_repo, tmp_path, stubbed
    ):
        import json

        stubbed.setattr(dspy, "GEPA", stub_optimizer(greedy))
        self.run(hermes_repo, tmp_path, write=True)
        run_dir = next((tmp_path / "out" / "tools").iterdir())
        verdict = json.loads((run_dir / "cross_tool_report.json").read_text())["verdict"]

        assert verdict["accepted"] is False
        regressed = {r["tool"]: r["stolen_by"] for r in verdict["regressions"]}
        assert "read_file" in regressed
        assert regressed["read_file"].get("search_files")

    def test_strict_gates_block_on_the_missing_benchmark(self, hermes_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(tidy))
        metrics = self.run(hermes_repo, tmp_path, write=True, strict_gates=True)
        assert metrics["gates_passed"] is False
        assert metrics["written"] is False


# ──────────────────────────────────────────────────────────────────────────
# PLAN.md's last Phase 2 constraint: descriptions must stay factually true.
# ──────────────────────────────────────────────────────────────────────────


class TestFactualAccuracyConstraint:
    def checker(self, hermes_repo):
        # entailment=False keeps this offline and deterministic; the LLM tier
        # has its own tests in test_accuracy.py.
        return build_accuracy_checker(load_catalog(hermes_repo), entailment=False)

    def enforce(self, hermes_repo, candidate, baseline):
        return enforce_constraints(
            candidate,
            baseline,
            ConstraintValidator(NO_GROWTH_LIMIT),
            accuracy=self.checker(hermes_repo),
        )

    def test_a_rewrite_that_invents_a_parameter_is_reverted(self, hermes_repo):
        baseline = bundle(read_file=("Read a text file.",))
        candidate = bundle(read_file=("Read a text file. Pass `recursive` to descend.",))
        kept, outcomes = self.enforce(hermes_repo, candidate, baseline)
        assert kept["read_file"].description == "Read a text file."
        assert outcomes[0].reverted and not outcomes[0].passed

    def test_the_messages_say_what_was_wrong(self, hermes_repo):
        baseline = bundle(read_file=("Read a text file.",))
        candidate = bundle(read_file=("Read a text file. Pass `recursive` to descend.",))
        _, outcomes = self.enforce(hermes_repo, candidate, baseline)
        assert any(
            m.startswith("factual_accuracy:") and "recursive" in m
            for m in outcomes[0].messages
        )

    def test_the_budget_messages_are_still_there(self, hermes_repo):
        baseline = bundle(read_file=("Read a text file.",))
        candidate = bundle(read_file=("Read a text file. Pass `recursive` to descend.",))
        _, outcomes = self.enforce(hermes_repo, candidate, baseline)
        assert any(m.startswith("size_limit:") for m in outcomes[0].messages)

    def test_an_accurate_rewrite_survives(self, hermes_repo):
        baseline = bundle(read_file=("Read a text file.",))
        candidate = bundle(read_file=("Read a text file. Use offset and limit for big files.",))
        kept, outcomes = self.enforce(hermes_repo, candidate, baseline)
        assert kept["read_file"].description.endswith("for big files.")
        assert outcomes[0].passed and not outcomes[0].reverted

    def test_a_value_outside_an_enum_is_reverted(self, hermes_repo):
        baseline = bundle(search_files=("Search file contents.",))
        candidate = bundle(search_files=("Search with target='both' for names and text.",))
        kept, _ = self.enforce(hermes_repo, candidate, baseline)
        assert kept["search_files"].description == "Search file contents."

    def test_an_inaccurate_parameter_description_is_reverted(self, hermes_repo):
        baseline = bundle(read_file=("Read a text file.", {"limit": "How many lines."}))
        candidate = bundle(read_file=("Read a text file.", {"limit": "How many; see `stride`."}))
        kept, outcomes = self.enforce(hermes_repo, candidate, baseline)
        assert kept["read_file"].params["limit"] == "How many lines."
        assert outcomes[0].target == "read_file.limit" and outcomes[0].reverted

    def test_one_inaccurate_rewrite_does_not_cost_the_others(self, hermes_repo):
        baseline = bundle(read_file=("Read a text file.",), terminal=("Run a command.",))
        candidate = bundle(
            read_file=("Read a file. Pass `recursive`.",),
            terminal=("Run a shell command in a persistent session.",),
        )
        kept, _ = self.enforce(hermes_repo, candidate, baseline)
        assert kept["read_file"].description == "Read a text file."
        assert kept["terminal"].description == "Run a shell command in a persistent session."

    def test_without_a_checker_the_old_behaviour_is_unchanged(self, hermes_repo):
        baseline = bundle(read_file=("Read a text file.",))
        candidate = bundle(read_file=("Read a text file. Pass `recursive` to descend.",))
        kept, outcomes = enforce_constraints(
            candidate, baseline, ConstraintValidator(NO_GROWTH_LIMIT)
        )
        assert kept["read_file"].description.endswith("to descend.")
        assert outcomes[0].passed

    def test_an_unselected_tool_is_never_factually_checked(self, hermes_repo):
        baseline = bundle(read_file=("Read a text file.",), terminal=("Run a command.",))
        candidate = bundle(
            read_file=("Read a text file.",), terminal=("Run it. Pass `recursive`.",)
        )
        kept, outcomes = enforce_constraints(
            candidate,
            baseline,
            ConstraintValidator(NO_GROWTH_LIMIT),
            allowed=["read_file"],
            accuracy=self.checker(hermes_repo),
        )
        assert kept["terminal"].description == "Run a command."
        assert outcomes == []


class TestAccuracyCheckerWiring:
    def test_the_checker_knows_every_tool_in_the_catalogue(self, hermes_repo):
        checker = build_accuracy_checker(load_catalog(hermes_repo), entailment=False)
        assert set(checker.facts) == {
            "read_file",
            "search_files",
            "write_file",
            "terminal",
            "vision_analyze",
        }

    def test_entailment_false_means_no_predictor(self, hermes_repo):
        checker = build_accuracy_checker(load_catalog(hermes_repo), entailment=False)
        assert checker.entailment is None

    def test_a_predictor_can_be_injected(self, hermes_repo):
        stub = object()
        checker = build_accuracy_checker(load_catalog(hermes_repo), entailment=stub)
        assert checker.entailment is stub

    def test_the_default_builds_one_but_does_not_call_it(self, hermes_repo):
        checker = build_accuracy_checker(load_catalog(hermes_repo))
        assert checker.entailment is not None
        with dspy.context(lm=None):
            assert checker.check_tool("read_file", "Read a file.", "Read a text file.") == []
        assert checker.entailment_ran is False


def overclaiming(bundle):
    """A rewrite that wins examples by promising something the schema forbids."""
    bundle["read_file"].description = (
        "Read a text file with line numbers. Pass `recursive` to walk directories."
    )


class TestFullRunStatistics:
    def run(self, repo, tmp_path, **kwargs):
        return evolve_tool_descriptions(
            hermes_repo=str(repo),
            dataset_path=str(build_dataset(tmp_path / "ds")),
            iterations=2,
            output_root=tmp_path / "out",
            **kwargs,
        )

    def test_metrics_carry_per_tool_statistics(self, hermes_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(tidy))
        metrics = self.run(hermes_repo, tmp_path, write=True)
        rows = {row["tool"]: row for row in metrics["per_tool"]}
        assert set(rows) == {"read_file", "search_files", "terminal", NO_TOOL}
        for row in rows.values():
            assert set(row) >= {
                "baseline_rate",
                "candidate_rate",
                "delta",
                "delta_ci",
                "p_worse",
                "underpowered",
                "min_detectable_shift",
            }

    def test_metrics_carry_the_chance_baseline_and_the_interval(
        self, hermes_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(tidy))
        metrics = self.run(hermes_repo, tmp_path, write=True)
        assert metrics["num_options"] == 6  # five tools plus 'none'
        assert metrics["chance_accuracy"] == pytest.approx(1 / 6)
        assert metrics["baseline_accuracy_ci"]["low"] <= metrics["baseline_accuracy"]
        assert metrics["candidate_accuracy_ci"]["high"] >= metrics["candidate_accuracy"]

    def test_a_tolerance_the_val_split_cannot_detect_is_reported(
        self, hermes_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(tidy))
        metrics = self.run(hermes_repo, tmp_path, write=True, regression_tolerance=0.05)
        assert metrics["regression_tolerance"] == 0.05
        assert metrics["underpowered_tools"]
        assert metrics["cross_tool_accepted"] is True

    def test_a_zero_tolerance_reports_nobody_as_underpowered(
        self, hermes_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(tidy))
        metrics = self.run(hermes_repo, tmp_path, write=True)
        assert metrics["underpowered_tools"] == []

    def test_the_saved_report_carries_the_comparisons(self, hermes_repo, tmp_path, stubbed):
        import json

        stubbed.setattr(dspy, "GEPA", stub_optimizer(tidy))
        self.run(hermes_repo, tmp_path, write=True)
        run_dir = next((tmp_path / "out" / "tools").iterdir())
        blob = json.loads((run_dir / "cross_tool_report.json").read_text())
        assert blob["verdict"]["comparisons"]
        assert blob["baseline"]["chance_accuracy"] == pytest.approx(1 / 6, abs=1e-4)
        assert blob["baseline"]["rates"]["read_file"]["outcomes"]

    def test_an_overclaiming_rewrite_is_reverted_and_never_written(
        self, hermes_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(overclaiming))
        before = (hermes_repo / "tools" / "file_tools.py").read_text()
        metrics = self.run(hermes_repo, tmp_path, write=True)

        assert metrics["factual_reverts"] == 1
        assert metrics["descriptions_changed"] == 0
        assert metrics["written"] is False
        assert (hermes_repo / "tools" / "file_tools.py").read_text() == before

    def test_the_entailment_tier_stays_offline(self, hermes_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", stub_optimizer(tidy))
        metrics = self.run(hermes_repo, tmp_path, write=True)
        assert metrics["entailment_ran"] is False

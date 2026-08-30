"""Phase 2's last two steps: what the run cost, and the PR it emits.

Everything here is offline. No API key, no network, no language model: DSPy's
call log is replaced with a list the test fills in itself, the selector's
forward pass is a keyword match, and the optimizer is a stub. Git is real,
because branch handling is the part that can damage a checkout and mocking it
would test nothing - but the repositories are built in ``tmp_path`` and nothing
is ever pushed.
"""

import json
import os
import subprocess
from pathlib import Path

import dspy
import pytest
from click.testing import CliRunner

from evolution.core.config import EvolutionConfig
from evolution.core.cost import CostReport, LMCall
from evolution.core.gates import GateResult, GateStatus
from evolution.core.pr_builder import GitError, PullRequestPlan, RejectedCandidate
from evolution.tools.cross_tool import CrossToolGuard, CrossToolReport
from evolution.tools.evolve_tool_descriptions import (
    EXIT_DEPLOYMENT_INCOMPLETE,
    ConstraintOutcome,
    benchmark_candidate,
    collect_rejections,
    evolve_tool_descriptions,
    main,
    pr_target_slug,
    score_lines,
)
from evolution.tools.tool_catalog import load_catalog
from evolution.tools.selection_eval import (
    ToolSelectionExample,
    ToolSelector,
    score_selection,
)
from tests.tools.test_evolve_tool_descriptions import build_dataset, keyword_forward

# Two calls DSPy priced, so the run has a real, checkable total.
PRICED_CALLS = [
    {
        "model": "openai/gpt-4.1",
        "usage": {"prompt_tokens": 1200, "completion_tokens": 300},
        "cost": 0.0125,
    },
    {
        "model": "openai/gpt-4.1-mini",
        "usage": {"prompt_tokens": 800, "completion_tokens": 100},
        "cost": 0.0025,
    },
]

# A model DSPy has no price for. Must never be summed as zero.
UNPRICED_CALLS = [
    {
        "model": "local/hermes-4",
        "usage": {"prompt_tokens": 4000, "completion_tokens": 900},
        "cost": None,
    }
]


# ──────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ──────────────────────────────────────────────────────────────────────────


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=str(repo), check=check, capture_output=True, text=True
    )


@pytest.fixture
def git_repo(hermes_repo):
    """The fake hermes-agent checkout, under real version control."""
    git(hermes_repo, "init", "-q")
    git(hermes_repo, "config", "user.email", "t@example.com")
    git(hermes_repo, "config", "user.name", "Tester")
    git(hermes_repo, "config", "commit.gpgsign", "false")
    git(hermes_repo, "add", "-A")
    git(hermes_repo, "commit", "-qm", "base")
    return hermes_repo


def head(repo):
    return git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def branches(repo):
    out = git(repo, "branch", "--format=%(refname:short)").stdout
    return set(out.split())


def evolve_branches(repo):
    return {b for b in branches(repo) if b.startswith("evolve/")}


def file_on(repo, ref, path):
    return git(repo, "show", f"{ref}:{path}").stdout


@pytest.fixture
def spending(monkeypatch):
    """Stand in for DSPy's global call log with a list the test controls."""
    log = []
    monkeypatch.setattr("evolution.core.cost._history", lambda: log)
    return log


@pytest.fixture
def stubbed(monkeypatch):
    """Neutralise every route to a language model."""
    monkeypatch.setattr(ToolSelector, "forward", keyword_forward)
    monkeypatch.setattr(dspy, "LM", lambda *a, **kw: object())
    monkeypatch.setattr(dspy, "configure", lambda **kw: None)
    return monkeypatch


def optimizer(mutate, spends=(), log=None):
    """A stub optimizer that edits the bundle and books *spends* while doing it."""

    class Stub:
        def __init__(self, *args, **kwargs):
            pass

        def compile(self, student, **kwargs):
            if log is not None:
                log.extend(spends)
            evolved = ToolSelector(student.bundle, student.signatures)
            mutate(evolved.bundle)
            return evolved

    return Stub


# The over-budget baseline repeats this sentence six times. A tidy rewrite drops
# it, so its presence or absence tells the two versions apart.
PADDING = "Prefer the purpose-built file tools"


def tidy_terminal(bundle):
    """A clean, in-budget rewrite of the one over-budget description."""
    bundle["terminal"].description = "Run a shell command in a persistent session."


def overclaim_and_tidy(bundle):
    """One rewrite the factual check must refuse, one it must keep."""
    bundle["read_file"].description = (
        "Read a text file with line numbers. Pass `recursive` to walk directories."
    )
    tidy_terminal(bundle)


def oversized_and_tidy(bundle):
    """One rewrite past the 500 char budget, one inside it."""
    bundle["read_file"].description = "Read a text file. " + ("y" * 600)
    tidy_terminal(bundle)


def greedy(bundle):
    """A rewrite that wins by stealing every other tool's selections.

    It stays on its own subject while doing so. An off-topic land-grab is
    caught earlier and more cheaply by the semantic_preservation constraint,
    which would leave the cross-tool guard untested; the candidate the guard
    exists for is the one that still sounds like itself.
    """
    bundle["search_files"].description = (
        "Search file contents or filenames with a regular expression, "
        "and grabs everything else, always."
    )


def run(repo, tmp_path, **kwargs):
    return evolve_tool_descriptions(
        hermes_repo=str(repo),
        dataset_path=str(build_dataset(tmp_path / "ds")),
        iterations=2,
        output_root=tmp_path / "out",
        **kwargs,
    )


def run_dir(tmp_path):
    return next((tmp_path / "out" / "tools").iterdir())


def body_of(tmp_path):
    return (run_dir(tmp_path) / "PULL_REQUEST.md").read_text()


def report_of(pairs, tools=("read_file", "terminal")):
    """A CrossToolReport over ``(expected, predicted)`` pairs, keyed by position.

    Two reports built from the same length of pairs align example for example,
    which is what lets the guard run its paired test.
    """
    scored = [
        score_selection(
            ToolSelectionExample(task=f"task {i}", correct_tool=expected), predicted
        )
        for i, (expected, predicted) in enumerate(pairs)
    ]
    return CrossToolReport.from_outcomes(scored, tools=list(tools))


# ──────────────────────────────────────────────────────────────────────────
# What the run cost
# ──────────────────────────────────────────────────────────────────────────


class TestCostIsMeasured:
    def test_metrics_carry_the_measured_cost(self, hermes_repo, tmp_path, stubbed, spending):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal, PRICED_CALLS, spending))
        metrics = run(hermes_repo, tmp_path, write=True)

        assert metrics["cost"]["calls"] == 2
        assert metrics["cost"]["known_cost_usd"] == pytest.approx(0.015)
        assert metrics["cost"]["total_tokens"] == 2400
        assert metrics["cost"]["complete"] is True

    def test_the_saved_metrics_file_carries_it_too(
        self, hermes_repo, tmp_path, stubbed, spending
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal, PRICED_CALLS, spending))
        run(hermes_repo, tmp_path, write=True)
        saved = json.loads((run_dir(tmp_path) / "metrics.json").read_text())
        assert saved["cost"]["known_cost_usd"] == pytest.approx(0.015)

    def test_every_model_the_run_touched_is_named(
        self, hermes_repo, tmp_path, stubbed, spending
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal, PRICED_CALLS, spending))
        metrics = run(hermes_repo, tmp_path, write=True)
        assert metrics["cost"]["models"] == {"openai/gpt-4.1": 1, "openai/gpt-4.1-mini": 1}

    def test_an_unpriced_call_is_not_summed_as_zero(
        self, hermes_repo, tmp_path, stubbed, spending
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal, UNPRICED_CALLS, spending))
        metrics = run(hermes_repo, tmp_path, write=True)

        assert metrics["cost"]["unpriced_calls"] == 1
        assert metrics["cost"]["complete"] is False
        assert metrics["cost"]["known_cost_usd"] == 0.0

    def test_a_run_that_reached_no_model_says_so(self, hermes_repo, tmp_path, stubbed, spending):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(hermes_repo, tmp_path, write=True)
        assert metrics["cost"]["calls"] == 0

    def test_the_console_summary_reports_the_cost(
        self, hermes_repo, tmp_path, stubbed, spending
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal, PRICED_CALLS, spending))
        result = cli_run(hermes_repo, tmp_path, ["--write"])
        assert "$0.0150" in result.output, result.output

    def test_the_console_does_not_round_an_unknown_price_down(
        self, hermes_repo, tmp_path, stubbed, spending
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal, UNPRICED_CALLS, spending))
        result = cli_run(hermes_repo, tmp_path, ["--write"])
        assert "at least" in result.output
        assert "no price available" in result.output

    def test_the_pr_body_carries_the_cost(self, git_repo, tmp_path, stubbed, spending):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal, PRICED_CALLS, spending))
        run(git_repo, tmp_path, write=True)
        assert "$0.0150" in body_of(tmp_path)


def cli_run(repo, tmp_path, extra):
    """Invoke the CLI in a scratch cwd so ./output never lands in the project."""
    runner = CliRunner()
    dataset = str(build_dataset(tmp_path / "ds"))
    scratch = tmp_path / "cli-cwd"
    scratch.mkdir(exist_ok=True)
    previous = os.getcwd()
    os.chdir(scratch)
    try:
        return runner.invoke(
            main,
            [
                "--hermes-repo", str(repo),
                "--dataset-path", dataset,
                "--iterations", "2",
                *extra,
            ],
            env={"COLUMNS": "200"},
        )
    finally:
        os.chdir(previous)


# ──────────────────────────────────────────────────────────────────────────
# A PR is built only when there is something to deploy
# ──────────────────────────────────────────────────────────────────────────


class TestBuiltOnlyOnWrite:
    def test_a_write_builds_the_branch(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True)

        assert metrics["written"] is True
        assert evolve_branches(git_repo) == {metrics["pull_request"]["branch"]}

    def test_no_write_builds_nothing(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=False)

        assert metrics["descriptions_changed"] == 1
        assert metrics["written"] is False
        assert metrics["pull_request"] is None
        assert evolve_branches(git_repo) == set()
        assert not (run_dir(tmp_path) / "PULL_REQUEST.md").exists()

    def test_a_dry_run_builds_nothing(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        assert run(git_repo, tmp_path, write=True, dry_run=True) is None
        assert evolve_branches(git_repo) == set()
        assert not (tmp_path / "out").exists()

    def test_a_rejected_candidate_builds_nothing(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(greedy))
        metrics = run(git_repo, tmp_path, write=True)

        assert metrics["cross_tool_accepted"] is False
        assert metrics["written"] is False
        assert evolve_branches(git_repo) == set()

    def test_a_candidate_that_changed_nothing_builds_nothing(
        self, git_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(lambda bundle: None))
        metrics = run(git_repo, tmp_path, write=True)

        assert metrics["descriptions_changed"] == 0
        assert evolve_branches(git_repo) == set()

    def test_no_create_pr_writes_the_files_and_stops(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True, create_pr=False)

        assert metrics["written"] is True
        assert metrics["pull_request"] is None
        assert evolve_branches(git_repo) == set()
        # The rewrite is still on disk, uncommitted, for the operator to use.
        assert PADDING not in (git_repo / "tools" / "shell_tools.py").read_text()

    def test_create_pr_defaults_to_the_config_field(self, git_repo, tmp_path, stubbed):
        """The field defaulted to True with nothing reading it. Now it decides."""
        assert EvolutionConfig().create_pr is True
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True)
        assert metrics["pull_request"] is not None

    def test_a_checkout_without_git_is_reported_not_crashed(
        self, hermes_repo, tmp_path, stubbed
    ):
        """The fake repo is not a git repo. The run must finish anyway."""
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(hermes_repo, tmp_path, write=True)

        assert metrics["written"] is True
        assert metrics["pull_request"] is None
        assert not (run_dir(tmp_path) / "PULL_REQUEST.md").exists()


# ──────────────────────────────────────────────────────────────────────────
# The branch, the body, and where they live
# ──────────────────────────────────────────────────────────────────────────


class TestBranchAndBody:
    def test_the_branch_name_matches_the_run_timestamp(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True)

        assert metrics["pull_request"]["branch"] == f"evolve/all-tools-{metrics['timestamp']}"
        # The artifacts directory is named for the same timestamp, so the branch
        # and the evidence for it can be lined up later without guessing.
        assert run_dir(tmp_path).name == metrics["timestamp"]

    def test_the_body_sits_with_the_run_artifacts(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True)
        assert metrics["pull_request"]["body_path"] == str(run_dir(tmp_path) / "PULL_REQUEST.md")

    def test_the_commit_carries_the_written_files(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True)
        branch = metrics["pull_request"]["branch"]

        committed = file_on(git_repo, branch, "tools/shell_tools.py")
        assert metrics["pull_request"]["files"] == ["tools/shell_tools.py"]
        assert "Run a shell command in a persistent session." in committed
        assert PADDING not in committed

    def test_one_tool_names_the_branch_after_itself(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True, tools=["terminal"])
        assert metrics["pull_request"]["branch"].startswith("evolve/terminal-")

    def test_a_toolset_run_names_the_branch_after_the_toolset(
        self, git_repo, tmp_path, stubbed
    ):
        def tidy_read(bundle):
            bundle["read_file"].description = "Read a text file with line numbers."

        stubbed.setattr(dspy, "GEPA", optimizer(tidy_read))
        metrics = run(git_repo, tmp_path, write=True, toolset="file")
        assert metrics["pull_request"]["branch"].startswith("evolve/file-toolset-")

    def test_the_body_carries_every_split_that_was_measured(
        self, git_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True)
        body = body_of(tmp_path)

        assert "| val |" in body
        assert "| holdout |" in body
        # No train row: this phase never scores train, and an invented number
        # would be worse than a missing one.
        assert "| train |" not in body

    def test_the_body_carries_the_gate_ladder(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True)
        body = body_of(tmp_path)

        assert "## Gates" in body
        assert "cross_tool" in body
        assert "tblite" in body

    def test_the_body_carries_the_paired_evidence(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True)
        body = body_of(tmp_path)

        assert "## Evidence" in body
        assert "cross-tool accepted" in body
        assert "tolerance" in body

    def test_the_body_carries_the_diff(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True)
        body = body_of(tmp_path)

        assert "```diff" in body
        # The removed padding and the shorter replacement are both in the diff.
        assert PADDING in body
        assert any(
            line.startswith("+") and "'Run a shell command in a persistent session.'" in line
            for line in body.splitlines()
        ), body

    def test_the_body_names_the_dataset_and_the_optimizer(
        self, git_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True)
        body = body_of(tmp_path)

        assert "Optimizer: GEPA, 2 iteration(s)" in body
        assert "12 train / 4 val / 8 holdout" in body

    def test_the_console_prints_the_branch_and_the_body_path(
        self, git_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        result = cli_run(git_repo, tmp_path, ["--write"])

        assert "Branch: evolve/all-tools-" in result.output, result.output
        assert "PULL_REQUEST.md" in result.output


# ──────────────────────────────────────────────────────────────────────────
# What was refused on the way, and why
# ──────────────────────────────────────────────────────────────────────────


class TestRejectedCandidatesReachTheBody:
    def test_a_factual_revert_is_named_with_its_real_reason(
        self, git_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(overclaim_and_tidy))
        metrics = run(git_repo, tmp_path, write=True)
        body = body_of(tmp_path)

        assert metrics["factual_reverts"] == 1
        assert "Rejected along the way" in body
        assert "`read_file`" in body
        assert "factual_accuracy:" in body
        assert "recursive" in body

    def test_a_budget_revert_is_named_with_its_real_reason(
        self, git_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(oversized_and_tidy))
        run(git_repo, tmp_path, write=True)
        body = body_of(tmp_path)

        assert "size_limit: Size exceeded" in body
        assert "/500 chars" in body

    def test_a_passing_check_is_never_quoted_as_a_rejection(
        self, git_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(overclaim_and_tidy))
        run(git_repo, tmp_path, write=True)
        rejected = body_of(tmp_path).split("## Rejected along the way")[1].split("##")[0]
        assert "Size OK" not in rejected
        assert "non_empty" not in rejected

    def test_a_clean_run_has_no_rejection_section(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True)
        assert "Rejected along the way" not in body_of(tmp_path)

    def test_the_reverted_reason_is_recorded_on_the_outcome(self):
        outcome = ConstraintOutcome(
            target="read_file",
            kind="tool_description",
            passed=False,
            reverted=True,
            messages=["size_limit: Size OK: 12/500 chars", "growth_limit: Growth exceeded: +80%"],
            failures=["growth_limit: Growth exceeded: +80%"],
        )
        assert outcome.reason() == "growth_limit: Growth exceeded: +80%"
        assert outcome.to_dict()["failures"] == ["growth_limit: Growth exceeded: +80%"]


class TestCollectRejections:
    def outcome(self, target, reverted, failures=()):
        return ConstraintOutcome(
            target=target,
            kind="tool_description",
            passed=not reverted,
            reverted=reverted,
            messages=list(failures),
            failures=list(failures),
        )

    def test_only_reverted_outcomes_are_collected(self):
        rejected = collect_rejections(
            [
                self.outcome("read_file", True, ["size_limit: 528/500"]),
                self.outcome("terminal", False),
            ]
        )
        assert [r.label for r in rejected] == ["read_file"]
        assert rejected[0].reason == "size_limit: 528/500"

    def test_a_revert_with_no_message_still_says_something(self):
        rejected = collect_rejections([self.outcome("read_file", True)])
        assert rejected[0].reason == "reverted to baseline"

    def test_a_regressed_tool_is_collected_per_split(self):
        verdict = self.verdict(regressed=True)
        rejected = collect_rejections([], verdict=verdict)
        assert any("val cross-tool" in r.label for r in rejected)
        assert all(r.reason for r in rejected)

    def test_the_holdout_guard_is_collected_too(self):
        verdict = self.verdict(regressed=True)
        rejected = collect_rejections([], holdout_verdict=verdict)
        assert any("holdout cross-tool" in r.label for r in rejected)

    def test_an_accepted_verdict_contributes_nothing(self):
        rejected = collect_rejections([], verdict=self.verdict(regressed=False))
        assert rejected == []

    def test_a_refusal_with_no_named_regression_is_still_reported(self):
        verdict = self.verdict(regressed=False)
        verdict.accepted = False
        verdict.reason = "overall accuracy did not improve"
        rejected = collect_rejections([], verdict=verdict)
        assert rejected == [
            RejectedCandidate(
                "whole candidate (val cross-tool)", "overall accuracy did not improve"
            )
        ]

    def verdict(self, regressed):
        before = report_of([("read_file", "read_file")] * 10)
        after = report_of(
            [("read_file", "terminal" if regressed else "read_file")] * 10
        )
        return CrossToolGuard().compare(before, after)


# ──────────────────────────────────────────────────────────────────────────
# The checkout is always handed back the way it was found
# ──────────────────────────────────────────────────────────────────────────


class TestTheRefIsRestored:
    def test_a_successful_run_leaves_the_original_ref_checked_out(
        self, git_repo, tmp_path, stubbed
    ):
        before = head(git_repo)
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True)
        assert head(git_repo) == before

    def test_the_change_lives_on_the_branch_not_on_the_checkout(
        self, git_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True)
        branch = metrics["pull_request"]["branch"]

        assert PADDING not in file_on(git_repo, branch, "tools/shell_tools.py")
        assert PADDING in (git_repo / "tools" / "shell_tools.py").read_text()

    def test_the_tracked_tree_is_left_clean(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True)
        dirty = git(git_repo, "status", "--porcelain", "--untracked-files=no").stdout
        assert dirty.strip() == ""

    def test_a_failed_push_still_restores_the_ref(self, git_repo, tmp_path, stubbed):
        """No remote is configured, so the push fails. The run must not strand us."""
        before = head(git_repo)
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True, push=True)

        assert head(git_repo) == before
        assert metrics["pull_request"] is not None
        assert evolve_branches(git_repo) == {metrics["pull_request"]["branch"]}

    def test_a_failed_open_still_restores_the_ref(self, git_repo, tmp_path, stubbed):
        before = head(git_repo)

        def refuse(self, *args, **kwargs):
            raise GitError("gh is not installed")

        stubbed.setattr(PullRequestPlan, "open", refuse)
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True, open_pr=True)
        assert head(git_repo) == before

    def test_a_failure_inside_the_build_still_restores_the_ref(
        self, git_repo, tmp_path, stubbed
    ):
        """build_pull_request makes the branch before it can fail, and returns no plan."""
        before = head(git_repo)

        def half_build(*, repo, target, timestamp, **kwargs):
            subprocess.run(
                ["git", "checkout", "-b", f"evolve/{target}-{timestamp}"],
                cwd=str(repo), capture_output=True, text=True, check=True,
            )
            raise GitError("git add failed: pathspec did not match")

        stubbed.setattr(
            "evolution.tools.evolve_tool_descriptions.build_pull_request", half_build
        )
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True)

        assert head(git_repo) == before
        assert metrics["pull_request"] is None

    def test_the_run_still_returns_its_metrics_after_a_git_failure(
        self, git_repo, tmp_path, stubbed
    ):
        def explode(**kwargs):
            raise GitError("something went wrong in git")

        stubbed.setattr(
            "evolution.tools.evolve_tool_descriptions.build_pull_request", explode
        )
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True)

        assert metrics["written"] is True
        assert metrics["pull_request"] is None
        assert (run_dir(tmp_path) / "metrics.json").exists()


# ──────────────────────────────────────────────────────────────────────────
# Nothing reaches the network unless it was asked for
# ──────────────────────────────────────────────────────────────────────────


class TestNothingIsPublished:
    def test_no_push_by_default(self, git_repo, tmp_path, stubbed):
        """A real remote is configured. The run must still leave it untouched."""
        remote = tmp_path / "origin.git"
        subprocess.run(
            ["git", "init", "--bare", "-q", str(remote)], check=True, capture_output=True
        )
        git(git_repo, "remote", "add", "origin", str(remote))

        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True)

        pushed = subprocess.run(
            ["git", "branch", "--format=%(refname:short)"],
            cwd=str(remote), capture_output=True, text=True,
        ).stdout.split()
        assert pushed == []
        assert metrics["pull_request"] is not None

    def test_push_is_never_called_without_the_flag(self, git_repo, tmp_path, stubbed):
        def forbidden(self, *args, **kwargs):
            raise AssertionError("push() must not run without --push")

        stubbed.setattr(PullRequestPlan, "push", forbidden)
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True)

    def test_open_is_never_called_without_the_flag(self, git_repo, tmp_path, stubbed):
        def forbidden(self, *args, **kwargs):
            raise AssertionError("open() must not run without --open-pr")

        stubbed.setattr(PullRequestPlan, "open", forbidden)
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True)

    def test_push_is_attempted_only_when_asked(self, git_repo, tmp_path, stubbed):
        """--push reaches push(). The call is intercepted; nothing leaves the machine."""
        calls = []

        def record(self, remote="origin"):
            calls.append(remote)
            return ""

        stubbed.setattr(PullRequestPlan, "push", record)
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True, push=True)
        assert calls == ["origin"]

    def test_open_uses_the_requested_base(self, git_repo, tmp_path, stubbed):
        calls = []

        def record(self, base="main"):
            calls.append(base)
            return ""

        stubbed.setattr(PullRequestPlan, "open", record)
        stubbed.setattr(PullRequestPlan, "push", lambda self, remote="origin": "")
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        run(git_repo, tmp_path, write=True, push=True, open_pr=True, pr_base="develop")
        assert calls == ["develop"]

    def test_the_console_says_plainly_that_nothing_was_pushed(
        self, git_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        result = cli_run(git_repo, tmp_path, ["--write"])
        assert "Nothing was pushed and no pull request was opened" in result.output

    def test_a_run_that_wrote_nothing_still_says_nothing_was_pushed(
        self, git_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        result = cli_run(git_repo, tmp_path, [])
        assert "nothing was pushed" in result.output

    def test_open_without_push_warns_before_it_tries(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(PullRequestPlan, "open", lambda self, base="main": "")
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        result = cli_run(git_repo, tmp_path, ["--write", "--open-pr"])
        assert "--open-pr without --push" in result.output


# ──────────────────────────────────────────────────────────────────────────
# CLI surface
# ──────────────────────────────────────────────────────────────────────────


class TestCli:
    def help(self):
        return CliRunner().invoke(main, ["--help"], env={"COLUMNS": "200"}).output

    def test_help_lists_the_deployment_options(self):
        output = self.help()
        for option in (
            "--create-pr",
            "--no-create-pr",
            "--push",
            "--no-push",
            "--open-pr",
            "--no-open-pr",
            "--pr-base",
        ):
            assert option in output

    def test_help_still_lists_every_option_it_used_to(self):
        output = self.help()
        for option in ("--tool", "--toolset", "--write", "--no-write", "--dry-run"):
            assert option in output

    def test_a_dry_run_says_it_builds_nothing(self, hermes_repo):
        result = CliRunner().invoke(
            main, ["--hermes-repo", str(hermes_repo), "--dry-run"], env={"COLUMNS": "200"}
        )
        assert "A dry run builds no branch and sends nothing." in result.output

    def test_a_dry_run_describes_the_branch_it_would_build(self, hermes_repo):
        result = CliRunner().invoke(
            main,
            ["--hermes-repo", str(hermes_repo), "--dry-run", "--write"],
            env={"COLUMNS": "200"},
        )
        assert "evolve/all-tools-<timestamp>" in result.output
        assert "would not push" in result.output.lower()

    def test_a_dry_run_with_no_create_pr_says_so(self, hermes_repo):
        result = CliRunner().invoke(
            main,
            ["--hermes-repo", str(hermes_repo), "--dry-run", "--write", "--no-create-pr"],
            env={"COLUMNS": "200"},
        )
        assert "Would not build a branch" in result.output

    def test_no_create_pr_is_honoured_end_to_end(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        result = cli_run(git_repo, tmp_path, ["--write", "--no-create-pr"])
        assert evolve_branches(git_repo) == set()
        assert "--no-create-pr" in result.output


# ──────────────────────────────────────────────────────────────────────────
# Small pure pieces
# ──────────────────────────────────────────────────────────────────────────


class TestTargetSlug:
    def test_a_single_tool_names_itself(self):
        assert pr_target_slug(["read_file"]) == "read_file"

    def test_a_toolset_run_names_the_toolset(self):
        assert pr_target_slug(["read_file", "write_file"], "file") == "file-toolset"

    def test_a_whole_catalogue_run_says_so(self):
        assert pr_target_slug(["read_file", "terminal"]) == "all-tools"

    def test_an_empty_selection_still_produces_a_name(self):
        assert pr_target_slug([]) == "all-tools"

    def test_nothing_unsafe_reaches_a_branch_name(self):
        assert pr_target_slug(["weird tool/name"]) == "weird-tool-name"
        assert "/" not in pr_target_slug([], "file system")
        assert pr_target_slug([], "file system") == "file-system-toolset"

    def test_a_single_tool_beats_the_toolset(self):
        assert pr_target_slug(["terminal"], "terminal") == "terminal"


class Holdout:
    def __init__(self, accuracy):
        self.tool_accuracy = accuracy


class TestScoreLines:
    def report(self, correct, total):
        pairs = [("read_file", "read_file")] * correct
        pairs += [("read_file", "terminal")] * (total - correct)
        return report_of(pairs)

    def test_val_is_always_there(self):
        lines = score_lines(self.report(5, 10), self.report(8, 10), val_examples=10)
        assert [line.split for line in lines] == ["val"]
        assert lines[0].baseline == pytest.approx(0.5)
        assert lines[0].evolved == pytest.approx(0.8)

    def test_holdout_comes_last_so_it_headlines_the_commit(self):
        lines = score_lines(
            self.report(5, 10),
            self.report(8, 10),
            val_examples=10,
            baseline_holdout=Holdout(0.4),
            candidate_holdout=Holdout(0.6),
            holdout_examples=12,
        )
        assert [line.split for line in lines] == ["val", "holdout"]
        assert lines[-1].delta == pytest.approx(0.2)

    def test_no_holdout_means_no_holdout_row(self):
        lines = score_lines(
            self.report(5, 10), self.report(8, 10), val_examples=10, baseline_holdout=None
        )
        assert len(lines) == 1

    def test_the_val_row_carries_the_chance_baseline(self):
        lines = score_lines(self.report(5, 10), self.report(8, 10), val_examples=10)
        assert "chance" in lines[0].detail
        assert "10 examples" in lines[0].detail


class TestCostReportShape:
    """A guard on the contract this module depends on, not a retest of cost.py."""

    def test_describe_flags_an_incomplete_total(self):
        report = CostReport(calls=[LMCall("m", 10, 5, None)])
        assert "at least" in report.describe()

    def test_to_dict_carries_what_metrics_json_promises(self):
        blob = CostReport(calls=[LMCall("m", 10, 5, 0.5)]).to_dict()
        assert set(blob) >= {
            "calls",
            "total_tokens",
            "known_cost_usd",
            "unpriced_calls",
            "complete",
            "models",
        }


# ──────────────────────────────────────────────────────────────────────────
# The exit status and the metrics both have to match what really happened
# ──────────────────────────────────────────────────────────────────────────


class TestTheExitStatusMatchesWhatHappened:
    """Phase 5 reads this status to decide whether an optimization was proposed.

    A run that reports failure after succeeding is as useless to it as one that
    reports success after a failed push, so both directions are pinned here.
    """

    def test_a_successful_run_exits_zero(self, git_repo, tmp_path, stubbed):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        result = cli_run(git_repo, tmp_path, ["--write"])
        assert result.exit_code == 0

    def test_a_successful_run_does_not_print_its_metrics_dict(
        self, git_repo, tmp_path, stubbed
    ):
        """The metrics belong in metrics.json, not dumped on the way out."""
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        result = cli_run(git_repo, tmp_path, ["--write"])
        assert "'elapsed_seconds'" not in result.output

    def test_a_dry_run_exits_zero(self, hermes_repo):
        result = CliRunner().invoke(
            main, ["--hermes-repo", str(hermes_repo), "--dry-run"], env={"COLUMNS": "200"}
        )
        assert result.exit_code == 0

    def test_a_failed_push_exits_deployment_incomplete(
        self, git_repo, tmp_path, stubbed
    ):
        """No remote is configured, so --push fails for real."""
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        result = cli_run(git_repo, tmp_path, ["--write", "--push"])
        assert result.exit_code == EXIT_DEPLOYMENT_INCOMPLETE

    def test_a_failed_open_exits_deployment_incomplete(
        self, git_repo, tmp_path, stubbed
    ):
        def refuse(self, *args, **kwargs):
            raise GitError("gh is not installed")

        stubbed.setattr(PullRequestPlan, "open", refuse)
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        result = cli_run(git_repo, tmp_path, ["--write", "--open-pr"])
        assert result.exit_code == EXIT_DEPLOYMENT_INCOMPLETE

    def test_a_checkout_without_git_still_exits_zero(self, hermes_repo, tmp_path, stubbed):
        """The write succeeded and nothing was asked for. Not a deployment failure."""
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        result = cli_run(hermes_repo, tmp_path, ["--write"])
        assert result.exit_code == 0

    def test_a_failed_push_is_recorded_in_the_saved_metrics(
        self, git_repo, tmp_path, stubbed
    ):
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True, push=True)

        assert metrics["deployment"]["status"] == "failed"
        assert metrics["deployment"]["pushed"] is False
        assert metrics["deployment"]["error"]

        saved = json.loads((run_dir(tmp_path) / "metrics.json").read_text())
        assert saved["deployment"]["status"] == "failed"

    def test_a_local_branch_is_never_recorded_as_pushed(
        self, git_repo, tmp_path, stubbed
    ):
        """Neither flag was passed, so nothing left the machine."""
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True)

        assert metrics["pull_request"] is not None
        assert metrics["deployment"]["status"] == "ok"
        assert metrics["deployment"]["pushed"] is False
        assert metrics["deployment"]["opened"] is False
        assert metrics["deployment"]["requested"] == {"push": False, "open_pr": False}

    def test_a_build_that_failed_is_not_recorded_as_ok(
        self, git_repo, tmp_path, stubbed
    ):
        def explode(**kwargs):
            raise GitError("something went wrong in git")

        stubbed.setattr(
            "evolution.tools.evolve_tool_descriptions.build_pull_request", explode
        )
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=True)

        assert metrics["pull_request"] is None
        assert metrics["deployment"]["status"] == "failed"

    def test_a_run_that_never_deployed_still_has_the_key(
        self, git_repo, tmp_path, stubbed
    ):
        """Same shape as pull_request: always present, None when it did not happen."""
        stubbed.setattr(dspy, "GEPA", optimizer(tidy_terminal))
        metrics = run(git_repo, tmp_path, write=False)
        assert metrics["deployment"] is None


# ──────────────────────────────────────────────────────────────────────────
# The benchmark has to measure the candidate, not the baseline twice
# ──────────────────────────────────────────────────────────────────────────


class TestTheBenchmarkSeesTheCandidate:
    """Both TBLite measurements used to run against the unmodified checkout.

    A gate that scores the baseline twice reports PASSED on evidence that
    cannot contain a regression, which is worse than not running it.
    """

    def bundle(self, repo):
        loaded = load_catalog(repo).bundle()
        loaded["read_file"].description = "EVOLVED read_file description."
        return loaded

    def test_an_unavailable_benchmark_is_not_worth_copying_a_repo(
        self, hermes_repo, monkeypatch
    ):
        ran = []
        monkeypatch.setattr(
            "evolution.tools.evolve_tool_descriptions.run_benchmark_gate",
            lambda *a, **kw: ran.append(a) or GateResult("tblite", GateStatus.PASSED, "x"),
        )
        baseline = GateResult("tblite", GateStatus.UNAVAILABLE, "benchmark not found")

        result = benchmark_candidate(hermes_repo, self.bundle(hermes_repo), baseline)

        assert result is baseline
        assert ran == []

    def test_the_measured_checkout_carries_the_candidate(
        self, hermes_repo, monkeypatch
    ):
        seen = {}

        def fake_gate(repo, name, **kwargs):
            seen["repo"] = Path(repo)
            seen["source"] = (Path(repo) / "tools" / "file_tools.py").read_text()
            seen["baseline"] = kwargs.get("baseline")
            return GateResult(name, GateStatus.PASSED, "ok", score=0.9)

        monkeypatch.setattr(
            "evolution.tools.evolve_tool_descriptions.run_benchmark_gate", fake_gate
        )
        baseline = GateResult("tblite", GateStatus.PASSED, "ok", score=0.95)

        benchmark_candidate(hermes_repo, self.bundle(hermes_repo), baseline)

        assert "EVOLVED read_file description." in seen["source"]
        assert seen["baseline"] == 0.95

    def test_the_operators_checkout_is_never_written_to(
        self, hermes_repo, monkeypatch
    ):
        before = (hermes_repo / "tools" / "file_tools.py").read_text()
        measured = {}

        def fake_gate(repo, name, **kwargs):
            measured["repo"] = Path(repo)
            return GateResult(name, GateStatus.PASSED, "ok", score=0.9)

        monkeypatch.setattr(
            "evolution.tools.evolve_tool_descriptions.run_benchmark_gate", fake_gate
        )
        baseline = GateResult("tblite", GateStatus.PASSED, "ok", score=0.95)

        benchmark_candidate(hermes_repo, self.bundle(hermes_repo), baseline)

        assert measured["repo"] != hermes_repo
        assert (hermes_repo / "tools" / "file_tools.py").read_text() == before

    def test_the_copy_does_not_outlive_the_measurement(self, hermes_repo, monkeypatch):
        measured = {}

        def fake_gate(repo, name, **kwargs):
            measured["repo"] = Path(repo)
            return GateResult(name, GateStatus.PASSED, "ok", score=0.9)

        monkeypatch.setattr(
            "evolution.tools.evolve_tool_descriptions.run_benchmark_gate", fake_gate
        )
        baseline = GateResult("tblite", GateStatus.PASSED, "ok", score=0.95)

        benchmark_candidate(hermes_repo, self.bundle(hermes_repo), baseline)
        assert not measured["repo"].exists()


class TestFactualRevertsCountsReverts:
    """`messages` holds every check that ran. Only `failures` refused anything."""

    def outcome(self, **kw):
        base = dict(
            target="read_file",
            kind="tool_description",
            passed=True,
            reverted=False,
            messages=[],
            failures=[],
        )
        base.update(kw)
        return ConstraintOutcome(**base)

    def count(self, outcomes):
        return sum(
            1
            for o in outcomes
            if o.reverted and any(m.startswith("factual_accuracy:") for m in o.failures)
        )

    def test_a_passing_factual_check_is_not_a_revert(self):
        passed = self.outcome(messages=["factual_accuracy: nothing unsupported"])
        assert self.count([passed]) == 0

    def test_a_failing_factual_check_is_a_revert(self):
        failed = self.outcome(
            passed=False,
            reverted=True,
            messages=["size_limit: Size OK", "factual_accuracy: claims recursive"],
            failures=["factual_accuracy: claims recursive"],
        )
        assert self.count([failed]) == 1

    def test_a_budget_revert_is_not_counted_as_a_factual_one(self):
        budget = self.outcome(
            passed=False,
            reverted=True,
            messages=["size_limit: 812/500 chars"],
            failures=["size_limit: 812/500 chars"],
        )
        assert self.count([budget]) == 0

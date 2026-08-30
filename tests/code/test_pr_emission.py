"""Tests for Phase 4's deployment step: cost accounting and the PR body.

Phase 4 already ends on a branch ``CodeOrganism`` created, so the thing under
test here is that the body is rendered for *that* branch and that no second
branch mechanism appears beside it.

Offline throughout. No LM is ever configured, so every cost report these runs
produce is genuinely empty and the priced cases are built from ``LMCall``
objects by hand. Nothing pushes: :meth:`PullRequestPlan.push` and
:meth:`PullRequestPlan.open` are replaced with recorders in every end to end
test, so a regression that started pushing would fail the assertion rather
than reach a remote. The temporary repos have no remotes configured either.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolution.code import evolve_tool_code as mod
from evolution.code.evolve_tool_code import (
    EVOLVER_COST_NOTE,
    PHASE_LABEL,
    REVIEW_NOTE,
    UNSANDBOXED,
    Candidate,
    CandidateOutcome,
    build_code_pull_request,
    code_rejected_candidates,
    code_score_lines,
    evolve_tool_code,
)
from evolution.code.fitness_code import BaselineSnapshot, CodeFitness
from evolution.code.organism import git_available
from evolution.code.safety import QualitySignals, SafetyReport
from evolution.core.cost import CostReport, LMCall
from evolution.core.gates import GateResult, GateStatus

BASELINE = '''"""Toy file tools."""


def read_lines(path, limit=10):
    """Return up to *limit* lines from *path*."""
    try:
        with open(path) as handle:
            return handle.read().splitlines()[:limit - 1]
    except OSError:
        return []
'''

# Fixes the off-by-one and leaves everything else alone.
FIXED = BASELINE.replace("[:limit - 1]", "[:limit]")

# Fixes the bug and breaks the contract: a new parameter on a public function.
# The signature guardrail catches this one before any test runs.
UNSAFE = FIXED.replace(
    "def read_lines(path, limit=10):",
    "def read_lines(path, limit=10, encoding='utf-8'):",
)

# Fixes the bug, keeps the signature, keeps the error handling - and quietly
# changes what a missing file returns. Nothing static can see that; the test
# suite can, which is exactly the case the hard pytest gate exists for.
BREAKS_TESTS = FIXED.replace("        return []\n", "        return None\n")

REPRO = '''import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())
from tools.file_tools import read_lines

with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
    handle.write("a\\nb\\nc\\n")
    path = handle.name

if read_lines(path, limit=3) == ["a", "b", "c"]:
    print("BUG_FIXED")
    sys.exit(0)
print("BUG_PRESENT")
sys.exit(1)
'''

# Green at baseline and green for the real fix, red for BREAKS_TESTS.
SUITE = '''from tools.file_tools import read_lines


def test_a_missing_file_reads_as_empty():
    assert read_lines("/definitely/not/here.txt") == []
'''


needs_git = pytest.mark.skipif(
    not git_available(), reason="git is not installed on this machine"
)


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )


def make_repo(root: Path) -> Path:
    """A miniature hermes-agent checkout with a suite that exercises the tool."""
    (root / "tools").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "tools" / "__init__.py").write_text("")
    (root / "tools" / "file_tools.py").write_text(BASELINE)
    (root / "tests" / "test_file_tools.py").write_text(SUITE)

    git(root.parent, "-c", "init.defaultBranch=main", "init", "-q", str(root))
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test Runner")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    return root


class FakeEvolver:
    """Stands in for the Darwinian Evolver CLI without running anything."""

    def __init__(self, *sources):
        self.sources = list(sources)

    def propose(self, job):
        return [
            Candidate(index=i, source=source, notes=f"candidate {i}", origin="fake")
            for i, source in enumerate(self.sources, start=1)
        ]


class Recorder:
    """Every attempt to leave this machine, recorded instead of performed."""

    def __init__(self):
        self.pushes = []
        self.opens = []


def install_recorder(monkeypatch, push_error=None, open_error=None) -> Recorder:
    recorder = Recorder()

    def push(self, remote="origin"):
        recorder.pushes.append((self.branch, remote))
        if push_error:
            raise mod.DeploymentGitError(push_error)
        return f"pushed {self.branch}"

    def open_pr(self, base="main"):
        recorder.opens.append((self.branch, base))
        if open_error:
            raise mod.DeploymentGitError(open_error)
        return "https://example.invalid/pull/1"

    monkeypatch.setattr(mod.PullRequestPlan, "push", push)
    monkeypatch.setattr(mod.PullRequestPlan, "open", open_pr)
    return recorder


def run_phase_four(repo, out_root, repro=None, sources=(FIXED,), **kwargs):
    """One pass, and everything the assertions need afterwards."""
    code = evolve_tool_code(
        tool="file_tools",
        hermes_repo=str(repo),
        python=sys.executable,
        evolver=FakeEvolver(*sources),
        output_root=out_root,
        repro_script=str(repro) if repro else None,
        **kwargs, sandbox=UNSANDBOXED,
        )
    metrics_paths = list(Path(out_root).rglob("metrics.json"))
    metrics = json.loads(metrics_paths[0].read_text()) if metrics_paths else None
    bodies = list(Path(out_root).rglob("PULL_REQUEST.md"))
    return SimpleNamespace(
        code=code,
        metrics=metrics,
        body=bodies[0].read_text() if bodies else None,
        body_path=bodies[0] if bodies else None,
        out_dir=metrics_paths[0].parent if metrics_paths else None,
    )


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "hermes-agent")


@pytest.fixture
def repro(tmp_path):
    path = tmp_path / "repro_issue_742.py"
    path.write_text(REPRO)
    return path


@pytest.fixture
def recorder(monkeypatch):
    return install_recorder(monkeypatch)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """One complete run with a winner and two different rejections.

    Module scoped because it drives real git and a real pytest subprocess per
    candidate, and every assertion below reads the same artifacts.
    """
    root = tmp_path_factory.mktemp("full")
    repo = make_repo(root / "hermes-agent")
    repro = root / "repro.py"
    repro.write_text(REPRO)

    with pytest.MonkeyPatch.context() as monkeypatch:
        recorder = install_recorder(monkeypatch)
        result = run_phase_four(
            repo,
            root / "out",
            repro=repro,
            sources=(FIXED, BREAKS_TESTS, UNSAFE),
            bug_issue="742",
            repro_runs=3,
            iterations=4,
        )
    result.repo = repo
    result.recorder = recorder
    return result


# ──────────────────────────────────────────────────────────────────────────
# The body, and the branch it belongs to
# ──────────────────────────────────────────────────────────────────────────


@needs_git
class TestPullRequestBody:
    def test_the_run_completes(self, full_run):
        assert full_run.code == 0
        assert full_run.metrics["winner"] == "c01"

    def test_a_body_is_written_beside_the_other_artifacts(self, full_run):
        assert full_run.body_path is not None
        assert full_run.body_path.name == "PULL_REQUEST.md"
        assert full_run.body_path.parent == full_run.out_dir
        assert (full_run.out_dir / "winner.diff").is_file()

    def test_the_body_belongs_to_the_branch_the_organism_made(self, full_run):
        branch = full_run.metrics["branch"]
        assert branch.startswith("evolve/code/file_tools-")
        assert full_run.metrics["pull_request"]["branch"] == branch
        assert branch in full_run.body

    def test_no_second_branch_is_created(self, full_run):
        """The organism's branch is the only one. A pr_builder branch here
        would mean two mechanisms checking out over one worktree."""
        branches = [
            line.strip(" *")
            for line in git(full_run.repo, "branch", "--list").stdout.splitlines()
        ]
        assert sorted(branches) == sorted(["main", full_run.metrics["branch"]])

    def test_the_body_names_the_phase(self, full_run):
        assert PHASE_LABEL in full_run.body

    def test_the_body_carries_the_winning_diff(self, full_run):
        assert "```diff" in full_run.body
        assert "-            return handle.read().splitlines()[:limit - 1]" in full_run.body
        assert "+            return handle.read().splitlines()[:limit]" in full_run.body

    def test_the_score_never_appears_without_its_evidence_coverage(self, full_run):
        # A reproduction and the quality heuristics ran; no benchmark did.
        assert "composite fitness" in full_run.body
        assert "evidence 70%" in full_run.body
        assert "no benchmark" in full_run.body

    def test_the_scores_are_baseline_versus_winner(self, full_run):
        assert "bug reproduction fix rate" in full_run.body
        assert "test suite pass rate" in full_run.body

    def test_the_statistics_are_the_ones_the_phase_computed(self, full_run):
        assert "McNemar" in full_run.body
        assert "Wilson interval" in full_run.body
        assert "fixed 3/3 run(s)" in full_run.body

    def test_the_body_says_nothing_merges(self, full_run):
        assert REVIEW_NOTE in full_run.body
        assert "human review" in full_run.body

    def test_the_body_tells_a_reviewer_how_to_read_the_diff(self, full_run):
        assert "git diff" in full_run.body
        assert "tools/file_tools.py" in full_run.body


# ──────────────────────────────────────────────────────────────────────────
# Cost
# ──────────────────────────────────────────────────────────────────────────


@needs_git
class TestCostReporting:
    def test_the_body_carries_a_cost_line(self, full_run):
        assert "- Cost:" in full_run.body
        # Nothing configured an LM, so the honest figure is "none recorded"
        # rather than $0.00 presented as a measurement.
        assert "no model calls recorded" in full_run.body

    def test_the_evolver_caveat_travels_with_the_cost(self, full_run):
        assert EVOLVER_COST_NOTE in full_run.body
        assert "Darwinian Evolver runs as a separate subprocess" in full_run.body

    def test_the_metrics_file_carries_the_cost_and_the_caveat(self, full_run):
        assert full_run.metrics["cost"]["calls"] == 0
        assert full_run.metrics["cost"]["known_cost_usd"] == 0
        assert full_run.metrics["cost_excludes"] == EVOLVER_COST_NOTE

    def test_a_measured_cost_is_reported_as_a_figure(self):
        plan = make_plan(cost=CostReport(calls=[LMCall("gpt", 100, 50, 0.02)]))
        assert "$0.0200" in plan.body
        assert EVOLVER_COST_NOTE in plan.body

    def test_an_unpriced_call_is_never_summed_as_zero(self):
        plan = make_plan(
            cost=CostReport(calls=[LMCall("local-model", 100, 50, None)])
        )
        assert "at least $" in plan.body
        assert "no price available" in plan.body


# ──────────────────────────────────────────────────────────────────────────
# Rejected candidates
# ──────────────────────────────────────────────────────────────────────────


@needs_git
class TestRejectedCandidates:
    def test_both_rejections_reach_the_body(self, full_run):
        assert "Rejected along the way" in full_run.body
        assert "2 candidate(s) were produced and refused" in full_run.body
        assert "`c02`" in full_run.body
        assert "`c03`" in full_run.body

    def test_a_safety_rejection_carries_the_guardrails_own_words(self, full_run):
        assert "refused by the safety guardrails" in full_run.body
        assert "signatures_frozen" in full_run.body
        assert "encoding" in full_run.body

    def test_a_pytest_rejection_says_the_hard_gate_caught_it(self, full_run):
        assert "refused by the hard pytest gate" in full_run.body

    def test_the_pytest_rejection_names_the_test_that_broke(self, full_run):
        assert "test_a_missing_file_reads_as_empty" in full_run.body

    def test_an_accepted_candidate_is_not_listed_as_rejected(self, full_run):
        rejected_section = full_run.body.split("Rejected along the way")[1]
        assert "`c01`" not in rejected_section.split("## Run")[0]


# ──────────────────────────────────────────────────────────────────────────
# Nothing leaves this machine unless it was asked for
# ──────────────────────────────────────────────────────────────────────────


@needs_git
class TestNothingLeavesTheMachine:
    def test_nothing_is_pushed_by_default(self, full_run):
        assert full_run.recorder.pushes == []

    def test_no_pr_is_opened_by_default(self, full_run):
        assert full_run.recorder.opens == []

    def test_the_test_repo_has_no_remote_at_all(self, full_run):
        assert git(full_run.repo, "remote").stdout.strip() == ""

    def test_the_console_says_nothing_was_sent(self, repo, repro, tmp_path, recorder, capsys):
        run_phase_four(repo, tmp_path / "out", repro=repro)
        printed = " ".join(capsys.readouterr().out.split())
        assert "Nothing was pushed and no PR was opened" in printed

    def test_a_push_happens_only_when_asked_for(self, repo, repro, tmp_path, recorder):
        result = run_phase_four(repo, tmp_path / "out", repro=repro, push=True)
        assert result.code == 0
        assert recorder.pushes == [(result.metrics["branch"], "origin")]
        assert recorder.opens == []

    def test_a_pr_is_opened_only_when_asked_for(self, repo, repro, tmp_path, recorder):
        result = run_phase_four(repo, tmp_path / "out", repro=repro, open_pr=True)
        assert result.code == 0
        assert recorder.opens == [(result.metrics["branch"], "main")]
        assert recorder.pushes == []

    def test_the_remote_and_base_are_honoured(self, repo, repro, tmp_path, recorder):
        result = run_phase_four(
            repo,
            tmp_path / "out",
            repro=repro,
            push=True,
            open_pr=True,
            remote="upstream",
            base="develop",
        )
        branch = result.metrics["branch"]
        assert recorder.pushes == [(branch, "upstream")]
        assert recorder.opens == [(branch, "develop")]

    def test_a_failed_push_is_reported_and_changes_the_exit_code(
        self, repo, repro, tmp_path, monkeypatch
    ):
        install_recorder(monkeypatch, push_error="no configured remote 'origin'")
        result = run_phase_four(repo, tmp_path / "out", repro=repro, push=True)
        assert result.code == 4
        # The run still produced its artifacts; only the send failed.
        assert result.body is not None

    def test_a_failed_pr_open_is_reported_and_changes_the_exit_code(
        self, repo, repro, tmp_path, monkeypatch
    ):
        install_recorder(monkeypatch, open_error="gh is not installed")
        result = run_phase_four(repo, tmp_path / "out", repro=repro, open_pr=True)
        assert result.code == 4

    def test_the_branch_is_still_restored_after_a_failed_push(
        self, repo, repro, tmp_path, monkeypatch
    ):
        install_recorder(monkeypatch, push_error="no configured remote 'origin'")
        run_phase_four(repo, tmp_path / "out", repro=repro, push=True)
        head = git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        assert head == "main"

    def test_pushing_without_a_body_still_pushes_nothing_extra(
        self, repo, repro, tmp_path, recorder
    ):
        result = run_phase_four(
            repo, tmp_path / "out", repro=repro, write_pr=False, push=True
        )
        assert result.body is None
        assert recorder.pushes == [(result.metrics["branch"], "origin")]


# ──────────────────────────────────────────────────────────────────────────
# When a body should not be built at all
# ──────────────────────────────────────────────────────────────────────────


@needs_git
class TestNothingToDeploy:
    def test_no_body_when_the_operator_declined_one(self, repo, repro, tmp_path, recorder):
        result = run_phase_four(repo, tmp_path / "out", repro=repro, write_pr=False)
        assert result.code == 0
        assert result.body is None
        assert result.metrics["pull_request"] is None
        # The rest of the run is untouched.
        assert result.metrics["winner"] == "c01"
        assert (result.out_dir / "winner.diff").is_file()

    def test_no_body_when_no_candidate_survived(self, repo, repro, tmp_path, recorder):
        result = run_phase_four(
            repo, tmp_path / "out", repro=repro, sources=(UNSAFE, BREAKS_TESTS)
        )
        assert result.code == 0
        assert result.metrics["winner"] is None
        assert result.metrics["pull_request"] is None
        assert result.body is None

    def test_no_body_when_the_winner_changed_nothing(self, repo, tmp_path, recorder):
        """An unchanged candidate is refused, so there is no winner and no
        document implying a change that does not exist."""
        result = run_phase_four(repo, tmp_path / "out", sources=(BASELINE,))
        assert result.metrics["winner"] is None
        assert result.body is None

    def test_a_dry_run_builds_nothing(self, repo, repro, tmp_path):
        stub = tmp_path / "stub-evolver"
        stub.write_text("#!/bin/sh\n")
        out_root = tmp_path / "out"
        code = evolve_tool_code(
            tool="file_tools",
            repro_script=str(repro),
            hermes_repo=str(repo),
            evolver_cmd=str(stub),
            dry_run=True,
            output_root=out_root, sandbox=UNSANDBOXED,
        )
        assert code == 0
        assert not out_root.exists()
        assert git(repo, "branch", "--list").stdout.strip() == "* main"


# ──────────────────────────────────────────────────────────────────────────
# build_code_pull_request on its own
# ──────────────────────────────────────────────────────────────────────────


def make_fitness(**kwargs) -> CodeFitness:
    defaults = dict(
        label="c01",
        accepted=True,
        total=0.8,
        safety=SafetyReport(results=[]),
        quality=QualitySignals(score=0.9),
        pytest_result=GateResult("pytest", GateStatus.PASSED, "42 passed in 3s"),
        evidence_coverage=0.2,
        missing_evidence=["bug_fix", "benchmark"],
    )
    defaults.update(kwargs)
    return CodeFitness(**defaults)


def make_plan(**kwargs):
    fitness = kwargs.pop("fitness", None) or make_fitness()
    outcome = CandidateOutcome(
        Candidate(index=1, source="after\n", notes="", origin="fake"), fitness, None
    )
    baseline = BaselineSnapshot(
        source="before\n",
        pytest_result=GateResult("pytest", GateStatus.PASSED, "42 passed in 3s"),
    )
    params = dict(
        repo=Path("/nowhere/hermes-agent"),
        branch="evolve/code/file_tools-20260731-010203",
        target="tools/file_tools.py",
        baseline=baseline,
        winner=outcome,
        diff="--- a\n+++ b\n",
        outcomes=[outcome],
        iterations=7,
    )
    params.update(kwargs)
    return build_code_pull_request(**params)


class TestBuildCodePullRequest:
    def test_the_plan_is_bound_to_the_branch_it_was_given(self):
        plan = make_plan()
        assert plan.branch == "evolve/code/file_tools-20260731-010203"

    def test_the_plan_does_not_claim_the_branch_as_its_own(self):
        """The organism created it and the organism restores it. A plan that
        thought it owned the branch would check out over that."""
        plan = make_plan()
        assert plan.created_branch is False
        assert plan.original_ref == ""
        # restore() is a no-op in that state, so it cannot fight the organism
        # even though the repo path here does not exist.
        assert plan.restore() is None

    def test_the_title_names_the_target(self):
        assert make_plan().title == "evolve: tools/file_tools.py"

    def test_the_title_names_the_issue_when_there_is_one(self):
        assert "(issue 742)" in make_plan(bug_issue="742").title

    def test_the_body_is_written_as_pull_request_md(self, tmp_path):
        path = make_plan().write_body(tmp_path)
        assert path == tmp_path / "PULL_REQUEST.md"
        assert path.read_text().startswith("Evolved `tools/file_tools.py`")

    def test_a_thin_score_says_how_thin_it_is(self):
        body = make_plan().body
        assert "evidence 20%" in body
        assert "no bug_fix or benchmark" in body

    def test_a_run_without_a_reproduction_says_nothing_was_proved(self):
        body = make_plan().body
        assert "Reproduction: none measured" in body

    def test_a_run_without_a_paired_suite_says_so_rather_than_no_change(self):
        body = make_plan().body
        assert "Paired test suite: not available" in body
        assert "not the same as no change" in body

    def test_the_files_are_the_one_file_that_moved(self):
        assert make_plan().files == ("tools/file_tools.py",)

    def test_the_optimizer_is_named_as_an_external_subprocess(self):
        body = make_plan().body
        assert "Darwinian Evolver, driven as an external CLI subprocess" in body
        assert "7 iteration(s)" in body

    def test_the_review_note_is_always_present(self):
        assert REVIEW_NOTE in make_plan().body

    def test_the_cost_caveat_is_present_even_with_no_cost_at_all(self):
        assert EVOLVER_COST_NOTE in make_plan(cost=None).body

    def test_a_rejected_candidate_with_no_recorded_reason_still_appears(self):
        rejected = make_fitness(label="c02", accepted=False, rejection_reason=None)
        outcome = CandidateOutcome(
            Candidate(index=2, source="x\n"), rejected, None
        )
        lines = code_rejected_candidates([outcome])
        assert lines[0].label == "c02"
        assert lines[0].reason == "refused, no reason recorded"

    def test_the_dataset_line_names_what_was_evaluated(self):
        baseline = BaselineSnapshot(
            source="before\n",
            pytest_result=GateResult("pytest", GateStatus.PASSED, "ok"),
            test_outcomes={"tests/test_a.py::test_one": True},
        )
        body = make_plan(baseline=baseline, repro_script="/tmp/repro_742.py",
                         repro_runs=5).body
        assert "hermes-agent pytest suite (1 test(s) recorded at baseline)" in body
        assert "reproduction repro_742.py x5 per candidate" in body

    def test_a_table_cell_cannot_break_the_table(self):
        """A gate message with a pipe in it must not add a column."""
        fitness = make_fitness(
            benchmark_results=[
                GateResult("swe-bench", GateStatus.PASSED, "12 | 20 tasks", score=0.6,
                           baseline=0.5)
            ]
        )
        outcome = CandidateOutcome(Candidate(index=1, source="x\n"), fitness, None)
        baseline = BaselineSnapshot(
            source="y\n",
            pytest_result=GateResult("pytest", GateStatus.PASSED, "ok"),
        )
        rows = code_score_lines(baseline, outcome)
        benchmark_row = [r for r in rows if r.split == "swe-bench"][0]
        assert "\\|" in benchmark_row.detail
        # Six structural pipes (five cells, closed) plus the escaped one in
        # the cell text. An unescaped pipe would raise the count and add a
        # phantom column.
        assert benchmark_row.row().count("|") == 7
        assert benchmark_row.row().endswith(" |")


# ──────────────────────────────────────────────────────────────────────────
# CLI surface
# ──────────────────────────────────────────────────────────────────────────


STUB_EVOLVER = '''import argparse
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

job = json.loads(pathlib.Path(args.job).read_text())
out = pathlib.Path(args.output)
(out / "candidates").mkdir(parents=True, exist_ok=True)
(out / "candidates" / "001.py").write_text(
    job["source"].replace("[:limit - 1]", "[:limit]")
)
'''


@needs_git
class TestCli:
    def stub(self, tmp_path):
        script = tmp_path / "stub_evolver.py"
        script.write_text(STUB_EVOLVER)
        return f"{sys.executable} {script}"

    def test_the_flags_are_documented(self):
        from click.testing import CliRunner

        output = CliRunner().invoke(mod.main, ["--help"]).output
        assert "--write-pr / --no-write-pr" in output
        assert "--push / --no-push" in output
        assert "--open-pr / --no-open-pr" in output

    def test_a_dry_run_says_it_would_send_nothing(self, repo, tmp_path):
        from click.testing import CliRunner

        result = CliRunner().invoke(
            mod.main,
            [
                "--tool", "file_tools",
                "--hermes-repo", str(repo),
                "--evolver-cmd", self.stub(tmp_path),
                "--dry-run",
            ],
        )
        printed = " ".join(result.output.split())
        assert result.exit_code == 0
        assert "Would not push" in printed
        assert "would not open a PR" in printed
        assert "A dry run builds nothing" in printed

    def test_a_dry_run_reports_the_flags_it_was_given(self, repo, tmp_path):
        from click.testing import CliRunner

        result = CliRunner().invoke(
            mod.main,
            [
                "--tool", "file_tools",
                "--hermes-repo", str(repo),
                "--evolver-cmd", self.stub(tmp_path),
                "--no-write-pr",
                "--push",
                "--open-pr",
                "--dry-run",
            ],
        )
        printed = " ".join(result.output.split())
        assert "Would not write a PR body" in printed
        assert "Would push to origin" in printed
        assert "would open a PR against main" in printed

    def test_the_default_cli_run_writes_a_body_and_sends_nothing(
        self, repo, repro, tmp_path, monkeypatch, recorder
    ):
        from click.testing import CliRunner

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(
            mod.main,
            [
                "--tool", "file_tools",
                "--hermes-repo", str(repo),
                "--repro-script", str(repro),
                "--evolver-cmd", self.stub(tmp_path),
                "--python", sys.executable,
            ],
        )
        assert result.exit_code == 0
        bodies = list((tmp_path / "output").rglob("PULL_REQUEST.md"))
        assert len(bodies) == 1
        assert PHASE_LABEL in bodies[0].read_text()
        assert recorder.pushes == []
        assert recorder.opens == []

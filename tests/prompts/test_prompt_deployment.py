"""Tests for what Phase 3 does after it decides: cost, branch, pull request.

Offline throughout, like the rest of this phase's tests. The optimizer, the
harness and the gate ladder are stubbed, dspy's call log is replaced with a
plain list this module appends to so a run can be made to spend money without a
model, and the git behaviour runs against real repositories in tmp_path because
the branch handling is the part that can damage someone's checkout.

Nothing here reaches the network. ``PullRequestPlan.push`` and
``PullRequestPlan.open`` are replaced with recorders by default, so a test can
assert that a push was attempted - and every other test fails loudly if one is
attempted that should not have been.
"""

import json
import subprocess

import dspy
import pytest
from click.testing import CliRunner

from evolution.core.gates import GateChain, GateResult, GateStatus
from evolution.core.pr_builder import GitError, PullRequestPlan
from evolution.prompts import evolve_prompt_section as ep
from evolution.prompts.behavioral_eval import (
    BehavioralOutcome,
    BehavioralReport,
    BehavioralSuite,
)
from evolution.prompts.sections import ActiveSessionReport
from tests.prompts.test_evolve_prompt_section import (
    EVOLVED_MARKER,
    PROMPT_SOURCE,
    flat,
    holdout_reports,
    same_length_optimize,
)


# ──────────────────────────────────────────────────────────────────────────
# Scaffolding
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def restore_dspy_settings():
    """evolve() configures a global default LM. Put the old one back."""
    previous = dspy.settings.lm
    yield
    dspy.configure(lm=previous)


def git(repo, *args) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(repo), check=True, capture_output=True, text=True
    ).stdout


def head(repo) -> str:
    return git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()


def branches(repo) -> list[str]:
    return [
        line.strip()
        for line in git(repo, "branch", "--format=%(refname:short)").splitlines()
        if line.strip()
    ]


def evolve_branch(repo) -> str:
    found = [b for b in branches(repo) if b.startswith("evolve/")]
    assert len(found) == 1, f"expected one evolve branch, found {found}"
    return found[0]


def _checkout(root):
    (root / "agent").mkdir(parents=True)
    (root / "agent" / "prompt_builder.py").write_text(PROMPT_SOURCE, encoding="utf-8")
    (root / "batch_runner.py").write_text("# fake runner\n", encoding="utf-8")
    return root


@pytest.fixture
def hermes(tmp_path):
    """A hermes-agent checkout that is a real git repo with one commit on main."""
    root = _checkout(tmp_path / "hermes-agent")
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "tester@example.com")
    git(root, "config", "user.name", "Tester")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "base")
    return root


@pytest.fixture
def plain(tmp_path):
    """The same checkout with no git repository anywhere in it."""
    return _checkout(tmp_path / "loose-checkout")


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """Working directory for output/, deliberately outside the repo under test."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def latest_run(run_dir):
    return sorted((run_dir / "output" / "prompts").iterdir())[-1]


def pr_body(run_dir) -> str:
    return (latest_run(run_dir) / "PULL_REQUEST.md").read_text(encoding="utf-8")


def metrics_of(run_dir) -> dict:
    return json.loads((latest_run(run_dir) / "metrics.json").read_text(encoding="utf-8"))


def priced_call(model="openai/gpt-4.1", prompt=1200, completion=300, cost=0.004, **extra):
    """One dspy history entry in the shape dspy records."""
    blob = {
        "model": model,
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        "cost": cost,
    }
    blob.update(extra)
    return blob


@pytest.fixture
def history(monkeypatch):
    """Stand in for dspy's global call log."""
    log: list = []
    monkeypatch.setattr("evolution.core.cost._history", lambda: log)
    return log


def _no_network(name):
    def refuse(self, *args, **kwargs):
        raise AssertionError(f"a test reached the network via PullRequestPlan.{name}")

    return refuse


@pytest.fixture
def stubbed(monkeypatch):
    """Replace everything that would need a model, a harness or a benchmark."""

    def fake_evaluate(
        self, system_prompt, harness, judge=None, section_name="", run_name="", scenarios=None
    ):
        targets = list(scenarios if scenarios is not None else self.scenarios)
        score = 0.9 if EVOLVED_MARKER in system_prompt else 0.4
        return BehavioralReport(
            outcomes=[
                BehavioralOutcome(
                    s.scenario_id, s.category, s.section_under_test, score, score >= 0.6
                )
                for s in targets
            ],
            harness="stub",
        )

    def fake_gates(**kwargs):
        return GateChain(strict=kwargs.get("strict", False)).run(
            GateResult(name="pytest", status=GateStatus.PASSED, message="12 passed"),
            GateResult(
                name="tblite",
                status=GateStatus.UNAVAILABLE,
                message="no benchmark in this checkout",
            ),
        )

    monkeypatch.setattr(ep, "_optimize_section", same_length_optimize)
    monkeypatch.setattr(BehavioralSuite, "evaluate", fake_evaluate)
    monkeypatch.setattr(ep, "run_gate_ladder", fake_gates)
    monkeypatch.setattr(
        ep, "detect_active_session", lambda *a, **k: ActiveSessionReport(active=False)
    )
    monkeypatch.setattr(PullRequestPlan, "push", _no_network("push"))
    monkeypatch.setattr(PullRequestPlan, "open", _no_network("open"))
    return monkeypatch


@pytest.fixture
def spending(stubbed, history):
    """An optimizer stub that records model calls on its way through."""

    def fake_optimize(
        section_name, baseline_text, trainset, valset, iterations, optimizer_model
    ):
        history.append(priced_call())
        history.append(priced_call(model="openai/gpt-4.1-mini", cost=0.0005))
        return same_length_optimize(
            section_name, baseline_text, trainset, valset, iterations, optimizer_model
        )

    stubbed.setattr(ep, "_optimize_section", fake_optimize)
    return history


def outcome_with_holdout(name, deltas, base=0.40, adjusted_p=None, category="memory_guidance"):
    """A SectionOutcome carrying a real paired comparison over hand-made scores."""
    categories = [category] * len(deltas)
    baseline, candidate = holdout_reports(
        categories, [base] * len(deltas), [base + d for d in deltas]
    )
    comparison = ep.compare_holdout(baseline, candidate, targeted_category=category)
    return ep.SectionOutcome(
        name=name,
        baseline_text="before",
        evolved_text="after",
        holdout=comparison,
        holdout_baseline=comparison.overall.baseline_mean,
        holdout_evolved=comparison.overall.candidate_mean,
        accepted=True,
        reason=comparison.reason,
        adjusted_p=adjusted_p,
        optimizer="GEPA",
    )


# ──────────────────────────────────────────────────────────────────────────
# Cost
# ──────────────────────────────────────────────────────────────────────────


class TestCost:
    def test_the_run_reports_what_it_spent(self, hermes, stubbed, spending, run_dir, capsys):
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes))
        output = flat(capsys.readouterr().out)

        assert "Cost" in output
        assert "2 call(s)" in output
        assert "$0.0045" in output
        assert "3,000 tokens" in output

    def test_the_cost_is_saved_with_the_run(self, hermes, stubbed, spending, run_dir):
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes))
        cost = metrics_of(run_dir)["cost"]

        assert cost["calls"] == 2
        assert cost["known_cost_usd"] == pytest.approx(0.0045)
        assert cost["total_tokens"] == 3000
        assert cost["complete"] is True
        assert cost["models"] == {"openai/gpt-4.1": 1, "openai/gpt-4.1-mini": 1}

    def test_only_the_calls_this_run_made_are_counted(
        self, hermes, stubbed, spending, run_dir
    ):
        """An earlier run's entries are still in dspy's log. They are not ours."""
        spending.insert(0, priced_call(cost=99.0))
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes))

        assert metrics_of(run_dir)["cost"]["known_cost_usd"] == pytest.approx(0.0045)

    def test_an_unpriced_call_is_not_quietly_counted_as_free(
        self, hermes, stubbed, history, run_dir, capsys
    ):
        def fake_optimize(
            section_name, baseline_text, trainset, valset, iterations, optimizer_model
        ):
            history.append(priced_call(cost=0.004))
            history.append(priced_call(model="local/hermes-4", cost=None))
            return same_length_optimize(
                section_name, baseline_text, trainset, valset, iterations, optimizer_model
            )

        stubbed.setattr(ep, "_optimize_section", fake_optimize)
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes))
        output = flat(capsys.readouterr().out)
        cost = metrics_of(run_dir)["cost"]

        assert "at least $0.0040" in output
        assert "1 call(s) with no price available" in output
        assert "lower bound" in output
        assert cost["unpriced_calls"] == 1
        assert cost["known_cost_usd"] == pytest.approx(0.004)
        assert cost["complete"] is False

    def test_a_run_that_spent_nothing_says_so(self, hermes, stubbed, history, run_dir, capsys):
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes))

        assert "no model calls recorded" in flat(capsys.readouterr().out)
        assert metrics_of(run_dir)["cost"]["calls"] == 0

    def test_the_cost_reaches_the_pr_body(self, hermes, stubbed, spending, run_dir):
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), write=True)

        assert "- Cost: 2 call(s)" in pr_body(run_dir)
        assert "$0.0045" in pr_body(run_dir)

    def test_an_incomplete_cost_stays_incomplete_in_the_pr_body(
        self, hermes, stubbed, history, run_dir
    ):
        def fake_optimize(
            section_name, baseline_text, trainset, valset, iterations, optimizer_model
        ):
            history.append(priced_call(model="local/hermes-4", cost=None))
            return same_length_optimize(
                section_name, baseline_text, trainset, valset, iterations, optimizer_model
            )

        stubbed.setattr(ep, "_optimize_section", fake_optimize)
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), write=True)

        assert "at least $" in pr_body(run_dir)
        assert "no price available" in pr_body(run_dir)


# ──────────────────────────────────────────────────────────────────────────
# A pull request only exists when something was deployed
# ──────────────────────────────────────────────────────────────────────────


class TestBuiltOnlyOnAWrite:
    def test_no_write_means_no_branch(self, hermes, stubbed, run_dir):
        code = ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes))

        assert code == 0
        assert branches(hermes) == ["main"]
        assert not (latest_run(run_dir) / "PULL_REQUEST.md").exists()
        assert metrics_of(run_dir)["pull_request"] is None

    def test_a_dry_run_builds_nothing(self, hermes, stubbed, run_dir):
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), dry_run=True
        )

        assert code == 0
        assert branches(hermes) == ["main"]
        assert not (run_dir / "output").exists()

    def test_a_dry_run_says_it_is_building_nothing(self, hermes, stubbed, run_dir, capsys):
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), dry_run=True)
        output = flat(capsys.readouterr().out)

        assert "a dry run builds neither" in output
        assert "nothing is ever pushed without --push" in output

    def test_a_refused_candidate_leaves_no_branch(self, hermes, stubbed, run_dir):
        """Nothing cleared the holdout, so --write writes nothing and branches nothing."""
        stubbed.setattr(
            ep,
            "_optimize_section",
            lambda section_name, baseline_text, **kw: (baseline_text * 4, "stub"),
        )
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), write=True
        )

        assert code == 0
        assert branches(hermes) == ["main"]
        assert not (latest_run(run_dir) / "PULL_REQUEST.md").exists()

    def test_a_write_builds_the_branch_and_the_body(self, hermes, stubbed, run_dir):
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), write=True
        )

        assert code == 0
        branch = evolve_branch(hermes)
        assert branch.startswith("evolve/MEMORY_GUIDANCE-")
        assert (latest_run(run_dir) / "PULL_REQUEST.md").exists()
        assert metrics_of(run_dir)["pull_request"]["branch"] == branch

    def test_several_sections_share_one_slug(self, hermes, stubbed, run_dir):
        ep.evolve(
            section_names=["MEMORY_GUIDANCE", "SKILLS_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
        )

        assert evolve_branch(hermes).startswith(f"evolve/{ep.MULTI_SECTION_TARGET}-")

    def test_the_change_lands_on_the_branch_and_the_checkout_comes_back(
        self, hermes, stubbed, run_dir
    ):
        """The commit belongs to the branch. The reviewer keeps their checkout."""
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), write=True)
        branch = evolve_branch(hermes)

        assert EVOLVED_MARKER in git(hermes, "show", f"{branch}:agent/prompt_builder.py")
        assert head(hermes) == "main"
        assert EVOLVED_MARKER not in (hermes / "agent" / "prompt_builder.py").read_text(
            encoding="utf-8"
        )
        assert git(hermes, "status", "--porcelain").strip() == ""

    def test_the_commit_message_carries_the_holdout_numbers(
        self, hermes, stubbed, run_dir
    ):
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), write=True)
        message = git(hermes, "log", "-1", "--pretty=%B", evolve_branch(hermes))

        assert "evolve: MEMORY_GUIDANCE" in message
        assert "holdout: 0.400 -> 0.900" in message
        assert "Cost:" in message

    def test_no_create_pr_leaves_the_write_in_the_working_tree(
        self, hermes, stubbed, run_dir, capsys
    ):
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
            create_pr=False,
        )

        assert code == 0
        assert branches(hermes) == ["main"]
        assert EVOLVED_MARKER in (hermes / "agent" / "prompt_builder.py").read_text(
            encoding="utf-8"
        )
        assert "No pull request requested" in flat(capsys.readouterr().out)

    def test_a_checkout_without_git_is_reported_not_fatal(self, plain, stubbed, run_dir, capsys):
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(plain), write=True
        )
        output = flat(capsys.readouterr().out)

        assert code == 0
        assert "not a git repository" in output
        assert EVOLVED_MARKER in (plain / "agent" / "prompt_builder.py").read_text(
            encoding="utf-8"
        )


# ──────────────────────────────────────────────────────────────────────────
# Nothing leaves the machine unless it was asked for by name
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def recorded_push(stubbed):
    """Record push and open calls instead of making them."""
    calls: dict[str, list] = {"push": [], "open": []}

    def fake_push(self, remote="origin"):
        calls["push"].append(remote)
        return ""

    def fake_open(self, base="main"):
        calls["open"].append(base)
        return "https://example.invalid/pr/1"

    stubbed.setattr(PullRequestPlan, "push", fake_push)
    stubbed.setattr(PullRequestPlan, "open", fake_open)
    return calls


class TestNetworkIsOptIn:
    def test_a_default_write_pushes_nothing_and_opens_nothing(
        self, hermes, stubbed, recorded_push, run_dir, capsys
    ):
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), write=True
        )
        output = flat(capsys.readouterr().out)

        assert code == 0
        assert recorded_push == {"push": [], "open": []}
        assert "Not pushed (--no-push is the default)" in output
        assert "No pull request opened (--no-open-pr is the default)" in output

    def test_push_happens_only_when_asked(
        self, hermes, stubbed, recorded_push, run_dir
    ):
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
            push=True,
        )

        assert code == 0
        assert recorded_push["push"] == ["origin"]
        assert recorded_push["open"] == []

    def test_opening_the_pr_uses_the_requested_base(
        self, hermes, stubbed, recorded_push, run_dir
    ):
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
            push=True,
            open_pr=True,
            pr_base="develop",
        )

        assert code == 0
        assert recorded_push["push"] == ["origin"]
        assert recorded_push["open"] == ["develop"]

    def test_open_pr_without_push_is_refused_before_anything_runs(
        self, hermes, stubbed, run_dir
    ):
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
            open_pr=True,
        )

        assert code == 1
        assert not (run_dir / "output").exists()
        assert branches(hermes) == ["main"]

    def test_push_without_write_is_refused_before_anything_runs(
        self, hermes, stubbed, run_dir, capsys
    ):
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), push=True
        )

        assert code == 1
        assert "need --write" in flat(capsys.readouterr().out)
        assert not (run_dir / "output").exists()

    def test_push_with_no_create_pr_is_refused_before_anything_runs(
        self, hermes, stubbed, run_dir, capsys
    ):
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
            create_pr=False,
            push=True,
        )

        assert code == 1
        assert "no branch to push" in flat(capsys.readouterr().out)
        assert not (run_dir / "output").exists()

    def test_a_failed_push_is_reported_and_the_checkout_still_comes_back(
        self, hermes, stubbed, run_dir, capsys
    ):
        def refuse(self, remote="origin"):
            raise GitError("git push failed: no configured remote named 'origin'")

        stubbed.setattr(PullRequestPlan, "push", refuse)
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
            push=True,
        )
        output = flat(capsys.readouterr().out)

        assert code == ep.EXIT_DEPLOYMENT_INCOMPLETE
        assert "Push failed" in output
        assert "nothing reached the remote" in output
        assert head(hermes) == "main"
        assert EVOLVED_MARKER in git(
            hermes, "show", f"{evolve_branch(hermes)}:agent/prompt_builder.py"
        )

    def test_a_failed_push_does_not_go_on_to_open_a_pr(
        self, hermes, stubbed, recorded_push, run_dir
    ):
        def refuse(self, remote="origin"):
            raise GitError("no configured remote")

        stubbed.setattr(PullRequestPlan, "push", refuse)
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
            push=True,
            open_pr=True,
        )

        assert code == ep.EXIT_DEPLOYMENT_INCOMPLETE
        assert recorded_push["open"] == []

    def test_a_missing_gh_is_reported_without_losing_the_branch(
        self, hermes, stubbed, recorded_push, run_dir, capsys
    ):
        def refuse(self, base="main"):
            raise GitError("gh is not installed, so the PR cannot be opened from here.")

        stubbed.setattr(PullRequestPlan, "open", refuse)
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
            push=True,
            open_pr=True,
        )
        output = flat(capsys.readouterr().out)

        assert code == ep.EXIT_DEPLOYMENT_INCOMPLETE
        assert "Could not open the pull request" in output
        assert "gh is not installed" in output
        assert head(hermes) == "main"
        assert evolve_branch(hermes)


class TestRefIsAlwaysRestored:
    def test_an_error_while_writing_the_body_still_restores_the_ref(
        self, hermes, stubbed, run_dir
    ):
        def explode(self, output_dir):
            raise OSError("disk full")

        stubbed.setattr(PullRequestPlan, "write_body", explode)
        with pytest.raises(OSError):
            ep.evolve(
                section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), write=True
            )

        assert head(hermes) == "main"
        assert EVOLVED_MARKER in git(
            hermes, "show", f"{evolve_branch(hermes)}:agent/prompt_builder.py"
        )

    def test_a_restore_that_fails_says_where_you_are(self, hermes, stubbed, run_dir, capsys):
        def refuse(self):
            raise GitError("git checkout main failed: local changes would be overwritten")

        stubbed.setattr(PullRequestPlan, "restore", refuse)
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), write=True
        )
        output = flat(capsys.readouterr().out)

        assert code == ep.EXIT_DEPLOYMENT_INCOMPLETE
        assert "Could not restore the checkout" in output
        assert "You are still on evolve/MEMORY_GUIDANCE-" in output


# ──────────────────────────────────────────────────────────────────────────
# What the body has to say
# ──────────────────────────────────────────────────────────────────────────


class TestPullRequestBody:
    @pytest.fixture
    def body(self, hermes, stubbed, spending, run_dir):
        ep.evolve(section_names=["MEMORY_GUIDANCE"], hermes_repo=str(hermes), write=True)
        return pr_body(run_dir)

    def test_it_names_the_phase_and_the_section(self, body):
        assert "MEMORY_GUIDANCE" in body
        assert ep.PR_PHASE in body

    def test_it_carries_the_holdout_scores(self, body):
        assert "| holdout | 0.400 | 0.900 | +0.500" in body
        assert "paired, n=" in body

    def test_it_carries_the_paired_evidence_the_comparison_already_rendered(self, body):
        assert "Wilcoxon signed-rank" in body
        assert "95% CI" in body
        assert "d=+" in body
        assert "memory_guidance (targeted)" in body
        assert "[held]" in body
        assert "- Power:" in body
        assert "- Verdict:" in body

    def test_it_carries_the_gate_ladder_including_what_could_not_run(self, body):
        assert "pytest (passed): 12 passed" in body
        assert "tblite (unavailable): no benchmark in this checkout" in body

    def test_it_carries_the_diff(self, body):
        assert "```diff" in body
        assert EVOLVED_MARKER in body

    def test_it_states_the_next_session_deployment_rule(self, body):
        assert "NEXT session" in body
        assert "nothing is hot-swapped" in body

    def test_it_names_the_dataset_the_harness_and_the_artifacts(self, body):
        assert "Eval dataset: behavioural suite" in body
        assert "train /" in body
        assert "Evaluation harness:" in body
        assert "Run artifacts" in body

    def test_it_names_the_optimizer_and_the_iterations(self, hermes, stubbed, run_dir):
        ep.evolve(
            section_names=["MEMORY_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
            iterations=3,
        )
        assert "- Optimizer: stub, 3 iteration(s)" in pr_body(run_dir)


# ──────────────────────────────────────────────────────────────────────────
# Rejected candidates, including the ones Holm dropped
# ──────────────────────────────────────────────────────────────────────────


class TestRejectedCandidatesReachTheBody:
    def test_a_section_dropped_by_the_holm_correction_is_listed_as_a_near_miss(
        self, hermes, stubbed, run_dir, capsys
    ):
        """One section clears alpha alone, then loses to the correction.

        holm_adjust is pinned rather than coaxed: the point of the test is that
        a section the correction dropped is visible to the reviewer, not that
        this particular pair of stub scores lands either side of 0.05.
        """
        stubbed.setattr(ep, "holm_adjust", lambda ps: [0.001, 0.400])
        code = ep.evolve(
            section_names=["MEMORY_GUIDANCE", "SKILLS_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
        )
        body = pr_body(run_dir)
        output = flat(capsys.readouterr().out)

        assert code == 0
        assert "multiple-comparison correction" in output
        assert "Rejected along the way" in body
        assert "`SKILLS_GUIDANCE`" in body
        assert "dropped by the Holm correction" in body
        assert "after correcting for 2 sections tested against one baseline" in body
        # The section that survived the correction is what shipped.
        branch = evolve_branch(hermes)
        assert branch.startswith("evolve/MEMORY_GUIDANCE-")
        on_branch = git(hermes, "show", f"{branch}:agent/prompt_builder.py")
        assert EVOLVED_MARKER in on_branch.split("SKILLS_GUIDANCE")[0]
        assert "evolve: MEMORY_GUIDANCE" in git(
            hermes, "log", "-1", "--pretty=%B", branch
        )

    def test_the_holm_adjusted_p_is_reported_for_the_section_that_survived(
        self, hermes, stubbed, run_dir
    ):
        stubbed.setattr(ep, "holm_adjust", lambda ps: [0.001, 0.400])
        ep.evolve(
            section_names=["MEMORY_GUIDANCE", "SKILLS_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
        )
        assert "Holm-adjusted p" in pr_body(run_dir)

    def test_a_constraint_failure_is_listed_next_to_the_section_that_shipped(
        self, hermes, stubbed, run_dir
    ):
        def optimize(section_name, baseline_text, trainset, valset, iterations, optimizer_model):
            if section_name == "SKILLS_GUIDANCE":
                return baseline_text * 4, "stub"  # blows the growth ceiling
            return same_length_optimize(
                section_name, baseline_text, trainset, valset, iterations, optimizer_model
            )

        stubbed.setattr(ep, "_optimize_section", optimize)
        ep.evolve(
            section_names=["MEMORY_GUIDANCE", "SKILLS_GUIDANCE"],
            hermes_repo=str(hermes),
            write=True,
        )
        body = pr_body(run_dir)

        assert "`SKILLS_GUIDANCE`: failed constraints" in body
        assert "1 candidate(s) were produced and refused" in body


# ──────────────────────────────────────────────────────────────────────────
# The pieces the body is assembled from
# ──────────────────────────────────────────────────────────────────────────


class TestPrTarget:
    def test_one_section_names_the_branch_after_itself(self):
        assert ep.pr_target(["MEMORY_GUIDANCE"]) == "MEMORY_GUIDANCE"

    def test_the_same_section_twice_is_still_one_section(self):
        assert ep.pr_target(["MEMORY_GUIDANCE", "MEMORY_GUIDANCE"]) == "MEMORY_GUIDANCE"

    def test_several_sections_share_a_stable_slug(self):
        assert (
            ep.pr_target(["MEMORY_GUIDANCE", "SKILLS_GUIDANCE"])
            == ep.MULTI_SECTION_TARGET
        )

    def test_nothing_at_all_still_produces_a_usable_branch_name(self):
        assert ep.pr_target([]) == ep.MULTI_SECTION_TARGET


class TestRepoRelative:
    def test_a_file_inside_the_repo_is_staged_by_its_relative_path(self, tmp_path):
        assert (
            ep.repo_relative(tmp_path, tmp_path / "agent" / "prompt_builder.py")
            == "agent/prompt_builder.py"
        )

    def test_a_file_outside_the_repo_keeps_its_own_path(self, tmp_path):
        outside = tmp_path.parent / "elsewhere.py"
        assert ep.repo_relative(tmp_path, outside) == outside.as_posix()


class TestScoreLines:
    def test_one_row_per_deployed_section_off_the_holdout(self):
        deployed = [
            outcome_with_holdout("MEMORY_GUIDANCE", [0.5] * 8),
            outcome_with_holdout("SKILLS_GUIDANCE", [0.3] * 8),
        ]
        lines = ep.holdout_score_lines(deployed)

        assert [line.split for line in lines] == [
            "holdout / MEMORY_GUIDANCE",
            "holdout / SKILLS_GUIDANCE",
        ]
        assert lines[0].baseline == pytest.approx(0.40)
        assert lines[0].evolved == pytest.approx(0.90)
        assert "8 up / 0 down / 0 unchanged" in lines[0].detail

    def test_a_single_section_does_not_need_its_name_in_the_split(self):
        lines = ep.holdout_score_lines([outcome_with_holdout("MEMORY_GUIDANCE", [0.5] * 8)])
        assert [line.split for line in lines] == ["holdout"]

    def test_a_section_with_no_holdout_contributes_no_row(self):
        bare = ep.SectionOutcome(name="MEMORY_GUIDANCE", baseline_text="a", evolved_text="b")
        assert ep.holdout_score_lines([bare]) == []


class TestStatistics:
    def test_the_comparison_speaks_for_itself(self):
        text = ep.holdout_statistics([outcome_with_holdout("MEMORY_GUIDANCE", [0.5] * 8)])

        assert "**MEMORY_GUIDANCE**" in text
        assert "Wilcoxon signed-rank over 8 holdout scenario(s)" in text
        assert "0.400 -> 0.900" in text
        assert "memory_guidance (targeted)" in text
        assert "- Power:" in text
        assert "- Verdict:" in text

    def test_an_adjusted_p_is_reported_when_one_applies(self):
        text = ep.holdout_statistics(
            [outcome_with_holdout("MEMORY_GUIDANCE", [0.5] * 8, adjusted_p=0.012)]
        )
        assert "Holm-adjusted p" in text
        assert "0.012" in text

    def test_a_section_with_no_holdout_contributes_nothing(self):
        bare = ep.SectionOutcome(name="MEMORY_GUIDANCE", baseline_text="a", evolved_text="b")
        assert ep.holdout_statistics([bare]) == ""


class TestRejectedCandidates:
    def test_the_deployed_section_is_not_in_its_own_rejected_list(self):
        shipped = outcome_with_holdout("MEMORY_GUIDANCE", [0.5] * 8)
        assert ep.rejected_candidates([shipped], [shipped]) == []

    def test_a_refused_section_carries_the_reason_it_was_refused(self):
        refused = ep.SectionOutcome(
            name="SKILLS_GUIDANCE",
            baseline_text="a",
            evolved_text="b",
            reason="failed constraints: size",
        )
        rejected = ep.rejected_candidates([refused], [])

        assert rejected[0].label == "SKILLS_GUIDANCE"
        assert rejected[0].reason == "failed constraints: size"

    def test_a_holm_drop_is_labelled_as_one(self):
        dropped = outcome_with_holdout(
            "SKILLS_GUIDANCE", [0.5] * 8, adjusted_p=0.4
        )
        dropped.reason = "p=0.008 alone, but 0.400 after correcting for 2 sections"
        rejected = ep.rejected_candidates([dropped], [])

        assert "dropped by the Holm correction" in rejected[0].reason
        assert "p=0.008 alone" in rejected[0].reason

    def test_a_section_that_survived_the_correction_is_not_labelled_a_drop(self):
        kept = outcome_with_holdout("MEMORY_GUIDANCE", [0.5] * 8, adjusted_p=0.001)
        kept.reason = "beaten on the tie-break"
        rejected = ep.rejected_candidates([kept], [])

        assert rejected[0].reason == "beaten on the tie-break"

    def test_a_section_with_no_recorded_reason_still_says_something(self):
        bare = ep.SectionOutcome(name="MEMORY_GUIDANCE", baseline_text="a", evolved_text="b")
        assert ep.rejected_candidates([bare], [])[0].reason == "no holdout evidence"


class TestGateLines:
    def test_a_ladder_that_never_ran_says_so(self):
        assert "not reached" in ep.gate_lines(None)[0]

    def test_an_empty_ladder_says_so(self):
        assert ep.gate_lines(GateChain()) == ["no gates ran"]

    def test_every_gate_is_listed_with_its_status(self):
        chain = GateChain().run(
            GateResult(name="pytest", status=GateStatus.PASSED, message="12 passed"),
            GateResult(name="tblite", status=GateStatus.UNAVAILABLE, message="not here"),
        )
        assert ep.gate_lines(chain) == [
            "pytest (passed): 12 passed",
            "tblite (unavailable): not here",
        ]


# ──────────────────────────────────────────────────────────────────────────
# Flags
# ──────────────────────────────────────────────────────────────────────────


class TestFlagDefaults:
    def test_a_pr_is_built_by_default_but_nothing_is_sent(self):
        params = {p.name: p for p in ep.main.params}
        assert params["create_pr"].default is True
        assert params["push"].default is False
        assert params["open_pr"].default is False
        assert params["pr_base"].default == "main"

    def test_the_flags_are_documented_in_the_help(self):
        result = CliRunner().invoke(ep.main, ["--help"])
        assert "--create-pr / --no-create-pr" in flat(result.output)
        assert "--push / --no-push" in flat(result.output)
        assert "--open-pr / --no-open-pr" in flat(result.output)
        assert "--pr-base" in flat(result.output)

    def test_the_cli_carries_the_deployment_flags_into_the_run(self, monkeypatch):
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)
            return 0

        monkeypatch.setattr(ep, "evolve", spy)
        CliRunner().invoke(
            ep.main,
            [
                "--section", "MEMORY_GUIDANCE",
                "--write",
                "--push",
                "--open-pr",
                "--pr-base", "release",
            ],
        )

        assert seen["create_pr"] is True
        assert seen["push"] is True
        assert seen["open_pr"] is True
        assert seen["pr_base"] == "release"

    def test_no_create_pr_reaches_the_run(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(ep, "evolve", lambda **kwargs: seen.update(kwargs) or 0)
        CliRunner().invoke(
            ep.main, ["--section", "MEMORY_GUIDANCE", "--no-create-pr"]
        )
        assert seen["create_pr"] is False

"""Tests for the deployment step.

Real git repositories in tmp_path, because the branch handling is the part that
can damage someone's checkout and mocking it would test nothing. Nothing here
touches a network: no test pushes, and ``gh`` is never invoked.
"""

import subprocess

import pytest

from evolution.core.cost import CostReport, LMCall
from evolution.core.pr_builder import (
    GitError,
    RejectedCandidate,
    ScoreLine,
    build_pull_request,
    render_body,
    require_clean_worktree,
)


def git(repo, *args):
    subprocess.run(
        ["git", *args], cwd=str(repo), check=True, capture_output=True, text=True
    )


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "Tester")
    git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "tool.py").write_text("DESCRIPTION = 'before'\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "base")
    return tmp_path


@pytest.fixture
def evolved(repo):
    (repo / "tool.py").write_text("DESCRIPTION = 'after'\n")
    return repo


def build(repo, **kw):
    kw.setdefault("target", "read_file")
    kw.setdefault("phase", "Phase 2")
    kw.setdefault("timestamp", "20260731_010203")
    kw.setdefault("files", ["tool.py"])
    return build_pull_request(repo=repo, **kw)


class TestBranch:
    def test_branch_follows_the_plan_naming(self, evolved):
        plan = build(evolved)
        assert plan.branch == "evolve/read_file-20260731_010203"

    def test_the_branch_really_exists_and_is_checked_out(self, evolved):
        plan = build(evolved)
        head = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(evolved), capture_output=True, text=True,
        ).stdout.strip()
        assert head == plan.branch

    def test_the_change_is_committed(self, evolved):
        build(evolved)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(evolved), capture_output=True, text=True,
        ).stdout.strip()
        assert status == ""

    def test_restore_returns_to_the_original_ref(self, evolved):
        before = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(evolved), capture_output=True, text=True,
        ).stdout.strip()
        plan = build(evolved)
        plan.restore()
        after = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(evolved), capture_output=True, text=True,
        ).stdout.strip()
        assert after == before

    def test_a_non_git_directory_is_refused(self, tmp_path):
        with pytest.raises(GitError, match="not a git repository"):
            build(tmp_path)

    def test_the_timestamp_is_supplied_not_read_from_the_clock(self, evolved):
        """Two runs with the same timestamp produce the same branch name."""
        first = build(evolved)
        first.discard()
        # discard() checked the original ref back out, which reset tool.py;
        # re-evolve it so the second run has the same change to commit.
        (evolved / "tool.py").write_text("DESCRIPTION = 'after'\n")
        second = build(evolved)
        second.discard()
        assert first.branch == second.branch == "evolve/read_file-20260731_010203"


class TestBody:
    def test_a_score_row_always_has_five_cells(self):
        """The detail cell exists even when empty, matching the table header."""
        with_detail = ScoreLine("val", 1.0, 1.0, detail="4 examples").row()
        without_detail = ScoreLine("train", 0.5, 0.75).row()
        assert with_detail == "| val | 1.000 | 1.000 | +0.000 | 4 examples |"
        assert without_detail == "| train | 0.500 | 0.750 | +0.250 |  |"

    def test_carries_every_split_plan_asks_for(self, evolved):
        plan = build(
            evolved,
            scores=[
                ScoreLine("train", 0.6, 0.7),
                ScoreLine("val", 0.55, 0.68),
                ScoreLine("holdout", 0.5, 0.64),
            ],
        )
        for split in ("train", "val", "holdout"):
            assert split in plan.body

    def test_includes_the_diff(self, evolved):
        body = build(evolved).body
        assert "```diff" in body
        assert "-DESCRIPTION = 'before'" in body
        assert "+DESCRIPTION = 'after'" in body

    def test_includes_the_cost(self, evolved):
        cost = CostReport(calls=[LMCall("m", 100, 50, 0.02)])
        assert "$0.0200" in build(evolved, cost=cost).body

    def test_an_unpriced_run_is_not_reported_as_cheap(self, evolved):
        cost = CostReport(calls=[LMCall("m", 100, 50, None)])
        body = build(evolved, cost=cost).body
        assert "at least" in body
        assert "no price available" in body

    def test_unmeasured_cost_says_so(self, evolved):
        assert "not measured" in build(evolved).body

    def test_lists_rejected_candidates(self, evolved):
        plan = build(
            evolved,
            rejected=[
                RejectedCandidate("cand-1", "size_limit: 528/500 chars"),
                RejectedCandidate("cand-2", "factual_accuracy: unknown parameter"),
            ],
        )
        assert "Rejected along the way" in plan.body
        assert "size_limit: 528/500 chars" in plan.body
        assert "factual_accuracy" in plan.body

    def test_lists_gates(self, evolved):
        plan = build(evolved, gates=["pytest: 2550 passed", "tblite: unavailable"])
        assert "tblite: unavailable" in plan.body

    def test_a_long_diff_is_clipped_with_a_pointer_to_the_branch(self):
        body = render_body(
            target="t", phase="p", scores=[], diff="\n".join(f"+line {i}" for i in range(900))
        )
        assert "clipped at 400 lines of 900" in body

    def test_body_is_written_next_to_the_run_artifacts(self, evolved, tmp_path):
        plan = build(evolved)
        path = plan.write_body(tmp_path / "out")
        assert path.name == "PULL_REQUEST.md"
        assert path.read_text() == plan.body


class TestCommitMessage:
    def test_follows_the_plan_shape(self, evolved):
        plan = build(
            evolved,
            scores=[ScoreLine("holdout", 0.5, 0.64)],
            optimizer="GEPA",
            iterations=10,
            dataset="synthetic, 120 examples",
        )
        assert plan.commit_message.startswith("evolve: read_file")
        assert "Optimizer: GEPA (10 iterations)" in plan.commit_message
        assert "Eval dataset: synthetic, 120 examples" in plan.commit_message
        assert "holdout: 0.500 -> 0.640 (+0.140)" in plan.commit_message

    def test_the_real_commit_carries_it(self, evolved):
        plan = build(evolved, scores=[ScoreLine("holdout", 0.5, 0.64)])
        message = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            cwd=str(evolved), capture_output=True, text=True,
        ).stdout
        assert "evolve: read_file" in message


class TestNothingIsPublished:
    def test_building_does_not_add_a_remote_or_push(self, evolved):
        build(evolved)
        remotes = subprocess.run(
            ["git", "remote"], cwd=str(evolved), capture_output=True, text=True
        ).stdout.strip()
        assert remotes == ""

    def test_push_is_a_separate_explicit_call(self, evolved):
        plan = build(evolved)
        # No remote configured, so a push must fail loudly rather than silently
        # succeed or be attempted as part of building.
        with pytest.raises(GitError):
            plan.push()

    def test_open_refuses_without_gh(self, evolved, monkeypatch):
        monkeypatch.setattr("evolution.core.pr_builder.shutil.which", lambda _: None)
        plan = build(evolved)
        with pytest.raises(GitError, match="gh is not installed"):
            plan.open()


class TestTheOperatorsWorkIsSafe:
    """A deployment step that eats uncommitted work is worse than no deployment.

    `git checkout -b` carries uncommitted edits onto the new branch, the commit
    absorbs them, and restoring the original ref leaves them only on a branch
    the operator never made. From the working tree it looks like deletion.
    """

    def test_a_dirty_target_file_is_refused_before_anything_happens(self, repo):
        (repo / "tool.py").write_text("DESCRIPTION = 'my uncommitted work'\n")
        with pytest.raises(GitError, match="uncommitted changes"):
            require_clean_worktree(repo, ["tool.py"])

    def test_allow_dirty_is_an_explicit_opt_in(self, repo):
        (repo / "tool.py").write_text("DESCRIPTION = 'my uncommitted work'\n")
        require_clean_worktree(repo, ["tool.py"], allow_dirty=True)

    def test_a_clean_target_passes(self, repo):
        require_clean_worktree(repo, ["tool.py"])

    def test_untracked_files_are_not_dirt(self, repo):
        (repo / "scratch.txt").write_text("notes\n")
        require_clean_worktree(repo, ["tool.py"])

    def test_unrelated_dirty_files_do_not_block(self, repo):
        (repo / "other.py").write_text("x = 1\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "other")
        (repo / "other.py").write_text("x = 2\n")
        require_clean_worktree(repo, ["tool.py"])


class TestAHalfBuiltBranchIsAbandoned:
    """A failure between `checkout -b` and the commit used to strand the caller.

    build_pull_request only returns a plan at the very end, so a caller that
    handles GitError had no object to call restore() on and was left standing on
    a branch it never asked for.
    """

    def _head(self, repo):
        return subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo), capture_output=True, text=True,
        ).stdout.strip()

    def test_a_failing_stage_puts_the_checkout_back(self, evolved):
        before = self._head(evolved)
        with pytest.raises(GitError):
            build(evolved, files=["does_not_exist.py"])
        assert self._head(evolved) == before

    def test_and_leaves_no_dead_branch_behind(self, evolved):
        with pytest.raises(GitError):
            build(evolved, files=["does_not_exist.py"])
        branches = subprocess.run(
            ["git", "branch"], cwd=str(evolved), capture_output=True, text=True
        ).stdout
        assert "evolve/" not in branches

    def test_discard_removes_the_branch_a_caller_no_longer_wants(self, evolved):
        before = self._head(evolved)
        plan = build(evolved)
        plan.discard()
        assert self._head(evolved) == before
        branches = subprocess.run(
            ["git", "branch"], cwd=str(evolved), capture_output=True, text=True
        ).stdout
        assert plan.branch not in branches

    def test_discard_is_safe_to_call_twice(self, evolved):
        plan = build(evolved)
        plan.discard()
        plan.discard()

    def test_abandoning_does_not_discard_unrelated_uncommitted_work(self, evolved):
        """The cleanup path must honour the promise require_clean_worktree makes.

        That function deliberately does not refuse over unrelated dirty files,
        on the grounds that they survive a branch switch. A forced checkout in
        the abandon path would make that untrue, and it would happen on the
        failure path, where nobody is looking.
        """
        (evolved / "notes.md").write_text("operator's uncommitted work\n")
        git(evolved, "add", "notes.md")
        (evolved / "other.txt").write_text("unstaged too\n")

        with pytest.raises(GitError):
            build(evolved, files=["does_not_exist.py"])

        assert (evolved / "notes.md").read_text() == "operator's uncommitted work\n"
        assert (evolved / "other.txt").read_text() == "unstaged too\n"

    def test_a_failing_commit_also_spares_unrelated_work(self, repo):
        """The commit itself failing must not cost the operator anything either.

        The unrelated file has to be *tracked and modified* to be at risk:
        untracked files survive a forced checkout, committed-then-edited ones
        are the ones it silently rewinds. A signing failure is used to make the
        commit fail, because hooks are deliberately skipped (see
        TestCommitHooksAreSkipped).
        """
        (repo / "notes.md").write_text("committed\n")
        git(repo, "add", "notes.md")
        git(repo, "commit", "-qm", "add notes")

        git(repo, "config", "commit.gpgsign", "true")
        git(repo, "config", "gpg.program", "/bin/false")

        (repo / "notes.md").write_text("operator's uncommitted work\n")
        (repo / "tool.py").write_text("DESCRIPTION = 'after'\n")

        with pytest.raises(GitError):
            build(repo)

        assert (repo / "notes.md").read_text() == "operator's uncommitted work\n"


class TestCommitHooksAreSkipped:
    """Phases 2 and 3 skip commit hooks, matching Phase 4's CodeOrganism.

    The two disagreed before: a checkout whose hooks Phase 4 ignored could still
    fail Phases 2 and 3. Skipping is the deliberate choice, because the commit
    lands on a scratch evolve/ branch nothing merges on its own.
    """

    def test_a_rejecting_pre_commit_hook_does_not_stop_the_run(self, evolved):
        hook = evolved / ".git" / "hooks" / "pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        plan = build(evolved)

        assert plan.branch.startswith("evolve/")
        head = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"],
            cwd=str(evolved), capture_output=True, text=True,
        ).stdout.strip()
        assert head.startswith("evolve: ")

    def test_a_rejecting_commit_msg_hook_does_not_stop_the_run(self, evolved):
        hook = evolved / ".git" / "hooks" / "commit-msg"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)

        plan = build(evolved)
        assert plan.branch.startswith("evolve/")


class TestUnrelatedStagedWorkSurvives:
    """`git commit` writes the whole index, not just what `git add` staged here.

    Work the operator had staged elsewhere would ride onto the evolve/ branch,
    and restore() would then leave it off the branch they were standing on.
    require_clean_worktree cannot see it: it only looks at the run's own paths.
    """

    @pytest.fixture
    def with_staged_work(self, repo):
        (repo / "notes.py").write_text("IMPORTANT = 'my staged work'\n")
        git(repo, "add", "notes.py")
        (repo / "tool.py").write_text("DESCRIPTION = 'after'\n")
        return repo

    def test_the_commit_does_not_absorb_it(self, with_staged_work):
        plan = build(with_staged_work)
        committed = subprocess.run(
            ["git", "show", "--name-only", "--format=", plan.branch],
            cwd=str(with_staged_work), capture_output=True, text=True, check=True,
        ).stdout.split()
        assert committed == ["tool.py"]

    def test_it_is_still_staged_after_the_run_restores(self, with_staged_work):
        plan = build(with_staged_work)
        plan.restore()
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(with_staged_work), capture_output=True, text=True, check=True,
        ).stdout.split()
        assert "notes.py" in staged
        assert (with_staged_work / "notes.py").read_text() == "IMPORTANT = 'my staged work'\n"

    def test_the_pr_body_diff_does_not_quote_it(self, with_staged_work):
        plan = build(with_staged_work)
        assert "IMPORTANT" not in plan.body
        assert "DESCRIPTION" in plan.body


class TestTheGuardFailsClosed:
    """An unreadable worktree is the one state where the run must not proceed."""

    def test_an_unreadable_worktree_is_refused_not_assumed_clean(self, repo, monkeypatch):
        def no_git(*args, **kwargs):
            raise GitError("git: command not found")

        monkeypatch.setattr("evolution.core.pr_builder._run", no_git)
        with pytest.raises(GitError, match="could not read the worktree state"):
            require_clean_worktree(repo, ["tool.py"])

    def test_allow_dirty_still_skips_the_question(self, repo, monkeypatch):
        def no_git(*args, **kwargs):
            raise GitError("git: command not found")

        monkeypatch.setattr("evolution.core.pr_builder._run", no_git)
        require_clean_worktree(repo, ["tool.py"], allow_dirty=True)

    def test_a_checkout_without_git_is_not_an_unreadable_one(self, tmp_path):
        """Nothing to strand and no branch to strand it on. Not this guard's problem."""
        (tmp_path / "tool.py").write_text("DESCRIPTION = 'x'\n")
        require_clean_worktree(tmp_path, ["tool.py"])

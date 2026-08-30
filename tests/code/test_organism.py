"""Tests for the git-backed code organism.

These run against real git repositories built in tmp_path. Mocking git would
test the mock: the whole value of this class is that branch creation, commits,
resets and branch restoration behave the way git actually behaves. No network,
no remotes, no LM.
"""

import json
import subprocess

import pytest

from evolution.code.organism import (
    CodeOrganism,
    DirtyWorktreeError,
    GitError,
    OrganismError,
    git_available,
    is_git_repo,
)

pytestmark = pytest.mark.skipif(
    not git_available(), reason="git is not installed on this machine"
)


BASELINE = '''"""Toy file tools."""


def read_lines(path, limit=10):
    """Return up to *limit* lines from *path*."""
    try:
        with open(path) as handle:
            return handle.read().splitlines()[:limit - 1]
    except OSError:
        return []
'''

FIXED = BASELINE.replace("[:limit - 1]", "[:limit]")


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=check,
    )


def current_branch(repo) -> str:
    return git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A miniature hermes-agent checkout with one commit."""
    root = tmp_path / "hermes-agent"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "file_tools.py").write_text(BASELINE)
    (root / "README.md").write_text("hermes-agent\n")

    git(root.parent, "-c", "init.defaultBranch=main", "init", "-q", str(root))
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test Runner")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    return root


class TestConstruction:
    def test_missing_repo_is_rejected(self, tmp_path):
        with pytest.raises(OrganismError, match="repo not found"):
            CodeOrganism(tmp_path / "nowhere", "tools/file_tools.py")

    def test_missing_target_is_rejected(self, repo):
        with pytest.raises(OrganismError, match="target file not found"):
            CodeOrganism(repo, "tools/ghost.py")

    def test_target_outside_repo_is_rejected(self, repo, tmp_path):
        outsider = tmp_path / "elsewhere.py"
        outsider.write_text("x = 1\n")
        with pytest.raises(OrganismError, match="outside the repo"):
            CodeOrganism(repo, outsider)

    def test_absolute_target_inside_repo_is_accepted(self, repo):
        organism = CodeOrganism(repo, repo / "tools" / "file_tools.py")
        assert organism.relpath == "tools/file_tools.py"

    def test_non_git_directory_is_rejected_at_start(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        (plain / "mod.py").write_text("x = 1\n")
        assert not is_git_repo(plain)
        with pytest.raises(GitError, match="not a git repository"):
            CodeOrganism(plain, "mod.py").start()

    def test_methods_require_an_open_organism(self, repo):
        organism = CodeOrganism(repo, "tools/file_tools.py")
        with pytest.raises(OrganismError, match="not open"):
            organism.mutate(FIXED, label="c01")
        with pytest.raises(OrganismError, match="not open"):
            organism.revert_last()


class TestLifecycle:
    def test_start_creates_a_branch_and_records_the_original(self, repo):
        original = current_branch(repo)
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            assert organism.is_open
            assert organism.branch != original
            assert organism.branch.startswith("evolve/code/file_tools-")
            assert organism.original_ref == original
            assert current_branch(repo) == organism.branch
        assert current_branch(repo) == original

    def test_custom_branch_name_is_honoured(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py", branch="fix/issue-742") as org:
            assert org.branch == "fix/issue-742"
            assert current_branch(repo) == "fix/issue-742"

    def test_baseline_source_is_the_file_at_branch_time(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            assert organism.baseline_source == BASELINE
            assert organism.baseline_sha

    def test_original_branch_is_restored_after_an_exception(self, repo):
        original = current_branch(repo)
        with pytest.raises(RuntimeError, match="boom"):
            with CodeOrganism(repo, "tools/file_tools.py") as organism:
                organism.mutate(FIXED, label="c01")
                raise RuntimeError("boom")
        assert current_branch(repo) == original
        assert (repo / "tools" / "file_tools.py").read_text() == BASELINE

    def test_uncommitted_candidate_does_not_survive_close(self, repo):
        original = current_branch(repo)
        organism = CodeOrganism(repo, "tools/file_tools.py").start()
        (repo / "tools" / "file_tools.py").write_text("garbage\n")
        organism.close()
        assert current_branch(repo) == original
        assert (repo / "tools" / "file_tools.py").read_text() == BASELINE

    def test_close_is_idempotent(self, repo):
        organism = CodeOrganism(repo, "tools/file_tools.py").start()
        organism.close()
        organism.close()
        assert not organism.is_open

    def test_start_is_idempotent(self, repo):
        organism = CodeOrganism(repo, "tools/file_tools.py").start()
        branch = organism.branch
        organism.start()
        assert organism.branch == branch
        organism.close()

    def test_detached_head_is_restored_as_a_sha(self, repo):
        sha = git(repo, "rev-parse", "HEAD").stdout.strip()
        git(repo, "checkout", "-q", "--detach", sha)
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            assert organism.original_ref == sha
        assert current_branch(repo) == "HEAD"
        assert git(repo, "rev-parse", "HEAD").stdout.strip() == sha

    def test_branch_survives_close_for_review(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            organism.mutate(FIXED, label="c01")
            branch = organism.branch
        branches = git(repo, "branch", "--list", branch).stdout
        assert branch in branches


class TestDirtyWorktree:
    def test_dirty_repo_is_refused_by_default(self, repo):
        (repo / "README.md").write_text("edited\n")
        with pytest.raises(DirtyWorktreeError, match="uncommitted changes"):
            CodeOrganism(repo, "tools/file_tools.py").start()

    def test_untracked_files_do_not_count_as_dirty(self, repo):
        (repo / "scratch.txt").write_text("notes\n")
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            assert organism.is_open
        assert (repo / "scratch.txt").exists()

    def test_allow_dirty_opts_in_and_restores_the_edit(self, repo):
        dirty_target = BASELINE + "\n# operator was mid-edit\n"
        (repo / "tools" / "file_tools.py").write_text(dirty_target)
        original = current_branch(repo)

        with CodeOrganism(repo, "tools/file_tools.py", allow_dirty=True) as organism:
            assert organism.baseline_source == dirty_target
            organism.mutate(FIXED, label="c01")

        assert current_branch(repo) == original
        assert (repo / "tools" / "file_tools.py").read_text() == dirty_target

    def test_unrelated_dirty_files_are_never_committed(self, repo):
        (repo / "README.md").write_text("operator edit\n")
        with CodeOrganism(repo, "tools/file_tools.py", allow_dirty=True) as organism:
            organism.mutate(FIXED, label="c01")
            status = git(repo, "status", "--porcelain", "--untracked-files=no").stdout
            assert "README.md" in status
        assert (repo / "README.md").read_text() == "operator edit\n"

    def test_revert_last_leaves_unrelated_dirty_files_alone(self, repo):
        """A rewind must not reach outside the target file.

        The evolution loop calls revert_last() once per candidate, so a
        repo-wide reset would take an operator's uncommitted work on the very
        first one.
        """
        (repo / "README.md").write_text("operator edit\n")
        with CodeOrganism(repo, "tools/file_tools.py", allow_dirty=True) as organism:
            organism.mutate(FIXED, label="c01")
            organism.revert_last()
            assert (repo / "README.md").read_text() == "operator edit\n"
        assert (repo / "README.md").read_text() == "operator edit\n"

    def test_revert_to_baseline_leaves_unrelated_dirty_files_alone(self, repo):
        (repo / "README.md").write_text("operator edit\n")
        with CodeOrganism(repo, "tools/file_tools.py", allow_dirty=True) as organism:
            organism.mutate(FIXED, label="c01")
            organism.mutate(FIXED + "# two\n", label="c02")
            organism.revert_to_baseline()
            assert (repo / "README.md").read_text() == "operator edit\n"
            assert organism.current_source() == organism.baseline_source
        assert (repo / "README.md").read_text() == "operator edit\n"

    def test_revert_last_leaves_unrelated_staged_work_alone(self, repo):
        """Staged-but-uncommitted work is just as unrecoverable as unstaged."""
        (repo / "README.md").write_text("staged edit\n")
        git(repo, "add", "README.md")
        with CodeOrganism(repo, "tools/file_tools.py", allow_dirty=True) as organism:
            organism.mutate(FIXED, label="c01")
            organism.revert_last()
            assert (repo / "README.md").read_text() == "staged edit\n"


class TestMutation:
    def test_mutate_writes_commits_and_diffs(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            mutation = organism.mutate(FIXED, label="c01")

            assert (repo / "tools" / "file_tools.py").read_text() == FIXED
            assert mutation.index == 1
            assert mutation.label == "c01"
            assert mutation.sha and mutation.parent_sha != mutation.sha
            assert "limit - 1" in mutation.diff
            assert not mutation.is_empty
            assert len(organism.lineage) == 1

    def test_every_candidate_becomes_a_commit(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            before = int(git(repo, "rev-list", "--count", "HEAD").stdout)
            organism.mutate(FIXED, label="c01")
            organism.mutate(BASELINE + "# two\n", label="c02")
            after = int(git(repo, "rev-list", "--count", "HEAD").stdout)
            assert after - before == 2
            assert [m.label for m in organism.lineage] == ["c01", "c02"]

    def test_an_unchanged_candidate_still_records_a_commit(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            mutation = organism.mutate(BASELINE, label="c01")
            assert mutation.is_empty
            assert len(organism.lineage) == 1

    def test_commit_message_names_the_candidate(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            organism.mutate(FIXED, label="c07")
            subject = git(repo, "log", "-1", "--format=%s").stdout.strip()
            assert "c07" in subject

    def test_only_the_target_file_is_staged(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            (repo / "README.md").write_text("sneaky\n")
            organism.mutate(FIXED, label="c01")
            changed = git(
                repo, "show", "--name-only", "--format=", "HEAD"
            ).stdout.split()
            assert changed == ["tools/file_tools.py"]


class TestRevert:
    def test_revert_last_restores_the_previous_content(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            organism.mutate(FIXED, label="c01")
            dropped = organism.revert_last()

            assert dropped.label == "c01"
            assert (repo / "tools" / "file_tools.py").read_text() == BASELINE
            assert organism.lineage == ()

    def test_revert_last_with_nothing_to_revert_is_a_no_op(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            assert organism.revert_last() is None

    def test_revert_last_undoes_only_one_mutation(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            organism.mutate(FIXED, label="c01")
            organism.mutate(FIXED + "# second\n", label="c02")
            organism.revert_last()
            assert (repo / "tools" / "file_tools.py").read_text() == FIXED
            assert [m.label for m in organism.lineage] == ["c01"]

    def test_revert_to_baseline_clears_everything(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            organism.mutate(FIXED, label="c01")
            organism.mutate(FIXED + "# second\n", label="c02")
            organism.revert_to_baseline()

            assert (repo / "tools" / "file_tools.py").read_text() == BASELINE
            assert organism.lineage == ()
            assert organism.diff_from_baseline() == ""

    def test_reapply_puts_a_reverted_winner_back(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            candidate = organism.mutate(FIXED, label="c01")
            organism.revert_last()
            final = organism.reapply(candidate)

            assert (repo / "tools" / "file_tools.py").read_text() == FIXED
            assert final.sha != candidate.sha
            assert "accepted candidate c01" in final.message
            assert "limit - 1" in organism.diff_from_baseline()


class TestDiffsAndReporting:
    def test_diff_from_baseline_is_empty_before_any_mutation(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            assert organism.diff_from_baseline() == ""

    def test_diff_working_tree_sees_uncommitted_edits(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            (repo / "tools" / "file_tools.py").write_text(FIXED)
            assert "limit - 1" in organism.diff_working_tree()

    def test_current_source_reads_the_working_tree(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            organism.mutate(FIXED, label="c01")
            assert organism.current_source() == FIXED

    def test_describe_is_json_serialisable(self, repo):
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            organism.mutate(FIXED, label="c01")
            blob = json.loads(json.dumps(organism.describe()))
            assert blob["target"] == "tools/file_tools.py"
            assert blob["mutations"][0]["label"] == "c01"
            assert blob["mutations"][0]["empty"] is False

    def test_original_branch_never_sees_the_mutation(self, repo):
        original = current_branch(repo)
        with CodeOrganism(repo, "tools/file_tools.py") as organism:
            organism.mutate(FIXED, label="c01")
        assert current_branch(repo) == original
        assert (repo / "tools" / "file_tools.py").read_text() == BASELINE

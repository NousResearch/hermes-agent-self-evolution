"""Tests for the Phase 4 entry point.

The external evolver is never really invoked: candidates come either from an
injected fake or from a stub script written into tmp_path that speaks the
adapter's job/output contract. The hermes-agent checkout is a real git repo
with a real (tiny) test suite and a real reproduction script, because the
command's whole job is to orchestrate those and it is not worth testing
against mocks of them.

Offline throughout: no LM is ever configured, and build_objective is exercised
on both the no-LM path and an injected-predictor path.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from evolution.code import evolve_tool_code as mod
from evolution.code.evolve_tool_code import (
    MUTATION_CONSTRAINTS,
    UNSANDBOXED,
    Candidate,
    EvolverError,
    EvolverJob,
    EvolverNotInstalled,
    ExternalEvolver,
    TargetNotFound,
    build_objective,
    evolve_tool_code,
    find_evolver,
    main,
    resolve_tool_file,
)
from evolution.code.organism import git_available

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

# Fixes the bug and breaks the contract: a new parameter on a public function.
UNSAFE = FIXED.replace(
    "def read_lines(path, limit=10):", "def read_lines(path, limit=10, encoding='utf-8'):"
)

# Fixes the bug by deleting the error handling around it.
UNGUARDED = '''"""Toy file tools."""


def read_lines(path, limit=10):
    """Return up to *limit* lines from *path*."""
    with open(path) as handle:
        return handle.read().splitlines()[:limit]
'''

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

# The same reproduction, made intermittent: it clears on every other run
# whatever the code says. A patch cannot fix this, and one lucky run cannot
# tell the difference.
FLAKY_REPRO = '''import pathlib
import sys

counter = pathlib.Path(__file__).with_suffix(".count")
count = (int(counter.read_text() or 0) if counter.exists() else 0) + 1
counter.write_text(str(count))

if count % 2:
    print("BUG_PRESENT")
    sys.exit(1)
print("BUG_FIXED")
sys.exit(0)
'''


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )


def current_branch(repo) -> str:
    return git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A miniature hermes-agent checkout: one buggy tool, one green test."""
    root = tmp_path / "hermes-agent"
    (root / "tools").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "tools" / "__init__.py").write_text("")
    (root / "tools" / "file_tools.py").write_text(BASELINE)
    (root / "tests" / "test_smoke.py").write_text("def test_smoke():\n    assert True\n")

    git(root.parent, "-c", "init.defaultBranch=main", "init", "-q", str(root))
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test Runner")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    return root


@pytest.fixture
def repro(tmp_path):
    path = tmp_path / "repro_issue_742.py"
    path.write_text(REPRO)
    return path


@pytest.fixture
def flaky_repro(tmp_path):
    path = tmp_path / "repro_flaky.py"
    path.write_text(FLAKY_REPRO)
    return path


def hide_the_evolver(monkeypatch):
    """Make every Darwinian Evolver command look absent, and nothing else.

    Blanking shutil.which outright would also hide git, which would send the
    run down the "git is not installed" path instead of the one under test.
    """
    import shutil as real_shutil

    real_which = real_shutil.which

    def which(name, *args, **kwargs):
        if any(candidate in name for candidate in ("evolv", "devolve")):
            return None
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(mod.shutil, "which", which)
    monkeypatch.delenv(mod.EVOLVER_ENV_VAR, raising=False)


class FakeEvolver:
    """Stands in for the Darwinian Evolver CLI without running anything."""

    def __init__(self, *sources):
        self.sources = list(sources)
        self.jobs: list[EvolverJob] = []

    def propose(self, job):
        self.jobs.append(job)
        return [
            Candidate(index=i, source=source, notes=f"candidate {i}", origin="fake")
            for i, source in enumerate(self.sources, start=1)
        ]


class BrokenEvolver:
    def propose(self, job):
        raise EvolverError("evolver exited 1 and produced no candidates")


class MissingEvolver:
    def propose(self, job):
        raise EvolverNotInstalled("could not run darwinian-evolver: [Errno 2]")


# ──────────────────────────────────────────────────────────────────────────
# Target resolution
# ──────────────────────────────────────────────────────────────────────────


class TestResolveToolFile:
    def test_bare_module_name_resolves_into_tools(self, repo):
        assert resolve_tool_file(repo, "file_tools").name == "file_tools.py"

    def test_filename_resolves(self, repo):
        assert resolve_tool_file(repo, "file_tools.py").name == "file_tools.py"

    def test_repo_relative_path_resolves(self, repo):
        resolved = resolve_tool_file(repo, "tools/file_tools.py")
        assert resolved == (repo / "tools" / "file_tools.py").resolve()

    def test_absolute_path_resolves(self, repo):
        target = repo / "tools" / "file_tools.py"
        assert resolve_tool_file(repo, str(target)) == target.resolve()

    def test_agent_directory_is_searched_too(self, repo):
        (repo / "agent").mkdir()
        (repo / "agent" / "prompt_builder.py").write_text("X = 1\n")
        assert resolve_tool_file(repo, "prompt_builder").parent.name == "agent"

    def test_unknown_tool_is_reported(self, repo):
        with pytest.raises(TargetNotFound, match="could not find a source file"):
            resolve_tool_file(repo, "does_not_exist")

    def test_a_file_outside_the_repo_is_refused(self, repo, tmp_path):
        outsider = tmp_path / "outsider.py"
        outsider.write_text("x = 1\n")
        with pytest.raises(TargetNotFound, match="outside the hermes-agent repo"):
            resolve_tool_file(repo, str(outsider))


# ──────────────────────────────────────────────────────────────────────────
# Evolver discovery
# ──────────────────────────────────────────────────────────────────────────


class TestFindEvolver:
    def test_nothing_installed_raises(self, monkeypatch):
        monkeypatch.setattr(mod.shutil, "which", lambda name: None)
        with pytest.raises(EvolverNotInstalled, match="not installed"):
            find_evolver(None, env={})

    def test_the_error_names_the_licence_constraint(self, monkeypatch):
        monkeypatch.setattr(mod.shutil, "which", lambda name: None)
        with pytest.raises(EvolverNotInstalled, match="AGPL v3"):
            find_evolver(None, env={})

    def test_a_command_on_path_is_found(self, monkeypatch):
        monkeypatch.setattr(
            mod.shutil, "which", lambda name: "/usr/local/bin/darwinian-evolver"
        )
        assert find_evolver(None, env={}) == ["/usr/local/bin/darwinian-evolver"]

    def test_an_explicit_command_wins(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod.shutil, "which", lambda name: None)
        script = tmp_path / "my-evolver"
        script.write_text("#!/bin/sh\n")
        assert find_evolver(str(script), env={}) == [str(script)]

    def test_an_explicit_command_may_carry_arguments(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod.shutil, "which", lambda name: None)
        script = tmp_path / "my-evolver"
        script.write_text("#!/bin/sh\n")
        assert find_evolver(f"{script} --quiet", env={}) == [str(script), "--quiet"]

    def test_an_explicit_command_that_does_not_exist_is_an_error(self, monkeypatch):
        monkeypatch.setattr(mod.shutil, "which", lambda name: None)
        with pytest.raises(EvolverNotInstalled, match="not executable"):
            find_evolver("/nowhere/evolver", env={})

    def test_the_environment_variable_is_honoured(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod.shutil, "which", lambda name: None)
        script = tmp_path / "env-evolver"
        script.write_text("#!/bin/sh\n")
        found = find_evolver(None, env={mod.EVOLVER_ENV_VAR: str(script)})
        assert found == [str(script)]


# ──────────────────────────────────────────────────────────────────────────
# External evolver adapter
# ──────────────────────────────────────────────────────────────────────────


STUB_HEADER = '''import argparse
import json
import pathlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

job = json.loads(pathlib.Path(args.job).read_text())
out = pathlib.Path(args.output)
'''


def make_stub(tmp_path, body, name="stub_evolver.py"):
    path = tmp_path / name
    path.write_text(STUB_HEADER + body)
    return [sys.executable, str(path)]


def make_job(source=BASELINE):
    return EvolverJob(
        target_path="tools/file_tools.py",
        source=source,
        objective="fix the off-by-one",
        iterations=3,
        bug_issue="742",
    )


class TestExternalEvolver:
    def test_candidates_are_read_from_the_output_directory(self, tmp_path):
        cmd = make_stub(
            tmp_path,
            '(out / "candidates").mkdir(parents=True, exist_ok=True)\n'
            '(out / "candidates" / "001.py").write_text(job["source"] + "# one\\n")\n'
            '(out / "candidates" / "002.py").write_text(job["source"] + "# two\\n")\n',
        )
        evolver = ExternalEvolver(cmd, repo=tmp_path, workdir=tmp_path / "work", sandbox=UNSANDBOXED)
        candidates = evolver.propose(make_job())

        assert [c.label for c in candidates] == ["c01", "c02"]
        assert candidates[0].source.endswith("# one\n")

    def test_candidates_are_read_from_jsonl(self, tmp_path):
        cmd = make_stub(
            tmp_path,
            'out.mkdir(parents=True, exist_ok=True)\n'
            '(out / "candidates.jsonl").write_text(\n'
            '    json.dumps({"source": job["source"] + "# jsonl\\n", "notes": "why"}) + "\\n"\n'
            ")\n",
        )
        evolver = ExternalEvolver(cmd, repo=tmp_path, workdir=tmp_path / "work", sandbox=UNSANDBOXED)
        candidates = evolver.propose(make_job())

        assert len(candidates) == 1
        assert candidates[0].notes == "why"

    def test_candidates_are_read_from_stdout(self, tmp_path):
        cmd = make_stub(
            tmp_path,
            'print(json.dumps({"source": job["source"] + "# stdout\\n"}))\n',
        )
        evolver = ExternalEvolver(cmd, repo=tmp_path, workdir=tmp_path / "work", sandbox=UNSANDBOXED)
        candidates = evolver.propose(make_job())

        assert len(candidates) == 1
        assert candidates[0].origin == "stdout"

    def test_a_candidate_may_point_at_a_file(self, tmp_path):
        (tmp_path / "elsewhere.py").write_text("# from a path\n")
        cmd = make_stub(
            tmp_path,
            'print(json.dumps({"path": str(pathlib.Path(args.output).parent.parent / "elsewhere.py")}))\n',
        )
        evolver = ExternalEvolver(cmd, repo=tmp_path, workdir=tmp_path / "work", sandbox=UNSANDBOXED)
        assert evolver.propose(make_job())[0].source == "# from a path\n"

    def test_the_job_file_carries_the_contract(self, tmp_path):
        cmd = make_stub(
            tmp_path,
            'print(json.dumps({"source": job["objective"] + "\\n" + str(job["iterations"])}))\n',
        )
        evolver = ExternalEvolver(cmd, repo=tmp_path, workdir=tmp_path / "work", sandbox=UNSANDBOXED)
        candidate = evolver.propose(make_job())[0]

        assert "fix the off-by-one" in candidate.source
        written = json.loads((tmp_path / "work" / "job.json").read_text())
        assert written["target_path"] == "tools/file_tools.py"
        assert written["constraints"] == list(MUTATION_CONSTRAINTS)
        assert written["bug_issue"] == "742"

    def test_no_candidates_is_an_error(self, tmp_path):
        cmd = make_stub(tmp_path, 'sys.stderr.write("nothing to do\\n")\nsys.exit(1)\n')
        evolver = ExternalEvolver(cmd, repo=tmp_path, workdir=tmp_path / "work", sandbox=UNSANDBOXED)
        with pytest.raises(EvolverError, match="produced no candidates"):
            evolver.propose(make_job())

    def test_candidates_from_a_failed_run_are_still_scored(self, tmp_path):
        cmd = make_stub(
            tmp_path,
            'print(json.dumps({"source": "# partial\\n"}))\nsys.exit(1)\n',
        )
        evolver = ExternalEvolver(cmd, repo=tmp_path, workdir=tmp_path / "work", sandbox=UNSANDBOXED)
        assert len(evolver.propose(make_job())) == 1
        assert evolver.last_returncode == 1

    def test_a_missing_binary_reports_as_not_installed(self, tmp_path):
        evolver = ExternalEvolver(
            ["/nonexistent/darwinian-evolver"], repo=tmp_path, workdir=tmp_path / "work", sandbox=UNSANDBOXED)
        with pytest.raises(EvolverNotInstalled, match="could not run"):
            evolver.propose(make_job())

    def test_a_hung_evolver_times_out(self, tmp_path):
        cmd = make_stub(tmp_path, "import time\ntime.sleep(30)\n")
        evolver = ExternalEvolver(
            cmd, repo=tmp_path, workdir=tmp_path / "work", timeout=1, sandbox=UNSANDBOXED)
        with pytest.raises(EvolverError, match="timed out"):
            evolver.propose(make_job())

    def test_a_relative_workdir_still_reaches_the_evolver(self, tmp_path, monkeypatch):
        # The evolver runs with cwd set to the repo, which is not the directory
        # the operator launched from. Relative job paths must not leak through.
        checkout = tmp_path / "hermes-agent"
        checkout.mkdir()
        cmd = make_stub(
            tmp_path,
            '(out / "candidates").mkdir(parents=True, exist_ok=True)\n'
            '(out / "candidates" / "001.py").write_text(job["source"])\n',
        )
        monkeypatch.chdir(tmp_path)
        evolver = ExternalEvolver(cmd, repo=checkout, workdir=Path("output/run-1"), sandbox=UNSANDBOXED)
        assert len(evolver.propose(make_job())) == 1

    def test_unparseable_stdout_lines_are_ignored(self, tmp_path):
        cmd = make_stub(
            tmp_path,
            'print("progress: 40%")\nprint("{not json}")\n'
            'print(json.dumps({"source": "# ok\\n"}))\n',
        )
        evolver = ExternalEvolver(cmd, repo=tmp_path, workdir=tmp_path / "work", sandbox=UNSANDBOXED)
        assert len(evolver.propose(make_job())) == 1


# ──────────────────────────────────────────────────────────────────────────
# Mutation brief
# ──────────────────────────────────────────────────────────────────────────


class FakePredictor:
    def __init__(self, objective="", raises=None):
        self.objective = objective
        self.raises = raises
        self.seen = None

    def __call__(self, **kwargs):
        self.seen = kwargs
        if self.raises:
            raise self.raises
        return type("Prediction", (), {"objective": self.objective})()


class TestBuildObjective:
    @pytest.fixture(autouse=True)
    def no_lm(self, monkeypatch):
        """Guarantee the no-LM path, whatever else the session configured.

        PR #127 in this repo broke because a test reached a dspy predictor with
        no LM loaded. The rule here is stronger: no test reaches a predictor at
        all, configured or not.
        """
        monkeypatch.setattr(mod.dspy.settings, "lm", None, raising=False)

    def test_lm_configured_reports_the_absence_of_a_model(self):
        assert mod._lm_configured() is False

    def test_without_an_lm_a_template_is_used(self):
        objective = build_objective("tools/file_tools.py", bug_issue="issue 742")
        assert "tools/file_tools.py" in objective
        assert "issue 742" in objective

    def test_an_injected_predictor_is_used(self):
        predictor = FakePredictor("Return all requested lines, not limit-1 of them.")
        objective = build_objective(
            "tools/file_tools.py",
            bug_issue="issue 742",
            reproduction=REPRO,
            predictor=predictor,
        )
        assert objective == "Return all requested lines, not limit-1 of them."
        assert predictor.seen["tool_module"] == "tools/file_tools.py"
        assert "Do not change any function signature" in predictor.seen["constraints"]

    def test_a_blank_prediction_falls_back_to_the_template(self):
        objective = build_objective("tools/file_tools.py", predictor=FakePredictor("  "))
        assert "smallest change" in objective

    def test_a_failing_predictor_falls_back_to_the_template(self):
        predictor = FakePredictor(raises=RuntimeError("no LM is loaded"))
        objective = build_objective("tools/file_tools.py", predictor=predictor)
        assert "tools/file_tools.py" in objective

    def test_the_reproduction_is_mentioned_when_supplied(self):
        assert "reproduction script" in build_objective("t.py", reproduction=REPRO)


# ──────────────────────────────────────────────────────────────────────────
# End to end
# ──────────────────────────────────────────────────────────────────────────


needs_git = pytest.mark.skipif(
    not git_available(), reason="git is not installed on this machine"
)


@needs_git
class TestEvolveToolCode:
    def test_a_winning_candidate_produces_a_branch_and_a_diff(
        self, repo, repro, tmp_path
    ):
        original = current_branch(repo)
        evolver = FakeEvolver(FIXED, UNSAFE)
        out_root = tmp_path / "out"

        code = evolve_tool_code(
            tool="file_tools",
            bug_issue="742",
            repro_script=str(repro),
            hermes_repo=str(repo),
            python=sys.executable,
            evolver=evolver,
            output_root=out_root, sandbox=UNSANDBOXED,
        )

        assert code == 0
        assert current_branch(repo) == original
        assert (repo / "tools" / "file_tools.py").read_text() == BASELINE

        metrics = json.loads(
            next(out_root.rglob("metrics.json")).read_text()
        )
        assert metrics["winner"] == "c01"
        assert metrics["baseline"]["bug_reproduces"] is True
        assert metrics["candidates"][1]["fitness"]["accepted"] is False

        diff = next(out_root.rglob("winner.diff")).read_text()
        assert "limit - 1" in diff

        branch = metrics["branch"]
        shipped = git(repo, "show", f"{branch}:tools/file_tools.py").stdout
        assert shipped == FIXED

    def test_the_evolver_receives_the_baseline_and_the_constraints(
        self, repo, repro, tmp_path
    ):
        evolver = FakeEvolver(FIXED)
        evolve_tool_code(
            tool="file_tools",
            repro_script=str(repro),
            hermes_repo=str(repo),
            python=sys.executable,
            iterations=4,
            evolver=evolver,
            output_root=tmp_path / "out", sandbox=UNSANDBOXED,
        )
        job = evolver.jobs[0]
        assert job.source == BASELINE
        assert job.target_path == "tools/file_tools.py"
        assert job.iterations == 4
        assert job.reproduction == REPRO
        assert job.constraints == MUTATION_CONSTRAINTS

    def test_unsafe_candidates_leave_nothing_behind(self, repo, repro, tmp_path):
        original = current_branch(repo)
        before_commits = git(repo, "rev-list", "--count", "HEAD").stdout.strip()
        out_root = tmp_path / "out"

        code = evolve_tool_code(
            tool="file_tools",
            repro_script=str(repro),
            hermes_repo=str(repo),
            python=sys.executable,
            evolver=FakeEvolver(UNSAFE, UNGUARDED),
            output_root=out_root, sandbox=UNSANDBOXED,
        )

        assert code == 0
        metrics = json.loads(next(out_root.rglob("metrics.json")).read_text())
        assert metrics["winner"] is None
        assert not list(out_root.rglob("winner.diff"))

        branch = metrics["branch"]
        assert (
            git(repo, "rev-list", "--count", branch).stdout.strip() == before_commits
        )
        assert current_branch(repo) == original

    def test_a_candidate_that_does_not_fix_the_bug_is_rejected(
        self, repo, repro, tmp_path
    ):
        cosmetic = BASELINE.replace('"""Toy file tools."""', '"""Toy file tools (tidied)."""')
        out_root = tmp_path / "out"

        evolve_tool_code(
            tool="file_tools",
            repro_script=str(repro),
            hermes_repo=str(repo),
            python=sys.executable,
            evolver=FakeEvolver(cosmetic),
            output_root=out_root, sandbox=UNSANDBOXED,
        )
        metrics = json.loads(next(out_root.rglob("metrics.json")).read_text())
        assert metrics["winner"] is None
        assert "bug not fixed" in metrics["candidates"][0]["fitness"]["rejection_reason"]

    def test_every_candidate_is_scored_against_the_baseline(self, repo, repro, tmp_path):
        # Two independent fixes: the second must not be scored on top of the first.
        other_fix = FIXED.replace(
            '    """Return up to *limit* lines from *path*."""',
            '    """Return up to *limit* lines from *path*."""\n    # second variant',
        )
        out_root = tmp_path / "out"
        evolve_tool_code(
            tool="file_tools",
            repro_script=str(repro),
            hermes_repo=str(repo),
            python=sys.executable,
            evolver=FakeEvolver(FIXED, other_fix),
            output_root=out_root, sandbox=UNSANDBOXED,
        )
        metrics_path = next(out_root.rglob("metrics.json"))
        metrics = json.loads(metrics_path.read_text())
        assert [c["fitness"]["accepted"] for c in metrics["candidates"]] == [True, True]
        # The winner's saved source sits beside metrics.json and must be the
        # variant scored under that label, not the other one layered on top.
        winner_source = metrics_path.parent / f"{metrics['winner']}.py"
        sources = {"c01": FIXED, "c02": other_fix}
        assert winner_source.read_text() == sources[metrics["winner"]]

    def test_the_branch_is_restored_even_when_a_candidate_explodes(
        self, repo, repro, tmp_path
    ):
        original = current_branch(repo)

        class ExplodingEvolver:
            def propose(self, job):
                raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            evolve_tool_code(
                tool="file_tools",
                repro_script=str(repro),
                hermes_repo=str(repo),
                python=sys.executable,
                evolver=ExplodingEvolver(),
                output_root=tmp_path / "out", sandbox=UNSANDBOXED,
        )
        assert current_branch(repo) == original

    def test_a_missing_evolver_exits_non_zero_without_touching_the_repo(
        self, repo, repro, tmp_path, monkeypatch
    ):
        hide_the_evolver(monkeypatch)
        original = current_branch(repo)

        code = evolve_tool_code(
            tool="file_tools",
            repro_script=str(repro),
            hermes_repo=str(repo),
            python=sys.executable,
            output_root=tmp_path / "out", sandbox=UNSANDBOXED,
        )

        assert code == 2
        assert current_branch(repo) == original
        assert git(repo, "branch", "--list").stdout.count("\n") == 1

    def test_an_evolver_that_fails_mid_run_exits_three(self, repo, repro, tmp_path):
        code = evolve_tool_code(
            tool="file_tools",
            repro_script=str(repro),
            hermes_repo=str(repo),
            python=sys.executable,
            evolver=BrokenEvolver(),
            output_root=tmp_path / "out", sandbox=UNSANDBOXED,
        )
        assert code == 3

    def test_an_evolver_that_disappears_mid_run_exits_two(self, repo, repro, tmp_path):
        code = evolve_tool_code(
            tool="file_tools",
            repro_script=str(repro),
            hermes_repo=str(repo),
            python=sys.executable,
            evolver=MissingEvolver(),
            output_root=tmp_path / "out", sandbox=UNSANDBOXED,
        )
        assert code == 2

    def test_dry_run_validates_without_branching(self, repo, repro, tmp_path):
        stub = tmp_path / "stub-evolver"
        stub.write_text("#!/bin/sh\n")
        branches_before = git(repo, "branch", "--list").stdout

        code = evolve_tool_code(
            tool="file_tools",
            repro_script=str(repro),
            hermes_repo=str(repo),
            evolver_cmd=str(stub),
            dry_run=True,
            output_root=tmp_path / "out", sandbox=UNSANDBOXED,
        )

        assert code == 0
        assert git(repo, "branch", "--list").stdout == branches_before
        assert not (tmp_path / "out").exists()

    def test_an_unknown_tool_exits_one(self, repo, tmp_path):
        assert (
            evolve_tool_code(
                tool="ghost_tools",
                hermes_repo=str(repo),
                evolver=FakeEvolver(FIXED),
                output_root=tmp_path / "out", sandbox=UNSANDBOXED,
        )
            == 1
        )

    def test_a_missing_repro_script_exits_one(self, repo, tmp_path):
        assert (
            evolve_tool_code(
                tool="file_tools",
                repro_script=str(tmp_path / "ghost.py"),
                hermes_repo=str(repo),
                evolver=FakeEvolver(FIXED),
                output_root=tmp_path / "out", sandbox=UNSANDBOXED,
        )
            == 1
        )

    def test_a_non_git_repo_exits_one(self, tmp_path):
        plain = tmp_path / "plain"
        (plain / "tools").mkdir(parents=True)
        (plain / "tools" / "file_tools.py").write_text(BASELINE)
        assert (
            evolve_tool_code(
                tool="file_tools",
                hermes_repo=str(plain),
                evolver=FakeEvolver(FIXED),
                output_root=tmp_path / "out", sandbox=UNSANDBOXED,
        )
            == 1
        )

    def test_a_red_baseline_stops_the_run(self, repo, repro, tmp_path):
        (repo / "tests" / "test_smoke.py").write_text(
            "def test_smoke():\n    assert False\n"
        )
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "break the suite")
        original = current_branch(repo)

        code = evolve_tool_code(
            tool="file_tools",
            repro_script=str(repro),
            hermes_repo=str(repo),
            python=sys.executable,
            evolver=FakeEvolver(FIXED),
            output_root=tmp_path / "out", sandbox=UNSANDBOXED,
        )

        assert code == 1
        assert current_branch(repo) == original

    def test_strict_mode_refuses_a_bug_that_does_not_reproduce(
        self, repo, repro, tmp_path
    ):
        (repo / "tools" / "file_tools.py").write_text(FIXED)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "already fixed")

        code = evolve_tool_code(
            tool="file_tools",
            repro_script=str(repro),
            hermes_repo=str(repo),
            python=sys.executable,
            strict_gates=True,
            evolver=FakeEvolver(FIXED),
            output_root=tmp_path / "out", sandbox=UNSANDBOXED,
        )
        assert code == 1

    def test_a_run_without_a_reproduction_still_scores_candidates(
        self, repo, tmp_path
    ):
        out_root = tmp_path / "out"
        code = evolve_tool_code(
            tool="file_tools",
            hermes_repo=str(repo),
            python=sys.executable,
            evolver=FakeEvolver(FIXED),
            output_root=out_root, sandbox=UNSANDBOXED,
        )
        assert code == 0
        metrics = json.loads(next(out_root.rglob("metrics.json")).read_text())
        assert metrics["candidates"][0]["fitness"]["repro"] is None
        assert metrics["winner"] == "c01"


# ──────────────────────────────────────────────────────────────────────────
# Uncertainty in the run record
# ──────────────────────────────────────────────────────────────────────────


@needs_git
class TestMeasuredEvidence:
    def run_once(self, repo, tmp_path, *sources, **kwargs):
        out_root = tmp_path / "out"
        code = evolve_tool_code(
            tool="file_tools",
            hermes_repo=str(repo),
            python=sys.executable,
            evolver=FakeEvolver(*sources),
            output_root=out_root,
            **kwargs, sandbox=UNSANDBOXED,
        )
        metrics = json.loads(next(out_root.rglob("metrics.json")).read_text())
        return code, metrics

    def test_the_reproduction_runs_as_often_as_asked(self, repo, repro, tmp_path):
        _, metrics = self.run_once(
            repo, tmp_path, FIXED, repro_script=str(repro), repro_runs=3
        )
        trials = metrics["candidates"][0]["fitness"]["repro_trials"]

        assert metrics["repro_runs"] == 3
        assert trials["runs"] == 3
        assert trials["fixes"] == 3
        assert trials["fix_rate_ci"]["low"] < 1.0
        assert metrics["baseline"]["repro_trials"]["runs"] == 3

    def test_a_flaky_reproduction_is_not_accepted_as_a_fix(
        self, repo, flaky_repro, tmp_path
    ):
        code, metrics = self.run_once(
            repo, tmp_path, FIXED, repro_script=str(flaky_repro), repro_runs=4
        )
        fitness = metrics["candidates"][0]["fitness"]

        assert code == 0
        assert metrics["winner"] is None
        assert fitness["accepted"] is False
        assert "bug not fixed" in fitness["rejection_reason"]
        assert fitness["repro_trials"]["flaky"] is True
        assert fitness["repro_trials"]["fixes"] == 2

    def test_a_single_run_would_have_believed_the_same_flake(
        self, repo, flaky_repro, tmp_path
    ):
        # The point of --repro-runs: one run of this script clears the bug and
        # the candidate ships. Four runs of it do not.
        code, metrics = self.run_once(
            repo, tmp_path, FIXED, repro_script=str(flaky_repro), repro_runs=1
        )
        assert code == 0
        assert metrics["candidates"][0]["fitness"]["repro_trials"]["runs"] == 1
        assert metrics["winner"] == "c01"

    def test_the_test_suite_is_compared_test_by_test(self, repo, repro, tmp_path):
        _, metrics = self.run_once(
            repo, tmp_path, FIXED, repro_script=str(repro)
        )
        suite = metrics["candidates"][0]["fitness"]["suite"]

        assert metrics["baseline"]["tests_measured"] == 1
        assert suite is not None
        assert suite["verdict"] == "identical outcomes"
        assert suite["paired"]["n"] == 1
        assert suite["newly_failing"] == []

    def test_every_score_arrives_with_its_evidence_coverage(
        self, repo, repro, tmp_path
    ):
        _, metrics = self.run_once(repo, tmp_path, FIXED, repro_script=str(repro))
        fitness = metrics["candidates"][0]["fitness"]

        # A reproduction and the quality heuristics ran; no benchmark did.
        assert fitness["evidence_coverage"] == 0.7
        assert fitness["missing_evidence"] == ["benchmark"]

    def test_a_score_with_no_reproduction_says_how_little_it_measured(
        self, repo, tmp_path
    ):
        _, metrics = self.run_once(repo, tmp_path, FIXED)
        fitness = metrics["candidates"][0]["fitness"]

        assert fitness["total"] == 1.0
        assert fitness["evidence_coverage"] == 0.2
        assert fitness["missing_evidence"] == ["bug_fix", "benchmark"]

    def test_two_candidates_within_noise_are_reported_as_arbitrary(
        self, repo, repro, tmp_path
    ):
        other_fix = FIXED.replace(
            '    """Return up to *limit* lines from *path*."""',
            '    """Return up to *limit* lines from *path*."""\n    # second variant',
        )
        _, metrics = self.run_once(
            repo, tmp_path, FIXED, other_fix, repro_script=str(repro)
        )
        ranking = metrics["ranking"]

        assert ranking["winner"] == metrics["winner"]
        assert ranking["margin"] == 0.0
        assert ranking["within_noise"] is True
        assert ranking["tied"] == ["c01", "c02"]
        assert "arbitrary" in ranking["summary"]

    def test_a_sole_survivor_is_ranked_against_nothing(self, repo, repro, tmp_path):
        _, metrics = self.run_once(
            repo, tmp_path, FIXED, UNSAFE, repro_script=str(repro)
        )
        ranking = metrics["ranking"]

        assert ranking["winner"] == "c01"
        assert ranking["runner_up"] is None
        assert ranking["margin"] is None
        assert ranking["considered"] == 1

    def test_no_survivor_leaves_no_ranking(self, repo, repro, tmp_path):
        _, metrics = self.run_once(
            repo, tmp_path, UNSAFE, repro_script=str(repro)
        )
        assert metrics["ranking"] is None
        assert metrics["winner"] is None


# ──────────────────────────────────────────────────────────────────────────
# CLI surface
# ──────────────────────────────────────────────────────────────────────────


@needs_git
class TestCli:
    def test_dry_run_exits_zero(self, repo, tmp_path):
        stub = tmp_path / "stub-evolver"
        stub.write_text("#!/bin/sh\n")
        result = CliRunner().invoke(
            main,
            [
                "--tool", "file_tools",
                "--hermes-repo", str(repo),
                "--evolver-cmd", str(stub),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_a_missing_evolver_exits_two(self, repo, monkeypatch):
        hide_the_evolver(monkeypatch)
        result = CliRunner().invoke(
            main, ["--tool", "file_tools", "--hermes-repo", str(repo)]
        )
        assert result.exit_code == 2

    def test_an_unknown_tool_exits_one(self, repo, tmp_path):
        stub = tmp_path / "stub-evolver"
        stub.write_text("#!/bin/sh\n")
        result = CliRunner().invoke(
            main,
            [
                "--tool", "ghost",
                "--hermes-repo", str(repo),
                "--evolver-cmd", str(stub),
                "--dry-run",
            ],
        )
        assert result.exit_code == 1

    def test_tool_is_required(self):
        result = CliRunner().invoke(main, [])
        assert result.exit_code != 0
        assert "--tool" in result.output

    def test_repro_runs_reaches_the_plan(self, repo, repro, tmp_path):
        stub = tmp_path / "stub-evolver"
        stub.write_text("#!/bin/sh\n")
        result = CliRunner().invoke(
            main,
            [
                "--tool", "file_tools",
                "--hermes-repo", str(repo),
                "--repro-script", str(repro),
                "--evolver-cmd", str(stub),
                "--repro-runs", "5",
                "--dry-run",
            ],
        )
        # Rich wraps console output to the terminal width, so compare on the
        # text rather than on where the line breaks landed.
        assert result.exit_code == 0
        assert "bug reproduction x5" in " ".join(result.output.split())

    def test_a_run_count_below_one_is_refused(self, repo, tmp_path):
        stub = tmp_path / "stub-evolver"
        stub.write_text("#!/bin/sh\n")
        result = CliRunner().invoke(
            main,
            [
                "--tool", "file_tools",
                "--hermes-repo", str(repo),
                "--evolver-cmd", str(stub),
                "--repro-runs", "0",
                "--dry-run",
            ],
        )
        assert result.exit_code != 0

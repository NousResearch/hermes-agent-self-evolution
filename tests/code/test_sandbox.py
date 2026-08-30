"""The code-evolution containment boundary, held to its own claims.

Two kinds of test live here, split by what they need from the machine.

The plumbing tests run everywhere: they use a permissive enforcer that wraps
nothing, because what they check - the environment allowlist, the disposable
checkout's shape, candidate path bounding, fail-closed refusal, cleanup on
timeout and crash - is this package's own code and must hold on a machine
with no sandbox backend at all.

The adversarial tests run only where a real enforcer exists (bubblewrap on
Linux, sandbox-exec on macOS) and are skipped elsewhere, which is honest:
they prove what the *kernel* refuses, and a machine that cannot enforce is
exactly the machine where the fail-closed tests above are the ones that
matter. Each one plays a hostile evolver: a stub that tries to read a parent
secret, read the operator's home directory, write outside the workspace, or
reach the network - while also emitting a well-formed candidate, so the same
run shows the legitimate contract still works from inside the boundary.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from evolution.code import sandbox as sbx
from evolution.code.evolve_tool_code import (
    UNSANDBOXED,
    Candidate,
    EvolverError,
    EvolverJob,
    ExternalEvolver,
    evolve_tool_code,
)
from evolution.code.sandbox import (
    CodeSandbox,
    SandboxError,
    SandboxUnavailable,
    bounded_candidate_path,
    sandbox_environment,
)


class PermissiveEnforcer:
    """Wraps nothing. Exists so the plumbing is testable without a kernel."""

    name = "permissive-test"

    def command(self, argv, *, workspace, writable, read_roots, allow_network):
        return list(argv)


BASELINE = "def add(a, b):\n    return a + b\n"
TUNED = BASELINE + "# tuned\n"


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo(tmp_path):
    """A miniature hermes-agent checkout with a probe that marks execution.

    ``test_probe.py`` writes a file into the directory pytest runs in, so
    after a full evolution pass the presence or absence of that file in the
    *real* repo is direct evidence of where candidate code actually executed.
    """
    root = tmp_path / "hermes-agent"
    (root / "tools").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "tools" / "__init__.py").write_text("")
    (root / "tools" / "file_tools.py").write_text(BASELINE)
    (root / "tests" / "test_smoke.py").write_text(
        "def test_smoke():\n    assert True\n"
    )
    (root / "tests" / "test_probe.py").write_text(
        "import pathlib\n\n"
        "def test_probe_marks_where_it_ran():\n"
        "    pathlib.Path('probe-executed.txt').write_text('here')\n"
        "    assert True\n"
    )
    git(root.parent, "-c", "init.defaultBranch=main", "init", "-q", str(root))
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test Runner")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    return root


STUB_HEADER = """import argparse
import json
import os
import pathlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

job = json.loads(pathlib.Path(args.job).read_text())
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)
"""

# Every hostile stub still emits one legitimate candidate: the same run that
# proves an escape is refused proves the honest contract still functions.
EMIT_CANDIDATE = (
    "print(json.dumps({'source': job['source'] + '# mutated\\n'}))\n"
)


def make_stub(workdir, body, name="stub_evolver.py"):
    """Write an evolver stub *inside the workdir*.

    The workdir is the one host directory a sandboxed run can read and write,
    so it is the only place a stub is guaranteed to be visible from inside
    the boundary on every platform.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / name
    path.write_text(STUB_HEADER + body)
    return [sys.executable, str(path)]


def make_job(source=BASELINE):
    return EvolverJob(
        target_path="tools/file_tools.py",
        source=source,
        objective="tune the implementation",
        iterations=1,
    )


class FakeEvolver:
    """In-process stand-in for the evolver CLI, for end-to-end plumbing."""

    def __init__(self, *sources):
        self.sources = list(sources)

    def propose(self, job):
        return [
            Candidate(index=i, source=source, origin="fake")
            for i, source in enumerate(self.sources, start=1)
        ]


@pytest.fixture
def contained_tempdirs(monkeypatch, tmp_path):
    """Route mkdtemp into a directory this test owns, so leftover workspaces
    are countable instead of lost among the system's other temp files."""
    holder = tmp_path / "sandbox-tempdirs"
    holder.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(holder))
    return holder


@pytest.fixture
def permissive(monkeypatch):
    """Make sandbox construction succeed everywhere, without enforcement."""
    monkeypatch.setattr(sbx, "available_enforcer", lambda: PermissiveEnforcer())


# ──────────────────────────────────────────────────────────────────────────
# The environment allowlist
# ──────────────────────────────────────────────────────────────────────────


PARENT_ENV = {
    "PATH": "/usr/bin",
    "LANG": "en_US.UTF-8",
    "AWS_SECRET_ACCESS_KEY": "not-for-children",
    "GITHUB_TOKEN": "ghp_nope",
    "OPENAI_API_KEY": "sk-parent",
    "SSH_AUTH_SOCK": "/tmp/agent.sock",
}


class TestTheEnvironmentIsAnAllowlist:
    def test_nothing_crosses_but_the_allowlist(self, tmp_path):
        env = sandbox_environment(
            PARENT_ENV, home=tmp_path / "h", tmp=tmp_path / "t"
        )
        assert env["PATH"] == "/usr/bin"
        assert env["LANG"] == "en_US.UTF-8"
        for secret in (
            "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN",
            "OPENAI_API_KEY", "SSH_AUTH_SOCK",
        ):
            assert secret not in env
        assert "not-for-children" not in json.dumps(env)

    def test_a_variable_crosses_only_when_named(self, tmp_path):
        env = sandbox_environment(
            PARENT_ENV,
            home=tmp_path / "h",
            tmp=tmp_path / "t",
            passthrough=("OPENAI_API_KEY",),
        )
        assert env["OPENAI_API_KEY"] == "sk-parent"
        assert "GITHUB_TOKEN" not in env

    def test_a_named_variable_absent_in_the_parent_is_skipped(self, tmp_path):
        env = sandbox_environment(
            {"PATH": "/usr/bin"},
            home=tmp_path / "h",
            tmp=tmp_path / "t",
            passthrough=("NEVER_SET_ANYWHERE",),
        )
        assert "NEVER_SET_ANYWHERE" not in env

    def test_home_and_tmpdir_point_into_the_workspace(self, tmp_path):
        env = sandbox_environment(
            PARENT_ENV, home=tmp_path / "h", tmp=tmp_path / "t"
        )
        assert env["HOME"] == str(tmp_path / "h")
        assert env["TMPDIR"] == str(tmp_path / "t")

    def test_git_is_pinned_inside_the_sandbox(self, tmp_path):
        env = sandbox_environment(
            PARENT_ENV, home=tmp_path / "h", tmp=tmp_path / "t"
        )
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert env["GIT_CONFIG_NOSYSTEM"] == "1"
        assert env["GIT_CONFIG_GLOBAL"] == str(tmp_path / "h" / ".gitconfig")


# ──────────────────────────────────────────────────────────────────────────
# Candidate path bounding
# ──────────────────────────────────────────────────────────────────────────


class TestCandidatePathsAreBounded:
    def test_an_absolute_path_outside_the_roots_is_refused(self, tmp_path):
        assert (
            bounded_candidate_path(
                "/etc/passwd", base=tmp_path, roots=(tmp_path,)
            )
            is None
        )

    def test_a_dotdot_component_is_refused_outright(self, tmp_path):
        inner = tmp_path / "inner"
        inner.mkdir()
        (tmp_path / "target.py").write_text("x = 1\n")
        # Even though it would land inside the root: upward traversal is not
        # something a candidate has any legitimate reason to express.
        assert (
            bounded_candidate_path(
                "../target.py", base=inner, roots=(tmp_path,)
            )
            is None
        )

    def test_a_relative_path_resolves_against_the_base(self, tmp_path):
        (tmp_path / "candidate.py").write_text("x = 1\n")
        resolved = bounded_candidate_path(
            "candidate.py", base=tmp_path, roots=(tmp_path,)
        )
        assert resolved == Path(os.path.realpath(tmp_path / "candidate.py"))

    def test_an_absolute_path_inside_a_root_is_accepted(self, tmp_path):
        (tmp_path / "candidate.py").write_text("x = 1\n")
        resolved = bounded_candidate_path(
            str(tmp_path / "candidate.py"), base=tmp_path, roots=(tmp_path,)
        )
        assert resolved is not None

    def test_a_symlink_pointing_out_of_the_roots_is_refused(self, tmp_path):
        secret = tmp_path / "outside" / "secret.py"
        secret.parent.mkdir()
        secret.write_text("password = 'hunter2'\n")
        root = tmp_path / "root"
        root.mkdir()
        (root / "link.py").symlink_to(secret)
        assert (
            bounded_candidate_path("link.py", base=root, roots=(root,)) is None
        )

    def test_the_adapter_refuses_an_escaping_path_in_every_mode(self, tmp_path):
        """Bounding holds even with the sandbox explicitly waived."""
        workdir = tmp_path / "work"
        cmd = make_stub(
            workdir,
            "print(json.dumps({'path': '/etc/passwd'}))\n"
            + EMIT_CANDIDATE,
        )
        evolver = ExternalEvolver(
            cmd, repo=tmp_path, workdir=workdir, sandbox=UNSANDBOXED
        )
        candidates = evolver.propose(make_job())
        assert len(candidates) == 1
        assert candidates[0].source.endswith("# mutated\n")

    def test_the_adapter_refuses_a_dotdot_escape(self, tmp_path):
        (tmp_path / "beyond.py").write_text("# beyond\n")
        workdir = tmp_path / "inner" / "work"
        cmd = make_stub(
            workdir,
            "print(json.dumps({'path': '../../beyond.py'}))\n",
        )
        evolver = ExternalEvolver(
            cmd, repo=tmp_path / "inner", workdir=workdir, sandbox=UNSANDBOXED
        )
        with pytest.raises(EvolverError, match="no candidates"):
            evolver.propose(make_job())

    def test_a_symlinked_candidate_file_is_refused(self, tmp_path):
        """Collection runs outside the sandbox, with this process's own
        privileges. A symlink dropped in candidates/ must not become an
        arbitrary-read primitive against the operator's files."""
        secret = tmp_path / "outside" / "id_rsa"
        secret.parent.mkdir()
        secret.write_text("PRIVATE KEY MATERIAL")
        workdir = tmp_path / "inner" / "work"
        cmd = make_stub(
            workdir,
            "(out / 'candidates').mkdir(parents=True, exist_ok=True)\n"
            f"os.symlink({str(secret)!r}, out / 'candidates' / '001.py')\n",
        )
        evolver = ExternalEvolver(
            cmd, repo=tmp_path / "inner", workdir=workdir, sandbox=UNSANDBOXED
        )
        with pytest.raises(EvolverError, match="no candidates"):
            evolver.propose(make_job())

    def test_a_symlinked_candidates_jsonl_is_ignored(self, tmp_path):
        decoy = tmp_path / "outside.jsonl"
        decoy.write_text(json.dumps({"source": "# from outside\n"}) + "\n")
        workdir = tmp_path / "inner" / "work"
        cmd = make_stub(
            workdir,
            f"os.symlink({str(decoy)!r}, out / 'candidates.jsonl')\n",
        )
        evolver = ExternalEvolver(
            cmd, repo=tmp_path / "inner", workdir=workdir, sandbox=UNSANDBOXED
        )
        with pytest.raises(EvolverError, match="no candidates"):
            evolver.propose(make_job())

    def test_executable_read_roots_cover_a_virtualenv(self):
        roots = sbx.executable_read_roots(sys.executable)
        directory = Path(os.path.realpath(sys.executable)).parent
        assert directory in roots
        if directory.name == "bin":
            assert directory.parent in roots

    def test_command_read_roots_cover_the_whole_command_line(self, tmp_path):
        script = tmp_path / "scripts" / "evolver.py"
        script.parent.mkdir()
        script.write_text("print('hi')\n")
        roots = sbx.command_read_roots(
            [sys.executable, str(script), "--flag", "not-a-path"]
        )
        assert Path(os.path.realpath(sys.executable)).parent in roots
        assert Path(os.path.realpath(script.parent)) in roots
        # Flags that name nothing on disk contribute nothing.
        assert all("not-a-path" not in str(r) for r in roots)

    def test_bubblewrap_binds_every_read_root_back(self, tmp_path):
        """The tmpfs over /tmp and home would otherwise hide an evolver or
        interpreter living there; the argv must mount each read root back,
        wherever it lives, before the writable binds."""
        under_tmp = tmp_path / "evolver-lives-here"
        under_tmp.mkdir()
        under_home = Path.home() / ".hse-imaginary-venv"
        workspace = tmp_path / "ws"
        workspace.mkdir()
        argv = sbx.BubblewrapEnforcer().command(
            ["evolver", "--job", "j"],
            workspace=workspace,
            writable=[workspace],
            read_roots=[under_tmp, under_home],
            allow_network=False,
        )
        joined = " ".join(argv)
        for root in (under_tmp, under_home):
            real = str(Path(os.path.realpath(root)))
            assert f"--ro-bind {real} {real}" in joined
            # After the tmpfs mounts, so the bind wins over the shadowing.
            assert joined.index("--tmpfs") < joined.index(f"--ro-bind {real}")
        # Writable binds come after every read-only layer.
        real_ws = str(Path(os.path.realpath(workspace)))
        assert joined.rindex(f"--bind {real_ws} {real_ws}") > joined.index(
            "--ro-bind /"
        )
        assert "--unshare-net" in argv

    def test_read_roots_cover_a_symlinked_interpreter_store(self, tmp_path):
        """uv's layout: the venv python points through a version-alias
        directory symlink (``store/cpython-3.12 -> cpython-3.12.14``), and
        the alias's parent must land in the roots - which realpath-based
        derivation cannot report, because realpath resolves the alias away
        without recording that its name has to be resolvable too."""
        tmp_path = Path(os.path.realpath(tmp_path))
        store = tmp_path / "store"
        install = store / "cpython-3.12.14"
        (install / "bin").mkdir(parents=True)
        binary = install / "bin" / "python3.12"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        (store / "cpython-3.12").symlink_to("cpython-3.12.14")
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        link = venv_bin / "python"
        link.symlink_to(store / "cpython-3.12" / "bin" / "python3.12")
        roots = sbx.executable_read_roots(str(link))
        assert venv_bin in roots  # the venv link's own parent
        assert store in roots  # the alias link's parent
        assert install / "bin" in roots  # the resolved binary's directory
        assert install in roots  # the prefix above bin/

    def test_bubblewrap_never_rebinds_the_hidden_mounts_themselves(
        self, tmp_path
    ):
        """A symlink chain can name home itself or / as a needed parent (a
        link sitting directly in the home root); binding those back would
        unseal the sandbox, so the enforcer refuses them and only roots
        strictly inside a hidden mount punch through."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        under_home = Path.home() / ".hse-imaginary-venv"
        argv = sbx.BubblewrapEnforcer().command(
            ["evolver"],
            workspace=workspace,
            writable=[workspace],
            read_roots=[
                Path("/"), Path("/tmp"), Path("/run"),
                Path.home(), under_home,
            ],
            allow_network=False,
        )
        joined = " ".join(argv)
        home = str(Path(os.path.realpath(Path.home())))
        real_under = str(Path(os.path.realpath(under_home)))
        assert f"--ro-bind {real_under} {real_under}" in joined
        assert f"--ro-bind {home} {home}" not in joined
        assert "--ro-bind /tmp /tmp" not in joined
        assert "--ro-bind /run /run" not in joined
        # The base root mount stays the only bind of / itself.
        assert joined.count("--ro-bind / /") == 1


# ──────────────────────────────────────────────────────────────────────────
# Fail closed
# ──────────────────────────────────────────────────────────────────────────


class TestNoEnforcerMeansNoRun:
    def test_require_enforcer_names_the_waiver(self, monkeypatch):
        monkeypatch.setattr(sbx, "available_enforcer", lambda: None)
        with pytest.raises(SandboxUnavailable, match="--unsandboxed"):
            sbx.require_enforcer()

    def test_the_adapter_refuses_rather_than_running_bare(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(sbx, "available_enforcer", lambda: None)
        workdir = tmp_path / "work"
        cmd = make_stub(workdir, EMIT_CANDIDATE)
        evolver = ExternalEvolver(cmd, repo=tmp_path, workdir=workdir)
        with pytest.raises(SandboxUnavailable):
            evolver.propose(make_job())

    def test_the_run_exits_5_before_mutating_anything(
        self, repo, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(sbx, "available_enforcer", lambda: None)
        code = evolve_tool_code(
            "file_tools",
            hermes_repo=str(repo),
            evolver=FakeEvolver(TUNED),
            output_root=tmp_path / "output",
        )
        assert code == 5
        # Refused before the organism started: no branch was created.
        branches = git(
            repo, "branch", "--list", "evolve/*", "--format=%(refname:short)"
        ).stdout.strip()
        assert branches == ""


# ──────────────────────────────────────────────────────────────────────────
# The disposable checkout
# ──────────────────────────────────────────────────────────────────────────


class TestTheDisposableCheckout:
    def test_it_is_a_clone_at_the_baseline_with_no_way_out(self, repo):
        baseline = git(repo, "rev-parse", "HEAD").stdout.strip()
        with CodeSandbox(
            repo, baseline_sha=baseline, enforcer=PermissiveEnforcer()
        ) as box:
            assert (box.checkout / ".git").exists()
            head = git(box.checkout, "rev-parse", "HEAD").stdout.strip()
            assert head == baseline
            remotes = git(box.checkout, "remote").stdout.strip()
            assert remotes == ""
            helper = git(
                box.checkout, "config", "--get", "credential.helper"
            ).stdout.strip()
            assert helper == ""

    def test_the_baseline_source_overrides_the_cloned_content(self, repo):
        dirty = BASELINE + "# uncommitted operator work\n"
        with CodeSandbox(
            repo,
            target_relpath="tools/file_tools.py",
            baseline_source=dirty,
            enforcer=PermissiveEnforcer(),
        ) as box:
            on_disk = (box.checkout / "tools" / "file_tools.py").read_text()
            assert on_disk == dirty

    def test_write_target_refuses_paths_outside_the_checkout(self, repo):
        with CodeSandbox(repo, enforcer=PermissiveEnforcer()) as box, \
                pytest.raises(SandboxError, match="outside"):
            box.write_target("../escape.py", "x = 1\n")

    def test_a_plain_directory_is_copied(self, tmp_path):
        source = tmp_path / "bare"
        source.mkdir()
        (source / "module.py").write_text("x = 1\n")
        with CodeSandbox(source, enforcer=PermissiveEnforcer()) as box:
            assert (box.checkout / "module.py").read_text() == "x = 1\n"

    def test_cleanup_removes_the_workspace(self, repo):
        box = CodeSandbox(repo, enforcer=PermissiveEnforcer())
        workspace = box.workspace
        assert workspace.exists()
        box.cleanup()
        assert not workspace.exists()
        box.cleanup()  # idempotent

    def test_the_workdir_survives_cleanup(self, repo, tmp_path):
        workdir = tmp_path / "artifacts"
        box = CodeSandbox(repo, workdir=workdir, enforcer=PermissiveEnforcer())
        (workdir / "job.json").write_text("{}")
        box.cleanup()
        assert (workdir / "job.json").exists()

    def test_run_refuses_a_caller_supplied_environment(self, repo):
        with CodeSandbox(repo, enforcer=PermissiveEnforcer()) as box, \
                pytest.raises(SandboxError, match="env_passthrough"):
            box.run([sys.executable, "-c", "pass"], env={"X": "1"})


class TestTimeoutAndCrashCleanUp:
    def test_a_timed_out_evolver_leaves_no_workspace_behind(
        self, repo, tmp_path, permissive, contained_tempdirs
    ):
        workdir = tmp_path / "work"
        cmd = make_stub(workdir, "import time\ntime.sleep(30)\n")
        evolver = ExternalEvolver(cmd, repo=repo, workdir=workdir, timeout=1)
        before = git(repo, "rev-parse", "HEAD").stdout.strip()
        with pytest.raises(EvolverError, match="timed out"):
            evolver.propose(make_job())
        assert list(contained_tempdirs.iterdir()) == []
        assert git(repo, "rev-parse", "HEAD").stdout.strip() == before
        assert (repo / "tools" / "file_tools.py").read_text() == BASELINE

    def test_a_crashed_evolver_leaves_no_workspace_and_an_untouched_repo(
        self, repo, tmp_path, permissive, contained_tempdirs
    ):
        workdir = tmp_path / "work"
        cmd = make_stub(
            workdir, "sys.stderr.write('kaboom\\n')\nsys.exit(2)\n"
        )
        evolver = ExternalEvolver(cmd, repo=repo, workdir=workdir)
        with pytest.raises(EvolverError, match="no candidates"):
            evolver.propose(make_job())
        assert list(contained_tempdirs.iterdir()) == []
        assert (repo / "tools" / "file_tools.py").read_text() == BASELINE


# ──────────────────────────────────────────────────────────────────────────
# End to end, without OS enforcement
# ──────────────────────────────────────────────────────────────────────────


class TestCandidateCodeNeverExecutesInTheRealRepo:
    def test_a_full_pass_runs_its_gates_in_the_checkout(
        self, repo, tmp_path, permissive, contained_tempdirs
    ):
        code = evolve_tool_code(
            "file_tools",
            hermes_repo=str(repo),
            evolver=FakeEvolver(TUNED),
            output_root=tmp_path / "output",
        )
        assert code == 0

        # The probe test writes a marker into whatever directory pytest runs
        # in. The real repo not having one is the direct observation that
        # baseline and candidate suites executed in the disposable checkout.
        assert not (repo / "probe-executed.txt").exists()

        # The deliverable is unchanged: lineage branch in the real repo.
        branches = git(
            repo, "branch", "--list", "evolve/code/*",
            "--format=%(refname:short)",
        ).stdout.strip()
        assert branches != ""

        # The workspace is gone; the artifacts and their provenance are not.
        assert list(contained_tempdirs.iterdir()) == []
        metrics_files = list((tmp_path / "output").rglob("metrics.json"))
        assert len(metrics_files) == 1
        recorded = json.loads(metrics_files[0].read_text())["sandbox"]
        assert recorded["enforcer"] == "permissive-test"
        assert recorded["allow_network"] is False

    def test_an_unsandboxed_run_records_that_it_was_one(
        self, repo, tmp_path
    ):
        code = evolve_tool_code(
            "file_tools",
            hermes_repo=str(repo),
            evolver=FakeEvolver(TUNED),
            output_root=tmp_path / "output",
            sandbox=UNSANDBOXED,
        )
        assert code == 0
        metrics_files = list((tmp_path / "output").rglob("metrics.json"))
        recorded = json.loads(metrics_files[0].read_text())["sandbox"]
        assert recorded == {"enforcer": None, "unsandboxed": True}


# ──────────────────────────────────────────────────────────────────────────
# Adversarial, with the real enforcer
# ──────────────────────────────────────────────────────────────────────────


needs_enforcer = pytest.mark.skipif(
    sbx.available_enforcer() is None,
    reason="no OS sandbox backend on this machine - the fail-closed tests "
    "above are the ones that bind here",
)


def run_hostile_stub(repo, workdir, body):
    """Drive a hostile stub through the real, ephemeral, enforced sandbox."""
    cmd = make_stub(workdir, body + EMIT_CANDIDATE)
    evolver = ExternalEvolver(cmd, repo=repo, workdir=workdir)
    return evolver.propose(make_job())


@needs_enforcer
class TestTheKernelHoldsTheBoundary:
    def test_a_parent_secret_is_not_observable(
        self, repo, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HSE_SANDBOX_PROBE_SECRET", "swordfish-9000")
        workdir = tmp_path / "work"
        candidates = run_hostile_stub(
            repo,
            workdir,
            "(out / 'env.json').write_text(json.dumps(dict(os.environ)))\n",
        )
        assert len(candidates) == 1
        seen = json.loads((workdir / "evolver_out" / "env.json").read_text())
        assert "HSE_SANDBOX_PROBE_SECRET" not in seen
        assert "swordfish-9000" not in json.dumps(seen)
        assert "PATH" in seen  # the allowlist crossed; the secret did not

    def test_the_operators_home_is_not_readable(self, repo, tmp_path):
        marker = f"TOP-SECRET-{os.getpid()}"
        sentinel = Path.home() / f".hse-read-probe-{os.getpid()}"
        sentinel.write_text(marker)
        try:
            workdir = tmp_path / "work"
            candidates = run_hostile_stub(
                repo,
                workdir,
                "try:\n"
                f"    content = open({str(sentinel)!r}).read()\n"
                "    result = {'ok': True, 'content': content}\n"
                "except Exception as exc:\n"
                "    result = {'ok': False, 'error': type(exc).__name__}\n"
                "(out / 'read.json').write_text(json.dumps(result))\n",
            )
            assert len(candidates) == 1
            result = json.loads(
                (workdir / "evolver_out" / "read.json").read_text()
            )
            assert result["ok"] is False
            assert marker not in json.dumps(result)
        finally:
            sentinel.unlink()

    def test_writes_do_not_escape_the_workspace(self, repo, tmp_path):
        pid = os.getpid()
        targets = [
            Path.home() / f".hse-escape-{pid}.txt",
            repo / "escaped.txt",
            Path(f"/tmp/hse-escape-{pid}.txt"),
        ]
        try:
            workdir = tmp_path / "work"
            body = "".join(
                "try:\n"
                f"    open({str(t)!r}, 'w').write('escaped')\n"
                "except Exception:\n"
                "    pass\n"
                for t in targets
            )
            candidates = run_hostile_stub(repo, workdir, body)
            assert len(candidates) == 1
            # Whether each attempt was denied outright or landed in a
            # namespace that no longer exists, the property is the same:
            # nothing arrived on the host.
            for target in targets:
                assert not target.exists(), target
        finally:
            for target in targets:
                target.unlink(missing_ok=True)

    def test_an_evolver_script_outside_the_workdir_still_runs(
        self, repo, tmp_path
    ):
        """The operator's command must work from wherever it lives.

        ``--evolver-cmd`` is routinely an interpreter plus a script, and the
        script can sit anywhere - including under a directory the sandbox
        mounts a tmpfs over. The command's own paths are bound back
        read-only, so the run the operator asked for is not refused by the
        boundary that exists to protect it.
        """
        elsewhere = tmp_path / "elsewhere"
        cmd = make_stub(elsewhere, EMIT_CANDIDATE)
        workdir = tmp_path / "work"
        evolver = ExternalEvolver(cmd, repo=repo, workdir=workdir)
        candidates = evolver.propose(make_job())
        assert len(candidates) == 1
        assert candidates[0].source.endswith("# mutated\n")

    def test_an_interpreter_reached_through_home_scoped_links_still_runs(
        self, repo, tmp_path
    ):
        """The regression for uv-managed interpreters under the user's home.

        uv reaches an interpreter through a version-alias directory symlink
        under home (``~/.local/share/uv/python/cpython-3.12-<platform> ->
        cpython-3.12.14-<platform>``), so the exec path crosses a link name
        that only exists inside the hidden home. Binding the realpath alone
        leaves that name unresolvable behind the tmpfs and the evolver dies
        with execvp ENOENT before it starts; the chain's parents have to be
        bound back too. This drives the same shape - a directory symlink
        under the real home in the middle of the interpreter's path -
        through the real enforcer.
        """
        real = Path(os.path.realpath(sys.executable))
        store = Path.home() / f".hse-uv-like-{os.getpid()}"
        store.mkdir()
        try:
            (store / "real-bin").symlink_to(real.parent)
            launcher = store / "real-bin" / real.name
            assert launcher.exists()
            workdir = tmp_path / "work"
            cmd = make_stub(workdir, EMIT_CANDIDATE)
            cmd[0] = str(launcher)
            evolver = ExternalEvolver(cmd, repo=repo, workdir=workdir)
            candidates = evolver.propose(make_job())
            assert len(candidates) == 1
            assert candidates[0].source.endswith("# mutated\n")
        finally:
            shutil.rmtree(store, ignore_errors=True)

    def test_the_network_is_unreachable(self, repo, tmp_path):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            workdir = tmp_path / "work"
            candidates = run_hostile_stub(
                repo,
                workdir,
                "import socket\n"
                "s = socket.socket()\n"
                "s.settimeout(5)\n"
                "try:\n"
                f"    s.connect(('127.0.0.1', {port}))\n"
                "    result = {'connected': True}\n"
                "except Exception as exc:\n"
                "    result = {'connected': False, "
                "'error': type(exc).__name__}\n"
                "(out / 'net.json').write_text(json.dumps(result))\n",
            )
            assert len(candidates) == 1
            result = json.loads(
                (workdir / "evolver_out" / "net.json").read_text()
            )
            assert result["connected"] is False
        finally:
            listener.close()

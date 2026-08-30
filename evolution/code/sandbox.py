"""Confine everything the code phase executes that it did not write itself.

Phase 4 runs two kinds of process no other phase runs: an external mutation
engine chosen by configuration, and the target repo's own test suite with a
candidate rewrite applied. Both execute content that arrived from outside this
package - the evolver binary is whatever ``--evolver-cmd`` names, and a
candidate is whatever that binary produced - so both run behind the same
boundary, and the boundary is enforced by the operating system rather than
promised by this module's own code.

The boundary has four independent legs, and each holds on its own:

1. **A disposable checkout.** The evolver and every fitness gate run against a
   clone of the target repo made at the baseline commit, with no remotes and
   with the credential helper disabled repo-locally. There is nothing to push
   to, no helper to hand out a token, and nothing a hung or hostile process
   can corrupt that a ``rm -rf`` of the workspace does not erase. The
   operator's checkout is never the working directory of an untrusted
   process.

2. **An explicit environment allowlist.** The child environment is built from
   named variables, never copied from the parent, so an API key, a
   ``GITHUB_TOKEN`` or an ``AWS_SECRET_ACCESS_KEY`` in the operator's shell
   is not observable inside the sandbox. ``HOME`` and ``TMPDIR`` point into
   the workspace. Anything the evolver legitimately needs - its own model
   key, say - is passed by name, which makes granting it a decision instead
   of a default.

3. **OS-level enforcement.** On Linux the child runs under `bubblewrap
   <https://github.com/containers/bubblewrap>`_: the filesystem is read-only
   outside the workspace, the invoking user's home directory is replaced by
   an empty tmpfs, and the process tree gets its own network, PID, IPC and
   UTS namespaces, so there is no network unless granted. On macOS it runs
   under ``sandbox-exec`` with a Seatbelt profile denying all network,
   denying reads of the user's home directory outside the interpreter and
   workspace, and denying writes everywhere but the workspace. The wrapper
   is applied to the *whole* child, so nothing a candidate spawns from
   inside pytest escapes it either.

4. **Fail closed.** When no enforcer is available the run refuses to start
   and says what to install. It does not fall back to running the untrusted
   process bare and hoping - an explicit ``--unsandboxed`` is the only way
   to get the old behaviour, and asking for it by name is the point.

Candidate ``path`` values are bounded separately (see
:func:`bounded_candidate_path`) because that check must hold even in
``--unsandboxed`` mode: a candidate that names ``/etc/passwd`` or walks
``..`` out of the run's own directories is refused everywhere, enforcement
or not.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path

__all__ = [
    "SANDBOX_ENV_PASSTHROUGH",
    "UNSANDBOXED",
    "BubblewrapEnforcer",
    "CodeSandbox",
    "SandboxError",
    "SandboxUnavailable",
    "SeatbeltEnforcer",
    "available_enforcer",
    "bounded_candidate_path",
    "command_read_roots",
    "executable_read_roots",
    "require_enforcer",
    "sandbox_environment",
]


class SandboxError(RuntimeError):
    """Raised when the sandbox cannot be built or driven."""


class SandboxUnavailable(SandboxError):
    """Raised when no OS-level enforcer exists and none was waived."""


class _Unsandboxed:
    """Sentinel: run without any sandbox, exactly as the code did before.

    A distinct type rather than ``None`` so that "no sandbox was passed,
    build one and fail closed" and "the operator explicitly waived the
    sandbox" can never be confused, in code or in a grep.
    """

    def __repr__(self) -> str:  # pragma: no cover - repr only
        return "UNSANDBOXED"


UNSANDBOXED = _Unsandboxed()


# ──────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────

# What crosses into the sandbox by default. Everything else in the parent
# environment - keys, tokens, session variables - simply does not exist on
# the other side. PATH crosses because the child must find its interpreter;
# the locale variables cross because their absence changes text encoding
# behaviour, which is a correctness matter, not a secret.
SANDBOX_ENV_PASSTHROUGH: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
)


def sandbox_environment(
    source: Mapping[str, str] | None = None,
    *,
    home: Path,
    tmp: Path,
    passthrough: Sequence[str] = (),
) -> dict[str, str]:
    """Build the child environment from an explicit allowlist.

    *passthrough* names additional parent variables to copy - the mechanism
    behind ``--sandbox-env``. A name listed but absent in the parent is
    silently skipped, so a run script can name a key it only sometimes has.

    The git variables pin every git a child might run to configuration inside
    the sandbox: no system gitconfig, a global gitconfig that lives in the
    sandbox home, and no terminal prompt for credentials - a git that wants
    to ask for a password fails instead of hanging the run.
    """
    parent = os.environ if source is None else source
    env: dict[str, str] = {}
    for name in (*SANDBOX_ENV_PASSTHROUGH, *passthrough):
        value = parent.get(name)
        if value is not None:
            env[name] = value
    env["HOME"] = str(home)
    env["TMPDIR"] = str(tmp)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_CONFIG_GLOBAL"] = str(Path(home) / ".gitconfig")
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ──────────────────────────────────────────────────────────────────────────
# Enforcers
# ──────────────────────────────────────────────────────────────────────────


def _quiet_run(cmd: Sequence[str], timeout: int = 15) -> bool:
    """True when *cmd* runs and exits 0. Used only for capability probes."""
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _profile_path_ok(path: Path) -> Path:
    """Refuse paths a Seatbelt profile string cannot carry safely."""
    text = str(path)
    if '"' in text or "\n" in text:
        raise SandboxError(
            f"cannot express {text!r} in a sandbox profile - "
            "move the workspace to a path without quotes or newlines"
        )
    return path


class SeatbeltEnforcer:
    """macOS ``sandbox-exec`` with a generated Seatbelt profile.

    The profile allows by default and then denies the three things that
    matter, because a deny-default profile kills the dynamic linker before
    ``main`` ever runs:

    - ``network*`` outright (unless network was granted),
    - ``file-write*`` everywhere except the writable roots and ``/dev`` -
      including the macOS per-user temp tree, which is why the sandbox
      environment points ``TMPDIR`` into the workspace,
    - ``file-read-data`` under the invoking user's home directory, excepting
      the interpreter and the run's own directories. Denying only the data
      operation leaves path metadata traversable, which the loader needs to
      reach an interpreter that lives under the home directory at all; file
      contents and directory listings under home stay unreadable.
    """

    name = "sandbox-exec (macOS Seatbelt)"
    hint = "sandbox-exec ships with macOS; if this probe fails, report it"

    def __init__(self) -> None:
        self._profiles: dict[tuple, Path] = {}

    @staticmethod
    def probe() -> bool:
        """True when a trivial allow-everything profile can run a process."""
        return _quiet_run(
            ["sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"]
        )

    def command(
        self,
        argv: Sequence[str],
        *,
        workspace: Path,
        writable: Sequence[Path],
        read_roots: Sequence[Path],
        allow_network: bool,
    ) -> list[str]:
        """Wrap *argv* so the kernel enforces the boundary around it."""
        key = (
            str(workspace),
            tuple(str(p) for p in writable),
            tuple(str(p) for p in read_roots),
            allow_network,
        )
        profile = self._profiles.get(key)
        if profile is None:
            profile = self._write_profile(
                workspace,
                writable=writable,
                read_roots=read_roots,
                allow_network=allow_network,
            )
            self._profiles[key] = profile
        return ["sandbox-exec", "-f", str(profile), *argv]

    def _write_profile(
        self,
        workspace: Path,
        *,
        writable: Sequence[Path],
        read_roots: Sequence[Path],
        allow_network: bool,
    ) -> Path:
        home = Path(os.path.realpath(os.path.expanduser("~")))
        write_ok = [
            _profile_path_ok(Path(os.path.realpath(p))) for p in writable
        ]
        # Everything writable must also be readable, and so must the
        # interpreter trees; only paths under home need naming, because
        # nothing outside home is read-denied in the first place.
        read_ok = sorted(
            {
                str(p)
                for p in (
                    *(Path(os.path.realpath(r)) for r in read_roots),
                    *write_ok,
                )
                if str(p).startswith(f"{home}{os.sep}")
            }
        )
        write_excepts = "".join(
            f' (require-not (subpath "{p}"))' for p in write_ok
        )
        read_excepts = "".join(
            f' (require-not (subpath "{_profile_path_ok(Path(p))}"))'
            for p in read_ok
        )
        lines = ["(version 1)", "(allow default)"]
        if not allow_network:
            lines.append("(deny network*)")
        lines.append(
            "(deny file-write* (require-all"
            ' (subpath "/")'
            f"{write_excepts}"
            ' (require-not (subpath "/dev"))))'
        )
        lines.append(
            "(deny file-read-data (require-all"
            f' (subpath "{_profile_path_ok(home)}")'
            f"{read_excepts}))"
        )
        profile = workspace / f"seatbelt-{len(self._profiles)}.sb"
        profile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return profile


class BubblewrapEnforcer:
    """Linux ``bwrap``: read-only root, hidden home, unshared namespaces.

    The whole filesystem is bind-mounted read-only, the invoking user's home
    directory is replaced with an empty tmpfs, ``/tmp`` and ``/run`` are
    fresh tmpfs, the read roots (interpreter trees, the evolver command's
    directory) are bound back read-only wherever they live, and only the
    writable roots are bound read-write. The child gets its own PID, IPC and
    UTS namespaces always, and its own empty network namespace unless
    network was granted.
    """

    name = "bubblewrap"
    hint = "install it (apt-get install bubblewrap / dnf install bubblewrap)"

    @staticmethod
    def probe() -> bool:
        """True when bwrap can build the namespaces this enforcer needs.

        A functional probe rather than ``which``: on kernels or LSM policies
        that refuse unprivileged user namespaces, the binary exists and every
        invocation fails, and that machine must count as *no enforcer*.
        """
        if shutil.which("bwrap") is None:
            return False
        return _quiet_run(
            [
                "bwrap", "--die-with-parent", "--unshare-net",
                "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                "/bin/sh", "-c", "true",
            ]
        )

    def command(
        self,
        argv: Sequence[str],
        *,
        workspace: Path,
        writable: Sequence[Path],
        read_roots: Sequence[Path],
        allow_network: bool,
    ) -> list[str]:
        """Wrap *argv* so the kernel enforces the boundary around it."""
        home = Path(os.path.realpath(os.path.expanduser("~")))
        cmd = ["bwrap", "--die-with-parent", "--unshare-pid",
               "--unshare-ipc", "--unshare-uts"]
        if not allow_network:
            cmd.append("--unshare-net")
        cmd += ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                "--tmpfs", "/tmp", "--tmpfs", "/run", "--tmpfs", str(home)]
        # Bind order matters: the tmpfs mounts go first, then every read root
        # is mounted back over them read-only, and the writable roots go last
        # so they win over every read-only layer. All read roots are bound,
        # not only the ones under home: the interpreter or the evolver command
        # can just as easily live under /tmp or /run, where the tmpfs would
        # otherwise hide it - re-binding a path that was already visible is
        # harmless.
        never_rebound = {"/", "/tmp", "/run", str(home)}
        for root in sorted(
            {str(Path(os.path.realpath(r))) for r in read_roots}
        ):
            # A read root may punch through a hidden mount only from strictly
            # inside it: binding home itself, an ancestor of home, /tmp or
            # /run wholesale would undo the hiding that defines this sandbox.
            # A command whose resolution genuinely needs one of those bound
            # fails closed at exec time instead of running unsealed.
            if root in never_rebound or str(home).startswith(f"{root}{os.sep}"):
                continue
            cmd += ["--ro-bind", root, root]
        for path in writable:
            real = str(Path(os.path.realpath(path)))
            cmd += ["--bind", real, real]
        cmd += ["--", *argv]
        return cmd


@functools.cache
def _probed(kind: str) -> bool:
    if kind == "seatbelt":
        return SeatbeltEnforcer.probe()
    if kind == "bwrap":
        return BubblewrapEnforcer.probe()
    return False


def available_enforcer():
    """The enforcer this platform can actually run, or None.

    Probed, not guessed: each candidate runs a trivial command under full
    enforcement once per process, so "the binary exists but the kernel
    refuses it" counts as unavailable rather than failing later mid-run.
    """
    if sys.platform == "darwin" and _probed("seatbelt"):
        return SeatbeltEnforcer()
    if sys.platform.startswith("linux") and _probed("bwrap"):
        return BubblewrapEnforcer()
    return None


def require_enforcer():
    """An enforcer, or :class:`SandboxUnavailable` naming the way out."""
    enforcer = available_enforcer()
    if enforcer is not None:
        return enforcer
    if sys.platform == "darwin":
        detail = SeatbeltEnforcer.hint
    elif sys.platform.startswith("linux"):
        detail = BubblewrapEnforcer.hint
    else:
        detail = f"no sandbox backend exists for {sys.platform}"
    raise SandboxUnavailable(
        "no OS-level sandbox is available on this machine, and code evolution "
        "runs a contributor-controlled evolver plus candidate code, so it "
        f"refuses to run unprotected ({detail}). Pass --unsandboxed to accept "
        "running that code with no isolation."
    )


# ──────────────────────────────────────────────────────────────────────────
# The sandbox
# ──────────────────────────────────────────────────────────────────────────


class CodeSandbox:
    """A disposable checkout, a scrubbed environment and an enforcer, together.

    Built once per run and shared by the evolver subprocess and every fitness
    gate, so a single boundary confines all of them. Use as a context manager
    or call :meth:`cleanup` from a ``finally``; either way the workspace and
    everything the sandboxed processes wrote inside it are deleted, which is
    also what makes a timed-out or crashed child cheap - the state it can
    have corrupted is state that was about to be erased.

    *workdir*, when given, is the run's artifact directory: it is made
    writable inside the sandbox so the evolver can leave its job file and
    candidates there, and it survives cleanup because it lives outside the
    workspace on purpose - a failed run's request and output are exactly what
    the operator inspects next.
    """

    def __init__(
        self,
        repo: Path,
        *,
        target_relpath: str | None = None,
        baseline_sha: str | None = None,
        baseline_source: str | None = None,
        workdir: Path | None = None,
        read_roots: Sequence[Path] = (),
        env_passthrough: Sequence[str] = (),
        allow_network: bool = False,
        enforcer=None,
        parent_env: Mapping[str, str] | None = None,
        git_timeout: int = 600,
    ) -> None:
        self.repo = Path(repo).expanduser().resolve()
        self.enforcer = enforcer if enforcer is not None else require_enforcer()
        self.allow_network = allow_network
        self.env_passthrough = tuple(env_passthrough)
        self.git_timeout = git_timeout

        self.workspace = Path(
            tempfile.mkdtemp(prefix="hse-code-sandbox-")
        ).resolve()
        self.home = self.workspace / "home"
        self.tmp = self.workspace / "tmp"
        self.checkout = self.workspace / "checkout"
        self.home.mkdir()
        self.tmp.mkdir()
        (self.home / ".gitconfig").write_text("", encoding="utf-8")

        self.workdir = Path(workdir).expanduser().resolve() if workdir else None
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)

        # The interpreter running this process is the one tests and stub
        # commands are launched with, so its trees are always readable -
        # including the parent of every symlink its path resolves through,
        # which for a uv-managed python includes a version-alias directory
        # under the invoking user's home.
        interpreter_roots = [
            Path(sys.prefix),
            Path(sys.base_prefix),
            *executable_read_roots(sys.executable),
        ]
        self.read_roots = tuple(
            Path(os.path.realpath(Path(p).expanduser()))
            for p in (*read_roots, *interpreter_roots)
        )
        self.writable = tuple(
            p for p in (self.workspace, self.workdir) if p is not None
        )

        self.environment = sandbox_environment(
            parent_env,
            home=self.home,
            tmp=self.tmp,
            passthrough=self.env_passthrough,
        )

        try:
            self._build_checkout(baseline_sha)
            if target_relpath and baseline_source is not None:
                self.write_target(target_relpath, baseline_source)
            self.target_relpath = target_relpath
        except BaseException:
            self.cleanup()
            raise

    # ── checkout ────────────────────────────────────────────────────────

    def _git(self, *args: str, cwd: Path) -> None:
        try:
            proc = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=str(cwd),
                timeout=self.git_timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SandboxError(f"git {' '.join(args)} failed: {exc}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise SandboxError(f"git {' '.join(args)} failed: {detail}")

    def _build_checkout(self, baseline_sha: str | None) -> None:
        """Clone the repo at the baseline, then cut every path outward.

        ``--no-hardlinks`` matters: a local clone shares object files by
        hardlink by default, and a shared inode is a write path from the
        sandbox back into the real repository's object store.

        A plain directory copies instead of cloning - callers driving the
        adapter directly against a bare tree keep working - and gets the
        same containment from the enforcer even without git history.
        """
        if (self.repo / ".git").exists():
            self._git(
                "clone", "--quiet", "--no-hardlinks",
                str(self.repo), str(self.checkout),
                cwd=self.workspace,
            )
            if baseline_sha:
                self._git(
                    "checkout", "--quiet", "--detach", baseline_sha,
                    cwd=self.checkout,
                )
            # No remote to push to, no helper to produce a credential, and
            # no maintenance process spawning in a directory that is about
            # to be deleted.
            self._git("remote", "remove", "origin", cwd=self.checkout)
            self._git("config", "credential.helper", "", cwd=self.checkout)
            self._git("config", "gc.auto", "0", cwd=self.checkout)
        else:
            shutil.copytree(self.repo, self.checkout)

    def write_target(self, relpath: str, source: str) -> Path:
        """Write *source* to *relpath* inside the checkout and return the path.

        This is how a candidate reaches the disposable tree for evaluation,
        and how a dirty-tree baseline (``allow_dirty``) overrides the cloned
        commit's content with what the operator actually had on disk.
        """
        path = (self.checkout / relpath).resolve()
        try:
            path.relative_to(self.checkout)
        except ValueError as exc:
            raise SandboxError(
                f"{relpath!r} resolves outside the sandbox checkout"
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    def import_file(self, path: Path) -> Path:
        """Copy a host file into the workspace so the sandbox can read it.

        The reproduction script lives wherever the operator keeps it, which
        the sandbox cannot see; the copy is what the job spec and the fitness
        runs point at.
        """
        source = Path(path).expanduser().resolve()
        dest_dir = self.workspace / "imported"
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / source.name
        shutil.copy2(source, dest)
        return dest

    # ── execution ───────────────────────────────────────────────────────

    def run(
        self,
        cmd: Sequence[str],
        *,
        capture_output: bool = True,
        text: bool = True,
        timeout: float | None = None,
        cwd: Path | str | None = None,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """Run *cmd* inside the sandbox. Signature-compatible with the
        ``subprocess.run`` calls the fitness runners make, so it drops in as
        their ``exec_fn``.

        The environment is always the sandbox's own; a caller-supplied one
        is refused rather than merged, because merging is exactly the
        full-inheritance mistake this class exists to end.
        """
        if "env" in kwargs:
            raise SandboxError(
                "CodeSandbox.run builds the child environment itself - "
                "pass variables through env_passthrough instead"
            )
        wrapped = self.enforcer.command(
            list(cmd),
            workspace=self.workspace,
            writable=self.writable,
            read_roots=self.read_roots,
            allow_network=self.allow_network,
        )
        return subprocess.run(  # noqa: PLW1510 - callers read returncode
            wrapped,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else str(self.checkout),
            env=self.environment,
            **kwargs,
        )

    def describe(self) -> dict:
        """What confined this run, for the metrics artifact."""
        return {
            "enforcer": getattr(self.enforcer, "name", type(self.enforcer).__name__),
            "allow_network": self.allow_network,
            "env_passthrough": list(self.env_passthrough),
            "checkout": str(self.checkout),
        }

    # ── lifecycle ───────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Delete the workspace. Idempotent, and safe on the error path."""
        shutil.rmtree(self.workspace, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.cleanup()
        return False


# ──────────────────────────────────────────────────────────────────────────
# Candidate path bounding
# ──────────────────────────────────────────────────────────────────────────


def _chain_parents(path: Path) -> tuple[Path, list[Path]]:
    """Follow *path* component by component, the way the kernel does.

    Returns the fully resolved target and the real parent directory of every
    symlink met on the way - intermediate components included, which is what
    ``os.path.realpath`` cannot report. The distinction is load-bearing: a
    uv-managed interpreter is reached through a version-alias directory
    symlink (``.../uv/python/cpython-3.12-<platform>`` pointing at the
    ``cpython-3.12.14-<platform>`` install), and a sandbox that binds back
    only the realpath result leaves the alias name unresolvable behind the
    tmpfs hiding the home directory it lives under. Resolution stops at the
    kernel's own 40-link bound; a looping path is returned as far as it got
    and fails at exec time exactly as it would outside the sandbox.
    """
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = Path(path.anchor)
    parents: list[Path] = []
    pending = deque(path.parts[1:])
    hops = 0
    while pending:
        part = pending.popleft()
        if part == ".":
            continue
        if part == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / part
        if os.path.islink(candidate):
            hops += 1
            if hops > 40:
                return candidate, parents
            parents.append(resolved)
            target = Path(os.readlink(candidate))
            if target.is_absolute():
                resolved = Path(target.anchor)
                pending.extendleft(reversed(target.parts[1:]))
            else:
                pending.extendleft(reversed(target.parts))
            continue
        resolved = candidate
    return resolved, parents


def executable_read_roots(executable: str) -> list[Path]:
    """The directories a sandboxed child must read to run *executable*.

    The real parent of every symlink in the binary's resolution chain -
    uv's version-alias directory included - then the resolved binary's own
    directory, and, when that directory is a ``bin/``, the prefix above it,
    because a virtualenv or uv-managed interpreter loads its standard
    library and site-packages from siblings of ``bin``, and a read
    exception covering only the binary would let it exec and then die on
    the first import.
    """
    found = shutil.which(executable) or executable
    target, roots = _chain_parents(Path(found).expanduser())
    directory = target.parent
    roots.append(directory)
    if directory.name == "bin":
        roots.append(directory.parent)
    return roots


def command_read_roots(argv: Sequence[str]) -> list[Path]:
    """The directories a sandboxed child must read to run *argv* - all of it.

    The executable's roots, and the parent directory of every later argument
    that names something on disk, because an evolver command is routinely an
    interpreter plus a script (``python /path/to/evolver.py``) or carries a
    config file by path, and a wrapper that makes the binary visible while
    hiding the script it was told to run refuses work the operator asked
    for. Arguments that name nothing on disk are flags and are skipped.
    """
    if not argv:
        return []
    roots = executable_read_roots(argv[0])
    for token in argv[1:]:
        candidate = Path(token).expanduser()
        try:
            exists = candidate.exists()
        except OSError:
            exists = False
        if exists:
            # Followed through its symlink chain exactly like the
            # executable: a script reached via a linked directory needs the
            # link's parent visible too.
            target, chain = _chain_parents(candidate)
            roots.extend(chain)
            roots.append(target.parent)
    return roots


def bounded_candidate_path(
    raw: str, *, base: Path, roots: Sequence[Path]
) -> Path | None:
    """Resolve a candidate-supplied ``path`` or refuse it.

    Returns the resolved path when it lies inside one of *roots* - the run's
    output directory and the checkout the evolver saw - and None otherwise.
    Refused outright, before any resolution: a ``..`` component (a candidate
    has no legitimate reason to walk upward) and, via the containment check,
    every absolute path outside the roots and every symlink that points out
    of them. ``os.path.realpath`` rather than ``Path.resolve`` because the
    containment claim must hold for the file actually opened, links and all.
    """
    candidate = Path(raw)
    if ".." in candidate.parts:
        return None
    if not candidate.is_absolute():
        candidate = Path(base) / candidate
    resolved = Path(os.path.realpath(candidate))
    for root in roots:
        real_root = Path(os.path.realpath(root))
        if resolved == real_root or real_root in resolved.parents:
            return resolved
    return None

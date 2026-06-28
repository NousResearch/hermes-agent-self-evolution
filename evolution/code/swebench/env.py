"""Docker-backed SWEbench environment — official eval grading via warm container.

The candidate source lives on disk inside a running container (at /testbed).
Verdict is produced by running spec.eval_script, which reinstalls the package,
resets+reapplies the gold test_patch (so F2P tests exist), and runs them between
sentinels. get_logs_eval then parses the captured output; get_eval_tests_report
maps to dataset ids. One eval per candidate state; every mutation resets the cache.

graded_report() attaches ``_eval_ok`` (bool) to distinguish infra failures
(build broken / timeout / sentinels missing) from real test outcomes.
"""

from __future__ import annotations

import io
import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional

from evolution.code.worktree import TestRun

# Stale .pyc files between same-size writes (same root cause as worktree.py:43).
_DOCKER_ENV = {"PYTHONDONTWRITEBYTECODE": "1"}

_EXEC_TIMEOUT = 600  # seconds for eval_script or read commands
_TAIL_CHARS = 8000   # feed to LM: enough for the failing-test summary/traceback
_LOG_SENTINEL = "eval_log"  # prefix for host-side temp log files


def _put_file(container, path: str, content: str) -> None:
    """Write a text file into the container via tar archive (avoids shell quoting)."""
    data = content.encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        info = tarfile.TarInfo(name=path.lstrip("/"))
        info.size = len(data)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    stream.seek(0)
    container.put_archive("/", stream.getvalue())


def _exec(container, cmd: str, workdir: str = "/testbed") -> tuple[int, str]:
    """Run a bash command in the container's conda testbed environment."""
    full = (
        "source /opt/miniconda3/bin/activate && conda activate testbed && " + cmd
    )
    res = container.exec_run(
        ["/bin/bash", "-c", full],
        workdir=workdir,
        environment=_DOCKER_ENV,
    )
    return res.exit_code, res.output.decode(errors="replace")


class SWEbenchEnv:
    """Docker container wrapping a single SWE-bench Lite instance.

    Lifecycle: :meth:`create` → (write_tool / run_test / apply_patch)* → :meth:`destroy`.
    Use as a context manager to guarantee cleanup.

    Verdict is obtained via official grading (eval_script + get_logs_eval +
    get_eval_tests_report), keyed to dataset ids — correct for all repos. The
    result is cached; every mutation that changes the candidate source resets
    self._graded so the next call re-evaluates.
    """

    def __init__(self, instance, spec, container):
        self._inst = instance
        self._f2p: tuple[str, ...] = instance.fail_to_pass
        self._p2p: tuple[str, ...] = instance.pass_to_pass
        self.repo: str = instance.repo
        self.gold_file: str = instance.gold_file
        self._spec = spec
        self._container = container
        self._graded: Optional[dict] = None
        self._last_eval_output: str = ""
        self.emulated: bool = False

    # -- construction -------------------------------------------------------

    @classmethod
    def create(cls, instance, *, run_id: str = "extval") -> "SWEbenchEnv":
        """Pull the prebuilt x86_64 instance image and start it under Rosetta.

        SWE-bench publishes a per-instance x86_64 image; we pull that and run it under
        Docker Desktop's Rosetta translation rather than building per-arch locally. This
        is uniform across repos — no repo is unreachable for lack of arm64 wheels, and no
        local image build means no build-failure noise — and Rosetta evaluates the
        numerical C-extension repos (xarray, scikit-learn) cleanly where QEMU segfaulted.
        ``emulated`` is always True: x86_64 under Rosetta on an arm64 host.
        """
        import docker
        from swebench.harness.docker_build import build_container, setup_logger
        from swebench.harness.test_spec.test_spec import make_test_spec

        client = docker.from_env()
        log_path = Path(tempfile.mktemp(prefix=f"sweb_{instance.instance_id}_", suffix=".log"))
        spec = make_test_spec(instance.raw, namespace="swebench", arch="x86_64")
        logger = setup_logger(spec.instance_id, log_path)
        if not spec.is_remote_image:
            raise RuntimeError(
                f"no prebuilt x86_64 image published for {instance.instance_id} "
                f"(spec.is_remote_image=False); see {log_path}")
        container = build_container(spec, client, run_id, logger, nocache=False)
        container.start()
        env = cls(instance, spec, container)
        env.emulated = True
        return env

    # -- verdict (the core) -------------------------------------------------

    def graded_report(self) -> dict:
        """Evaluate the current candidate via official eval_script + grading.

        Caches the result; resets on any write/patch/reset that changes source.
        Sets ``_eval_ok`` on the returned dict: False means infra failure (build
        broken, timeout, or no parseable sentinels) rather than a real test outcome.
        """
        if self._graded is not None:
            return self._graded

        from swebench.harness.grading import get_logs_eval, get_eval_tests_report

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"{_LOG_SENTINEL}_{self._spec.instance_id}_",
            suffix=".log",
            delete=False,
        ) as fh:
            log_fp = fh.name

        # Write eval_script into the container and run it with a container-side
        # timeout so a wedged test/install cannot hang the campaign.
        eval_path = f"/tmp/eval_{self._spec.instance_id}.sh"
        _put_file(self._container, eval_path, self._spec.eval_script)
        exit_code, raw_output = _exec(
            self._container,
            f"timeout {_EXEC_TIMEOUT} bash {eval_path}",
            workdir="/testbed",
        )

        timed_out = (exit_code == 124)

        try:
            Path(log_fp).write_text(raw_output)
            status_map, applied_ok = get_logs_eval(self._spec, log_fp)
        finally:
            os.unlink(log_fp)

        self._last_eval_output = raw_output

        gold = {
            "FAIL_TO_PASS": list(self._f2p),
            "PASS_TO_PASS": list(self._p2p),
        }
        report = get_eval_tests_report(status_map, gold)

        eval_ok = bool(applied_ok) and bool(status_map) and not timed_out
        report["_eval_ok"] = eval_ok
        report["_timed_out"] = timed_out

        self._graded = report
        return self._graded

    # -- gate seam -----------------------------------------------------------

    def failing_tests(self, *ids: str) -> set[str]:
        """Return the subset of ``ids`` that failed in the last graded eval.

        Conservative: an id not present in success ∪ failure of either scope
        (i.e. never graded) is treated as FAILING so it cannot certify as fixed.
        """
        rep = self.graded_report()
        graded_ok = (
            set(rep["FAIL_TO_PASS"]["success"])
            | set(rep["PASS_TO_PASS"]["success"])
        )
        failures = (
            set(rep["FAIL_TO_PASS"]["failure"])
            | set(rep["PASS_TO_PASS"]["failure"])
        )
        graded = graded_ok | failures
        return {i for i in ids if i not in graded or i in failures}

    # -- run_test (for repair engine) ----------------------------------------

    def run_test(self, *ids: str, **_kw) -> TestRun:
        """Run one graded eval; return a TestRun whose output feeds the LM."""
        t0 = time.monotonic()
        rep = self.graded_report()
        duration = time.monotonic() - t0

        f2p_fail = set(rep["FAIL_TO_PASS"]["failure"])
        p2p_fail = set(rep["PASS_TO_PASS"]["failure"])
        all_fail = f2p_fail | p2p_fail
        passed = not any(i in all_fail for i in ids) if ids else not all_fail

        raw = self._last_eval_output
        # Slice to the region between sentinels so the LM sees test failures,
        # not install/setup noise.  Falls back to the whole log if sentinels absent.
        from swebench.harness.constants import START_TEST_OUTPUT, END_TEST_OUTPUT
        start_idx = raw.find(START_TEST_OUTPUT)
        end_idx = raw.find(END_TEST_OUTPUT)
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            raw = raw[start_idx + len(START_TEST_OUTPUT):end_idx]
        output = raw[-_TAIL_CHARS:] if len(raw) > _TAIL_CHARS else raw

        # Best-effort exit code: 0 if passed, 1 if not.
        exit_code = 0 if passed else 1
        return TestRun(passed=passed, output=output,
                       duration_seconds=duration, exit_code=exit_code)

    # -- file I/O -----------------------------------------------------------

    def read_tool(self, relpath: str) -> str:
        """Read a file from /testbed inside the container."""
        _, out = _exec(self._container, f"cat /testbed/{relpath}")
        return out

    def write_tool(self, relpath: str, src: str) -> None:
        """Write ``src`` into /testbed/``relpath`` and invalidate the verdict cache."""
        _put_file(self._container, f"/testbed/{relpath}", src)
        self._graded = None

    def apply_patch(self, diff: str) -> None:
        """Apply a unified diff to /testbed and invalidate the verdict cache.

        Raises RuntimeError unless git apply succeeds.
        """
        patch_path = "/tmp/candidate.patch"
        _put_file(self._container, patch_path, diff)
        code, out = _exec(self._container, f"git apply -v {patch_path}")
        if code != 0:
            raise RuntimeError(f"git apply failed (exit {code}):\n{out[-2000:]}")
        self._graded = None

    def base_source(self, relpath: str) -> str:
        """The committed (base_commit) content of ``relpath`` from git HEAD."""
        _, out = _exec(self._container, f"git show HEAD:{relpath}")
        return out

    def changed_files(self) -> list[str]:
        """Repo-relative source paths with a working-tree change.

        Filters build artifacts so only real source changes are visible to the gate.
        """
        _, out = _exec(
            self._container,
            "git status --porcelain --untracked-files=all",
        )
        result: list[str] = []
        for line in out.splitlines():
            if len(line) <= 3:
                continue
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1].strip()
            if any(
                skip in path
                for skip in ("__pycache__", ".pytest_cache", ".egg-info", "coverage")
            ):
                continue
            if path.endswith((".pyc", ".egg-link")):
                continue
            result.append(path)
        return result

    def reset_file(self, relpath: str) -> None:
        """Restore ``relpath`` to HEAD and purge any pycache; invalidates cache."""
        _exec(self._container, f"git checkout -- {relpath}")
        # Purge any adjacent __pycache__ so the restored file runs fresh.
        stem = relpath.rsplit("/", 1)[-1].removesuffix(".py")
        _exec(self._container, f"find /testbed -path '*/__pycache__/{stem}*.pyc' -delete")
        self._graded = None

    def assert_authoritative(self, package: str) -> None:
        """Confirm /testbed exists (lightweight sanity check for Docker env)."""
        code, _ = _exec(self._container, "test -d /testbed")
        if code != 0:
            raise RuntimeError(
                f"assert_authoritative({package!r}): /testbed not found in container"
            )

    # -- lifecycle ----------------------------------------------------------

    def destroy(self) -> None:
        """Stop and remove the container (best-effort)."""
        try:
            self._container.stop(timeout=10)
        except Exception:
            pass
        try:
            self._container.remove(force=True)
        except Exception:
            pass

    def __enter__(self) -> "SWEbenchEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.destroy()

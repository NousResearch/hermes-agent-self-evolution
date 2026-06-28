"""Tests for SWEbenchEnv.

Pure unit tests (no Docker) cover the changed_files filter and cache-invalidation
contract. The slow integration tests build real containers for a flask and django
Lite instance and verify that the buggy base reproduces the F2P failure and that
applying the gold patch resolves it without P2P regression.
"""

from __future__ import annotations

import importlib.util

import pytest


def _swebench_stack_available() -> bool:
    """True only when the optional ``swebench`` extra is installed (docker + swebench +
    datasets). CI runs a plain ``uv sync`` without the extra, so these integration tests
    must skip there rather than error on the lazy imports inside the env/loader."""
    return all(
        importlib.util.find_spec(mod) is not None
        for mod in ("docker", "swebench", "datasets")
    )


_requires_swebench_stack = pytest.mark.skipif(
    not _swebench_stack_available(),
    reason="integration: requires the 'swebench' extra (uv sync --extra swebench) "
    "and a running Docker daemon",
)


# ---------------------------------------------------------------------------
# Pure unit tests (no Docker)
# ---------------------------------------------------------------------------


class _FakeExec:
    """Steers _exec return values per-command prefix."""

    def __init__(self, responses: dict[str, tuple[int, str]]):
        self._responses = responses
        self.calls: list[str] = []

    def __call__(self, container, cmd: str, workdir: str = "/testbed") -> tuple[int, str]:
        self.calls.append(cmd)
        for prefix, val in self._responses.items():
            if cmd.startswith(prefix):
                return val
        return 0, ""


class _FakeContainer:
    pass


class _FakeInstance:
    fail_to_pass = ("tests/test_x.py::test_fail",)
    pass_to_pass = ("tests/test_x.py::test_pass",)
    repo = "pallets/flask"
    gold_file = "src/flask/app.py"
    raw = {}


def _make_env(exec_fn=None):
    """Build a SWEbenchEnv with fake container; patch _exec if provided."""
    from evolution.code.swebench import env as env_mod
    from unittest.mock import patch

    inst = _FakeInstance()

    class _FakeSpec:
        instance_id = "test__instance-1"
        eval_script = "#!/bin/bash\necho done"

    with patch.object(env_mod, "_exec", exec_fn or (lambda *a, **kw: (0, ""))):
        e = env_mod.SWEbenchEnv.__new__(env_mod.SWEbenchEnv)
        e._inst = inst
        e._f2p = inst.fail_to_pass
        e._p2p = inst.pass_to_pass
        e.repo = inst.repo
        e.gold_file = inst.gold_file
        e._spec = _FakeSpec()
        e._container = _FakeContainer()
        e._graded = None
        e._last_eval_output = ""
    return e


def test_changed_files_filters_build_artifacts(monkeypatch):
    """changed_files must strip __pycache__, .pyc, .egg-info, .pytest_cache, coverage."""
    from evolution.code.swebench import env as env_mod

    status_output = (
        " M src/flask/app.py\n"
        "?? src/flask/__pycache__/app.cpython-311.pyc\n"
        "?? src/flask/__pycache__/\n"
        "?? .pytest_cache/\n"
        "?? src/flask.egg-info/SOURCES.txt\n"
        "?? coverage/\n"
        " M src/flask/cli.py\n"
        " M src/flask/cli.pyc\n"
    )

    def fake_exec(container, cmd, workdir="/testbed"):
        return 0, status_output

    monkeypatch.setattr(env_mod, "_exec", fake_exec)
    e = _make_env()
    # Re-attach the monkeypatched _exec (the env uses the module-level name).
    result = e.changed_files()
    assert "src/flask/app.py" in result
    assert "src/flask/cli.py" in result
    # All artifact paths must be absent.
    for path in result:
        assert "__pycache__" not in path
        assert ".pytest_cache" not in path
        assert ".egg-info" not in path
        assert "coverage" not in path
        assert not path.endswith(".pyc")


def test_write_tool_invalidates_graded_cache(monkeypatch, tmp_path):
    """write_tool must reset _graded so the next graded_report re-evaluates."""
    from evolution.code.swebench import env as env_mod

    monkeypatch.setattr(env_mod, "_put_file", lambda *a, **kw: None)
    monkeypatch.setattr(env_mod, "_exec", lambda *a, **kw: (0, ""))

    e = _make_env()
    e._graded = {"FAIL_TO_PASS": {"success": [], "failure": []},
                 "PASS_TO_PASS": {"success": [], "failure": []}}
    assert e._graded is not None

    e.write_tool("src/flask/app.py", "# new content")
    assert e._graded is None


def test_apply_patch_invalidates_graded_cache(monkeypatch):
    """apply_patch must reset _graded on success."""
    from evolution.code.swebench import env as env_mod

    monkeypatch.setattr(env_mod, "_put_file", lambda *a, **kw: None)
    monkeypatch.setattr(env_mod, "_exec", lambda *a, **kw: (0, ""))

    e = _make_env()
    e._graded = {"FAIL_TO_PASS": {"success": [], "failure": []},
                 "PASS_TO_PASS": {"success": [], "failure": []}}

    e.apply_patch("--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n")
    assert e._graded is None


def test_apply_patch_raises_on_failure(monkeypatch):
    """apply_patch must raise RuntimeError and NOT clear cache on git apply failure."""
    from evolution.code.swebench import env as env_mod

    monkeypatch.setattr(env_mod, "_put_file", lambda *a, **kw: None)
    monkeypatch.setattr(env_mod, "_exec", lambda *a, **kw: (1, "error: patch failed"))

    e = _make_env()
    pre = {"FAIL_TO_PASS": {"success": [], "failure": []},
           "PASS_TO_PASS": {"success": [], "failure": []}}
    e._graded = pre

    with pytest.raises(RuntimeError, match="git apply failed"):
        e.apply_patch("bad diff")
    # Cache was not cleared — the source was not changed.
    assert e._graded is pre


def test_reset_file_invalidates_graded_cache(monkeypatch):
    """reset_file must reset _graded."""
    from evolution.code.swebench import env as env_mod

    monkeypatch.setattr(env_mod, "_exec", lambda *a, **kw: (0, ""))

    e = _make_env()
    e._graded = {"FAIL_TO_PASS": {"success": [], "failure": []},
                 "PASS_TO_PASS": {"success": [], "failure": []}}

    e.reset_file("src/flask/app.py")
    assert e._graded is None


def test_failing_tests_resolves_from_graded_report(monkeypatch):
    """failing_tests returns ids in F2P.failure ∪ P2P.failure, plus ungraded ids."""
    from evolution.code.swebench import env as env_mod

    e = _make_env()
    e._graded = {
        "FAIL_TO_PASS": {"success": [], "failure": ["tests/t.py::test_a"]},
        "PASS_TO_PASS": {"success": ["tests/t.py::test_b"], "failure": []},
        "_eval_ok": True,
        "_timed_out": False,
    }

    result = e.failing_tests("tests/t.py::test_a", "tests/t.py::test_b", "tests/t.py::test_c")
    # test_a: explicit failure → included
    # test_b: explicit success → excluded
    # test_c: never graded (not in success ∪ failure) → conservative: included
    assert result == {"tests/t.py::test_a", "tests/t.py::test_c"}


# ---------------------------------------------------------------------------
# Integration tests (require Docker)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@_requires_swebench_stack
def test_flask_bug_reproduces_and_gold_resolves():
    """Full integration: build container, confirm bug, apply gold, confirm green."""
    import time

    from evolution.code.swebench.loader import load_single_file_lite
    from evolution.code.swebench.env import SWEbenchEnv

    instances = load_single_file_lite()
    flask_insts = [i for i in instances if i.repo == "pallets/flask"]
    assert flask_insts, "No flask instance found in single-file Lite set"
    inst = flask_insts[0]

    t0 = time.monotonic()
    with SWEbenchEnv.create(inst) as env:
        assert env.emulated is True  # every instance runs the prebuilt x86_64 image under Rosetta

        # Write buggy base (restores base_commit state — eval_script will apply test_patch).
        buggy_src = env.base_source(inst.gold_file)
        env.write_tool(inst.gold_file, buggy_src)

        failing_buggy = env.failing_tests(*inst.fail_to_pass)
        assert failing_buggy, (
            f"Expected F2P failures on buggy base for {inst.instance_id}; "
            f"got none. F2P={inst.fail_to_pass}"
        )

        # Apply the gold patch; cache is already cleared by write_tool above,
        # but apply_patch also clears it.
        env.apply_patch(inst.gold_patch)

        failing_f2p_after = env.failing_tests(*inst.fail_to_pass)
        assert not failing_f2p_after, (
            f"F2P tests still failing after gold patch for {inst.instance_id}: "
            f"{failing_f2p_after}"
        )

        failing_p2p_after = env.failing_tests(*inst.pass_to_pass)
        assert not failing_p2p_after, (
            f"P2P regression after gold patch for {inst.instance_id}: "
            f"{failing_p2p_after}"
        )

    elapsed = time.monotonic() - t0
    print(f"\n[flask integration] wall-clock {elapsed:.0f}s  instance={inst.instance_id}")


@pytest.mark.slow
@_requires_swebench_stack
def test_django_bug_reproduces_and_gold_resolves():
    """Django integration: build container, confirm bug, apply gold, confirm green.

    Django is the harder parser case (django-specific test runner output format);
    this regression-protects the F2P/P2P grading path for django instances.
    """
    import time

    from evolution.code.swebench.loader import load_single_file_lite
    from evolution.code.swebench.env import SWEbenchEnv

    instances = load_single_file_lite(limit=80)
    django_insts = [i for i in instances if i.repo == "django/django"]
    assert django_insts, "No django/django instance found in first 80 single-file Lite instances"
    inst = django_insts[0]

    t0 = time.monotonic()
    with SWEbenchEnv.create(inst) as env:
        # Write buggy base so eval_script starts from the base_commit state.
        buggy_src = env.base_source(inst.gold_file)
        env.write_tool(inst.gold_file, buggy_src)

        failing_buggy = set(inst.fail_to_pass) & env.failing_tests(*inst.fail_to_pass)
        assert failing_buggy, (
            f"Expected at least one F2P failure on buggy base for {inst.instance_id}; "
            f"got none. F2P={inst.fail_to_pass}"
        )

        env.apply_patch(inst.gold_patch)

        failing_f2p_after = env.failing_tests(*inst.fail_to_pass)
        assert not failing_f2p_after, (
            f"F2P tests still failing after gold patch for {inst.instance_id}: "
            f"{failing_f2p_after}"
        )

        failing_p2p_after = env.failing_tests(*inst.pass_to_pass)
        assert not failing_p2p_after, (
            f"P2P regression after gold patch for {inst.instance_id}: "
            f"{failing_p2p_after}"
        )

    elapsed = time.monotonic() - t0
    print(f"\n[django integration] wall-clock {elapsed:.0f}s  instance={inst.instance_id}")


@pytest.mark.slow
@_requires_swebench_stack
def test_xarray_x86_under_rosetta_resolves():
    """Numerical C-extension repo under Rosetta. xarray has no arm64 wheels, so it runs
    via the prebuilt x86_64 image. This regression-protects that the eval runs to
    completion under Rosetta — the exact path QEMU segfaulted on (pandas/numpy C
    extensions): the bug reproduces on the base, the eval is clean (``_eval_ok``), and the
    gold patch resolves F2P with no P2P regression.
    """
    import time

    from evolution.code.swebench.loader import load_single_file_lite
    from evolution.code.swebench.env import SWEbenchEnv

    instances = load_single_file_lite()
    xarray_inst = next(
        (i for i in instances if i.instance_id == "pydata__xarray-3364"), None
    )
    assert xarray_inst is not None, "pydata__xarray-3364 not found in single-file Lite set"

    t0 = time.monotonic()
    with SWEbenchEnv.create(xarray_inst) as env:
        assert env.emulated is True  # x86_64 under Rosetta

        buggy_src = env.base_source(xarray_inst.gold_file)
        env.write_tool(xarray_inst.gold_file, buggy_src)

        buggy_report = env.graded_report()
        assert buggy_report.get("_eval_ok"), (
            f"eval did not run to completion under Rosetta for {xarray_inst.instance_id}: "
            f"_eval_ok={buggy_report.get('_eval_ok')} _timed_out={buggy_report.get('_timed_out')}"
        )

        failing_buggy = set(xarray_inst.fail_to_pass) & env.failing_tests(*xarray_inst.fail_to_pass)
        assert failing_buggy, (
            f"Expected F2P failures on buggy base for {xarray_inst.instance_id}; "
            f"got none. F2P={xarray_inst.fail_to_pass}"
        )

        env.apply_patch(xarray_inst.gold_patch)

        failing_f2p_after = env.failing_tests(*xarray_inst.fail_to_pass)
        assert not failing_f2p_after, (
            f"F2P still failing after gold patch for {xarray_inst.instance_id}: {failing_f2p_after}"
        )
        failing_p2p_after = env.failing_tests(*xarray_inst.pass_to_pass)
        assert not failing_p2p_after, (
            f"P2P regression after gold patch for {xarray_inst.instance_id}: {failing_p2p_after}"
        )

    elapsed = time.monotonic() - t0
    print(f"\n[xarray rosetta] wall-clock {elapsed:.0f}s  instance={xarray_inst.instance_id}")

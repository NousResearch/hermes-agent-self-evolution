"""Guards against subsystems that exist, pass their own tests, and are unreachable.

This is the defect class that produced the original audit's finding 06
(`create_pr` was a config field with no implementation) and then recurred in
this codebase twice: a ~500-line agent-in-the-loop harness with 37 passing
tests that no orchestrator imported, and a canary deployment module with no
caller and no CLI flag. Both were reported as shipped.

Unit tests cannot catch that — every piece worked. These tests assert the
pieces are *connected*.
"""

from __future__ import annotations

import ast
import importlib
import tomllib
from dataclasses import fields
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVOLUTION = REPO / "evolution"


def _python_sources() -> list[Path]:
    return [p for p in EVOLUTION.rglob("*.py") if "__pycache__" not in p.parts]


def _all_source_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in _python_sources())


class TestEveryConfigFieldIsRead:
    """A config field nothing reads is a promise the code does not keep."""

    def test_no_unread_fields(self):
        from evolution.core.config import EvolutionConfig

        config_src = (EVOLUTION / "core" / "config.py").read_text()
        elsewhere = "\n".join(
            p.read_text(encoding="utf-8", errors="replace")
            for p in _python_sources()
            if p != EVOLUTION / "core" / "config.py"
        )

        unread = [
            f.name
            for f in fields(EvolutionConfig)
            if f.name not in elsewhere and config_src.count(f.name) <= 2
        ]
        assert unread == [], (
            "EvolutionConfig fields that nothing outside config.py reads: "
            + ", ".join(unread)
        )


class TestSubsystemsAreReachable:
    """Each of these was built, tested, and wired to nothing at least once."""

    @pytest.mark.parametrize(
        "symbol,module_hint",
        [
            ("AgentEvaluator", "evolution.core.agent_runner"),
            ("tasks_from_examples", "evolution.core.agent_runner"),
            ("deploy_canary", "evolution.deploy.canary"),
            ("evaluate_canary", "evolution.deploy.canary"),
            ("rollback_canary", "evolution.deploy.canary"),
            ("build_default_gate", "evolution.code.admission"),
            ("run_sidecar", "evolution.code.sidecar"),
            ("arm_from_eval_run", "evolution.core.report"),
        ],
    )
    def test_symbol_has_a_caller_outside_its_own_module(self, symbol, module_hint):
        defining = Path(module_hint.replace(".", "/") + ".py")
        callers = [
            p
            for p in _python_sources()
            if p.relative_to(REPO) != defining
            and symbol in p.read_text(encoding="utf-8", errors="replace")
        ]
        assert callers, (
            f"{symbol} is defined in {module_hint} and nothing else in evolution/ "
            "uses it — it is unreachable from any entry point."
        )


class TestCliFlagsExist:
    """Flags whose absence would silently disable a whole subsystem."""

    def _options(self, module: str) -> str:
        mod = importlib.import_module(module)
        return "\n".join(
            str(getattr(p, "opts", "")) for p in getattr(mod.main, "params", [])
        )

    @pytest.mark.parametrize(
        "module,flag",
        [
            ("evolution.skills.evolve_skill", "--agent-eval"),
            ("evolution.skills.evolve_skill", "--canary"),
            ("evolution.skills.evolve_skill", "--create-pr"),
            ("evolution.skills.evolve_skill", "--run-tests"),
            ("evolution.code.evolve_code", "--create-pr"),
            ("evolution.code.evolve_code", "--suggest"),
            ("evolution.monitor.run_rotation", "--phases"),
            ("evolution.deploy.canary_cli", "--evaluate"),
            ("evolution.deploy.canary_cli", "--rollback"),
        ],
    )
    def test_flag_is_exposed(self, module, flag):
        assert flag in self._options(module), f"{module} does not expose {flag}"


class TestEntryPointsResolve:
    """Every console script in pyproject must actually import and be callable."""

    def _scripts(self) -> dict[str, str]:
        data = tomllib.loads((REPO / "pyproject.toml").read_text())
        return data.get("project", {}).get("scripts", {})

    def test_there_are_scripts_declared(self):
        assert self._scripts(), "no console scripts declared"

    def test_each_script_target_exists(self):
        broken = []
        for name, target in self._scripts().items():
            module_path, _, attr = target.partition(":")
            try:
                module = importlib.import_module(module_path)
            except Exception as exc:  # noqa: BLE001
                broken.append(f"{name}: cannot import {module_path} ({exc})")
                continue
            if not callable(getattr(module, attr, None)):
                broken.append(f"{name}: {target} is not callable")
        assert broken == [], "\n".join(broken)

    def test_every_cli_module_has_a_script(self):
        """A CLI nobody can invoke by name is a CLI most people never find."""
        declared = {t.partition(":")[0] for t in self._scripts().values()}
        cli_modules = set()
        for path in _python_sources():
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            has_main = any(
                isinstance(node, ast.FunctionDef) and node.name == "main"
                for node in tree.body
            )
            uses_click = "@click.command()" in path.read_text(encoding="utf-8", errors="replace")
            if has_main and uses_click:
                cli_modules.add(
                    str(path.relative_to(REPO)).replace("/", ".").removesuffix(".py")
                )
        missing = sorted(cli_modules - declared)
        assert missing == [], f"CLI modules with no console script: {missing}"


class TestAgplBoundaryStillHolds:
    """Repeated here because it is a licensing invariant, not a code detail."""

    def test_nothing_imports_the_agpl_engine(self):
        offenders = []
        for path in _python_sources():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip().startswith(("import darwinian_evolver", "from darwinian_evolver")):
                    offenders.append(f"{path.relative_to(REPO)}: {line.strip()}")
        assert offenders == [], "AGPL code imported into the MIT tree:\n" + "\n".join(offenders)

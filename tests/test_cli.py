"""Tests for the ``hermes-evolve`` entry point.

Offline: the only subcommand invoked for real is ``status``, which does pure
source inspection. The phase subcommands are checked for resolvability, not
executed, so no optimization ever starts here.
"""

import click
import pytest
from click.testing import CliRunner

from evolution.cli import _SUBCOMMANDS, cli

TOOL_SOURCE = '''"""Tools."""

READ_SCHEMA = {
    "name": "read_file",
    "description": "Read a file.",
    "parameters": {"type": "object", "properties": {}, "required": []}
}

LONG_SCHEMA = {
    "name": "verbose_tool",
    "description": "%s",
    "parameters": {"type": "object", "properties": {}, "required": []}
}
''' % ("x" * 620)

PROMPT_SOURCE = '''MEMORY_GUIDANCE = (
    "Remember things."
)

SKILLS_GUIDANCE = (
    "Save skills."
)
'''


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "file_tools.py").write_text(TOOL_SOURCE)
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "prompt_builder.py").write_text(PROMPT_SOURCE)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: d\n---\n\nBody\n")
    return tmp_path


@pytest.fixture
def runner():
    return CliRunner()


class TestGroup:
    def test_help_lists_every_phase(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for name in ("skill", "tools", "prompt", "code", "monitor", "status"):
            assert name in result.output

    def test_help_does_not_need_a_repo(self, runner):
        assert runner.invoke(cli, ["--help"]).exit_code == 0

    def test_unknown_command_fails_cleanly(self, runner):
        result = runner.invoke(cli, ["nonsense"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    @pytest.mark.parametrize("name", sorted(_SUBCOMMANDS))
    def test_every_subcommand_resolves_to_a_command(self, name):
        resolved = cli.get_command(click.Context(cli), name)
        assert isinstance(resolved, click.Command), name

    @pytest.mark.parametrize("name", sorted(_SUBCOMMANDS))
    def test_every_subcommand_has_its_own_help(self, runner, name):
        result = runner.invoke(cli, [name, "--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_subcommand_names_match_the_phase_modules(self):
        assert set(_SUBCOMMANDS) == {"skill", "tools", "prompt", "code", "monitor"}


class TestStatus:
    def test_counts_targets_per_phase(self, runner, repo):
        result = runner.invoke(cli, ["status", "--hermes-repo", str(repo)])
        assert result.exit_code == 0
        assert "Skill files" in result.output
        assert "Tool descriptions" in result.output
        assert "Prompt sections" in result.output

    def test_reports_pytest_gate_as_available(self, runner, repo):
        result = runner.invoke(cli, ["status", "--hermes-repo", str(repo)])
        assert "pytest" in result.output
        assert "available" in result.output

    def test_reports_absent_benchmarks_as_unavailable(self, runner, repo):
        result = runner.invoke(cli, ["status", "--hermes-repo", str(repo)])
        assert "tblite" in result.output
        assert "unavailable" in result.output

    def test_flags_over_budget_descriptions(self, runner, repo):
        result = runner.invoke(cli, ["status", "--hermes-repo", str(repo)])
        assert "verbose_tool" in result.output
        assert "exceed" in result.output

    def test_missing_repo_exits_nonzero(self, runner, tmp_path):
        result = runner.invoke(cli, ["status", "--hermes-repo", str(tmp_path / "ghost")])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_undiscoverable_repo_explains_how_to_fix(self, runner, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
        monkeypatch.setattr(
            "evolution.core.config.get_hermes_agent_path",
            lambda: (_ for _ in ()).throw(FileNotFoundError("nope")),
        )
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 1
        assert "HERMES_AGENT_REPO" in result.output

    def test_empty_repo_reports_zeroes_without_crashing(self, runner, tmp_path):
        result = runner.invoke(cli, ["status", "--hermes-repo", str(tmp_path)])
        assert result.exit_code == 0
        assert "Optimization targets" in result.output

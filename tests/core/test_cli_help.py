"""Smoke tests for CLI help."""

from click.testing import CliRunner

from evolution.cli import main


def test_cli_help_lists_core_commands():
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "init" in result.output
    assert "repo" in result.output
    assert "targets" in result.output
    assert "runs" in result.output

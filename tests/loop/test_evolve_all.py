"""Tests for the unified evolve_all CLI (all phases mocked)."""

from unittest.mock import MagicMock, patch, PropertyMock
from click.testing import CliRunner
from evolution.loop.evolve_all import main


def test_dry_run_does_not_call_phases():
    runner = CliRunner()
    with patch("evolution.loop.evolve_all.load_history") as mock_load, \
         patch("evolution.loop.evolve_all.run_phases") as mock_run:
        mock_load.return_value = MagicMock()
        type(mock_load.return_value).runs = PropertyMock(return_value=[])
        mock_load.return_value.latest_per_phase.return_value = {}
        result = runner.invoke(main, ["--dry-run"])
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    mock_run.assert_not_called()


def test_phases_filter():
    runner = CliRunner()
    with patch("evolution.loop.evolve_all.load_history") as mock_load, \
         patch("evolution.loop.evolve_all.run_phases") as mock_run:
        mock_load.return_value = MagicMock()
        type(mock_load.return_value).runs = PropertyMock(return_value=[])
        mock_load.return_value.latest_per_phase.return_value = {}
        mock_run.return_value = MagicMock(status="success", phases={}, total_elapsed=0)
        mock_run.return_value.summary_metrics.return_value = {"total_improvement": 0, "phases_run": 2, "phases_skipped": 0, "phases_failed": 0}
        result = runner.invoke(main, ["--phases", "skills,tools", "--force"])
    assert result.exit_code == 0
    _, kwargs = mock_run.call_args
    assert kwargs["phases"] == ["skills", "tools"]


def test_force_flag_runs_all():
    runner = CliRunner()
    with patch("evolution.loop.evolve_all.load_history") as mock_load, \
         patch("evolution.loop.evolve_all.run_phases") as mock_run:
        mock_load.return_value = MagicMock()
        type(mock_load.return_value).runs = PropertyMock(return_value=[MagicMock(phases={"skills": {}})])
        mock_load.return_value.latest_per_phase.return_value = {"skills": "r1"}
        mock_run.return_value = MagicMock(status="success", phases={}, total_elapsed=0)
        mock_run.return_value.summary_metrics.return_value = {"total_improvement": 0, "phases_run": 4, "phases_skipped": 0, "phases_failed": 0}
        result = runner.invoke(main, ["--force"])
    assert result.exit_code == 0


def test_up_to_date_message():
    runner = CliRunner()
    with patch("evolution.loop.evolve_all.load_history") as mock_load:
        mock_load.return_value = MagicMock()
        type(mock_load.return_value).runs = PropertyMock(return_value=[])
        mock_load.return_value.latest_per_phase.return_value = {
            "skills": "r1", "tools": "r1", "guidance": "r1", "params": "r1",
        }
        result = runner.invoke(main)
    assert result.exit_code == 0
    assert "up-to-date" in result.output.lower()


def test_history_appended_after_run():
    runner = CliRunner()
    with patch("evolution.loop.evolve_all.load_history") as mock_load, \
         patch("evolution.loop.evolve_all.append_run") as mock_append, \
         patch("evolution.loop.evolve_all.run_phases") as mock_run:
        mock_load.return_value = MagicMock()
        type(mock_load.return_value).runs = PropertyMock(return_value=[])
        mock_load.return_value.latest_per_phase.return_value = {}
        mock_run.return_value = MagicMock(status="success", phases={}, total_elapsed=0)
        mock_run.return_value.summary_metrics.return_value = {"total_improvement": 0, "phases_run": 4, "phases_skipped": 0, "phases_failed": 0}
        result = runner.invoke(main, ["--force", "--write-back"])
    assert result.exit_code == 0
    mock_append.assert_called_once()

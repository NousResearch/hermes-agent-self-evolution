"""CLI integration tests for evolution.validation.closed_loop."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from evolution.validation.agent_runner import AgentRunResult, TaskRunContext
from evolution.validation.closed_loop import main
from evolution.validation.report import (
    PhaseResult,
    TaskResult,
    ValidationReport,
    WinLoss,
)


def _stub_report(tool: str, decision: str) -> ValidationReport:
    empty_phase = PhaseResult(
        pass_rate=0.0, n_passed=0, n_failed=0, n_abstained=0, tasks=[],
    )
    return ValidationReport(
        schema_version="1",
        tool=tool,
        task_suite_path="x",
        task_suite_sha256="abc",
        baseline=empty_phase,
        evolved=empty_phase,
        delta=WinLoss(n_wins=0, n_losses=0, n_ties=0, pass_rate_change=0.0),
        decision=decision,
        decision_reasons=[],
    )


@pytest.fixture
def cli_fixtures(tmp_path):
    hermes_repo = tmp_path / "hermes-agent"
    (hermes_repo / "tools").mkdir(parents=True)
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(json.dumps({
        "task_id": "t1", "user_message": "do", "expected_tools": ["patch"],
    }) + "\n")
    baseline = tmp_path / "baseline.py"
    baseline.write_text("# baseline\n")
    evolved = tmp_path / "evolved.py"
    evolved.write_text("# evolved\n")
    return {
        "hermes_repo": hermes_repo, "tasks_path": tasks_path,
        "baseline": baseline, "evolved": evolved,
    }


class TestCli:
    def test_exits_zero_on_pass(self, cli_fixtures, tmp_path):
        report = _stub_report("patch", "pass")
        with patch(
            "evolution.validation.closed_loop.ClosedLoopValidator"
        ) as mock_validator_cls, patch(
            "evolution.validation.closed_loop.HermesToolDescriptionInstaller"
        ), patch(
            "evolution.validation.closed_loop.HermesAgentRunner"
        ):
            mock_validator_cls.return_value.validate.return_value = report
            result = CliRunner().invoke(main, [
                "--tool", "patch",
                "--hermes-repo", str(cli_fixtures["hermes_repo"]),
                "--tasks", str(cli_fixtures["tasks_path"]),
                "--baseline", str(cli_fixtures["baseline"]),
                "--evolved", str(cli_fixtures["evolved"]),
                "--output-dir", str(tmp_path / "out"),
            ])
        assert result.exit_code == 0, result.output

    def test_exits_one_on_regression(self, cli_fixtures, tmp_path):
        report = _stub_report("patch", "regression")
        with patch(
            "evolution.validation.closed_loop.ClosedLoopValidator"
        ) as mock_validator_cls, patch(
            "evolution.validation.closed_loop.HermesToolDescriptionInstaller"
        ), patch(
            "evolution.validation.closed_loop.HermesAgentRunner"
        ):
            mock_validator_cls.return_value.validate.return_value = report
            result = CliRunner().invoke(main, [
                "--tool", "patch",
                "--hermes-repo", str(cli_fixtures["hermes_repo"]),
                "--tasks", str(cli_fixtures["tasks_path"]),
                "--baseline", str(cli_fixtures["baseline"]),
                "--evolved", str(cli_fixtures["evolved"]),
                "--output-dir", str(tmp_path / "out"),
            ])
        assert result.exit_code == 1, result.output

    def test_writes_validation_report_json(self, cli_fixtures, tmp_path):
        report = _stub_report("patch", "pass")
        out_dir = tmp_path / "out"
        with patch(
            "evolution.validation.closed_loop.ClosedLoopValidator"
        ) as mock_validator_cls, patch(
            "evolution.validation.closed_loop.HermesToolDescriptionInstaller"
        ), patch(
            "evolution.validation.closed_loop.HermesAgentRunner"
        ):
            mock_validator_cls.return_value.validate.return_value = report
            CliRunner().invoke(main, [
                "--tool", "patch",
                "--hermes-repo", str(cli_fixtures["hermes_repo"]),
                "--tasks", str(cli_fixtures["tasks_path"]),
                "--baseline", str(cli_fixtures["baseline"]),
                "--evolved", str(cli_fixtures["evolved"]),
                "--output-dir", str(out_dir),
            ])
        report_path = out_dir / "validation_report.json"
        assert report_path.exists()
        loaded = json.loads(report_path.read_text())
        assert loaded["tool"] == "patch"
        assert loaded["decision"] == "pass"

    def test_validator_called_with_resolved_paths(self, cli_fixtures, tmp_path):
        """Click's path types resolve to absolute paths — make sure the
        validator sees the file paths the user passed, not whatever
        defaults Click might inject."""
        captured = {}
        report = _stub_report("patch", "pass")

        def _capture_inputs(inputs):
            captured["inputs"] = inputs
            return report

        mock_validator = MagicMock()
        mock_validator.validate.side_effect = _capture_inputs

        with patch(
            "evolution.validation.closed_loop.ClosedLoopValidator",
            return_value=mock_validator,
        ), patch(
            "evolution.validation.closed_loop.HermesToolDescriptionInstaller"
        ), patch(
            "evolution.validation.closed_loop.HermesAgentRunner"
        ):
            CliRunner().invoke(main, [
                "--tool", "patch",
                "--hermes-repo", str(cli_fixtures["hermes_repo"]),
                "--tasks", str(cli_fixtures["tasks_path"]),
                "--baseline", str(cli_fixtures["baseline"]),
                "--evolved", str(cli_fixtures["evolved"]),
                "--output-dir", str(tmp_path / "out"),
            ])

        inputs = captured["inputs"]
        assert inputs.tool_name == "patch"
        assert inputs.baseline_artifact == cli_fixtures["baseline"]
        assert inputs.evolved_artifact == cli_fixtures["evolved"]

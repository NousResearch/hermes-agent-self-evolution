"""--target {hermes,claude} wiring for evolve_prompt_section (dry-run; no LM/agent calls)."""
from click.testing import CliRunner

from evolution.prompts.evolve_prompt_section import main

_SUITE = (
    '{"task_id":"a","expected_action":"convention","required_cmd_substr":["bin/check"],'
    '"forbidden_cmd_substr":["pytest"],"user_message":"run tests","fixture_setup":{}}\n'
    '{"task_id":"b","expected_action":"convention","required_cmd_substr":["bin/run"],'
    '"forbidden_cmd_substr":["python app.py"],"user_message":"run app","fixture_setup":{}}\n'
)


def test_claude_target_dry_run_seeds_from_region(tmp_path):
    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "# Project\n<!-- evolve:CONV start -->\nseed conventions\n<!-- evolve:CONV end -->\n"
    )
    suite = tmp_path / "s.jsonl"
    suite.write_text(_SUITE)
    res = CliRunner().invoke(main, [
        "--target", "claude", "--section", "CONV", "--claude-md", str(claude_md),
        "--tasks", str(suite), "--dry-run", "--output-dir", str(tmp_path / "out"),
    ])
    assert res.exit_code == 0, res.output
    assert "Target: claude" in res.output


def test_claude_target_requires_claude_md(tmp_path):
    suite = tmp_path / "s.jsonl"
    suite.write_text(_SUITE)
    res = CliRunner().invoke(main, [
        "--target", "claude", "--section", "CONV",
        "--tasks", str(suite), "--dry-run", "--output-dir", str(tmp_path / "out"),
    ])
    assert res.exit_code != 0  # missing --claude-md


def test_hermes_target_still_requires_hermes_repo(tmp_path):
    suite = tmp_path / "s.jsonl"
    suite.write_text(_SUITE)
    res = CliRunner().invoke(main, [
        "--target", "hermes", "--section", "X",
        "--tasks", str(suite), "--dry-run", "--output-dir", str(tmp_path / "out"),
    ])
    assert res.exit_code != 0  # missing --hermes-repo

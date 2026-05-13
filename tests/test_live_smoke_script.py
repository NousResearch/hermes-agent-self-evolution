"""Tests for the optional live smoke runner without calling providers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_live_skill_evolution_smoke.py"


def load_smoke_module():
    spec = importlib.util.spec_from_file_location("run_live_skill_evolution_smoke", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_live_smoke_print_command_uses_default_fixture_and_dataset(capsys):
    smoke = load_smoke_module()

    exit_code = smoke.main(["--print-command-only"])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "evolution.skills.evolve_skill" in captured
    assert "--eval-source golden" in captured
    assert "examples/golden-datasets/demo-skill" in captured
    assert "examples/hermes-agent-fixture" in captured
    assert "--iterations 1" in captured


def test_provider_env_check_reports_likely_missing_openai_key(monkeypatch):
    smoke = load_smoke_module()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    missing = smoke.missing_provider_env(["openai/gpt-4.1", "openai/gpt-4.1-mini"])

    assert missing == ["OPENAI_API_KEY"]


def test_build_command_includes_configurable_output_dir(tmp_path):
    smoke = load_smoke_module()
    args = smoke.parse_args([
        "--print-command-only",
        "--output-dir",
        str(tmp_path / "artifacts"),
    ])

    command = smoke.build_command(args)

    assert "--output-dir" in command
    assert str(tmp_path / "artifacts") in command


def test_validate_inputs_finds_skill_outside_testing_category(tmp_path):
    smoke = load_smoke_module()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    hermes_repo = tmp_path / "hermes-agent"
    skill_dir = hermes_repo / "skills" / "research" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo\n---\n\n# Demo\n"
    )
    args = smoke.parse_args([
        "--print-command-only",
        "--dataset-path",
        str(dataset_path),
        "--hermes-repo",
        str(hermes_repo),
    ])

    smoke.validate_inputs(args)


def test_printed_command_is_shell_quoted_for_paths_with_spaces(tmp_path, capsys):
    smoke = load_smoke_module()
    dataset_path = tmp_path / "dataset with spaces"
    dataset_path.mkdir()
    hermes_repo = tmp_path / "hermes repo"
    skill_dir = hermes_repo / "skills" / "testing" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo\n---\n\n# Demo\n"
    )

    exit_code = smoke.main([
        "--print-command-only",
        "--dataset-path",
        str(dataset_path),
        "--hermes-repo",
        str(hermes_repo),
    ])

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "'" in captured
    assert "dataset with spaces" in captured

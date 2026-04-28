"""Tests for trace import and failure-to-dataset CLI commands."""

import json

from click.testing import CliRunner

from evolution.cli import main
from evolution.db.store import EvolutionStore


def _setup_trace_target(tmp_path):
    runner = CliRunner()
    root = tmp_path / ".evolution"
    repo_path = tmp_path / "hermes-agent"
    skill_dir = repo_path / "skills" / "testing" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: test-skill\ndescription: Test\n---\n\n# Test\n")
    assert runner.invoke(main, ["--root", str(root), "init"]).exit_code == 0
    assert runner.invoke(main, ["--root", str(root), "repo", "add", "hermes-agent", "--path", str(repo_path)]).exit_code == 0
    assert runner.invoke(main, ["--root", str(root), "targets", "scan", "--repo", "hermes-agent"]).exit_code == 0
    return runner, root


def _write_trace_file(path):
    rows = [
        {
            "task_input": "Review PR auth",
            "observed_output": "Looks fine",
            "expected_behavior": "Flag missing auth check",
            "status": "failure",
            "failure_reason": "missed auth bug",
        },
        {
            "task_input": "Review PR tests",
            "observed_output": "No tests needed",
            "expected_behavior": "Require regression tests",
            "status": "failure",
            "failure_reason": "ignored tests",
        },
        {
            "task_input": "Review PR docs",
            "observed_output": "Ship it",
            "expected_behavior": "Request exact docs path",
            "status": "failure",
            "failure_reason": "missed evidence path",
        },
        {
            "task_input": "Review PR style",
            "observed_output": "Good",
            "expected_behavior": "No issue",
            "status": "success",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_cli_traces_import_list_and_dataset_build(tmp_path):
    runner, root = _setup_trace_target(tmp_path)
    trace_file = tmp_path / "traces.jsonl"
    _write_trace_file(trace_file)

    import_result = runner.invoke(
        main,
        [
            "--root", str(root),
            "traces", "import",
            "--target", "skill:test-skill",
            "--source", "hermes-session",
            "--path", str(trace_file),
        ],
    )

    assert import_result.exit_code == 0, import_result.output
    assert "Imported 4 traces for skill:test-skill" in import_result.output

    list_result = runner.invoke(main, ["--root", str(root), "traces", "list", "--target", "skill:test-skill", "--status", "failure"])
    assert list_result.exit_code == 0, list_result.output
    assert list_result.output.count("trace_") == 3
    assert "missed auth bug" in list_result.output

    dataset_result = runner.invoke(
        main,
        [
            "--root", str(root),
            "traces", "dataset",
            "--target", "skill:test-skill",
            "--version", "trace-v1",
        ],
    )
    assert dataset_result.exit_code == 0, dataset_result.output
    assert "Built trace dataset dataset_" in dataset_result.output
    assert "3 examples" in dataset_result.output
    dataset_id = next(part for part in dataset_result.output.split() if part.startswith("dataset_"))

    store = EvolutionStore(root / "evolution.db")
    dataset = store.get_dataset(dataset_id)
    examples = store.list_eval_examples(dataset_id)
    assert dataset["source"] == "traces"
    assert dataset["version"] == "trace-v1"
    assert dataset["split_spec_json"] == {"train": 1, "val": 1, "holdout": 1}
    assert len(examples) == 3
    assert {example["split"] for example in examples} == {"train", "val", "holdout"}
    assert all(example["metadata_json"]["trace_id"].startswith("trace_") for example in examples)


def test_cli_traces_import_reports_invalid_json_without_traceback(tmp_path):
    runner, root = _setup_trace_target(tmp_path)
    trace_file = tmp_path / "invalid.jsonl"
    trace_file.write_text("{not-json}\n")

    result = runner.invoke(
        main,
        [
            "--root", str(root),
            "traces", "import",
            "--target", "skill:test-skill",
            "--source", "hermes-session",
            "--path", str(trace_file),
        ],
    )

    assert result.exit_code != 0
    assert "Invalid JSON on line 1" in result.output
    assert "Traceback" not in result.output
    store = EvolutionStore(root / "evolution.db")
    target = store.get_target_by_name("skill", "test-skill")
    assert store.list_attempt_traces(target_id=target["id"]) == []


def test_cli_traces_import_blocks_secret_values_without_persisting(tmp_path):
    runner, root = _setup_trace_target(tmp_path)
    trace_file = tmp_path / "traces.jsonl"
    trace_file.write_text(
        json.dumps(
            {
                "task_input": "safe task",
                "observed_output": "OPENAI_API_KEY=sk-" + "a" * 30,
                "expected_behavior": "redact secret",
                "status": "failure",
            }
        )
        + "\n"
    )

    result = runner.invoke(
        main,
        [
            "--root", str(root),
            "traces", "import",
            "--target", "skill:test-skill",
            "--source", "hermes-session",
            "--path", str(trace_file),
        ],
    )

    assert result.exit_code != 0
    assert "secret scan failed" in result.output
    assert "sk-" not in result.output
    store = EvolutionStore(root / "evolution.db")
    target = store.get_target_by_name("skill", "test-skill")
    assert store.list_attempt_traces(target_id=target["id"]) == []

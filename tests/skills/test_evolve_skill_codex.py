"""Tests for the codex-batched evolution CLI entrypoint."""

import json

from click.testing import CliRunner

from evolution.skills.evolve_skill_codex import main


class TestCodexEntrypoint:
    def test_cli_rejects_live_generation_without_allow_flag(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, [
            "--skill", "obsidian",
            "--eval-source", "synthetic",
            "--iterations", "1",
            "--hermes-repo", str(tmp_path),
        ])

        assert result.exit_code != 0
        assert "not implemented" in result.output.lower()

    def test_cli_requires_dataset_path_for_cached_mode(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, [
            "--skill", "obsidian",
            "--eval-source", "cached",
            "--iterations", "1",
            "--hermes-repo", str(tmp_path),
        ])

        assert result.exit_code != 0
        assert "dataset-path" in result.output

    def test_cli_dry_run_succeeds_with_cached_dataset_path(self, tmp_path):
        runner = CliRunner()
        dataset_dir = tmp_path / "datasets" / "skills" / "obsidian"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "train.jsonl").write_text('{"task_input":"a","expected_behavior":"b"}\n')
        (dataset_dir / "val.jsonl").write_text('{"task_input":"a","expected_behavior":"b"}\n')
        (dataset_dir / "holdout.jsonl").write_text('{"task_input":"a","expected_behavior":"b"}\n')

        result = runner.invoke(main, [
            "--skill", "obsidian",
            "--eval-source", "cached",
            "--dataset-path", str(dataset_dir),
            "--iterations", "1",
            "--dry-run",
            "--hermes-repo", str(tmp_path),
        ])

        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_cli_non_dry_run_writes_artifacts_with_mocked_dependencies(self, monkeypatch, tmp_path):
        runner = CliRunner()
        dataset_dir = tmp_path / "datasets" / "skills" / "obsidian"
        dataset_dir.mkdir(parents=True)
        sample = '{"task_input":"a","expected_behavior":"b"}\n'
        (dataset_dir / "train.jsonl").write_text(sample)
        (dataset_dir / "val.jsonl").write_text(sample)
        (dataset_dir / "holdout.jsonl").write_text(sample)

        skills_dir = tmp_path / "skills" / "note-taking" / "obsidian"
        skills_dir.mkdir(parents=True)
        skill_path = skills_dir / "SKILL.md"
        skill_path.write_text(
            "---\nname: obsidian\ndescription: test skill\n---\n\nOriginal body\n"
        )

        monkeypatch.setattr(
            "evolution.skills.evolve_skill_codex.CodexRunner.run_json_task",
            lambda self, prompt, timeout_seconds: (
                {"candidate_skill_markdown": "---\nname: obsidian\ndescription: test skill\n---\n\nImproved body\n"}
                if "mutation" in prompt.lower()
                else {
                    "baseline_score": 0.4,
                    "candidate_score": 0.6,
                    "improvement": 0.2,
                    "per_example": [{"task_input": "a", "baseline_score": 0.4, "candidate_score": 0.6, "reason": "clearer"}],
                    "recommendation": {
                        "winner": "candidate",
                        "reason": "candidate is clearer",
                        "confidence": 0.81,
                    },
                }
            ),
        )

        result = runner.invoke(main, [
            "--skill", "obsidian",
            "--eval-source", "cached",
            "--dataset-path", str(dataset_dir),
            "--iterations", "1",
            "--hermes-repo", str(tmp_path),
        ])

        assert result.exit_code == 0, result.output
        assert "Output saved to" in result.output

        output_root = tmp_path / "output" / "obsidian"
        runs = list(output_root.iterdir())
        assert len(runs) == 1
        run_dir = runs[0]

        assert (run_dir / "baseline_skill.md").exists()
        assert (run_dir / "evolved_skill.md").exists()
        assert (run_dir / "metrics.json").exists()

        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["baseline_score"] == 0.4
        assert metrics["candidate_score"] == 0.6
        assert metrics["improvement"] == 0.2
        assert metrics["per_example"][0]["reason"] == "clearer"
        assert metrics["recommendation"]["winner"] == "candidate"
        assert metrics["recommendation"]["reason"] == "candidate is clearer"
        assert metrics["recommendation"]["confidence"] == 0.81
        assert metrics["prompt_version"] == "v1"
        assert metrics["budget"]["max_codex_calls"] == 3
        assert metrics["budget"]["phase_timeout_seconds"] == 180

    def test_cli_rejects_iterations_above_one_until_loop_exists(self, tmp_path):
        runner = CliRunner()
        dataset_dir = tmp_path / "datasets" / "skills" / "obsidian"
        dataset_dir.mkdir(parents=True)
        sample = '{"task_input":"a","expected_behavior":"b"}\n'
        (dataset_dir / "train.jsonl").write_text(sample)
        (dataset_dir / "val.jsonl").write_text(sample)
        (dataset_dir / "holdout.jsonl").write_text(sample)

        result = runner.invoke(main, [
            "--skill", "obsidian",
            "--eval-source", "cached",
            "--dataset-path", str(dataset_dir),
            "--iterations", "2",
            "--dry-run",
            "--hermes-repo", str(tmp_path),
        ])

        assert result.exit_code != 0
        assert "iterations=1" in result.output.lower()

    def test_cli_reports_missing_skill_cleanly(self, tmp_path):
        runner = CliRunner()
        dataset_dir = tmp_path / "datasets" / "skills" / "obsidian"
        dataset_dir.mkdir(parents=True)
        sample = '{"task_input":"a","expected_behavior":"b"}\n'
        (dataset_dir / "train.jsonl").write_text(sample)
        (dataset_dir / "val.jsonl").write_text(sample)
        (dataset_dir / "holdout.jsonl").write_text(sample)

        result = runner.invoke(main, [
            "--skill", "obsidian",
            "--eval-source", "cached",
            "--dataset-path", str(dataset_dir),
            "--iterations", "1",
            "--hermes-repo", str(tmp_path),
        ])

        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_cli_reports_codex_timeout_cleanly(self, monkeypatch, tmp_path):
        runner = CliRunner()
        dataset_dir = tmp_path / "datasets" / "skills" / "obsidian"
        dataset_dir.mkdir(parents=True)
        sample = '{"task_input":"a","expected_behavior":"b"}\n'
        (dataset_dir / "train.jsonl").write_text(sample)
        (dataset_dir / "val.jsonl").write_text(sample)
        (dataset_dir / "holdout.jsonl").write_text(sample)

        skills_dir = tmp_path / "skills" / "note-taking" / "obsidian"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("---\nname: obsidian\ndescription: test skill\n---\n\nOriginal body\n")

        def fake_run_json_task(self, prompt, timeout_seconds):
            raise RuntimeError("Codex timed out after 180s")

        monkeypatch.setattr(
            "evolution.skills.evolve_skill_codex.CodexRunner.run_json_task",
            fake_run_json_task,
        )

        result = runner.invoke(main, [
            "--skill", "obsidian",
            "--eval-source", "cached",
            "--dataset-path", str(dataset_dir),
            "--iterations", "1",
            "--hermes-repo", str(tmp_path),
        ])

        assert result.exit_code != 0
        assert "Codex timed out" in result.output

    def test_cli_reports_constraint_failure_cleanly(self, monkeypatch, tmp_path):
        runner = CliRunner()
        dataset_dir = tmp_path / "datasets" / "skills" / "obsidian"
        dataset_dir.mkdir(parents=True)
        sample = '{"task_input":"a","expected_behavior":"b"}\n'
        (dataset_dir / "train.jsonl").write_text(sample)
        (dataset_dir / "val.jsonl").write_text(sample)
        (dataset_dir / "holdout.jsonl").write_text(sample)

        skills_dir = tmp_path / "skills" / "note-taking" / "obsidian"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("---\nname: obsidian\ndescription: test skill\n---\n\nOriginal body\n")

        monkeypatch.setattr(
            "evolution.skills.evolve_skill_codex.CodexRunner.run_json_task",
            lambda self, prompt, timeout_seconds: {"candidate_skill_markdown": "broken"},
        )

        result = runner.invoke(main, [
            "--skill", "obsidian",
            "--eval-source", "cached",
            "--dataset-path", str(dataset_dir),
            "--iterations", "1",
            "--hermes-repo", str(tmp_path),
        ])

        assert result.exit_code != 0
        assert "failed constraints" in result.output.lower()

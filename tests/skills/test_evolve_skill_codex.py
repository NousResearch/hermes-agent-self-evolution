"""Tests for the codex-batched evolution CLI entrypoint."""

import json
import os
import subprocess
from pathlib import Path
import sys

from click.testing import CliRunner

from evolution.skills.evolve_skill_codex import main


class TestCodexEntrypoint:
    def test_codex_entrypoint_imports_without_dspy_in_base_python(self, tmp_path):
        base_python = Path(sys.base_prefix) / "bin" / "python3"
        if not base_python.exists():
            raise AssertionError(f"Base interpreter not found: {base_python}")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())

        result = subprocess.run(
            [
                str(base_python),
                "-c",
                "import evolution.skills.evolve_skill_codex as m; print(m.__name__)",
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=Path.cwd(),
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "evolution.skills.evolve_skill_codex"

    def test_cli_rejects_unsupported_eval_source_at_parse_time(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, [
            "--skill", "obsidian",
            "--eval-source", "synthetic",
            "--iterations", "1",
            "--hermes-repo", str(tmp_path),
        ])

        assert result.exit_code != 0
        assert "invalid value" in result.output.lower()
        assert "cached" in result.output.lower()

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
        assert metrics["budget"]["max_examples"] == 8
        assert "budget_strict" not in metrics["budget"]
        assert "max_candidates_per_iteration" not in metrics["budget"]

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

    def test_cli_rejects_candidate_when_evaluation_prefers_baseline(self, monkeypatch, tmp_path):
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
            lambda self, prompt, timeout_seconds: (
                {"candidate_skill_markdown": "---\nname: obsidian\ndescription: test skill\n---\n\nImproved body\n"}
                if "mutation" in prompt.lower()
                else {
                    "baseline_score": 0.7,
                    "candidate_score": 0.6,
                    "improvement": -0.1,
                    "per_example": [{"task_input": "a", "baseline_score": 0.7, "candidate_score": 0.6, "reason": "baseline clearer"}],
                    "recommendation": {
                        "winner": "baseline",
                        "reason": "candidate regressed",
                        "confidence": 0.82,
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

        assert result.exit_code != 0
        assert "rejected" in result.output.lower()
        output_root = tmp_path / "output" / "obsidian"
        assert not output_root.exists()

    def test_cli_limits_holdout_examples_to_configured_max_examples(self, monkeypatch, tmp_path):
        runner = CliRunner()
        dataset_dir = tmp_path / "datasets" / "skills" / "obsidian"
        dataset_dir.mkdir(parents=True)
        sample = '{"task_input":"a","expected_behavior":"b"}\n'
        (dataset_dir / "train.jsonl").write_text(sample)
        (dataset_dir / "val.jsonl").write_text(sample)
        (dataset_dir / "holdout.jsonl").write_text(
            "".join(
                json.dumps({"task_input": f"task-{idx}", "expected_behavior": "b"}) + "\n"
                for idx in range(3)
            )
        )

        skills_dir = tmp_path / "skills" / "note-taking" / "obsidian"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("---\nname: obsidian\ndescription: test skill\n---\n\nOriginal body\n")

        captured = {}

        def fake_run_json_task(self, prompt, timeout_seconds):
            if "mutation" in prompt.lower():
                return {"candidate_skill_markdown": "---\nname: obsidian\ndescription: test skill\n---\n\nImproved body\n"}
            holdout_json = prompt.split("Holdout: ", 1)[1]
            captured["holdout"] = json.loads(holdout_json)
            return {
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

        monkeypatch.setenv("HERMES_EVOLUTION_MAX_EXAMPLES", "1")
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

        assert result.exit_code == 0, result.output
        assert len(captured["holdout"]) == 1

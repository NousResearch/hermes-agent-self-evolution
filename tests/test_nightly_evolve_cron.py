"""Tests for the nightly self-evolution cron controller."""

from pathlib import Path

from scripts import nightly_evolve_cron as cron
from scripts.nightly_evolve_cron import Target, resolve_eval_dataset_args, summarize


def _target(tmp_path: Path) -> Target:
    return Target(
        profile="babbage",
        profile_root=tmp_path / "hermes-home",
        skill="kanban-orchestrator",
        domain="coordination",
        title="Chief of Staff",
    )


def test_auto_eval_prefers_golden_dataset_when_present(tmp_path):
    golden_dir = tmp_path / "datasets" / "golden" / "skills" / "kanban-orchestrator"
    golden_dir.mkdir(parents=True)
    (golden_dir / "train.jsonl").write_text("{}\n")

    eval_source, dataset_path = resolve_eval_dataset_args(
        tmp_path,
        _target(tmp_path),
        {"source": "auto"},
    )

    assert eval_source == "golden"
    assert dataset_path == str(golden_dir)


def test_auto_eval_falls_back_to_synthetic_without_golden_dataset(tmp_path):
    eval_source, dataset_path = resolve_eval_dataset_args(
        tmp_path,
        _target(tmp_path),
        {"source": "auto"},
    )

    assert eval_source == "synthetic"
    assert dataset_path is None


def test_explicit_synthetic_does_not_use_golden_dataset(tmp_path):
    golden_dir = tmp_path / "datasets" / "golden" / "skills" / "kanban-orchestrator"
    golden_dir.mkdir(parents=True)
    (golden_dir / "golden.jsonl").write_text("{}\n")

    eval_source, dataset_path = resolve_eval_dataset_args(
        tmp_path,
        _target(tmp_path),
        {"source": "synthetic"},
    )

    assert eval_source == "synthetic"
    assert dataset_path is None


def test_summarize_reports_gate_status_and_application_state(tmp_path):
    artifact = tmp_path / "artifact"
    log = tmp_path / "run.log"
    summary = summarize(
        _target(tmp_path),
        "candidate",
        artifact,
        log,
        {
            "baseline_score": 0.5,
            "evolved_score": 0.7,
            "improvement": 0.2,
            "constraints_passed": True,
        },
    )

    assert "gate: constraints-passed" in summary
    assert "applied: no" in summary
    assert "review: candidate-review-needed" in summary


def test_summarize_reports_constraint_rejection(tmp_path):
    summary = summarize(
        _target(tmp_path),
        "constraints-failed",
        tmp_path / "artifact",
        tmp_path / "run.log",
        None,
    )

    assert "gate: constraints-failed" in summary
    assert "review: rejected" in summary
    assert "applied: no" in summary


def test_codex_models_validate_against_hermes_oauth_not_openai_key(monkeypatch):
    monkeypatch.setattr(cron, "codex_auth_available", lambda *, refresh_if_expiring=False: True)

    policy = {
        "models": {
            "eval": "openai-codex/gpt-5.4-mini",
            "optimizer": "openai-codex/gpt-5.4-mini",
            "fallback_eval": "openai-codex/gpt-5.4-mini",
            "fallback_optimizer": "openai-codex/gpt-5.4-mini",
        }
    }

    assert cron.required_key_for_model("openai-codex/gpt-5.4-mini") is None
    assert cron.validate_model_env(policy, env={}) == []


def test_codex_models_report_missing_hermes_oauth_once(monkeypatch):
    monkeypatch.setattr(cron, "codex_auth_available", lambda *, refresh_if_expiring=False: False)

    missing = cron.validate_model_env(
        {
            "models": {
                "eval": "openai-codex/gpt-5.4-mini",
                "optimizer": "openai-codex/gpt-5.4-mini",
            }
        },
        env={},
    )

    assert missing == ["OpenAI-Codex OAuth credentials (run `hermes model`)"]

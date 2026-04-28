"""Tests for executable content package validation gates."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from evolution.cli import main
from evolution.content.validator import validate_content_package


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _base_manifest(root: Path, *, scope: list[str] | None = None, proof_artifacts: dict | None = None) -> dict:
    (root / "presentation.md").write_text("# Capability presentation\n")
    (root / "index.html").write_text("<html><body>Capability presentation</body></html>\n")
    manifest = {
        "run_id": "test_run_001",
        "stage": "capability_presentation",
        "generated_at": "2026-04-28T00:00:00Z",
        "owner_profile": "video-production",
        "artifacts": {
            "markdown": str(root / "presentation.md"),
            "html": str(root / "index.html"),
        },
        "capability_scope": scope or ["Remotion animation"],
    }
    if proof_artifacts is not None:
        manifest["proof_artifacts"] = proof_artifacts
    _write_json(root / "manifest.json", manifest)
    return manifest


def test_validator_emits_runtime_and_qa_verdict_for_passing_proof_package(tmp_path):
    project = tmp_path / "strong-package"
    project.mkdir()
    proof_dir = project / "proof-assets"
    proof_dir.mkdir()
    remotion_proof = proof_dir / "remotion-frame-grab.png"
    remotion_proof.write_bytes(b"not really a png, but inspectable bytes")
    _base_manifest(project, proof_artifacts={"remotion": str(remotion_proof)})

    result = validate_content_package(project, write=True, now="2026-04-28T00:00:00Z")

    assert result["verdict"] == "pass"
    assert (project / "runtime_check.json").exists()
    assert (project / "qa_verdict.json").exists()
    runtime = json.loads((project / "runtime_check.json").read_text())
    qa = json.loads((project / "qa_verdict.json").read_text())
    assert runtime["stage"] == "runtime_preflight"
    assert qa["verdict"] == "pass"
    assert qa["generated_by"] == "hermes-evolve content validate"


def test_validator_holds_feature_card_deck_without_proof_assets(tmp_path):
    project = tmp_path / "weak-deck"
    project.mkdir()
    _base_manifest(project, scope=["Remotion animation", "GSAP", "Three.js", "AntV charts"])

    result = validate_content_package(project, write=True, now="2026-04-28T00:00:00Z")

    assert result["verdict"] == "hold"
    assert (project / "runtime_check.json").exists()
    qa_path = project / "qa_verdict.json"
    assert qa_path.exists()
    qa = json.loads(qa_path.read_text())
    failed_ids = {check["id"] for check in qa["checks"] if check["status"] == "hold"}
    assert "proof_assets_dir" in failed_ids
    assert "proof_assets_non_empty" in failed_ids
    assert "claimed_visual_stack_proof" in failed_ids
    assert "human_quality_gate" in failed_ids


def test_cli_content_validate_writes_reports_and_exits_nonzero_on_hold(tmp_path):
    project = tmp_path / "weak-cli-deck"
    project.mkdir()
    _base_manifest(project, scope=["Remotion animation", "Lottie/dotLottie"])

    result = CliRunner().invoke(main, ["content", "validate", str(project)])

    assert result.exit_code != 0
    assert "verdict=hold" in result.output
    assert "qa_verdict=" in result.output
    assert (project / "runtime_check.json").exists()
    assert (project / "qa_verdict.json").exists()


def test_cli_content_validate_allow_hold_returns_zero_for_audit_mode(tmp_path):
    project = tmp_path / "weak-audit-deck"
    project.mkdir()
    _base_manifest(project, scope=["Remotion animation"])

    result = CliRunner().invoke(main, ["content", "validate", str(project), "--allow-hold"])

    assert result.exit_code == 0, result.output
    assert "verdict=hold" in result.output

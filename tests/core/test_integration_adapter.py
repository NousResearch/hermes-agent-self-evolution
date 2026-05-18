"""Tests for Hermes integration adapter compatibility checks."""

from pathlib import Path

import pytest

from evolution.core.integration_adapter import HermesIntegrationAdapter


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# test\n")


def test_detect_version_from_pyproject(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nversion = \"1.2.3\"\n")
    adapter = HermesIntegrationAdapter(tmp_path)
    assert adapter.detect_version() == "1.2.3"


def test_compatibility_matrix_reports_missing_components(tmp_path: Path):
    _touch(tmp_path / "batch_runner.py")
    _touch(tmp_path / "agent" / "trajectory.py")
    adapter = HermesIntegrationAdapter(tmp_path)

    matrix = adapter.compatibility_matrix()
    missing = [row for row in matrix if row.required and not row.exists]
    assert missing
    assert any(row.component == "prompt_builder" for row in missing)


def test_assert_compatible_raises_with_missing_required(tmp_path: Path):
    adapter = HermesIntegrationAdapter(tmp_path)
    with pytest.raises(RuntimeError, match="Missing required components"):
        adapter.assert_compatible()


def test_assert_compatible_passes_when_all_components_exist(tmp_path: Path):
    for _, (rel_path, _) in HermesIntegrationAdapter.REQUIRED_COMPONENTS.items():
        _touch(tmp_path / rel_path)

    adapter = HermesIntegrationAdapter(tmp_path)
    adapter.assert_compatible()

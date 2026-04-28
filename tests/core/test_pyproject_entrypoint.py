"""Tests for package entrypoint registration."""

from pathlib import Path


def test_pyproject_declares_hermes_evolve_script():
    pyproject = Path("pyproject.toml").read_text()

    assert "[project.scripts]" in pyproject
    assert 'hermes-evolve = "evolution.cli:main"' in pyproject

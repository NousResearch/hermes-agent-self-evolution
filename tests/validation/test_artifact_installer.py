"""Tests for HermesToolDescriptionInstaller._extract_description.

Verifies the installer accepts both input shapes the harness sees in
practice:
  - Hermes tool-module .py file (a hand-edited baseline)
  - MCP-shape manifest .json (evolve_tool's output, plumbed through
    --benchmark-cmd as EVOLVED_PATH / BASELINE_PATH)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolution.validation.artifact_installer import HermesToolDescriptionInstaller


# ---- Hermes .py fixture ----

_PY_FIXTURE = '''\
"""Stub tool module."""

from tools.registry import registry

PATCH_SCHEMA = {
    "name": "patch",
    "description": "Apply targeted edits.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

registry.register(PATCH_SCHEMA, lambda **kw: None)
'''


# ---- MCP JSON fixture ----

_JSON_FIXTURE = {
    "tools": [
        {
            "name": "patch",
            "description": "Apply targeted edits from JSON.",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        }
    ]
}


@pytest.fixture
def fake_hermes_repo(tmp_path):
    """A minimal hermes-agent-shaped repo with a registry stub + the patch
    tool. Just enough that HermesToolSource discovers it."""
    repo = tmp_path / "hermes-agent"
    tools = repo / "tools"
    tools.mkdir(parents=True)
    # Minimal registry stub so the AST walker doesn't choke on imports.
    (tools / "__init__.py").write_text("")
    (tools / "registry.py").write_text(
        "class _Registry:\n"
        "    def register(self, schema, handler):\n"
        "        pass\n"
        "registry = _Registry()\n"
    )
    (tools / "file_tools.py").write_text(_PY_FIXTURE)
    return repo


class TestExtractDescription:
    def test_extracts_from_hermes_py_module(self, fake_hermes_repo, tmp_path):
        evolved_py = tmp_path / "evolved_file_tools.py"
        evolved_py.write_text(_PY_FIXTURE.replace(
            '"description": "Apply targeted edits."',
            '"description": "Evolved patch description."',
        ))
        installer = HermesToolDescriptionInstaller(
            hermes_repo=fake_hermes_repo,
            tool_name="patch",
        )
        desc = installer._extract_description(evolved_py)
        assert desc == "Evolved patch description."

    def test_extracts_from_mcp_manifest_json(self, fake_hermes_repo, tmp_path):
        evolved_json = tmp_path / "evolved_manifest.json"
        evolved_json.write_text(json.dumps(_JSON_FIXTURE))
        installer = HermesToolDescriptionInstaller(
            hermes_repo=fake_hermes_repo,
            tool_name="patch",
        )
        desc = installer._extract_description(evolved_json)
        assert desc == "Apply targeted edits from JSON."

    def test_extracts_from_mcp_manifest_json_relative_path(
        self, fake_hermes_repo, tmp_path, monkeypatch
    ):
        # Reproduces the --benchmark-cmd flow: EVOLVED_PATH arrives as a
        # path relative to the orchestrator's CWD (e.g.
        # "output/tools/<ts>/evolved_manifest.json"). The prior impl piped
        # this through MCPManifestSource.find_manifest, which prepended the
        # input's own parent and doubled the path.
        out_dir = tmp_path / "output" / "tools" / "run1"
        out_dir.mkdir(parents=True)
        (out_dir / "evolved_manifest.json").write_text(json.dumps(_JSON_FIXTURE))
        monkeypatch.chdir(tmp_path)
        installer = HermesToolDescriptionInstaller(
            hermes_repo=fake_hermes_repo,
            tool_name="patch",
        )
        rel_path = Path("output/tools/run1/evolved_manifest.json")
        desc = installer._extract_description(rel_path)
        assert desc == "Apply targeted edits from JSON."

    def test_json_target_tool_missing_raises(self, fake_hermes_repo, tmp_path):
        evolved_json = tmp_path / "evolved_manifest.json"
        evolved_json.write_text(json.dumps({
            "tools": [
                {"name": "other_tool", "description": "x", "inputSchema": {}}
            ]
        }))
        installer = HermesToolDescriptionInstaller(
            hermes_repo=fake_hermes_repo,
            tool_name="patch",
        )
        with pytest.raises(KeyError):
            installer._extract_description(evolved_json)

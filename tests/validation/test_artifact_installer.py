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


# ---------------------------------------------------------------------------
# SkillFileInstaller
# ---------------------------------------------------------------------------


from evolution.validation.artifact_installer import SkillFileInstaller


@pytest.fixture
def baseline_skill(tmp_path):
    """A read-only source skill directory the installer must not mutate."""
    src_root = tmp_path / "source_skills" / "systematic_debugging"
    src_root.mkdir(parents=True)
    skill_md = src_root / "SKILL.md"
    skill_md.write_text(
        "---\nname: systematic_debugging\n---\n\n# Baseline body\n"
    )
    # Sibling file the skill might reference — must also be copied.
    (src_root / "examples.md").write_text("baseline examples")
    return skill_md


class TestSkillFileInstallerConstruction:
    def test_copies_skill_directory_into_workdir(self, baseline_skill, tmp_path):
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        installer = SkillFileInstaller(
            skill_source_path=baseline_skill,
            skill_name="systematic_debugging",
            workdir=workdir,
        )
        # The target_path lives under the workdir.
        assert installer.target_path == (
            workdir / "skills" / "systematic_debugging" / "SKILL.md"
        )
        assert installer.target_path.is_file()
        # Sibling files are copied too — agents may reference them.
        sibling = workdir / "skills" / "systematic_debugging" / "examples.md"
        assert sibling.is_file()
        assert sibling.read_text() == "baseline examples"

    def test_skills_src_attribute_points_at_skills_root(self, baseline_skill, tmp_path):
        # The validator threads this into TaskRunContext so the runner
        # can stage it into its per-task sandbox.
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        installer = SkillFileInstaller(
            skill_source_path=baseline_skill,
            skill_name="systematic_debugging",
            workdir=workdir,
        )
        assert installer.skills_src == workdir / "skills"

    def test_missing_source_raises(self, tmp_path):
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        with pytest.raises(FileNotFoundError):
            SkillFileInstaller(
                skill_source_path=tmp_path / "does_not_exist.md",
                skill_name="x",
                workdir=workdir,
            )

    def test_missing_workdir_raises(self, baseline_skill, tmp_path):
        with pytest.raises(NotADirectoryError):
            SkillFileInstaller(
                skill_source_path=baseline_skill,
                skill_name="x",
                workdir=tmp_path / "does_not_exist",
            )


class TestSkillFileInstallerInstall:
    def test_install_overwrites_target_with_candidate_text(self, baseline_skill, tmp_path):
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        installer = SkillFileInstaller(
            skill_source_path=baseline_skill,
            skill_name="systematic_debugging",
            workdir=workdir,
        )
        candidate = tmp_path / "candidate.md"
        candidate.write_text("# Evolved body\n")
        sha = installer.install(candidate)
        assert installer.target_path.read_text() == "# Evolved body\n"
        # Returned sha matches the post-install bytes.
        import hashlib
        assert sha == hashlib.sha256(b"# Evolved body\n").hexdigest()

    def test_install_does_not_mutate_source(self, baseline_skill, tmp_path):
        """The whole point of the workdir copy: the user's actual skill
        on disk is never touched, even when we install a candidate."""
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        original = baseline_skill.read_text()
        installer = SkillFileInstaller(
            skill_source_path=baseline_skill,
            skill_name="systematic_debugging",
            workdir=workdir,
        )
        candidate = tmp_path / "candidate.md"
        candidate.write_text("# Different body\n")
        installer.install(candidate)
        assert baseline_skill.read_text() == original


class TestSkillFileInstallerVerifyBackup:
    def _installer(self, baseline_skill, tmp_path):
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        return SkillFileInstaller(
            skill_source_path=baseline_skill,
            skill_name="systematic_debugging",
            workdir=workdir,
        )

    def test_accepts_valid_utf8_backup(self, baseline_skill, tmp_path):
        installer = self._installer(baseline_skill, tmp_path)
        backup = tmp_path / "backup.md"
        backup.write_text("# Some skill text\n")
        # Does not raise.
        installer.verify_backup(backup)

    def test_rejects_empty_backup(self, baseline_skill, tmp_path):
        installer = self._installer(baseline_skill, tmp_path)
        backup = tmp_path / "backup.md"
        backup.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            installer.verify_backup(backup)

    def test_rejects_invalid_utf8(self, baseline_skill, tmp_path):
        installer = self._installer(baseline_skill, tmp_path)
        backup = tmp_path / "backup.md"
        backup.write_bytes(b"\xff\xfe\x00\x00invalid")
        with pytest.raises(ValueError, match="not valid UTF-8"):
            installer.verify_backup(backup)

"""build_backend — the single per-target construction site."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from evolution.prompts.backend import build_backend
from evolution.validation.claude_runner import (
    DEFAULT_CLAUDE_TIMEOUT_SECONDS,
    ClaudeCodeAgentRunner,
)
from evolution.validation.hermes_runner import DEFAULT_TASK_TIMEOUT_SECONDS, HermesAgentRunner


def _hermes_repo(tmp_path: Path) -> Path:
    (tmp_path / "agent").mkdir(parents=True, exist_ok=True)
    (tmp_path / "agent" / "prompt_builder.py").write_text(textwrap.dedent('''\
        """Stub."""
        MEMORY_GUIDANCE = "baseline text"
    '''))
    return tmp_path


def _claude_md(tmp_path: Path) -> Path:
    p = tmp_path / "CLAUDE.md"
    p.write_text("# top\n<!-- evolve:CONV start -->\nseed conventions\n<!-- evolve:CONV end -->\n")
    return p


def _kw(**over):
    base = dict(
        section_name="X", hermes_repo=None, claude_md=None, output_dir=Path("/tmp"),
        agent_model=None, task_timeout_seconds=None, baseline_override_file=None,
    )
    base.update(over)
    return base


def test_hermes_backend(tmp_path):
    repo = _hermes_repo(tmp_path)
    b = build_backend("hermes", **_kw(section_name="MEMORY_GUIDANCE", hermes_repo=repo,
                                      output_dir=tmp_path))
    assert isinstance(b.runner, HermesAgentRunner)
    assert b.runner.timeout_seconds == DEFAULT_TASK_TIMEOUT_SECONDS
    assert b.installer.target_path == repo / "agent" / "prompt_builder.py"
    assert b.deploy_target == repo / "agent" / "prompt_builder.py"
    assert b.baseline_text == "baseline text"


def test_claude_backend(tmp_path):
    md = _claude_md(tmp_path)
    b = build_backend("claude", **_kw(section_name="CONV", claude_md=md, output_dir=tmp_path))
    assert isinstance(b.runner, ClaudeCodeAgentRunner)
    assert b.runner.timeout_seconds == DEFAULT_CLAUDE_TIMEOUT_SECONDS  # per-target default
    # validation target is the throwaway workdir file, NOT the user's CLAUDE.md
    assert b.installer.target_path == tmp_path / "claude_workdir" / "append_system_prompt.txt"
    assert b.deploy_target == md
    assert b.baseline_text.strip() == "seed conventions"


def test_explicit_timeout_wins(tmp_path):
    b = build_backend("claude", **_kw(section_name="CONV", claude_md=_claude_md(tmp_path),
                                      output_dir=tmp_path, task_timeout_seconds=42))
    assert b.runner.timeout_seconds == 42


def test_missing_required_args_raise(tmp_path):
    with pytest.raises(ValueError, match="hermes-repo"):
        build_backend("hermes", **_kw(section_name="X", output_dir=tmp_path))
    with pytest.raises(ValueError, match="claude-md"):
        build_backend("claude", **_kw(section_name="X", output_dir=tmp_path))
    with pytest.raises(ValueError, match="unknown --target"):
        build_backend("codex", **_kw(output_dir=tmp_path))


def test_fail_fast_on_missing_section(tmp_path):
    with pytest.raises(KeyError):
        build_backend("hermes", **_kw(section_name="NOPE", hermes_repo=_hermes_repo(tmp_path),
                                      output_dir=tmp_path))


def test_claude_override_seeds_new_region_without_failfast(tmp_path):
    # region absent + override file -> allowed (creates region on --apply)
    md = tmp_path / "CLAUDE.md"
    md.write_text("# no region yet\n")
    ovr = tmp_path / "seed.txt"
    ovr.write_text("forceful conventions")
    b = build_backend("claude", **_kw(section_name="NEWCONV", claude_md=md, output_dir=tmp_path,
                                      baseline_override_file=ovr))
    assert b.baseline_text == "forceful conventions"


def test_empty_baseline_rejected(tmp_path):
    md = tmp_path / "CLAUDE.md"
    md.write_text("<!-- evolve:CONV start -->\n   \n<!-- evolve:CONV end -->\n")
    with pytest.raises(ValueError, match="empty"):
        build_backend("claude", **_kw(section_name="CONV", claude_md=md, output_dir=tmp_path))

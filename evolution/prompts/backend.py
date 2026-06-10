"""PromptBackend — the single per-target strategy for prompt-section evolution.

``build_backend`` is the ONE place that branches on ``--target``; it bundles the
source / installer / runner and the baseline so ``evolve_prompt_section`` carries no
``if target`` branches. Adding a third backend means a factory branch + adapter
classes, nothing in the driver.

Two operations are deliberately distinct and target DIFFERENT files for claude:
  - ``install_candidate`` (validation inner loop) → the installer's throwaway target
  - ``deploy`` (``--apply``) → the source's real artifact (prompt_builder.py / CLAUDE.md)
Never merge them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evolution.prompts.claude_prompt_source import ClaudeCodePromptSource
from evolution.prompts.hermes_prompt_source import HermesPromptSource
from evolution.prompts.prompt_source import PromptSource
from evolution.validation.agent_runner import AgentRunner
from evolution.validation.artifact_installer import (
    ClaudeAppendPromptInstaller,
    HermesPromptSectionInstaller,
    SupportsInlineInstall,
)
from evolution.validation.claude_runner import (
    DEFAULT_CLAUDE_TIMEOUT_SECONDS,
    ClaudeCodeAgentRunner,
)
from evolution.validation.hermes_runner import (
    DEFAULT_TASK_TIMEOUT_SECONDS,
    HermesAgentRunner,
)


@dataclass(frozen=True)
class PromptBackend:
    """Everything the driver needs for one target, selected once."""

    source: PromptSource              # read baseline; deploy on --apply
    installer: SupportsInlineInstall  # validation splice target (guard + ClosedLoopValidator)
    runner: AgentRunner
    baseline_text: str
    deploy_target: Path               # what --apply actually writes (for the log line)

    def install_candidate(self, text: str) -> None:
        """Install a candidate for validation (inner loop) — via the installer."""
        self.installer.install_text(text)

    def deploy(self, section_name: str, text: str) -> None:
        """Write the evolved section to the real artifact (--apply) — via the source."""
        self.source.write(section_name, text)


def build_backend(
    target: str,
    *,
    section_name: str,
    hermes_repo: Optional[Path],
    claude_md: Optional[Path],
    output_dir: Path,
    agent_model: Optional[str],
    task_timeout_seconds: Optional[int],
    baseline_override_file: Optional[Path],
) -> PromptBackend:
    """Construct the backend for ``target``. The sole per-target branch site.

    Resolves the effective timeout once (explicit ``task_timeout_seconds`` wins,
    else the per-target default), validates required args + section existence, and
    computes the baseline (override file or the live section)."""
    if target == "hermes":
        if hermes_repo is None:
            raise ValueError("--hermes-repo is required for --target hermes")
        resolved_repo = Path(hermes_repo).resolve()
        source: PromptSource = HermesPromptSource(resolved_repo)
        timeout = task_timeout_seconds or DEFAULT_TASK_TIMEOUT_SECONDS
    elif target == "claude":
        if claude_md is None:
            raise ValueError("--claude-md is required for --target claude")
        source = ClaudeCodePromptSource(Path(claude_md))
        timeout = task_timeout_seconds or DEFAULT_CLAUDE_TIMEOUT_SECONDS
    else:
        raise ValueError(f"unknown --target {target!r} (expected 'hermes' or 'claude')")

    # Fail fast on a non-existent section BEFORE any LM spend — except the one
    # legitimate case of claude seeding a brand-new CLAUDE.md region from an override
    # (the region is created on --apply).
    if not (target == "claude" and baseline_override_file is not None):
        source.read(section_name)

    if baseline_override_file is not None:
        baseline_text = Path(baseline_override_file).read_text(encoding="utf-8")
    else:
        baseline_text = source.read(section_name)
    if not baseline_text.strip():
        raise ValueError(
            f"baseline for section {section_name!r} is empty — add seed text to the "
            f"section/region or pass a non-empty --baseline-override-file."
        )

    if target == "hermes":
        installer: SupportsInlineInstall = HermesPromptSectionInstaller(
            resolved_repo, section_name
        )
        runner: AgentRunner = HermesAgentRunner(timeout_seconds=timeout, model=agent_model)
        deploy_target = installer.target_path
    else:
        workdir = Path(output_dir) / "claude_workdir"
        workdir.mkdir(parents=True, exist_ok=True)
        installer = ClaudeAppendPromptInstaller(workdir=workdir, baseline_text=baseline_text)
        runner = ClaudeCodeAgentRunner(
            append_prompt_file=installer.target_path,
            model=agent_model or "sonnet",
            timeout_seconds=timeout,
        )
        deploy_target = Path(claude_md)  # type: ignore[arg-type]  # non-None for claude

    return PromptBackend(
        source=source,
        installer=installer,
        runner=runner,
        baseline_text=baseline_text,
        deploy_target=deploy_target,
    )

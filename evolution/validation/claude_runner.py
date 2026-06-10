"""ClaudeCodeAgentRunner — drive ``claude -p`` headlessly and map to AgentRunResult.

Implements the AgentRunner protocol so the agnostic scorer/validator consume it
unchanged. The candidate prompt-section under test is injected via
``--append-system-prompt-file`` (the installer writes the candidate to that file;
``claude`` reads it fresh each invocation), so the user's real CLAUDE.md is never
touched during validation.

Invocation recipe (verified against claude 2.1.169):
  - subscription auth via CLAUDE_CODE_OAUTH_TOKEN env (NOT --bare; bare ignores the token)
  - hermetic per-run: HOME=fresh tmp (no ambient ~/.claude CLAUDE.md / plugins / memory),
    --strict-mcp-config (no MCP servers), --no-session-persistence
  - cwd + --add-dir = fixture_dir
  - --output-format stream-json --verbose streams assistant tool_use blocks + a final
    result event carrying total_cost_usd + token usage

CONTAINMENT: the agent runs with ``--permission-mode bypassPermissions`` (no
interactive prompts in headless use), so filesystem confinement is enforced at the OS
level by wrapping the subprocess in macOS ``sandbox-exec`` with a profile that denies
all writes except under the fixture dir, the run's HOME, and temp dirs. The Claude Code
``sandbox`` *setting* is NOT used — it only confines Bash, leaving the native Write/Edit
tools free to escape (empirically confirmed). When OS sandboxing is unavailable (non-
macOS, or sandbox-exec missing), the runner REFUSES to run rather than silently
executing an unconfined agent — set ``require_sandbox=False`` to override deliberately.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import litellm

from evolution.core.lm_timing_callback import (
    COST_LEDGER,
    CostCeilingExceeded,
    CostLedger,
)
from evolution.validation.agent_runner import AgentRunResult, TaskRunContext

DEFAULT_CLAUDE_TIMEOUT_SECONDS = 300
_DEFAULT_ALLOWED_TOOLS = ["Read", "Edit", "Write", "Bash", "Glob", "Grep"]
# Temp roots the agent (and claude itself) legitimately writes to, beyond the
# fixture dir + per-run HOME. macOS canonicalizes /tmp and /var/folders under /private.
_SANDBOX_TEMP_ROOTS = ("/private/tmp", "/private/var/folders", "/dev")


class SandboxUnavailableError(RuntimeError):
    """OS-level filesystem sandboxing is not available and was not waived."""


class ClaudeCodeAgentRunner:
    """Invoke ``claude -p`` (OS-sandboxed) and parse the resulting stream-json."""

    def __init__(
        self,
        *,
        append_prompt_file: Optional[Path] = None,
        model: str = "sonnet",
        timeout_seconds: int = DEFAULT_CLAUDE_TIMEOUT_SECONDS,
        allowed_tools: Optional[list[str]] = None,
        claude_command: str = "claude",
        cost_ledger: CostLedger = COST_LEDGER,
        require_sandbox: bool = True,
    ) -> None:
        # The installer owns ``append_prompt_file``; ``claude`` reads it via
        # --append-system-prompt-file each run, so re-installing a new candidate
        # is picked up without reconstructing the runner. Resolve to absolute:
        # ``claude`` runs with cwd=fixture_dir, so a relative path (the default
        # output/<...> tree is relative) would be resolved under the fixture and
        # silently not found, abstaining every task.
        self.append_prompt_file = (
            Path(append_prompt_file).resolve() if append_prompt_file is not None else None
        )
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.allowed_tools = allowed_tools or list(_DEFAULT_ALLOWED_TOOLS)
        self.claude_command = claude_command
        self.cost_ledger = cost_ledger
        self.require_sandbox = require_sandbox

    def run(self, ctx: TaskRunContext) -> AgentRunResult:
        home = Path(tempfile.mkdtemp(prefix="cl_claude_home_"))
        try:
            (home / ".claude").mkdir(parents=True, exist_ok=True)
            env = {
                **os.environ,
                "HOME": str(home),
                "CLAUDE_CODE_OAUTH_TOKEN": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", ""),
                **ctx.extra_env,
            }
            claude_argv = [
                self.claude_command, "-p", ctx.user_message,
                "--output-format", "stream-json", "--verbose",
                "--model", self.model,
                "--no-session-persistence",
                "--strict-mcp-config",
                "--permission-mode", "bypassPermissions",
                "--add-dir", str(ctx.fixture_dir),
                "--allowedTools", ",".join(self.allowed_tools),
            ]
            if self.append_prompt_file is not None:
                claude_argv += ["--append-system-prompt-file", str(self.append_prompt_file)]
            argv = self._wrap_in_sandbox(
                claude_argv, write_roots=[Path(ctx.fixture_dir).resolve(), home.resolve()]
            )
            start = time.time()
            try:
                proc = subprocess.run(
                    argv, env=env, cwd=str(ctx.fixture_dir),
                    capture_output=True, text=True,
                    timeout=self.timeout_seconds, check=False,
                )
            except subprocess.TimeoutExpired:
                result = AgentRunResult(
                    tool_calls_seq=[], final_text_tail="",
                    duration_seconds=time.time() - start,
                    error=f"claude -p timed out after {self.timeout_seconds}s",
                )
            except FileNotFoundError as exc:
                result = AgentRunResult(
                    tool_calls_seq=[], final_text_tail="",
                    duration_seconds=time.time() - start,
                    error=f"claude command not found: {exc}",
                )
            else:
                result = _parse_stream_json(
                    proc.stdout, duration_seconds=time.time() - start,
                    stderr_tail=proc.stderr[-500:],
                )
        finally:
            shutil.rmtree(home, ignore_errors=True)
        # Record + enforce the cost ceiling eagerly: Layer-1/convention scoring
        # makes no in-process LM call, so the BaseLM.__call__ guard would never
        # fire for an agent-cost overrun. Check + raise here instead.
        self.cost_ledger.record_agent_cost(result.agent_cost_usd)
        state = self.cost_ledger.get_abort_state()
        if state is not None:
            raise CostCeilingExceeded(*state)
        return result

    def _wrap_in_sandbox(self, claude_argv: list[str], *, write_roots: list[Path]) -> list[str]:
        """Prepend an OS sandbox that denies writes outside ``write_roots`` + temp.

        Returns the argv to exec. Raises SandboxUnavailableError when no OS sandbox
        is available and ``require_sandbox`` is set — refusing to run an unconfined
        agent rather than silently escaping (the prior failure mode)."""
        if sys.platform == "darwin" and shutil.which("sandbox-exec"):
            return ["sandbox-exec", "-p", _macos_write_sandbox_profile(write_roots), *claude_argv]
        if self.require_sandbox:
            raise SandboxUnavailableError(
                "No OS filesystem sandbox available on this platform "
                f"({sys.platform}); refusing to run claude unconfined under "
                "bypassPermissions. Pass require_sandbox=False to override."
            )
        return claude_argv


def _macos_write_sandbox_profile(write_roots: list[Path]) -> str:
    """SBPL profile: allow everything, then deny all writes, then re-allow writes
    only under the given roots + temp dirs. Confines Write/Edit/Bash alike."""
    roots = list(write_roots) + [Path(p) for p in _SANDBOX_TEMP_ROOTS]
    allows = "\n".join(f'    (subpath "{r}")' for r in roots)
    return (
        "(version 1)\n"
        "(allow default)\n"
        "(deny file-write*)\n"
        f"(allow file-write*\n{allows})\n"
    )


def _price_from_tokens(model: Optional[str], tokens: dict) -> Optional[float]:
    """Compute USD cost from token counts via litellm when the agent reported $0.

    Returns None when the model is unknown/unpriced or token counts are zero, so a
    truly-uncaptured run stays flagged rather than being faked as $0. Mirrors the
    Hermes runner's fallback for subscription/OAuth runs (which report $0 cost)."""
    if not model:
        return None
    inp = tokens.get("input_tokens") or 0
    out = tokens.get("output_tokens") or 0
    if inp + out == 0:
        return None
    try:
        pin, pout = litellm.cost_per_token(model=model, prompt_tokens=inp, completion_tokens=out)
        total = pin + pout
        return total if total > 0 else None
    except Exception:  # noqa: BLE001 — litellm raises widely for unknown models
        return None


def _parse_stream_json(
    stdout: str, *, duration_seconds: float, stderr_tail: str = ""
) -> AgentRunResult:
    """Parse ``claude -p --output-format stream-json`` events into an AgentRunResult.

    Events: ``system/init`` (model), ``assistant`` (content blocks incl. ``tool_use``),
    and a final ``result`` (total_cost_usd, usage, final text, is_error). Cost resolves
    actual(>0) → litellm-computed-from-tokens → uncaptured (a billed subscription run
    reports $0, so we price it from tokens rather than silently counting it free).
    """
    tool_calls_seq: list[str] = []
    tool_calls_with_args: list[dict] = []
    final_text = ""
    cost: Optional[float] = None
    tokens: dict = {}
    model_name: Optional[str] = None
    err: Optional[str] = None
    saw_result = False

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = ev.get("type")
        if etype == "system" and ev.get("subtype") == "init":
            model_name = ev.get("model") or model_name
        elif etype == "assistant":
            for block in (ev.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name")
                    if name:
                        tool_calls_seq.append(str(name))
                        args = block.get("input")
                        tool_calls_with_args.append(
                            {"name": str(name),
                             "arguments": args if isinstance(args, dict) else {}}
                        )
        elif etype == "result":
            saw_result = True
            final_text = str(ev.get("result") or "")
            cost = ev.get("total_cost_usd")
            u = ev.get("usage") or {}
            tokens = {
                "input_tokens": u.get("input_tokens", 0),
                "output_tokens": u.get("output_tokens", 0),
                "cache_read_tokens": u.get("cache_read_input_tokens", 0),
                "cache_write_tokens": u.get("cache_creation_input_tokens", 0),
            }
            if ev.get("is_error"):
                err = f"claude result is_error: {final_text[:200]}"

    if err is None and not saw_result:
        err = f"no result event in claude stream-json output (stderr: {stderr_tail[:200]})"
    # Degraded/empty completion (transport drop, interrupted run): a result event
    # that produced no tools, no text, and no tokens is not artifact-quality
    # evidence — abstain rather than score it as a (false) loss/pass.
    if err is None and not tool_calls_seq and not final_text.strip() \
            and (tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0)) == 0:
        err = "claude returned an empty/degraded result (no tools, text, or tokens)"

    if cost is not None and cost > 0:
        agent_cost, source = cost, "actual"
    else:
        computed = _price_from_tokens(model_name, tokens)
        agent_cost, source = (computed, "computed") if computed is not None else (None, "uncaptured")

    return AgentRunResult(
        tool_calls_seq=tool_calls_seq,
        final_text_tail=final_text[-4096:],
        duration_seconds=duration_seconds,
        model_name=model_name,
        error=err,
        tool_calls_with_args=tool_calls_with_args,
        agent_cost_usd=agent_cost,
        agent_cost_source=source,  # type: ignore[arg-type]
        agent_tokens=tokens,
    )

"""Drive Hermes Agent via ``hermes -z`` one-shot mode and parse the result.

Sandboxes the subprocess by overriding ``HERMES_HOME`` (Hermes's own
config seam) and the process's ``HOME`` to the same tmp dir, and
running with ``cwd`` set to the task's per-task fixture directory.
That keeps the agent's interactive-shell aliases and personal
``.config`` paths invisible, and any file writes the agent does default
into a directory the harness controls.

The agent runs with ``HERMES_YOLO_MODE=1`` (Hermes one-shot mode sets
this itself) so it auto-approves tool prompts; v1 has no further
sandboxing. The agent can still write anywhere it has UID for.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import litellm

from evolution.core.lm_timing_callback import (
    COST_LEDGER,
    CostCeilingExceeded,
    CostLedger,
)
from evolution.validation.agent_runner import (
    AgentCostSource,
    AgentRunResult,
    TaskRunContext,
)

logger = logging.getLogger(__name__)


_FINAL_TEXT_TAIL_BYTES = 4096
DEFAULT_TASK_TIMEOUT_SECONDS = 120

# Known LiteLLM provider prefixes the rest of the framework uses
# (DSPy / litellm convention: ``<provider>/<model>``). The hermes -m flag
# interprets the same shape as openrouter-style routing — passing
# ``openai/gpt-4o-mini`` silently switches base_url to openrouter.ai and
# breaks auth for direct-provider configs, producing a 0-turn session that
# the saturation pre-flight misreports as "validator too weak". We strip
# these prefixes at the hermes boundary so users get the behavior they
# expect when they pass the same model string they use elsewhere.
_LITELLM_PROVIDER_PREFIXES = (
    "openai/",
    "anthropic/",
    "azure/",
    "gemini/",
    "cohere/",
    "bedrock/",
    "mistral/",
)


def _price_from_tokens(model: Optional[str], agent_tokens: dict) -> Optional[float]:
    """Price a run from captured token counts via litellm when hermes didn't report a cost.

    Returns None if the model is unknown, unset, token counts are zero, or
    litellm prices the run at $0 (recognized-but-unpriced model — a $0 here is a
    pricing gap, not a free run, so it must flow to the uncaptured flag rather
    than be trusted).

    The computed cost is a **lower bound**: it prices ``input_tokens`` +
    ``output_tokens`` only and omits cache/reasoning tokens, so it may UNDER-count
    cache- or reasoning-heavy runs (a known follow-up). Adding those tokens here
    risks double-counting against litellm's own cache accounting, so it's deferred.
    """
    if not model:
        return None
    input_t = agent_tokens.get("input_tokens") or 0
    output_t = agent_tokens.get("output_tokens") or 0
    if input_t + output_t == 0:
        return None
    try:
        pin, pout = litellm.cost_per_token(
            model=model, prompt_tokens=input_t, completion_tokens=output_t
        )
        total = pin + pout
        return total if total > 0 else None
    except Exception:  # noqa: BLE001 — litellm raises widely for unknown models
        return None


def _strip_litellm_provider_prefix(model: str) -> str:
    """Strip a known LiteLLM provider prefix from a model name.

    Returns ``model`` unchanged when no recognized prefix is present, so
    openrouter-style routing through an unrecognized vendor still works.
    """
    for prefix in _LITELLM_PROVIDER_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


class HermesAgentRunner:
    """Invoke ``hermes -z`` and parse the resulting session JSON.

    The exit code is intentionally ignored — Hermes one-shot returns 0
    on almost every code path including agent-loop crashes. The session
    JSON is the only source of truth for what the agent did.
    """

    def __init__(
        self,
        hermes_command: str = "hermes",
        timeout_seconds: int = DEFAULT_TASK_TIMEOUT_SECONDS,
        user_config_path: Optional[Path] = None,
        model: Optional[str] = None,
        cost_ledger: CostLedger = COST_LEDGER,
    ) -> None:
        self.hermes_command = hermes_command
        self.timeout_seconds = timeout_seconds
        self.cost_ledger = cost_ledger
        # If set, copied into the sandboxed HERMES_HOME so the agent picks
        # up the user's credentials. Defaults to ``~/.hermes/config.yaml``.
        self.user_config_path = (
            user_config_path
            if user_config_path is not None
            else Path.home() / ".hermes" / "config.yaml"
        )
        # Optional per-invocation model override (passed as ``hermes -z -m
        # <model>``). When unset, Hermes uses whatever is configured in
        # the sandboxed ``config.yaml``. Useful for closed-loop validation
        # against a deliberately weaker agent model than the user's
        # daily-driver default — saturation on capable models hides
        # behavioral signal that a weaker model would expose.
        #
        # Normalize LiteLLM-style provider prefixes (``openai/``, etc.)
        # before storing: hermes -m treats ``<provider>/<model>`` as
        # openrouter routing which silently switches the base_url. See
        # ``_strip_litellm_provider_prefix`` for the full rationale.
        if model is not None:
            normalized = _strip_litellm_provider_prefix(model)
            if normalized != model:
                logger.info(
                    "Stripped LiteLLM provider prefix from hermes -m model: "
                    "%r → %r (avoids accidental openrouter routing)",
                    model, normalized,
                )
            self.model = normalized
        else:
            self.model = None

    def run(self, ctx: TaskRunContext) -> AgentRunResult:
        message = ctx.user_message
        sandbox = Path(tempfile.mkdtemp(prefix="cl_hermes_home_"))
        try:
            self._prime_sandbox(sandbox, ctx)
            env = {
                **os.environ,
                "HERMES_HOME": str(sandbox),
                "HOME": str(sandbox),
                **ctx.extra_env,
            }
            argv = [self.hermes_command, "-z", message]
            if self.model is not None:
                # Insert before the -z so hermes parses it as a global flag,
                # not as part of the -z prompt value.
                argv = [self.hermes_command, "-m", self.model, "-z", message]
            start = time.time()
            try:
                subprocess.run(
                    argv,
                    env=env,
                    cwd=str(ctx.fixture_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                result = AgentRunResult(
                    tool_calls_seq=[],
                    final_text_tail="",
                    duration_seconds=time.time() - start,
                    error=f"hermes -z timed out after {self.timeout_seconds}s",
                )
            except FileNotFoundError as exc:
                result = AgentRunResult(
                    tool_calls_seq=[],
                    final_text_tail="",
                    duration_seconds=time.time() - start,
                    error=f"hermes command not found: {exc}",
                )
            else:
                duration = time.time() - start

                # Modern hermes persists the session to a SQLite ``state.db`` in
                # HERMES_HOME (one-shot ``-z`` no longer writes ``session_*.json``).
                db_path = sandbox / "state.db"
                if not db_path.is_file():
                    result = AgentRunResult(
                        tool_calls_seq=[],
                        final_text_tail="",
                        duration_seconds=duration,
                        error="no session written by hermes -z (state.db absent)",
                    )
                else:
                    try:
                        result = parse_session_from_db(db_path, duration_seconds=duration)
                    except Exception as exc:  # noqa: BLE001 — keep result bound so the run is recorded
                        logger.warning("failed to parse session db: %s", exc)
                        result = AgentRunResult(
                            tool_calls_seq=[],
                            final_text_tail="",
                            duration_seconds=duration,
                            error=f"session db parse failed: {exc}",
                        )
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)
        self.cost_ledger.record_agent_cost(result.agent_cost_usd)
        # Enforce the ceiling eagerly: Layer-1/test_command scoring makes no
        # in-process LM call, so the BaseLM.__call__ guard would never fire for
        # an agent-cost overrun. Check + raise here instead.
        state = self.cost_ledger.get_abort_state()
        if state is not None:
            raise CostCeilingExceeded(*state)
        return result

    def _prime_sandbox(self, sandbox: Path, ctx: TaskRunContext) -> None:
        (sandbox / "sessions").mkdir(parents=True, exist_ok=True)
        if self.user_config_path.exists():
            shutil.copy2(self.user_config_path, sandbox / "config.yaml")
        if ctx.skills_src is not None and ctx.skills_src.is_dir():
            # Copy (not symlink) so an in-task write by the agent corrupts
            # only this sandbox, not the installer's persistent candidate.
            shutil.copytree(ctx.skills_src, sandbox / "skills")


def parse_session_result(
    session_path: Path,
    *,
    duration_seconds: float,
) -> AgentRunResult:
    """Read a Hermes session JSON and extract the tool-call sequence + final text.

    Retained for the legacy ``session_*.json`` shape and unit tests that
    exercise the message extractors with hand-crafted fixtures. The live
    runner reads ``state.db`` via ``parse_session_from_db``.
    """
    try:
        data = json.loads(session_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return AgentRunResult(
            tool_calls_seq=[],
            final_text_tail="",
            duration_seconds=duration_seconds,
            error=f"could not parse session JSON at {session_path}: {exc}",
            session_path=session_path,
        )

    messages = data.get("messages") or []
    return _result_from_messages(
        messages,
        duration_seconds=duration_seconds,
        model_name=data.get("model"),
        session_path=session_path,
    )


def parse_session_from_db(
    db_path: Path,
    *,
    duration_seconds: float,
) -> AgentRunResult:
    """Reconstruct an ``AgentRunResult`` from a Hermes ``state.db``.

    Modern hermes persists each session to SQLite. We read the most-recent
    session's messages and normalize them into the same message-dict shape the
    legacy JSON path produced, so the existing extractors work unchanged. The
    ``messages.tool_calls`` column holds the tool-call list verbatim — current
    hermes writes the flat ``{"name", "arguments"}`` shape; the extractors also
    accept the older OpenAI-nested ``{"function": {...}}`` shape.

    A row whose ``tool_calls`` column won't parse as JSON aborts with an
    ``error`` result (the task abstains) rather than being silently read as
    "agent invoked no tools" — that would score a DB-format regression as a
    behavioral failure and contaminate the fitness signal.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return AgentRunResult(
            tool_calls_seq=[],
            final_text_tail="",
            duration_seconds=duration_seconds,
            error=f"could not open session DB at {db_path}: {exc}",
            session_path=db_path,
        )
    try:
        conn.row_factory = sqlite3.Row
        # Attempt the extended SELECT that includes cost/token columns added in
        # recent hermes builds. Fall back to the minimal id/model SELECT when
        # those columns are absent (schema-drift against an older hermes binary)
        # so the run still contributes behavioral signal rather than crashing.
        try:
            session = conn.execute(
                "SELECT id, model, actual_cost_usd, estimated_cost_usd, "
                "cost_status, input_tokens, output_tokens, cache_read_tokens, "
                "cache_write_tokens, reasoning_tokens "
                "FROM sessions ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            _has_cost_cols = True
        except sqlite3.OperationalError as exc:
            # Only treat a genuine missing-column as schema drift. A locked /
            # corrupt / missing-table OperationalError must re-raise so the outer
            # sqlite3.Error handler abstains rather than mislabeling it as drift.
            if "no such column" not in str(exc).lower():
                raise
            session = conn.execute(
                "SELECT id, model FROM sessions ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            _has_cost_cols = False
        if session is None:
            return AgentRunResult(
                tool_calls_seq=[],
                final_text_tail="",
                duration_seconds=duration_seconds,
                error=f"session DB at {db_path} has no sessions",
                session_path=db_path,
            )
        rows = conn.execute(
            "SELECT role, content, tool_calls FROM messages "
            "WHERE session_id = ? ORDER BY id",
            (session["id"],),
        ).fetchall()
    except sqlite3.Error as exc:
        return AgentRunResult(
            tool_calls_seq=[],
            final_text_tail="",
            duration_seconds=duration_seconds,
            error=f"could not read session DB at {db_path}: {exc}",
            session_path=db_path,
        )
    finally:
        conn.close()

    # Resolve cost fields from the session row (or leave uncaptured on drift).
    if _has_cost_cols:
        actual = session["actual_cost_usd"]
        estimated = session["estimated_cost_usd"]
        _tok_keys = ("input_tokens", "output_tokens", "cache_read_tokens",
                     "cache_write_tokens", "reasoning_tokens")
        agent_tokens = {k: session[k] for k in _tok_keys if session[k] is not None}

        # Resolution order:
        #   actual (> 0, or == 0 with zero tokens as genuine-zero) →
        #   computed via litellm from tokens →
        #   estimated (> 0) →
        #   uncaptured
        # A $0 hermes cost alongside real token usage is an unpriced model, not a
        # free run, so it flows into the computed or uncaptured path rather than
        # being trusted as $0.
        _total_tokens = sum(agent_tokens.values())
        model_name: Optional[str] = session["model"]
        if actual is not None and (actual > 0.0 or _total_tokens == 0):
            agent_cost_usd: Optional[float] = actual
            agent_cost_source = "actual"
        else:
            computed = _price_from_tokens(model_name, agent_tokens)
            if computed is not None:
                agent_cost_usd = computed
                agent_cost_source = "computed"
            elif estimated is not None and estimated > 0.0:
                agent_cost_usd = estimated
                agent_cost_source = "estimated"
            else:
                agent_cost_usd = None
                agent_cost_source = "uncaptured"
    else:
        agent_cost_usd = None
        agent_cost_source = "uncaptured"
        agent_tokens: dict = {}

    messages: list[dict] = []
    for row in rows:
        raw_calls = row["tool_calls"]
        parsed_calls: Any = None
        if raw_calls:
            try:
                parsed_calls = json.loads(raw_calls)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(
                    "malformed tool_calls JSON in session %s at %s (%s); "
                    "abstaining rather than scoring the task as a no-op",
                    session["id"], db_path, exc,
                )
                return AgentRunResult(
                    tool_calls_seq=[],
                    final_text_tail="",
                    duration_seconds=duration_seconds,
                    error=f"malformed tool_calls JSON in session DB at {db_path}: {exc}",
                    session_path=db_path,
                )
        messages.append({
            "role": row["role"],
            "content": row["content"] or "",
            "tool_calls": parsed_calls,
        })
    return _result_from_messages(
        messages,
        duration_seconds=duration_seconds,
        model_name=session["model"],
        session_path=db_path,
        agent_cost_usd=agent_cost_usd,
        agent_cost_source=agent_cost_source,
        agent_tokens=agent_tokens,
    )


def _result_from_messages(
    messages: list[dict],
    *,
    duration_seconds: float,
    model_name: Optional[str],
    session_path: Optional[Path],
    agent_cost_usd: Optional[float] = None,
    agent_cost_source: AgentCostSource = "uncaptured",
    agent_tokens: Optional[dict] = None,
) -> AgentRunResult:
    """Build an ``AgentRunResult`` from a normalized message list."""
    return AgentRunResult(
        tool_calls_seq=_extract_tool_call_names(messages),
        final_text_tail=_extract_final_text_tail(messages),
        duration_seconds=duration_seconds,
        model_name=model_name,
        session_path=session_path,
        tool_calls_with_args=_extract_tool_calls_with_args(messages),
        agent_cost_usd=agent_cost_usd,
        agent_cost_source=agent_cost_source,
        agent_tokens=agent_tokens if agent_tokens is not None else {},
    )


def _extract_tool_call_names(messages: list[dict]) -> list[str]:
    """Pull tool names from every assistant turn's tool_calls.

    Hermes session JSON has been observed to carry tool_calls in two
    shapes depending on the model:
      - OpenAI-style nested:   {"function": {"name": "X", "arguments": ...}}
      - Flat:                  {"name": "X", "arguments": ...}
    Handle both. Multi-tool-call assistant turns contribute each call
    in order.
    """
    out: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = _call_name(call)
            if name:
                out.append(name)
    return out


def _call_name(call: dict) -> Optional[str]:
    fn = call.get("function")
    if isinstance(fn, dict):
        nested = fn.get("name")
        if nested:
            return str(nested)
    flat = call.get("name")
    return str(flat) if flat else None


def _extract_tool_calls_with_args(messages: list[dict]) -> list[dict]:
    """Return ``[{name, arguments}, ...]`` for each assistant tool call.

    Arguments are parsed from the LLM-emitted JSON string. Malformed or
    non-object arguments fall back to ``{}`` rather than dropping the
    call — the Layer 2 judge can still treat "memory was invoked with
    empty args" as a behavior signal. Handles both OpenAI-nested and flat
    tool_call shapes, mirroring ``_extract_tool_call_names``.
    """
    out: list[dict] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            name = _call_name(call)
            if not name:
                continue
            args_raw = _call_arguments_raw(call)
            try:
                args = json.loads(args_raw) if args_raw else {}
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "malformed arguments for tool call %r (%r); using {} — a "
                    "content judge will see an empty-args call",
                    name, args_raw[:120],
                )
                args = {}
            if not isinstance(args, dict):
                args = {}
            out.append({"name": name, "arguments": args})
    return out


def _call_arguments_raw(call: dict) -> str:
    fn = call.get("function")
    if isinstance(fn, dict):
        nested = fn.get("arguments")
        if isinstance(nested, str):
            return nested
        if isinstance(nested, dict):
            return json.dumps(nested)
    flat = call.get("arguments")
    if isinstance(flat, str):
        return flat
    if isinstance(flat, dict):
        return json.dumps(flat)
    return ""


def _extract_final_text_tail(messages: list[dict]) -> str:
    """Last 4096 chars of the last assistant message with text content."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content[-_FINAL_TEXT_TAIL_BYTES:]
    return ""

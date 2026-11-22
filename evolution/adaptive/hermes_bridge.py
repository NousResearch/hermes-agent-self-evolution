"""Subprocess bridge: run ONE Hermes Agent turn through its documented Python API.

``HermesExecutor`` runs this file's source as::

    <hermes python> -c <source of this file> REQUEST.json RESULT.json

with the hermes-agent checkout as the working directory, so ``import run_agent``
resolves to that checkout. This follows hermes-agent's own guide
``website/docs/guides/python-library.md`` ("Using Hermes as a Python Library"),
which says to run the embedding program from the (activated) checkout.

Hermes interfaces called (verified in NousResearch/hermes-agent at caafe5389a):

* ``run_agent.AIAgent(**kwargs)`` -- ``run_agent.py`` class ``AIAgent``; its
  ``__init__`` forwards the keyword arguments to ``agent.agent_init.init_agent``.
* ``AIAgent.run_conversation(user_message=..., task_id=...) -> dict`` --
  ``agent/turn_facade.py`` ``TurnFacadeMixin.run_conversation``. The result dict
  is built in ``agent/turn_finalizer.py`` (``final_response``, ``completed``,
  ``failed``, ``partial``, ``interrupted``, ``error``, ``turn_exit_reason``,
  ``api_calls``, ``*_tokens``, ``estimated_cost_usd``, ``cost_status``,
  ``guardrail``, ``messages``) or by the early-exit envelopes in
  ``agent/conversation_loop.py``.
* ``AIAgent.close()`` -- ``run_agent.py``; releases clients and task resources.

Stdlib only. This module must never import Hermes at module import time: the
unit tests import it to exercise the pure helpers and ``main`` against a stub
``run_agent`` module, with no network and no API keys.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time

# Run-result keys copied verbatim when present (see agent/turn_finalizer.py).
_BOOL_KEYS = ("completed", "failed", "partial", "interrupted")
_INT_KEYS = ("api_calls", "input_tokens", "output_tokens", "total_tokens",
             "cache_read_tokens", "cache_write_tokens")
_STR_KEYS = ("turn_exit_reason", "cost_status", "cost_source", "model", "provider",
             "failure_reason")
_MAX_TEXT = 200_000   # cap on final_response characters written back
_MAX_ERROR = 2_000


def build_agent_kwargs(request: dict) -> dict:
    """Keyword arguments for ``run_agent.AIAgent`` from a bridge request.

    Every isolation switch is fixed here, not taken from the request, so a
    strategy cannot turn memory, context files, trajectories or background
    review back on. ``enabled_toolsets=[]`` means *no tools*: hermes-agent's
    ``model_tools._select_tool_names`` treats only ``None`` as "all toolsets".
    """
    enabled = request["enabled_toolsets"]
    disabled = request["disabled_toolsets"]
    if not isinstance(enabled, list) or not isinstance(disabled, list):
        raise ValueError("toolset selections must be lists (None would enable every toolset)")
    kwargs = {
        "model": request.get("model") or "",
        "quiet_mode": True,                  # python-library.md: always set when embedding
        "enabled_toolsets": list(enabled),
        "disabled_toolsets": list(disabled),
        "max_iterations": int(request["max_iterations"]),
        "run_budget_seconds": float(request["run_budget_seconds"]),
        "ephemeral_system_prompt": request.get("system_prompt") or None,
        "save_trajectories": False,
        "skip_context_files": True,          # no AGENTS.md / SOUL.md from the cwd
        "skip_memory": True,                 # no persistent memory read or write
        "skip_background_review": True,      # no post-turn review fork (may write skills/memory)
        "checkpoints_enabled": False,
    }
    if request.get("provider"):
        kwargs["provider"] = request["provider"]
    if request.get("max_tokens") is not None:
        kwargs["max_tokens"] = int(request["max_tokens"])
    return kwargs


def _tool_call_names(messages) -> list[str]:
    """Names of every tool the assistant called in ``messages`` (OpenAI message shape)."""
    names: list[str] = []
    for message in messages or ():
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            if not isinstance(call, dict):
                names.append("<unparseable>")
                continue
            function = call.get("function")
            name = function.get("name") if isinstance(function, dict) else call.get("name")
            names.append(str(name) if name else "<unnamed>")
    return names


def _finite_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def summarise_result(result: dict, latency_s: float) -> dict:
    """JSON-safe summary of a ``run_conversation`` result dict."""
    if not isinstance(result, dict):
        raise TypeError(f"run_conversation returned {type(result).__name__}, expected dict")
    summary: dict = {"latency_s": float(latency_s)}
    final = result.get("final_response")
    summary["final_response"] = final[:_MAX_TEXT] if isinstance(final, str) else ""
    for key in _BOOL_KEYS:
        if isinstance(result.get(key), bool):
            summary[key] = result[key]
    for key in _INT_KEYS:
        value = result.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            summary[key] = value
    for key in _STR_KEYS:
        if result.get(key) is not None:
            summary[key] = str(result[key])
    summary["estimated_cost_usd"] = _finite_or_none(result.get("estimated_cost_usd"))
    if result.get("error"):
        summary["error"] = str(result["error"])[:_MAX_ERROR]
    guardrail = result.get("guardrail")
    if isinstance(guardrail, dict):
        summary["guardrail"] = {k: guardrail[k] for k in ("action", "code", "tool_name")
                                if isinstance(guardrail.get(k), str)}
    summary["tool_calls"] = _tool_call_names(result.get("messages"))
    return summary


def run(request: dict) -> dict:
    """Run one Hermes turn; returns ``summarise_result`` output."""
    repo = request.get("hermes_repo")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    from run_agent import AIAgent  # documented entry point (python-library.md)

    agent = AIAgent(**build_agent_kwargs(request))
    try:
        start = time.perf_counter()
        result = agent.run_conversation(user_message=request["prompt"],
                                        task_id=request["task_id"])
        latency = time.perf_counter() - start
    finally:
        with contextlib.suppress(Exception):
            agent.close()
    return summarise_result(result, latency)


def _write_json(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False)
    os.replace(tmp, path)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: hermes_bridge REQUEST.json RESULT.json", file=sys.stderr)
        return 2
    request_path, result_path = argv
    with open(request_path, encoding="utf-8") as handle:
        request = json.load(handle)
    try:
        # Hermes may print despite quiet_mode; keep the bridge's own stdout clean.
        with contextlib.redirect_stdout(sys.stderr):
            payload = {"ok": True, **run(request)}
    except Exception as exc:  # reported as an execution error, never raised to the loop
        payload = {"ok": False, "exception_type": type(exc).__name__,
                   "error": f"{type(exc).__name__}: {exc}"[:_MAX_ERROR]}
    _write_json(result_path, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""OpenAI-compatible HTTP shim that routes chat completions through the
Claude Code CLI's OAuth subscription (via the official ``claude-agent-sdk``).

This lets tools that speak OpenAI (litellm, dspy, etc.) use a Claude Code
subscription instead of an Anthropic Console API key. Useful when:

  * you already pay for Claude Code and don't want a second metered API key
  * you want token usage to count against your subscription, not a separate
    billing account

USAGE
-----
Start the shim (defaults to 127.0.0.1:8765)::

    python scripts/claude_oai_shim.py

Point an OpenAI-compatible client at it::

    export OPENAI_BASE_URL=http://127.0.0.1:8765/v1
    export OPENAI_API_KEY=anything   # ignored, but most clients require it
    python -m evolution.skills.evolve_skill \\
        --skill github-code-review \\
        --optimizer-model openai/claude-opus-4-7 \\
        --eval-model openai/claude-haiku-4-5

Optional: set ``SHIM_API_KEY`` and pass ``Authorization: Bearer <key>`` so
only trusted local clients can use the endpoint.

MODEL ROUTING
-------------
Use OpenAI-style model names with a ``claude-*`` suffix. The shim maps
families to Claude Code's three tiers; specific minor versions resolve to
the latest of that family.

    claude-opus-*    -> opus
    claude-sonnet-*  -> sonnet
    claude-haiku-*   -> haiku

ENDPOINTS
---------
GET  /health                  -> {"ok": true}
POST /v1/chat/completions     -> OpenAI Chat Completions response shape

CAVEATS
-------
- Streaming is not yet supported (``stream: true`` returns HTTP 400).
- Token usage counts are best-effort and may be 0 if the SDK doesn't expose
  them for the current model.
- This routes through your Claude Code subscription. Review your plan's
  acceptable-use terms before automating against it.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

MODEL_MAP = {
    "opus": os.getenv("CLAUDE_SHIM_OPUS", "claude-opus-4-1-20250805"),
    "sonnet": os.getenv("CLAUDE_SHIM_SONNET", "claude-sonnet-4-5"),
    "haiku": os.getenv("CLAUDE_SHIM_HAIKU", "claude-haiku-4-5"),
}

DEFAULT_MAX_MESSAGE_CHARS = 500_000


def _resolve_model(name: str) -> str:
    """Map an OpenAI-style model name to a Claude tier."""
    n = name.lower()
    if "opus" in n:
        return MODEL_MAP["opus"]
    if "haiku" in n:
        return MODEL_MAP["haiku"]
    if "sonnet" in n or "claude" in n:
        return MODEL_MAP["sonnet"]
    if "gpt" in n or "o1" in n or "o3" in n:
        logger.warning(
            "Model %r does not look like a Claude name; routing to sonnet",
            name,
        )
    return MODEL_MAP["sonnet"]


def _messages_to_prompt(messages: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Flatten OpenAI messages into a single prompt + optional system."""
    system_parts: list[str] = []
    convo_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            convo_parts.append(f"Assistant: {content}")
        else:
            convo_parts.append(f"User: {content}")
    system = "\n\n".join(system_parts) if system_parts else None
    prompt = "\n\n".join(convo_parts).strip()
    return prompt, system


def _usage_tokens(usage: Any, key: str) -> int:
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    return int(getattr(usage, key, 0) or 0)


def _openai_error(
    message: str,
    *,
    err_type: str = "invalid_request_error",
    code: str | None = None,
    status: int = 400,
) -> JSONResponse:
    body: dict[str, Any] = {
        "error": {"message": message, "type": err_type, "code": code},
    }
    return JSONResponse(status_code=status, content=body)


def _verify_api_key(
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    expected = os.environ.get("SHIM_API_KEY")
    if not expected:
        return
    if not creds or creds.credentials != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _total_message_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(str(part.get("text", "")))
        else:
            total += len(str(content))
    return total


app = FastAPI(title="claude-oai-shim", version="0.2.0")


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    _: None = Depends(_verify_api_key),
) -> JSONResponse:
    body = await request.json()
    if body.get("stream"):
        return _openai_error(
            "Streaming is not yet supported by this shim.",
            err_type="invalid_request_error",
        )

    messages = body.get("messages", [])
    if not messages:
        return _openai_error("messages must be a non-empty array")

    max_chars = int(
        os.environ.get("SHIM_MAX_MESSAGE_CHARS", DEFAULT_MAX_MESSAGE_CHARS)
    )
    if _total_message_chars(messages) > max_chars:
        return _openai_error(
            f"Total message content exceeds limit of {max_chars} characters",
        )

    model = body.get("model", "claude-sonnet-4-5")
    from claude_agent_sdk import ClaudeAgentOptions, query

    resolved = _resolve_model(model)
    prompt, system = _messages_to_prompt(messages)
    options = ClaudeAgentOptions(model=resolved, system_prompt=system)

    text_chunks: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        async for message in query(prompt=prompt, options=options):
            content = getattr(message, "content", None)
            if content:
                for block in content:
                    t = getattr(block, "text", None)
                    if t:
                        text_chunks.append(t)
            u = getattr(message, "usage", None)
            if u:
                usage["input_tokens"] = _usage_tokens(u, "input_tokens")
                usage["output_tokens"] = _usage_tokens(u, "output_tokens")
    except Exception as exc:
        logger.exception("claude-agent-sdk query failed")
        return _openai_error(
            str(exc),
            err_type="server_error",
            status=500,
        )

    response_text = "".join(text_chunks).strip()
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": resolved,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage["input_tokens"],
                "completion_tokens": usage["output_tokens"],
                "total_tokens": usage["input_tokens"] + usage["output_tokens"],
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude Code OpenAI shim")
    parser.add_argument("--host", default=os.getenv("CLAUDE_SHIM_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("CLAUDE_SHIM_PORT", "8765")),
    )
    args = parser.parse_args()
    if os.environ.get("SHIM_API_KEY"):
        logger.info("SHIM_API_KEY is set — /v1/chat/completions requires Bearer auth")
    else:
        logger.warning(
            "SHIM_API_KEY is not set — any local process can call this endpoint"
        )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

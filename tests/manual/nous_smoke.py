"""Manual end-to-end smoke for the Nous Portal OAuth + agent_key flow.

Why this exists:
  We have unit tests for NousLM that mock httpx.Client at the Python
  layer. Those catch shape bugs but don't validate that the real network
  call we'd make against `portal.nousresearch.com` carries the right
  headers, body, and bearer. The user has no Nous Portal account, so
  we can't run a true end-to-end smoke against the real portal.

What this script does:
  Spin up a stdlib http.server on a random localhost port that pretends
  to be the Nous portal. Routes:
    POST /api/oauth/token       — refresh_token grant
    POST /api/oauth/agent-key   — agent_key mint
    POST /v1/chat/completions   — OpenAI-compat inference (so the actual
                                  inference call returns 200 too)

  Construct a NousLM via the real resolver pointed at the local server,
  drive several scenarios (initial mint, OAuth refresh + mint, mid-run
  401 recovery), and assert the recorded HTTP exchange matches expected
  shape.

How to run:
  uv run python tests/manual/nous_smoke.py

  Exits 0 on success, prints a recorded-requests summary, and 1 on any
  failed assertion. Not part of CI — heavyweight (spins up a server)
  and not needed on every commit.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import litellm

# Override module-level URLs BEFORE importing nous_lm so the constants
# pick up the local server. Real callers would set HERMES_PORTAL_BASE_URL
# in their shell; we set it here for the in-process smoke.
import os
_PORT_HOLDER = {"port": 0}


# ---------------------------------------------------------------------------
# Mock portal server
# ---------------------------------------------------------------------------


class _RecordingHandler(BaseHTTPRequestHandler):
    """Routes the four endpoints we care about and records every request."""

    recorded: deque = deque()
    # Behavior knobs flipped per scenario from the main thread.
    behavior: Dict[str, Any] = {
        "refresh_status": 200,
        "refresh_body": None,        # set per scenario
        "mint_status": 200,
        "mint_body": None,
        "mint_call_count": 0,
        "mint_first_status": None,   # 401 then 200, for the refresh-retry test
        "infer_status": 200,
        "infer_call_count": 0,
        "infer_first_status": None,  # 401 then 200, for the inference-retry test
    }

    def log_message(self, format, *args):  # silence default access logs
        pass

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(length) if length > 0 else b""

    def _record(self, body: bytes) -> Dict[str, Any]:
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = body.decode("utf-8", errors="replace")
        entry = {
            "method": self.command,
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "body": parsed,
        }
        self.recorded.append(entry)
        return entry

    def _respond(self, status: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):  # noqa: N802 — http.server convention
        body = self._read_body()
        self._record(body)

        if self.path.endswith("/api/oauth/token"):
            self._respond(
                self.behavior["refresh_status"],
                self.behavior["refresh_body"]
                or {
                    "access_token": "REFRESHED-OAUTH",
                    "refresh_token": "REFRESHED-REFRESH",
                    "expires_in": 86400,
                    "token_type": "Bearer",
                },
            )
            return

        if self.path.endswith("/api/oauth/agent-key"):
            self.behavior["mint_call_count"] += 1
            # First-call override (used to simulate "stale OAuth → mint 401 → refresh + retry")
            if (
                self.behavior["mint_first_status"] is not None
                and self.behavior["mint_call_count"] == 1
            ):
                self._respond(
                    self.behavior["mint_first_status"],
                    {"error": "invalid_token"},
                )
                return
            future = datetime.now(tz=timezone.utc) + timedelta(seconds=1800)
            self._respond(
                self.behavior["mint_status"],
                self.behavior["mint_body"]
                or {
                    "api_key": f"MINTED-AGENT-KEY-{self.behavior['mint_call_count']}",
                    "key_id": "test-key-id",
                    "expires_at": future.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                    "expires_in": 1800,
                    "reused": False,
                },
            )
            return

        if "/chat/completions" in self.path:
            self.behavior["infer_call_count"] += 1
            if (
                self.behavior["infer_first_status"] is not None
                and self.behavior["infer_call_count"] == 1
            ):
                self._respond(
                    self.behavior["infer_first_status"],
                    {"error": {"code": "invalid_api_key", "message": "401"}},
                )
                return
            self._respond(
                self.behavior["infer_status"],
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "OK"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
            return

        self._respond(404, {"error": "unknown_route", "path": self.path})


# ---------------------------------------------------------------------------
# Scenario harness
# ---------------------------------------------------------------------------


def _start_server() -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    _PORT_HOLDER["port"] = server.server_port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _reset_recordings():
    _RecordingHandler.recorded.clear()
    _RecordingHandler.behavior.update(
        {
            "refresh_status": 200,
            "refresh_body": None,
            "mint_status": 200,
            "mint_body": None,
            "mint_call_count": 0,
            "mint_first_status": None,
            "infer_status": 200,
            "infer_call_count": 0,
            "infer_first_status": None,
        }
    )


def _make_lm(*, port: int, **state):
    """Construct a NousLM pointed at the local mock server."""
    from evolution.core.nous_lm import NousLM, _reset_state_for_tests

    _reset_state_for_tests()
    base_url = f"http://127.0.0.1:{port}"
    defaults = dict(
        access_token="seed-oauth",
        refresh_token="seed-refresh",
        oauth_expires_at=time.time() + 86400,
        agent_key=None,
        agent_key_expires_at=None,
        portal_base_url=base_url,
        inference_base_url=f"{base_url}/v1",
        # cache=False so each smoke scenario actually hits the wire — DSPy's
        # response cache ignores api_key/api_base in the cache key, which
        # would otherwise let one scenario's response leak into the next.
        # num_retries=0 because LiteLLM's internal retry-on-401 would
        # transparently recover before NousLM.forward's 401 handler sees
        # the failure, masking our re-mint logic.
        cache=False,
        num_retries=0,
    )
    defaults.update(state)
    return NousLM(model="openai/test-model", **defaults)


def _summary(label: str) -> str:
    lines = [f"\n=== {label} ==="]
    for i, r in enumerate(_RecordingHandler.recorded):
        auth = r["headers"].get("Authorization", "<none>")
        body_preview = (
            json.dumps(r["body"])[:80] if not isinstance(r["body"], str) else r["body"][:80]
        )
        lines.append(
            f"  [{i}] {r['method']} {r['path']:30} auth={auth[:40]:40} body={body_preview}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenario_initial_mint(port: int) -> List[str]:
    """Fresh construction with no agent_key → one mint POST, no refresh."""
    failures = []
    _reset_recordings()
    lm = _make_lm(port=port)

    if _RecordingHandler.behavior["mint_call_count"] != 1:
        failures.append(
            f"expected 1 mint call, got {_RecordingHandler.behavior['mint_call_count']}"
        )
    refresh_calls = sum(
        1 for r in _RecordingHandler.recorded if r["path"].endswith("/api/oauth/token")
    )
    if refresh_calls != 0:
        failures.append(
            f"OAuth refresh fired unnecessarily ({refresh_calls} call(s))"
        )
    mint_record = next(
        r for r in _RecordingHandler.recorded if r["path"].endswith("/api/oauth/agent-key")
    )
    if mint_record["headers"].get("Authorization") != "Bearer seed-oauth":
        failures.append(
            f"mint POST should carry seed access_token as Bearer; got "
            f"{mint_record['headers'].get('Authorization')}"
        )
    if mint_record["body"].get("min_ttl_seconds") != 1800:
        failures.append(
            f"mint POST should request 1800s min TTL; got {mint_record['body']}"
        )
    if not str(lm.kwargs["api_key"]).startswith("MINTED-AGENT-KEY"):
        failures.append(
            f"NousLM api_key should be the minted agent_key; got {lm.kwargs['api_key']}"
        )

    print(_summary("initial_mint"))
    return failures


def scenario_oauth_expiring_refreshes_then_mints(port: int) -> List[str]:
    """When OAuth is expiring AND no agent_key, expect refresh THEN mint
    (in that order), with the mint POST using the refreshed access_token.
    """
    failures = []
    _reset_recordings()
    lm = _make_lm(port=port, oauth_expires_at=time.time() + 30)

    paths = [r["path"] for r in _RecordingHandler.recorded]
    if not paths or not paths[0].endswith("/api/oauth/token"):
        failures.append(
            f"expected OAuth refresh first; got call sequence {paths}"
        )
    if len(paths) < 2 or not paths[1].endswith("/api/oauth/agent-key"):
        failures.append(
            f"expected mint as second call; got call sequence {paths}"
        )
    if len(_RecordingHandler.recorded) >= 2:
        mint_record = _RecordingHandler.recorded[1]
        # The refresh response in our mock returns access_token=REFRESHED-OAUTH;
        # the mint POST must use it as Bearer (proves the refresh-then-mint
        # ordering wires correctly).
        if mint_record["headers"].get("Authorization") != "Bearer REFRESHED-OAUTH":
            failures.append(
                f"mint POST should carry REFRESHED-OAUTH as Bearer; got "
                f"{mint_record['headers'].get('Authorization')}"
            )
    # Refresh response should also have rotated the refresh_token in shared state.
    if lm._shared_state.refresh_token != "REFRESHED-REFRESH":
        failures.append(
            f"rotated refresh_token should be persisted; got "
            f"{lm._shared_state.refresh_token}"
        )

    print(_summary("oauth_expiring_refreshes_then_mints"))
    return failures


def scenario_inference_uses_minted_agent_key(port: int) -> List[str]:
    """End-to-end: construct LM (mints), then make a real LiteLLM call.
    The inference POST's Authorization header must be the MINTED agent_key
    — proving we're not silently routing the OAuth access_token through
    as the inference Bearer (the bug this whole PR fixes).
    """
    failures = []
    _reset_recordings()
    lm = _make_lm(port=port)

    try:
        lm(messages=[{"role": "user", "content": "hello"}])
    except Exception as exc:
        failures.append(f"inference call raised unexpectedly: {type(exc).__name__}: {exc}")

    infer_records = [r for r in _RecordingHandler.recorded if "/chat/completions" in r["path"]]
    if not infer_records:
        failures.append("no inference call recorded")
    else:
        auth = infer_records[0]["headers"].get("Authorization", "")
        if not auth.startswith("Bearer MINTED-AGENT-KEY"):
            failures.append(
                f"inference Bearer should be the minted agent_key; got {auth}"
            )

    print(_summary("inference_uses_minted_agent_key"))
    return failures


def scenario_inference_401_triggers_remint_and_retry(port: int) -> List[str]:
    """Inference 401 (e.g., agent_key revoked mid-run) → force re-mint
    via NousLM's forward 401 handler, then retry the inference once.
    """
    failures = []
    _reset_recordings()
    _RecordingHandler.behavior["infer_first_status"] = 401
    lm = _make_lm(port=port)

    try:
        lm(messages=[{"role": "user", "content": "hello"}])
    except Exception as exc:
        failures.append(f"inference call raised after retry: {type(exc).__name__}: {exc}")

    infer_count = sum(
        1 for r in _RecordingHandler.recorded if "/chat/completions" in r["path"]
    )
    mint_count = sum(
        1 for r in _RecordingHandler.recorded if r["path"].endswith("/api/oauth/agent-key")
    )
    # Expect: 1 initial mint (constructor), 1 first inference (401), 1 force re-mint, 1 retry inference (200)
    if infer_count != 2:
        failures.append(f"expected 2 inference calls (1 fail + 1 retry); got {infer_count}")
    if mint_count != 2:
        failures.append(f"expected 2 mint calls (initial + force re-mint); got {mint_count}")

    print(_summary("inference_401_triggers_remint_and_retry"))
    return failures


def scenario_oauth_invalid_grant_surfaces_error(port: int) -> List[str]:
    """Refresh failure with invalid_grant must raise HermesProviderError
    pointing operator at `hermes model`.
    """
    from evolution.core.hermes_provider import HermesProviderError

    failures = []
    _reset_recordings()
    _RecordingHandler.behavior["refresh_status"] = 400
    _RecordingHandler.behavior["refresh_body"] = {
        "error": "invalid_grant",
        "error_description": "refresh token is no longer valid",
    }

    raised = None
    try:
        _make_lm(port=port, oauth_expires_at=time.time() + 30)
    except HermesProviderError as exc:
        raised = str(exc)

    if raised is None:
        failures.append("expected HermesProviderError; nothing raised")
    elif "hermes model" not in raised:
        failures.append(f"recovery hint missing from error: {raised}")

    print(_summary("oauth_invalid_grant_surfaces_error"))
    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    server = _start_server()
    port = _PORT_HOLDER["port"]
    print(f"Mock Nous Portal listening on http://127.0.0.1:{port}")

    # Suppress LiteLLM background telemetry chatter that pollutes the smoke
    # output without affecting wire-level behavior.
    litellm.suppress_debug_info = True

    all_failures: List[str] = []
    for scenario in (
        scenario_initial_mint,
        scenario_oauth_expiring_refreshes_then_mints,
        scenario_inference_uses_minted_agent_key,
        scenario_inference_401_triggers_remint_and_retry,
        scenario_oauth_invalid_grant_surfaces_error,
    ):
        failures = scenario(port)
        for f in failures:
            all_failures.append(f"{scenario.__name__}: {f}")

    server.shutdown()

    print("\n" + "=" * 60)
    if all_failures:
        print(f"FAIL: {len(all_failures)} assertion(s) failed:")
        for f in all_failures:
            print(f"  - {f}")
        return 1
    print("PASS: All assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

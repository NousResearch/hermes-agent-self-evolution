"""Hermes OpenAI-Codex OAuth integration helpers.

Self-evolution normally hands model strings to DSPy/LiteLLM. Hermes' Codex
OAuth backend is different: credentials live in Hermes' auth store and the
transport is the Codex Responses API wrapped by ``CodexAuxiliaryClient``.
This module keeps that integration in one place without exposing tokens.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

CODEX_MODEL_PREFIX = "openai-codex/"
CODEX_PROVIDER = "openai-codex"
DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def is_openai_codex_model(model: str | None) -> bool:
    """Return True for self-evolution's Hermes Codex OAuth model prefix."""
    return str(model or "").strip().startswith(CODEX_MODEL_PREFIX)


def codex_model_id(model: str) -> str:
    """Strip ``openai-codex/`` and return the Codex backend model id."""
    value = str(model or "").strip()
    if not value.startswith(CODEX_MODEL_PREFIX):
        raise ValueError(f"Not an OpenAI-Codex model string: {model!r}")
    bare = value[len(CODEX_MODEL_PREFIX) :].strip()
    if not bare:
        raise ValueError("OpenAI-Codex model string must include a model id after 'openai-codex/'")
    return bare


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_hermes_source_paths() -> list[Path]:
    """Likely Hermes Agent source checkouts, ordered by operator intent."""
    candidates: list[Path] = []
    for env_key in (
        "HERMES_AGENT_SOURCE_REPO",
        "HERMES_SOURCE_REPO",
        "HERMES_AGENT_CODE_REPO",
        # HERMES_AGENT_REPO is often the profile root in this project; keep it
        # as a candidate but validate it contains source before adding to sys.path.
        "HERMES_AGENT_REPO",
    ):
        value = os.getenv(env_key, "").strip()
        if value:
            candidates.append(Path(value).expanduser())

    home = Path.home()
    root = _repo_root()
    candidates.extend(
        [
            home / "hermes-agent",
            home / "dev" / "hermes-agent",
            home / ".hermes" / "hermes-agent",
            root.parent / "hermes-agent",
            root.parent.parent / "hermes-agent",
        ]
    )

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _looks_like_hermes_source(path: Path) -> bool:
    return (path / "agent" / "auxiliary_client.py").is_file() and (path / "hermes_cli" / "auth.py").is_file()


def ensure_hermes_source_importable() -> Path | None:
    """Ensure Hermes Agent source modules can be imported.

    Returns the source path added/found when discovery was needed, or None when
    modules were already importable. Raises ImportError with a non-secret setup
    hint when no checkout can be found.
    """
    try:
        importlib.import_module("agent.auxiliary_client")
        importlib.import_module("hermes_cli.auth")
        return None
    except ModuleNotFoundError:
        pass

    for candidate in _candidate_hermes_source_paths():
        if not _looks_like_hermes_source(candidate):
            continue
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
        try:
            importlib.import_module("agent.auxiliary_client")
            importlib.import_module("hermes_cli.auth")
            return candidate
        except ModuleNotFoundError:
            continue

    raise ImportError(
        "Could not import Hermes Agent Codex support. Set HERMES_AGENT_SOURCE_REPO "
        "to the hermes-agent source checkout containing agent/ and hermes_cli/."
    )


def _load_codex_dependencies() -> tuple[Callable[..., dict[str, Any]], Any, Any, Callable[[str], dict[str, str]]]:
    """Load Hermes' auth helper, OpenAI proxy, and CodexAuxiliaryClient lazily."""
    ensure_hermes_source_importable()
    auth_mod = importlib.import_module("hermes_cli.auth")
    aux_mod = importlib.import_module("agent.auxiliary_client")
    header_fn = getattr(aux_mod, "_codex_cloudflare_headers", lambda _token: {})
    return (
        auth_mod.resolve_codex_runtime_credentials,
        aux_mod.OpenAI,
        aux_mod.CodexAuxiliaryClient,
        header_fn,
    )


def codex_auth_available(*, refresh_if_expiring: bool = False) -> bool:
    """Return True when Hermes has a usable-looking Codex OAuth access token.

    This intentionally returns only a boolean; callers must not log credentials.
    ``refresh_if_expiring`` defaults to False so cron dry-runs can validate local
    configuration without forcing a token refresh network call.
    """
    try:
        resolve_codex_runtime_credentials, _, _, _ = _load_codex_dependencies()
        creds = resolve_codex_runtime_credentials(refresh_if_expiring=refresh_if_expiring)
    except Exception:
        return False
    return bool(str(creds.get("api_key") or "").strip())


def build_codex_auxiliary_client(model_id: str, *, refresh_if_expiring: bool = True) -> Any:
    """Build Hermes' CodexAuxiliaryClient for a bare Codex model id.

    The returned client exposes ``chat.completions.create`` for DSPy while the
    underlying transport uses Hermes' Codex OAuth credentials and Responses API
    adapter. No secret values are included in raised error messages.
    """
    bare_model = str(model_id or "").strip()
    if not bare_model:
        raise ValueError("A bare Codex model id is required")

    resolve_codex_runtime_credentials, OpenAI, CodexAuxiliaryClient, header_fn = _load_codex_dependencies()
    creds = resolve_codex_runtime_credentials(refresh_if_expiring=refresh_if_expiring)
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError(
            "No Hermes OpenAI-Codex OAuth credentials are available. "
            "Run `hermes model` and authenticate/select the OpenAI Codex provider."
        )

    base_url = str(creds.get("base_url") or "").strip().rstrip("/") or DEFAULT_CODEX_BASE_URL
    real_client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=header_fn(api_key),
    )
    return CodexAuxiliaryClient(real_client, bare_model)

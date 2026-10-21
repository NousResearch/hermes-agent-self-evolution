"""Shared DSPy construction for BYOK endpoints and account-aware Copilot models.

EVOLUTION_API_BASE/KEY configure custom endpoints. copilot/<id> uses
COPILOT_GITHUB_TOKEN and COPILOT_GH_HOST (or GH_HOST; default github.com).
Copilot account metadata selects the wire protocol for each model.
"""

import json
from functools import lru_cache
import os
import time
import urllib.parse
import urllib.request

import dspy

_COPILOT_HEADERS = {
    "editor-version": "vscode/1.104.1",
    "Copilot-Integration-Id": "vscode-chat",
    "User-Agent": "GitHubCopilotChat/0.26.7",
}
_copilot_cache: dict = {}


def _is_copilot_endpoint(base: str) -> bool:
    host = urllib.parse.urlsplit(base).hostname or ""
    return host.endswith(".githubcopilot.com") or (
        host.startswith("copilot-api.") and host.endswith(".ghe.com")
    )


def _validate_copilot_endpoint(base: str, host: str) -> str:
    url = urllib.parse.urlsplit(base)
    allowed = (url.hostname or "").endswith(".githubcopilot.com") or url.hostname == "copilot-api." + host
    if url.scheme != "https" or not allowed or url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}:
        raise ValueError("Invalid discovered Copilot endpoint")
    return base.rstrip("/")


def _read_json(url, token):
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token, **_COPILOT_HEADERS})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def _copilot_credentials() -> tuple[str, str]:
    host = os.environ.get("COPILOT_GH_HOST", os.environ.get("GH_HOST", "github.com")).lower()
    if host != "github.com" and not (host.endswith(".ghe.com") and urllib.parse.urlsplit("https://" + host).hostname == host and "/" not in host and ":" not in host):
        raise ValueError("Invalid Copilot GitHub host")
    gh_token = os.environ.get("COPILOT_GITHUB_TOKEN")
    if not gh_token:
        raise RuntimeError("copilot/<model> needs the COPILOT_GITHUB_TOKEN environment variable")
    source = (host, gh_token)
    c = _copilot_cache
    if c.get("source") == source and c["exp"] - 60 > time.time():
        return c["token"], c["base"]
    if host == "github.com":
        request = urllib.request.Request(
            "https://api.github.com/copilot_internal/v2/token",
            headers={"Authorization": "token " + gh_token, **_COPILOT_HEADERS},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.load(response)
        token = data["token"]
    else:
        data = _read_json("https://api." + host + "/copilot_internal/user", gh_token)
        token = gh_token
    base = _validate_copilot_endpoint(data["endpoints"]["api"], host)
    c.update(source=source, token=token, base=base, exp=data.get("expires_at", time.time() + 1500))
    return token, base


@lru_cache(maxsize=1)
def _copilot_models(base, token):
    return _read_json(base.rstrip("/") + "/models", token)["data"]


def _copilot_model(base, token, model):
    for item in _copilot_models(base, token):
        if item["id"] == model and item.get("policy", {}).get("state") != "disabled":
            return item
    raise ValueError(f"Copilot model is not advertised or enabled: {model}")


def _copilot_options(model, base, token, kwargs):
    metadata = _copilot_model(base, token, model)
    endpoints = metadata.get("supported_endpoints") or ["/chat/completions"]
    headers = {**kwargs.get("extra_headers", {}), **_COPILOT_HEADERS, "Authorization": "Bearer " + token}
    kwargs.update(api_key=token, api_base=base, extra_headers=headers)
    if model.startswith("claude-") and "/v1/messages" in endpoints:
        if kwargs.get("max_tokens") is None:
            kwargs["max_tokens"] = 4096
        return "anthropic/" + model, kwargs
    if "/responses" in endpoints and ("/chat/completions" not in endpoints or model.startswith(("gpt-5", "gpt-6"))):
        kwargs.update(model_type="responses", store=False)
        kwargs.setdefault("reasoning_effort", "low")
    elif "/chat/completions" not in endpoints:
        raise ValueError(f"No supported Copilot transport for {model}")
    return "openai/" + model, kwargs


class _CopilotLM(dspy.LM):
    """Refresh auth and preserve structured outputs on Copilot's native APIs."""

    def __init__(self, *args, credentials=None, **kwargs):
        self._credentials = credentials or _copilot_credentials
        super().__init__(*args, **kwargs)

    def _request_kwargs(self, kwargs):
        token, base = self._credentials()
        if self.model.startswith("anthropic/"):
            format_ = kwargs.get("response_format")
            schema = format_.model_json_schema() if isinstance(format_, type) and hasattr(format_, "model_json_schema") else None
            if isinstance(format_, dict) and format_.get("type") == "json_schema":
                schema = format_["json_schema"]["schema"]
            if schema is not None:
                # LiteLLM's legacy response_format mapping emits deprecated output_format.
                # Carry the same schema through the supported Anthropic field instead.
                kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
                kwargs["output_config"] = {
                    **kwargs.get("output_config", {}), "format": {"type": "json_schema", "schema": schema}
                }
        headers = {**self.kwargs.get("extra_headers", {}), **kwargs.get("extra_headers", {}), **_COPILOT_HEADERS, "Authorization": "Bearer " + token}
        return {**kwargs, "api_key": token, "api_base": base, "extra_headers": headers}

    def __call__(self, prompt=None, *, messages=None, **kwargs):
        return super().__call__(prompt, messages=messages, **self._request_kwargs(kwargs))

    async def acall(self, prompt=None, *, messages=None, **kwargs):
        return await super().acall(prompt, messages=messages, **self._request_kwargs(kwargs))

    def forward(self, prompt=None, messages=None, **kwargs):
        return super().forward(prompt, messages=messages, **self._request_kwargs(kwargs))

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return await super().aforward(prompt, messages=messages, **self._request_kwargs(kwargs))


def make_lm(model: str, **kwargs) -> dspy.LM:
    if model.startswith("copilot/"):
        token, base = _copilot_credentials()
        model, kwargs = _copilot_options(model.split("/", 1)[1], base, token, kwargs)
        return _CopilotLM(model, **kwargs)
    if os.environ.get("EVOLUTION_API_BASE"):
        kwargs.setdefault("api_base", os.environ["EVOLUTION_API_BASE"])
    if os.environ.get("EVOLUTION_API_KEY"):
        kwargs.setdefault("api_key", os.environ["EVOLUTION_API_KEY"])
    if _is_copilot_endpoint(kwargs.get("api_base", "")):
        base = kwargs["api_base"].rstrip("/")
        if not base.startswith("https://"):
            raise ValueError("Copilot endpoint requires HTTPS")
        token = kwargs.get("api_key")
        if not token:
            raise RuntimeError("Copilot endpoint requires EVOLUTION_API_KEY or api_key")
        model, kwargs = _copilot_options(model.split("/", 1)[-1], base, token, kwargs)
        return _CopilotLM(model, credentials=lambda: (token, base), **kwargs)
    return dspy.LM(model, **kwargs)

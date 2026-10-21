"""Exercise MiniMax routing through DSPy and the SDK's serialized HTTP request."""

import json
from pathlib import Path

import dspy
import httpx
import pytest

from evolution.core.config import EvolutionConfig, MINIMAX_MODELS


_TEST_KEY = "offline-test-key"


@pytest.mark.parametrize("base_url", [
    "https://api.minimax.io/v1",
    "https://api.minimax.cn/v1",
    "https://custom.invalid/v1",
])
@pytest.mark.parametrize("model", [
    f"minimax/{MINIMAX_MODELS[0]}",
    "MiniMax-M3",
    "openai/MiniMax-M2.7",
    "MiniMax-M2.7-highspeed",
    "minimax/custom-model",
])
def test_serialized_chat_request(monkeypatch, base_url, model):
    requests = []

    def send(client, request, **kwargs):
        requests.append(request)
        return httpx.Response(200, request=request, json={
            "id": "offline-response", "object": "chat.completion", "created": 0,
            "model": model.split("/")[-1],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    monkeypatch.setattr(httpx.Client, "send", send)
    config = EvolutionConfig(
        hermes_agent_path=Path("."), minimax_api_key=_TEST_KEY,
        minimax_base_url=base_url,
    )
    lm = config.make_lm(model)
    lm.cache = False
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "https://image.invalid/test.png"}},
    ]}]
    with dspy.context(lm=lm):
        assert lm(messages=messages) == ["OK"]
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == f"{base_url}/chat/completions"
    assert request.headers["authorization"] == "Bearer offline-test-key"
    body = json.loads(request.content)
    assert body["model"] == model.split("/")[-1]
    assert body["messages"] == messages
    assert body["temperature"] == 1.0
    for unsupported in ("service_tier", "reasoning_split", "thinking"):
        assert unsupported not in body
    assert body.get("reasoning_effort") != "none"

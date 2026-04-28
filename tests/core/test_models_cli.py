"""Tests for model comparison CLI commands."""

from __future__ import annotations

import json

from click.testing import CliRunner

from evolution import cli as cli_module
from evolution.cli import main


def test_cli_models_compare_requires_explicit_prompt():
    result = CliRunner().invoke(
        main,
        ["models", "compare", "--model", "deepseek-v4-flash"],
    )

    assert result.exit_code != 0
    assert "Provide --prompt or --prompt-file" in result.output


def test_cli_models_compare_emits_json_for_supplied_models(monkeypatch):
    def fake_compare(**kwargs):
        assert kwargs["provider"] == "deepseek"
        assert kwargs["models"] == ["deepseek-v4-flash", "deepseek-v4-pro"]
        assert kwargs["prompt"] == "classify this comment"
        assert kwargs["max_tokens"] == 128
        return [
            {
                "model": "deepseek-v4-flash",
                "ok": True,
                "latency_ms": 420,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "output_text": '{"tag":"pricing"}',
                "error": None,
            },
            {
                "model": "deepseek-v4-pro",
                "ok": True,
                "latency_ms": 940,
                "prompt_tokens": 10,
                "completion_tokens": 32,
                "total_tokens": 42,
                "output_text": '{"tag":"pricing","why":"specific"}',
                "error": None,
            },
        ]

    monkeypatch.setattr(cli_module, "compare_chat_models", fake_compare)

    result = CliRunner().invoke(
        main,
        [
            "models",
            "compare",
            "--provider",
            "deepseek",
            "--model",
            "deepseek-v4-flash",
            "--model",
            "deepseek-v4-pro",
            "--prompt",
            "classify this comment",
            "--max-tokens",
            "128",
            "--json-output",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["provider"] == "deepseek"
    assert [row["model"] for row in payload["results"]] == ["deepseek-v4-flash", "deepseek-v4-pro"]


def test_cli_models_compare_prints_table_without_api_key(monkeypatch):
    def fake_compare(**kwargs):
        return [
            {
                "model": "deepseek-v4-flash",
                "ok": True,
                "latency_ms": 111,
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "output_text": "flash output",
                "error": None,
            },
            {
                "model": "deepseek-v4-pro",
                "ok": False,
                "latency_ms": 222,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "output_text": "",
                "error": "rate limited",
            },
        ]

    monkeypatch.setattr(cli_module, "compare_chat_models", fake_compare)

    result = CliRunner().invoke(
        main,
        [
            "models",
            "compare",
            "--model",
            "deepseek-v4-flash",
            "--model",
            "deepseek-v4-pro",
            "--prompt",
            "tag this",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "deepseek-v4-flash ok latency_ms=111 tokens=3" in result.output
    assert "deepseek-v4-pro error latency_ms=222 tokens=0" in result.output
    assert "rate limited" in result.output
    assert "API_KEY" not in result.output

"""Tests for AWS Bedrock resolution in the Hermes-aware LM resolver.

Bedrock differs from every other provider: no inline ``api_key``, no env-var
key, no ``auth.json`` credential pool. Auth flows through boto3's default
chain (``AWS_BEARER_TOKEN_BEDROCK``, ``AWS_ACCESS_KEY_ID``,
``AWS_PROFILE``, IAM role, IMDS), and LiteLLM's ``bedrock/<model>`` provider
auto-resolves it. The framework only surfaces the region kwarg + optional
profile name from ``~/.hermes/config.yaml``.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import litellm
import pytest

from evolution.core.auth_check import (
    _HERMES_AUTH_COMMAND_BY_PROVIDER,
    is_auth_error,
)
from evolution.core.hermes_provider import (
    _BEDROCK_DEFAULT_MODEL,
    _BEDROCK_DEFAULT_REGION,
    HermesProviderError,
    ResolvedLM,
    resolve_default_lm,
)


@pytest.fixture
def hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Empty ~/.hermes-equivalent dir, with provider AND AWS env vars cleared.

    The AWS-side clearing matters: the resolver consults ``AWS_REGION`` /
    ``AWS_DEFAULT_REGION`` for region fallback, and a leaky local env would
    silently override the test's expected behavior.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return home


def _write_config(home: Path, body: str) -> None:
    (home / "config.yaml").write_text(textwrap.dedent(body).lstrip())


# ---------------------------------------------------------------------------
# Provider mapping: explicit `provider: bedrock`
# ---------------------------------------------------------------------------


class TestBedrockProviderMapping:
    def test_explicit_bedrock_with_model_default(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              default: us.anthropic.claude-sonnet-4-6
              provider: bedrock
            bedrock:
              region: us-east-2
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "bedrock/us.anthropic.claude-sonnet-4-6"
        assert lm.lm_kwargs == {"aws_region_name": "us-east-2"}
        assert "bedrock(region=us-east-2)" in lm.source

    def test_cross_region_inference_profile_passes_through(self, hermes_home):
        # us./apac./eu. inference-profile prefixes are part of the model ID
        # and must reach LiteLLM intact — they're how Bedrock routes to
        # cross-region pools.
        _write_config(
            hermes_home,
            """
            model:
              default: apac.anthropic.claude-haiku-4-5
              provider: bedrock
            bedrock:
              region: ap-southeast-2
            """,
        )
        lm = resolve_default_lm(role="eval", hermes_home=hermes_home)
        assert lm.model == "bedrock/apac.anthropic.claude-haiku-4-5"
        assert lm.lm_kwargs["aws_region_name"] == "ap-southeast-2"

    def test_aws_profile_name_flows_through(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              default: anthropic.claude-3-5-sonnet-20241022-v2:0
              provider: bedrock
            bedrock:
              region: us-west-2
              aws_profile_name: my-bedrock-profile
            """,
        )
        lm = resolve_default_lm(role="judge", hermes_home=hermes_home)
        assert lm.lm_kwargs == {
            "aws_region_name": "us-west-2",
            "aws_profile_name": "my-bedrock-profile",
        }

    def test_no_api_key_emitted(self, hermes_home):
        # Bedrock auth comes from boto3's chain. The resolver must not pack
        # any api_key kwarg — LiteLLM would then think it's an explicit
        # OpenAI-style key and skip boto3 resolution.
        _write_config(
            hermes_home,
            """
            model:
              default: us.anthropic.claude-sonnet-4-6
              provider: bedrock
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert "api_key" not in lm.lm_kwargs


# ---------------------------------------------------------------------------
# Region resolution chain
# ---------------------------------------------------------------------------


class TestBedrockRegionResolution:
    def test_config_region_takes_precedence_over_env(self, hermes_home, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-1")
        _write_config(
            hermes_home,
            """
            model:
              default: us.anthropic.claude-sonnet-4-6
              provider: bedrock
            bedrock:
              region: eu-west-1
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["aws_region_name"] == "eu-west-1"

    def test_aws_region_env_used_when_config_missing(self, hermes_home, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        _write_config(
            hermes_home,
            """
            model:
              default: us.anthropic.claude-sonnet-4-6
              provider: bedrock
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["aws_region_name"] == "us-east-1"

    def test_aws_default_region_fallback(self, hermes_home, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        _write_config(
            hermes_home,
            """
            model:
              default: us.anthropic.claude-sonnet-4-6
              provider: bedrock
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["aws_region_name"] == "us-west-2"

    def test_hardcoded_default_when_nothing_set(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              default: us.anthropic.claude-sonnet-4-6
              provider: bedrock
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.lm_kwargs["aws_region_name"] == _BEDROCK_DEFAULT_REGION


# ---------------------------------------------------------------------------
# Standalone fallback model
# ---------------------------------------------------------------------------


class TestBedrockStandaloneDefault:
    def test_standalone_default_model_when_unset(self, hermes_home):
        _write_config(
            hermes_home,
            """
            model:
              provider: bedrock
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == f"bedrock/{_BEDROCK_DEFAULT_MODEL}"


# ---------------------------------------------------------------------------
# Credential-pool isolation: Bedrock must NOT touch auth.json
# ---------------------------------------------------------------------------


class TestBedrockSkipsCredentialPool:
    def test_no_pool_lookup_attempted(self, hermes_home):
        # Stuff auth.json with a bogus "bedrock" pool entry that, if read,
        # would surface as an api_key in lm_kwargs. The resolver must
        # silently ignore it (boto3 owns Bedrock auth, not us).
        (hermes_home / "auth.json").write_text(
            json.dumps(
                {
                    "credential_pool": {
                        "bedrock": [
                            {
                                "access_token": "this-should-not-appear",
                                "priority": 0,
                            }
                        ]
                    }
                }
            )
        )
        _write_config(
            hermes_home,
            """
            model:
              default: us.anthropic.claude-sonnet-4-6
              provider: bedrock
            bedrock:
              region: us-east-1
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert "api_key" not in lm.lm_kwargs
        assert "this-should-not-appear" not in str(lm.lm_kwargs.values())


# ---------------------------------------------------------------------------
# Provider aliases (aws, aws-bedrock, amazon, amazon-bedrock)
# ---------------------------------------------------------------------------


class TestBedrockAliases:
    @pytest.mark.parametrize(
        "alias", ["aws", "aws-bedrock", "amazon", "amazon-bedrock"]
    )
    def test_alias_resolves_to_bedrock(self, hermes_home, alias):
        _write_config(
            hermes_home,
            f"""
            model:
              default: us.anthropic.claude-sonnet-4-6
              provider: {alias}
            bedrock:
              region: us-east-1
            """,
        )
        lm = resolve_default_lm(role="optimizer", hermes_home=hermes_home)
        assert lm.model == "bedrock/us.anthropic.claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Auto-detection guard: provider: auto must NOT pick Bedrock
# ---------------------------------------------------------------------------


class TestBedrockNotAutoDetected:
    def test_auto_detect_does_not_pick_bedrock_with_aws_env(
        self, hermes_home, monkeypatch
    ):
        # Even with AWS env vars set, provider: auto should never silently
        # route to Bedrock — AWS_PROFILE / AWS_ACCESS_KEY_ID are commonly
        # set for non-Bedrock reasons (S3, DynamoDB, etc.).
        monkeypatch.setenv("AWS_PROFILE", "my-aws-profile")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        # No other provider configured → resolver should raise the
        # "no model could be resolved" error, not pick Bedrock.
        with pytest.raises(HermesProviderError, match="No model could be resolved"):
            resolve_default_lm(role="optimizer", hermes_home=hermes_home)


# ---------------------------------------------------------------------------
# Auth-error classification: NoCredentialsError → is_auth_error
# ---------------------------------------------------------------------------


class TestBedrockAuthErrorClassification:
    def test_no_credentials_error_classified_as_auth(self):
        # botocore.exceptions.NoCredentialsError surfaces this exact message
        # through LiteLLM's bedrock provider. The string-pattern matcher
        # must catch it so preflight surfaces a Rich panel + recovery hint
        # rather than a Python traceback.
        exc = litellm.AuthenticationError(
            message="Unable to locate credentials. You can configure credentials by running 'aws configure'.",
            llm_provider="bedrock",
            model="bedrock/us.anthropic.claude-sonnet-4-6",
        )
        assert is_auth_error(exc) is True

    def test_recovery_message_present_for_bedrock(self):
        # The user-facing panel must include actionable AWS-credential
        # guidance — env var, profile, or instance role. None of those are
        # `hermes auth add`.
        cmd = _HERMES_AUTH_COMMAND_BY_PROVIDER.get("bedrock", "")
        assert "AWS_PROFILE" in cmd or "AWS_BEARER_TOKEN_BEDROCK" in cmd

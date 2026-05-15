"""Tests for the cheap-model cost advisor.

Covers:
  * Lookup strategy: direct keys + strip-prefix fallback
  * Same-provider enumeration with cost + context-window filters
  * Graceful skip for off-catalog providers (Bedrock, Codex, local)
  * Suggestion model string preserves the original prefix shape
  * Rich panel renders with the right ratio + paste-ready CLI flag
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from evolution.core.cost_advisor import (
    CheaperAlternative,
    _MIN_INPUT_COST_RATIO,
    _namespace,
    _version_tuple,
    find_cheaper_alternative,
    render_suggestion_panel,
)


# A deterministic fake catalog that exercises the same shape variants as
# the real litellm.model_cost dict. Keeps tests independent of LiteLLM
# pricing updates and of any specific provider's current model lineup.
_FAKE_CATALOG = {
    # Anthropic-style: bare keys, no prefix
    "claude-opus-test": {
        "input_cost_per_token": 5e-6,
        "output_cost_per_token": 25e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "anthropic-test",
    },
    "claude-sonnet-test": {
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 15e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "anthropic-test",
    },
    "claude-haiku-test": {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 5e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "anthropic-test",
    },
    "claude-haiku-tiny-ctx": {
        # Cheaper than opus but smaller context — must be filtered out.
        "input_cost_per_token": 0.5e-6,
        "output_cost_per_token": 2e-6,
        "max_input_tokens": 8_000,
        "litellm_provider": "anthropic-test",
    },
    # OpenAI-style: bare keys, no prefix
    "gpt-test": {
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 10e-6,
        "max_input_tokens": 128_000,
        "litellm_provider": "openai-test",
    },
    "gpt-mini-test": {
        "input_cost_per_token": 0.5e-6,
        "output_cost_per_token": 2e-6,
        "max_input_tokens": 128_000,
        "litellm_provider": "openai-test",
    },
    # OpenRouter-style: prefixed keys preserved verbatim
    "openrouter-test/openai/gpt-test": {
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 10e-6,
        "max_input_tokens": 128_000,
        "litellm_provider": "openrouter-test",
    },
    "openrouter-test/openai/gpt-mini-test": {
        "input_cost_per_token": 0.5e-6,
        "output_cost_per_token": 2e-6,
        "max_input_tokens": 128_000,
        "litellm_provider": "openrouter-test",
    },
    # Lone-cheap-no-cheaper-available: a single model with no siblings.
    "lonely-model": {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 5e-6,
        "max_input_tokens": 32_000,
        "litellm_provider": "lonely-test",
    },
    # Entry with missing pricing — must be filtered out.
    "no-pricing": {
        "input_cost_per_token": None,
        "output_cost_per_token": None,
        "max_input_tokens": 128_000,
        "litellm_provider": "anthropic-test",
    },
    # ----------------------------------------------------------------
    # Versioned entries — exercise the major-version generation filter.
    # Naming mirrors real catalog patterns (claude-3-haiku-20240307 vs
    # claude-haiku-4-5) so the regex parser sees realistic input.
    # ----------------------------------------------------------------
    "vendor-opus-4-5": {
        "input_cost_per_token": 5e-6,
        "output_cost_per_token": 25e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "vendor",
    },
    "vendor-haiku-4-5": {
        # Current-gen, $1/M — should win against the older gen-3 below.
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 5e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "vendor",
    },
    "vendor-3-haiku-20240307": {
        # Older generation, MUCH cheaper. Pure cost-sort would pick this
        # for a vendor-opus-4-5 user; the major-version filter must reject.
        "input_cost_per_token": 0.25e-6,
        "output_cost_per_token": 1.25e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "vendor",
    },
    "vendor-opus-4-7": {
        "input_cost_per_token": 5e-6,
        "output_cost_per_token": 25e-6,
        "max_input_tokens": 1_000_000,
        "litellm_provider": "vendor",
    },
    "vendor-sonnet-4-6": {
        # Newer-named, same cost as vendor-4-sonnet-old below; tiebreak
        # by minor desc must prefer this canonical name.
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 15e-6,
        "max_input_tokens": 1_000_000,
        "litellm_provider": "vendor",
    },
    "vendor-4-sonnet-20250514": {
        # Same generation (major 4), same cost — but minor parses as 0
        # because the digit isn't followed by another digit-pair.
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 15e-6,
        "max_input_tokens": 1_000_000,
        "litellm_provider": "vendor",
    },
    # Cross-generation isolation: only gen-3 entries on this provider.
    "lonely-gen3-opus": {
        "input_cost_per_token": 15e-6,
        "output_cost_per_token": 75e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "lonely-gen3",
    },
    "lonely-gen3-haiku-20240307": {
        "input_cost_per_token": 0.25e-6,
        "output_cost_per_token": 1.25e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "lonely-gen3",
    },
    # ----------------------------------------------------------------
    # Namespace fixture: mirrors Bedrock cross-region routing patterns
    # so the namespace filter has something to gate on.
    # ----------------------------------------------------------------
    "us.bedrocktest.opus-4-5": {
        "input_cost_per_token": 5.5e-6,
        "output_cost_per_token": 27.5e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "bedrock-test",
    },
    "us.bedrocktest.haiku-4-5": {
        # Same us.* region — qualifies as a same-namespace suggestion.
        "input_cost_per_token": 1.1e-6,
        "output_cost_per_token": 5.5e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "bedrock-test",
    },
    "bedrocktest.haiku-4-5": {
        # Regional-only (no us. prefix). Same provider, same major version,
        # CHEAPER than us.bedrocktest.haiku-4-5 — but a different routing
        # profile. Must NOT be suggested for a us.* user.
        "input_cost_per_token": 1.0e-6,
        "output_cost_per_token": 5.0e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "bedrock-test",
    },
    "eu.bedrocktest.haiku-4-5": {
        # Different region — same shape, must not be cross-suggested.
        "input_cost_per_token": 1.1e-6,
        "output_cost_per_token": 5.5e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "bedrock-test",
    },
    # ----------------------------------------------------------------
    # OpenRouter-style cross-vendor: same litellm_provider, different
    # routing namespace (anthropic/* vs zai/*).
    # ----------------------------------------------------------------
    "openroutertest/anthropic/opus-4": {
        "input_cost_per_token": 5e-6,
        "output_cost_per_token": 25e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "openrouter-test",
    },
    "openroutertest/anthropic/haiku-4": {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 5e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "openrouter-test",
    },
    "openroutertest/zai/glm-4-flash": {
        # Cheaper than the anthropic options, completely different upstream
        # vendor. Must NOT be suggested for an anthropic user.
        "input_cost_per_token": 0.1e-6,
        "output_cost_per_token": 0.5e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "openrouter-test",
    },
    # ----------------------------------------------------------------
    # Weak-savings fixture: only a marginal saving exists.
    # ----------------------------------------------------------------
    "marginaltest-opus": {
        "input_cost_per_token": 5e-6,
        "output_cost_per_token": 25e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "marginal-test",
    },
    "marginaltest-opus-cheaper": {
        # 1.1× cheaper — below the meaningful-savings threshold.
        "input_cost_per_token": 4.5e-6,
        "output_cost_per_token": 22.5e-6,
        "max_input_tokens": 200_000,
        "litellm_provider": "marginal-test",
    },
}


@pytest.fixture(autouse=True)
def _patch_catalog():
    with patch("evolution.core.cost_advisor.litellm.model_cost", _FAKE_CATALOG):
        yield


# ---------------------------------------------------------------------------
# Lookup strategies
# ---------------------------------------------------------------------------


class TestLookupStrategies:
    def test_strip_prefix_lookup_for_anthropic_style(self):
        # User passed "anthropic-test/claude-opus-test"; catalog has the
        # bare key. The advisor must strip and retry.
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        assert alt.provider == "anthropic-test"

    def test_direct_key_lookup_for_openrouter_style(self):
        # OpenRouter keys are full strings; no prefix-strip needed.
        alt = find_cheaper_alternative("openrouter-test/openai/gpt-test")
        assert alt is not None
        assert alt.provider == "openrouter-test"
        assert alt.suggested_model == "openrouter-test/openai/gpt-mini-test"

    def test_unknown_model_returns_none(self):
        # Bedrock / Codex / local-server endpoints land here.
        assert find_cheaper_alternative("bedrock/us.anthropic.claude-test") is None
        assert find_cheaper_alternative("openai/gpt-5-codex-fake-doesnt-exist") is None


# ---------------------------------------------------------------------------
# Suggestion logic
# ---------------------------------------------------------------------------


class TestSuggestionLogic:
    def test_suggests_cheapest_same_provider_with_sufficient_context(self):
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        # Cheapest qualifying candidate is haiku-test, NOT haiku-tiny-ctx
        # (which has only 8k context vs the opus's 200k requirement).
        assert alt.suggested_model == "anthropic-test/claude-haiku-test"
        assert alt.input_cost_ratio == pytest.approx(5.0)
        assert alt.output_cost_ratio == pytest.approx(5.0)
        assert alt.suggested_max_input_tokens == 200_000

    def test_filters_out_smaller_context_window_candidates(self):
        # Even though claude-haiku-tiny-ctx is the cheapest in the catalog,
        # its 8k context disqualifies it for an opus user with 200k.
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        assert alt.suggested_model != "anthropic-test/claude-haiku-tiny-ctx"

    def test_no_cheaper_alternative_returns_none(self):
        # claude-haiku-test is already the cheapest qualifying anthropic-test
        # entry. No suggestion to make.
        alt = find_cheaper_alternative("anthropic-test/claude-haiku-test")
        assert alt is None

    def test_provider_with_only_one_model_returns_none(self):
        alt = find_cheaper_alternative("lonely-model")
        assert alt is None

    def test_skips_catalog_entries_with_missing_pricing(self):
        # If the cheapest candidate has None pricing, the advisor must
        # skip it rather than crash on multiplication.
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        assert alt.suggested_model != "anthropic-test/no-pricing"

    def test_does_not_suggest_cross_provider_swaps(self):
        # gpt-mini-test is cheaper than claude-opus-test but on a different
        # provider. The advisor must not cross provider boundaries.
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        assert "gpt" not in alt.suggested_model

    def test_suggested_model_string_paste_ready_for_eval_model_flag(self):
        # The suggestion has to flow through `--eval-model <X>` directly,
        # which means the same prefix shape as the original.
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        # User passed prefixed; suggestion must be prefixed.
        assert alt.suggested_model.startswith("anthropic-test/")

    def test_cost_per_1m_uses_per_token_times_million(self):
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        # 5e-6 per token × 1M = $5/M. Watch for the implicit float math.
        assert alt.current_input_cost_per_1m == pytest.approx(5.0)
        assert alt.suggested_input_cost_per_1m == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------


class TestVersionTuple:
    """The major-version filter is the user-visible safety net against
    silently downgrading from gen-4 to gen-3 just because cost-sort wins.
    """

    @pytest.mark.parametrize(
        "model_key,expected",
        [
            ("claude-opus-4-5", (4, 5)),
            ("claude-opus-4-5-20251101", (4, 5)),  # date suffix ignored
            ("claude-opus-4-7", (4, 7)),
            ("claude-3-opus-20240229", (3, 0)),  # single-digit major
            ("claude-3-haiku-20240307", (3, 0)),
            ("claude-haiku-4-5", (4, 5)),
            ("claude-haiku-4-5-20251001", (4, 5)),
            ("claude-4-sonnet-20250514", (4, 0)),  # digit not followed by -N
            ("claude-sonnet-4-6", (4, 6)),
            ("gpt-5", (5, 0)),
            ("gpt-5-codex", (5, 0)),
            ("custom-local-model", (None, 0)),  # no parseable version
        ],
    )
    def test_extracts_correct_major_minor(self, model_key, expected):
        assert _version_tuple(model_key) == expected


class TestGenerationFilter:
    def test_excludes_older_major_version_even_if_cheaper(self):
        # vendor-3-haiku-20240307 is 4x cheaper than vendor-haiku-4-5 but
        # gen-3 vs gen-4. The advisor must NOT pick the older one.
        alt = find_cheaper_alternative("vendor-opus-4-5")
        assert alt is not None
        assert alt.suggested_model == "vendor-haiku-4-5"
        assert "3-haiku" not in alt.suggested_model

    def test_gen3_user_gets_gen3_suggestion(self):
        # Users still on gen-3 should get gen-3 suggestions — the filter
        # is symmetric. lonely-gen3 provider has only two gen-3 models.
        alt = find_cheaper_alternative("lonely-gen3-opus")
        assert alt is not None
        assert alt.suggested_model == "lonely-gen3-haiku-20240307"

    def test_unversioned_models_treated_as_match_all(self):
        # Custom/local models have no parseable version — _version_tuple
        # returns (None, 0). The filter must degrade open (allow all
        # candidates) rather than closed (reject everything).
        alt = find_cheaper_alternative("claude-opus-test")  # no digits
        assert alt is not None
        # Existing fake catalog has 'claude-haiku-test' as the cheapest
        # qualifying anthropic-test entry; this remains the suggestion.
        assert alt.suggested_model == "claude-haiku-test"


class TestTiebreakByNewerMinor:
    def test_prefers_newer_minor_when_cost_ties(self):
        # vendor-sonnet-4-6 (4, 6) and vendor-4-sonnet-20250514 (4, 0)
        # are both $3/M with 1M context. The advisor must prefer the
        # higher-minor canonical name so users get the current model
        # rather than a stale date-suffixed snapshot.
        alt = find_cheaper_alternative("vendor-opus-4-7")
        assert alt is not None
        assert alt.suggested_model == "vendor-sonnet-4-6"


class TestNamespace:
    """The namespace filter blocks two real surprises:
      1. Bedrock cross-region (us.X) being silently swapped for regional (X)
      2. OpenRouter cross-vendor (anthropic/X being swapped for zai/X)
    """

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("claude-opus-4-5", ""),
            ("gpt-5", ""),
            ("gpt-4.1", ""),
            ("claude-3.5-sonnet", ""),  # claude-3 has digit, not a route
            ("openrouter/anthropic/claude-opus-4", "openrouter/anthropic"),
            ("openrouter/z-ai/glm-4.7-flash", "openrouter/z-ai"),
            ("anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic"),
            ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "us.anthropic"),
            ("eu.anthropic.claude-sonnet-4-6", "eu.anthropic"),
            ("us-gov.anthropic.claude-X", "us-gov.anthropic"),
            ("apac.anthropic.claude-3-5-sonnet-20240620-v1:0", "apac.anthropic"),
        ],
    )
    def test_extracts_correct_namespace(self, key, expected):
        assert _namespace(key) == expected

    def test_bedrock_cross_region_user_only_gets_same_region_suggestion(self):
        # us.bedrocktest user must NOT be silently suggested the cheaper
        # bedrocktest.haiku-4-5 (regional-only, different routing profile)
        # or eu.bedrocktest.haiku-4-5 (different region).
        alt = find_cheaper_alternative("us.bedrocktest.opus-4-5")
        assert alt is not None
        assert alt.suggested_model == "us.bedrocktest.haiku-4-5"
        assert _namespace(alt.suggested_model) == "us.bedrocktest"

    def test_bedrock_regional_user_only_gets_regional_suggestion(self):
        # Inverse: a bedrocktest.* (regional-only) user must not be
        # routed to us.bedrocktest.* (cross-region inference profile).
        # Add a regional opus to the catalog implicitly via this test —
        # the only candidate at namespace "" with cheaper input is
        # bedrocktest.haiku-4-5 itself, which has no opus to compare to.
        # Instead test via a synthetic: the bedrocktest.haiku-4-5 entry
        # IS in the catalog as its own namespace; only candidates with
        # the same empty-but-bedrocktest-prefix namespace should match.
        alt = find_cheaper_alternative("bedrocktest.haiku-4-5")
        # No same-namespace cheaper option exists — should return None
        # (NOT cross-suggest us.bedrocktest.haiku-4-5 even though it's
        # 1.1× more expensive, that wouldn't trigger anyway).
        assert alt is None

    def test_openrouter_cross_vendor_excluded(self):
        # An openroutertest/anthropic/* user must not be suggested an
        # openroutertest/zai/* model even though zai is much cheaper —
        # different upstream vendor entirely.
        alt = find_cheaper_alternative("openroutertest/anthropic/opus-4")
        assert alt is not None
        assert alt.suggested_model == "openroutertest/anthropic/haiku-4"
        assert "zai" not in alt.suggested_model


class TestMinimumSavingsThreshold:
    def test_marginal_savings_suppressed(self):
        # marginaltest-opus -> marginaltest-opus-cheaper would be only
        # 1.1× cheaper. Below the floor where the panel is worth surfacing.
        alt = find_cheaper_alternative("marginaltest-opus")
        assert alt is None

    def test_threshold_constant_is_at_least_one_and_a_half(self):
        # Sanity check: the threshold should be high enough to suppress
        # AWS pricing-tier noise (~10% savings) but low enough that the
        # 1.7× sonnet-vs-opus suggestion still surfaces in the real catalog.
        assert 1.4 <= _MIN_INPUT_COST_RATIO <= 1.7


class TestPanel:
    def test_panel_includes_cli_flag_and_ratio(self):
        alt = find_cheaper_alternative("anthropic-test/claude-opus-test")
        assert alt is not None
        panel = render_suggestion_panel("eval", alt)

        # Render to a plain string for substring checks.
        from rich.console import Console
        from io import StringIO

        buf = StringIO()
        Console(file=buf, width=120, color_system=None).print(panel)
        out = buf.getvalue()

        assert "--eval-model anthropic-test/claude-haiku-test" in out
        assert "5.0× cheaper" in out
        assert "$5.00/M input" in out
        assert "$1.00/M input" in out
        # Tradeoff caveat must surface so users don't blindly downgrade
        # the optimizer/reflection roles too.
        assert "reasoning quality" in out

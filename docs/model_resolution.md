# Model resolution

How `agent-self-evolution` decides which model to call for each LM role.

## TL;DR

If you have Hermes Agent configured (`~/.hermes/config.yaml` exists), the framework uses your Hermes-configured model and provider automatically — for the optimizer, reflection, eval, and judge roles. No env vars to set.

If you don't have Hermes, set any standard provider env var (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, etc.) and it works.

If neither, the framework exits with a message listing what was tried and how to fix it.

## Resolution order

For each role (`optimizer`, `reflection`, `eval`, `judge`), the resolver walks this chain top-to-bottom and stops at the first usable result:

1. **Explicit CLI override** — `--optimizer-model`, `--reflection-model`, `--eval-model` on the command line. The string is passed straight through to LiteLLM; no `api_base` or `api_key` is inferred (you rely on env vars for credentials, exactly as the previous version of the framework did).

2. **`~/.hermes/config.yaml` → `model.provider`** — when set and not `"auto"`, this picks the provider directly. The model name comes from `model.default`. Credentials follow the chain in the next section.

3. **`~/.hermes/config.yaml` → `model.provider: "auto"` (or unset)** — slim auto-detect: tries each provider in priority order (`anthropic` → `openrouter` → `openai` → `nous` → `gemini` → `copilot` → ...) and picks the first one with a usable credential anywhere.

4. **No Hermes config at all** — same auto-detect, but only env vars are checked.

5. **Nothing usable** — `HermesProviderError` with a message that lists every step that was tried.

For a chosen provider, credentials resolve in this order:

1. `model.api_key` from `~/.hermes/config.yaml` (when the provider matches)
2. The provider's standard env var (e.g. `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`)
3. The highest-priority entry in `~/.hermes/auth.json` `credential_pool[<provider>]` (lowest `priority` integer wins, mirroring Hermes's own convention)

Reflection role specifically: when `--reflection-model` is unset, falls back to the resolved optimizer model. Reasoning models (gpt-5-class, Claude with extended thinking) work best for the reflection LM; if your provider doesn't have one, the optimizer model is a reasonable fallback.

## Provider mapping

The Hermes provider name maps to LiteLLM as follows. For each, the framework constructs a `dspy.LM(model_string, **lm_kwargs)` call.

| Hermes `provider` | LiteLLM `model` | `lm_kwargs` | Notes |
|---|---|---|---|
| `anthropic` | `anthropic/<model>` | `api_key` (+ `api_base` if set) | Native Anthropic Messages API |
| `openrouter` | `openrouter/<model>` | `api_key` | Multi-model gateway |
| `openai` | `openai/<model>` | `api_key` (+ `api_base` if set) | OpenAI direct |
| `gemini` | `gemini/<model>` | `api_key` | Native Google AI Studio API |
| `nous` | `openai/<model>` | `api_base=https://inference-api.nousresearch.com/v1`, `api_key` | OpenAI-compat endpoint |
| `copilot` | `openai/<model>` | `api_base=https://api.githubcopilot.com`, `api_key` | Uses GITHUB_TOKEN |
| `custom` | `openai/<model>` | `api_base` from config (required), `api_key` if needed | The escape hatch — point at any OpenAI-compat endpoint |
| `ollama`, `vllm`, `llamacpp` | `openai/<model>` | `api_base` from config (required), `api_key=EMPTY` placeholder | Local servers; aliases for `custom` |
| `lmstudio` | `openai/<model>` | `api_base=http://127.0.0.1:1234/v1` (default), `api_key=EMPTY` | LM Studio local server |
| `zai`, `kimi-coding`, `minimax`, `huggingface`, `nvidia`, `arcee`, `ollama-cloud`, `kilocode`, `ai-gateway`, `xiaomi` | `openai/<model>` | Provider's canonical `api_base`, `api_key` from env or pool | OpenAI-wire-compatible HTTP |
| `bedrock` (aliases: `aws`, `aws-bedrock`, `amazon`, `amazon-bedrock`) | `bedrock/<model-id>` | `aws_region_name` (+ optional `aws_profile_name`) | AWS Bedrock via boto3 default credential chain — see [AWS Bedrock setup](#aws-bedrock-setup) below |
| `openai-codex` | `openai/<model>` | `api_base=https://chatgpt.com/backend-api/codex`, `api_key=<oauth-bearer>`, `extra_headers` for Cloudflare, `model_type=responses` | Custom DSPy LM with OAuth refresh — see [OpenAI Codex Responses API](#openai-codex-responses-api) below |

**Wire-mode flip:** if the resolved `api_base` contains `/anthropic` (z.ai with `/anthropic` suffix, MiniMax with `/anthropic` suffix), the model string flips to `anthropic/<model>` — Hermes does the same auto-detection.

## Local-server setups

### vLLM

```yaml
# ~/.hermes/config.yaml
model:
  default: meta-llama/Llama-3.3-70B-Instruct
  provider: custom
  base_url: http://localhost:8000/v1
```

No `api_key` needed. The framework passes `api_key=EMPTY` to LiteLLM, which most local servers tolerate.

### Ollama

```yaml
model:
  default: llama3.3
  provider: ollama
  base_url: http://localhost:11434/v1
```

### LM Studio

```yaml
model:
  default: qwen2.5-coder-7b
  provider: lmstudio
  # base_url defaults to http://127.0.0.1:1234/v1; override if you've changed the port
```

### llama.cpp

```yaml
model:
  default: my-quantized-model
  provider: llamacpp
  base_url: http://localhost:8080/v1
```

## AWS Bedrock setup

Bedrock auth flows through boto3's default credential chain — `AWS_BEARER_TOKEN_BEDROCK`, `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`, `AWS_PROFILE`, IAM role, or IMDS. The framework never reads AWS credentials itself; it only surfaces the region (and optional profile name) to LiteLLM, which calls boto3 under the hood.

```yaml
# ~/.hermes/config.yaml
model:
  default: us.anthropic.claude-sonnet-4-6
  provider: bedrock
bedrock:
  region: us-east-2
  aws_profile_name: my-bedrock-profile  # optional; omit to use default chain
```

Region resolution: `bedrock.region` in `config.yaml` → `AWS_REGION` env var → `AWS_DEFAULT_REGION` env var → `us-east-1` (hardcoded fallback).

Cross-region inference profiles (`us.anthropic.claude-sonnet-4-6`, `apac.anthropic.claude-haiku-4-5`, etc.) work unchanged — the leading region prefix is part of the model ID and reaches Bedrock as-is.

If `model.default` is unset, the framework falls back to `us.anthropic.claude-sonnet-4-6` so a bare `provider: bedrock` config runs without further configuration.

Bedrock is **never auto-detected** when `provider: auto` (or unset) — AWS env vars are commonly set for non-Bedrock reasons (S3, DynamoDB), so silently routing the optimizer to Bedrock would surprise users. Set `provider: bedrock` explicitly to opt in.

When boto3 can't find any credentials, the call surfaces as `litellm.AuthenticationError("Unable to locate credentials...")`. The preflight catches this and renders the recovery hint (`export AWS_PROFILE=...` / `export AWS_BEARER_TOKEN_BEDROCK=...` / "run from an instance with Bedrock permissions") in the standard Rich panel.

**What's not supported:** Bedrock Guardrails (Hermes uses these via boto3 directly; LiteLLM doesn't expose them). If Guardrails are required for your evolution runs, wrap the framework with your own moderation layer — the cost ledger and cost-ceiling work unchanged.

## OpenAI Codex Responses API

For users with a ChatGPT subscription, Hermes can route through OpenAI's Codex Responses API. Run `hermes auth add openai-codex` (Hermes-side) to populate `~/.hermes/auth.json` with an OAuth access + refresh token, then point your `config.yaml` at it:

```yaml
# ~/.hermes/config.yaml
model:
  default: gpt-5-codex     # or gpt-5, depending on what your plan grants
  provider: openai-codex
```

The framework reads OAuth credentials from `auth.json credential_pool["openai-codex"]` (highest-priority entry wins) and instantiates a `CodexLM` — a thin `dspy.LM` subclass that adds two things on top of stock DSPy:

1. **Cloudflare-mitigation headers**: `originator: codex_cli_rs`, a `codex_cli_rs`-prefixed `User-Agent`, and `ChatGPT-Account-ID` extracted from the OAuth JWT. Without these, every call from a non-residential IP (CI runners, VPS, cloud-hosted agents) gets a 403 from Cloudflare regardless of OAuth correctness.
2. **In-memory OAuth refresh**: the access token expires every ~30 minutes; before each call the LM checks `expires_at` against a 120s skew window and refreshes via the OAuth `refresh_token` grant if needed. Multiple LM roles (optimizer, reflection, eval, judge) share one process-wide refresh state so a four-thread evolution doesn't trigger four parallel refreshes (which would `refresh_token_reused` three of them).

Refresh is **in-memory only** — the framework does not write back to `~/.hermes/auth.json`. Long evolutions (>30 minutes on a fresh token) need to re-run `hermes auth add openai-codex` if the on-disk store also needs to be refreshed, but each evolve invocation refreshes its own runtime state independently.

**What's not supported:** streaming via the Responses endpoint (evolution doesn't stream), Codex-specific reasoning-effort overrides (DSPy's defaults work for gpt-5-class), and tool-call message conversion beyond what DSPy's `_convert_chat_request_to_responses_request` already handles. If a Codex 401 surfaces during a run, the standard auth-error panel renders with the `hermes auth add openai-codex` recovery hint.

## Nous Portal OAuth + agent_key

Nous Portal uses a two-stage credential model that's different from every other provider:

1. **OAuth access_token** (long-lived, days). Refreshable via the standard `refresh_token` grant.
2. **agent_key** (short-lived, ~30 minutes). Minted from the access_token via a Nous-specific `POST /api/oauth/agent-key`. The inference endpoint requires the **agent_key** as Bearer — not the access_token.

Run `hermes model` and select Nous Portal to populate `~/.hermes/auth.json` with both. Then point `config.yaml` at Nous:

```yaml
# ~/.hermes/config.yaml
model:
  default: Hermes-4-405B
  provider: nous
```

When the resolver detects a Nous credential pool entry with a `refresh_token` (signals OAuth-managed flow), the framework instantiates a `NousLM` subclass that:

1. **Mints a fresh agent_key at preflight time** by POSTing to `{portal}/api/oauth/agent-key` with the OAuth access_token as Bearer.
2. **Refreshes the OAuth access_token in-memory** when it's within 120s of expiry — POSTed to `{portal}/api/oauth/token` with the standard refresh_token grant. Mirrors Hermes's own refresh-first-then-mint sequencing in `hermes_cli/auth.py`.
3. **Re-mints on inference 401** (mid-run agent_key revocation or expiration). The four LM roles (optimizer, reflection, eval, judge) coordinate through a shared lock so a four-thread evolution doesn't race the portal's single-use refresh-token rotation.

The portal URL is overridable via `HERMES_PORTAL_BASE_URL` (Hermes's own env var name; sharing keeps configs portable for stage / mock setups).

Refresh + mint state is **in-memory only** — the framework never writes back to `~/.hermes/auth.json`. For evolution sessions running longer than the on-disk agent_key TTL (~30 minutes since the last `hermes model`), the in-process refresh handles it. For multi-day sessions, periodic `hermes model` keeps the on-disk store fresh.

**What's not supported:** auxiliary endpoints (vision / web-extract / session-search models from `auxiliary.*` config), streaming, and `auth.json` writeback. Pool entries without `refresh_token` (env-var-style `NOUS_API_KEY` setups) fall through to the existing direct-pass-through path — note that path probably doesn't actually work for Nous inference (the access_token isn't a valid Bearer), but we don't try to "upgrade" those users silently.

A runnable smoke harness at `tests/manual/nous_smoke.py` validates the Nous wire flow against a local mock portal (no real Nous Portal account required). Run via `uv run python tests/manual/nous_smoke.py`.

## Per-role overrides

When your provider exposes multiple models, you can pick a different one per role to manage cost. Common pattern: a frontier model for the optimizer + reflection LMs (where reasoning matters), a cheaper model for eval + judge (where you'll make many calls):

```bash
uv run python -m evolution.skills.evolve_skill \
    --skill my-skill \
    --optimizer-model anthropic/claude-opus-4-5 \
    --reflection-model anthropic/claude-opus-4-5 \
    --eval-model anthropic/claude-haiku-4-5
```

Or for OpenRouter:

```bash
uv run python -m evolution.skills.evolve_skill \
    --skill my-skill \
    --optimizer-model openrouter/anthropic/claude-opus-4-5 \
    --reflection-model openrouter/openai/gpt-5 \
    --eval-model openrouter/anthropic/claude-haiku-4-5
```

When the `--<role>-model` flag is set, the resolver does not infer `api_base` or `api_key` — pass the env var (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, etc.) yourself. This matches the previous behavior of the framework.

## Cost considerations

If your Hermes is configured for a single frontier model (e.g. Claude Opus), defaulting all four roles to it can be expensive. A typical evolution run hits the eval + judge LMs ~100x and the optimizer + reflection LMs ~10x. If your eval-LM-per-call cost is $0.10, eval alone is ~$10 per run; on Opus it would be ~$50.

Three ways to manage this:

1. **Per-role overrides** (above) — pick a cheaper model from the same provider for eval + judge.
2. **Use `--budget light`** — fewer GEPA iterations, fewer total LM calls.
3. **Cost advisor** — when `--eval-model` is unset, the framework checks `litellm.model_cost` after preflight and surfaces a Rich panel suggesting a cheaper same-provider model with sufficient context window when one exists. The panel includes a paste-ready `--eval-model` flag. Pass `--no-cost-suggest` to suppress.

The `output/<run>/run_config.json` includes a `resolved_lms` block showing exactly which model + endpoint was used per role, with `api_key` redacted. Inspect it after a run to confirm what you paid for.

## Standalone (no Hermes) setup

The framework checks env vars in this priority order when `~/.hermes/config.yaml` is absent or `provider: "auto"` is set:

1. `ANTHROPIC_API_KEY` → uses `anthropic/claude-opus-4-5` by default
2. `OPENROUTER_API_KEY` → uses `openrouter/anthropic/claude-opus-4-5`
3. `OPENAI_API_KEY` → uses `openai/gpt-4.1`
4. `NOUS_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GITHUB_TOKEN`, others — when only the env var is set without a Hermes config, the resolver picks the model name from a sane built-in default per provider.

For finer control, pass `--optimizer-model` etc. explicitly and skip Hermes resolution entirely.

## Stale credentials and the auth-error path

The framework validates LM credentials before doing any optimization work. On every `evolve` run (unless you pass `--no-preflight`), it makes one tiny ~$0.0001 `litellm.completion` call per unique LM combo. If a credential is bad, the run fails fast with a Rich-formatted error panel that names the model and includes the right recovery command for your provider — no Python traceback, no 5-minute doomed run, no dataset-gen budget burned before the failure surfaces.

Per-provider recovery commands the framework suggests:

| Provider | Recovery |
|---|---|
| `anthropic` | `hermes auth add anthropic` |
| `nous` | `hermes model` (then select Nous Portal) |
| `gemini` | `hermes login --provider google-gemini-cli` |
| `openrouter`, `openai`, `kimi-coding`, etc. | `export <PROVIDER>_API_KEY=...` |
| `copilot` | `gh auth login` |

(`hermes login` was deprecated upstream — current Hermes commands are `hermes auth add <provider>` or `hermes model`. Gemini still uses the old `hermes login --provider` form.)

If a credential goes bad mid-run (rare — long sessions on short-TTL OAuth, key revocation in flight), the same `HermesProviderError` surfaces from the next LM call rather than producing silent `score=0.0` evaluations. The mid-run path is defense-in-depth on top of preflight; under normal use the preflight catches everything.

The framework does not refresh OAuth tokens — that's Hermes's job. Honor the `last_status: exhausted` + `last_error_reset_at` fields Hermes writes to `auth.json` when it detects a bad credential; we skip those entries until Hermes's cooldown passes, mirroring Hermes's own pool rotation logic.

### Skipping preflight

Pass `--no-preflight` to skip the credential probe. Useful when:
- You just ran `evolve` successfully a minute ago and know the creds are good
- You're iterating in a tight loop and want to shave the ~1s preflight latency
- Your provider's `litellm.completion` probe is flaky (some custom endpoints don't like `max_tokens=1`)

## Troubleshooting

**Error: "No model could be resolved for role=optimizer."**

The framework couldn't find any usable provider. Either configure Hermes (see the [Hermes Agent README](https://github.com/NousResearch/hermes-agent)), set `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY` / `OPENAI_API_KEY`, or pass `--optimizer-model anthropic/claude-opus-4-5` (or your provider's equivalent) explicitly.

**Error: "Auto-detected provider 'X' but no model name configured."**

Your env var is set but your Hermes config (or absence thereof) doesn't pin a model name. Pass `--optimizer-model <provider>/<model>` or set `model.default` in `~/.hermes/config.yaml`.

**LiteLLM error: "Model not found" or 404**

Hermes can use server-side aliases (e.g. `gpt-5.4-mini` even when OpenAI's catalog calls it something else). LiteLLM doesn't know about Hermes aliases — it sends whatever name you give it straight to the endpoint. If the endpoint rejects the name, fix it in `~/.hermes/config.yaml` `model.default` or override with `--optimizer-model <real-model-name>`.

**The wrong model is being used for eval/judge — I expected a cheaper one.**

The framework defaults all four roles to Hermes's single `model.default`. To use a cheaper model for eval, pass `--eval-model` explicitly.

## Future work

This module currently does not:

- Honor `auxiliary.*` provider config from `config.yaml` (Hermes's vision/web-extract/session-search routing)
- OAuth refresh for Qwen, Spotify, or Google Gemini providers (Codex and Nous Portal handled in-memory — see their dedicated sections above; the other OAuth providers in Hermes don't have demand from the evolution use case yet)

The slim resolver lives at `evolution/core/hermes_provider.py`. The mapping table is sourced from `hermes_cli/auth.py` constants — drift is possible; update by reference when Hermes adds providers.

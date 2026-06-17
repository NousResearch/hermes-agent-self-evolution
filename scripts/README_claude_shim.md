# Claude OAuth Shim

OpenAI-compatible HTTP shim that routes chat completions through the Claude
Code CLI's OAuth subscription via the official
[`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/).

## Why

Tools in this repo (DSPy, litellm) speak OpenAI by default and want
`OPENAI_API_KEY`. If you already pay for Claude Code and would rather use that
subscription than provision a second Anthropic Console API key, this shim
gives you an OpenAI-compatible endpoint backed by your CC OAuth tokens.

## Install

```bash
pip install -e ".[claude-shim]"
```

(or install the three deps directly: `claude-agent-sdk fastapi uvicorn[standard]`)

You must also have an authenticated Claude Code CLI on the machine
(`~/.claude/credentials.json` populated — happens automatically when you log
into the CLI).

## Run

```bash
python scripts/claude_oai_shim.py
# listens on http://127.0.0.1:8765

# optional
export SHIM_API_KEY=your-local-secret
python scripts/claude_oai_shim.py --port 8765 --host 127.0.0.1
```

When `SHIM_API_KEY` is set, clients must send `Authorization: Bearer <key>`.

## Use with `evolve_skill`

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8765/v1
export OPENAI_API_KEY=anything   # or your SHIM_API_KEY when auth is enabled

python -m evolution.skills.evolve_skill \
    --skill github-code-review \
    --optimizer-model openai/claude-opus-4-7 \
    --eval-model openai/claude-haiku-4-5 \
    --iterations 5
```

## Model routing

OpenAI-style names with a `claude-*` suffix map to Claude Code's three tiers
(override defaults with `CLAUDE_SHIM_OPUS`, `CLAUDE_SHIM_SONNET`,
`CLAUDE_SHIM_HAIKU`):

| OpenAI name             | Default routed to          |
|-------------------------|----------------------------|
| `claude-opus-*`         | `claude-opus-4-1-20250805` |
| `claude-sonnet-*`       | `claude-sonnet-4-5`        |
| `claude-haiku-*`        | `claude-haiku-4-5`         |

Non-Claude names (e.g. `gpt-4`) log a warning and fall back to sonnet.

## Endpoints

- `GET  /health` → `{"ok": true}`
- `POST /v1/chat/completions` → OpenAI Chat Completions response shape

## Caveats

- **Streaming is not supported** (`stream: true` returns HTTP 400).
- Token usage counts are best-effort.
- Review your Claude Code plan's acceptable-use terms before automating
  against it.

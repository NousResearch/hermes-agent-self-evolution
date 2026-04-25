#!/usr/bin/env bash
# Wrapper to run hermes-agent-self-evolution with provider selection
#set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load secrets from Hermes' secure .env
if [[ -f "$HOME/.hermes/.env" ]]; then
    set -a
    source "$HOME/.hermes/.env"
    set +a
fi

# Load local config
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

PROVIDER="${PROVIDER:-openrouter}"

if [[ "$PROVIDER" == "nous" ]]; then
    # Nous Research: OpenRouter-powered with coding-plan discounts
    OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-anthropic/claude-sonnet-4.6}"
    EVAL_MODEL="${EVAL_MODEL:-moonshotai/kimi-k2.6}"
    if [[ -n "${NOUS_API_KEY:-}" ]]; then
        export OPENROUTER_API_KEY="$NOUS_API_KEY"
    fi
    echo "🧬 Provider: Nous Research (Kimi K2.6 evaluator is FREE)"
else
    # OpenRouter defaults
    OPTIMIZER_MODEL="${OPTIMIZER_MODEL:-anthropic/claude-sonnet-4.6}"
    EVAL_MODEL="${EVAL_MODEL:-google/gemini-3.1-flash-lite}"
    JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-sonnet-4.6}"

    if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
        echo "ERROR: OPENROUTER_API_KEY is not set. Set it in ~/.hermes/.env or $SCRIPT_DIR/.env"
        exit 1
    fi
    echo "🧬 Provider: OpenRouter"
fi

source "$SCRIPT_DIR/venv/bin/activate"

exec python -m evolution.skills.evolve_skill \
    --optimizer-model "$OPTIMIZER_MODEL" \
    --eval-model "$EVAL_MODEL" \
    "$@"

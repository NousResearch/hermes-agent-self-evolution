#!/bin/bash
# Hermes self-evolution nightly runner
# Targets: github-code-review skill
# LLM: cx/gpt-5.4 via local endpoint http://localhost:20128/v1

set -euo pipefail

cd "$(dirname "$0")"

# Local OpenAI-compatible endpoint
export OPENAI_API_BASE="http://localhost:20128/v1"
export OPENAI_API_KEY="dummy-local-key"
# LiteLLM also respects these
export OPENAI_BASE_URL="http://localhost:20128/v1"

# Activate venv
# shellcheck disable=SC1091
source venv/bin/activate

SKILL="${SKILL:-github-code-review}"
ITERATIONS="${ITERATIONS:-10}"
MODEL="openai/cx/gpt-5.4"

LOG_DIR="$HOME/.hermes/self-evolution/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/evolve-${SKILL}-${STAMP}.log"

echo "[$(date -Iseconds)] Starting evolution: skill=$SKILL iters=$ITERATIONS model=$MODEL" | tee -a "$LOG"

python -m evolution.skills.evolve_skill \
  --skill "$SKILL" \
  --iterations "$ITERATIONS" \
  --optimizer-model "$MODEL" \
  --eval-model "$MODEL" \
  2>&1 | tee -a "$LOG"

echo "[$(date -Iseconds)] Evolution run complete. Log: $LOG" | tee -a "$LOG"
